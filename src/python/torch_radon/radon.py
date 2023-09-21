import typing
import warnings

import numpy as np
import torch

from . import cuda_backend
from .differentiable_functions import RadonForward, RadonBackprojection
from .filtering import FourierFilters
from .projection import Projection
from .utils import normalize_shape, expose_projection_attributes
from .volumes import Volume2D, Volume3D

warnings.simplefilter('default')


class ExecCfgGeneratorBase:

    def __init__(self):
        pass

    def __call__(self, vol_cfg, proj_cfg, is_half):
        if proj_cfg.projection_type == 2:
            ch = 4 if is_half else 1
            return cuda_backend.ExecCfg(8, 16, 8, ch)

        return cuda_backend.ExecCfg(16, 16, 1, 4)


class BaseRadon:
    """The base class for all Radon Modules"""

    def __init__(
        self,
        volume: typing.Union[Volume2D, Volume3D],
        projection: Projection,
    ):

        self.volume = volume
        self.projection = projection
        self.exec_cfg_generator = ExecCfgGeneratorBase()

        # caches used to avoid reallocation of resources
        self.tex_cache = cuda_backend.TextureCache(8)
        self.fft_cache = cuda_backend.FFTCache(8)
        self.fourier_filters = FourierFilters()

    def _check_input(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        return x

    def forward(
        self,
        image: torch.Tensor,
        angles: torch.Tensor,
        exec_cfg: cuda_backend.ExecCfg = None,
    ):
        r"""Radon forward projection.

        :param image: PyTorch GPU tensor.
        :param angles: PyTorch GPU tensor indicating the measuring angles, if None the angles given to the constructor are used
        :returns: PyTorch GPU tensor containing sinograms.
        """
        image = self._check_input(image)
        angles = self._check_input(angles)
        return RadonForward.apply(
            image,
            angles,
            self.tex_cache,
            self.volume.to_cfg(),
            self.projection.cfg,
            exec_cfg,
        )

    def backward(
        self,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        exec_cfg: cuda_backend.ExecCfg = None,
    ):
        r"""Radon backward projection.

        :param sinogram: PyTorch GPU tensor containing sinograms.
        :param angles: PyTorch GPU tensor indicating the measuring angles, if None the angles given to the constructor
            are used
        :returns: PyTorch GPU tensor containing backprojected volume.
        """
        sinogram = self._check_input(sinogram)
        angles = self._check_input(angles)
        return RadonBackprojection.apply(
            sinogram,
            angles,
            self.tex_cache,
            self.volume.to_cfg(),
            self.projection.cfg,
            exec_cfg,
        )

    @normalize_shape(2)
    def filter_sinogram(self, sinogram, filter_name="ramp"):
        size = sinogram.size(2)
        n_angles = sinogram.size(1)

        # Pad sinogram to improve accuracy
        padded_size = max(64, int(2**np.ceil(np.log2(2 * size))))
        pad = padded_size - size
        padded_sinogram = torch.nn.functional.pad(sinogram.float(),
                                                  (0, pad, 0, 0))

        sino_fft = cuda_backend.rfft(padded_sinogram,
                                     self.fft_cache) / np.sqrt(padded_size)

        # get filter and apply
        f = self.fourier_filters.get(padded_size, filter_name, sinogram.device)
        filtered_sino_fft = sino_fft * f

        # Inverse fft
        filtered_sinogram = cuda_backend.irfft(
            filtered_sino_fft, self.fft_cache) / np.sqrt(padded_size)
        filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi /
                                                              (2 * n_angles))

        return filtered_sinogram.to(dtype=sinogram.dtype)


class ParallelBeam(BaseRadon):
    r"""
    |
    .. image:: https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/
            master/pictures/parallelbeam.svg?sanitize=true
        :align: center
        :width: 400px
    |

    Class that implements Radon projection for the Parallel Beam geometry.

    :param det_count: *Required*. Number of rays that will be projected.
    :param angles: *Required*. Array containing the list of measuring angles. Can be a Numpy array, a PyTorch tensor or a tuple
        `(start, end, num_angles)` defining a range.
    :param det_spacing: Distance between two contiguous rays. By default is `1.0`.
    :param volume: Specifies the volume position and scale. By default a uniform volume is used.
        To create a non-uniform volume specify an instance of :class:`torch_radon.Volume2D`.

    """

    def __init__(
            self,
            det_count: int,
            det_spacing: float = 1.0,
            volume: Volume2D = Volume2D(),
    ):
        super().__init__(
            volume,
            Projection.parallel_beam(det_count, det_spacing),
        )


expose_projection_attributes(ParallelBeam, [("det_count", "det_count_u"),
                                            ("det_spacing", "det_spacing_u")])


class FanBeam(BaseRadon):
    r"""
    |
    .. image:: https://raw.githubusercontent.com/matteo-ronchetti/torch-radon/
            master/pictures/fanbeam.svg?sanitize=true
        :align: center
        :width: 400px
    |

    Class that implements Radon projection for the Fanbeam geometry.

    :param det_count: *Required*. Number of rays that will be projected.
    :param angles: *Required*. Array containing the list of measuring angles. Can be a Numpy array, a PyTorch tensor or a tuple
        `(start, end, num_angles)` defining a range.
    :param src_dist: Distance between the source of rays and the origin. If not specified is set equals to :attr:`det_count`.
    :param det_dist: Distance between the detector plane and the origin. If not specified is set equals to :attr:`det_dist`.
    :param det_spacing: Distance between two contiguous rays. By default is `(src_dist + det_dist) / src_dist`.
    :param volume: Specifies the volume position and scale. By default a square uniform volume is used.
        To create a non-uniform volume specify an instance of :class:`torch_radon.Volume2D`.

    """

    def __init__(
        self,
        det_count: int,
        src_dist: float = None,
        det_dist: float = None,
        det_spacing: float = None,
        volume: Volume2D = None,
    ):

        if src_dist is None:
            src_dist = det_count

        if det_dist is None:
            det_dist = src_dist

        if det_spacing is None:
            det_spacing = (src_dist + det_dist) / src_dist

        if volume is None:
            volume = Volume2D()

        projection = Projection.fanbeam(
            src_dist,
            det_dist,
            det_count,
            det_spacing,
        )

        super().__init__(volume, projection)


class ConeBeam(BaseRadon):

    def __init__(
        self,
        det_count_u: int,
        src_dist: float = None,
        det_dist: float = None,
        det_count_v: int = -1,
        det_spacing_u: float = 1.0,
        det_spacing_v: float = -1.0,
        pitch: float = 0.0,
        base_z: float = 0.0,
        volume: Volume3D = None,
    ):

        if src_dist is None:
            src_dist = det_count_u

        if det_dist is None:
            det_dist = src_dist

        det_count_v = det_count_v if det_count_v > 0 else det_count_u
        det_spacing_v = det_spacing_v if det_spacing_v > 0 else det_spacing_u

        if volume is None:
            volume = Volume3D()

        projection = Projection.coneflat(
            src_dist,
            det_dist,
            det_count_u,
            det_spacing_u,
            det_count_v,
            det_spacing_v,
            pitch,
            base_z,
        )

        super().__init__(volume, projection)


expose_projection_attributes(ConeBeam, [
    "det_count_u", "det_count_v", "det_spacing_u", "det_spacing_v",
    ("src_dist", "s_dist"), ("det_dist", "d_dist"), "pitch",
    ("base_z", "initial_z")
])
