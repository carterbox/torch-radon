import typing
import warnings

import numpy as np
import torch
import torch.fft

from . import cuda_backend
from .differentiable_functions import RadonForward, RadonBackprojection
from .filtering import FourierFilters
from .projection import Projection
from .utils import normalize_shape, ShapeNormalizer, expose_projection_attributes
from .volumes import Volume2D, Volume3D

warnings.simplefilter("default")


class ExecCfgGeneratorBase:
    def __init__(self):
        pass

    def __call__(self, vol_cfg, proj_cfg, is_half):
        if proj_cfg.projection_type == 2:
            ch = 4 if is_half else 1
            return cuda_backend.ExecCfg(8, 16, 8, ch)

        return cuda_backend.ExecCfg(16, 16, 1, 4)


class BaseRadon:
    def __init__(
        self,
        angles: typing.Union[torch.Tensor, typing.Tuple[int, int, int]],
        volume: typing.Union[Volume2D, Volume3D],
        projection: Projection,
    ):
        # allows angles to be specified as (start_angle, end_angle, n_angles)
        if isinstance(angles, tuple) and len(angles) == 3:
            start_angle, end_angle, n_angles = angles
            angles = np.linspace(
                start_angle,
                end_angle,
                n_angles,
                endpoint=False,
            )

        # make sure that angles are a PyTorch tensor
        if not isinstance(angles, torch.Tensor):
            angles = torch.FloatTensor(angles)

        self.angles = angles
        self.volume = volume
        self.projection = projection
        self.exec_cfg_generator = ExecCfgGeneratorBase()

        # caches used to avoid reallocation of resources
        self.tex_cache = cuda_backend.TextureCache(8)
        self.fourier_filters = FourierFilters()

    def _move_parameters_to_device(self, device):
        if device != self.angles.device:
            self.angles = self.angles.to(device)

    def _check_input(self, x):
        if not x.is_contiguous():
            x = x.contiguous()

        if x.dtype == torch.float16:
            assert (
                x.size(0) % 4 == 0
            ), f"Batch size must be multiple of 4 when using half precision. Got batch size {x.size(0)}"

        return x

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor = None,
        exec_cfg: cuda_backend.ExecCfg = None,
    ):
        r"""Radon forward projection.

        :param x: PyTorch GPU tensor.
        :param angles: PyTorch GPU tensor indicating the measuring angles, if None the angles given to the constructor are used
        :returns: PyTorch GPU tensor containing sinograms.
        """
        x = self._check_input(x)
        self._move_parameters_to_device(x.device)

        angles = angles if angles is not None else self.angles

        shape_normalizer = ShapeNormalizer(self.volume.num_dimensions())
        x = shape_normalizer.normalize(x)

        self.volume.height = x.size(-2)
        self.volume.width = x.size(-1)
        if self.volume.num_dimensions() == 3:
            self.volume.depth = x.size(-3)

        self.projection.cfg.n_angles = len(angles)

        y = RadonForward.apply(
            x,
            angles,
            self.tex_cache,
            self.volume.to_cfg(),
            self.projection.cfg,
            self.exec_cfg_generator,
            exec_cfg,
        )

        return shape_normalizer.unnormalize(y)

    def backward(
        self,
        sinogram,
        angles: torch.Tensor = None,
        volume: typing.Union[Volume2D, Volume3D] = None,
        exec_cfg: cuda_backend.ExecCfg = None,
    ):
        r"""Radon backward projection.

        :param sinogram: PyTorch GPU tensor containing sinograms.
        :param angles: PyTorch GPU tensor indicating the measuring angles, if None the angles given to the constructor
            are used
        :returns: PyTorch GPU tensor containing backprojected volume.
        """
        sinogram = self._check_input(sinogram)
        volume = self.volume if volume is None else volume

        assert (
            volume.has_size()
        ), "Must use forward before calling backward or specify a volume"

        self._move_parameters_to_device(sinogram.device)

        angles = angles if angles is not None else self.angles

        shape_normalizer = ShapeNormalizer(self.volume.num_dimensions())
        sinogram = shape_normalizer.normalize(sinogram)

        self.projection.cfg.n_angles = len(angles)

        y = RadonBackprojection.apply(
            sinogram,
            angles,
            self.tex_cache,
            volume.to_cfg(),
            self.projection.cfg,
            self.exec_cfg_generator,
            exec_cfg,
        )

        return shape_normalizer.unnormalize(y)

    @normalize_shape(2)
    def filter_sinogram(
        self,
        sinogram: torch.Tensor,
        filter_name: typing.Literal[
            "ramp", "shepp-logan", "cosine", "hamming", "hann"
        ] = "ramp",
    ):
        size = sinogram.size(2)
        n_angles = sinogram.size(1)

        # Pad sinogram to improve accuracy
        padded_size = max(64, int(2 ** np.ceil(np.log2(2 * size))))
        pad = padded_size - size
        padded_sinogram = torch.nn.functional.pad(sinogram.float(), (0, pad, 0, 0))

        sino_fft = torch.fft.rfft(padded_sinogram, norm="ortho")

        # get filter and apply
        f = self.fourier_filters.get(padded_size, filter_name, sinogram.device)
        filtered_sino_fft = sino_fft * f

        # Inverse fft
        filtered_sinogram = torch.fft.irfft(filtered_sino_fft, norm="ortho")
        filtered_sinogram = filtered_sinogram[:, :, :-pad] * (np.pi / (2 * n_angles))

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
        angles: typing.Union[list, np.array, torch.Tensor, tuple],
        det_spacing: float = 1.0,
        volume: Volume2D = None,
    ):
        if volume is None:
            volume = Volume2D()

        projection = Projection.parallel_beam(det_count, det_spacing)

        super().__init__(angles, volume, projection)


expose_projection_attributes(
    ParallelBeam, [("det_count", "det_count_u"), ("det_spacing", "det_spacing_u")]
)


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
        angles: typing.Union[list, np.array, torch.Tensor, tuple],
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

        projection = Projection.fanbeam(src_dist, det_dist, det_count, det_spacing)

        super().__init__(angles, volume, projection)


class ConeBeam(BaseRadon):
    def __init__(
        self,
        det_count_u: int,
        angles: typing.Union[list, np.array, torch.Tensor, tuple],
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

        super().__init__(angles, volume, projection)

    def filter_sinogram(
        self,
        sinogram: torch.Tensor,
        filter_name: typing.Literal[
            "ramp", "shepp-logan", "cosine", "hamming", "hann"
        ] = "ramp",
        v_chunk_size: int | None = 32,
    ):
        r"""Filter cone-beam projections along the detector-u axis.

        The input shape is ``[..., angles, det_v, det_u]``. Filtering is
        independent for each detector-v row, so processing detector-v in chunks
        is mathematically equivalent to filtering all rows at once while using
        less peak memory for the FFT intermediates.
        """
        shape_normalizer = ShapeNormalizer(3)
        sinogram = shape_normalizer.normalize(sinogram)

        batch_size, n_angles, det_v, det_u = sinogram.shape
        chunk_size = det_v if v_chunk_size is None else int(v_chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"v_chunk_size must be positive or None, got {v_chunk_size}")

        if chunk_size >= det_v:
            # Fast path: filter the 4D tensor directly — no permutes or reshapes.
            padded_size = max(64, int(2 ** np.ceil(np.log2(2 * det_u))))
            pad = padded_size - det_u
            padded = torch.nn.functional.pad(sinogram.float(), (0, pad))
            sino_fft = torch.fft.rfft(padded, norm="ortho")
            f = self.fourier_filters.get(padded_size, filter_name, sinogram.device)
            filtered = torch.fft.irfft(sino_fft * f, norm="ortho")
            result = filtered[..., :-pad] * (np.pi / (2 * n_angles))
            result = result.to(dtype=sinogram.dtype)
        else:
            # Chunked path: work in [batch, det_v, angles, det_u] layout so that
            # slicing along det_v and writing results back are contiguous.
            permuted = sinogram.permute(0, 2, 1, 3)
            filtered = torch.empty(
                batch_size, det_v, n_angles, det_u,
                device=sinogram.device, dtype=sinogram.dtype,
            )
            for start in range(0, det_v, chunk_size):
                end = min(start + chunk_size, det_v)
                chunk = permuted[:, start:end].reshape(
                    batch_size * (end - start), n_angles, det_u
                )
                chunk = super().filter_sinogram(chunk, filter_name=filter_name)
                filtered[:, start:end] = chunk.reshape(
                    batch_size, end - start, n_angles, det_u
                )
            result = filtered.permute(0, 2, 1, 3)

        return shape_normalizer.unnormalize(result)

    def fdk_preweight(self, sinogram: torch.Tensor):
        r"""Apply the standard cone-beam FDK distance preweight."""
        shape_normalizer = ShapeNormalizer(3)
        sinogram = shape_normalizer.normalize(sinogram)

        det_v = sinogram.shape[-2]
        det_u = sinogram.shape[-1]
        device = sinogram.device
        # Compute the weight grid in float32 even for half-precision inputs to
        # avoid overflow in source_to_detector**2 and precision loss in the
        # sqrt/division. The result is cast back to the input dtype below.
        compute_dtype = torch.float32

        u = (
            torch.arange(det_u, device=device, dtype=compute_dtype)
            - (det_u - 1) / 2
        ) * self.det_spacing_u
        v = (
            torch.arange(det_v, device=device, dtype=compute_dtype)
            - (det_v - 1) / 2
        ) * self.det_spacing_v

        # Broadcast u and v into a (det_v, det_u) weight grid without allocating
        # a full meshgrid (saves one det_v × det_u tensor).
        uu = u.view(1, -1)
        vv = v.view(-1, 1)

        source_to_detector = self.src_dist + self.det_dist
        weight = source_to_detector / torch.sqrt(source_to_detector**2 + uu**2 + vv**2)
        weight = weight * (source_to_detector / self.src_dist)
        weight = weight.to(dtype=sinogram.dtype)

        weighted = sinogram * weight.view(1, 1, det_v, det_u)
        return shape_normalizer.unnormalize(weighted)

    def fdk(
        self,
        sinogram: torch.Tensor,
        filter_name: typing.Literal[
            "ramp", "shepp-logan", "cosine", "hamming", "hann"
        ] = "ramp",
        v_chunk_size: int | None = 32,
        angles: torch.Tensor = None,
        volume: Volume3D = None,
        exec_cfg: cuda_backend.ExecCfg = None,
    ):
        r"""FDK reconstruction for circular cone-beam projections.

        The expected sinogram shape is ``[..., angles, det_v, det_u]``. The
        returned tensor has shape ``[..., depth, height, width]``.
        """
        sinogram = self.fdk_preweight(sinogram)
        sinogram = self.filter_sinogram(
            sinogram,
            filter_name=filter_name,
            v_chunk_size=v_chunk_size,
        )

        reconstruction = self.backward(
            sinogram,
            angles=angles,
            volume=volume,
            exec_cfg=exec_cfg,
        )

        # The backprojection kernel divides by det_spacing_u * det_spacing_v
        # internally. The ramp filter is defined on physical detector-u
        # coordinates, so the remaining det_spacing_v factor is applied here
        # (the det_spacing_u factor cancels with the kernel's scaling).
        source_to_detector = self.src_dist + self.det_dist
        bp_scale = (
            self.det_spacing_v
            * (self.src_dist / source_to_detector) ** 2
        )
        return reconstruction * bp_scale


expose_projection_attributes(
    ConeBeam,
    [
        "det_count_u",
        "det_count_v",
        "det_spacing_u",
        "det_spacing_v",
        ("src_dist", "s_dist"),
        ("det_dist", "d_dist"),
        "pitch",
        ("base_z", "initial_z"),
    ],
)
