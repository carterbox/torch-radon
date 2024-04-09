import torch

from . import cuda_backend


def _generate_config(
    proj_cfg: cuda_backend.ProjectionCfg,
    is_half: bool,
):
    if proj_cfg.projection_type == 2:
        ch = 4 if is_half else 1
        return cuda_backend.ExecCfg(8, 16, 8, ch)

    return cuda_backend.ExecCfg(16, 16, 1, 4)


class RadonForward(torch.autograd.Function):
    """Perform the forward Radon transformtion from real to sinogram space."""

    @staticmethod
    def forward(
        ctx,
        image: torch.Tensor,
        angles: torch.Tensor,
        tex_cache: cuda_backend.TextureCache,
        vol_cfg: cuda_backend.VolumeCfg,
        proj_cfg: cuda_backend.ProjectionCfg,
    ):
        exec_cfg = _generate_config(
            ctx.proj_cfg,
            image.dtype == torch.half,
        )
        sinogram = cuda_backend.forward(
            image,
            angles,
            tex_cache,
            vol_cfg,
            proj_cfg,
            exec_cfg,
        )
        ctx.tex_cache = tex_cache
        ctx.vol_cfg = vol_cfg
        ctx.proj_cfg = proj_cfg.copy()
        ctx.save_for_backward(angles)

        return sinogram

    @staticmethod
    def backward(
        ctx,
        grad_sinogram: torch.Tensor,
    ):
        (angles,) = ctx.saved_tensors
        exec_cfg = _generate_config(
            ctx.proj_cfg,
            grad_sinogram.dtype == torch.half,
        )
        grad_image = cuda_backend.backward(
            grad_sinogram,
            angles,
            ctx.tex_cache,
            ctx.vol_cfg,
            ctx.proj_cfg,
            exec_cfg,
        )
        return grad_image, None, None, None, None, None, None


class RadonBackprojection(torch.autograd.Function):
    """Perform the adjoint Radon transformation from sinogram to real space.

    Parameters
    ----------
    sinogram : (batch, channels, angles, width)
    angles : (batch, angles)

    """

    @staticmethod
    def forward(
        ctx,
        sinogram: torch.Tensor,
        angles: torch.Tensor,
        tex_cache: cuda_backend.TextureCache,
        vol_cfg: cuda_backend.VolumeCfg,
        proj_cfg: cuda_backend.ProjectionCfg,
    ):
        exec_cfg = _generate_config(
            ctx.proj_cfg,
            sinogram.dtype == torch.half,
        )
        image = cuda_backend.backward(
            sinogram,
            angles,
            tex_cache,
            vol_cfg,
            proj_cfg,
            exec_cfg,
        )
        ctx.tex_cache = tex_cache
        ctx.vol_cfg = vol_cfg
        ctx.proj_cfg = proj_cfg.copy()
        ctx.save_for_backward(angles)

        return image

    @staticmethod
    def backward(
        ctx,
        grad_image: torch.Tensor,
    ):
        (angles,) = ctx.saved_tensors
        exec_cfg = _generate_config(
            ctx.proj_cfg,
            grad_image.dtype == torch.half,
        )
        grad_sinogram = cuda_backend.forward(
            grad_image,
            angles,
            ctx.tex_cache,
            ctx.vol_cfg,
            ctx.proj_cfg,
            exec_cfg,
        )
        return grad_sinogram, None, None, None, None, None, None
