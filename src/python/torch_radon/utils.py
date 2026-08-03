import torch


def _normalize_shape(x: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Size]:
    old_shape = x.size()[:-d]
    x = x.view(-1, *(x.size()[-d:]))
    return x, old_shape


def _unnormalize_shape(
    y: torch.Tensor | tuple[torch.Tensor, ...],
    old_shape: torch.Size,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(y, torch.Tensor):
        y = y.view(*old_shape, *(y.size()[1:]))
    elif isinstance(y, tuple):
        y = tuple(yy.view(*old_shape, *(yy.size()[1:])) for yy in y)

    return y


class ShapeNormalizer:
    """Stateful helper that flattens and restores tensor batch dimensions.

    Many of the CUDA kernels operate on a single flat batch axis, but users may
    pass tensors with several leading batch dimensions (e.g. channels). This
    class collapses those leading dimensions into one before the kernel runs
    (:meth:`normalize`) and expands them back afterward (:meth:`unnormalize`).

    Attributes:
        d: Number of trailing (spatial) dimensions kept separate from the batch.
        old_shape: The batch dimensions saved by the last call to
            :meth:`normalize`, used by :meth:`unnormalize` to restore the
            original shape. ``None`` until :meth:`normalize` has been called.
    """

    def __init__(self, d: int):
        """Initialize the normalizer.

        Args:
            d: Number of trailing (spatial) dimensions to keep separate from
                the flattened batch axis.
        """
        self.d = d
        self.old_shape = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the leading batch dimensions of ``x`` into a single axis.

        The original batch shape is stored internally so that
        :meth:`unnormalize` can restore it later.

        Args:
            x: Tensor whose last ``self.d`` dimensions are spatial.

        Returns:
            The reshaped tensor with a single leading batch dimension.
        """
        x, self.old_shape = _normalize_shape(x, self.d)
        return x

    def unnormalize(
        self,
        y: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Restore the batch dimensions saved by the last call to :meth:`normalize`.

        Args:
            y: The output data, either a single tensor or a tuple of tensors.

        Returns:
            The reshaped data with the same container type as ``y``.

        Raises:
            Exception: If called before :meth:`normalize`.
        """
        if self.old_shape is None:
            raise Exception("Calling `unnormalize` before `normalize` ")

        return _unnormalize_shape(y, self.old_shape)


def normalize_shape(d: int):
    """Decorator that flattens and restores tensor batch dimensions around ``f``.

    This is a stateless alternative to :class:`ShapeNormalizer` for methods whose
    only work between normalizing and unnormalizing is the call to ``f`` itself.
    A tensor with shape ``(batch_1, ..., batch_n, s_1, ..., s_d)`` is reshaped to
    ``(batch, s_1, s_2, ...., s_d)``, fed to ``f``, and the output is reshaped to
    ``(batch_1, ..., batch_n, s_1, ..., s_o)``.

    Use :class:`ShapeNormalizer` instead when other work must happen between the
    normalize and unnormalize steps.

    :param d: Number of non-batch dimensions.
    """

    def wrap(f):
        def wrapped(self, x: torch.Tensor, *args, **kwargs):
            x, old_shape = _normalize_shape(x, d)

            y = f(self, x, *args, **kwargs)

            return _unnormalize_shape(y, old_shape)

        wrapped.__doc__ = f.__doc__
        return wrapped

    return wrap


def projection_property_maker(name):
    @property
    def prop(self):
        return getattr(self.projection.cfg, name)

    @prop.setter
    def prop(self, value):
        setattr(self.projection.cfg, name, value)

    return prop


def expose_projection_attributes(pyclass, attributes: list):
    """Exposes the attributes of the projection directly from the class.
    For example, after exposing "det_spacing_u" (internal name) as "det_spacing" (exposed name), setting radon.det_spacing = 32
    is equivalent to setting  radon.projection.cfg.det_spacing_u = 32

    Args:
        pyclass: A python class (not instance of a class but the actual class)
        attributes: List of attributes, each element can be a string (then exposed_name = internal_name) or a tuple (exposed_name, internal_name)
    """
    for x in attributes:
        exposed_name, internal_name = x if isinstance(x, tuple) else (x, x)
        setattr(pyclass, exposed_name, projection_property_maker(internal_name))
