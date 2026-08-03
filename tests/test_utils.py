import torch

from torch_radon.utils import ShapeNormalizer


def test_tuple_input_returns_tuple():
    """ShapeNormalizer.unnormalize returns a tuple (not a list) for tuple input."""
    normalizer = ShapeNormalizer(2)
    x = torch.randn(2, 3, 16, 16)
    flat = normalizer.normalize(x)

    out = normalizer.unnormalize((flat, flat))

    assert isinstance(out, tuple)
    assert out[0].shape == (2, 3, 16, 16)
    assert out[1].shape == (2, 3, 16, 16)
