import numpy as np
import torch
import pytest
import random
import itertools

import torch_radon

# torch_radon.cuda_backend.set_log_level(0)

from .utils import (
    random_symbolic_function,
    symbolic_discretize,
    symbolic_forward,
    TestHelper,
    assert_equal,
)

random.seed(42)
device = torch.device("cuda")
test_helper = TestHelper("parallel_beam")

# (batch_size, angles, volume, spacing, det_count)
params = []

# check different batch sizes
for batch_size in [1, 3, 17, 32]:
    params.append((batch_size, torch.linspace(0, np.pi, 128), None, 1.0, 128))

# check few and many angles which are not multiples of 16
for angles in [torch.linspace(0, np.pi, 19), torch.linspace(0, np.pi, 803)]:
    params.append((4, angles, None, 1.0, 128))

# change volume size
for height, width in [(128, 256), (256, 128), (75, 149), (81, 81)]:
    s = max(height, width)
    volume = torch_radon.volumes.Volume2D()
    volume.set_size(height, width)
    params.append((4, torch.linspace(0, np.pi, 64), volume, 2.0, s))

# change volume scale and center
for center in [(0, 0), (17, -25), (53, 49)]:
    for voxel_size in [(1, 1), (0.75, 0.75), (1.5, 1.5), (0.7, 1.3), (1.3, 0.7)]:
        det_count = int(
            179 * max(voxel_size[0], 1) * max(voxel_size[1], 1) * np.sqrt(2)
        )
        volume = torch_radon.volumes.Volume2D(center, voxel_size)
        volume.set_size(179, 123)
        params.append((4, torch.linspace(0, np.pi, 128), volume, 2.0, det_count))

for spacing in [1.0, 0.5, 1.3, 2.0]:
    for det_count in [79, 128, 243]:
        for src_dist, det_dist in [(128, 128), (64, 128), (128, 64), (503, 503)]:
            volume = torch_radon.volumes.Volume2D()
            volume.set_size(128, 128)
            params.append(
                (4, torch.linspace(0, np.pi, 128), volume, spacing, det_count)
            )


@pytest.mark.parametrize("batch_size, angles, volume, spacing, det_count", params)
def test_error(batch_size, angles, volume, spacing, det_count):
    if volume is None:
        volume = torch_radon.volumes.Volume2D()
        volume.set_size(det_count, det_count)

    radon = torch_radon.ParallelBeam(
        det_count,
        angles,
        spacing,
        volume,
    )
    angles = angles.to(device)

    f = random_symbolic_function(radon.volume.height, radon.volume.width)
    x = symbolic_discretize(f, radon.volume.height, radon.volume.width)

    f.scale(*radon.volume.voxel_size)
    f.move(*radon.volume.center)

    tx = torch.FloatTensor(x).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    y = symbolic_forward(f, angles.cpu(), radon.projection.cfg).cpu().numpy()
    ty = radon.forward(tx, angles)
    assert_equal(ty.size(0), batch_size)

    max_error = 2e-3 * (512 / y.shape[0]) * (512 / y.shape[1])

    description = f"Angles: {angles}\nVolume: {volume}\nSpacing: {spacing}, Count: {det_count}, Precision: float"
    test_helper.compare_images(y, ty, max_error, description)

    back_max_error = 1e-3
    test_helper.backward_check(tx, ty, radon, description, back_max_error, angles)

    if batch_size % 4 == 0:
        ty = radon.forward(tx.half(), angles)

        description = f"Angles: {angles}\nVolume: {volume}\nSpacing: {spacing}, Count: {det_count}, Precision: half"
        test_helper.compare_images(y, ty, max_error, description)


def test_simple_integrals(image_size=17):
    """Check that the forward radon operator works correctly at 0 and PI/2.

    When we project at angles 0 and PI/2, the foward operator should be the
    same as taking the sum over the object array along each axis.
    """
    volume = torch_radon.volumes.Volume2D()
    volume.set_size(image_size, image_size)

    angles = torch.tensor(
        [0.0, np.pi, -np.pi / 2, np.pi / 2],
        dtype=torch.float32,
        device="cuda",
    )
    radon = torch_radon.ParallelBeam(
        volume=volume,
        angles=angles,
        det_spacing=1.0,
        det_count=image_size,
    )

    original = torch.zeros(
        image_size,
        image_size,
        dtype=torch.float32,
        device="cuda",
    )
    original[image_size // 4, :] += 1
    original[:, image_size // 2] += 1

    data = radon.forward(original, angles)
    data0 = torch.sum(original, axis=0)
    data1 = torch.sum(original, axis=1)

    print("\n", data0.cpu().numpy())
    print("\n", data[0].cpu().numpy())
    print("\n", data[1].cpu().numpy())
    torch.testing.assert_allclose(data[0], data0)
    torch.testing.assert_allclose(data[1], data0)
    print("\n")
    print("\n", data1.cpu().numpy())
    print("\n", data[2].cpu().numpy())
    print("\n", data[3].cpu().numpy())
    torch.testing.assert_allclose(data[2], data1)
    torch.testing.assert_allclose(data[3], torch.flip(data1, (0,)))


def manual_integrals(x):
    angles = torch.tensor(
        # [
        # [0, 0],
        #     [0, -torch.pi / 2],
        #     [-torch.pi / 2, 0],
        [-torch.pi / 2, -torch.pi / 2],
        # ],
    ).float()

    integrals = torch.empty(
        (*x.shape[:-2], 2, x.shape[-1]), dtype=x.dtype, layout=x.layout, device=x.device
    )

    for b in range(x.shape[0]):
        # if b % 4 == 0:
        # integrals[b, :, 0, :] = torch.sum(x[b], dim=-2)  # pi
        # integrals[b, :, 1, :] = torch.sum(x[b], dim=-2)  # pi
        # if b % 4 == 1:
        #     integrals[b, :, 0, :] = torch.sum(x[b], dim=-2)  # pi
        #     integrals[b, :, 1, :] = torch.sum(x[b], dim=-1)  # pi/2
        # if b % 4 == 2:
        #     integrals[b, :, 0, :] = torch.sum(x[b], dim=-1)  # pi/2
        #     integrals[b, :, 1, :] = torch.sum(x[b], dim=-2)  # pi
        # if b % 4 == 3:
        integrals[b, :, 0, :] = torch.sum(x[b], dim=-1)  # pi/2
        integrals[b, :, 1, :] = torch.sum(x[b], dim=-1)  # pi/2

    assert integrals.shape == (*x.shape[:-2], 2, x.shape[-1])

    return angles, integrals


@pytest.mark.parametrize(
    "batch,channel,dtype",
    itertools.product(
        [
            1,
            # 3,
            4,
            8,
        ],
        [
            1,
            # 3,
            # 4,
            # 8,
        ],
        [
            "float",
            # "half",
        ],
    ),
)
def test_complex_integrals(batch, channel, dtype):
    image_size = 3
    convert = dict(float=torch.float, half=torch.half)

    x = (
        torch.arange(0, batch * channel * image_size * image_size)
        .reshape(batch, channel, image_size, image_size)
        .type(convert[dtype])
    )
    # for c in range(x.shape[1]):
    #     x[:, c] = c+1

    angles, integrals = manual_integrals(x)

    x = x.to("cuda")
    angles = angles.to("cuda")

    volume = torch_radon.volumes.Volume2D()
    volume.set_size(image_size, image_size)

    radon = torch_radon.ParallelBeam(
        volume=volume,
        angles=angles,
        det_spacing=1.0,
        det_count=image_size,
    )
    data = radon.forward(x, angles=angles).to("cpu")

    print()
    print("integrals - radon")
    for b in range(batch):
        for c in range(channel):
            print(f"batch: {b}, channel: {c}")
            for a in range(2):
                print(f"  {integrals[b, c, a]} - {data[b, c, a]}")
    assert data.shape == integrals.shape
    assert torch.equal(data, integrals)


def manual_back(s):
    angles = torch.tensor(
        # [
        [0, 0],
        #     [0, -torch.pi / 2],
        #     [-torch.pi / 2, 0],
        #     [-torch.pi / 2, -torch.pi / 2],
        # ],
    ).float()

    images = torch.zeros(
        (*s.shape[:-2], s.shape[-1], s.shape[-1]),
        dtype=s.dtype,
        layout=s.layout,
        device=s.device,
    )

    for b in range(s.shape[0]):
        # if b % len(angles) == 0:
        images[b, :, :, :] += s[b, :, 0:1, :]  # pi
        images[b, :, :, :] += s[b, :, 1:2, :]  # pi
    # if b % len(angles) == 1:
    #     images[b, :, :, :] += s[b, :, 0:1, :]  # pi
    #     images[b, :, :, :] += s[b, :, 1, :][..., None]  # pi/2
    # if b % len(angles) == 2:
    #     images[b, :, :, :] += s[b, :, 0, :][..., None]  # pi/2
    #     images[b, :, :, :] += s[b, :, 1:2, :]  # pi
    # if b % len(angles) == 3:
    #     images[b, :, :, :] += s[b, :, 0, :][..., None]  # pi/2
    #     images[b, :, :, :] += s[b, :, 1, :][..., None]  # pi/2

    return angles, images


@pytest.mark.parametrize(
    "batch,channel,dtype",
    itertools.product(
        [
            1,
            # 3,
            4,
            8,
        ],
        [
            1,
            # 3,
            # 4,
            # 8,
        ],
        [
            "float",
            # "half",
        ],
    ),
)
def test_complex_back(batch, channel, dtype):
    image_size = 3
    num_angles = 2
    convert = dict(float=torch.float, half=torch.half)

    sino = (
        torch.randperm(batch * channel * num_angles * image_size)
        .reshape(batch, channel, num_angles, image_size)
        .type(convert[dtype])
    )
    # sino[:] = 0
    # sino[..., 1] = 1
    # sino[..., 2] = -2

    angles, images = manual_back(sino)

    sino = sino.to("cuda")
    angles = angles.to("cuda")

    volume = torch_radon.volumes.Volume2D()
    volume.set_size(image_size, image_size)

    radon = torch_radon.ParallelBeam(
        angles=angles,
        volume=volume,
        det_spacing=1.0,
        det_count=image_size,
    )
    back = radon.backward(sino, angles=angles).to("cpu")

    print()
    print("back - radon")
    for b in range(batch):
        for c in range(channel):
            print(f"batch: {b}, channel: {c}")
            for w in range(image_size):
                print(f"  {images[b, c, w, :]} - {back[b, c, w, :]}")
    assert back.shape == images.shape
    assert torch.equal(back, images)


def test_simple_back(image_size=5):
    data = torch.zeros(4, image_size, device="cuda")
    data[:, image_size // 4] = torch.tensor([1, 2, 3, 4], device="cuda")

    volume = torch_radon.volumes.Volume2D()
    volume.set_size(image_size, image_size)
    angles = torch.tensor(
        [0.0, np.pi / 2, np.pi, -np.pi / 2],
        dtype=torch.float32,
        device="cuda",
    )
    radon = torch_radon.ParallelBeam(
        angles=angles,
        volume=volume,
        det_spacing=1.0,
        det_count=image_size,
    )

    original = radon.backward(data, angles)
    print("\n", original)

    ref = torch.tensor(
        [
            [0.0, 1.0, 0.0, 3.0, 0.0],
            [4.0, 5.0, 4.0, 7.0, 4.0],
            [0.0, 1.0, 0.0, 3.0, 0.0],
            [2.0, 3.0, 2.0, 5.0, 2.0],
            [0.0, 1.0, 0.0, 3.0, 0.0],
        ],
    )

    assert torch.equal(original.cpu(), ref)


@pytest.mark.parametrize(
    "batch,channel,dtype",
    itertools.product(
        [
            1,
            # 3,
            4,
            8,
        ],
        [
            1,
            # 3,
            # 4,
            # 8,
        ],
        [
            "float",
            # 'half',
        ],
    ),
)
def test_adjoint(batch, channel, dtype):
    """Test whether the backward and forward operators are true adjoints.

    If they are true adjoints then the inner products as computed below should
    be equal. Seems to be not a great adjoint due to only equal up to 2 sigfigs.
    """

    image_size = 64
    nangles = 7
    convert = dict(float=torch.float, half=torch.half)
    device = torch.device("cuda")

    image0 = torch.abs(
        torch.rand(
            (batch, channel, image_size, image_size),
            dtype=convert[dtype],
            device=device,
        )
    )
    sino0 = torch.abs(
        torch.rand(
            (batch, channel, nangles, image_size),
            dtype=convert[dtype],
            device=device,
        )
    )
    angles = torch.linspace(
        start=0,
        end=torch.pi,
        steps=nangles,
        device=device,
        dtype=torch.float32,
    )

    volume = torch_radon.volumes.Volume2D()
    volume.set_size(image_size, image_size)

    radon = torch_radon.ParallelBeam(
        angles=angles,
        volume=volume,
        det_spacing=1.0,
        det_count=image_size,
    )

    sino1 = radon.forward(x=image0, angles=angles)
    assert sino1.shape == sino0.shape
    image1 = radon.backward(sinogram=sino0, angles=angles)
    assert image1.shape == image0.shape
    a = torch.inner(sino1.flatten(), sino0.flatten())
    b = torch.inner(image0.flatten(), image1.flatten())
    print()
    print("<Fm,   m> = {:.5g}{:+.5g}j".format(a.item(), 0))
    print("< d, F*d> = {:.5g}{:+.5g}j".format(b.item(), 0))
