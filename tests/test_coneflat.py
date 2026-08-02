import astra
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import torch_radon

from .utils import relative_error, assert_less

device = torch.device('cuda')

full_angles = np.linspace(0, 2*np.pi, 128).astype(np.float32)
many_angles = np.linspace(0, np.pi, 90).astype(np.float32)

params = []
for batch_size in [1, 8]:
    for volume_size in [64, 81]:
        for angles in [full_angles, many_angles]:
            for spacing in [1.0, 0.5, 1.3, 2.0]:
                for distances in [(1.5, 1.5), (2.0, 2.0), (1.2, 3.0)]:
                    for det_count in [1.0, 1.5]:
                        params.append((device, batch_size, volume_size, angles, spacing, distances, det_count))

half_params = [x for x in params if x[1] % 4 == 0]


def center_of_mass_3d(x):
    z, y, x_idx = np.indices(x.shape)
    weights = np.maximum(x, 0)
    total = weights.sum()
    return np.array([
        (z * weights).sum() / total,
        (y * weights).sum() / total,
        (x_idx * weights).sum() / total,
    ])


@pytest.mark.parametrize('voxel_size', [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.3, 0.7, 2.1)])
def test_coneflat_backprojection_respects_voxel_size(voxel_size):
    volume_size = 32
    det_count = 48
    angles = np.linspace(0, 2*np.pi, 48, endpoint=False).astype(np.float32)

    volume = torch_radon.volumes.Volume3D(voxel_size=voxel_size)
    volume.set_size(volume_size, volume_size, volume_size)
    radon = torch_radon.ConeBeam(
        det_count,
        angles,
        src_dist=volume_size * 6,
        det_dist=volume_size * 3,
        det_count_v=det_count,
        det_spacing_u=2.0,
        det_spacing_v=2.0,
        volume=volume,
    )

    x = torch.zeros(1, volume_size, volume_size, volume_size, device=device)
    x[:, 14:18, 14:18, 14:18] = 1.0

    y = radon.forward(x)
    bp = radon.backward(y).detach().cpu().numpy()[0]

    np.testing.assert_allclose(center_of_mass_3d(bp), np.array([15.5, 15.5, 15.5]), atol=0.08)


@pytest.mark.parametrize('filter_name', ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'])
def test_coneflat_filter_chunking_matches_full(filter_name):
    volume_size = 16
    det_count_u = 31
    det_count_v = 17
    angles = np.linspace(0, 2*np.pi, 21, endpoint=False).astype(np.float32)

    volume = torch_radon.volumes.Volume3D()
    volume.set_size(volume_size, volume_size, volume_size)
    radon = torch_radon.ConeBeam(
        det_count_u,
        angles,
        src_dist=volume_size * 4,
        det_dist=volume_size * 2,
        det_count_v=det_count_v,
        det_spacing_u=1.3,
        det_spacing_v=1.1,
        volume=volume,
    )

    sinogram = torch.randn(2, len(angles), det_count_v, det_count_u, device=device)
    full = radon.filter_sinogram(sinogram, filter_name=filter_name, v_chunk_size=None)
    chunked = radon.filter_sinogram(sinogram, filter_name=filter_name, v_chunk_size=4)

    torch.testing.assert_close(chunked, full, rtol=0, atol=0)


@pytest.mark.parametrize('voxel_size', [1.0, 2.0])
def test_coneflat_fdk_matches_astra_scale(voxel_size):
    volume_size = 32
    det_count = 48
    angles = np.linspace(0, 2*np.pi, 90, endpoint=False).astype(np.float32)
    src_dist = volume_size * 4.0 * voxel_size
    det_dist = volume_size * 2.0 * voxel_size
    det_spacing = 1.5 * voxel_size

    z, y, x = np.indices((volume_size, volume_size, volume_size))
    center = (volume_size - 1) / 2
    phantom = (
        ((x - center)**2 + ((y - center) * 1.2)**2 + ((z - center) * 0.8)**2)
        < (volume_size * 0.22)**2
    ).astype(np.float32)
    phantom += 0.5 * (
        (((x - (center + 4)) * 1.2)**2 + (y - (center - 3))**2 + ((z - center) * 1.1)**2)
        < (volume_size * 0.13)**2
    ).astype(np.float32)

    volume_half_size = volume_size * voxel_size / 2
    vol_geom = astra.create_vol_geom(
        volume_size,
        volume_size,
        volume_size,
        -volume_half_size,
        volume_half_size,
        -volume_half_size,
        volume_half_size,
        -volume_half_size,
        volume_half_size,
    )
    proj_geom = astra.create_proj_geom(
        'cone',
        det_spacing,
        det_spacing,
        det_count,
        det_count,
        angles,
        src_dist,
        det_dist,
    )
    proj_id, astra_y = astra.create_sino3d_gpu(phantom, proj_geom, vol_geom)
    rec_id = astra.data3d.create('-vol', vol_geom)
    alg_id = None
    try:
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        astra_rec = astra.data3d.get(rec_id)
    finally:
        if alg_id is not None:
            astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)

    volume = torch_radon.volumes.Volume3D(voxel_size=(voxel_size, voxel_size, voxel_size))
    volume.set_size(volume_size, volume_size, volume_size)
    radon = torch_radon.ConeBeam(
        det_count,
        angles,
        src_dist,
        det_dist,
        det_count_v=det_count,
        det_spacing_u=det_spacing,
        det_spacing_v=det_spacing,
        volume=volume,
    )

    torch_phantom = torch.tensor(phantom, device=device).unsqueeze(0)
    torch_rec = radon.fdk(radon.forward(torch_phantom), v_chunk_size=8).detach().cpu().numpy()[0]

    fdk_error = relative_error(astra_rec, torch_rec)
    scale = np.sum(torch_rec * astra_rec) / (np.sum(torch_rec * torch_rec) + 1e-12)

    assert_less(fdk_error, 3e-3)
    np.testing.assert_allclose(scale, 1.0, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize('device, batch_size, volume_size, angles, det_spacing, distances, det_count', params)
def test_fanflat_error(device, batch_size, volume_size, angles, det_spacing, distances, det_count):
    # generate random images
    det_count = int(det_count * volume_size)
    x = np.random.uniform(0.0, 1.0, (volume_size, volume_size, volume_size)).astype(np.float32)

    s_dist, d_dist = distances
    s_dist *= volume_size
    d_dist *= volume_size

    # astra
    vol_geom = astra.create_vol_geom(x.shape[1], x.shape[2], x.shape[0])
    proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, det_count, det_count, angles, s_dist, d_dist)

    proj_id, astra_y = astra.create_sino3d_gpu(x, proj_geom, vol_geom)

    rec_id = astra.data3d.create('-vol', vol_geom)

    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)

    astra_y = astra_y.transpose(1, 0, 2)
    astra_bp = astra.data3d.get(rec_id)

    # TODO clean astra structures

    # our implementation
    volume = torch_radon.volumes.Volume3D()
    volume.set_size(volume_size, volume_size, volume_size)
    radon = torch_radon.ConeBeam(det_count, angles, s_dist, d_dist, det_spacing_u=det_spacing, volume=volume)
    x = torch.FloatTensor(x).view(1, x.shape[0], x.shape[1], x.shape[2]).repeat(batch_size, 1, 1, 1).to(device)

    our_fp = radon.forward(x)
    our_bp = radon.backward(our_fp)

    our_fp = our_fp.cpu().numpy()
    batch_error = max([relative_error(our_fp[0], our_fp[i]) for i in range(1, batch_size)] + [0])
    forward_error = relative_error(astra_y, our_fp[0])

    our_bp = our_bp.cpu().numpy()
    batch_error_back = max([relative_error(our_bp[0], our_bp[i]) for i in range(1, batch_size)] + [0])
    back_error = relative_error(astra_bp, our_bp[0])

    if not forward_error < 2e-2:
        fig, ax = plt.subplots(3, 3)
        ax = ax.ravel()
        ax[0].imshow(astra_y[0])
        ax[1].imshow(our_fp[0, 0])
        ax[2].imshow(np.abs(our_fp[0, 0] - astra_y[0]))
        ax[3].imshow(astra_y[len(angles)//2])
        ax[4].imshow(our_fp[0, len(angles)//2])
        ax[5].imshow(np.abs(our_fp[0, len(angles)//2] - astra_y[len(angles)//2]))
        ax[6].imshow(astra_y[-1])
        ax[7].imshow(our_fp[0, -1])
        ax[8].imshow(np.abs(our_fp[0, -1] - astra_y[-1]))
        plt.show()

    print(f"batch: {batch_size}, size: {volume_size}, angles: {len(angles)}, spacing: {det_spacing}, distances: {distances}, det_count:{det_count}, forward: {forward_error}, back: {back_error}")

    # TODO better checks
    assert_less(batch_error, 1e-6)
    assert_less(forward_error, 2e-2)
    assert_less(batch_error_back, 1e-6)
    assert_less(back_error, 3e-3)


@pytest.mark.parametrize('device, batch_size, volume_size, angles, det_spacing, distances, det_count', half_params)
def test_half(device, batch_size, volume_size, angles, det_spacing, distances, det_count):
    # generate random images
    det_count = int(det_count * volume_size)
    x = np.random.uniform(0.0, 1.0, (batch_size, volume_size, volume_size, volume_size)).astype(np.float32)

    s_dist, d_dist = distances
    s_dist *= volume_size
    d_dist *= volume_size

    volume = torch_radon.volumes.Volume3D()
    volume.set_size(volume_size, volume_size, volume_size)
    radon = torch_radon.ConeBeam(det_count, angles, s_dist, d_dist, det_spacing_u=det_spacing, volume=volume)
    x = torch.FloatTensor(x).to(device)

    single_fp = radon.forward(x) / len(angles)
    single_bp = radon.backward(single_fp)

    half_fp = radon.forward(x.half()) / len(angles)
    half_bp = radon.backward(half_fp)

    forward_error = relative_error(single_fp.cpu().numpy(), half_fp.float().cpu().numpy())
    back_error = relative_error(single_bp.cpu().numpy(), half_bp.float().cpu().numpy())

    print(f"batch: {batch_size}, size: {volume_size}, angles: {len(angles)}, spacing: {det_spacing}, distances: {distances}, det_count:{det_count}, forward: {forward_error}, back: {back_error}")

    # TODO better checks
    assert_less(forward_error, 3e-3)
    assert_less(back_error, 3e-3)
