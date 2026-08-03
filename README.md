`carterbox/torch-radon` is a fork of `matteo-ronchetti/torch-radon` with some
modules removed (shearlets, reconstruction) and the build system replaced with
the extension system from PyTorch. This fork is maintained separately because
the upstream project is unmaintained. If the upstream project becomes active
again, this fork will attempt to merge its improvements upstream.

# TorchRadon: Fast Differentiable Routines for Computed Tomography

TorchRadon is a PyTorch extension written in CUDA that implements
differentiable routines for solving computed tomography (CT) reconstruction
problems.

The library is designed to help researchers working on CT problems to combine
deep learning and model-based approaches.

Main features:
 - Forward projections, back projections and FDK reconstruction are
 **differentiable** and integrated with PyTorch `.backward()` .
 - Up to 125x **faster** than Astra Toolbox.
 - **Batch operations**: fully exploit the power of modern GPUs by processing
 multiple images in parallel.
 - **Transparent API**: all operations are seamlessly integrated with PyTorch,
  gradients can  be  computed using `.backward()` , half precision can be used
  with Nvidia AMP.
 - **Half precision**: storing data in half precision allows to get sensible
 speedups when  doing  Radon  forward  and  backward projections with a very
 small accuracy loss.

Implemented operations:
 - Parallel Beam projections
 - Fan Beam projections
 - 3D Cone Beam projections
 - FDK reconstruction for circular Cone Beam CT

## Cone Beam FDK

`ConeBeam` provides an FDK helper for circular cone-beam geometries:

```python
volume = torch_radon.Volume3D(voxel_size=(sx, sy, sz))
volume.set_size(depth, height, width)
radon = torch_radon.ConeBeam(
    det_count_u,
    angles,
    src_dist=source_to_origin,
    det_dist=origin_to_detector,
    det_count_v=det_count_v,
    det_spacing_u=det_spacing_u,
    det_spacing_v=det_spacing_v,
    volume=volume,
)

projections = radon.forward(x)
reconstruction = radon.fdk(projections, filter_name="ramp", v_chunk_size=32)
```

The cone-beam filter is applied along the detector-u axis. `v_chunk_size`
controls detector-v chunking during filtering, reducing FFT peak memory while
preserving numerical equivalence with full-volume filtering.

## Speed

TorchRadon is much faster than competing libraries:

![benchmark](https://raw.githubusercontent.com/matteo-ronchetti/tomography-benchmarks/master/figures/tesla_t4_barplot.png)

See the [Tomography Benchmarks
repository](https://github.com/matteo-ronchetti/tomography-benchmarks) for more
detailed benchmarks.

## Installation

Currently only Linux is supported. Windows not supported mainly because there not yet a Windows package for PyTorch on the conda-forge channel.

## Install via the Conda package manager and the conda-forge channel

Please read about how to setup and use the conda package manager before attempting the following command.

```bash
conda install --channel conda-forge carterbox-torch-radon
```

No PYPI packages will be provided because pip was not designed for mixed-language software distribution.

## Install from source

Source builds require a local CUDA toolkit whose version matches the CUDA
version used by your PyTorch installation. Check the PyTorch CUDA version first:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

Then select the matching CUDA toolkit and install without build isolation, so
the extension is compiled against the PyTorch package in your active
environment:

```bash
git clone https://github.com/carterbox/torch-radon.git
cd torch-radon

# Example for PyTorch built with CUDA 12.4:
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"

python -m pip install --no-build-isolation -e .
```

## Cite

If you are using TorchRadon in your research, please cite the following paper:
```

@article{torch_radon,
Author = {Matteo Ronchetti},
Title = {TorchRadon: Fast Differentiable Routines for Computed Tomography},
Year = {2020},
Eprint = {arXiv:2009.14788},
journal={arXiv preprint arXiv:2009.14788},
}

```

## Testing

Install testing dependencies with `pip install .[testing]`
then test with:
```shell script
pytest tests/
```
