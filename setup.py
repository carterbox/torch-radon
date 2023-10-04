"""Since this package is a pytorch extension, this setup file uses the custom
CUDAExtension build system from pytorch. This ensures that compatible compiler
args, headers, etc for pytorch.

Read more at the pytorch docs:
https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension
"""
import os.path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    packages=["torch_radon"],
    package_dir={
        "": "src/python",
    },
    ext_modules=[
        CUDAExtension(
            name="torch_radon_cuda",
            sources=[
                "src/backprojection.cu",
                "src/fft.cu",
                "src/forward.cu",
                "src/log.cpp",
                "src/noise.cu",
                "src/parameter_classes.cu",
                "src/pytorch.cpp",
                "src/symbolic.cpp",
                "src/texture.cu",
            ],
            include_dirs=[os.path.abspath("include")],
            extra_compile_args={
                "cxx": [
                    # GNU++14 required for hexfloat extension used in rmath.h
                    "-std=gnu++14",
                ],
                "nvcc": [
                    # __half conversions required in backprojection
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
            libraries=[
                'cufft',
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
