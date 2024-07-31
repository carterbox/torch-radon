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
                "src/backward.cu",
                "src/forward.cu",
                "src/log.cpp",
                "src/noise.cu",
                "src/cfg.cu",
                "src/pytorch.cpp",
                "src/symbolic.cpp",
                "src/texture.cu",
            ],
            include_dirs=[os.path.abspath("include")],
            extra_compile_args={
                "cxx": [
                    # GNU++ >=14 required hexfloat extension in rmath.h
                    # C++ >=17 required pytorch>=2.1
                    "-std=c++17",
                    "-fvisibility=hidden",
                ],
                "nvcc": [
                    # __half conversions required in backprojection
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            },
            libraries=[
                'curand',
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
