import os

if os.getenv("TORCH_RADON_DOC_BUILD"):
    print("Not importing cuda backend because this is just the doc build")

    class VolumeCfg:
        """A placeholder for the pybind11 VolumeCfg class"""
        pass

    class ProjectionCfg:
        """A placeholder for the pybind11 ProjectionCfg class"""

        pass

    class TextureCache:
        """A placeholder for the pybind11 TextureCfg class"""

        pass

    class ExecCfg:
        """A placeholder for the pybind11 ExecCfg class"""

        pass
else:
    from torch_radon_cuda import *
