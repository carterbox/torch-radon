#pragma once

#include "cfg.h"
#include "texture.h"

namespace radon {

template<typename T>
void
forward_cuda(const T* x,
             const float* angles,
             T* y,
             TextureCache& tex_cache,
             const VolumeCfg& vol_cfg,
             const ProjectionCfg& proj_cfg,
             const ExecCfg& exec_cfg,
             const int batch_size,
             const int device,
             const int channels,
             const int angle_batch_size);

template<typename T>
void
forward_cuda_3d(const T* x,
                const float* angles,
                T* y,
                TextureCache& tex_cache,
                const VolumeCfg& vol_cfg,
                const ProjectionCfg& proj_cfg,
                const ExecCfg& exec_cfg,
                const int batch_size,
                const int channels,
                const int device,
                const int angle_batch_size);

template<typename T>
void
backward_cuda(const T* x,
              const float* angles,
              T* y,
              TextureCache& tex_cache,
              const VolumeCfg& vol_cfg,
              const ProjectionCfg& proj_cfg,
              const ExecCfg& exec_cfg,
              const int batch_size,
              const int device);

template<typename T>
void
backward_cuda_3d(const T* x,
                 const float* angles,
                 T* y,
                 TextureCache& tex_cache,
                 const VolumeCfg& vol_cfg,
                 const ProjectionCfg& proj_cfg,
                 const ExecCfg& exec_cfg,
                 const int batch_size,
                 const int device);

} // namespace radon
