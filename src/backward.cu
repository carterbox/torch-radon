#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#include "radon.h"
#include "texture.h"
#include "utils.h"

/**
 * @brief
 *
 * @tparam parallel_beam
 * @tparam texture_channels
 * @tparam T the type of the real-space image
 * @param output The real space image
 * @param texture A texture containing the diffraction patterns
 * @param angles
 * @param vol_cfg
 * @param proj_cfg
 */
// launch expectation is x and y are spread across the x,y dimensions of
// threads and blocks the z dimension is for the batches/channels
// The number of z blocks should equal the number of textures so that the
// threads in a z block have the same angles guarantee i.e. BlockDim.z must be 1
// or zero
template<bool parallel_beam, int texture_channels, typename T>
__global__ void
backward_kernel(T* __restrict__ output,
                cudaTextureObject_t texture,
                const float* __restrict__ angles,
                const VolumeCfg vol_cfg,
                const ProjectionCfg proj_cfg,
                const int angle_batch_size,
                const int n_angles,
                const int real_channels)
{
  // Calculate image coordinates
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  // Linear thread id within a block
  const uint tid = threadIdx.y * blockDim.x + threadIdx.x;

  const float cx = vol_cfg.width / 2.0f;
  const float cy = vol_cfg.height / 2.0f;
  const float cr = proj_cfg.det_count_u / 2.0f;

  const float dx = (float(x) - cx) * vol_cfg.sx + vol_cfg.dx + 0.5f;
  const float dy = (float(y) - cy) * vol_cfg.sy + vol_cfg.dy + 0.5f;

  const float ids = __fdividef(1.0f, proj_cfg.det_spacing_u);
  const float sdx = dx * ids;
  const float sdy = dy * ids;

  // (batch, channel, height, width)
  const int texture_id = blockIdx.z; // assert BlockDim.z == 1 or 0
  const int batch_id = texture_id * texture_channels / real_channels;
  const int angle_offset = n_angles * (batch_id % angle_batch_size);
  const int base =
    x + vol_cfg.width * (y + vol_cfg.height * (texture_channels * (texture_id)));
  const int pitch = vol_cfg.width * vol_cfg.height;

  // keep sin and cos packed toghether to save one memory load in the main loop
  // __shared__ memory is shared between threads in a block
  __shared__ float2 sincos[4096];

  for (int i = tid; i < proj_cfg.n_angles; i += blockDim.x * blockDim.y) {
    float2 tmp;
    tmp.x = -__sinf(angles[i + angle_offset]);
    tmp.y = __cosf(angles[i + angle_offset]);
    sincos[i] = tmp;
  }
  __syncthreads();

  if (x < vol_cfg.width && y < vol_cfg.height) {
    float accumulator[texture_channels];
#pragma unroll
    for (int i = 0; i < texture_channels; i++)
      accumulator[i] = 0.0f;

    if (parallel_beam) {
      const int n_angles = proj_cfg.n_angles;

      // keep a float version of i to avoid expensive int2float conversions
      // inside the main loop
      float fi = 0.5f;
#pragma unroll(16)
      for (int i = 0; i < n_angles; i++) {
        float j = sincos[i].y * sdx + sincos[i].x * sdy + cr;
        if (texture_channels == 1) {
          accumulator[0] += tex2DLayered<float>(texture, j, fi, texture_id);
        } else {
          // read 4 values at the given position and accumulate
          float4 read = tex2DLayered<float4>(texture, j, fi, texture_id);
          accumulator[0] += read.x;
          accumulator[1] += read.y;
          accumulator[2] += read.z;
          accumulator[3] += read.w;
        }
        fi += 1.0f;
      }
    } else {
      const float k = proj_cfg.s_dist + proj_cfg.d_dist;
      const int n_angles = proj_cfg.n_angles;

      // keep a float version of i to avoid expensive int2float conversions
      // inside the main loop
      float fi = 0.5f;
#pragma unroll(16)
      for (int i = 0; i < n_angles; i++) {
        float iden;
        float den = fmaf(sincos[i].y, -dy, sincos[i].x * dx + proj_cfg.s_dist);

        // iden = __fdividef(k, den);
        asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(iden) : "f"(k), "f"(den));

        float j = (sincos[i].y * sdx + sincos[i].x * sdy) * iden + cr;

        if (texture_channels == 1) {
          accumulator[0] +=
            tex2DLayered<float>(texture, j, fi, texture_id) * iden;
        } else {
          // read 4 values at the given position and accumulate
          float4 read = tex2DLayered<float4>(texture, j, fi, texture_id);
          accumulator[0] += read.x * iden;
          accumulator[1] += read.y * iden;
          accumulator[2] += read.z * iden;
          accumulator[3] += read.w * iden;
        }
        fi += 1.0f;
      }
    }

#pragma unroll
    for (int b = 0; b < texture_channels; b++) {
      output[base + b * pitch] = accumulator[b] * ids;
    }
  }
}

template<typename T>
void
radon::backward_cuda(const T* x,
                     const float* angles,
                     T* y,
                     TextureCache& tex_cache,
                     const VolumeCfg& vol_cfg,
                     const ProjectionCfg& proj_cfg,
                     const ExecCfg& exec_cfg,
                     const int batch_size,
                     const int channels,
                     const int device,
                     const int angle_batch_size,
                     const int n_angles)
{
  constexpr bool is_float = std::is_same<T, float>::value;
  constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;

  LOG_DEBUG("Radon backward 2D. Height: " << vol_cfg.height
                                          << " width: " << vol_cfg.width
                                          << " channels: " << channels);
  LOG_DEBUG("Radon backward 2D. Det count: "
            << proj_cfg.det_count_u << " angles: " << n_angles
            << " angles_batch_size: " << angle_batch_size
            << " batch_size: " << batch_size);

  // If the number of channels is a multiple of 4, then we can use the texture
  // channels to decrease the thread grid size. NOTE: CUDA also supports
  // textures with 2 channels.
  int texture_channels = 1;
  if (channels % 4 == 0) {
    texture_channels = 4;
  }
  if (texture_channels > 4) {
    throw std::invalid_argument("This more than 4 channels is unsupported!");
  }
  const int grid_size_z = batch_size * channels / texture_channels;

  // copy x into CUDA Array (allocating it if needed) and bind to texture
  Texture* tex = tex_cache.get({ device,
                                 grid_size_z,
                                 proj_cfg.n_angles,
                                 proj_cfg.det_count_u,
                                 true,
                                 texture_channels,
                                 precision });
  tex->put(x);

  // dim3 block_dim(3, 3);
  // dim3 grid_dim(1, 1, grid_size_z);
  dim3 block_dim = exec_cfg.get_block_dim();
  dim3 grid_dim =
    exec_cfg.get_grid_size(vol_cfg.width, vol_cfg.height, grid_size_z);

  LOG_DEBUG("Block Size x:" << block_dim.x << " y:" << block_dim.y
                            << " z:" << block_dim.z);
  LOG_DEBUG("Grid Size x:" << grid_dim.x << " y:" << grid_dim.y
                           << " z:" << grid_dim.z);

  switch (texture_channels) {
    case 1:
      if (proj_cfg.projection_type == FANBEAM) {
        backward_kernel<false, 1, T><<<grid_dim, block_dim>>>(y,
                                                              tex->texture,
                                                              angles,
                                                              vol_cfg,
                                                              proj_cfg,
                                                              angle_batch_size,
                                                              n_angles,
                                                              channels);
      } else {
        backward_kernel<true, 1, T><<<grid_dim, block_dim>>>(y,
                                                             tex->texture,
                                                             angles,
                                                             vol_cfg,
                                                             proj_cfg,
                                                             angle_batch_size,
                                                             n_angles,
                                                             channels);
      }
      break;
    case 4:
      if (proj_cfg.projection_type == FANBEAM) {
        backward_kernel<false, 4, T><<<grid_dim, block_dim>>>(y,
                                                              tex->texture,
                                                              angles,
                                                              vol_cfg,
                                                              proj_cfg,
                                                              angle_batch_size,
                                                              n_angles,
                                                              channels);
      } else {
        backward_kernel<true, 4, T><<<grid_dim, block_dim>>>(y,
                                                             tex->texture,
                                                             angles,
                                                             vol_cfg,
                                                             proj_cfg,
                                                             angle_batch_size,
                                                             n_angles,
                                                             channels);
      }
      break;
    default:
      throw std::invalid_argument("This is an unsupported number of channels!");
  }
}

template void
radon::backward_cuda<float>(const float* x,
                            const float* angles,
                            float* y,
                            TextureCache& tex_cache,
                            const VolumeCfg& vol_cfg,
                            const ProjectionCfg& proj_cfg,
                            const ExecCfg& exec_cfg,
                            const int batch_size,
                            const int channels,
                            const int device,
                            const int angle_batch_size,
                            const int n_angles);
template void
radon::backward_cuda<__half>(const __half* x,
                             const float* angles,
                             __half* y,
                             TextureCache& tex_cache,
                             const VolumeCfg& vol_cfg,
                             const ProjectionCfg& proj_cfg,
                             const ExecCfg& exec_cfg,
                             const int batch_size,
                             const int channels,
                             const int device,
                             const int angle_batch_size,
                             const int n_angles);

template<int channels, typename T>
__global__ void
backward_kernel_3d(T* __restrict__ output,
                   cudaTextureObject_t texture,
                   const float* __restrict__ angles,
                   const VolumeCfg vol_cfg,
                   const ProjectionCfg proj_cfg)
{
  // TODO consider pitch and initial_z
  // Calculate volume coordinates
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint z = blockIdx.z * blockDim.z + threadIdx.z;
  const uint tid =
    (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

  const uint index = (z * vol_cfg.height + y) * vol_cfg.width + x;
  const uint pitch = vol_cfg.depth * vol_cfg.height * vol_cfg.width;

  const float cx = vol_cfg.width / 2.0f;
  const float cy = vol_cfg.height / 2.0f;
  const float cz = vol_cfg.depth / 2.0f;
  const float cu = proj_cfg.det_count_u / 2.0f;
  const float cv = proj_cfg.det_count_v / 2.0f;

  const float dx = (float(x) - cx) * vol_cfg.sx + vol_cfg.dx + 0.5f;
  const float dy = (float(y) - cy) * vol_cfg.sy + vol_cfg.dy + 0.5f;
  const float dz =
    (float(z) - cz) * vol_cfg.sz + vol_cfg.dz + 0.5f - proj_cfg.initial_z;

  const float inv_det_spacing_u = __fdividef(1.0f, proj_cfg.det_spacing_u);
  const float inv_det_spacing_v = __fdividef(1.0f, proj_cfg.det_spacing_v);
  const float ids = inv_det_spacing_u * inv_det_spacing_v;

  const float sdx = dx * inv_det_spacing_u;
  const float sdy = dy * inv_det_spacing_u;
  const float sdz = dz * inv_det_spacing_v;
  const float pitch_speed = -proj_cfg.pitch * 0.1591549f * inv_det_spacing_v;

  // using a single float3 array creates 3 memory loads, while float2+float ==>
  // 2 loads
  __shared__ float2 sincos[4096];
  __shared__ float pitch_dz[4096];

  for (int i = tid; i < proj_cfg.n_angles; i += 256) {
    float2 tmp;
    tmp.x = __sinf(angles[i]);
    tmp.y = __cosf(angles[i]);
    pitch_dz[i] = angles[i] * pitch_speed;
    sincos[i] = tmp;
  }
  __syncthreads();

  if (x < vol_cfg.width && y < vol_cfg.height && z < vol_cfg.depth) {
    float accumulator[channels];

#pragma unroll
    for (int i = 0; i < channels; i++)
      accumulator[i] = 0.0f;

    const float k = proj_cfg.s_dist + proj_cfg.d_dist;

#pragma unroll(4)
    for (int i = 0; i < proj_cfg.n_angles; i++) {
      float alpha = fmaf(-dx, sincos[i].x, proj_cfg.s_dist) + sincos[i].y * dy;
      float beta = sincos[i].y * sdx + sincos[i].x * sdy;

      // float k_over_alpha = __fdividef(k, alpha);
      float k_over_alpha;
      asm("div.approx.ftz.f32 %0, %1, %2;"
          : "=f"(k_over_alpha)
          : "f"(k), "f"(alpha));

      float u = k_over_alpha * beta + cu;
      float v = k_over_alpha * (sdz + pitch_dz[i]) + cv;
      float scale = k_over_alpha * k_over_alpha;

      if (channels == 1) {
        accumulator[0] += tex2DLayered<float>(texture, u, v, i) * scale;
      } else {
        // read 4 values at the given position and accumulate
        float4 read = tex2DLayered<float4>(texture, u, v, i);
        accumulator[0] += read.x * scale;
        accumulator[1] += read.y * scale;
        accumulator[2] += read.z * scale;
        accumulator[3] += read.w * scale;
      }
    }

#pragma unroll
    for (int b = 0; b < channels; b++) {
      output[b * pitch + index] = accumulator[b] * ids;
    }
  }
}

template<typename T>
void
radon::backward_cuda_3d(const T* x,
                        const float* angles,
                        T* y,
                        TextureCache& tex_cache,
                        const VolumeCfg& vol_cfg,
                        const ProjectionCfg& proj_cfg,
                        const ExecCfg& exec_cfg,
                        const int batch_size,
                        const int channels,
                        const int device,
                        const int angle_batch_size,
                        const int n_angles)
{
  constexpr bool is_float = std::is_same<T, float>::value;
  constexpr int precision = is_float ? PRECISION_FLOAT : PRECISION_HALF;

  // If the number of channels is a multiple of 4, then we can use the texture
  // channels to decrease the thread grid size. NOTE: CUDA also supports
  // textures with 2 channels.
  int texture_channels = 1;
  if (channels % 4 == 0) {
    texture_channels = 4;
  }

  Texture* tex = tex_cache.get({ device,
                                 proj_cfg.n_angles,
                                 proj_cfg.det_count_v,
                                 proj_cfg.det_count_u,
                                 true,
                                 texture_channels,
                                 precision });

  dim3 grid_dim =
    exec_cfg.get_grid_size(vol_cfg.width, vol_cfg.height, vol_cfg.depth);
  const dim3 block_dim = exec_cfg.get_block_dim();

  for (int i = 0; i < batch_size; i += channels) {
    T* local_y = &y[i * vol_cfg.depth * vol_cfg.height * vol_cfg.width];
    tex->put(
      &x[i * proj_cfg.n_angles * proj_cfg.det_count_v * proj_cfg.det_count_u]);

    // Invoke kernel
    if (channels == 1) {
      backward_kernel_3d<1><<<grid_dim, block_dim>>>(
        local_y, tex->texture, angles, vol_cfg, proj_cfg);
    } else {
      if (is_float) {
        backward_kernel_3d<4><<<grid_dim, block_dim>>>(
          local_y, tex->texture, angles, vol_cfg, proj_cfg);
      } else {
        backward_kernel_3d<4><<<grid_dim, block_dim>>>(
          (__half*)local_y, tex->texture, angles, vol_cfg, proj_cfg);
      }
    }
  }
}

template void
radon::backward_cuda_3d<float>(const float* x,
                               const float* angles,
                               float* y,
                               TextureCache& tex_cache,
                               const VolumeCfg& vol_cfg,
                               const ProjectionCfg& proj_cfg,
                               const ExecCfg& exec_cfg,
                               const int batch_size,
                               const int channels,
                               const int device,
                               const int angle_batch_size,
                               const int n_angles);

template void
radon::backward_cuda_3d<__half>(const __half* x,
                                const float* angles,
                                __half* y,
                                TextureCache& tex_cache,
                                const VolumeCfg& vol_cfg,
                                const ProjectionCfg& proj_cfg,
                                const ExecCfg& exec_cfg,
                                const int batch_size,
                                const int channels,
                                const int device,
                                const int angle_batch_size,
                                const int n_angles);
