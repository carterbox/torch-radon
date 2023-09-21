#include <cuda_fp16.h>
#include <iostream>
#include <string>

#include "defines.h"
#include "texture.h"
#include "utils.h"

TextureConfig::TextureConfig(int device,
                             int depth,
                             int height,
                             int width,
                             bool layered,
                             int channels,
                             int precision)
  : device(device)
  , depth(depth)
  , height(height)
  , width(width)
  , is_layered(layered)
  , channels(channels)
  , precision(precision)
{
}

bool
TextureConfig::operator==(const TextureConfig& o) const
{
  return this->device == o.device && this->width == o.width &&
         this->height == o.height && this->is_layered == o.is_layered &&
         this->depth == o.depth && this->channels == o.channels &&
         this->precision == o.precision;
}

TextureType
TextureConfig::get_texture_type() const
{
  if (this->is_layered && this->height == 0)
    return TEX_1D_LAYERED;
  if (this->is_layered)
    return TEX_2D_LAYERED;
  return TEX_3D;
}

TextureConfig
create_1Dlayered_texture_config(int device,
                                int size,
                                int layers,
                                int channels,
                                int precision)
{
  return TextureConfig(device, layers, 0, size, true, channels, precision);
}

std::ostream&
operator<<(std::ostream& os, TextureConfig const& m)
{
  std::string precision = m.precision == PRECISION_FLOAT ? "float" : "half";

  return os << "(device: " << m.device << ", depth: " << m.depth
            << ", height: " << m.height << ", width: " << m.width
            << ", channels: " << m.channels << ", precision: " << precision
            << ", " << (m.is_layered ? "layered" : "not layered") << ")";
}

/// Assume data ordered        (height, channel, width) for 1D layered
/// Assume data ordered (depth, channel, height, width) for 2D layered
/// Assume data ordered (channel, depth, height, width) for 3D
template<int texture_type, typename T>
__global__ void
write_to_surface(const float* data,
                 cudaSurfaceObject_t surface,
                 const int width,
                 const int height,
                 const int depth)
{
  constexpr int channels = sizeof(T) / 4;
  static_assert(std::is_same<T, float1>::value ||
                  std::is_same<T, float2>::value ||
                  std::is_same<T, float4>::value,
                "Only float1, float2, and float4 are supported.");

  int pitch;
  switch (texture_type) {
    case TEX_1D_LAYERED:
      pitch = width;
      break;
    case TEX_2D_LAYERED:
      pitch = height * width;
      break;
    case TEX_3D:
      pitch = depth * height * width;
      break;
  }

  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;
       x += blockDim.x * gridDim.x) {
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;
         y += blockDim.y * gridDim.y) {
      for (int z = blockIdx.z * blockDim.z + threadIdx.z; z < depth;
           z += blockDim.z * gridDim.z) {

        int offset;
        switch (texture_type) {
          case TEX_1D_LAYERED:
            offset = ((y)*channels + 0) * width + x;
            break;
          case TEX_2D_LAYERED:
            offset = (((z)*channels + 0) * height + y) * width + x;
            break;
          case TEX_3D:
            offset = (((0) * depth + z) * height + y) * width + x;
            break;
        }

        T tmp;
        if constexpr (channels >= 1) {
          tmp.x = data[0 * pitch + offset];
        }
        if constexpr (channels >= 2) {
          tmp.y = data[1 * pitch + offset];
        }
        if constexpr (channels == 4) {
          tmp.z = data[2 * pitch + offset];
          tmp.w = data[3 * pitch + offset];
        }

        switch (texture_type) {
          case TEX_1D_LAYERED:
            surf1DLayeredwrite<T>(tmp, surface, x * sizeof(T), y);
            break;
          case TEX_2D_LAYERED:
            surf2DLayeredwrite<T>(tmp, surface, x * sizeof(T), y, z);
            break;
          case TEX_3D:
            surf3Dwrite<T>(tmp, surface, x * sizeof(T), y, z);
            break;
        }
      }
    }
  }
}

template<int texture_type>
__global__ void
write_half_to_surface(const __half* data,
                      cudaSurfaceObject_t surface,
                      const int width,
                      const int height,
                      const int depth)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < width && y < height && z < depth) {
    const int pitch = width * height * depth;
    const int offset = (z * height + y) * width + x;

    __half tmp[4];
    for (int i = 0; i < 4; i++)
      tmp[i] = data[i * pitch + offset];

    switch (texture_type) {
      case TEX_1D_LAYERED:
        surf1DLayeredwrite<float2>(
          *(float2*)tmp, surface, x * sizeof(float2), y);
        break;
      case TEX_2D_LAYERED:
        surf2DLayeredwrite<float2>(
          *(float2*)tmp, surface, x * sizeof(float2), y, z);
        break;
      case TEX_3D:
        surf3Dwrite<float2>(*(float2*)tmp, surface, x * sizeof(float2), y, z);
        break;
    }
  }
}

cudaChannelFormatDesc
get_channel_desc(int channels, int precision)
{
  if (precision == PRECISION_FLOAT) {
    if (channels == 1) {
      return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    }
    if (channels == 4) {
      return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    }
  }
  if (precision == PRECISION_HALF && channels == 4) {
    return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
  }

  LOG_WARNING("Unsupported number of channels and precision (channels:"
              << channels << ", precision: " << precision << ")");
  return cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
}

Texture::Texture(TextureConfig c)
  : cfg(c)
{
  checkCudaErrors(cudaSetDevice(this->cfg.device));

  LOG_INFO("Allocating Texture " << this->cfg);

  // Allocate CUDA array
  cudaChannelFormatDesc channelDesc =
    get_channel_desc(cfg.channels, cfg.precision);
  auto allocation_type = cfg.is_layered ? cudaArrayLayered : cudaArrayDefault;

  const cudaExtent extent = make_cudaExtent(cfg.width, cfg.height, cfg.depth);
  checkCudaErrors(
    cudaMalloc3DArray(&array, &channelDesc, extent, allocation_type));

  // Create resource descriptor
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = array;

  // Specify texture object parameters
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  checkCudaErrors(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));

  // Create surface object
  checkCudaErrors(cudaCreateSurfaceObject(&surface, &resDesc));
}

void
Texture::put(const float* data)
{
  if (this->cfg.precision == PRECISION_HALF) {
    LOG_WARNING("Putting half precision data into a float texture");
  }

  checkCudaErrors(cudaSetDevice(this->cfg.device));

  // Pytorch (channel first) has a different memory order than CUDA textures
  // (channel last), so we have to use a special copy method when using
  // multiple channels
  if (cfg.channels == 1) {
    // if using a single channel use cudaMemcpy to copy data into array
    cudaMemcpy3DParms myparms = { 0 };
    myparms.srcPos = make_cudaPos(0, 0, 0);
    myparms.dstPos = make_cudaPos(0, 0, 0);
    myparms.srcPtr = make_cudaPitchedPtr(
      (void*)data, cfg.width * sizeof(float), cfg.width, max(cfg.height, 1));
    myparms.dstArray = this->array;

    myparms.extent = make_cudaExtent(cfg.width, max(cfg.height, 1), cfg.depth);

    myparms.kind = cudaMemcpyDeviceToDevice;
    checkCudaErrors(cudaMemcpy3D(&myparms));
  } else if (cfg.channels == 4) {
    // else if using multiple channels use custom kernel to copy the data
    int texture_type = cfg.get_texture_type();
    if (texture_type == TEX_1D_LAYERED) {
      dim3 grid_dim(roundup_div(cfg.width, 16), roundup_div(cfg.depth, 16));
      LOG_DEBUG("[TORCH RADON] Copying 1D Texture " << this->cfg);
      write_to_surface<TEX_1D_LAYERED, float4>
        <<<grid_dim, dim3(16, 16)>>>(data,
                                     this->surface,
                                     max(cfg.width, 1),
                                     max(cfg.height, 1),
                                     max(cfg.depth, 1));
    } else {
      dim3 grid_dim(
        roundup_div(cfg.width, 16), roundup_div(cfg.height, 16), cfg.depth);
      if (texture_type == TEX_2D_LAYERED) {
        LOG_DEBUG("[TORCH RADON] Copying 2D Texture " << this->cfg);
        write_to_surface<TEX_2D_LAYERED, float4>
          <<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(data,
                                             this->surface,
                                             max(cfg.width, 1),
                                             max(cfg.height, 1),
                                             max(cfg.depth, 1));
      } else {
        LOG_DEBUG("[TORCH RADON] Copying 3D Texture " << this->cfg);
        write_to_surface<TEX_3D, float4>
          <<<grid_dim, dim3(16, 16)>>>(data,
                                       this->surface,
                                       max(cfg.width, 1),
                                       max(cfg.height, 1),
                                       max(cfg.depth, 1));
      }
    }
#ifdef DEBUG
    checkCudaErrors(cudaDeviceSynchronize());
#endif
  } else {
    throw std::invalid_argument("There can only be 1 or 4 texture channels!");
  }
}

void
Texture::put(const __half* data)
{
  if (this->cfg.precision == PRECISION_FLOAT) {
    LOG_WARNING("Putting single precision data into a half precision texture");
  }

  checkCudaErrors(cudaSetDevice(this->cfg.device));

  int texture_type = cfg.get_texture_type();
  if (texture_type == TEX_1D_LAYERED) {
    dim3 grid_dim(roundup_div(cfg.width, 16), roundup_div(cfg.depth, 16));
    write_half_to_surface<TEX_1D_LAYERED><<<grid_dim, dim3(16, 16)>>>(
      (__half*)data, this->surface, cfg.width, cfg.depth, 1);
  } else {
    dim3 grid_dim(
      roundup_div(cfg.width, 16), roundup_div(cfg.height, 16), cfg.depth);
    if (texture_type == TEX_2D_LAYERED) {
      write_half_to_surface<TEX_2D_LAYERED><<<grid_dim, dim3(16, 16)>>>(
        (__half*)data, this->surface, cfg.width, cfg.height, cfg.depth);
    } else {
      write_half_to_surface<TEX_3D><<<grid_dim, dim3(16, 16)>>>(
        (__half*)data, this->surface, cfg.width, cfg.height, cfg.depth);
    }
  }
}

bool
Texture::matches(TextureConfig& c)
{
  return c == this->cfg;
}

Texture::~Texture()
{
  LOG_DEBUG("[TORCH RADON] Freeing Texture " << this->cfg);

  if (this->array != nullptr) {
    checkCudaErrors(cudaSetDevice(this->cfg.device));
    checkCudaErrors(cudaDestroyTextureObject(this->texture));
    checkCudaErrors(cudaDestroySurfaceObject(this->surface));
    checkCudaErrors(cudaFreeArray(this->array));
    this->array = nullptr;
  }
}
