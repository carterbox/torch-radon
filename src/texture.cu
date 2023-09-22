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
  // FIXME: Add a docstring which explains exactly what makes each texture type.
  if (is_layered && depth == 0 && height >= 0 && width > 0)
    return TEX_1D_LAYERED;
  if (is_layered && depth >= 0 && height > 0 && width > 0)
    return TEX_2D_LAYERED;
  if (!is_layered && depth > 0 && height > 0 && width > 0)
    return TEX_3D;
  throw std::invalid_argument(
    "This TextureConfig does not match any TextureType!");
}

TextureConfig
create_1Dlayered_texture_config(int device,
                                int size,
                                int layers,
                                int channels,
                                int precision)
{
  return TextureConfig(device, 0, layers, size, true, channels, precision);
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
template<int texture_type, typename T, typename D>
__global__ void
write_to_surface(const D* data,
                 cudaSurfaceObject_t surface,
                 const int width,
                 const int height,
                 const int depth)
{
  constexpr int channels = sizeof(T) / sizeof(D);
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

        D tmp[channels];
        for (int i = 0; i < channels; i++)
          tmp[i] = data[i * pitch + offset];

        switch (texture_type) {
          case TEX_1D_LAYERED:
            surf1DLayeredwrite<T>(*(T*)tmp, surface, x * sizeof(T), y);
            break;
          case TEX_2D_LAYERED:
            surf2DLayeredwrite<T>(*(T*)tmp, surface, x * sizeof(T), y, z);
            break;
          case TEX_3D:
            surf3Dwrite<T>(*(T*)tmp, surface, x * sizeof(T), y, z);
            break;
        }
      }
    }
  }
}

cudaChannelFormatDesc
get_channel_desc(int channels, int precision)
{
  int bitsPerChannel[4] = { 0, 0, 0, 0 };
  for (int c = 0; c < channels; ++c) {
    bitsPerChannel[c] = (precision == PRECISION_FLOAT) ? 32 : 16;
  }
  return cudaCreateChannelDesc(bitsPerChannel[0],
                               bitsPerChannel[1],
                               bitsPerChannel[2],
                               bitsPerChannel[3],
                               cudaChannelFormatKindFloat);
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

  cudaMemcpy3DParms myparms = { 0 };

  // Pytorch (channel first) has a different memory order than CUDA textures
  // (channel last), so we have to use a special copy method when using
  // multiple channels
  switch (cfg.channels) {
    case 1:
      myparms.srcPos = make_cudaPos(0, 0, 0);
      myparms.dstPos = make_cudaPos(0, 0, 0);
      myparms.srcPtr = make_cudaPitchedPtr((void*)data,
                                           cfg.width * sizeof(float),
                                           max(cfg.width, 1),
                                           max(cfg.height, 1));
      myparms.dstArray = this->array;
      myparms.extent = make_cudaExtent(
        max(cfg.width, 1), max(cfg.height, 1), max(cfg.depth, 1));
      myparms.kind = cudaMemcpyDeviceToDevice;
      checkCudaErrors(cudaMemcpy3D(&myparms));
      break;
    case 2:
      launchAccordingToSwitch<float2>(data);
      break;
    case 4:
      // else if using multiple channels use custom kernel to copy the data
      launchAccordingToSwitch<float4>(data);
      break;
  }
}

void
Texture::put(const __half* data)
{
  if (this->cfg.precision == PRECISION_FLOAT) {
    LOG_WARNING("Putting single precision data into a half precision texture");
  }

  checkCudaErrors(cudaSetDevice(this->cfg.device));

  cudaMemcpy3DParms myparms = { 0 };

  switch (cfg.channels) {
    case 1:
      myparms.srcPos = make_cudaPos(0, 0, 0);
      myparms.dstPos = make_cudaPos(0, 0, 0);
      myparms.srcPtr = make_cudaPitchedPtr((void*)data,
                                           cfg.width * sizeof(__half),
                                           max(cfg.width, 1),
                                           max(cfg.height, 1));
      myparms.dstArray = this->array;
      myparms.extent = make_cudaExtent(
        max(cfg.width, 1), max(cfg.height, 1), max(cfg.depth, 1));
      myparms.kind = cudaMemcpyDeviceToDevice;
      checkCudaErrors(cudaMemcpy3D(&myparms));
      break;
    case 2:
      launchAccordingToSwitch<float1>(data);
      break;
    case 4:
      launchAccordingToSwitch<float2>(data);
      break;
  }
}

/**
 * @brief A wrapper around the launch of writeToSurface which allows us to
   instantiate more templates without repeated code
 *
 * @tparam T The type used during memory copy
 * @tparam D The type of the data
 * @param data
 */
template<typename T, typename D>
void
Texture::launchAccordingToSwitch(const D* data)
{
  const TextureType texture_type = cfg.get_texture_type();

  dim3 grid_dim(
    roundup_div(cfg.width, 16), roundup_div(cfg.height, 16), cfg.depth);

  switch (texture_type) {

    case TEX_1D_LAYERED:
      write_to_surface<TEX_1D_LAYERED, T>
        <<<grid_dim, dim3(16, 16)>>>(data,
                                     this->surface,
                                     max(cfg.width, 1),
                                     max(cfg.height, 1),
                                     max(cfg.depth, 1));
      break;

    case TEX_2D_LAYERED:
      write_to_surface<TEX_2D_LAYERED, T>
        <<<grid_dim, dim3(16, 16)>>>(data,
                                     this->surface,
                                     max(cfg.width, 1),
                                     max(cfg.height, 1),
                                     max(cfg.depth, 1));
      break;

    case TEX_3D:
    default:
      write_to_surface<TEX_3D, T>
        <<<grid_dim, dim3(16, 16)>>>(data,
                                     this->surface,
                                     max(cfg.width, 1),
                                     max(cfg.height, 1),
                                     max(cfg.depth, 1));

      break;
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
