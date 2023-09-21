#ifndef TORCH_RADON_TEXTURE_CACHE_H
#define TORCH_RADON_TEXTURE_CACHE_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cache.h"
#include "defines.h"
#include "utils.h"

enum TextureType
{
  TEX_1D_LAYERED = 0,
  TEX_2D_LAYERED = 1,
  TEX_3D = 2
};

/**
 * @brief Stores information about a CUDA texture
 *
 */
class TextureConfig
{
public:
  int device;

  int depth;
  int height;
  int width;

  bool is_layered;

  int channels;
  int precision;

  /**
   * @brief Construct a new Texture Config object
   *
   * @param device The CUDA device to use
   * @param depth The depth of the texture memory in elements
   * @param height The height of the texture memory in elements
   * @param width The width of the texture memory in elements
   * @param is_layered Whether to use a layered CUDA texture
   * @param channels The number of channels in the data
   * @param precision Whether the data is float (1) or half (0)
   */
  TextureConfig(int device,
                int depth,
                int height,
                int width,
                bool is_layered,
                int channels,
                int precision);

  bool operator==(const TextureConfig& o) const;

  TextureType get_texture_type() const;
};

TextureConfig
create_1Dlayered_texture_config(int device,
                                int size,
                                int layers,
                                int channels,
                                int precision);

std::ostream&
operator<<(std::ostream& os, TextureConfig const& m);

/**
 * @brief Stores a CUDA Surface and Texture Objects together. Provides methods
 * to load data into these.
 *
 */
class Texture
{
  /**
   * @brief Holds a pointer to the actual texture memory
   *
   */
  cudaArray* array = nullptr;
  TextureConfig cfg;

public:

  /**
  * @brief A surface object that shares memory with the texture object
  *
  */
  cudaSurfaceObject_t surface = 0;

  /**
   * @brief A texture object that shares memory with the surface object
   *
   */
  cudaTextureObject_t texture = 0;

  Texture(TextureConfig c);

  /**
   * @brief Loads data from a normal device memory into CUDA Texture/Surface Memory.
   *
   * The data is loaded into either single or four channel memory.
   *
   * @param data
   */
  void put(const float* data);
  void put(const __half* data);

  bool matches(TextureConfig& k);

  ~Texture();
};

typedef Cache<TextureConfig, Texture> TextureCache;

#endif
