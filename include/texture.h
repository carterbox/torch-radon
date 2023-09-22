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
 * TextureConfig is used as a unique identifier for the Cache object, so that
 * textures may be reused instead of reallocated.
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
   * @param channels The number of channels in the texture memory
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

  /**
   * @brief Get the texture type object
   *
   * Guesses whether the Texture is 1D, 2D, or 3D and with or without layers
   * based on the attributes of the texture.
   *
   * @return TextureType
   */
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
 * @brief Stores a CUDA Surface and Texture Object together. Provides methods
 * to loading data into these.
 *
 */
class Texture
{
private:
  /**
   * @brief Holds a pointer to the actual texture memory
   *
   */
  cudaArray* array = nullptr;
  TextureConfig cfg;

  template<typename T, typename D>
  void launchAccordingToSwitch(const D* data);

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
   * @brief Loads data from a normal device memory into CUDA Texture/Surface
   * Memory.
   *
   * The data is loaded into either single or four channel memory.
   *
   * @param data
   */
  void put(const float* data);
  void put(const __half* data);

  /**
   * @brief Return whether the this texture has the same TextureConfig as k.
   *
   * @param k The TextureConfig to compare with
   * @return true
   * @return false
   */
  bool matches(TextureConfig& k);

  ~Texture();
};

/**
 * @brief A Cache for Textures.
 *
 */
typedef Cache<TextureConfig, Texture> TextureCache;

#endif
