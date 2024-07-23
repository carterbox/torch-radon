#ifndef TORCH_RADON_UTILS_H
#define TORCH_RADON_UTILS_H

#include "defines.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>

template<typename... Args>
std::string
string_format(const std::string& format, Args... args)
{
  size_t size =
    snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(),
                     buf.get() + size - 1); // We don't want the '\0' inside
}

inline int
roundup_div(const int x, const int y)
{
  return x / y + (x % y != 0);
}

// inline unsigned int next_power_of_two(unsigned int v) {
//     v--;
//     v |= v >> 1;
//     v |= v >> 2;
//     v |= v >> 4;
//     v |= v >> 8;
//     v |= v >> 16;
//     v++;
//     return v;
// }

template<typename T>
void
check_cuda(T result, const char* func, const char* file, const int line)
{
  if (result) {
    fprintf(stderr,
            "CUDA error at %s (%s:%d) error code: %d, error string: %s\n",
            func,
            file,
            line,
            static_cast<unsigned int>(result),
            cudaGetErrorString(result));
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

#endif
