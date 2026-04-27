#pragma once
#include <hip/hip_runtime.h>

#include "hcu_fp8.h"

template <typename T> struct PrintTraits {
  static __device__ void print_var(const char *msg, T val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "dtype=unknown value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, (const void *)&val);
  }

  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, T val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
           "index=%d, dtype=unknown value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, (const void *)&val);
  }
};

#define DEFINE_PRINT_TRAIT(TYPE, NAME, FORMAT, CAST_TYPE)                      \
  template <> struct PrintTraits<TYPE> {                                       \
    static __device__ void print_var(const char *msg, TYPE val) {              \
      printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "        \
             "dtype=" NAME " value=" FORMAT "\n",                              \
             msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,             \
             threadIdx.y, threadIdx.z, (CAST_TYPE)val);                        \
    }                                                                          \
    static __device__ void print_buffer(const char *msg, const char *buf_name, \
                                        int index, TYPE val) {                 \
      printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "        \
             "buffer=%s, index=%d, dtype=" NAME " value=" FORMAT "\n",         \
             msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,             \
             threadIdx.y, threadIdx.z, buf_name, index, (CAST_TYPE)val);       \
    }                                                                          \
  }

DEFINE_PRINT_TRAIT(char, "char", "%d", int);
DEFINE_PRINT_TRAIT(signed char, "signed char", "%d", int);
// Do not specialize unsigned char: ck_tile::fp8_t / bf8_t (fp8_e4_t / fp8_e5_t) may
// alias to uint8_t (unsigned char). One trait for fp8_e4_t covers that storage type;
// do not add a separate fp8_e5_t trait when it would duplicate the same underlying type.
DEFINE_PRINT_TRAIT(short, "short", "%d", int);
// Do not specialize unsigned short: on CK/HCU, bfloat16_t (bf16_t) aliases to
// unsigned short; a second trait would duplicate PrintTraits<unsigned short>.
DEFINE_PRINT_TRAIT(int, "int", "%d", int);
DEFINE_PRINT_TRAIT(unsigned int, "uint", "%u", unsigned int);
DEFINE_PRINT_TRAIT(long, "long", "%ld", long);
DEFINE_PRINT_TRAIT(unsigned long, "ulong", "%lu", unsigned long);
DEFINE_PRINT_TRAIT(long long, "long long", "%lld", long long);

DEFINE_PRINT_TRAIT(float, "float", "%f", float);
DEFINE_PRINT_TRAIT(double, "double", "%lf", double);
DEFINE_PRINT_TRAIT(half_t, "half_t", "%f", float);
DEFINE_PRINT_TRAIT(bfloat16_t, "bfloat16_t", "%f", float);

// fp8_e5_t often shares the same underlying type as fp8_e4_t on CK (both uint8);
// defining both would re-specialize PrintTraits<unsigned char>.
DEFINE_PRINT_TRAIT(fp8_e4_t, "fp8_e4_t", "%f", float);

//
template <> struct PrintTraits<bool> {
  static __device__ void print_var(const char *msg, bool val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=bool "
           "value=%s\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, val ? "true" : "false");
  }
  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, bool val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
           "index=%d, dtype=bool value=%s\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, val ? "true" : "false");
  }
};

template <typename T> struct PrintTraits<T *> {
  static __device__ void print_var(const char *msg, T *val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): "
           "dtype=pointer value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, (void *)val);
  }
  static __device__ void print_buffer(const char *msg, const char *buf_name,
                                      int index, T *val) {
    printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): buffer=%s, "
           "index=%d, dtype=pointer value=%p\n",
           msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
           threadIdx.z, buf_name, index, (void *)val);
  }
};

template <typename T> __device__ void debug_print_var(const char *msg, T var) {
  PrintTraits<T>::print_var(msg, var);
}

// Template declaration for device-side debug printing (buffer only)
template <typename T>
__device__ void debug_print_buffer_value(const char *msg, const char *buf_name,
                                         int index, T var) {
  PrintTraits<T>::print_buffer(msg, buf_name, index, var);
}

__device__ void debug_print_msg(const char *msg) {
  printf("msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d)\n", msg,
         blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
         threadIdx.z);
}
