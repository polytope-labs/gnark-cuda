#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// CUDA port of gnark-hip/include/types.h.
// Only the runtime include and the error-check macro differ from the HIP
// original; the DEVICE_* qualifier macros and the POD field/curve structs are
// identical under nvcc.

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            return err;                                                         \
        }                                                                       \
    } while (0)

#define DEVICE __device__
#define HOST __host__
#define GLOBAL __global__
#define DEVICE_HOST __device__ __host__
#define DEVICE_INLINE __device__ __forceinline__
#define DEVICE_NOINLINE __device__ __noinline__

// BLS12-381 scalar field element: 4×uint64 little-endian, Montgomery form.
// Byte-identical to gnark-crypto fr.Element ([4]uint64) and to icicle's
// bls12_381 scalar_t (storage<8> = [8]uint32). 32 bytes.
struct Fr { uint64_t limbs[4]; };
