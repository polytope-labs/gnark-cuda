#include "fr.h"
#include "types.h"
#include <cuda_runtime.h>

// CUDA port of gnark-hip/src/vecops.hip. Element-wise Fr vector ops. Kernel
// bodies identical to HIP; only launch syntax and error types change. These
// internal launchers (vec_mul, vec_denominators, vec_from_mont, vec_to_mont)
// are called by the gpu_* wrappers in api.cu.

__global__ void vec_add_kernel(Fr* r, const Fr* a, const Fr* b, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr::add(r[tid], a[tid], b[tid]);
}

__global__ void vec_sub_kernel(Fr* r, const Fr* a, const Fr* b, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr::sub(r[tid], a[tid], b[tid]);
}

__global__ void vec_mul_kernel(Fr* r, const Fr* a, const Fr* b, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr::mul(r[tid], a[tid], b[tid]);
}

__global__ void vec_neg_kernel(Fr* r, const Fr* a, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr::neg(r[tid], a[tid]);
}

__global__ void vec_scalar_mul_kernel(Fr* r, const Fr* a, const Fr* scalar, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr::mul(r[tid], a[tid], *scalar);
}

__global__ void vec_denominators_kernel(Fr* r, const Fr* twiddles, const Fr* coset, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    Fr t;
    fr::mul(t, twiddles[tid], *coset);
    Fr one;
    fr::set_one(one);
    fr::sub(t, t, one);
    fr::inv(r[tid], t);
}

// Montgomery form conversion kernels
__global__ void vec_to_mont_kernel(Fr* r, const Fr* a, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr::to_mont(r[tid], a[tid]);
}

__global__ void vec_from_mont_kernel(Fr* r, const Fr* a, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    fr::from_mont(r[tid], a[tid]);
}

// --- Internal host launchers (called by api.cu gpu_* wrappers) ---

extern "C" {

cudaError_t vec_add(Fr* d_r, const Fr* d_a, const Fr* d_b, uint32_t n, cudaStream_t stream) {
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    vec_add_kernel<<<dim3(grid), dim3(block), 0, stream>>>(d_r, d_a, d_b, n);
    return cudaGetLastError();
}

cudaError_t vec_sub(Fr* d_r, const Fr* d_a, const Fr* d_b, uint32_t n, cudaStream_t stream) {
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    vec_sub_kernel<<<dim3(grid), dim3(block), 0, stream>>>(d_r, d_a, d_b, n);
    return cudaGetLastError();
}

cudaError_t vec_mul(Fr* d_r, const Fr* d_a, const Fr* d_b, uint32_t n, cudaStream_t stream) {
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    vec_mul_kernel<<<dim3(grid), dim3(block), 0, stream>>>(d_r, d_a, d_b, n);
    return cudaGetLastError();
}

cudaError_t vec_from_mont(Fr* d_r, const Fr* d_a, uint32_t n, cudaStream_t stream) {
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    vec_from_mont_kernel<<<dim3(grid), dim3(block), 0, stream>>>(d_r, d_a, n);
    return cudaGetLastError();
}

cudaError_t vec_to_mont(Fr* d_r, const Fr* d_a, uint32_t n, cudaStream_t stream) {
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    vec_to_mont_kernel<<<dim3(grid), dim3(block), 0, stream>>>(d_r, d_a, n);
    return cudaGetLastError();
}

cudaError_t vec_denominators(Fr* d_r, const Fr* d_twiddles, const Fr* d_coset, uint32_t n, cudaStream_t stream) {
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    vec_denominators_kernel<<<dim3(grid), dim3(block), 0, stream>>>(d_r, d_twiddles, d_coset, n);
    return cudaGetLastError();
}

} // extern "C"
