#include "ratio.h"
#include "fr.h"
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

// CUDA port of gnark-hip/src/ratio.hip. The one non-trivial change vs HIP is the
// prefix scan: rocprim::inclusive_scan -> cub::DeviceScan::InclusiveScan. The
// FrMulOp functor is CUB-compatible as-is (__host__ __device__ binary operator).

__global__ void ratio_copy_terms_kernel(
    const Fr* __restrict__ l,
    const Fr* __restrict__ r,
    const Fr* __restrict__ o,
    const Fr* __restrict__ s1,
    const Fr* __restrict__ s2,
    const Fr* __restrict__ s3,
    const Fr* __restrict__ twiddles0,
    const Fr* __restrict__ challenges,
    uint32_t n,
    Fr* __restrict__ out_num,
    Fr* __restrict__ out_den)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (tid == 0) {
        fr::set_one(out_num[0]);
        fr::set_one(out_den[0]);
    }
    if (tid >= n - 1) return;

    Fr beta = challenges[0];
    Fr gamma = challenges[1];
    Fr u = challenges[2];
    Fr u2 = challenges[3];

    Fr id0 = twiddles0[tid];
    Fr id1, id2;
    fr::mul(id1, id0, u);
    fr::mul(id2, id0, u2);

    Fr num, den, tmp;
    fr::set_one(num);
    fr::set_one(den);

    fr::mul(tmp, beta, id0);
    fr::add(tmp, tmp, gamma);
    fr::add(tmp, tmp, l[tid]);
    fr::mul(num, num, tmp);

    fr::mul(tmp, beta, id1);
    fr::add(tmp, tmp, gamma);
    fr::add(tmp, tmp, r[tid]);
    fr::mul(num, num, tmp);

    fr::mul(tmp, beta, id2);
    fr::add(tmp, tmp, gamma);
    fr::add(tmp, tmp, o[tid]);
    fr::mul(num, num, tmp);

    fr::mul(tmp, beta, s1[tid]);
    fr::add(tmp, tmp, gamma);
    fr::add(tmp, tmp, l[tid]);
    fr::mul(den, den, tmp);

    fr::mul(tmp, beta, s2[tid]);
    fr::add(tmp, tmp, gamma);
    fr::add(tmp, tmp, r[tid]);
    fr::mul(den, den, tmp);

    fr::mul(tmp, beta, s3[tid]);
    fr::add(tmp, tmp, gamma);
    fr::add(tmp, tmp, o[tid]);
    fr::mul(den, den, tmp);

    out_num[tid + 1] = num;
    out_den[tid + 1] = den;
}

cudaError_t ratio_copy_terms(
    const Fr* d_l,
    const Fr* d_r,
    const Fr* d_o,
    const Fr* d_s1,
    const Fr* d_s2,
    const Fr* d_s3,
    const Fr* d_twiddles0,
    const Fr* d_challenges,
    uint32_t n,
    Fr* d_out_num,
    Fr* d_out_den,
    cudaStream_t stream)
{
    uint32_t block = 256;
    uint32_t grid = ((n - 1) + block - 1) / block;
    ratio_copy_terms_kernel<<<dim3(grid), dim3(block), 0, stream>>>(
        d_l, d_r, d_o, d_s1, d_s2, d_s3,
        d_twiddles0, d_challenges, n, d_out_num, d_out_den);
    return cudaGetLastError();
}

struct FrMulOp {
    // __device__-only: cub::DeviceScan invokes this on-device, and fr::mul is a
    // __device__ function, so a __host__ __device__ operator() would fail to
    // compile its host instantiation.
    __device__ Fr operator()(const Fr& a, const Fr& b) const {
        Fr r;
        fr::mul(r, a, b);
        return r;
    }
};

__global__ void ratio_apply_inverse_kernel(Fr* coeffs, const Fr* den, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    Fr invd;
    fr::inv(invd, den[tid]);
    fr::mul(coeffs[tid], coeffs[tid], invd);
}

cudaError_t ratio_prefix_scan(Fr* d_data, uint32_t n, cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        nullptr, temp_bytes, d_data, d_data, FrMulOp{}, (int)n, stream));
    void* temp_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&temp_storage, temp_bytes));
    cudaError_t err = cub::DeviceScan::InclusiveScan(
        temp_storage, temp_bytes, d_data, d_data, FrMulOp{}, (int)n, stream);
    cudaFree(temp_storage);
    return err;
}

cudaError_t ratio_apply_inverse(Fr* d_coeffs, const Fr* d_den, uint32_t n, cudaStream_t stream) {
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    ratio_apply_inverse_kernel<<<dim3(grid), dim3(block), 0, stream>>>(d_coeffs, d_den, n);
    return cudaGetLastError();
}
