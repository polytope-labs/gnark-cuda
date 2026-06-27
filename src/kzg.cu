// Device synthetic division for KZG opening: q(X) = (f(X) - f(a)) / (X - a).
//
// q[j] = a^{-(j+1)} * S[j+1], S[j] = sum_{k>=j} f[k]*a^k (inclusive suffix sum).
// The a^k / a^{-k} power tables are built on-device via a multiplicative prefix
// scan, so only the scalar a (and a^{-1}, 1) cross the PCIe bus, not 512MB of
// powers.
#include "fr.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cub/device/device_scan.cuh>

struct FrAddOp {
    __device__ Fr operator()(const Fr& a, const Fr& b) const {
        Fr r;
        fr::add(r, a, b);
        return r;
    }
};
struct FrMulOp {
    __device__ Fr operator()(const Fr& a, const Fr& b) const {
        Fr r;
        fr::mul(r, a, b);
        return r;
    }
};

__global__ void kzg_revmul_kernel(Fr* out, const Fr* in, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[n - 1 - i];
}
__global__ void kzg_mulvec_kernel(Fr* r, const Fr* a, const Fr* b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr::mul(r[i], a[i], b[i]);
}
// d_pow[0]=one, d_pow[k>=1]=a ; d_inv[0]=one, d_inv[k>=1]=ainv  (seed for prefix product)
__global__ void kzg_seed_kernel(Fr* d_pow, Fr* d_inv, const Fr* one, const Fr* a, const Fr* ainv, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0) { d_pow[0] = *one; d_inv[0] = *one; }
    else { d_pow[i] = *a; d_inv[i] = *ainv; }
}
// q[j] = S[j+1] * invpow[j+1] for j in [0, n-1)
__global__ void kzg_qshift_kernel(Fr* q, const Fr* S, const Fr* invpow, uint32_t n) {
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j + 1 < n) fr::mul(q[j], S[j + 1], invpow[j + 1]);
}

// d_a, d_ainv, d_one: single Fr each (device). d_q: n-1 coeffs.
extern "C" int gpu_kzg_divide(const void* d_f, const void* d_a, const void* d_ainv,
                              const void* d_one, uint32_t n, void* d_q, void* stream) {
    if (n < 2) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    Fr *pow = nullptr, *inv = nullptr, *h = nullptr, *hr = nullptr, *Sr = nullptr, *S = nullptr;
    void *tmp = nullptr, *tmp2 = nullptr;
    size_t tb = 0, tb2 = 0;
    cudaError_t err = cudaSuccess;
    uint32_t B = 256, G = (n + B - 1) / B;
    size_t bytes = (size_t)n * sizeof(Fr);

    if ((err = cudaMalloc(&pow, bytes)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&inv, bytes)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&h, bytes)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&hr, bytes)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&Sr, bytes)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&S, bytes)) != cudaSuccess) goto done;

    // build a^k / a^{-k} on device
    kzg_seed_kernel<<<G, B, 0, s>>>(pow, inv, (const Fr*)d_one, (const Fr*)d_a, (const Fr*)d_ainv, n);
    if ((err = cub::DeviceScan::InclusiveScan(nullptr, tb, pow, pow, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&tmp, tb)) != cudaSuccess) goto done;
    if ((err = cub::DeviceScan::InclusiveScan(tmp, tb, pow, pow, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    if ((err = cub::DeviceScan::InclusiveScan(tmp, tb, inv, inv, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;

    kzg_mulvec_kernel<<<G, B, 0, s>>>(h, (const Fr*)d_f, pow, n);   // h[k]=f[k]*a^k
    kzg_revmul_kernel<<<G, B, 0, s>>>(hr, h, n);
    if ((err = cub::DeviceScan::InclusiveScan(nullptr, tb2, hr, Sr, FrAddOp(), (int)n, s)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&tmp2, tb2)) != cudaSuccess) goto done;
    if ((err = cub::DeviceScan::InclusiveScan(tmp2, tb2, hr, Sr, FrAddOp(), (int)n, s)) != cudaSuccess) goto done;
    kzg_revmul_kernel<<<G, B, 0, s>>>(S, Sr, n);                    // S[j]=suffix sum
    kzg_qshift_kernel<<<G, B, 0, s>>>((Fr*)d_q, S, inv, n);         // q[j]=S[j+1]*a^{-(j+1)}
    err = cudaStreamSynchronize(s);

done:
    if (pow) cudaFree(pow);
    if (inv) cudaFree(inv);
    if (h) cudaFree(h);
    if (hr) cudaFree(hr);
    if (Sr) cudaFree(Sr);
    if (S) cudaFree(S);
    if (tmp) cudaFree(tmp);
    if (tmp2) cudaFree(tmp2);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_kzg_divide] failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
