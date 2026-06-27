// Device-resident FrVector support for the plonk2-style resident prover.
//
// Two groups:
//  (1) CUDA-runtime helpers the resident orchestration needs but the icicle
//      delegation layer never exposed: events, pinned host memory, mem info,
//      cross-stream waits, async D2D.
//  (2) Element-wise Fr vector ops on AoS-Montgomery device buffers. The trivial
//      ones are fresh kernels; add/sub reuse the existing vecops.cu launchers.
//      batch_invert and scale_by_powers reuse the cub prefix-product machinery
//      already proven in ratio.cu / kzg.cu.
//
// Representation contract (unchanged): every field void* is a flat AoS buffer of
// Fr in Montgomery form; "zero" is all-zero bytes; a scalar arg d_c/d_g is a
// single device Fr.
#include "fr.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cub/device/device_scan.cuh>

// Existing element-wise launchers from vecops.cu (host functions, extern-callable).
extern "C" cudaError_t vec_add(Fr* d_r, const Fr* d_a, const Fr* d_b, uint32_t n, cudaStream_t stream);
extern "C" cudaError_t vec_sub(Fr* d_r, const Fr* d_a, const Fr* d_b, uint32_t n, cudaStream_t stream);
extern "C" cudaError_t vec_mul(Fr* d_r, const Fr* d_a, const Fr* d_b, uint32_t n, cudaStream_t stream);

namespace {
constexpr uint32_t kBlock = 256;
inline uint32_t grid(uint32_t n) { return (n + kBlock - 1) / kBlock; }

struct FrMulOp {
    __device__ Fr operator()(const Fr& a, const Fr& b) const {
        Fr r;
        fr::mul(r, a, b);
        return r;
    }
};

__global__ void scalar_mul_kernel(Fr* v, const Fr* c, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fr::mul(v[i], v[i], *c);
}
__global__ void addmul_kernel(Fr* v, const Fr* a, const Fr* b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fr t;
    fr::mul(t, a[i], b[i]);
    fr::add(v[i], v[i], t);
}
__global__ void add_scalar_mul_kernel(Fr* v, const Fr* a, const Fr* c, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fr t;
    fr::mul(t, a[i], *c);
    fr::add(v[i], v[i], t);
}
// seed for scale_by_powers: pow[0]=1, pow[i>=1]=g  (prefix-product -> g^i)
__global__ void powers_seed_kernel(Fr* pow, const Fr* g, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i == 0) fr::set_one(pow[0]);
    else pow[i] = *g;
}
__global__ void reverse_kernel(Fr* out, const Fr* in, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[n - 1 - i];
}
__global__ void inv_one_kernel(Fr* out, const Fr* in) { fr::inv(out[0], in[0]); }
// batch-invert combine: out[i] = pre[i-1] * revscan[n-2-i] * invAll
//   pre[k]     = prod_{j<=k} v[j]              (inclusive prefix product)
//   revscan[k] = prod_{j>=n-1-k} v[j]          (inclusive scan of reversed v)
//   invAll     = 1 / pre[n-1]
__global__ void batch_invert_combine_kernel(Fr* v, const Fr* pre, const Fr* revscan,
                                            const Fr* invAll, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Fr a, b;
    if (i == 0) fr::set_one(a);
    else a = pre[i - 1];
    if (i == n - 1) fr::set_one(b);
    else b = revscan[n - 2 - i];
    Fr t;
    fr::mul(t, a, b);
    fr::mul(v[i], t, *invAll);
}
} // namespace

extern "C" {

// ── (1) device / stream / event / pinned / mem ───────────────────────────────

int gpu_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess ? 0 : -1;
}
int gpu_device_sync(void) {
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}
void* gpu_event_create(void) {
    cudaEvent_t e = nullptr;
    if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) return nullptr;
    return (void*)e;
}
void gpu_event_record(void* event, void* stream) {
    cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
}
void gpu_stream_wait_event(void* stream, void* event) {
    cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
}
void gpu_event_destroy(void* event) {
    if (event) cudaEventDestroy((cudaEvent_t)event);
}
int gpu_alloc_pinned(void** ptr, size_t bytes) {
    return cudaHostAlloc(ptr, bytes, cudaHostAllocDefault) == cudaSuccess ? 0 : -1;
}
void gpu_free_pinned(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}
int gpu_memcpy_d2d_on_stream(void* dst, const void* src, size_t size, void* stream) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

// ── (2) FrVector element-wise ops ────────────────────────────────────────────

int gpu_vec_add(void* r, const void* a, const void* b, uint32_t n, void* stream) {
    return vec_add((Fr*)r, (const Fr*)a, (const Fr*)b, n, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}
int gpu_vec_sub(void* r, const void* a, const void* b, uint32_t n, void* stream) {
    return vec_sub((Fr*)r, (const Fr*)a, (const Fr*)b, n, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}
int gpu_vec_scalar_mul(void* v, const void* d_c, uint32_t n, void* stream) {
    scalar_mul_kernel<<<grid(n), kBlock, 0, (cudaStream_t)stream>>>((Fr*)v, (const Fr*)d_c, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
int gpu_vec_addmul(void* v, const void* a, const void* b, uint32_t n, void* stream) {
    addmul_kernel<<<grid(n), kBlock, 0, (cudaStream_t)stream>>>((Fr*)v, (const Fr*)a, (const Fr*)b, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
int gpu_vec_add_scalar_mul(void* v, const void* a, const void* d_c, uint32_t n, void* stream) {
    add_scalar_mul_kernel<<<grid(n), kBlock, 0, (cudaStream_t)stream>>>((Fr*)v, (const Fr*)a, (const Fr*)d_c, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
int gpu_vec_set_zero(void* v, uint32_t n, void* stream) {
    return cudaMemsetAsync(v, 0, (size_t)n * sizeof(Fr), (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

// v[i] *= g^i  (g^i built on-device via prefix product of [1,g,g,...])
int gpu_vec_scale_by_powers(void* v, const void* d_g, uint32_t n, void* stream) {
    if (n == 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    Fr* pow = nullptr;
    void* tmp = nullptr;
    size_t tb = 0;
    cudaError_t err = cudaSuccess;
    if ((err = cudaMalloc(&pow, (size_t)n * sizeof(Fr))) != cudaSuccess) goto done;
    powers_seed_kernel<<<grid(n), kBlock, 0, s>>>(pow, (const Fr*)d_g, n);
    if ((err = cub::DeviceScan::InclusiveScan(nullptr, tb, pow, pow, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&tmp, tb)) != cudaSuccess) goto done;
    if ((err = cub::DeviceScan::InclusiveScan(tmp, tb, pow, pow, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    if (vec_mul((Fr*)v, (const Fr*)v, pow, n, s) != cudaSuccess) { err = cudaGetLastError(); goto done; }
    err = cudaStreamSynchronize(s);
done:
    if (pow) cudaFree(pow);
    if (tmp) cudaFree(tmp);
    if (err != cudaSuccess) { fprintf(stderr, "[gpu_vec_scale_by_powers] %s\n", cudaGetErrorString(err)); return -1; }
    return 0;
}

// v[i] = 1/v[i]  (Montgomery batch inversion, parallel pre/suf-product form)
int gpu_vec_batch_invert(void* v, uint32_t n, void* stream) {
    if (n == 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    Fr *pre = nullptr, *rev = nullptr, *invAll = nullptr;
    void* tmp = nullptr;
    size_t tb = 0, tb2 = 0;
    cudaError_t err = cudaSuccess;
    size_t bytes = (size_t)n * sizeof(Fr);
    if ((err = cudaMalloc(&pre, bytes)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&rev, bytes)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&invAll, sizeof(Fr))) != cudaSuccess) goto done;
    // prefix product -> pre
    if ((err = cub::DeviceScan::InclusiveScan(nullptr, tb, (Fr*)v, pre, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    if ((err = cudaMalloc(&tmp, tb)) != cudaSuccess) goto done;
    if ((err = cub::DeviceScan::InclusiveScan(tmp, tb, (Fr*)v, pre, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    // reversed input -> rev, then prefix product in place -> revscan
    reverse_kernel<<<grid(n), kBlock, 0, s>>>(rev, (const Fr*)v, n);
    if ((err = cub::DeviceScan::InclusiveScan(nullptr, tb2, rev, rev, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    if (tb2 > tb) { cudaFree(tmp); tmp = nullptr; if ((err = cudaMalloc(&tmp, tb2)) != cudaSuccess) goto done; }
    if ((err = cub::DeviceScan::InclusiveScan(tmp, tb2, rev, rev, FrMulOp(), (int)n, s)) != cudaSuccess) goto done;
    // invAll = 1 / pre[n-1]
    inv_one_kernel<<<1, 1, 0, s>>>(invAll, pre + (n - 1));
    batch_invert_combine_kernel<<<grid(n), kBlock, 0, s>>>((Fr*)v, pre, rev, invAll, n);
    err = cudaStreamSynchronize(s);
done:
    if (pre) cudaFree(pre);
    if (rev) cudaFree(rev);
    if (invAll) cudaFree(invAll);
    if (tmp) cudaFree(tmp);
    if (err != cudaSuccess) { fprintf(stderr, "[gpu_vec_batch_invert] %s\n", cudaGetErrorString(err)); return -1; }
    return 0;
}

} // extern "C"
