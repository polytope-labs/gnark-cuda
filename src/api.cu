// =============================================================================
// gnark-cuda api.cu — the icicle-delegation layer + api-resident bespoke kernels.
//
// CUDA port of gnark-hip/src/api.hip. This file exports the `gpu_*` C-ABI
// (see include/gnark_cuda.h) that the gnark-crypto seam binds to. Generic
// primitives (MSM/NTT/vec) delegate to a self-built open-icicle CUDA backend;
// the small bespoke PLONK kernels are ported in-place from the HIP original.
//
// CORRECTNESS STATUS (see STATUS.md):
//   * Device mgmt / memory / streams / bitreverse / MSM / bespoke kernels /
//     wrappers  -> faithful ports, expected correct (gated by M2/M3 KATs).
//   * NTT family + fused FFT (gpu_ntt*, gpu_fft_scale_fft*, gpu_plonk_restore_
//     batch) -> BEST-EFFORT mapping of gnark-hip's explicit DIT/DIF/bit-reverse/
//     1-over-n semantics onto icicle's Ordering + always-scaled inverse. This is
//     the #1 fragile area (plan §4/§6.7/§8.5). Every branch tagged [FRAGILE]
//     MUST pass the M4 per-mode round-trip tests on real sm_120 hardware before
//     being trusted. Do not assume correctness from the fact that it compiles.
// =============================================================================

#include "gnark_cuda.h"
#include "plonk.h"
#include "ratio.h"
#include "fr.h"
#include <cuda_runtime.h>
#include <cstring>

#include "icicle/runtime.h"
#include "icicle/device.h"
#include "icicle/errors.h"
#include "icicle/ntt.h"
#include "icicle/msm.h"
#include "icicle/vec_ops.h"
#include "icicle/api/bls12_381.h"

using namespace icicle;
using bls12_381::scalar_t;
using bls12_381::affine_t;
using bls12_381::projective_t;

// Internal launchers defined in vecops.cu. (vec_from_mont/vec_to_mont are part
// of the public C-ABI declared in gnark_cuda.h and defined in vecops.cu, so we
// do not redeclare them here.)
extern "C" cudaError_t vec_mul(Fr* d_r, const Fr* d_a, const Fr* d_b, uint32_t n, cudaStream_t stream);
extern "C" cudaError_t vec_denominators(Fr* d_r, const Fr* d_twiddles, const Fr* d_coset, uint32_t n, cudaStream_t stream);

static inline int ok(eIcicleError e) { return e == eIcicleError::SUCCESS ? 0 : -1; }

// Multiply each element by the field element n (= 2^log_n). Used to undo
// icicle's mandatory 1/n on inverse NTT for the no-scale variants. n < 2^32.
__global__ void ntt_scale_by_n_kernel(Fr* data, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    Fr n_can = {{(uint64_t)n, 0, 0, 0}};
    Fr n_mont; fr::to_mont(n_mont, n_can);
    Fr tmp; fr::mul(tmp, data[tid], n_mont);
    data[tid] = tmp;
}

// Batch element-wise multiply: polys[p][i] *= scale[i] for all p != skip_idx.
__global__ void batch_vec_mul_kernel(Fr** polys, const Fr* scale,
                                     uint32_t n, uint32_t count, int skip_idx) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)count * n;
    if (tid >= total) return;
    uint32_t poly = tid / n;
    if ((int)poly == skip_idx) return;
    uint32_t idx = tid % n;
    Fr* data = polys[poly];
    Fr tmp;
    fr::mul(tmp, data[idx], scale[idx]);
    data[idx] = tmp;
}

extern "C" {

// --- Device management -------------------------------------------------------

int gpu_init(int device_id) {
    // Load the self-built open-icicle CUDA backend from ICICLE_BACKEND_INSTALL_DIR.
    icicle_load_backend_from_env_or_default();
    Device dev{"CUDA", device_id};
    return ok(icicle_set_device(dev));
}

int gpu_device_count() {
    int count = 0;
    icicle_get_device_count(count);
    return count;
}

// icicle's active device is per-thread. Go cgo calls run on arbitrary OS
// threads, so every goroutine doing GPU work must (re)select CUDA or icicle
// silently dispatches to the CPU backend and dereferences device pointers as
// host memory. Cheap; call at the start of each GPU entry point.
int gpu_set_device(int device_id) {
    Device dev{"CUDA", device_id};
    return ok(icicle_set_device(dev));
}

void gpu_sync() {
    icicle_device_synchronize();
}

// --- Memory management -------------------------------------------------------

void* gpu_malloc(size_t size) {
    void* ptr = nullptr;
    return icicle_malloc(&ptr, size) == eIcicleError::SUCCESS ? ptr : nullptr;
}

void gpu_free(void* ptr) {
    if (ptr) icicle_free(ptr);
}

int gpu_memcpy_h2d(void* dst, const void* src, size_t size) {
    return ok(icicle_copy_to_device(dst, src, size));
}

int gpu_memcpy_d2h(void* dst, const void* src, size_t size) {
    return ok(icicle_copy_to_host(dst, src, size));
}

int gpu_memcpy_d2d(void* dst, const void* src, size_t size) {
    return ok(icicle_copy(dst, src, size));
}

int gpu_memcpy_h2d_on_stream(void* dst, const void* src, size_t size, void* stream) {
    return ok(icicle_copy_to_device_async(dst, src, size, (icicleStreamHandle)stream));
}

int gpu_memcpy_d2h_on_stream(void* dst, const void* src, size_t size, void* stream) {
    return ok(icicle_copy_to_host_async(dst, src, size, (icicleStreamHandle)stream));
}

// --- Streams -----------------------------------------------------------------

void* gpu_stream_create() {
    icicleStreamHandle s = nullptr;
    return icicle_create_stream(&s) == eIcicleError::SUCCESS ? (void*)s : nullptr;
}

void gpu_stream_sync(void* stream) {
    icicle_stream_synchronize((icicleStreamHandle)stream);
}

void gpu_stream_destroy(void* stream) {
    if (stream) icicle_destroy_stream((icicleStreamHandle)stream);
}

// --- NTT (delegated to icicle) ----------------------------------------------
//
// gnark-hip exposed NTTs in terms of explicit DIT/DIF + bit-reverse + a
// "noscale" flag that suppresses the 1/n on inverse. icicle instead takes an
// `Ordering` (kNN/kNR/kRN) and ALWAYS applies 1/n on inverse. The mapping below
// reproduces the gnark-hip surface; the [FRAGILE] tags mark every place whose
// ordering/scaling equivalence must be confirmed by the M4 round-trip tests.
//
// Reference (gnark-hip api.hip):
//   gpu_ntt           = ntt_execute                  (natural in/out, inv scales 1/n)
//   gpu_ntt_dif       = ntt_execute then bitreverse  (natural in, bit-rev out)
//   gpu_ntt_dit       = ntt_execute_nobr             (bit-rev in, natural out)
//   gpu_ntt_dit_nosc  = ntt_execute_nobr_noscale     (bit-rev in, natural out, no 1/n)
//   gpu_ntt_dif_nosc  = BR -> nobr_noscale -> BR
//   gpu_ntt_coset     = ntt_coset_execute            (coset_gen, natural in/out)
//   gpu_ntt_coset_dit = ntt_coset_execute_nobr

static NTTConfig<scalar_t> make_ntt_cfg(Ordering ord, icicleStreamHandle s) {
    NTTConfig<scalar_t> cfg = default_ntt_config<scalar_t>();
    cfg.ordering = ord;
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    cfg.is_async = (s != nullptr);
    cfg.stream = s;
    return cfg;
}

// In-place icicle NTT (input aliased as output). dir: kForward / kInverse.
static int icicle_ntt_inplace(Fr* d, uint32_t log_n, NTTDir dir, Ordering ord,
                              icicleStreamHandle s) {
    NTTConfig<scalar_t> cfg = make_ntt_cfg(ord, s);
    scalar_t* p = reinterpret_cast<scalar_t*>(d);
    return ok(bls12_381_ntt(p, (int)(1u << log_n), dir, &cfg, p));
}

static uint32_t g_ntt_domain_log = 0;  // size the domain is currently built for

int gpu_ntt_init(uint32_t log_n) {
    if (g_ntt_domain_log >= log_n) return 0;     // existing (larger) domain covers it
    if (g_ntt_domain_log != 0) bls12_381_ntt_release_domain();
    // Build the domain from GNARK's 2-adic root of unity so icicle's NTT basis
    // matches gnark's FFT. omega_{2^32} (canonical, from gnark-crypto fr) raised
    // to 2^(32-log_n) gives the primitive 2^log_n-th root. icicle's internal
    // scalar form is canonical (it strips Montgomery on input), so we pass the
    // canonical root; gnark's Montgomery *data* then passes R through linearly.
    // Build for the LARGEST size requested; smaller NTTs use derived sub-roots.
    static const uint64_t OMEGA32_CANON[4] = {
        0x3829971f439f0d2bULL, 0xb63683508c2280b9ULL,
        0xd09b681922c813b4ULL, 0x16a2a19edfe81f20ULL};
    scalar_t root;
    memcpy(&root, OMEGA32_CANON, sizeof(root));
    for (uint32_t i = 0; i < 32u - log_n; i++) root = root * root;
    NTTInitDomainConfig cfg = default_ntt_init_domain_config();
    if (bls12_381_ntt_init_domain(&root, &cfg) != eIcicleError::SUCCESS) return -1;
    g_ntt_domain_log = log_n;
    return 0;
}

void gpu_ntt_cleanup() {
    g_ntt_domain_log = 0;
    bls12_381_ntt_release_domain();
}

int gpu_ntt(void* d, uint32_t log_n, int direction, void* stream) {
    // [FRAGILE] natural-order both ends.
    return icicle_ntt_inplace((Fr*)d, log_n,
                              direction == 0 ? NTTDir::kForward : NTTDir::kInverse,
                              Ordering::kNN, (icicleStreamHandle)stream);
}

int gpu_ntt_dif(void* d, uint32_t log_n, int direction, void* stream) {
    // [FRAGILE] natural in, bit-reversed out -> Ordering::kNR.
    return icicle_ntt_inplace((Fr*)d, log_n,
                              direction == 0 ? NTTDir::kForward : NTTDir::kInverse,
                              Ordering::kNR, (icicleStreamHandle)stream);
}

int gpu_ntt_dit(void* d, uint32_t log_n, int direction, void* stream) {
    // [FRAGILE] bit-reversed in, natural out -> Ordering::kRN.
    return icicle_ntt_inplace((Fr*)d, log_n,
                              direction == 0 ? NTTDir::kForward : NTTDir::kInverse,
                              Ordering::kRN, (icicleStreamHandle)stream);
}

// The "_noscale" inverse variants suppress the 1/n that icicle ALWAYS applies on
// inverse. We undo it by multiplying the result by n (montgomery data x canonical
// n via the kernel's montgomery mul). Forward direction needs no compensation.
int gpu_ntt_dit_noscale(void* d, uint32_t log_n, int direction, void* stream) {
    NTTDir dir = direction == 0 ? NTTDir::kForward : NTTDir::kInverse;
    if (icicle_ntt_inplace((Fr*)d, log_n, dir, Ordering::kRN, (icicleStreamHandle)stream) != 0) return -1;
    if (direction != 0) {
        uint32_t n = 1u << log_n;
        ntt_scale_by_n_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>((Fr*)d, n);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    return 0;
}

int gpu_ntt_dif_noscale(void* d, uint32_t log_n, int direction, void* stream) {
    NTTDir dir = direction == 0 ? NTTDir::kForward : NTTDir::kInverse;
    if (icicle_ntt_inplace((Fr*)d, log_n, dir, Ordering::kNR, (icicleStreamHandle)stream) != 0) return -1;
    if (direction != 0) {
        uint32_t n = 1u << log_n;
        ntt_scale_by_n_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>((Fr*)d, n);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    return 0;
}

// --- Coset NTT: coset-shift multiply + NTT (matches gnark domain coset FFT) ---
// gnark's coset FFT shifts by g = FrMultiplicativeGen = 7. Forward coset:
// data[i]*=g^i then FFT. Inverse coset: IFFT (incl. 1/n) then data[i]*=g^{-i}.
// g^{±i} is computed per-thread (square-and-multiply on the index) to avoid a
// 2^25-element (1 GB) coset-powers table. Mirrors gnark-hip's ntt_coset_execute.
__device__ Fr g_coset_gen[1];      // 7 in Montgomery form
__device__ Fr g_coset_gen_inv[1];  // 7^{-1} in Montgomery form

__global__ void coset_gen_setup_kernel() {
    Fr seven; seven.limbs[0] = 7; seven.limbs[1] = 0; seven.limbs[2] = 0; seven.limbs[3] = 0;
    fr::to_mont(g_coset_gen[0], seven);
    fr::inv(g_coset_gen_inv[0], g_coset_gen[0]);
}

static bool g_coset_gen_ready = false;
static int ensure_coset_gen(cudaStream_t s) {
    if (g_coset_gen_ready) return 0;
    coset_gen_setup_kernel<<<1, 1, 0, s>>>();
    if (cudaStreamSynchronize(s) != cudaSuccess) return -1;
    g_coset_gen_ready = true;
    return 0;
}

// Two-level coset-powers table: l1[j]=g^j (j<B), l2[k]=g^{kB}, so
// g^i = l2[i>>logB] * l1[i & (B-1)] — one lookup-multiply per element. Replaces
// a per-thread square-and-multiply (which warp-diverges + dominated divideByZH
// at 33M elements). The two small tables are built once per (inverse,logB).
static Fr* d_coset_l1 = nullptr;
static Fr* d_coset_l2 = nullptr;
static const uint32_t COSET_LMAX = 1u << 14;  // per-level table capacity

__global__ void coset_table_build_kernel(Fr* l1, Fr* l2, uint32_t B, uint32_t nL2, int inverse) {
    Fr g = inverse ? g_coset_gen_inv[0] : g_coset_gen[0];
    fr::set_one(l1[0]);
    for (uint32_t j = 1; j < B; j++) fr::mul(l1[j], l1[j - 1], g);
    Fr gB; fr::mul(gB, l1[B - 1], g);   // g^B
    fr::set_one(l2[0]);
    for (uint32_t k = 1; k < nL2; k++) fr::mul(l2[k], l2[k - 1], gB);
}

__global__ void coset_mul2_kernel(Fr* data, const Fr* l1, const Fr* l2, uint32_t n, uint32_t logB) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t lo = tid & ((1u << logB) - 1);
    uint32_t hi = tid >> logB;
    Fr p; fr::mul(p, l2[hi], l1[lo]);
    fr::mul(data[tid], data[tid], p);
}

// Multiply data[i] by g^{±i} (inverse selects g^{-1}) via the two-level table.
static int coset_scale_fast(Fr* d, uint32_t log_n, int inverse, cudaStream_t s) {
    if (d_coset_l1 == nullptr) {
        if (cudaMalloc(&d_coset_l1, COSET_LMAX * sizeof(Fr)) != cudaSuccess) return -1;
        if (cudaMalloc(&d_coset_l2, COSET_LMAX * sizeof(Fr)) != cudaSuccess) return -1;
    }
    uint32_t logB = log_n / 2;
    if (logB > 13) logB = 13;
    uint32_t B = 1u << logB;
    uint32_t nL2 = (1u << log_n) >> logB;
    if (nL2 == 0) nL2 = 1;
    if (B > COSET_LMAX || nL2 > COSET_LMAX) return -1;
    coset_table_build_kernel<<<1, 1, 0, s>>>(d_coset_l1, d_coset_l2, B, nL2, inverse);
    uint32_t n = 1u << log_n;
    uint32_t block = 256, grid = (n + block - 1) / block;
    coset_mul2_kernel<<<grid, block, 0, s>>>(d, d_coset_l1, d_coset_l2, n, logB);
    return 0;
}

// Coset NTT in the kNN (natural-in/natural-out) ordering.
int gpu_ntt_coset(void* d, uint32_t log_n, int direction, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    if (ensure_coset_gen(s) != 0) return -1;
    if (direction == 1) { // inverse: IFFT (+1/n) then * g^{-i}
        if (gpu_ntt(d, log_n, 1, stream) != 0) return -1;
        if (coset_scale_fast((Fr*)d, log_n, 1, s) != 0) return -1;
    } else {              // forward: * g^i then FFT
        if (coset_scale_fast((Fr*)d, log_n, 0, s) != 0) return -1;
        if (gpu_ntt(d, log_n, 0, stream) != 0) return -1;
    }
    return 0;
}

// Coset NTT in DIT ordering (bit-reversed input -> natural output for inverse).
// This is the path gnark's a.ToCanonical(bigDomain) takes for a LagrangeCoset /
// BitReverse polynomial (divideByZH): DIT inverse FFT then * g^{-i} in natural order.
int gpu_ntt_coset_dit(void* d, uint32_t log_n, int direction, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    if (ensure_coset_gen(s) != 0) return -1;
    if (direction == 1) { // inverse: DIT IFFT (bit-rev in -> natural out, +1/n) then * g^{-i}
        if (gpu_ntt_dit(d, log_n, 1, stream) != 0) return -1;
        if (coset_scale_fast((Fr*)d, log_n, 1, s) != 0) return -1;
    } else {              // forward: * g^i then DIT FFT
        if (coset_scale_fast((Fr*)d, log_n, 0, s) != 0) return -1;
        if (gpu_ntt_dit(d, log_n, 0, stream) != 0) return -1;
    }
    return 0;
}

int gpu_bitreverse(void* d, uint32_t log_n, void* stream) {
    VecOpsConfig cfg = default_vec_ops_config();
    cfg.is_a_on_device = true;
    cfg.is_result_on_device = true;
    cfg.is_async = (stream != nullptr);
    cfg.stream = (icicleStreamHandle)stream;
    scalar_t* p = reinterpret_cast<scalar_t*>(d);
    return ok(bls12_381_bit_reverse(p, (uint64_t)(1u << log_n), &cfg, p));
}

// --- Fused FFT: IFFT -> multiply -> FFT (device-resident) --------------------
// [FRAGILE] Mirrors gnark-hip's gpu_fft_scale_fft using the DIT/DIF mapping
// above. Inherits all the NTT-mapping caveats; M4 round-trips the whole fused
// op against a CPU IFFT->scale->FFT reference for every (ifft_dir,fft_dir).

int gpu_fft_scale_fft(void* d_data, void* d_scale, uint32_t log_n,
                      int ifft_dir, int fft_dir, void* stream) {
    icicleStreamHandle s = (icicleStreamHandle)stream;
    uint32_t n = 1u << log_n;
    // Phase 1: inverse FFT (DIT=kRN bit-rev-in / DIF=kNR natural-in).
    if ((ifft_dir == 1 ? gpu_ntt_dif(d_data, log_n, 1, stream)
                       : gpu_ntt_dit(d_data, log_n, 1, stream)) != 0) return -1;
    // Phase 2: element-wise scale on device.
    if (vec_mul((Fr*)d_data, (Fr*)d_data, (Fr*)d_scale, n, (cudaStream_t)s) != cudaSuccess) return -1;
    // Phase 3: forward FFT.
    if ((fft_dir == 1 ? gpu_ntt_dif(d_data, log_n, 0, stream)
                      : gpu_ntt_dit(d_data, log_n, 0, stream)) != 0) return -1;
    return 0;
}

// d_polys is a DEVICE array of `count` Fr* pointers. Loop the per-poly fused FFT
// (correctness-first; batched NTT optimization deferred). Copy the pointer array
// to host to iterate.
int gpu_fft_scale_fft_batch(void* d_polys, uint32_t count, int skip_idx,
                            void* d_scale, uint32_t log_n,
                            int ifft_dir, int fft_dir, void* stream) {
    if (count == 0) return 0;
    void** hostPtrs = (void**)malloc(count * sizeof(void*));
    if (!hostPtrs) return -1;
    if (icicle_copy_to_host(hostPtrs, d_polys, count * sizeof(void*)) != eIcicleError::SUCCESS) { free(hostPtrs); return -1; }
    int rc = 0;
    for (uint32_t i = 0; i < count; i++) {
        if ((int)i == skip_idx) continue;
        if (gpu_fft_scale_fft(hostPtrs[i], d_scale, log_n, ifft_dir, fft_dir, stream) != 0) { rc = -1; break; }
    }
    free(hostPtrs);
    return rc;
}

// --- MSM (delegated to icicle) ----------------------------------------------
// [VERIFY @ M3] Result coordinate convention: icicle projective_t vs gnark
// G1Jac. If icicle's projective is homogeneous (x=X/Z) rather than Jacobian
// (x=X/Z^2) the host result must be converted. Asserted in the M3 MSM KAT.

// gnark BLS12-381 fp Montgomery 1 (= R mod p), from gnark-crypto fp Element.SetOne().
static const uint64_t FP_ONE_MONT[6] = {
    0x760900000002fffdULL, 0xebf4000bc40c0002ULL, 0x5f48985753c758baULL,
    0x77ce585370525745ULL, 0x5c071a97a256ec6dULL, 0x15f65ec3fa80e493ULL};

// gpu_msm expects CANONICAL (non-Montgomery) device points + scalars. The caller
// (gnark-crypto gpu.go) converts Montgomery->canonical and caches the canonical
// points, because (a) icicle's in-MSM Montgomery path is buggy when the MSM is
// chunked under memory pressure (see ICICLE_MSM_MONTGOMERY_CHUNKING_BUG.md), and
// (b) doing the conversion outside lets the reused SRS points be cached.
int gpu_msm(const void* d_points, const void* d_scalars, uint32_t n,
            void* h_result, uint32_t window_size, void* stream) {
    MSMConfig cfg = default_msm_config();
    cfg.stream = (icicleStreamHandle)stream;
    cfg.are_scalars_on_device = true;
    cfg.are_points_on_device = true;
    cfg.are_results_on_device = false;            // write result to host
    cfg.are_scalars_montgomery_form = false;      // inputs are already canonical
    cfg.are_points_montgomery_form = false;
    if (window_size > 0) cfg.c = (int)window_size;
    cfg.is_async = false;

    projective_t proj;                            // icicle homogeneous projective (canonical)
    const bool dbg = getenv("GPU_MSM_DEBUG") != nullptr;
    if (dbg) fprintf(stderr, "[gpu_msm] n=%u calling bls12_381_msm\n", n);
    if (bls12_381_msm(reinterpret_cast<const scalar_t*>(d_scalars),
                      reinterpret_cast<const affine_t*>(d_points),
                      (int)n, &cfg, &proj) != eIcicleError::SUCCESS)
        return -1;
    if (dbg) fprintf(stderr, "[gpu_msm] msm ok, marshaling\n");

    // Marshal -> gnark G1Jac {X,Y,Z fp Montgomery}, 3*48 bytes. icicle projective
    // is HOMOGENEOUS (affine = X/Z), gnark wants Jacobian (affine = X/Z^2), so we
    // route via affine. icicle output is canonical -> convert into Montgomery.
    uint64_t* out = reinterpret_cast<uint64_t*>(h_result);     // X[6] Y[6] Z[6]
    const uint64_t* pz = reinterpret_cast<const uint64_t*>(&proj) + 12; // proj.z
    bool is_inf = true;
    for (int i = 0; i < 6; i++) if (pz[i] != 0) { is_inf = false; break; }
    if (is_inf) {                                 // gnark infinity: X=1, Y=1, Z=0
        for (int i = 0; i < 6; i++) { out[i] = FP_ONE_MONT[i]; out[6+i] = FP_ONE_MONT[i]; out[12+i] = 0; }
        return 0;
    }
    if (dbg) fprintf(stderr, "[gpu_msm] not inf, to_affine\n");
    affine_t aff;  bls12_381_to_affine(&proj, &aff);           // canonical affine
    if (dbg) fprintf(stderr, "[gpu_msm] to_affine ok, convert_montgomery\n");
    affine_t affM; VecOpsConfig vc = default_vec_ops_config();
    bls12_381_affine_convert_montgomery(&aff, 1, /*is_into=*/true, &vc, &affM);
    if (dbg) fprintf(stderr, "[gpu_msm] marshal done\n");
    const uint64_t* axy = reinterpret_cast<const uint64_t*>(&affM);  // x[6] y[6]
    for (int i = 0; i < 6; i++) { out[i] = axy[i]; out[6+i] = axy[6+i]; out[12+i] = FP_ONE_MONT[i]; }
    return 0;                                     // {x_m, y_m, 1_m} = Jacobian of affine
}

// Size of the icicle homogeneous projective result, so the Go caller can size the
// host landing buffer for gpu_msm_async / gpu_msm_marshal.
size_t gpu_msm_proj_bytes(void) { return sizeof(projective_t); }

// gpu_msm_async runs the MSM ASYNCHRONOUSLY on `stream`, writing the raw icicle
// projective result into the host buffer h_proj (>= gpu_msm_proj_bytes). It does
// NOT sync and does NOT marshal, so several MSMs issued on distinct streams
// overlap on the GPU. The caller syncs each stream, then calls gpu_msm_marshal.
// h_proj and the scalar buffer must stay alive until the sync.
int gpu_msm_async(const void* d_points, const void* d_scalars, uint32_t n,
                  void* h_proj, uint32_t window_size, void* stream) {
    MSMConfig cfg = default_msm_config();
    cfg.stream = (icicleStreamHandle)stream;
    cfg.are_scalars_on_device = true;
    cfg.are_points_on_device = true;
    cfg.are_results_on_device = false;            // result D2H'd to h_proj (async)
    cfg.are_scalars_montgomery_form = false;
    cfg.are_points_montgomery_form = false;
    if (window_size > 0) cfg.c = (int)window_size;
    cfg.is_async = true;
    if (bls12_381_msm(reinterpret_cast<const scalar_t*>(d_scalars),
                      reinterpret_cast<const affine_t*>(d_points),
                      (int)n, &cfg, reinterpret_cast<projective_t*>(h_proj)) != eIcicleError::SUCCESS)
        return -1;
    return 0;
}

// gpu_msm_marshal converts a settled icicle projective (from gpu_msm_async, after
// the stream is synced) into a gnark G1Jac {X,Y,Z fp Montgomery}. Pure host work.
int gpu_msm_marshal(const void* h_proj, void* h_result) {
    const projective_t* proj = reinterpret_cast<const projective_t*>(h_proj);
    uint64_t* out = reinterpret_cast<uint64_t*>(h_result);
    const uint64_t* pz = reinterpret_cast<const uint64_t*>(proj) + 12;
    bool is_inf = true;
    for (int i = 0; i < 6; i++) if (pz[i] != 0) { is_inf = false; break; }
    if (is_inf) {
        for (int i = 0; i < 6; i++) { out[i] = FP_ONE_MONT[i]; out[6+i] = FP_ONE_MONT[i]; out[12+i] = 0; }
        return 0;
    }
    affine_t aff; bls12_381_to_affine(const_cast<projective_t*>(proj), &aff);
    affine_t affM; VecOpsConfig vc = default_vec_ops_config();
    bls12_381_affine_convert_montgomery(&aff, 1, /*is_into=*/true, &vc, &affM);
    const uint64_t* axy = reinterpret_cast<const uint64_t*>(&affM);
    for (int i = 0; i < 6; i++) { out[i] = axy[i]; out[6+i] = axy[6+i]; out[12+i] = FP_ONE_MONT[i]; }
    return 0;
}

// --- Bespoke PLONK kernels (api-resident; ported from HIP) -------------------

__global__ void plonk_scatter_result_kernel(const Fr* src, Fr* dst, uint32_t n, uint32_t rho,
                                            uint32_t iter, uint32_t shift_bits) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t out_idx = ((uint64_t)rho * tid) + iter;
    out_idx = __brevll(out_idx) >> shift_bits;
    dst[out_idx] = src[tid];
}

__global__ void plonk_divide_zh_kernel(Fr* data, const Fr* inv_rho, uint32_t n, uint32_t rho,
                                       uint32_t shift_bits) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t i_rev = __brevll((uint64_t)tid) >> shift_bits;
    Fr tmp;
    fr::mul(tmp, data[tid], inv_rho[i_rev % rho]);
    data[tid] = tmp;
}

__global__ void plonk_fold_quotient_kernel(const Fr* h1, const Fr* h2, const Fr* h3,
                                           Fr* out, const Fr* zeta_n_plus_2, uint32_t n) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const Fr z = *zeta_n_plus_2;
    Fr t;
    fr::mul(t, h3[tid], z);
    fr::add(t, t, h2[tid]);
    fr::mul(t, t, z);
    fr::add(out[tid], t, h1[tid]);
}

int gpu_plonk_evaluate(void** d_polys, const void* d_twiddles0,
                       const void* d_bp, const void* d_challenges,
                       const void* d_precomp_denoms,
                       uint32_t n, uint32_t npolys, uint32_t nbBsbGates,
                       void* d_result, void* stream) {
    return plonk_evaluate_constraints(
        (Fr**)d_polys, (const Fr*)d_twiddles0, (const Fr*)d_bp,
        (const Fr*)d_challenges, (const Fr*)d_precomp_denoms,
        n, npolys, nbBsbGates, (Fr*)d_result, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

int gpu_plonk_scatter_result(const void* d_src, void* d_dst, uint32_t n, uint32_t rho,
                             uint32_t iter, uint32_t shift_bits, void* stream) {
    uint32_t block = 256, grid = (n + block - 1) / block;
    plonk_scatter_result_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const Fr*)d_src, (Fr*)d_dst, n, rho, iter, shift_bits);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int gpu_plonk_divide_zh(void* d_data, const void* d_inv_rho, uint32_t n, uint32_t rho,
                        uint32_t shift_bits, void* stream) {
    uint32_t block = 256, grid = (n + block - 1) / block;
    plonk_divide_zh_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (Fr*)d_data, (const Fr*)d_inv_rho, n, rho, shift_bits);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int gpu_plonk_fold_quotient(const void* d_h1, const void* d_h2, const void* d_h3,
                            void* d_out, const void* d_zeta_n_plus_2, uint32_t n, void* stream) {
    uint32_t block = 256, grid = (n + block - 1) / block;
    plonk_fold_quotient_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const Fr*)d_h1, (const Fr*)d_h2, (const Fr*)d_h3, (Fr*)d_out,
        (const Fr*)d_zeta_n_plus_2, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int gpu_ratio_copy_terms(const void* d_l, const void* d_r, const void* d_o,
                         const void* d_s1, const void* d_s2, const void* d_s3,
                         const void* d_twiddles0, const void* d_challenges,
                         uint32_t n, void* d_out_num, void* d_out_den, void* stream) {
    return ratio_copy_terms(
        (const Fr*)d_l, (const Fr*)d_r, (const Fr*)d_o,
        (const Fr*)d_s1, (const Fr*)d_s2, (const Fr*)d_s3,
        (const Fr*)d_twiddles0, (const Fr*)d_challenges,
        n, (Fr*)d_out_num, (Fr*)d_out_den, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

int gpu_ratio_prefix_scan(void* d_data, uint32_t n, void* stream) {
    return ratio_prefix_scan((Fr*)d_data, n, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

int gpu_ratio_apply_inverse(void* d_coeffs, const void* d_den, uint32_t n, void* stream) {
    return ratio_apply_inverse((Fr*)d_coeffs, (const Fr*)d_den, n, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

int gpu_vec_mul(void* d_r, const void* d_a, const void* d_b, uint32_t n, void* stream) {
    return vec_mul((Fr*)d_r, (const Fr*)d_a, (const Fr*)d_b, n, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

int gpu_vec_denominators(void* d_r, const void* d_twiddles, const void* d_coset, uint32_t n, void* stream) {
    return vec_denominators((Fr*)d_r, (const Fr*)d_twiddles, (const Fr*)d_coset, n, (cudaStream_t)stream) == cudaSuccess ? 0 : -1;
}

// Convert affine G1 points Montgomery->canonical on device (d_dst may alias d_src).
// Used by the seam to convert + cache reused MSM bases once.
int gpu_affine_from_mont(void* d_dst, const void* d_src, uint32_t n, void* stream) {
    VecOpsConfig vc = default_vec_ops_config();
    vc.is_a_on_device = true; vc.is_result_on_device = true;
    vc.stream = (icicleStreamHandle)stream; vc.is_async = (stream != nullptr);
    return ok(bls12_381_affine_convert_montgomery(
        reinterpret_cast<const affine_t*>(d_src), (size_t)n, /*is_into=*/false, &vc,
        reinterpret_cast<affine_t*>(d_dst)));
}

// --- Batch restore (PLONK post-rho) ------------------------------------------
// Lagrange Regular -> Canonical Regular (inverse FFT with 1/n) then multiply by
// the coset-inverse scale_powers, per poly. Loop over the device pointer array.
int gpu_plonk_restore_batch(void** d_polys, uint32_t count, int skip_idx,
                            const void* d_scale_powers, uint32_t log_n, void* stream) {
    if (count == 0) return 0;
    uint32_t n = 1u << log_n;
    void** hostPtrs = (void**)malloc(count * sizeof(void*));
    if (!hostPtrs) return -1;
    if (icicle_copy_to_host(hostPtrs, d_polys, count * sizeof(void*)) != eIcicleError::SUCCESS) { free(hostPtrs); return -1; }
    int rc = 0;
    for (uint32_t i = 0; i < count; i++) {
        if ((int)i == skip_idx) continue;
        // inverse FFT (natural in/out + 1/n) == ntt_execute inverse
        if (gpu_ntt(hostPtrs[i], log_n, /*inverse=*/1, stream) != 0) { rc = -1; break; }
        if (vec_mul((Fr*)hostPtrs[i], (Fr*)hostPtrs[i], (const Fr*)d_scale_powers, n, (cudaStream_t)stream) != cudaSuccess) { rc = -1; break; }
    }
    free(hostPtrs);
    return rc;
}

// --- Polynomial evaluation at a point (chunked Horner) -----------------------

int gpu_poly_eval(const void* d_coeffs, uint32_t n, const void* d_point,
                  void* h_result, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    const uint32_t NUM_CHUNKS = EVAL_CHUNKS;
    uint32_t chunk_size = (n + NUM_CHUNKS - 1) / NUM_CHUNKS;
    uint32_t actual_chunks = (n + chunk_size - 1) / chunk_size;

    Fr* d_partials = nullptr; Fr* d_point_power = nullptr; Fr* d_result = nullptr;
    cudaError_t err;
    if ((err = cudaMalloc(&d_partials, NUM_CHUNKS * sizeof(Fr))) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&d_point_power, sizeof(Fr))) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&d_result, sizeof(Fr))) != cudaSuccess) goto cleanup;

    poly_eval_phase1_kernel<<<1, NUM_CHUNKS, 0, s>>>(
        (const Fr*)d_coeffs, (const Fr*)d_point, d_partials, d_point_power, n, chunk_size);
    if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;

    poly_eval_phase2_kernel<<<1, 1, 0, s>>>(d_partials, d_point_power, d_result, actual_chunks);
    if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;

    if ((err = cudaStreamSynchronize(s)) != cudaSuccess) goto cleanup;
    err = cudaMemcpy(h_result, d_result, sizeof(Fr), cudaMemcpyDeviceToHost);

cleanup:
    if (d_partials) cudaFree(d_partials);
    if (d_point_power) cudaFree(d_point_power);
    if (d_result) cudaFree(d_result);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_poly_eval] failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Async, no-alloc, no-sync variant of gpu_poly_eval: the caller supplies scratch
// (d_partials of EVAL_CHUNKS Fr, d_point_power of 1 Fr) and a device d_result, and
// syncs the stream once after issuing a whole batch. d_partials/d_point_power may be
// reused across the batch since the kernels serialize on the stream.
int gpu_poly_eval_async(const void* d_coeffs, uint32_t n, const void* d_point,
                        void* d_partials, void* d_point_power, void* d_result, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    const uint32_t NUM_CHUNKS = EVAL_CHUNKS;
    uint32_t chunk_size = (n + NUM_CHUNKS - 1) / NUM_CHUNKS;
    uint32_t actual_chunks = (n + chunk_size - 1) / chunk_size;
    poly_eval_phase1_kernel<<<1, NUM_CHUNKS, 0, s>>>(
        (const Fr*)d_coeffs, (const Fr*)d_point, (Fr*)d_partials, (Fr*)d_point_power, n, chunk_size);
    poly_eval_phase2_kernel<<<1, 1, 0, s>>>(
        (Fr*)d_partials, (Fr*)d_point_power, (Fr*)d_result, actual_chunks);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// --- Linearized polynomial ---------------------------------------------------

int gpu_plonk_linearized_poly(
    const void* d_blindedZ, const void* d_s3, const void* d_ql, const void* d_qr,
    const void* d_qm, const void* d_qo, const void* d_qk, const void* d_hFolded,
    const void* d_scalars,
    uint32_t n, uint32_t n_blindedZ, uint32_t n_hFolded,
    void* d_result, void* stream)
{
    uint32_t block = 256, grid = (n_blindedZ + block - 1) / block;
    plonk_linearized_poly_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const Fr*)d_blindedZ, (const Fr*)d_s3, (const Fr*)d_ql, (const Fr*)d_qr,
        (const Fr*)d_qm, (const Fr*)d_qo, (const Fr*)d_qk, (const Fr*)d_hFolded,
        (const Fr*)d_scalars, n, n_blindedZ, n_hFolded, (Fr*)d_result);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpu_plonk_linearized_poly] launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

} // extern "C"
