#pragma once

#include <stdint.h>
#include <stddef.h>

// =============================================================================
// gnark-cuda public C-ABI.
//
// This is the AUTHORITATIVE contract that libgnark_cuda.so must export. It is
// byte-identical to the `gpu_*` C-ABI that gnark-hip exported (libgnark_hip),
// which the gnark-crypto seam binds to in the cgo preambles of
//   gnark-crypto/ecc/bls12-381/gpu/{gpu,plonk,ratio}.go
// Re-exporting the same symbols means every Go file above the C boundary
// compiles unchanged — only the cgo LDFLAGS/CFLAGS and the cuda build tag (hip in the AMD libgnark_hip)
// change. DO NOT rename, reorder args, or change types of any symbol here.
//
// Representation contract: every `void*` that points at field data is a buffer
// of `Fr` (4×uint64 little-endian Montgomery, 32 bytes), identical to
// gnark-crypto fr.Element. Device pointers are flat allocations safe for
// `uintptr + offset` arithmetic. `stream` is an opaque handle (icicleStreamHandle
// under the hood); NULL means the default stream.
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// --- Device management -------------------------------------------------------
int  gpu_init(int device_id);
int  gpu_set_device(int device_id);   // (re)select CUDA on the calling thread
int  gpu_device_count(void);
void gpu_sync(void);

// --- Memory management -------------------------------------------------------
void* gpu_malloc(size_t size);
void  gpu_free(void* ptr);
int   gpu_memcpy_h2d(void* dst, const void* src, size_t size);
int   gpu_memcpy_d2h(void* dst, const void* src, size_t size);
int   gpu_memcpy_d2d(void* dst, const void* src, size_t size);
int   gpu_memcpy_h2d_on_stream(void* dst, const void* src, size_t size, void* stream);
int   gpu_memcpy_d2h_on_stream(void* dst, const void* src, size_t size, void* stream);

// --- Streams -----------------------------------------------------------------
void* gpu_stream_create(void);
void  gpu_stream_sync(void* stream);
void  gpu_stream_destroy(void* stream);

// --- NTT (delegated to icicle bls12_381_ntt) ---------------------------------
// `direction`: 0 = forward, 1 = inverse. `log_n` = log2(domain size).
int  gpu_ntt_init(uint32_t log_n);
void gpu_ntt_cleanup(void);
int  gpu_ntt(void* d_data, uint32_t log_n, int direction, void* stream);
int  gpu_ntt_dit(void* d_data, uint32_t log_n, int direction, void* stream);
int  gpu_ntt_dif(void* d_data, uint32_t log_n, int direction, void* stream);
int  gpu_ntt_dit_noscale(void* d_data, uint32_t log_n, int direction, void* stream);
int  gpu_ntt_dif_noscale(void* d_data, uint32_t log_n, int direction, void* stream);
int  gpu_ntt_coset(void* d_data, uint32_t log_n, int direction, void* stream);
int  gpu_ntt_coset_dit(void* d_data, uint32_t log_n, int direction, void* stream);
int  gpu_bitreverse(void* d_data, uint32_t log_n, void* stream);

// --- MSM (delegated to icicle bls12_381_msm) ---------------------------------
// d_points, d_scalars: device buffers (Montgomery). h_result: host G1 (projective).
int  gpu_msm(const void* d_points, const void* d_scalars, uint32_t n,
             void* h_result, uint32_t window_size, void* stream);
size_t gpu_msm_proj_bytes(void);
int  gpu_msm_async(const void* d_points, const void* d_scalars, uint32_t n, void* h_proj, uint32_t window_size, void* stream);
int  gpu_msm_marshal(const void* h_proj, void* h_result);

// --- Fused FFT: IFFT -> element-wise multiply -> FFT, all on device ----------
int  gpu_fft_scale_fft(void* d_data, void* d_scale, uint32_t log_n,
                       int ifft_dir, int fft_dir, void* stream);
int  gpu_fft_scale_fft_batch(void* d_polys, uint32_t count, int skip_idx,
                             void* d_scale, uint32_t log_n,
                             int ifft_dir, int fft_dir, void* stream);

// --- Bespoke PLONK kernels (ported from gnark-hip) ---------------------------
int  gpu_plonk_evaluate(void** d_polys, const void* d_twiddles0,
                        const void* d_bp, const void* d_challenges,
                        const void* d_precomp_denoms,
                        uint32_t n, uint32_t npolys, uint32_t nbBsbGates,
                        void* d_result, void* stream);
int  gpu_plonk_scatter_result(const void* d_src, void* d_dst, uint32_t n, uint32_t rho,
                              uint32_t iter, uint32_t shift_bits, void* stream);
int  gpu_plonk_divide_zh(void* d_data, const void* d_inv_rho, uint32_t n, uint32_t rho,
                         uint32_t shift_bits, void* stream);
int  gpu_plonk_fold_quotient(const void* d_h1, const void* d_h2, const void* d_h3,
                             void* d_out, const void* d_zeta_n_plus_2, uint32_t n, void* stream);
int  gpu_plonk_restore_batch(void** d_polys, uint32_t count, int skip_idx,
                             const void* d_scale_powers, uint32_t log_n, void* stream);
int  gpu_poly_eval(const void* d_coeffs, uint32_t n, const void* d_point,
                   void* h_result, void* stream);
int  gpu_poly_eval_async(const void* d_coeffs, uint32_t n, const void* d_point,
                         void* d_partials, void* d_point_power, void* d_result, void* stream);
int  gpu_kzg_divide(const void* d_f, const void* d_a, const void* d_ainv, const void* d_one, uint32_t n, void* d_q, void* stream);
int  gpu_plonk_linearized_poly(
        const void* d_blindedZ, const void* d_s3, const void* d_ql, const void* d_qr,
        const void* d_qm, const void* d_qo, const void* d_qk, const void* d_hFolded,
        const void* d_scalars,
        uint32_t n, uint32_t n_blindedZ, uint32_t n_hFolded,
        void* d_result, void* stream);

// --- Copy-constraint ratio (grand product) -----------------------------------
int  gpu_ratio_copy_terms(const void* d_l, const void* d_r, const void* d_o,
                          const void* d_s1, const void* d_s2, const void* d_s3,
                          const void* d_twiddles0, const void* d_challenges,
                          uint32_t n, void* d_out_num, void* d_out_den, void* stream);
int  gpu_ratio_prefix_scan(void* d_data, uint32_t n, void* stream);
int  gpu_ratio_apply_inverse(void* d_coeffs, const void* d_den, uint32_t n, void* stream);

// --- Vector ops --------------------------------------------------------------
int  vec_from_mont(void* d_r, const void* d_a, uint32_t n, void* stream);
int  vec_to_mont(void* d_r, const void* d_a, uint32_t n, void* stream);
int  gpu_affine_from_mont(void* d_dst, const void* d_src, uint32_t n, void* stream);  // G1 affine Montgomery->canonical
int  gpu_vec_mul(void* d_r, const void* d_a, const void* d_b, uint32_t n, void* stream);
int  gpu_vec_denominators(void* d_r, const void* d_twiddles, const void* d_coset, uint32_t n, void* stream);

// ── plonk2-style resident layer (frvec.cu): device/stream/event/pinned + FrVector ops ──
int   gpu_mem_get_info(size_t* free_bytes, size_t* total_bytes);
int   gpu_device_sync(void);
void* gpu_event_create(void);
void  gpu_event_record(void* event, void* stream);
void  gpu_stream_wait_event(void* stream, void* event);
void  gpu_event_destroy(void* event);
int   gpu_alloc_pinned(void** ptr, size_t bytes);
void  gpu_free_pinned(void* ptr);
int   gpu_memcpy_d2d_on_stream(void* dst, const void* src, size_t size, void* stream);
int   gpu_vec_add(void* r, const void* a, const void* b, uint32_t n, void* stream);
int   gpu_vec_sub(void* r, const void* a, const void* b, uint32_t n, void* stream);
int   gpu_vec_scalar_mul(void* v, const void* d_c, uint32_t n, void* stream);
int   gpu_vec_addmul(void* v, const void* a, const void* b, uint32_t n, void* stream);
int   gpu_vec_add_scalar_mul(void* v, const void* a, const void* d_c, uint32_t n, void* stream);
int   gpu_vec_set_zero(void* v, uint32_t n, void* stream);
int   gpu_vec_scale_by_powers(void* v, const void* d_g, uint32_t n, void* stream);
int   gpu_vec_powers(void* v, const void* d_g, uint32_t n, void* stream);
int   gpu_vec_batch_invert(void* v, uint32_t n, void* stream);


#ifdef __cplusplus
} // extern "C"
#endif
