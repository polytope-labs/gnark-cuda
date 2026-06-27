#pragma once

#include "types.h"

// CUDA port of gnark-hip/include/ratio.h. Copy-constraint ratio (grand product).

// out_num[i] = Π_j (entry_j(w^i) + beta*id_j(w^i) + gamma)
// out_den[i] = Π_j (entry_j(w^i) + beta*sigma_j(w^i) + gamma)
// challenges = [beta, gamma, u, u^2]. All buffers device-resident.
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
    cudaStream_t stream = 0);

// In-place inclusive product scan (cub::DeviceScan::InclusiveScan, FrMulOp).
cudaError_t ratio_prefix_scan(Fr* d_data, uint32_t n, cudaStream_t stream = 0);

// In-place coeffs[i] *= inv(den[i]) for i in [0, n).
cudaError_t ratio_apply_inverse(Fr* d_coeffs, const Fr* d_den, uint32_t n, cudaStream_t stream = 0);
