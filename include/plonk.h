#pragma once

#include "types.h"

// CUDA port of gnark-hip/include/plonk.h. Internal launcher + kernel decls used
// by api.cu. NTT/MSM are NOT declared here — those are delegated to icicle.

// Evaluate PLONK constraints at all n domain points (gate + ordering + local).
// d_polys: array of npolys device pointers (Fr[n] each, Lagrange Regular).
// d_twiddles0: w^i evaluation points (Fr[n]).
// d_bp: blinding coeffs [bl0,bl1, br0,br1, bo0,bo1, bz0,bz1,bz2] (Fr[9]).
// d_challenges: [alpha, beta, gamma, cs, css, coset, cosetExpNm1, cardInv] (Fr[8]).
// d_precomp_denoms: (coset*w^i - 1)^{-1} (Fr[n]). d_result: output (Fr[n]).
cudaError_t plonk_evaluate_constraints(
    Fr** d_polys,
    const Fr* d_twiddles0,
    const Fr* d_bp,
    const Fr* d_challenges,
    const Fr* d_precomp_denoms,
    uint32_t n,
    uint32_t npolys,
    uint32_t nbBsbGates,
    Fr* d_result,
    cudaStream_t stream);

// Polynomial evaluation kernels (chunked Horner), launched from api.cu.
#define EVAL_CHUNKS 256

__global__ void poly_eval_phase1_kernel(
    const Fr* __restrict__ coeffs,
    const Fr* __restrict__ point,
    Fr* __restrict__ partials,
    Fr* __restrict__ point_chunk_power,
    uint32_t n,
    uint32_t chunk_size);

__global__ void poly_eval_phase2_kernel(
    const Fr* __restrict__ partials,
    const Fr* __restrict__ point_chunk_power,
    Fr* __restrict__ result,
    uint32_t n_chunks);

// Linearized polynomial kernel. Scalars: [s1,s2,rl,lZeta,rZeta,oZeta,alpha2Lag,zhZeta].
__global__ void plonk_linearized_poly_kernel(
    const Fr* __restrict__ blindedZ,
    const Fr* __restrict__ trace_s3,
    const Fr* __restrict__ ql,
    const Fr* __restrict__ qr,
    const Fr* __restrict__ qm,
    const Fr* __restrict__ qo,
    const Fr* __restrict__ qk,
    const Fr* __restrict__ hFolded,
    const Fr* __restrict__ scalars,
    uint32_t n,
    uint32_t n_blindedZ,
    uint32_t n_hFolded,
    Fr* __restrict__ result);
