#include "plonk.h"
#include "fr.h"
#include <cuda_runtime.h>

// CUDA port of gnark-hip/src/plonk.hip. Kernel bodies are identical to the HIP
// original (pure Fr arithmetic); only the launch syntax and error types change.

// Polynomial indices (must match gnark prove.go)
enum {
    ID_L  = 0, ID_R  = 1, ID_O  = 2,
    ID_Z  = 3, ID_ZS = 4,
    ID_Ql = 5, ID_Qr = 6, ID_Qm = 7, ID_Qo = 8, ID_Qk = 9,
    ID_S1 = 10, ID_S2 = 11, ID_S3 = 12,
    ID_Qci = 13
};

// Evaluate degree-1 blind: bp[0] + bp[1]*x
__device__ __forceinline__ void eval_blind1(Fr& r, const Fr* bp, const Fr& x) {
    Fr tmp;
    fr::mul(tmp, bp[1], x);
    fr::add(r, bp[0], tmp);
}

// Evaluate degree-2 blind: bp[0] + bp[1]*x + bp[2]*x²
__device__ __forceinline__ void eval_blind2(Fr& r, const Fr* bp, const Fr& x) {
    Fr x2, t1, t2;
    fr::mul(t1, bp[1], x);
    fr::sqr(x2, x);
    fr::mul(t2, bp[2], x2);
    fr::add(r, bp[0], t1);
    fr::add(r, r, t2);
}

// Challenge layout in d_challenges[8]:
// [0]=alpha [1]=beta [2]=gamma [3]=cs [4]=css [5]=coset [6]=cosetExpNm1 [7]=cardInv

__global__ void plonk_constraint_kernel(
    Fr** __restrict__ polys,
    const Fr* __restrict__ twiddles0,
    const Fr* __restrict__ d_bp,        // 9 Fr values
    const Fr* __restrict__ d_challenges, // 8 Fr values
    const Fr* __restrict__ precomp_denoms,
    uint32_t n,
    uint32_t nbBsbGates,
    Fr* __restrict__ d_result)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Load challenges into registers
    Fr alpha = d_challenges[0];
    Fr beta  = d_challenges[1];
    Fr gamma = d_challenges[2];
    Fr cs    = d_challenges[3];
    Fr css   = d_challenges[4];
    Fr coset = d_challenges[5];
    Fr cosetExpNm1 = d_challenges[6];
    Fr cardInv     = d_challenges[7];

    // Load polynomial evaluations
    Fr uL  = polys[ID_L][tid];
    Fr uR  = polys[ID_R][tid];
    Fr uO  = polys[ID_O][tid];
    Fr uZ  = polys[ID_Z][tid];
    Fr uZS = polys[ID_ZS][(tid + 1) % n]; // shifted
    Fr uQl = polys[ID_Ql][tid];
    Fr uQr = polys[ID_Qr][tid];
    Fr uQm = polys[ID_Qm][tid];
    Fr uQo = polys[ID_Qo][tid];
    Fr uQk = polys[ID_Qk][tid];
    Fr uS1 = polys[ID_S1][tid];
    Fr uS2 = polys[ID_S2][tid];
    Fr uS3 = polys[ID_S3][tid];

    // Scale S1, S2, S3 by beta
    fr::mul(uS1, uS1, beta);
    fr::mul(uS2, uS2, beta);
    fr::mul(uS3, uS3, beta);

    // Add blinding: bp layout [bl0,bl1, br0,br1, bo0,bo1, bz0,bz1,bz2]
    Fr tw = twiddles0[tid];
    Fr y;

    eval_blind1(y, &d_bp[0], tw); fr::add(uL, uL, y);   // L += bl(ω^i)
    eval_blind1(y, &d_bp[2], tw); fr::add(uR, uR, y);   // R += br(ω^i)
    eval_blind1(y, &d_bp[4], tw); fr::add(uO, uO, y);   // O += bo(ω^i)
    eval_blind2(y, &d_bp[6], tw); fr::add(uZ, uZ, y);   // Z += bz(ω^i)

    Fr tw_next = twiddles0[(tid + 1) % n];
    eval_blind2(y, &d_bp[6], tw_next); fr::add(uZS, uZS, y); // ZS += bz(ω^{i+1})

    // === Gate constraint ===
    Fr ic, tmp;
    fr::mul(ic, uQl, uL);
    fr::mul(tmp, uQr, uR);    fr::add(ic, ic, tmp);
    fr::mul(tmp, uQm, uL);
    fr::mul(tmp, tmp, uR);     fr::add(ic, ic, tmp);
    fr::mul(tmp, uQo, uO);    fr::add(ic, ic, tmp);
    fr::add(ic, ic, uQk);

    for (uint32_t i = 0; i < nbBsbGates; i++) {
        Fr q0 = polys[ID_Qci + 2*i][tid];
        Fr q1 = polys[ID_Qci + 2*i + 1][tid];
        fr::mul(tmp, q0, q1);
        fr::add(ic, ic, tmp);
    }

    // === Ordering constraint ===
    Fr id;
    fr::mul(id, tw, coset);
    fr::mul(id, id, beta);

    Fr a, b, c, r, l;
    fr::add(a, gamma, uL); fr::add(a, a, id);
    fr::mul(b, id, cs);    fr::add(b, b, uR); fr::add(b, b, gamma);
    fr::mul(c, id, css);   fr::add(c, c, uO); fr::add(c, c, gamma);
    fr::mul(r, a, b); fr::mul(r, r, c); fr::mul(r, r, uZ);

    fr::add(a, uS1, uL); fr::add(a, a, gamma);
    fr::add(b, uS2, uR); fr::add(b, b, gamma);
    fr::add(c, uS3, uO); fr::add(c, c, gamma);
    fr::mul(l, a, b); fr::mul(l, l, c); fr::mul(l, l, uZS);

    Fr ordering;
    fr::sub(ordering, l, r);

    // === Local constraint ===
    Fr lone;
    fr::mul(lone, cosetExpNm1, cardInv);
    fr::mul(lone, lone, precomp_denoms[tid]);

    Fr local_c, one_m;
    fr::set_one(one_m);
    fr::sub(local_c, uZ, one_m);
    fr::mul(local_c, local_c, lone);

    // === Combine: gate + α·ordering + α²·local ===
    Fr result;
    fr::mul(result, local_c, alpha);
    fr::add(result, result, ordering);
    fr::mul(result, result, alpha);
    fr::add(result, result, ic);

    d_result[tid] = result;
}

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
    cudaStream_t stream)
{
    (void)npolys;
    uint32_t block = 256;
    uint32_t grid = (n + block - 1) / block;
    plonk_constraint_kernel<<<dim3(grid), dim3(block), 0, stream>>>(
        d_polys, d_twiddles0, d_bp, d_challenges, d_precomp_denoms,
        n, nbBsbGates, d_result);
    return cudaGetLastError();
}

// =============================================================================
// Polynomial evaluation at a point — chunked Horner's method
// =============================================================================

__global__ void poly_eval_phase1_kernel(
    const Fr* __restrict__ coeffs,
    const Fr* __restrict__ point,
    Fr* __restrict__ partials,
    Fr* __restrict__ point_chunk_power,
    uint32_t n,
    uint32_t chunk_size)
{
    uint32_t tid = threadIdx.x;
    if (tid >= EVAL_CHUNKS) return;

    uint32_t lo = tid * chunk_size;
    uint32_t hi = lo + chunk_size;
    if (hi > n) hi = n;

    Fr pt = *point;
    Fr acc;
    fr::set_zero(acc);

    // Horner from high to low: acc = c[hi-1] + pt*(c[hi-2] + pt*(...))
    if (lo < hi) {
        acc = coeffs[hi - 1];
        for (int i = (int)hi - 2; i >= (int)lo; i--) {
            fr::mul(acc, acc, pt);
            fr::add(acc, acc, coeffs[i]);
        }
    }
    partials[tid] = acc;

    // Thread 0 computes point^chunk_size
    if (tid == 0) {
        Fr pc;
        fr::set_one(pc);
        for (uint32_t i = 0; i < chunk_size; i++) {
            fr::mul(pc, pc, pt);
        }
        *point_chunk_power = pc;
    }
}

__global__ void poly_eval_phase2_kernel(
    const Fr* __restrict__ partials,
    const Fr* __restrict__ point_chunk_power,
    Fr* __restrict__ result,
    uint32_t n_chunks)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    Fr pck = *point_chunk_power;
    Fr acc;
    fr::set_zero(acc);

    if (n_chunks > 0) {
        acc = partials[n_chunks - 1];
        for (int i = (int)n_chunks - 2; i >= 0; i--) {
            fr::mul(acc, acc, pck);
            fr::add(acc, acc, partials[i]);
        }
    }
    *result = acc;
}

// =============================================================================
// Linearized polynomial kernel
// =============================================================================
// Scalars layout in d_scalars[8]:
//   [0]=s1  [1]=s2  [2]=rl  [3]=lZeta  [4]=rZeta  [5]=oZeta
//   [6]=alpha2Lag  [7]=zhZeta

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
    Fr* __restrict__ result)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_blindedZ) return;

    // Load scalar constants
    Fr s1      = scalars[0];
    Fr s2      = scalars[1];
    Fr rl      = scalars[2];
    Fr lZeta   = scalars[3];
    Fr rZeta   = scalars[4];
    Fr oZeta   = scalars[5];
    Fr alpha2Lag = scalars[6];
    Fr zhZeta  = scalars[7];

    // blindedZ[i] * (s2 + alpha2Lag)
    Fr s2_plus_a2l;
    fr::add(s2_plus_a2l, s2, alpha2Lag);

    Fr acc, tmp;
    fr::mul(acc, blindedZ[tid], s2_plus_a2l);

    if (tid < n) {
        // trace_s3[i] * s1
        fr::mul(tmp, trace_s3[tid], s1);
        fr::add(acc, acc, tmp);

        // qm[i] * rl
        fr::mul(tmp, qm[tid], rl);
        fr::add(acc, acc, tmp);

        // ql[i] * lZeta
        fr::mul(tmp, ql[tid], lZeta);
        fr::add(acc, acc, tmp);

        // qr[i] * rZeta
        fr::mul(tmp, qr[tid], rZeta);
        fr::add(acc, acc, tmp);

        // qo[i] * oZeta
        fr::mul(tmp, qo[tid], oZeta);
        fr::add(acc, acc, tmp);

        // + qk[i]
        fr::add(acc, acc, qk[tid]);
    }

    // - hFolded[i] * zhZeta
    if (tid < n_hFolded) {
        fr::mul(tmp, hFolded[tid], zhZeta);
        fr::sub(acc, acc, tmp);
    }

    result[tid] = acc;
}
