#pragma once

#include "types.h"

// =============================================================================
// BLS12-381 Fr — 64-bit limbs, CIOS Montgomery multiply.
//
// Ported VERBATIM from gnark-hip/include/fr.h: the only device intrinsic used
// is __umul64hi, which is native to CUDA, so this header compiles unchanged
// under nvcc. The Montgomery constants (P, R_MOD_P, R2_MOD_P, MU) MUST match
// gnark-crypto and icicle bit-for-bit — verified at runtime in M3.
// =============================================================================

namespace fr {

__device__ __constant__ const uint64_t P[4] = {
    0xffffffff00000001ULL, 0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL, 0x73eda753299d7d48ULL
};

__device__ __constant__ const uint64_t R_MOD_P[4] = {
    0x00000001fffffffeULL, 0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL
};

__device__ __constant__ const uint64_t R2_MOD_P[4] = {
    0xc999e990f3f29c6dULL, 0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL, 0x0748d9d99f59ff11ULL
};

__device__ __constant__ const uint64_t MU = 0xfffffffeffffffffULL;

DEVICE_INLINE void adc(uint64_t& r, uint64_t& carry, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t s1 = a + b;
    uint64_t c1 = s1 < a ? 1ULL : 0ULL;
    r = s1 + c;
    carry = c1 + (r < s1 ? 1ULL : 0ULL);
}

DEVICE_INLINE void sbb(uint64_t& r, uint64_t& borrow, uint64_t a, uint64_t b, uint64_t bin) {
    uint64_t s = a - b;
    uint64_t b1 = s > a ? 1ULL : 0ULL;
    r = s - bin;
    borrow = b1 + (r > s ? 1ULL : 0ULL);
}

DEVICE_INLINE void add(Fr& r, const Fr& a, const Fr& b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) adc(r.limbs[i], carry, a.limbs[i], b.limbs[i], carry);
    uint64_t borrow = 0;
    uint64_t tmp[4];
    for (int i = 0; i < 4; i++) sbb(tmp[i], borrow, r.limbs[i], P[i], borrow);
    uint64_t use_tmp = carry | (borrow == 0 ? 1ULL : 0ULL);
    for (int i = 0; i < 4; i++) r.limbs[i] = use_tmp ? tmp[i] : r.limbs[i];
}

DEVICE_INLINE void sub(Fr& r, const Fr& a, const Fr& b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) sbb(r.limbs[i], borrow, a.limbs[i], b.limbs[i], borrow);
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) adc(r.limbs[i], carry, r.limbs[i], P[i], carry);
    }
}

DEVICE_INLINE void mul(Fr& r, const Fr& a, const Fr& b) {
    uint64_t t[5] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo = a.limbs[i] * b.limbs[j];
            uint64_t hi = __umul64hi(a.limbs[i], b.limbs[j]);
            uint64_t sum = t[j] + lo;
            uint64_t c1 = sum < t[j] ? 1ULL : 0ULL;
            sum += carry;
            c1 += sum < carry ? 1ULL : 0ULL;
            t[j] = sum;
            carry = hi + c1;
        }
        t[4] += carry;

        uint64_t m = t[0] * MU;
        carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t lo = m * P[j];
            uint64_t hi = __umul64hi(m, P[j]);
            uint64_t sum = t[j] + lo;
            uint64_t c1 = sum < t[j] ? 1ULL : 0ULL;
            sum += carry;
            c1 += sum < carry ? 1ULL : 0ULL;
            t[j] = sum;
            carry = hi + c1;
        }
        t[4] += carry;

        for (int j = 0; j < 4; j++) t[j] = t[j + 1];
        t[4] = 0;
    }

    uint64_t borrow = 0;
    uint64_t tmp[4];
    for (int i = 0; i < 4; i++) sbb(tmp[i], borrow, t[i], P[i], borrow);
    for (int i = 0; i < 4; i++) r.limbs[i] = borrow ? t[i] : tmp[i];
}

DEVICE_INLINE void sqr(Fr& r, const Fr& a) { mul(r, a, a); }

DEVICE_INLINE void neg(Fr& r, const Fr& a) {
    uint64_t is_zero = 0;
    for (int i = 0; i < 4; i++) is_zero |= a.limbs[i];
    if (is_zero == 0) { for (int i = 0; i < 4; i++) r.limbs[i] = 0; return; }
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) sbb(r.limbs[i], borrow, P[i], a.limbs[i], borrow);
}

DEVICE_INLINE void pow(Fr& r, const Fr& a, const uint64_t exp[4]) {
    for (int i = 0; i < 4; i++) r.limbs[i] = R_MOD_P[i];
    Fr base; for (int i = 0; i < 4; i++) base.limbs[i] = a.limbs[i];
    for (int i = 0; i < 4; i++) {
        uint64_t word = exp[i];
        for (int bit = 0; bit < 64; bit++) {
            if (word & 1) mul(r, r, base);
            sqr(base, base);
            word >>= 1;
        }
    }
}

DEVICE_INLINE void inv(Fr& r, const Fr& a) {
    const uint64_t pm2[4] = {
        0xfffffffeffffffffULL, 0x53bda402fffe5bfeULL,
        0x3339d80809a1d805ULL, 0x73eda753299d7d48ULL
    };
    pow(r, a, pm2);
}

DEVICE_INLINE void set_zero(Fr& r) { for (int i = 0; i < 4; i++) r.limbs[i] = 0; }
DEVICE_INLINE void set_one(Fr& r) { for (int i = 0; i < 4; i++) r.limbs[i] = R_MOD_P[i]; }
DEVICE_INLINE bool is_zero(const Fr& a) { uint64_t z = 0; for (int i = 0; i < 4; i++) z |= a.limbs[i]; return z == 0; }
DEVICE_INLINE bool eq(const Fr& a, const Fr& b) { uint64_t d = 0; for (int i = 0; i < 4; i++) d |= (a.limbs[i] ^ b.limbs[i]); return d == 0; }

DEVICE_INLINE void to_mont(Fr& r, const Fr& a) {
    Fr r2; for (int i = 0; i < 4; i++) r2.limbs[i] = R2_MOD_P[i];
    mul(r, a, r2);
}
DEVICE_INLINE void from_mont(Fr& r, const Fr& a) {
    Fr one = {{1, 0, 0, 0}};
    mul(r, a, one);
}

} // namespace fr
