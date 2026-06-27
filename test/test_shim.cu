// M3/M4 correctness harness for libgnark_cuda, run on a real sm_120 GPU.
//   A. field/Montgomery: gpu_vec_mul(x, 1_mont) == x   (exercises fr::mul + layout)
//   B. MSM KAT: gpu_msm over 10 Montgomery points/scalars == MSM_KAT_EXPECTED
//   C. NTT round-trip: inverse(forward(v)) == v  (probes the kNN NTT mapping)
//
// Diagnostic-heavy on purpose: prints actual vs expected limbs so the form
// (Montgomery vs canonical, projective convention) is observable, not guessed.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "gnark_cuda.h"
#include "icicle/runtime.h"
#include "icicle/api/bls12_381.h"
#include "icicle/vec_ops.h"

using namespace icicle;

// POD layout matching gnark-hip types.h (and icicle scalar_t/affine_t bytes).
struct Fp { uint64_t limbs[6]; };
struct G1Affine { Fp x; Fp y; };
#include "msm_kat.h"   // MSM_KAT_N, MSM_KAT_POINTS, MSM_KAT_SCALARS, MSM_KAT_EXPECTED (uses Fr, G1Affine)

static const uint64_t ONE_MONT[4] = { // R mod p (Montgomery 1), from fr.h R_MOD_P
    0x00000001fffffffeULL, 0x5884b7fa00034802ULL, 0x998c4fefecbc4ff5ULL, 0x1824b159acc5056fULL};

static int g_fail = 0;
static void check(bool ok, const char* name) {
    printf("[%s] %s\n", ok ? " OK " : "FAIL", name);
    if (!ok) g_fail++;
}
static void dump(const char* tag, const uint64_t* a, int n) {
    printf("    %s:", tag); for (int i=0;i<n;i++) printf(" %016lx", a[i]); printf("\n");
}

// ---------- A. field / Montgomery via gpu_vec_mul ----------
static void testFieldMul() {
    const uint32_t n = MSM_KAT_N;
    Fr ones[MSM_KAT_N];
    for (uint32_t i=0;i<n;i++) memcpy(ones[i].limbs, ONE_MONT, sizeof(ONE_MONT));
    void *da=gpu_malloc(n*sizeof(Fr)), *db=gpu_malloc(n*sizeof(Fr)), *dr=gpu_malloc(n*sizeof(Fr));
    gpu_memcpy_h2d(da, MSM_KAT_SCALARS, n*sizeof(Fr));
    gpu_memcpy_h2d(db, ones, n*sizeof(Fr));
    int rc = gpu_vec_mul(dr, da, db, n, nullptr);
    gpu_sync();
    Fr out[MSM_KAT_N];
    gpu_memcpy_d2h(out, dr, n*sizeof(Fr));
    bool ok = (rc==0) && memcmp(out, MSM_KAT_SCALARS, n*sizeof(Fr))==0;
    if (!ok) { dump("got[0] ", out[0].limbs,4); dump("want[0]", MSM_KAT_SCALARS[0].limbs,4); }
    check(ok, "A. gpu_vec_mul(x, 1_mont) == x  (fr::mul + Montgomery layout)");
    gpu_free(da); gpu_free(db); gpu_free(dr);
}

// ---------- B. MSM KAT ----------
static void testMsmKat() {
    const uint32_t n = MSM_KAT_N;
    void *dp=gpu_malloc(n*sizeof(G1Affine)), *ds=gpu_malloc(n*sizeof(Fr));
    gpu_memcpy_h2d(dp, MSM_KAT_POINTS, n*sizeof(G1Affine));   // Montgomery
    gpu_memcpy_h2d(ds, MSM_KAT_SCALARS, n*sizeof(Fr));        // Montgomery
    // gpu_msm now expects CANONICAL inputs (the seam does this + caches points).
    gpu_affine_from_mont(dp, dp, n, nullptr);
    vec_from_mont(ds, ds, n, nullptr);
    // gpu_msm writes a gnark G1Jac {X,Y,Z=1_mont}; with Z=1 the affine is the
    // first 96 bytes (X,Y in Montgomery) == MSM_KAT_EXPECTED.
    uint64_t jac[18] = {0};
    int rc = gpu_msm(dp, ds, n, jac, 0, nullptr);
    gpu_sync();
    bool ok = (rc==0) && memcmp(jac, &MSM_KAT_EXPECTED, sizeof(G1Affine))==0;
    if (!ok) { dump("got.x ", jac, 6); dump("want.x", (const uint64_t*)&MSM_KAT_EXPECTED.x, 6); }
    check(ok, "B. gpu_msm KAT matches (canonical-in, Jacobian-out)");
    gpu_free(dp); gpu_free(ds);
}

// ---------- C. NTT round-trip (kNN) ----------
static void testNttRoundtrip() {
    const uint32_t log_n = 4, n = 1u<<log_n;
    if (gpu_ntt_init(log_n) != 0) { check(false, "C. gpu_ntt_init"); return; }
    Fr v[16];
    for (uint32_t i=0;i<n;i++) v[i] = MSM_KAT_SCALARS[i % MSM_KAT_N];
    void* dv = gpu_malloc(n*sizeof(Fr));
    gpu_memcpy_h2d(dv, v, n*sizeof(Fr));
    int r1 = gpu_ntt(dv, log_n, 0, nullptr);   // forward
    int r2 = gpu_ntt(dv, log_n, 1, nullptr);   // inverse
    gpu_sync();
    Fr out[16];
    gpu_memcpy_d2h(out, dv, n*sizeof(Fr));
    bool ok = (r1==0 && r2==0) && memcmp(out, v, n*sizeof(Fr))==0;
    if (!ok) { dump("got[1] ", out[1].limbs,4); dump("want[1]", v[1].limbs,4); }
    check(ok, "C. gpu_ntt inverse(forward(v)) == v  (kNN round-trip)");
    gpu_ntt_cleanup();
    gpu_free(dv);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: survive a crash
    printf("step: main start\n");
    if (gpu_init(0) != 0) { printf("gpu_init failed\n"); return 1; }
    printf("=== libgnark_cuda M3/M4 harness (device count=%d) ===\n", gpu_device_count());
    printf("step: testFieldMul\n");  testFieldMul();
    printf("step: testMsmKat\n");    testMsmKat();
    printf("step: testNttRoundtrip\n"); testNttRoundtrip();
    printf("=== %s (%d failures) ===\n", g_fail==0 ? "ALL PASS" : "FAILURES", g_fail);
    return g_fail==0 ? 0 : 1;
}
