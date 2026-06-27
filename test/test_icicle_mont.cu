// Pure-icicle repro: is the large-n Montgomery-MSM divergence an icicle bug or
// our API misuse? No gnark, no gnark-cuda shim — only icicle's C-API and its own
// generated data. For each N we compute the SAME MSM two ways on the CUDA device:
//   (1) canonical inputs, are_*_montgomery_form = false
//   (2) the same inputs converted to Montgomery, are_*_montgomery_form = true
// icicle strips Montgomery internally, so both MUST yield the same point.
// If they diverge at large N (but match at small N), it's an icicle CUDA bug.

#include <cstdio>
#include <cstring>
#include <vector>
#include "icicle/runtime.h"
#include "icicle/device.h"
#include "icicle/msm.h"
#include "icicle/vec_ops.h"
#include "icicle/api/bls12_381.h"

using namespace icicle;
using bls12_381::scalar_t;
using bls12_381::affine_t;
using bls12_381::projective_t;

static bool run_n(int N) {
  std::vector<scalar_t> sc(N);
  std::vector<affine_t> pt(N);
  printf("  N=%d: gen scalars...\n", N);
  scalar_t::rand_host_many(sc.data(), N);            // canonical (icicle native)
  printf("  gen points...\n");
  projective_t::rand_host_many(pt.data(), N);        // canonical affine (icicle idiom)
  printf("  upload + msm...\n");

  void *dsc = nullptr, *dpt = nullptr, *dscM = nullptr, *dptM = nullptr;
  icicle_malloc(&dsc,  (size_t)N * sizeof(scalar_t));
  icicle_malloc(&dpt,  (size_t)N * sizeof(affine_t));
  icicle_malloc(&dscM, (size_t)N * sizeof(scalar_t));
  icicle_malloc(&dptM, (size_t)N * sizeof(affine_t));
  icicle_copy_to_device(dsc, sc.data(), (size_t)N * sizeof(scalar_t));
  icicle_copy_to_device(dpt, pt.data(), (size_t)N * sizeof(affine_t));
  printf("  copied; running canonical msm...\n");

  MSMConfig cfg = default_msm_config();
  cfg.are_scalars_on_device = true;
  cfg.are_points_on_device = true;
  cfg.are_results_on_device = false;
  cfg.is_async = false;
  cfg.are_scalars_montgomery_form = false;
  cfg.are_points_montgomery_form = false;

  // (1) canonical MSM
  projective_t Rc;
  eIcicleError e1 = bls12_381_msm((const scalar_t*)dsc, (const affine_t*)dpt, N, &cfg, &Rc);
  printf("  canonical msm done (err=%d); converting + montgomery msm...\n", (int)e1);

  // convert canonical -> Montgomery on device
  VecOpsConfig vc = default_vec_ops_config();
  vc.is_a_on_device = true; vc.is_result_on_device = true; vc.is_async = false;
  bls12_381_scalar_convert_montgomery((const scalar_t*)dsc, (size_t)N, /*is_into=*/true, &vc, (scalar_t*)dscM);
  printf("    scalar->mont done\n");
  bls12_381_affine_convert_montgomery((const affine_t*)dpt, (size_t)N, /*is_into=*/true, &vc, (affine_t*)dptM);
  printf("    affine->mont done\n");

  // (2) Montgomery MSM (icicle strips internally)
  MSMConfig cfgM = cfg;
  cfgM.are_scalars_montgomery_form = true;
  cfgM.are_points_montgomery_form = true;
  projective_t Rm;
  eIcicleError e2 = bls12_381_msm((const scalar_t*)dscM, (const affine_t*)dptM, N, &cfgM, &Rm);
  printf("    montgomery msm done (err=%d); comparing...\n", (int)e2);

  // Compare via host-side to_affine (bls12_381_eq runs a device kernel on host
  // pointers -> would crash). Same point -> same normalized affine bytes.
  affine_t Ac, Am;
  bls12_381_to_affine(&Rc, &Ac);
  bls12_381_to_affine(&Rm, &Am);
  bool eq = (memcmp(&Ac, &Am, sizeof(affine_t)) == 0);
  printf("N=2^%-2d (%-8d)  canonical-MSM == montgomery-MSM ? %s\n",
         (int)__builtin_ctz((unsigned)N), N, eq ? "YES" : "NO   <-- MISMATCH");
  icicle_free(dsc); icicle_free(dpt); icicle_free(dscM); icicle_free(dptM);
  return eq;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  icicle_load_backend_from_env_or_default();
  Device dev{"CUDA", 0};
  icicle_set_device(dev);
  // Hog most of GPU memory (strict cudaMalloc) so icicle's MSM must CHUNK (it
  // decides via cudaMemGetInfo's free-memory). Leave ~3GB so the 8M MSM can run
  // but is forced into multiple chunks — the memory-pressure conditions of the
  // real prove that the clean run never hit.
  void* hog = nullptr;
  size_t freeb = 0, totalb = 0; cudaMemGetInfo(&freeb, &totalb);
  size_t hog_bytes = freeb > (3ULL << 30) ? freeb - (3ULL << 30) : 0;
  while (hog_bytes > (1ULL << 30) && cudaMalloc(&hog, hog_bytes) != cudaSuccess) hog_bytes -= (512ULL << 20);
  cudaMemGetInfo(&freeb, &totalb);
  printf("== forced chunking: hogged %.1f GB, %.1f GB free ==\n", hog_bytes / 1e9, freeb / 1e9);

  int bad = 0;
  for (int logn : {10, 12, 14, 16, 18, 20, 23}) {
    if (!run_n(1 << logn)) bad++;
  }
  if (hog) icicle_free(hog);
  printf("\n=> %s\n", bad ? "MONTGOMERY MISMATCH in icicle CUDA MSM at large n (pure-icicle repro; not our code)"
                          : "all sizes match (no icicle bug)");
  return bad;
}
