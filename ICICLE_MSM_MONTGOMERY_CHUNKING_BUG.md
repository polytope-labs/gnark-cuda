# Bug: CUDA MSM returns wrong result with `are_*_montgomery_form=true` when the MSM is chunked

**Component:** open-icicle CUDA backend — MSM (`bls12_381`, and by symmetry all curves)
**Severity:** correctness (silent wrong result, no error returned)
**Found:** 2026-06-26, RTX 5090 (sm_120), CUDA 12.8.1, open-icicle built `-DCUDA_BACKEND=local -DCURVE=bls12_381`

## Summary

`bls12_381_msm` (and the templated `msm`) returns an **incorrect point** when **both**:

1. `config.are_scalars_montgomery_form == true` and/or `config.are_points_montgomery_form == true`, **and**
2. the MSM is large enough — relative to *free* GPU memory — to be split into multiple chunks by `multi_chunked_msm`.

With canonical-form inputs (`are_*_montgomery_form=false`) the same MSM is correct at every size. With Montgomery inputs the result is correct only while the MSM fits in a **single chunk**; once chunking kicks in, the result is wrong.

This is silent — `eIcicleError::SUCCESS` is returned.

## Why it hides

- icicle's own MSM tests (`tests/test_curve_api.cpp::MSM_test`) use `logn=12` (~4096 points) on an otherwise-empty GPU, so they (a) use canonical inputs and (b) never trigger chunking. The Montgomery + chunked path is untested.
- Chunking is decided dynamically from **free** memory (`cuda_msm.cuh::get_min_nof_chunks` → `compute_required_memory` → `cudaMemGetInfo`), so the same `(N, inputs)` can be single-chunk on a clean GPU and multi-chunk under memory pressure. The bug only appears in the latter.

## Suspected root cause

In `backend/cuda/src/msm/cuda_msm.cuh::multi_chunked_msm` the per-chunk Montgomery state is tracked by:

```cpp
bool internal_are_points_montgomery_form =
    (batch_same_points && i) || config.precompute_factor > 1
      ? false
      : config.are_points_montgomery_form;   // lines ~1319-1323
```

and each chunk's `bucket_method_msm` converts its slice from Montgomery in place (`upload_scalars`/`upload_points`, lines ~503-547). The scalar side passes `config.are_scalars_montgomery_form` unconditionally to every chunk (line ~1349). The interaction across chunks (which slices get converted, in which buffer, and whether the conversion is double-applied or skipped) appears to be the defect — non-batched multi-chunk MSM with Montgomery inputs yields a wrong sum.

## Minimal reproduction

`test_icicle_mont.cu` (pure icicle C-API; no external framework). For each N it computes the **same** MSM two ways on the CUDA device and compares (via `to_affine` + memcmp):
- (1) canonical inputs, `are_*_montgomery_form=false`
- (2) the same inputs converted to Montgomery (`convert_montgomery`, `is_into=true`), `are_*_montgomery_form=true`

A `cudaMalloc` "hog" leaves ~3 GB free to force chunking of the 8M case.

```
== forced chunking: hogged 29.9 GB, 3.2 GB free ==
N=2^10 … 2^20   canonical-MSM == montgomery-MSM ?  YES
N=2^23 (8M)     canonical-MSM == montgomery-MSM ?  NO   <-- MISMATCH
```

Without the hog (clean GPU, 8M fits in one chunk) all sizes match — confirming the trigger is chunking, not size per se.

Build:
```
nvcc -std=c++17 -O2 -arch=sm_120 test_icicle_mont.cu -o test_icicle_mont \
  -I<icicle>/include -I<icicle>/backend/cuda/include \
  -L<install>/lib -licicle_field_bls12_381 -licicle_curve_bls12_381 -licicle_device \
  --expt-relaxed-constexpr
ICICLE_BACKEND_INSTALL_DIR=<install>/lib/backend ./test_icicle_mont
```

(Repro source: `test/test_icicle_mont.cu` in this repo.)

## Workaround (what we do)

Convert inputs Montgomery→canonical **explicitly** on device once (`bls12_381_{scalar,affine}_convert_montgomery`, `is_into=false`), then call MSM with `are_*_montgomery_form=false`. This avoids icicle's internal per-chunk conversion entirely and is correct at all sizes regardless of chunking. (It's also cacheable for reused bases.)
