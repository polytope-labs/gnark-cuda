# Device-Resident BLS12-381 PLONK Prover — Build Plan

> Rebuild of the apk-proofs PLONK prover (n=2^23, RTX 5090) to keep field data
> device-resident end-to-end on top of the existing icicle shim. Modeled on Linea
> `gpu/plonk2` (gbotrel) + wnark `backend/accelerated/webgpu` (ivokub). Keeps
> AoS-Montgomery Fr (no SoA), keeps every proven kernel, borrows the orchestration
> + handle abstraction + transfer/compute overlap + proving-key residency cache.

See the milestone tasks (m0–m6). Full plan body below.

# Device-Resident BLS12-381 PLONK Prover — Definitive Build Plan

This plan rebuilds the apk-proofs PLONK prover (n=2^23, RTX 5090) to keep field data **device-resident end-to-end** on top of the existing icicle shim. We keep AoS-Montgomery `Fr` (no SoA transpose), keep every proven kernel (MSM, NTT, `gpu_kzg_divide`, `gpu_ratio_*`, `gpu_plonk_linearized_poly`, `gpu_plonk_evaluate`), and borrow only the **orchestration + handle abstraction + transfer/compute overlap** from Linea and the **proving-key residency cache** from wnark.

Ground-truth anchors verified in this session:
- Shim ABI: `/home/seun/Projects/gnark-cuda/include/gnark_cuda.h` (115 lines, all `gpu_*` void*+n+stream, no FrVector, no events, no pinned, no d2d-on-stream, no `vec_add`/`vec_sub`/`vec_scalar_mul` public symbols despite the launchers existing in `vecops.cu:68/75`).
- `vecops.cu`: `vec_add`/`vec_sub`/`vec_mul`/`vec_from_mont`/`vec_to_mont`/`vec_denominators` host launchers exist; `vec_neg_kernel`/`vec_scalar_mul_kernel` are **kernels only, no launcher** (`vecops.cu:28,34`).
- `api.cu:286-325`: the two-level coset power-table (`coset_table_build_kernel`/`coset_mul2_kernel`/`coset_scale_fast`) exists but is hardcoded to `g_coset_gen` (g=7) and private to `gpu_ntt_coset*`.
- `gpu.go`: residency reuse layer (canonCache, pointsCache LRU, fftPool, streamPool, `MSMDeviceScalarsWithStats` at :299) is already in place. `MSMDeviceScalarsWithStats` is the device-scalar MSM centerpiece and already does `vec_from_mont` correctly (no R^-1 fixup needed — unlike the Linea ref).
- `resident.go`: `CosetIFFTInverseDevice`, `InverseButterfliesDevice`, `InverseFFTDevice`, `VecMulDevice`, `LinearizedPolyDevice`, `KzgDivideDevice` already wrap device-pointer ops.
- `prove_gpu.go`: today only the **quotient rho-loop** is device-offloaded (`proverGPUContext`); commit/Z/linearize/open still round-trip per-op via `register.go` hooks.
- Integration seam: `register.go` registers per-op `gpuMultiExp`/`gpuFFT*`/`gpuDividePolyByXminusA` into gnark-crypto. The prover entry is `Prove()` at `prove.go:218`.

---

## 1. Package layout

### New Go package: `internal/gpu/bls12381/p2` (the device-resident orchestration layer)

A new sub-package so we don't churn the existing per-op hook layer. `p2` = "plonk2-style residency". Everything is `//go:build cuda` with `!cuda` stubs.

| File | Responsibility |
|---|---|
| `internal/gpu/bls12381/p2/device.go` | `Device` type: owns the fixed `streams [4]unsafe.Pointer` + `events [16]unsafe.Pointer`, `StreamID`/`EventID` int→handle lookup, `Bind()`(=`SetDevice`+`LockOSThread`), `Sync()`, `SyncStream(id)`, `RecordEvent(s,e)`, `WaitEvent(s,e)`, `MemGetInfo()`. Wraps the new C device/stream/event ABI. |
| `internal/gpu/bls12381/p2/frvector.go` | `FrVector{ptr unsafe.Pointer; n int; dev *Device}` + the full method set (§3). Thin AoS wrapper over `gpu_malloc(n*32)` + the new vec-op C calls. `runtime.SetFinalizer` backstop. |
| `internal/gpu/bls12381/p2/fft.go` | `FFTDomain{logN,size,dev}` thin wrapper over `NTTInit`. Methods `FFT`/`FFTInverse`/`BitReverse`/`CosetFFT`/`CosetFFTInverse` dispatching the existing `gpu_ntt_*`/`gpu_bitreverse`/`gpu_ntt_coset*` on a `StreamID`. |
| `internal/gpu/bls12381/p2/msm.go` | `G1MSM{basesDevPtr, n, dev}`: wraps `getCanonicalPoints` as a handle (bases resident, converted once). `MultiExp(v *FrVector) G1Jac` = `MSMDeviceScalarsWithStats(bases, v.ptr, ...)`. `MultiExpBatch` for the H1/H2/H3 + LRO batch commits. |
| `internal/gpu/bls12381/p2/pinned.go` | `PinnedFrBuffer{ptr, data []fr.Element}` via `gpu_alloc_pinned` + `unsafe.Slice`. Landing zones for the unavoidable D2H (quotient h1/h2/h3, FS commit scalars). |
| `internal/gpu/bls12381/p2/kernels.go` | Go wrappers for the fused PLONK kernels on `FrVector` handles: `ZComputeFactors`/`ZPrefixProduct` (=`gpu_ratio_*`), `PlonkEvaluate`/`ScatterResult`/`DivideZH`/`FoldQuotient`/`RestoreBatch` (existing bespoke kernels), `LinearizedPoly` (=`gpu_plonk_linearized_poly`), `KzgDivide`, `PolyEval`. Plus the **new** `BatchInvert`, `ScaleByPowers`, `ComputeL1Den`, `ReduceBlindedCoset`, `SubtractBlindingHead`. |
| `internal/gpu/bls12381/p2/enabled_cuda.go` / `enabled_nogpu.go` | `Enabled = true/false` so non-GPU builds compile. |

### Proving-key residency cache: `backend/plonk/bls12-381/`

| File | Responsibility |
|---|---|
| `prove_resident.go` (`//go:build cuda`, NEW) | The end-to-end device-resident orchestration (§4). Holds `residentPK` (the long-lived selector/SRS/twiddle/perm residency cache, §1.3) on `*ProvingKey`, and the phase functions `commitLRO`/`buildZ`/`computeQuotient`/`linearize`/`open` operating on `FrVector`s. |
| `prove_resident_nogpu.go` (`//go:build !cuda`, NEW) | Stubs. |
| `prove.go` (KEEP, minimal edit) | At `Prove()` (:218): if `p2.Enabled && residentEligible(spr)` dispatch to the resident path; else current path. One branch only. |

### What we KEEP vs REPLACE

**KEEP unchanged:**
- The entire shim's proven kernels: NTT family, MSM, `gpu_kzg_divide`, `gpu_ratio_*`, `gpu_plonk_*`, `gpu_vec_mul`, `gpu_affine_from_mont`, coset power-table.
- `gpu.go` reuse layer (canonCache, pointsCache, fftPool, streamPool, `MSMDeviceScalarsWithStats`, `NTTInit`/`NTTMaxLogN=25`).
- `register.go` per-op hooks — **kept for the CPU-fallback / sub-threshold / non-apk path and as the m0-m5 reference oracle**. The resident path bypasses them in the inner loop.
- `resident.go` wrappers (reused by `p2/kernels.go`).
- `verify.go`, `setup.go`, `marshal.go` — untouched.

**REPLACE (only on the resident path):**
- `prove_gpu.go`'s quotient-only offload → subsumed by `prove_resident.go`'s full pipeline (keep `prove_gpu.go` as the m0-m3 fallback for the not-yet-ported phases).
- The per-op `gpuFFT`/`gpuMultiExp` host round-trips inside the prove inner loop → resident `FrVector` ops.

---

## 2. C ABI delta (add to `gnark-cuda`)

All new symbols go in `include/gnark_cuda.h` and are implemented in `src/` (new `src/device.cu` for stream/event/pinned; extend `src/vecops.cu` for the element-wise ops; extend `src/api.cu` for `scale_by_powers` and `memcpy_d2d_on_stream`). The representation contract is unchanged: every field `void*` is an AoS-Montgomery `Fr` buffer.

### A. Device / stream / event management — NEW thin cuda-runtime wrappers

```c
// src/device.cu — new file. Pure CUDA runtime, no icicle.
int   gpu_mem_get_info(size_t* free_bytes, size_t* total_bytes);   // cudaMemGetInfo
void* gpu_event_create(void);                                       // cudaEventCreateWithFlags(cudaEventDisableTiming)
void  gpu_event_record(void* event, void* stream);                  // cudaEventRecord
void  gpu_stream_wait_event(void* stream, void* event);             // cudaStreamWaitEvent  (device-side cross-stream dep)
void  gpu_event_destroy(void* event);                               // cudaEventDestroy
```

- `gpu_stream_create`/`gpu_stream_sync`/`gpu_stream_destroy`: **already exist** (`gnark_cuda.h:44-46`). We keep the opaque-handle model (better than Linea's fixed-index `gnark_gpu_create_stream(ctx,id)`); the **Device holds the fixed `streams[4]`/`events[16]` array on the Go side**, mapping `StreamID`→handle. No `ctx` needed (context-less shim).
- **Critical caveat (icicle stream binding, see Risk R3):** icicle NTT/MSM take an `icicleStreamHandle`. Our `gpu_stream_create` returns exactly that handle as `void*`, so a stream created here is usable by both raw `cudaStreamWaitEvent` (cast to `cudaStream_t`) **and** icicle kernels. Verify in m0 that the `icicleStreamHandle` is a `cudaStream_t` under the hood (it is, in open-icicle CUDA backend) — if not, we maintain a parallel `cudaStream_t` and pass the icicle handle separately.

### B. Pinned host memory — NEW

```c
int  gpu_alloc_pinned(void** ptr, size_t bytes);   // cudaHostAlloc(cudaHostAllocDefault)
void gpu_free_pinned(void* ptr);                    // cudaFreeHost
```

Without these, `gpu_memcpy_*_on_stream` from pageable Go slices serialize the DMA (the multi-stream API is cosmetic). Needed for true Transfer/Compute overlap (m6).

### C. Async D2D — NEW (the only missing memcpy variant)

```c
int gpu_memcpy_d2d_on_stream(void* dst, const void* src, size_t size, void* stream);  // icicle_copy_..._async or cudaMemcpyAsync(D2D)
```

`gpu_memcpy_d2d` (sync) already exists (`gnark_cuda.h:39`). Only the on-stream variant is missing.

### D. FrVector element-wise ops — mix of "expose existing launcher" and "new trivial kernel"

| New C symbol | Implementation | Status |
|---|---|---|
| `int gpu_vec_add(void* r, const void* a, const void* b, uint32_t n, void* stream)` | calls existing `vec_add` launcher (`vecops.cu:68`) | **launcher EXISTS — ABI-only** |
| `int gpu_vec_sub(void* r, const void* a, const void* b, uint32_t n, void* stream)` | existing `vec_sub` (`vecops.cu:75`) | **launcher EXISTS — ABI-only** |
| `int gpu_vec_scalar_mul(void* v, const void* d_c, uint32_t n, void* stream)` | new launcher around existing `vec_scalar_mul_kernel` (`vecops.cu:34`); `d_c` = single device Fr | **kernel EXISTS, needs launcher+ABI** |
| `int gpu_vec_addmul(void* v, const void* a, const void* b, uint32_t n, void* stream)` | NEW kernel `v[i]+=a[i]*b[i]` (one `fr::mul`+`fr::add`) | **NEW trivial kernel** |
| `int gpu_vec_add_scalar_mul(void* v, const void* a, const void* d_c, uint32_t n, void* stream)` | NEW kernel `v[i]+=a[i]*(*c)` | **NEW trivial kernel** |
| `int gpu_vec_set_zero(void* v, uint32_t n, void* stream)` | `cudaMemsetAsync(v,0,n*32,stream)` — zero is valid Montgomery zero | **NEW (memset, no kernel)** |
| `int gpu_vec_scale_by_powers(void* v, const void* d_g, uint32_t n, void* stream)` | generalize `coset_table_build_kernel` (`api.cu:290`) to take an **arbitrary device `g`** instead of `g_coset_gen`; reuse `coset_mul2_kernel` | **NEEDS-VARIANT: generalize existing g=7 table** |
| `int gpu_vec_batch_invert(void* v, void* temp, uint32_t n, void* stream)` | NEW: two-level Montgomery batch invert. Reuse the `cub::DeviceScan::InclusiveScan` + `FrMulOp` prefix-product machinery already in `ratio.cu:121` / `kzg.cu:69`. prefix-product into `temp`, one `fr::inv` of the total, sweep back. | **NEW — the one non-trivial kernel** |

### E. PLONK helper kernels missing for full residency — NEW small kernels

| New C symbol | Implementation | Status |
|---|---|---|
| `int gpu_compute_l1_den(void* out, const void* d_coset_gen, const void* d_twiddles, uint32_t n, void* stream)` | NEW: `out[i] = cosetGen*omega^i - 1`. Trivial (`vec_denominators` already does `coset*tw[i]-1`; this is that **without** the final `inv` — caller `BatchInvert`s). | **NEW trivial (variant of `vec_denominators`)** |
| `int gpu_reduce_blinded_coset(void* dst, const void* src, const void* d_tail, uint32_t tail_len, const void* d_cosetPowN, uint32_t n, void* stream)` | NEW: `dst[i]=src[i]+src[n+j]*cosetPowN` for `j<tail_len`. Small kernel; `d_tail` is a few device Fr (uploaded once). | **NEW trivial** |
| `int gpu_subtract_blinding_head(void* v, const void* d_tail, uint32_t tail_len, void* stream)` | NEW: `v[i]-=tail[i]` for `i<tail_len`. Tiny. | **NEW trivial** |

We do **not** add `PlonkPermBoundary`/`PlonkGateAccum` separately — our `gpu_plonk_evaluate` (`api.cu:497`, the big fused gate+ordering+local kernel) already covers the same math in one launch and is proven by the current quotient rho-loop. We do **not** add `Butterfly4Inverse` — we keep our existing scatter/divide_zh/fold rho-loop quotient decomposition (`gpu_plonk_scatter_result`/`gpu_plonk_divide_zh`/`gpu_plonk_fold_quotient`), which is proven.

### F. MSM async/device-result — NEEDS-VARIANT (m6 only, optional)

`gpu_msm` forces `is_async=false` + result-to-host. For m6 overlap, add an optional stream-honoring variant that drops the internal `gpu_sync` and lets the caller fence with an event. Low priority — the result is a single G1 point and must go to host for Fiat-Shamir anyway.

**Summary of genuinely-new CUDA:** `gpu_vec_addmul`, `gpu_vec_add_scalar_mul`, `gpu_vec_batch_invert` (the only non-trivial one), `gpu_compute_l1_den`, `gpu_reduce_blinded_coset`, `gpu_subtract_blinding_head`, plus the generalized `gpu_vec_scale_by_powers`. Everything else is ABI-exposure of existing launchers or thin cuda-runtime/memset wrappers.

---

## 3. FrVector Go API (AoS-Montgomery, NOT SoA)

```go
// internal/gpu/bls12381/p2/frvector.go   //go:build cuda
type FrVector struct {
    ptr unsafe.Pointer // gpu_malloc(n*32), flat AoS-Montgomery Fr buffer
    n   int
    dev *Device
}
```

Every method takes a variadic `...StreamID` (default `StreamCompute=0`), pins the OS thread + `SetDevice` via `dev.bind()`, and dispatches the `_on_stream` C variant when a stream is given. **No transpose anywhere** — `ptr` is exactly what every existing kernel already consumes.

| Method | C call | Source |
|---|---|---|
| `NewFrVector(dev, n)` | `gpu_malloc(n*32)` + `SetFinalizer` | new |
| `(*FrVector).Free()` | `gpu_free` (idempotent, clear finalizer) | new |
| `Len() int` | metadata | — |
| `CopyFromHost(src fr.Vector, ...s)` | `gpu_memcpy_h2d[_on_stream]` (raw bytes, no transpose) | reuse `MemcpyH2D` |
| `CopyToHost(dst fr.Vector, ...s)` | `gpu_memcpy_d2h[_on_stream]` | reuse `MemcpyD2H` |
| `CopyFromDevice(src, ...s)` | `gpu_memcpy_d2d[_on_stream]` | **needs `_on_stream` (§2C)** |
| `SetZero(...s)` | `gpu_vec_set_zero` | new (memset) |
| `Mul(a,b,...s)` | `gpu_vec_mul` | **exists** |
| `Add(a,b,...s)` | `gpu_vec_add` | launcher exists, ABI-only |
| `Sub(a,b,...s)` | `gpu_vec_sub` | launcher exists, ABI-only |
| `AddMul(a,b,...s)` | `gpu_vec_addmul` | new kernel |
| `ScalarMul(c fr.Element,...s)` | `gpu_vec_scalar_mul` (upload c→1 device Fr) | kernel exists, needs launcher |
| `AddScalarMul(a, c fr.Element,...s)` | `gpu_vec_add_scalar_mul` | new kernel |
| `ScaleByPowers(g fr.Element,...s)` | `gpu_vec_scale_by_powers` | generalized table |
| `BatchInvert(temp *FrVector,...s)` | `gpu_vec_batch_invert` | new kernel |
| `Commit(msm *G1MSM) G1Jac` | `MSMDeviceScalarsWithStats(msm.bases, v.ptr, n, &res)` | **exists, the centerpiece** |
| `Eval(z fr.Element) fr.Element` | `gpu_poly_eval` (result→host, 1 Fr) | exists |

Scalar-by-value convention: upload the `fr.Element` to a 1-element device Fr (`gpu_malloc(32)`+h2d) and pass its pointer, matching the existing `gpu_kzg_divide` `d_a`/`d_ainv`/`d_one` convention. (A tiny per-call malloc; pool a handful of 1-Fr scratch slots on the Device to avoid churn.)

`FFTDomain` (fft.go) and `G1MSM` (msm.go) likewise wrap existing calls; `CosetFFT = ScaleByPowers(g) → FFT(DIF) → BitReverse` composed on the Go side, **or** the fused `gpu_ntt_coset` single call for the 4× quotient throughput path.

---

## 4. Device-resident prove flow

One persistent `residentPK` (on `*ProvingKey`, built once, §1.3): resident `dQl,dQr,dQm,dQo,dQk,dS1,dS2,dS3,dQcp` (canonical FrVectors), `dPerm` (int64 perm table via `gpu_malloc`+h2d), SRS bases as a `G1MSM` handle (canonical, converted once), NTT domain (`NTTInit(25)`), `dTwiddles0`. Plus ~12 reusable n-sized scratch FrVectors allocated once (mirrors Linea `allocPersistentBufs`). Guarded by `pk.prepareMu`; populated flags flipped only after successful upload (wnark `markPopulated` pattern).

Streams: `StreamCompute=0` (default), `StreamTransfer=1`, `StreamMSM=2`. Events gate cross-stream deps in the quotient loop only (m6).

| Phase | Resident buffers | Device ops | Stream | Host fence (structural) |
|---|---|---|---|---|
| **Prep PK** (once/key) | all selectors→FrVectors via `iFFT(Lagrange)`; SRS→`G1MSM`; perm→`dPerm`; twiddles | `CopyFromHost`→`BitReverse`→`FFTInverse` per selector | Compute | none (warmup) |
| **Solve + BSB22** | — (host solve) | committed values→`CopyFromHost`→`BitReverse`→`FFTInverse`→`Commit` | Compute/MSM | **commit point** (BSB22 digest→hash→witness) |
| **Commit L/R/O** | `dLCan,dRCan,dOCan` (resident heads kept), `dQkSrc` | per wire: `CopyFromHost`→`BitReverse`→`FFTInverse`→`CopyFromDevice`(keep Can); blinding via `SubtractBlindingHead`; `Commit` batch (reuse `MultiExpBatch`) | Compute; MSM | **proof.LRO[0..2]→gamma,beta** |
| **Grand product Z** | `dL,dR,dO`(evals), `dZCan` | `ZComputeFactors`(=`gpu_ratio_copy_terms`)→`BatchInvert`(den)→`Mul`→`ZPrefixProduct`(=`gpu_ratio_prefix_scan`+`apply_inverse`)→`iFFT`→`SubtractBlindingHead`→`Commit` | Compute; MSM | **proof.Z→alpha** |
| **Quotient** | resident `*Can` + selectors; 4 coset blocks | per coset: `ReduceBlindedCoset`→`CosetFFT`; selectors `CopyFromDevice`→`CosetFFT`; `ComputeL1Den`→`BatchInvert`; `gpu_plonk_evaluate`(fused perm+gate); per-qcp `Mul`+`AddScalarMul`. Then `iCosetFFT`+scatter/divide_zh/fold | Compute (m0-m5); +Transfer/MSM overlap (m6) | **one Sync + D2H h1/h2/h3; proof.H→zeta** |
| **Linearize** | `dZCan,dS3,dQl..dQk,dHFolded` resident | `gpu_plonk_linearized_poly` (`LinearizedPolyDevice`); per-qcp `AddScalarMul` | Compute | `CopyToHost(linPol[:n])`; **linPolDigest** |
| **Open** | resident `dS1,dS2,dQcp`,`dZCan` | `s1/s2/qcp ζ` via `PolyEval`(device); `KzgDivide`(=`gpu_kzg_divide`) for open-Z & batch-fold quotient; `Commit` | Compute; MSM | eval D2H (single Fr each); **batch-open H** |

Reused-as-is: `gpu_plonk_linearized_poly`, `gpu_kzg_divide`, `gpu_ratio_*`, `gpu_plonk_evaluate`/`scatter`/`divide_zh`/`fold`, `MSMDeviceScalarsWithStats`, `gpu_ntt_coset*`. Structurally-unavoidable host fences = exactly the Fiat-Shamir commit points (LRO, Z, H, BSB22, linPol, batch-H) where a G1 point must hit host to feed the transcript and produce the next challenge as a host scalar.

---

## 5. Milestone plan (ordered, each compilable + verifiable against the apk `TestPlonkProveAndVerify`)

Verification oracle for every milestone: the apk proof **must still verify**. Until a phase is ported, it stays on the current `prove_gpu.go`/`register.go` path, so the proof is always valid. Smallest-first, no big-bang.

- **m0 — Scaffolding builds, proof unchanged.** Add all §2 C symbols (stub-implement the new kernels but don't call them yet); add `p2.Device`/`FrVector`/`FFTDomain`/`G1MSM`/`Pinned` with `enabled_cuda.go`. Add `prove_resident.go` with `residentEligible()` returning **false**. *Verify:* `-tags cuda` builds; apk test passes on the unchanged path. Plus a standalone `p2` unit test round-tripping `CopyFromHost`/`CopyToHost` and each new vec op vs a CPU `fr.Vector` reference (this is where `BatchInvert`, `ScaleByPowers`, `addmul` Montgomery correctness is proven in isolation — see R2).

- **m1 — Resident proving-key cache, proof verifies via existing per-op path.** Build `residentPK`: upload selectors/SRS/perm/twiddles once, keyed by domain identity, populated flags. Don't yet drive the pipeline from it — only assert the cached canonical selectors **byte-match** the host-computed ones, and that the resident `G1MSM` produces the same LRO commitment as the current `gpuMultiExp`. *Verify:* apk passes; assert resident selectors == CPU selectors; second prove of same circuit re-uses cache (no re-upload — log it).

- **m2 — iFFT + commit L/R/O resident.** Flip `commitLRO` to the resident path (`FFTDomain` + `G1MSM.MultiExpBatch` + `SubtractBlindingHead`); everything downstream (Z, quotient, linearize, open) stays on the old path, fed from `CopyToHost` of the resident Can buffers. *Verify:* apk passes; LRO commitments bit-identical to old path; gamma/beta unchanged.

- **m3 — Grand-product Z on-device via `BatchInvert`.** Resident `buildZ` using `gpu_ratio_*` + the new `BatchInvert` (this is the first real use of the non-trivial new kernel in the full pipeline). Z commitment from resident buffer. *Verify:* apk passes; Z commitment bit-identical; alpha unchanged. This independently validates Montgomery-form correctness through `BatchInvert` end-to-end (R2).

- **m4 — Quotient resident.** Replace `prove_gpu.go`'s offload with the `prove_resident.go` 4-coset loop driven by resident `*Can`/selectors (`ReduceBlindedCoset`/`ComputeL1Den`/`gpu_plonk_evaluate`/scatter/divide_zh/fold). Single-stream (no events yet). *Verify:* apk passes; h1/h2/h3 D2H match old rho-loop; proof.H unchanged; zeta unchanged. **Gate: re-run M4 NTT-mode round-trip KATs on sm_120 first** (the `[FRAGILE]` ordering risk, R-NTT).

- **m5 — Linearize + open resident.** `LinearizedPolyDevice` + device `PolyEval` for s1/s2/qcp ζ + `KzgDivide` for the openings, fed from resident buffers. *Verify:* apk passes; batched-proof H and ZShifted.H bit-identical; full proof verifies with **zero per-op `gpuFFT` round-trips** in the inner loop (assert via a hook counter).

- **m6 — Stream overlap.** Introduce `StreamTransfer`/`StreamMSM` + events + pinned buffers in the quotient loop and around commits. Pure performance; **no proof-content change**, so verification is identical apk pass + a wall-clock delta against the m5 baseline (target: beat the ~13s PCIe ceiling per `project_gnark_prover_perf_ceiling`).

---

## 6. Risk register

- **R1 — AoS-vs-SoA confusion.** Mitigation: the contract is fixed in `gnark_cuda.h:17-21` (AoS-Montgomery, flat). We **do not port any transpose**. Every new kernel indexes `Fr*[tid]` exactly like the existing `vecops.cu`. The Linea SoA `copy_aos_to_soa_kernel` is reference-only and never touched. m0's `CopyFromHost`/`CopyToHost` round-trip unit test catches any accidental transpose immediately.

- **R2 — Montgomery form through `BatchInvert` (and the new fused ops).** The one genuinely-new non-trivial kernel. Risk: `fr::inv` on the prefix-product, or `addmul`/`scale_by_powers`, mishandling the Montgomery R-factor. Mitigation: `BatchInvert` reuses the **already-proven** `FrMulOp`+`cub::DeviceScan` from `ratio.cu`/`kzg.cu` (Montgomery-correct in production). m0 unit-tests every new op against a CPU `fr.Element` reference **before** any pipeline use; m3 independently validates `BatchInvert` end-to-end via the Z commitment bit-match. Note our advantage over wnark: we DMA Montgomery bytes directly, no boundary conversion.

- **R3 — icicle stream binding.** Risk: a `cudaStream_t` created for `cudaStreamWaitEvent` isn't the same handle icicle NTT/MSM enqueue on, so events don't actually order icicle kernels. Mitigation: m0 explicitly verifies `gpu_stream_create` returns an `icicleStreamHandle` that is a `cudaStream_t` (open-icicle CUDA backend: it is). If not, the Device keeps a parallel `cudaStream_t`↔`icicleStreamHandle` pair per StreamID and records events on the cuda handle. Until m6 everything runs on the default stream (proven today), so this risk is isolated to the perf milestone and cannot break correctness.

- **R4 — Fiat-Shamir fence is structural, not removable.** Risk: over-aggressively trying to keep commits on-device breaks the transcript. Mitigation: explicitly enumerate the unavoidable D2H points (LRO, Z, H, BSB22, linPol, batch-H — §4) as host fences by design; `Commit` always lands the G1 point on host (`gpu_msm` already does). We overlap *around* these fences (m6), never try to remove them.

- **R5 — VRAM budget at n=2^23.** Resident set: 9 selectors (256MB each = 2.3GB) + SRS bases (~768MB) + twiddles (to 2^25) + perm (3n int64 = 192MB) + ~12 scratch FrVectors (3GB) + 4 coset blocks ≈ pushing 9-10GB before icicle MSM/NTT working buffers. Mitigation: add `gpu_mem_get_info` (§2A) and gate; reuse the existing `fftPool`/`releaseReusableMemory` LRU; the RTX 5090 has 32GB so single-domain is comfortable, but the NTTInit-to-2^25 twiddle alloc already competes (`gpu.go:421-443` has the retry/release path) — keep that. If pressure appears, port Linea's MSM `offload_points`/`reload_points` (not built in m0-m5; only if `mem_get_info` shows <2GB headroom before the quotient cosets).

- **R-NTT — `[FRAGILE]` NTT ordering on sm_120.** The DIT/DIF/noscale/coset mapping onto icicle's Ordering+mandatory-1/n is annotated `[FRAGILE]` (`api.cu:368`). A wrong ordering silently produces wrong polys. Mitigation: gate **m4** behind re-running the per-mode round-trip KATs on sm_120 (forward∘inverse==identity, coset round-trip, fused `scale_fft` vs CPU reference) before trusting the resident quotient.

Relevant files: `/home/seun/Projects/gnark-cuda/include/gnark_cuda.h`, `/home/seun/Projects/gnark-cuda/src/{api.cu,vecops.cu,ratio.cu,kzg.cu}` (+ new `src/device.cu`), `/home/seun/Projects/gnark/internal/gpu/bls12381/{gpu.go,resident.go}` (+ new `p2/` package), `/home/seun/Projects/gnark/backend/plonk/bls12-381/{prove.go,prove_gpu.go}` (+ new `prove_resident.go`), `/home/seun/Projects/gnark/backend/accelerated/cuda/register.go`.