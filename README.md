# gnark-cuda

`libgnark_cuda` — a CUDA acceleration shim for the [gnark](https://github.com/Consensys/gnark)
BLS12-381 **PLONK** prover, built on the MIT-licensed
[open-icicle](https://github.com/ingonyama-zk/open-icicle) backend.

It re-exports the exact `gpu_*` C-ABI that gnark-crypto's GPU seam binds, so the
Go above the C boundary compiles unchanged — only the cgo flags and the `hip`
build tag change. MSM and NTT are delegated to open-icicle; the bespoke PLONK
kernels (constraint evaluation, grand-product, scatter, divide-by-Zh, fused FFT,
coset NTT) live here.

On an RTX 5090 (Blackwell, sm_120) the apk-proofs circuit (7,097,608 constraints)
**proves and verifies in ~13.5 s** — ~7.3× the ~98 s CPU baseline — with the
quotient rho-loop running fully device-resident.

## Layout

| Path | What |
|---|---|
| `include/gnark_cuda.h` | the authoritative `gpu_*` C-ABI contract |
| `include/{fr,types}.h` | BLS12-381 Fr arithmetic (Montgomery) |
| `include/{plonk,ratio}.h` | bespoke kernel declarations |
| `src/api.cu` | icicle delegation (MSM/NTT) + coset NTT + fused FFT |
| `src/{plonk,ratio,vecops}.cu` | the ported PLONK kernels |
| `test/` | `test_shim` (field/MSM/NTT KATs) + `test_icicle_mont` (the icicle Montgomery-chunking repro) |
| `integration/` | reference copies of the gnark prover files (`prove*.go`) that drive the shim |
| `ICICLE_MSM_MONTGOMERY_CHUNKING_BUG.md` | upstream bug report |

## Build

Requires CUDA 12.8+, CMake, Ninja, and an open-icicle install for `bls12_381`.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DICICLE_INSTALL_DIR=/path/to/icicle-install \
  -DICICLE_SRC_DIR=/path/to/open-icicle/icicle
cmake --build build -j
```

## Integration

The prover lives in forks that pin this shim via cgo flags + the `cuda` build tag:

- **gnark** fork — `backend/plonk/bls12-381/prove{,_gpu,_nogpu}.go` (mirrored in `integration/`)
- **gnark-crypto** fork — the `ecc/bls12-381/gpu` seam re-pointed to `libgnark_cuda`

Build the prover with `-tags cuda` and cgo `LDFLAGS` pointing at `build/` +
the icicle libs.
