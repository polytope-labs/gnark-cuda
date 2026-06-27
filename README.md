# gnark-cuda

`libgnark_cuda` — a CUDA acceleration shim for the [gnark](https://github.com/Consensys/gnark)
BLS12-381 **PLONK** prover, built on the MIT-licensed
[open-icicle](https://github.com/ingonyama-zk/open-icicle) backend.

It exports a stable `gpu_*` C-ABI that the gnark fork binds via cgo. MSM and NTT
are delegated to open-icicle; the bespoke PLONK kernels (constraint evaluation,
grand-product, scatter, divide-by-Zh, fused FFT, coset NTT) live here.

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

The GPU prover lives entirely in the **gnark** fork, behind the `cuda` build tag.
The **gnark-crypto** fork carries only ~50 lines of nil-default registration
hooks — no GPU code, no cgo, no dependency on this shim:

- `ecc/bls12-381/multiexp.go` — `RegisterGPUMultiExp` + a guarded call site in `G1Jac.MultiExp`
- `ecc/bls12-381/fr/fft/fft.go` — `RegisterGPUFFT` + guarded call sites in `Domain.FFT` / `FFTInverse`

Everything GPU sits in the gnark fork:

- `internal/gpu/bls12381/` — the cgo bindings to `libgnark_cuda`; `gpu.go`
  carries the `#cgo LDFLAGS` (`-lgnark_cuda -licicle_field_bls12_381
  -licicle_curve_bls12_381 -licicle_device -lcudart -lstdc++`)
- `backend/accelerated/cuda/` — the `gpuMultiExp` / `gpuFFT{,Inverse,InverseCoset}`
  impls; its `init()` installs them via `RegisterGPUMultiExp` / `RegisterGPUFFT`,
  so importing this package (with `-tags cuda`) transparently routes MSM/FFT to the GPU
- `backend/plonk/bls12-381/prove{,_gpu,_nogpu}.go` (mirrored in `integration/`) —
  `prove_gpu.go` holds the device-resident quotient rho-loop; `prove.go` dispatches
  to it and falls back to the CPU path (`prove_nogpu.go`) without the tag

Downstream, a circuit opts in with a one-line side-effect import under the same
tag (e.g. the apk fork's `circuits/apk/gpu_register.go`:
`import _ "github.com/consensys/gnark/backend/accelerated/cuda"`).

Both forks are pinned via `replace` to `github.com/polytope-labs/{gnark,gnark-crypto}`.
Build with `-tags cuda` and point cgo at this shim + the icicle libs via the env:

```sh
export CGO_CFLAGS="-I<gnark-cuda>/include"
export CGO_LDFLAGS="-L<gnark-cuda>/build -L<icicle-install>/lib -L/usr/local/cuda/lib64"
go build -tags cuda ./...
```

Without `-tags cuda`, the hooks stay nil and the prover is pure-CPU upstream gnark.
