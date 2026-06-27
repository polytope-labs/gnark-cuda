cmake_minimum_required(VERSION 3.24)
project(gnark_cuda LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# gnark-cuda: a CUDA shared library that re-exports the byte-identical `gpu_*`
# C-ABI of libgnark_hip, so the gnark-crypto cgo seam links against it unchanged.
# Generic primitives (MSM/NTT/vec) delegate to a self-built open-icicle CUDA
# backend; the bespoke PLONK kernels are ported from gnark-hip.
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Blackwell / RTX 5090 = sm_120. Requires CUDA Toolkit >= 12.8 (RunPod template
# ships CUDA 13.0). Override with -DCMAKE_CUDA_ARCHITECTURES=... for other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Where the self-built open-icicle install lives (see ICICLE_CUDA_PORT_PLAN §0.2):
#   cmake -S icicle -B build -DCURVE=bls12_381 -DCUDA_BACKEND=local \
#         -DCMAKE_INSTALL_PREFIX=$HOME/icicle-install -DCUDA_ARCH=120
set(ICICLE_INSTALL_DIR "$ENV{HOME}/icicle-install" CACHE PATH "open-icicle install prefix")
# The trimmed open-icicle install does not ship all public headers (e.g.
# icicle/runtime.h); they live in the source tree. Point at it for includes.
set(ICICLE_SRC_DIR "" CACHE PATH "open-icicle source dir (for icicle/include headers)")

find_library(ICICLE_DEVICE   NAMES icicle_device              HINTS ${ICICLE_INSTALL_DIR}/lib)
find_library(ICICLE_FIELD    NAMES icicle_field_bls12_381     HINTS ${ICICLE_INSTALL_DIR}/lib)
find_library(ICICLE_CURVE    NAMES icicle_curve_bls12_381     HINTS ${ICICLE_INSTALL_DIR}/lib)

if(NOT ICICLE_DEVICE OR NOT ICICLE_FIELD OR NOT ICICLE_CURVE)
  message(WARNING
    "open-icicle libs not found under ${ICICLE_INSTALL_DIR}/lib. "
    "Build them first (see ICICLE_CUDA_PORT_PLAN.md §0.2) or pass "
    "-DICICLE_INSTALL_DIR=<prefix>. The bespoke-kernel sources still compile "
    "without them; only the icicle-delegating api.cu needs them at link time.")
endif()

add_library(gnark_cuda SHARED
  src/plonk.cu
  src/ratio.cu
    src/kzg.cu
  src/vecops.cu
  src/frvec.cu
  src/api.cu            # icicle-delegation layer + api-resident bespoke kernels
)

target_include_directories(gnark_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ICICLE_INSTALL_DIR}/include
  # icicle public headers (modular_arithmetic.h etc.) pull in CUDA-backend
  # private headers (gpu-utils/*, ptx.h), so both source include roots are needed.
  $<$<BOOL:${ICICLE_SRC_DIR}>:${ICICLE_SRC_DIR}/icicle/include>
  $<$<BOOL:${ICICLE_SRC_DIR}>:${ICICLE_SRC_DIR}/icicle/backend/cuda/include>
)

target_compile_options(gnark_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(gnark_cuda PUBLIC
  ${ICICLE_FIELD} ${ICICLE_CURVE} ${ICICLE_DEVICE}
)

set_target_properties(gnark_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON   # ratio.cu uses cub::DeviceScan
  POSITION_INDEPENDENT_CODE ON
)

# Optional standalone C++ correctness harness (M2/M3 gates), built when present.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_shim.cu)
  add_executable(test_shim test/test_shim.cu)
  target_link_libraries(test_shim PRIVATE gnark_cuda
    ${ICICLE_FIELD} ${ICICLE_CURVE} ${ICICLE_DEVICE})
  target_include_directories(test_shim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
    ${ICICLE_INSTALL_DIR}/include
    $<$<BOOL:${ICICLE_SRC_DIR}>:${ICICLE_SRC_DIR}/icicle/include>
    $<$<BOOL:${ICICLE_SRC_DIR}>:${ICICLE_SRC_DIR}/icicle/backend/cuda/include>)
endif()
