#!/usr/bin/env bash
# Builds the Avalon engine runtime into build/runtime from the commit that
# engine.lock pins. The runtime is never committed nor embedded in the library:
# a release attaches it (pinned URL, size and SHA-256), a development node reads
# build/runtime from the mounted source (src/launch.rs, find_runtime).
#
#   scripts/build-runtime.sh [--source <engine checkout>]
#
# Layout the plugin expects (src/launch.rs, is_runtime):
#   build/runtime/bin/server     the engine's OpenAI-compatible server
#   build/runtime/lib/           shared libraries the server needs, if any
#   build/runtime/RUNTIME.json   {"version", "engine_commit", "platform"}
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck disable=SC1091
source "$root/engine.lock"
source_dir="$root/build/runtime-src"
while (($# > 0)); do
  case "$1" in
    --source) source_dir="$(cd "$2" && pwd -P)"; shift 2 ;;
    *) printf 'build-runtime: unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ "$source_dir" == "$root/build/runtime-src" ]]; then
  if [[ ! -d "$source_dir/.git" ]]; then
    git clone --filter=blob:none "$ENGINE_REPOSITORY" "$source_dir"
  fi
  git -C "$source_dir" fetch --quiet origin "$ENGINE_BRANCH"
  git -C "$source_dir" checkout --quiet --detach "$ENGINE_COMMIT"
fi
commit="$(git -C "$source_dir" rev-parse HEAD)"

out="$root/build/runtime"
staging="$root/build/runtime.new"
rm -rf "$staging"
mkdir -p "$staging/bin" "$staging/lib"

# --- engine-specific: llama-server of the Avalon fork -----------------------
# Backend: AVALON_BACKEND=hip|cuda|cpu (default: hip when /opt/rocm exists,
# cuda when nvcc is found, else cpu). AMDGPU_TARGETS (default gfx1100;gfx1201)
# and CUDA_ARCHITECTURES (default: the release's wide list; pin one in
# development) pick the GPU code. ggml and llama are linked statically; the
# ROCm or CUDA libraries the server needs are copied into lib/.
backend="${AVALON_BACKEND:-}"
if [[ -z "$backend" ]]; then
  if [[ -d /opt/rocm ]]; then backend=hip
  elif command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then backend=cuda
  else backend=cpu; fi
fi
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
case "$backend" in
  hip)
    cmake_args+=(-DGGML_HIP=ON "-DAMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx1100;gfx1201}"
      -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++) ;;
  cuda)
    nvcc="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    # NCCL only serves several cards in one process and the node image has no libnccl.
    cmake_args+=(-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF "-DCMAKE_CUDA_COMPILER=$nvcc"
      "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-75;80;86;89;90;120}") ;;
  cpu) ;;
  *) printf 'build-runtime: unknown AVALON_BACKEND %s\n' "$backend" >&2; exit 2 ;;
esac
# A release runs on CPUs other than the builder's: AVALON_PORTABLE=1 drops -march=native.
[[ "${AVALON_PORTABLE:-0}" == 1 ]] && cmake_args+=(-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON)
build_dir="$root/build/engine-$backend"
cmake -S "$source_dir" -B "$build_dir" "${cmake_args[@]}"
cmake --build "$build_dir" -j"$(nproc)" --target llama-server llama-stage-server
cp "$build_dir/bin/llama-server" "$staging/bin/server"
# an earlier stage of a model split across machines (src/launch.rs, StageRole::Serve)
cp "$build_dir/bin/llama-stage-server" "$staging/bin/stage-server"
# The expert ranking the cache is prefilled from, when the engine ships one.
find "$source_dir" -maxdepth 2 -name 'moe-cache-ranking*.bin' -exec cp {} "$staging/" \; 2>/dev/null || true
# Carry the GPU runtime libraries the node image lacks.
if [[ "$(uname -s)" == Linux ]]; then
  ldd "$staging/bin/server" | awk '/=> \// {print $3}' \
    | grep -E '/opt/rocm|/usr/local/cuda|libcudart|libcublas|libhip|librocblas|libhipblas|libamdhip' \
    | while read -r lib; do cp -L "$lib" "$staging/lib/"; done
  # rocBLAS / hipBLASLt read their kernel libraries at run time.
  for dir in /opt/rocm/lib/rocblas/library /opt/rocm/lib/hipblaslt/library; do
    if [[ "$backend" == hip && -d "$dir" ]]; then
      mkdir -p "$staging/lib/$(basename "$(dirname "$dir")")"
      cp -r "$dir" "$staging/lib/$(basename "$(dirname "$dir")")/"
    fi
  done
fi
# ---------------------------------------------------------------------------

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) platform=linux-x86_64 ;;
  Linux-aarch64) platform=linux-aarch64 ;;
  Darwin-arm64) platform=macos-aarch64 ;;
  *) platform="$(uname -s)-$(uname -m)" ;;
esac
printf '{"version": "%s", "engine_commit": "%s", "platform": "%s", "backend": "%s"}\n' \
  "${commit:0:12}" "$commit" "$platform" "$backend" > "$staging/RUNTIME.json"
rm -rf "$out"
mv "$staging" "$out"
printf 'build-runtime: %s at %s\n' "$out" "${commit:0:12}"
