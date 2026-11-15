#!/usr/bin/env bash
# Builds the Avalon engine runtime into build/runtime from the commit that
# engine.lock pins. The runtime is never committed nor embedded in the library:
# a release attaches it (pinned URL, size and SHA-256), a development node reads
# build/runtime from the mounted source (src/launch.rs, find_runtime).
#
#   scripts/build-runtime.sh [--source <engine checkout>] [--hip]
#
# --hip builds the AMD runtime into build/runtime-hip inside ROCm's image
# (docker/runtime-hip.Dockerfile; ROCM_VERSION, default 7.2.4), or on the host
# when /opt/rocm exists. AMDGPU_TARGETS picks the cards (default
# gfx1100;gfx1101;gfx1102;gfx1201). The plugin runs it on AMD cards.
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
    --hip) export AVALON_BACKEND=hip; shift ;;
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
commit="${AVALON_COMMIT:-$(git -C "$source_dir" rev-parse HEAD)}"

# HIP without a ROCm on the host: run this script again inside ROCm's image.
if [[ "${AVALON_BACKEND:-}" == hip && ! -d /opt/rocm && -z "${AVALON_IN_CONTAINER:-}" ]]; then
  rocm_version="${ROCM_VERSION:-7.2.4}"
  image="pom-plugin-avalon-runtime:rocm$rocm_version"
  docker build --progress=plain --build-arg ROCM_VERSION="$rocm_version" \
    -f "$root/docker/runtime-hip.Dockerfile" -t "$image" "$root/docker"
  exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e AVALON_IN_CONTAINER=1 -e AVALON_COMMIT="$commit" \
    -e AVALON_BACKEND=hip -e AMDGPU_TARGETS="${AMDGPU_TARGETS:-}" -e AVALON_PORTABLE="${AVALON_PORTABLE:-0}" \
    -e BUILD_JOBS="${BUILD_JOBS:-}" -v "$root:/plugin" -v "$source_dir:/src:ro" \
    "$image" /plugin/scripts/build-runtime.sh --source /src
fi


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
out="$root/build/runtime"
[[ "$backend" == hip ]] && out="$root/build/runtime-hip"
staging="$out.new"
rm -rf "$staging"
mkdir -p "$staging/bin" "$staging/lib"
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF)
case "$backend" in
  hip)
    cmake_args+=(-DGGML_HIP=ON "-DAMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx1100;gfx1101;gfx1102;gfx1201}"
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
cmake --build "$build_dir" -j"${BUILD_JOBS:-$(nproc)}" --target llama-server llama-stage-server
cp "$build_dir/bin/llama-server" "$staging/bin/server"
# an earlier stage of a model split across machines (src/launch.rs, StageRole::Serve)
cp "$build_dir/bin/llama-stage-server" "$staging/bin/stage-server"
# The expert ranking the cache is prefilled from, when the engine ships one.
find "$source_dir" -maxdepth 2 -name 'moe-cache-ranking*.bin' -exec cp {} "$staging/" \; 2>/dev/null || true
# Carry the GPU runtime libraries the node image lacks.
if [[ "$(uname -s)" == Linux && "$backend" == cuda ]]; then
  ldd "$staging/bin/server" | awk '/=> \// {print $3}' \
    | grep -E '/usr/local/cuda|libcudart|libcublas' \
    | while read -r lib; do cp -L "$lib" "$staging/lib/"; done
elif [[ "$backend" == hip ]]; then
  # every library ldd resolves and the ones HIP opens itself (libamd_comgr),
  # with what they need in turn, but the C/C++ runtime the node image has
  rocm=/opt/rocm
  skip='^(linux-vdso|ld-linux|libc\.|libm\.|libdl\.|libpthread\.|librt\.|libstdc\+\+\.|libgcc_s\.)'
  search="$rocm/lib:$rocm/lib/llvm/lib:$rocm/lib/rocm_sysdeps/lib"
  copy_deps() {
    LD_LIBRARY_PATH="$search" ldd "$1" | awk '/=> \//{print $1, $3}' | while read -r name path; do
      if [[ ! "$name" =~ $skip && ! -e "$staging/lib/$name" ]]; then
        cp -L "$path" "$staging/lib/$name"; echo "$staging/lib/$name"
      fi
    done
  }
  pending="$(copy_deps "$staging/bin/server")"$'\n'"$(copy_deps "$staging/bin/stage-server")"
  for opened in libamd_comgr.so.3 libamd_comgr.so.2; do
    found="$(find -L "$rocm/lib" -maxdepth 1 -name "$opened" | head -1)"
    if [[ -n "$found" ]]; then cp -L "$found" "$staging/lib/$opened"; pending+=$'\n'"$staging/lib/$opened"; break; fi
  done
  while [[ -n "${pending//[[:space:]]/}" ]]; do
    next=""
    while read -r lib; do [[ -z "$lib" ]] || next+="$(copy_deps "$lib")"$'\n'; done <<< "$pending"
    pending="$next"
  done
  # rocBLAS / hipBLASLt read their kernels next to their .so: only this build's cards
  archs="${AMDGPU_TARGETS:-gfx1100;gfx1101;gfx1102;gfx1201}"
  for lib in rocblas hipblaslt; do
    [[ -d "$rocm/lib/$lib/library" ]] || continue
    mkdir -p "$staging/lib/$lib/library"
    find -L "$rocm/lib/$lib/library" -maxdepth 1 -type f | while read -r file; do
      name="$(basename "$file")"; keep=0
      [[ "$name" =~ gfx[0-9a-f]+ ]] || keep=1
      for a in ${archs//;/ }; do [[ "$name" == *"${a}"* ]] && keep=1; done
      if ((keep)); then cp -L "$file" "$staging/lib/$lib/library/"; fi
    done
  done
  if LD_LIBRARY_PATH="$staging/lib" ldd "$staging/bin/server" | grep 'not found'; then
    echo 'build-runtime: the server has unresolved libraries' >&2; exit 1
  fi
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
