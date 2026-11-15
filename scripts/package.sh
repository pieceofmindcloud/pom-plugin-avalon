#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
platform=""
version=""
output="${root}/dist-release"
runtime=""
runtime_hip=""
repository="${GITHUB_REPOSITORY:-pieceofmindcloud/pom-plugin-avalon}"

usage() {
  printf '%s\n' \
    'Usage: scripts/package.sh --platform <linux-x86_64|macos-aarch64|windows-x86_64> --version <semver> [options]' \
    '  --output <dir>   package directory (default: dist-release)' \
    '  --runtime <tar.gz>      the NVIDIA runtime archive the release attaches (pinned into the library)' \
    '  --runtime-hip <tar.gz>  the AMD runtime archive the release attaches (pinned into the library)'
}

die() {
  printf 'package: %s\n' "$1" >&2
  exit 2
}

while (($# > 0)); do
  case "$1" in
    --platform) (($# >= 2)) || die '--platform requires a value'; platform="$2"; shift 2 ;;
    --version) (($# >= 2)) || die '--version requires a value'; version="$2"; shift 2 ;;
    --output) (($# >= 2)) || die '--output requires a value'; output="$2"; shift 2 ;;
    --runtime) (($# >= 2)) || die '--runtime requires a value'; runtime="$2"; shift 2 ;;
    --runtime-hip) (($# >= 2)) || die '--runtime-hip requires a value'; runtime_hip="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "$platform" ]] || die '--platform is required'
[[ -n "$version" ]] || die '--version is required'
command -v jq >/dev/null 2>&1 || die 'jq is required'

plugin_code="$(jq -er '.plugin_code | strings' "$root/ui/manifest.json")" || die 'manifest has no plugin code'
release_code="$(jq -er '.plugin_code | strings' "$root/release/manifest.json")" || die 'release manifest has no plugin code'
[[ "$plugin_code" =~ ^[a-z][a-z0-9_]{1,63}$ ]] || die 'manifest plugin code is invalid'
[[ "$plugin_code" == "$release_code" ]] || die 'release manifest plugin code does not match the UI manifest'
[[ "$(jq -er '.schema | numbers' "$root/release/manifest.json")" == 1 ]] || die 'release manifest schema is unsupported'

case "$platform" in
  linux-x86_64) os="linux"; arch="x86_64" ;;
  macos-aarch64) os="macos"; arch="aarch64" ;;
  windows-x86_64) os="windows"; arch="x86_64" ;;
  *) die "unsupported platform: $platform" ;;
esac

# The runtimes the release attaches: each is copied beside the package under its asset name, and its URL, SHA-256
# and size are compiled into the library (src/runtime.rs), which downloads it on the machine's first load.
mkdir -p "$output"
pin_runtime() {   # <archive> <asset name> <variable prefix>
  local archive="$1" asset="$2" prefix="$3"
  [[ -f "$archive" ]] || die "no runtime archive at $archive"
  [[ "$(cd "$(dirname "$archive")" && pwd -P)/$(basename "$archive")" == "$output/$asset" ]] || cp "$archive" "$output/$asset"
  export "${prefix}_URL=https://github.com/${repository}/releases/download/v${version}/${asset}"
  export "${prefix}_SHA256=$(sha256sum "$output/$asset" | cut -d' ' -f1)"
  export "${prefix}_SIZE=$(stat -c %s "$output/$asset")"
}
[[ -z "$runtime" ]] || pin_runtime "$runtime" "avalon-runtime-cuda-${platform}.tar.gz" AVALON_RUNTIME
[[ -z "$runtime_hip" ]] || pin_runtime "$runtime_hip" "avalon-runtime-hip-${platform}.tar.gz" AVALON_RUNTIME_HIP

build_output="$("$root/scripts/build.sh" --platform "$platform" --output "$output")"
artifact="$(printf '%s\n' "$build_output" | sed -n 's/^artifact=//p')"
sha256="$(printf '%s\n' "$build_output" | sed -n 's/^sha256=//p')"
size="$(printf '%s\n' "$build_output" | sed -n 's/^size=//p')"
target="$(printf '%s\n' "$build_output" | sed -n 's/^target=//p')"

metadata="${output}/pom-plugin-avalon-${platform}.metadata.json"
public_manifest="${output}/pom-plugin-${platform}.json"
preferences="$(jq -c '.preferences' "$root/release/manifest.json")" || die 'release manifest preferences are invalid'
feature_set="$(jq -c '.feature_set' "$root/release/manifest.json")" || die 'release manifest feature set is invalid'
cat > "$metadata" <<JSON
{
  "plugin_code": "${plugin_code}",
  "version": "${version}",
  "platform": "${platform}",
  "target": "${target}",
  "plugin_abi": 1,
  "sha256": "${sha256}",
  "size": ${size},
  "artifact": "$(basename "$artifact")",
  "feature_set": ${feature_set},
  "preferences": ${preferences}
}
JSON
jq -n \
  --slurpfile contract "$root/release/manifest.json" \
  --arg release_id "v${version}" \
  --arg version "$version" \
  --arg os "$os" \
  --arg arch "$arch" \
  --arg asset "$(basename "$artifact")" \
  --arg sha256 "$sha256" \
  --argjson size "$size" \
  '($contract[0]) + {
    release_id: $release_id,
    version: $version,
    os: $os,
    arch: $arch,
    asset: $asset,
    sha256: $sha256,
    size: $size
  }' > "$public_manifest" || die 'could not generate the GitHub release manifest'

printf 'artifact=%s\nmetadata=%s\npublic_manifest=%s\nsha256=%s\nsize=%s\nversion=%s\nplatform=%s\n' \
  "$artifact" "$metadata" "$public_manifest" "$sha256" "$size" "$version" "$platform"
