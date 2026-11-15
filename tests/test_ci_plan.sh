#!/usr/bin/env bash
# The release workflow: Linux x86_64 with the NVIDIA and AMD runtimes pinned into the plugin.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }

workflow="$root/.github/workflows/release.yml"
[[ -f "$workflow" ]] || fail 'workflow is missing'
grep -q '^name: Release$' "$workflow" || fail 'workflow name'
grep -q 'backend: \[cuda, hip\]' "$workflow" || fail 'both runtimes must be built'
grep -q -- '--runtime runtimes/avalon-runtime-cuda.tar.gz --runtime-hip runtimes/avalon-runtime-hip.tar.gz' "$workflow" \
  || fail 'the package must pin both runtimes'
grep -q 'AVALON_PORTABLE=1' "$workflow" || fail 'release runtimes must be portable'
# Stable releases also go to the license server, POM's fallback for GitHub.
grep -q 'POM_RELEASE_TOKEN' "$workflow" || fail 'workflow must publish stable releases to the license server'
grep -q 'softprops/action-gh-release' "$workflow" || fail 'GitHub release step is missing'
grep -q 'dist-release/pom-plugin-linux-x86_64.json' "$workflow" || fail 'public update manifest must be attached'
grep -q 'persist-credentials: false' "$workflow" || fail 'checkout token should not persist'

printf 'test_ci_plan: ok\n'
