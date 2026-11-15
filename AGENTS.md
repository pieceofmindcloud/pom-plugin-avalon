# Project agent memory

- POM plugin `avalon` (`pom-plugin-avalon`), created from the POM `creating-pom-plugins` skill. The contract lives in the POM repository: `docs/subsystems/plugins-ui.md` and `docs/subsystems/enterprise-licensing.md`.
- `src/abi.rs` is the C ABI v1 boundary (entry `pom_avalon_plugin_v1` in `src/lib.rs`); keep domain code out of it. `src/lib.rs` dispatches `host.*`, `ui.*`, `rpc.*` and `peer.*` operations; `src/context.rs` reads `host.configure`.
- `query` must answer at once: the node waits 3 s and one slow call blocks every other. Slow work (downloads, child processes, indexing) runs on a thread the plugin owns.
- The plugin host speaks IPC over its own stdin/stdout: any child process gets `Stdio::null()` for stdin and a log file for its output, never the inherited streams.
- Secrets (`gateway.api_key`, `host_api.token`, upstream tokens) never appear in `rpc.*` replies, `host.services`, logs or screens.
- `ui/manifest.json` is the `pom-plugin-ui/v1` screen contract; `release/manifest.json` is the release contract (ABI, feature set, preferences). `ui/src/host/runtime.ts` and `ui/src/host/ready.ts` are the only files that read `__POM_HOST__`. CSS classes use the `pb-` prefix, which `ui/build.mjs` renames to the plugin code.
- `scripts/build-ui.sh` generates the ignored `ui/dist/`; `scripts/package.sh` builds an installable artifact and the per-platform GitHub manifest. The POM local rebuild runs `development.rebuild` with the source read-only (`scripts/dev-ui.sh` writes to `POM_PLUGIN_OUT_DIR`).
- The engine is the Avalon fork of llama.cpp (one model family, `qwen4exp` / Qwen3.8-Flash-Next, experts in RAM with a GPU expert cache). It runs as `llama-server` under the plugin: `src/launch.rs` maps `engine.load` to its flags (loopback, `--jinja`, KV type from `kv`, Avalon's `-b/-ub 9536`, MTP from the model folder's `MTP/*.gguf` with `--spec-type draft-mtp`), CUDA cards through `CUDA_VISIBLE_DEVICES`, AMD through `HIP_VISIBLE_DEVICES` from `gpu_devices`. `engine.lock` pins the fork's `pom` branch; its feature branches (`feature/external-tensor-data`, `feature/hidden-state-input`) are cherry-picks from the POM's llama.cpp fork, neutral, merged into `pom`.
- `scripts/build-runtime.sh` builds `llama-server` statically (`AVALON_BACKEND=hip|cuda|cpu`, `AMDGPU_TARGETS`, `CUDA_ARCHITECTURES`, `AVALON_PORTABLE=1` for a release) and copies ROCm/CUDA libraries into `build/runtime/lib`. Build from a worktree of the fork, never the checkout being edited.
- Verify with `scripts/build-ui.sh && POM_UI_REQUIRE_DIST=1 cargo test --locked && cargo fmt --check && cargo clippy --locked --all-targets -- -D warnings && tests/test_ci_plan.sh`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project. Do not repeat what the codebase already shows; point to the authoritative file or command instead. Prefer rewriting or pruning existing entries over appending new ones. When updating this file, preserve this bar for all agents and keep entries concise.
