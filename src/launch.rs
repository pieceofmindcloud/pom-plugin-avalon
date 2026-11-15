//! What is particular to Avalon: where its runtime is, how its server is
//! started for one deployment, and what it can do. Everything else in the
//! plugin (supervision, the POM generation route, the service, the status)
//! is engine-agnostic. Adapt this file first; `feature-levels.md` of the
//! `pom-inference-engine` skill lists what each constant unlocks.
//!
//! The default below starts an OpenAI-compatible server in the llama.cpp
//! `llama-server` style. Replace the program, the arguments and the health
//! path with the engine's own, keeping: loopback only, the port given here,
//! the model path the POM sends, and no secret on the command line.

use crate::engine::LoadRequest;
use std::path::{Path, PathBuf};

/// GGUF `general.architecture` values this engine runs; the POM's deploy
/// screen offers the service only for these. Empty = the engine reads no
/// GGUF and the service is offered for nothing until this is filled.
pub const ARCHITECTURES: &[&str] = &["qwen4exp"];
/// The engine reads images through the deployment's projector (`mmproj_path`).
pub const VISION: bool = false;
/// The engine runs a POM layer slice as one stage of a split (feature L6).
pub const STAGES: bool = false;
/// Architectures whose stages this engine cuts itself from the whole model
/// (`range` in `engine.load`, announced as `stage_ranges`). Leave empty when
/// the engine reads the node's header-only virtual slice instead (a llama.cpp
/// derivative with the external tensor data loader).
pub const STAGE_RANGES: &[&str] = &[];
/// POM settings that apply to this engine's deployments (`pom_settings`).
pub const POM_SETTINGS: &[&str] = &["cache_types", "batch", "sampling"];
/// How long a model may take to load before the load is an error.
pub const LOAD_TIMEOUT_SECS: u64 = 1800;
/// The environment variable an operator sets to a runtime built by hand.
pub const RUNTIME_ENV: &str = "POM_AVALON_RUNTIME_DIR";

/// One start of the engine's server.
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Added to a cleared environment (`PATH` and `HOME` are kept).
    pub env: Vec<(String, String)>,
    /// A GET that answers 200 once the model is loaded.
    pub health_path: String,
    /// The model name the engine's OpenAI API expects in requests.
    pub served_model: String,
}

/// What a usable runtime directory holds.
pub fn is_runtime(dir: &Path) -> bool {
    dir.join("bin/server").is_file()
}

/// The runtime, first match wins: the operator's directory, then the
/// `build/runtime` of the source this library was built from (the POM's
/// development rebuild compiles the mounted source, so the node reads the
/// runtime straight from the mount), then the plugin directory's copy.
pub fn find_runtime(plugin_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        std::env::var_os(RUNTIME_ENV).map(PathBuf::from),
        Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("build/runtime")),
        Some(plugin_dir.join("runtime")),
    ];
    candidates.into_iter().flatten().find(|dir| is_runtime(dir))
}

/// The version a runtime reports, for the dashboard (`RUNTIME.json`).
pub fn runtime_version(runtime: &Path) -> Option<String> {
    let text = std::fs::read_to_string(runtime.join("RUNTIME.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value["version"].as_str().map(str::to_owned)
}

/// The MTP draft head beside the model: `MTP/*.gguf` in the model's folder,
/// as Avalon's model repository ships it.
pub fn mtp_draft(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?.join("MTP");
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "gguf"))
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The deployment's sampling defaults and the llama-server flag of each.
const SAMPLING_FLAGS: &[(&str, &str)] = &[
    ("temperature", "--temp"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
    ("repeat_penalty", "--repeat-penalty"),
    ("repeat_last_n", "--repeat-last-n"),
    ("presence_penalty", "--presence-penalty"),
    ("frequency_penalty", "--frequency-penalty"),
    ("dry_multiplier", "--dry-multiplier"),
    ("dry_base", "--dry-base"),
    ("dry_allowed_length", "--dry-allowed-length"),
    ("dry_penalty_last_n", "--dry-penalty-last-n"),
    ("seed", "--seed"),
];

/// An older node sends only `kv`, in its plugin names: `int8` is llama.cpp's `q8_0`.
fn kv_name(kv: &str) -> String {
    match kv {
        "int8" | "k8v4" => "q8_0".into(),
        "q4_0" => "q4_0".into(),
        other if !other.is_empty() => other.into(),
        _ => "f16".into(),
    }
}

/// The node's AMD cards as HIP numbers them, from `gpu_devices`.
fn amd_ordinals(request: &LoadRequest) -> Vec<String> {
    request
        .gpu_devices
        .iter()
        .filter(|device| device["vendor"].as_str() == Some("amd"))
        .filter_map(|device| device["ordinal"].as_u64().or(device["index"].as_u64()))
        .map(|ordinal| ordinal.to_string())
        .collect()
}

fn setting_u64(request: &LoadRequest, key: &str) -> Option<u64> {
    request
        .settings
        .get(key)
        .and_then(serde_json::Value::as_u64)
}

pub fn command(runtime: &Path, request: &LoadRequest, port: u16) -> Result<Launch, String> {
    let mut args: Vec<String> = vec![
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--model".into(),
        request.model_path.display().to_string(),
        "--alias".into(),
        request.model_label.clone(),
        "--ctx-size".into(),
        request.context.to_string(),
        "--parallel".into(),
        request.sessions.to_string(),
        "--jinja".into(),
        "--flash-attn".into(),
        "on".into(),
        "--cache-type-k".into(),
        request
            .cache_type_k
            .clone()
            .unwrap_or_else(|| kv_name(&request.kv)),
        "--cache-type-v".into(),
        request
            .cache_type_v
            .clone()
            .unwrap_or_else(|| kv_name(&request.kv)),
        "--no-webui".into(),
    ];
    // The deployment's sampling defaults: llama-server applies them to a request that does not set them.
    for (key, flag) in SAMPLING_FLAGS {
        if let Some(number) = request.sampling.get(*key).filter(|value| value.is_number()) {
            args.extend([flag.to_string(), number.to_string()]);
        }
    }
    if let Some(threads) = setting_u64(request, "threads").filter(|threads| *threads > 0) {
        args.extend(["--threads".into(), threads.to_string()]);
    }
    // Avalon reads a prompt in steps of 9536 tokens; the deployment's batch replaces it when set.
    let batch = request.n_batch.unwrap_or(9536);
    let ubatch = request.n_ubatch.unwrap_or(batch).min(batch);
    args.extend([
        "--batch-size".into(),
        batch.to_string(),
        "--ubatch-size".into(),
        ubatch.to_string(),
    ]);
    let mtp = request
        .settings
        .get("mtp")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    if let Some(draft) = mtp_draft(&request.model_path).filter(|_| mtp) {
        let draft_n = setting_u64(request, "draft_n").unwrap_or(2).clamp(1, 8);
        args.extend([
            "--model-draft".into(),
            draft.display().to_string(),
            "--spec-type".into(),
            "draft-mtp".into(),
            "--spec-draft-n-max".into(),
            draft_n.to_string(),
            "--ubatch-size-draft".into(),
            "256".into(),
            "--cache-type-k-draft".into(),
            "q8_0".into(),
            "--cache-type-v-draft".into(),
            "q8_0".into(),
        ]);
    }
    if let Some(mmproj) = request.mmproj_path.as_ref().filter(|_| VISION) {
        args.extend(["--mmproj".into(), mmproj.display().to_string()]);
    }
    let mut env = vec![(
        "LD_LIBRARY_PATH".into(),
        runtime.join("lib").display().to_string(),
    )];
    // Avalon's own switches, read from the environment by the engine.
    if request
        .settings
        .get("tail_skip")
        .and_then(serde_json::Value::as_bool)
        == Some(false)
    {
        env.push(("GGML_CUDA_MOE_TAIL_SKIP".into(), "0".into()));
    }
    if request
        .settings
        .get("gpu_stages")
        .and_then(serde_json::Value::as_bool)
        == Some(false)
    {
        env.push(("LLAMA_STAGES".into(), "0".into()));
    }
    let tensile = runtime.join("lib/rocblas/library");
    if tensile.is_dir() {
        env.push((
            "ROCBLAS_TENSILE_LIBPATH".into(),
            tensile.display().to_string(),
        ));
    }
    if !request.gpus.is_empty() {
        // numbered as nvidia-smi and the node number them, not fastest first
        let gpus: Vec<String> = request.gpus.iter().map(u32::to_string).collect();
        env.push(("CUDA_DEVICE_ORDER".into(), "PCI_BUS_ID".into()));
        env.push(("CUDA_VISIBLE_DEVICES".into(), gpus.join(",")));
    } else {
        let amd = amd_ordinals(request);
        if !amd.is_empty() {
            env.push(("HIP_VISIBLE_DEVICES".into(), amd.join(",")));
        }
    }
    Ok(Launch {
        program: runtime.join("bin/server"),
        args,
        env,
        health_path: "/health".into(),
        served_model: request.model_label.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_server_listens_on_loopback_on_the_given_port() {
        let request =
            LoadRequest::from_json(&json!({"model_path": "/models/m.gguf", "gpus": [1]})).unwrap();
        let launch = command(Path::new("/rt"), &request, 4242).unwrap();
        let args = launch.args.join(" ");
        assert!(args.contains("--host 127.0.0.1 --port 4242"));
        assert!(args.contains("--model /models/m.gguf"));
        assert!(launch
            .env
            .contains(&("CUDA_VISIBLE_DEVICES".into(), "1".into())));
        assert!(!args.contains("key"), "no secret on the command line");
        assert!(
            args.contains("--cache-type-k f16 --cache-type-v f16"),
            "{args}"
        );
        let old = LoadRequest::from_json(&json!({"model_path": "/m.gguf", "kv": "int8"})).unwrap();
        let args = command(Path::new("/rt"), &old, 1).unwrap().args.join(" ");
        assert!(
            args.contains("--cache-type-k q8_0"),
            "an older node's int8 is q8_0"
        );
        let new = LoadRequest::from_json(&json!({"model_path": "/m.gguf", "kv": "int8",
            "cache_type_k": "q8_0", "cache_type_v": "q4_0", "n_batch": 2048, "n_ubatch": 512,
            "sampling": {"temperature": 0.6, "seed": 7}}))
        .unwrap();
        let args = command(Path::new("/rt"), &new, 1).unwrap().args.join(" ");
        assert!(
            args.contains("--cache-type-k q8_0 --cache-type-v q4_0"),
            "{args}"
        );
        assert!(
            args.contains("--temp 0.6") && args.contains("--seed 7"),
            "{args}"
        );
        assert!(
            args.contains("--batch-size 2048 --ubatch-size 512"),
            "{args}"
        );
        assert!(!args.contains("--model-draft"), "no MTP folder, no draft");
    }

    #[test]
    fn the_mtp_head_beside_the_model_is_used_unless_turned_off() {
        let dir = std::env::temp_dir().join(format!("avalon-mtp-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("MTP")).unwrap();
        std::fs::write(dir.join("m.gguf"), b"GGUF").unwrap();
        std::fs::write(dir.join("MTP/mtp-head.gguf"), b"GGUF").unwrap();
        let model = dir.join("m.gguf").display().to_string();
        let on = LoadRequest::from_json(&json!({"model_path": model})).unwrap();
        let args = command(Path::new("/rt"), &on, 1).unwrap().args.join(" ");
        assert!(args.contains("--spec-type draft-mtp --spec-draft-n-max 2"));
        let off = LoadRequest::from_json(&json!({"model_path": model, "settings": {"mtp": false}}))
            .unwrap();
        let args = command(Path::new("/rt"), &off, 1).unwrap().args.join(" ");
        assert!(!args.contains("--model-draft"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn amd_cards_are_chosen_through_hip() {
        let request = LoadRequest::from_json(&json!({"model_path": "/m.gguf",
            "gpu_devices": [{"index": 0, "vendor": "amd", "ordinal": 0}]}))
        .unwrap();
        let launch = command(Path::new("/rt"), &request, 1).unwrap();
        assert!(launch
            .env
            .contains(&("HIP_VISIBLE_DEVICES".into(), "0".into())));
    }
}
