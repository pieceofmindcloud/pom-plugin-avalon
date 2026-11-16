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
pub const STAGES: bool = true;
/// Architectures whose stages this engine cuts itself from the whole model
/// (`range` in `engine.load`, announced as `stage_ranges`). Leave empty when
/// the engine reads the node's header-only virtual slice instead (a llama.cpp
/// derivative with the external tensor data loader).
pub const STAGE_RANGES: &[&str] = &[];
/// POM settings that apply to this engine's deployments (`pom_settings`).
pub const POM_SETTINGS: &[&str] = &[
    "cache_types",
    "batch",
    "sampling",
    "inference_role",
    "prefill_min_tokens",
    "hidden_state_dtype",
];
/// How long a model may take to load before the load is an error.
pub const LOAD_TIMEOUT_SECS: u64 = 1800;
/// The environment variable an operator sets to a runtime built by hand.
pub const RUNTIME_ENV: &str = "POM_AVALON_RUNTIME_DIR";
/// The same for the AMD (HIP) runtime.
pub const RUNTIME_HIP_ENV: &str = "POM_AVALON_RUNTIME_HIP_DIR";

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
    /// An earlier stage of a split model: the loopback port and secret it serves the next stage on (`engine.link`).
    pub link: Option<(u16, String)>,
}

/// What this machine runs of a model split by layers across machines (`link` in `engine.load`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageRole {
    /// The whole model, or a split's last stage that is also its first (no link to make).
    Whole,
    /// The last stage: it serves the requests and takes its input rows from the stage before it.
    Head { prev_port: u16, token: String },
    /// An earlier stage: it serves its rows to the next stage, and takes its own from the one before when it is a
    /// middle stage.
    Serve { prev: Option<(u16, String)> },
}

/// The stage this load is: from `link` (the POM gives `{}` to the first stage, `{token, prev_port}` to a later one)
/// and the slice's own `pom.*` keys (whether it holds the model's last layer).
pub fn stage_role(request: &LoadRequest) -> Result<StageRole, String> {
    let Some(link) = &request.link else {
        return Ok(StageRole::Whole);
    };
    let prev = match (link["prev_port"].as_u64(), link["token"].as_str()) {
        (Some(port), Some(token)) if port > 0 && port <= u64::from(u16::MAX) => {
            Some((port as u16, token.to_owned()))
        }
        (None, None) => None,
        _ => return Err("the stage link needs both prev_port and token".into()),
    };
    let last = crate::gguf::stage_of(&request.model_path).is_none_or(|stage| stage.last());
    Ok(match (last, prev) {
        (true, Some((prev_port, token))) => StageRole::Head { prev_port, token },
        (true, None) => StageRole::Whole,
        (false, prev) => StageRole::Serve { prev },
    })
}

/// What a usable runtime directory holds.
pub fn is_runtime(dir: &Path) -> bool {
    dir.join("bin/server").is_file()
}

/// Which build of the engine a load runs on: the AMD one when the node gives
/// no NVIDIA card and has an AMD card (the node lists NVIDIA cards in `gpus`
/// and every card in `gpu_devices`).
pub fn wants_hip(request: &LoadRequest) -> bool {
    request.gpus.is_empty() && !amd_ordinals(request).is_empty()
}

/// The runtime, first match wins: the operator's directory, then the
/// `build/runtime` (`build/runtime-hip` for AMD cards) of the source this
/// library was built from (the POM's development rebuild compiles the mounted
/// source, so the node reads the runtime straight from the mount), then the
/// plugin directory's copy (`runtime` or `runtime-hip`).
pub fn find_runtime(plugin_dir: &Path, hip: bool) -> Option<PathBuf> {
    let name = if hip { "runtime-hip" } else { "runtime" };
    let candidates = [
        std::env::var_os(if hip { RUNTIME_HIP_ENV } else { RUNTIME_ENV }).map(PathBuf::from),
        Some(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("build")
                .join(name),
        ),
        Some(plugin_dir.join(name)),
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
    // a virtual slice lives apart from the model: its draft is beside the files it reads
    let source = crate::gguf::external_source(model);
    let model = source.as_deref().unwrap_or(model);
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

/// What a saved session is bound to: the model file and the KV types (a slot file of another
/// model or KV layout does not restore).
pub fn session_fingerprint(request: &LoadRequest) -> String {
    let model = request
        .model_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let k = request
        .cache_type_k
        .clone()
        .unwrap_or_else(|| kv_name(&request.kv));
    let v = request
        .cache_type_v
        .clone()
        .unwrap_or_else(|| kv_name(&request.kv));
    format!("{model}|{k}|{v}|ctx{}", request.context)
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

/// Flash attention, on unless the deployment turns it off (a fallback for a card where it misbehaves).
fn flash_attn(request: &LoadRequest) -> bool {
    request
        .settings
        .get("flash_attn")
        .and_then(serde_json::Value::as_bool)
        != Some(false)
}

/// The V cache type: a quantized V cache needs flash attention, so without it V stays f16.
fn cache_v(request: &LoadRequest, flash: bool) -> String {
    let v = request
        .cache_type_v
        .clone()
        .unwrap_or_else(|| kv_name(&request.kv));
    if flash || v == "f16" || v == "f32" || v == "bf16" {
        v
    } else {
        "f16".into()
    }
}

fn setting_u64(request: &LoadRequest, key: &str) -> Option<u64> {
    request
        .settings
        .get(key)
        .and_then(serde_json::Value::as_u64)
}

pub fn command(runtime: &Path, request: &LoadRequest, port: u16) -> Result<Launch, String> {
    let role = stage_role(request)?;
    let whole = role == StageRole::Whole;
    let flash = flash_attn(request);
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
        if flash { "on" } else { "off" }.into(),
        "--cache-type-k".into(),
        request
            .cache_type_k
            .clone()
            .unwrap_or_else(|| kv_name(&request.kv)),
        "--cache-type-v".into(),
        cache_v(request, flash),
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
    if let Some(draft) =
        mtp_draft(&request.model_path).filter(|_| mtp && !matches!(role, StageRole::Serve { .. }))
    {
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
            if flash { "q8_0" } else { "f16" }.into(),
        ]);
    }
    // Conversation sessions: llama-server saves and restores a slot in the node's session folder.
    // a stage's sessions hold only its own layers: they are kept on a whole model only
    if let Some(dir) = request.session_dir.as_ref().filter(|_| whole) {
        args.extend(["--slot-save-path".into(), dir.display().to_string()]);
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
    // the RAM the experts take: an arena of that size instead of every expert copied when they fit
    if let Some(gb) = setting_u64(request, "expert_ram_gb").filter(|gb| *gb > 0) {
        env.push(("GGML_CUDA_MOE_ARENA_MB".into(), (gb * 1024).to_string()));
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
            // Avalon sizes each kernel's scratch on its dispatch (HSA_SCRATCH_SINGLE_LIMIT=1 MiB); ROCr's release of
            // it after the dispatch never finished on an RX 7800 XT (gfx1101): the context hung at creation, or later
            // at the first kernel needing more scratch. Kept, not released (freriadoc/Avalon#60).
            if std::env::var_os("HSA_NO_SCRATCH_RECLAIM").is_none() {
                env.push(("HSA_NO_SCRATCH_RECLAIM".into(), "1".into()));
            }
        }
    }
    let mut program = runtime.join("bin/server");
    let mut health_path = "/health".to_owned();
    let mut link = None;
    match role {
        StageRole::Whole => {}
        StageRole::Head { prev_port, token } => {
            args.extend(["--stage-prev".into(), format!("127.0.0.1:{prev_port}")]);
            // a restored checkpoint would hold this stage's layers only: the earlier stage would not follow it
            args.extend(["--ctx-checkpoints".into(), "0".into()]);
            env.push(("LLAMA_STAGE_PREV_TOKEN".into(), token));
        }
        StageRole::Serve { prev } => {
            program = runtime.join("bin/stage-server");
            let serve_port = crate::http::free_port()?;
            let secret = crate::http::random_hex(24);
            // the stage server answers no HTTP: it is ready once its port takes a connection
            health_path = String::new();
            args.retain(|arg| arg != "--no-webui" && arg != "--jinja");
            args.extend(["--stage-serve".into(), format!("127.0.0.1:{serve_port}")]);
            env.push(("LLAMA_STAGE_TOKEN".into(), secret.clone()));
            if let Some((prev_port, token)) = prev {
                args.extend(["--stage-prev".into(), format!("127.0.0.1:{prev_port}")]);
                env.push(("LLAMA_STAGE_PREV_TOKEN".into(), token));
            }
            link = Some((serve_port, secret));
        }
    }
    Ok(Launch {
        program,
        args,
        env,
        health_path,
        served_model: request.model_label.clone(),
        link,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use serde_json::Value;

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

    /// A header-only GGUF with the POM slice keys.
    fn slice(path: &Path, start: u32, end: u32, total: u32) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        for (key, value) in [
            ("pom.start_layer", start),
            ("pom.end_layer", end),
            ("pom.original_block_count", total),
        ] {
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(4u32.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn each_stage_of_a_split_runs_its_own_program() {
        let dir = std::env::temp_dir().join(format!("avalon-stage-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (first, last) = (dir.join("first.gguf"), dir.join("last.gguf"));
        slice(&first, 0, 23, 48);
        slice(&last, 24, 47, 48);
        let at = |path: &Path, link: Value| {
            LoadRequest::from_json(
                &json!({"model_path": path.display().to_string(), "link": link,
                "session_dir": dir.display().to_string()}),
            )
            .unwrap()
        };

        // the first stage serves its rows; the POM reaches it through engine.link
        let launch = command(Path::new("/rt"), &at(&first, json!({})), 1).unwrap();
        assert!(launch.program.ends_with("bin/stage-server"));
        let (serve_port, secret) = launch.link.clone().unwrap();
        assert!(launch
            .args
            .join(" ")
            .contains(&format!("--stage-serve 127.0.0.1:{serve_port}")));
        assert!(launch.env.contains(&("LLAMA_STAGE_TOKEN".into(), secret)));
        assert!(launch.health_path.is_empty());
        assert!(
            !launch.args.join(" ").contains("--slot-save-path"),
            "no sessions on a stage"
        );

        // the last stage serves the requests and reads its input from the relay
        let launch = command(
            Path::new("/rt"),
            &at(&last, json!({"token": "t", "prev_port": 4100})),
            1,
        )
        .unwrap();
        assert!(launch.program.ends_with("bin/server"));
        let args = launch.args.join(" ");
        assert!(
            args.contains("--stage-prev 127.0.0.1:4100") && args.contains("--ctx-checkpoints 0")
        );
        assert!(launch
            .env
            .contains(&("LLAMA_STAGE_PREV_TOKEN".into(), "t".into())));
        assert!(launch.link.is_none());

        // a middle stage does both
        let middle = dir.join("middle.gguf");
        slice(&middle, 16, 31, 48);
        let launch = command(
            Path::new("/rt"),
            &at(&middle, json!({"token": "t", "prev_port": 4100})),
            1,
        )
        .unwrap();
        let args = launch.args.join(" ");
        assert!(args.contains("--stage-serve") && args.contains("--stage-prev 127.0.0.1:4100"));

        assert!(stage_role(&at(&last, json!({"token": "t"}))).is_err());
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
        assert!(wants_hip(&request));
        assert!(launch
            .env
            .iter()
            .any(|(key, _)| key == "HSA_NO_SCRATCH_RECLAIM"));
        let no_fa =
            LoadRequest::from_json(&json!({"model_path": "/m.gguf", "cache_type_v": "q4_0",
            "settings": {"flash_attn": false}}))
            .unwrap();
        let args = command(Path::new("/rt"), &no_fa, 1).unwrap().args.join(" ");
        assert!(args.contains("--flash-attn off") && args.contains("--cache-type-v f16"));
        let capped = LoadRequest::from_json(
            &json!({"model_path": "/m.gguf", "settings": {"expert_ram_gb": 30}}),
        )
        .unwrap();
        assert!(command(Path::new("/rt"), &capped, 1)
            .unwrap()
            .env
            .contains(&("GGML_CUDA_MOE_ARENA_MB".into(), "30720".into())));
        let nvidia = LoadRequest::from_json(&json!({"model_path": "/m.gguf", "gpus": [0],
            "gpu_devices": [{"index": 0, "vendor": "nvidia", "ordinal": 0},
                            {"index": 1, "vendor": "amd", "ordinal": 0}]}))
        .unwrap();
        assert!(!wants_hip(&nvidia));
    }
}
