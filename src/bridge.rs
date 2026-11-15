//! The generation route the POM gateway calls (`engine.upstream`), on
//! loopback, in front of the engine's own server.
//!
//! The POM renders a conversation into one prompt text and posts
//! `POST /pom/generate` with `{prompt, chat_template, thinking, max_tokens,
//! stop, temperature, top_p, top_k, min_p, presence_penalty,
//! frequency_penalty, seed, session?, images?, probe?}` and
//! `Authorization: Bearer <token>`. The answer is one JSON object per line:
//!
//! - `{"t": text}` answer text and `{"r": text}` reasoning, in order;
//! - `{"p": [read, total]}` prompt progress (optional);
//! - `{"done": {finish, prompt_tokens, completion_tokens, reasoning_tokens,
//!   cached_tokens, timings, stop_sequence}}`, or `{"error": message}`.
//!
//! This first level translates to the engine's OpenAI-compatible streaming
//! API: `chat_template: true` becomes one user turn on `/v1/chat/completions`
//! (the engine applies the model's template), `false` a raw
//! `/v1/completions`. Reasoning must stay structured (`reasoning_content`),
//! never `<think>` text in `t`. Tool calls stay in the answer text in the
//! model's own form: the POM parses them. A capability the engine lacks is a
//! clear `{"error"}` line, never a silent fallback. Conversation sessions and
//! the prefill/decode split are `crate::sessions`: the routes answer 501 when
//! the engine was loaded without the node's session folder.

use crate::http::{self, NdjsonWriter, Request};
use serde_json::{json, Map, Value};
use std::io::BufRead;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// The numeric sampling fields the POM forwards from the client.
pub const SAMPLING_KEYS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "seed",
];

/// How long the engine may stay silent inside one generation.
const ENGINE_SILENCE: Duration = Duration::from_secs(600);

/// Where the bridge reaches the engine, while it is ready: its loopback port
/// and the model name its API expects.
pub type Target = Arc<dyn Fn() -> Option<(u16, String)> + Send + Sync>;

pub struct Bridge {
    pub port: u16,
    pub token: String,
    pub in_flight: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
}

impl Bridge {
    pub fn start(target: Target) -> Result<Self, String> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
        let port = listener
            .local_addr()
            .map_err(|error| error.to_string())?
            .port();
        let token = http::random_hex(24);
        let in_flight = Arc::new(AtomicU32::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (secret, count, stopping) = (token.clone(), Arc::clone(&in_flight), Arc::clone(&stop));
        thread::Builder::new()
            .name(format!("pom-plugin-{}-bridge", crate::PLUGIN_CODE))
            .spawn(move || {
                for stream in listener.incoming() {
                    if stopping.load(Ordering::SeqCst) {
                        return;
                    }
                    let Ok(stream) = stream else { continue };
                    let (target, secret, count) =
                        (Arc::clone(&target), secret.clone(), Arc::clone(&count));
                    let _ = thread::Builder::new()
                        .name(format!("pom-plugin-{}-request", crate::PLUGIN_CODE))
                        .spawn(move || serve(&stream, &target, &secret, &count));
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(Self {
            port,
            token,
            in_flight,
            stop,
        })
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wakes the accept loop so it sees the flag and releases the port.
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

struct InFlight(Arc<AtomicU32>);

impl InFlight {
    fn enter(count: &Arc<AtomicU32>) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(count))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn serve(stream: &TcpStream, target: &Target, secret: &str, count: &Arc<AtomicU32>) {
    let Ok(request) = http::read_request(stream) else {
        return;
    };
    let presented = request
        .header("authorization")
        .and_then(|value| value.strip_prefix("Bearer "))
        .unwrap_or_default();
    if !http::same_secret(presented, secret) {
        let _ = http::respond_json(stream, 401, &json!({"error": "unauthorized"}));
        return;
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/pom/generate") => {
            let _guard = InFlight::enter(count);
            generate(stream, target, &request);
        }
        ("GET", "/pom/status") => {
            let ready = target().is_some();
            let _ = http::respond_json(
                stream,
                200,
                &json!({"loaded": ready, "in_flight": count.load(Ordering::SeqCst)}),
            );
        }
        (method, path) if path.starts_with("/pom/session") => {
            session_route(stream, target, method, path, &request);
        }
        _ => {
            let _ = http::respond_json(stream, 404, &json!({"error": "not found"}));
        }
    }
}

/// The engine's request for one POM generation, and its route.
pub fn engine_request(body: &Value, model: &str) -> Result<(&'static str, Value), String> {
    let prompt = body["prompt"].as_str().ok_or("'prompt' must be a string")?;
    let chat = body["chat_template"].as_bool().unwrap_or(true);
    let probing = !probe_labels(body).is_empty();
    let images = body["images"].as_array().cloned().unwrap_or_default();
    if !images.is_empty() && !crate::launch::VISION {
        return Err("this engine does not read images".into());
    }
    let mut request = Map::new();
    request.insert("model".into(), json!(model));
    request.insert("stream".into(), json!(true));
    request.insert("stream_options".into(), json!({"include_usage": true}));
    if let Some(max) = body["max_tokens"].as_u64().filter(|max| *max > 0) {
        request.insert("max_tokens".into(), json!(max));
    }
    if let Some(stop) = body["stop"].as_array().filter(|stop| !stop.is_empty()) {
        request.insert("stop".into(), json!(stop));
    }
    for key in SAMPLING_KEYS {
        if body[*key].is_number() {
            request.insert((*key).into(), body[*key].clone());
        }
    }
    // A typed decision: the first token's most likely candidates, so the bridge reads each label's probability.
    if probing {
        request.insert("max_tokens".into(), json!(1));
        request.insert("temperature".into(), json!(0.0));
    }
    if !chat {
        if probing {
            request.insert("logprobs".into(), json!(PROBE_CANDIDATES));
        }
        request.insert("prompt".into(), json!(prompt));
        return Ok(("/v1/completions", Value::Object(request)));
    }
    let content = if images.is_empty() {
        json!(prompt)
    } else {
        let mut parts: Vec<Value> = images
            .iter()
            .map(|url| json!({"type": "image_url", "image_url": {"url": url}}))
            .collect();
        parts.push(json!({"type": "text", "text": prompt}));
        Value::Array(parts)
    };
    request.insert(
        "messages".into(),
        json!([{"role": "user", "content": content}]),
    );
    // The thinking switch most chat templates read (llama.cpp, vLLM, SGLang).
    let thinking = body["thinking"].as_bool().unwrap_or(true);
    request.insert(
        "chat_template_kwargs".into(),
        json!({"enable_thinking": thinking}),
    );
    if probing {
        request.insert("logprobs".into(), json!(true));
        request.insert("top_logprobs".into(), json!(PROBE_CANDIDATES));
    }
    Ok(("/v1/chat/completions", Value::Object(request)))
}

/// How many first-token candidates a decision reads (OpenAI's ceiling for `top_logprobs`).
const PROBE_CANDIDATES: u32 = 20;

/// The labels of a typed decision (`probe.labels`), empty when the request is not one.
pub fn probe_labels(body: &Value) -> Vec<String> {
    body["probe"]["labels"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

/// The first generated token's candidates as `(text, logprob)`: the chat form
/// (`logprobs.content[0].top_logprobs`) or the completion form (`logprobs.top_logprobs[0]`, a map).
pub fn first_token_candidates(choice: &Value) -> Option<Vec<(String, f64)>> {
    let logprobs = &choice["logprobs"];
    if let Some(top) = logprobs["content"][0]["top_logprobs"].as_array() {
        return Some(
            top.iter()
                .filter_map(|entry| {
                    Some((
                        entry["token"].as_str()?.to_owned(),
                        entry["logprob"].as_f64()?,
                    ))
                })
                .collect(),
        );
    }
    let top = &logprobs["top_logprobs"][0];
    if let Some(map) = top.as_object() {
        return Some(
            map.iter()
                .filter_map(|(token, value)| Some((token.clone(), value.as_f64()?)))
                .collect(),
        );
    }
    None
}

/// Each label's probability among the candidates: the mass of the tokens that spell it (leading space aside).
pub fn label_probabilities(labels: &[String], candidates: &[(String, f64)]) -> Vec<f64> {
    labels
        .iter()
        .map(|label| {
            candidates
                .iter()
                .filter(|(token, _)| token.trim() == label.trim())
                .map(|(_, logprob)| logprob.exp())
                .sum()
        })
        .collect()
}

/// A turn of a named conversation: its slot and its prompt's tokens.
#[derive(Debug)]
pub struct Turn {
    pub key: String,
    pub slot: u32,
    pub ids: Vec<i64>,
    /// A prefill turn not worth running (the decode machine holds most of the prompt).
    pub skipped: Option<String>,
}

/// One JSON call to the engine's own API.
fn engine_json(port: u16, method: &str, path: &str, body: &Value) -> Result<Value, String> {
    let payload = body.to_string();
    let response = http::request(
        method,
        port,
        path,
        &[("content-type", "application/json")],
        Some(payload.as_bytes()),
        ENGINE_SILENCE,
    )
    .map_err(|error| format!("{path}: {error}"))?;
    let status = response.status;
    let mut text = String::new();
    let mut reader = response.body;
    std::io::Read::read_to_string(&mut reader, &mut text).map_err(|error| error.to_string())?;
    if status != 200 {
        return Err(format!("{path} answered {status}: {}", text.trim()));
    }
    serde_json::from_str(&text).map_err(|error| format!("{path}: {error}"))
}

/// The tokens the engine reads for this request: the templated conversation, or the raw prompt.
fn prompt_ids(port: u16, engine_body: &Value) -> Result<Vec<i64>, String> {
    let text = if let Some(prompt) = engine_body["prompt"].as_str() {
        prompt.to_owned()
    } else {
        let mut template = json!({"messages": engine_body["messages"]});
        if let Some(kwargs) = engine_body.get("chat_template_kwargs") {
            template["chat_template_kwargs"] = kwargs.clone();
        }
        engine_json(port, "POST", "/apply-template", &template)?["prompt"]
            .as_str()
            .ok_or("/apply-template gave no prompt")?
            .to_owned()
    };
    let tokens = engine_json(
        port,
        "POST",
        "/tokenize",
        &json!({"content": text, "add_special": true, "parse_special": true}),
    )?;
    Ok(tokens["tokens"]
        .as_array()
        .ok_or("/tokenize gave no tokens")?
        .iter()
        .filter_map(Value::as_i64)
        .collect())
}

/// Before a turn of a named conversation: its slot, the saved session restored there when the slot does
/// not hold it and the session starts the prompt; a prefill turn's worth decided. None: no session.
fn prepare_turn(port: u16, body: &Value, engine_body: &mut Value) -> Result<Option<Turn>, String> {
    let Some(key) = body["session"]
        .as_str()
        .filter(|key| crate::sessions::valid_key(key))
    else {
        if body["prefill_only"] == true {
            return Err("a prefill turn needs a conversation (session)".into());
        }
        return Ok(None);
    };
    if !crate::sessions::enabled() {
        if body["prefill_only"] == true {
            return Err("a prefill turn needs the node's session folder".into());
        }
        return Ok(None);
    }
    let ids = prompt_ids(port, engine_body)?;
    if body["prefill_only"] == true {
        let holds = body["decode_holds"].as_u64().unwrap_or(0) as usize;
        let new = ids.len().saturating_sub(holds);
        let min = crate::sessions::prefill_min_tokens() as usize;
        if new < min.max(1) {
            return Ok(Some(Turn {
                key: key.to_owned(),
                slot: 0,
                ids,
                skipped: Some(format!("{new} new tokens, under {min}")),
            }));
        }
    }
    let (slot, holds) = crate::sessions::slot_for(key).ok_or("sessions are off")?;
    if !holds {
        if let Some(saved) = crate::sessions::saved_ids(key) {
            if crate::sessions::worth_restoring(&saved, &ids) {
                let restored = engine_json(
                    port,
                    "POST",
                    &format!("/slots/{slot}?action=restore"),
                    &json!({"filename": crate::sessions::file_name(key, "full")}),
                );
                if restored.is_err() {
                    // the slot's content is unknown now; the engine reads the prompt as it is
                    crate::sessions::release_slot(slot);
                }
            }
        }
    }
    engine_body["id_slot"] = json!(slot);
    engine_body["cache_prompt"] = json!(true);
    Ok(Some(Turn {
        key: key.to_owned(),
        slot,
        ids,
        skipped: None,
    }))
}

/// After a turn: the slot saved to the session folder with the prompt's tokens.
fn save_turn(port: u16, turn: &Turn) -> Result<Value, String> {
    engine_json(
        port,
        "POST",
        &format!("/slots/{}?action=save", turn.slot),
        &json!({"filename": crate::sessions::file_name(&turn.key, "full")}),
    )?;
    crate::sessions::write_ids(&turn.key, &turn.ids)?;
    Ok(json!({"ids": turn.ids.len(), "delta": false}))
}

/// `/pom/session*`: the files the POM moves between machines, and the snapshot actions.
fn session_route(stream: &TcpStream, target: &Target, method: &str, path: &str, request: &Request) {
    let reply = |status: u16, body: Value| {
        let _ = http::respond_json(stream, status, &body);
    };
    if !crate::sessions::enabled() {
        return reply(
            501,
            json!({"error": "this engine was loaded without a session folder"}),
        );
    }
    let key = request.query.get("key").cloned().unwrap_or_default();
    let kind = request.query.get("kind").cloned().unwrap_or_default();
    let number = |name: &str| {
        request
            .query
            .get(name)
            .and_then(|value| value.parse::<u64>().ok())
    };
    match (method, path) {
        ("GET", "/pom/session/offer") => match crate::sessions::offer(&key) {
            Some(offer) if crate::sessions::valid_key(&key) => reply(200, offer),
            _ => reply(400, json!({"error": "invalid session key"})),
        },
        ("GET", "/pom/session/pending") => reply(200, json!({"pending": false})),
        ("GET", "/pom/session/file") => {
            match crate::sessions::read_piece(
                &key,
                &kind,
                number("offset").unwrap_or(0),
                number("len").unwrap_or(0),
            ) {
                Ok(Some((bytes, size))) => {
                    let _ = http::respond_bytes(stream, &bytes, size);
                }
                Ok(None) => reply(404, json!({"error": "no such session file"})),
                Err(error) => reply(400, json!({"error": error})),
            }
        }
        ("PUT", "/pom/session/chunk") => {
            let (Some(offset), Some(total)) = (number("offset"), number("total")) else {
                return reply(400, json!({"error": "offset and total are required"}));
            };
            match crate::sessions::write_piece(&key, &kind, offset, total, &request.body) {
                Ok(whole) => reply(200, json!({"whole": whole})),
                Err(error) => reply(400, json!({"error": error})),
            }
        }
        ("POST", "/pom/session") => {
            let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
            let Some((port, _)) = target() else {
                return reply(503, json!({"error": "the engine is not ready"}));
            };
            match body["action"].as_str() {
                Some(action @ ("save" | "restore")) => {
                    let filename = body["filename"].as_str().unwrap_or_default();
                    if !crate::sessions::valid_key(filename) {
                        return reply(400, json!({"error": "a bare file name is required"}));
                    }
                    let started = Instant::now();
                    let slot = body["slot"].as_u64().unwrap_or(0);
                    match engine_json(
                        port,
                        "POST",
                        &format!("/slots/{slot}?action={action}"),
                        &json!({"filename": filename}),
                    ) {
                        Ok(answer) => reply(
                            200,
                            json!({
                                "tokens": answer["n_saved"].as_u64().or(answer["n_restored"].as_u64()).unwrap_or(0),
                                "bytes": answer["n_written"].as_u64().or(answer["n_read"].as_u64()).unwrap_or(0),
                                "ms": started.elapsed().as_millis() as u64,
                            }),
                        ),
                        Err(error) => reply(400, json!({"error": error})),
                    }
                }
                // deltas are never offered by this engine; the POM sends whole sessions
                Some(other) => reply(
                    400,
                    json!({"error": format!("this engine has no session action '{other}'")}),
                ),
                None => reply(400, json!({"error": "an action is required"})),
            }
        }
        _ => reply(404, json!({"error": "not found"})),
    }
}

/// What one streamed engine event adds to the answer.
#[derive(Debug, Default, PartialEq)]
pub struct Delta {
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub finish: Option<String>,
    pub usage: Option<Value>,
    pub timings: Option<Value>,
    /// The first token's candidates, when the request asked for them (a decision).
    pub candidates: Option<Vec<(String, f64)>>,
}

/// One `data:` payload of the engine's server-sent events.
pub fn parse_event(data: &str) -> Result<Delta, String> {
    let event: Value = serde_json::from_str(data).map_err(|error| error.to_string())?;
    if let Some(error) = event.get("error") {
        return Err(error["message"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| error.to_string()));
    }
    let choice = &event["choices"][0];
    let nonempty = |value: &Value| {
        value
            .as_str()
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    let delta = &choice["delta"];
    Ok(Delta {
        text: nonempty(&delta["content"]).or_else(|| nonempty(&choice["text"])),
        reasoning: nonempty(&delta["reasoning_content"]).or_else(|| nonempty(&delta["reasoning"])),
        finish: choice["finish_reason"].as_str().map(str::to_owned),
        usage: event
            .get("usage")
            .filter(|usage| usage.is_object())
            .cloned(),
        timings: event
            .get("timings")
            .filter(|timings| timings.is_object())
            .cloned(),
        candidates: first_token_candidates(choice),
    })
}

/// The POM's `done` line from what the stream said.
pub fn done_line(
    finish: Option<&str>,
    usage: &Value,
    timings: Option<Value>,
    first_token: Option<Duration>,
    total: Duration,
) -> Value {
    let count = |key: &str| usage[key].as_u64().unwrap_or(0);
    let prompt_tokens = count("prompt_tokens");
    let completion_tokens = count("completion_tokens");
    let reasoning_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap_or(0);
    let cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    // llama.cpp-style timings: the engine's own when it sends them, else the
    // bridge's measure (prompt = time to the first token).
    let timings = timings.unwrap_or_else(|| {
        let prompt_ms = first_token.unwrap_or(total).as_secs_f64() * 1000.0;
        json!({
            "prompt_n": prompt_tokens,
            "prompt_ms": prompt_ms,
            "predicted_n": completion_tokens,
            "predicted_ms": total.as_secs_f64() * 1000.0 - prompt_ms,
        })
    });
    json!({"done": {
        "finish": match finish {
            Some("length") => "length",
            _ => "stop",
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "timings": timings,
        "stop_sequence": Value::Null,
    }})
}

fn generate(stream: &TcpStream, target: &Target, request: &Request) {
    let fail = |message: String| {
        if let Ok(mut out) = NdjsonWriter::start(stream) {
            let _ = out.line(&json!({"error": message}));
            let _ = out.finish();
        }
    };
    let Some((port, model)) = target() else {
        return fail("the engine is not ready".into());
    };
    let body: Value = match serde_json::from_slice(&request.body) {
        Ok(body) => body,
        Err(error) => return fail(format!("the body is not JSON: {error}")),
    };
    let labels = probe_labels(&body);
    let prefill_only = body["prefill_only"] == true;
    let (path, engine_body) = match engine_request(&body, &model) {
        Ok(request) => request,
        Err(error) => return fail(error),
    };
    let started = Instant::now();
    let mut engine_body = engine_body;
    let turn = match prepare_turn(port, &body, &mut engine_body) {
        Ok(turn) => turn,
        Err(error) => return fail(error),
    };
    if let Some(Turn {
        skipped: Some(reason),
        ..
    }) = &turn
    {
        // the decode machine already holds most of the prompt: a prefill here is not worth it
        if let Ok(mut out) = NdjsonWriter::start(stream) {
            let _ = out.line(
                &json!({"done": {"finish": "stop", "prompt_tokens": 0, "completion_tokens": 0,
                "prefill": {"skipped": reason}}}),
            );
            let _ = out.finish();
        }
        return;
    }
    let payload = engine_body.to_string();
    let response = match http::request(
        "POST",
        port,
        path,
        &[
            ("content-type", "application/json"),
            ("accept", "text/event-stream"),
        ],
        Some(payload.as_bytes()),
        ENGINE_SILENCE,
    ) {
        Ok(response) => response,
        Err(error) => return fail(format!("the engine did not answer: {error}")),
    };
    let Ok(mut out) = NdjsonWriter::start(stream) else {
        return;
    };
    let mut body = response.body;
    if response.status != 200 {
        let mut text = String::new();
        let _ = std::io::Read::read_to_string(&mut body, &mut text);
        let _ = out.line(
            &json!({"error": format!("the engine answered {}: {}", response.status, text.trim())}),
        );
        let _ = out.finish();
        return;
    }
    let (mut finish, mut usage, mut timings, mut first) = (None, json!({}), None, None);
    let mut candidates: Option<Vec<(String, f64)>> = None;
    let mut line = String::new();
    loop {
        line.clear();
        match body.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                let _ = out.line(&json!({"error": format!("the engine stream broke: {error}")}));
                let _ = out.finish();
                return;
            }
        }
        let Some(data) = line.trim_end().strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let delta = match parse_event(data) {
            Ok(delta) => delta,
            Err(error) => {
                let _ = out.line(&json!({"error": error}));
                let _ = out.finish();
                return;
            }
        };
        if delta.text.is_some() || delta.reasoning.is_some() {
            first.get_or_insert_with(|| started.elapsed());
        }
        if prefill_only {
            // a prefill turn reads the prompt and keeps it; its one token is not the answer
            finish = delta.finish.or(finish);
            usage = delta.usage.unwrap_or(usage);
            timings = delta.timings.or(timings);
            continue;
        }
        let sent = delta
            .reasoning
            .map(|text| out.line(&json!({"r": text})))
            .unwrap_or(Ok(()))
            .and_then(|()| {
                delta
                    .text
                    .map(|text| out.line(&json!({"t": text})))
                    .unwrap_or(Ok(()))
            });
        if sent.is_err() {
            // The POM went away: dropping the engine's connection cancels it.
            return;
        }
        if candidates.is_none() {
            candidates = delta.candidates;
        }
        finish = delta.finish.or(finish);
        usage = delta.usage.unwrap_or(usage);
        timings = delta.timings.or(timings);
    }
    let mut done = done_line(finish.as_deref(), &usage, timings, first, started.elapsed());
    if !labels.is_empty() {
        let Some(candidates) = candidates else {
            let _ = out.line(
                &json!({"error": "the engine gave no first-token probabilities for the decision"}),
            );
            let _ = out.finish();
            return;
        };
        done["done"]["probe"] = json!({"probabilities": label_probabilities(&labels, &candidates)});
    }
    if let Some(turn) = &turn {
        match save_turn(port, turn) {
            Ok(saved) => {
                done["done"]["session"] = json!({"key": turn.key, "saved": saved});
                if prefill_only {
                    done["done"]["prefill"] = json!({"tokens": turn.ids.len()});
                }
            }
            Err(error) => {
                let _ = out.line(&json!({"error": format!("the session was not saved: {error}")}));
                let _ = out.finish();
                return;
            }
        }
    }
    let _ = out.line(&done);
    let _ = out.finish();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn a_chat_turn_becomes_one_user_message_with_the_thinking_switch() {
        let (path, request) = engine_request(
            &json!({"prompt": "hi", "chat_template": true, "thinking": false, "max_tokens": 9,
                    "stop": ["</s>"], "temperature": 0.2, "top_k": 40, "unknown": 1}),
            "m",
        )
        .unwrap();
        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(
            request["messages"],
            json!([{"role": "user", "content": "hi"}])
        );
        assert_eq!(request["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(request["max_tokens"], 9);
        assert_eq!(request["stop"], json!(["</s>"]));
        assert_eq!(request["temperature"], 0.2);
        assert_eq!(request["top_k"], 40);
        assert!(request.get("unknown").is_none());
    }

    #[test]
    fn a_raw_completion_keeps_the_prompt_as_it_is() {
        let (path, request) =
            engine_request(&json!({"prompt": "<s>x", "chat_template": false}), "m").unwrap();
        assert_eq!(path, "/v1/completions");
        assert_eq!(request["prompt"], "<s>x");
        assert!(request.get("messages").is_none());
    }

    #[test]
    fn what_the_engine_cannot_do_is_refused_by_name() {
        let (path, probe) = engine_request(
            &json!({"prompt": "x", "probe": {"labels": ["Yes", "No"]}}),
            "m",
        )
        .unwrap();
        assert_eq!(path, "/v1/chat/completions");
        assert_eq!(probe["max_tokens"], 1);
        assert_eq!(probe["top_logprobs"], 20);
        let chat = json!({"logprobs": {"content": [{"top_logprobs": [
            {"token": "Yes", "logprob": -0.1}, {"token": " Yes", "logprob": -3.0}, {"token": "No", "logprob": -2.5}]}]}});
        let found = first_token_candidates(&chat).unwrap();
        let probabilities =
            label_probabilities(&["Yes".into(), "No".into(), "Maybe".into()], &found);
        assert!((probabilities[0] - ((-0.1f64).exp() + (-3.0f64).exp())).abs() < 1e-9);
        assert!((probabilities[1] - (-2.5f64).exp()).abs() < 1e-9);
        assert_eq!(probabilities[2], 0.0);
        let legacy = json!({"logprobs": {"top_logprobs": [{"A": -0.2}]}});
        assert_eq!(first_token_candidates(&legacy).unwrap()[0].0, "A");
        // a prefill turn without a conversation is refused before the engine is asked
        let mut engine_body = json!({});
        let refused = prepare_turn(
            1,
            &json!({"prompt": "x", "prefill_only": true}),
            &mut engine_body,
        );
        assert!(refused.unwrap_err().contains("session"));
        assert!(engine_request(&json!({}), "m").is_err());
    }

    #[test]
    fn reasoning_and_answer_stay_apart() {
        let delta = parse_event(r#"{"choices":[{"delta":{"reasoning_content":"why"}}]}"#).unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("why"));
        assert_eq!(delta.text, None);
        let end = parse_event(
            r#"{"choices":[{"delta":{"content":"ok"},"finish_reason":"length"}],"usage":{"prompt_tokens":3}}"#,
        )
        .unwrap();
        assert_eq!(end.text.as_deref(), Some("ok"));
        assert_eq!(end.finish.as_deref(), Some("length"));
        assert!(parse_event(r#"{"error":{"message":"boom"}}"#)
            .unwrap_err()
            .contains("boom"));
    }

    #[test]
    fn the_done_line_carries_counts_and_timings() {
        let usage = json!({"prompt_tokens": 10, "completion_tokens": 4,
            "completion_tokens_details": {"reasoning_tokens": 2},
            "prompt_tokens_details": {"cached_tokens": 6}});
        let done = done_line(
            Some("stop"),
            &usage,
            None,
            Some(Duration::from_millis(100)),
            Duration::from_millis(300),
        );
        assert_eq!(done["done"]["finish"], "stop");
        assert_eq!(done["done"]["reasoning_tokens"], 2);
        assert_eq!(done["done"]["cached_tokens"], 6);
        assert_eq!(done["done"]["timings"]["prompt_n"], 10);
        assert!((done["done"]["timings"]["prompt_ms"].as_f64().unwrap() - 100.0).abs() < 1.0);
        let own = done_line(
            None,
            &usage,
            Some(json!({"draft_n": 5})),
            None,
            Duration::ZERO,
        );
        assert_eq!(own["done"]["timings"]["draft_n"], 5);
    }

    /// An engine server that streams a fixed answer, chunked.
    fn fake_engine() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let request = http::read_request(&stream).unwrap();
                assert_eq!(request.path, "/v1/chat/completions");
                let events = [
                    r#"{"choices":[{"delta":{"reasoning_content":"think"}}]}"#,
                    r#"{"choices":[{"delta":{"content":"Hello"}}]}"#,
                    r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
                ];
                let mut stream = &stream;
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
                    .unwrap();
                for data in events
                    .iter()
                    .map(|event| format!("data: {event}\n\n"))
                    .chain(["data: [DONE]\n\n".to_owned()])
                {
                    write!(stream, "{:x}\r\n{data}\r\n", data.len()).unwrap();
                }
                stream.write_all(b"0\r\n\r\n").unwrap();
            }
        });
        port
    }

    fn post(port: u16, token: &str, body: &Value) -> (u16, Vec<Value>) {
        let payload = body.to_string();
        let auth = format!("Bearer {token}");
        let mut response = http::request(
            "POST",
            port,
            "/pom/generate",
            &[("authorization", &auth)],
            Some(payload.as_bytes()),
            Duration::from_secs(10),
        )
        .unwrap();
        let mut text = String::new();
        response.body.read_to_string(&mut text).unwrap();
        let lines = text
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        (response.status, lines)
    }

    #[test]
    fn the_route_streams_the_engine_answer_as_pom_lines() {
        let engine = fake_engine();
        let bridge = Bridge::start(Arc::new(move || Some((engine, "m".to_owned())))).unwrap();
        let (status, lines) = post(
            bridge.port,
            &bridge.token,
            &json!({"prompt": "hi", "chat_template": true}),
        );
        assert_eq!(status, 200);
        assert_eq!(lines[0], json!({"r": "think"}));
        assert_eq!(lines[1], json!({"t": "Hello"}));
        assert_eq!(lines[2]["done"]["completion_tokens"], 2);
        assert_eq!(lines.len(), 3);
        let (status, _) = post(bridge.port, "wrong", &json!({"prompt": "hi"}));
        assert_eq!(status, 401);
    }

    #[test]
    fn a_bridge_without_an_engine_answers_an_error_line() {
        let bridge = Bridge::start(Arc::new(|| None)).unwrap();
        let (status, lines) = post(bridge.port, &bridge.token, &json!({"prompt": "hi"}));
        assert_eq!(status, 200);
        assert!(lines[0]["error"].as_str().unwrap().contains("not ready"));
    }
}
