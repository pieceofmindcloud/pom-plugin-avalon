//! Conversation sessions (feature level L4): a conversation's KV state saved
//! in the node's session folder, so another machine that serves the whole
//! model continues it without reading the prompt again.
//!
//! The engine is llama-server, which saves and restores one slot to a file
//! (`--slot-save-path`, `POST /slots/<id>?action=save|restore`). The bridge
//! gives each conversation a slot (`id_slot` in the request), and:
//!
//! - before a turn whose slot does not hold the conversation, it restores the
//!   conversation's saved file when its tokens start the prompt;
//! - after the turn, it saves the slot (`<key>.full`) and the prompt's tokens
//!   (`<key>.full_ids`, JSON), and the `done` line says
//!   `session: {key, saved: {ids, delta: false}}`;
//! - the POM moves those two files between machines with `/pom/session/offer`,
//!   `/file` and `/chunk` (no delta: the POM then sends the whole session).
//!
//! A session restores only under the same model file and KV types (`config`).

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Instant;

/// The two files of a session the POM moves (`delta` and `delta_meta` are never written here).
pub const KINDS: [&str; 2] = ["full", "full_ids"];
/// The largest file a peer may send: a whole session of a long conversation.
pub const MAX_FILE: u64 = 1 << 30;
/// A restore is worth it only when the saved start covers this much of the prompt.
const MIN_RESTORE_TOKENS: usize = 64;

pub struct Config {
    pub dir: PathBuf,
    pub slots: u32,
    /// The settings a session is bound to: model file and KV types.
    pub fingerprint: String,
}

#[derive(Default)]
struct State {
    config: Option<Config>,
    /// Bumped by every `configure`, so an unload clears only the configuration it made.
    generation: u64,
    /// `prefill` or `decode` in a prefill/decode split, and the split's threshold.
    role: Option<String>,
    prefill_min_tokens: Option<u32>,
    /// Which conversation each slot holds, and when it was last used.
    slots: HashMap<u32, (String, Instant)>,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::default()))
}

/// Sessions on (the engine was loaded with a session folder) or off; returns this configuration's generation.
pub fn configure(config: Option<Config>) -> u64 {
    let mut state = state().lock().unwrap_or_else(PoisonError::into_inner);
    state.slots.clear();
    state.generation += 1;
    if let Some(config) = &config {
        let _ = fs::create_dir_all(&config.dir);
    }
    state.config = config;
    state.generation
}

/// Sessions off, unless a later `configure` replaced the one of `generation`.
pub fn clear(generation: u64) {
    let mut state = state().lock().unwrap_or_else(PoisonError::into_inner);
    if state.generation == generation {
        state.config = None;
        state.slots.clear();
        state.role = None;
    }
}

pub fn set_role(role: Option<String>, prefill_min_tokens: Option<u32>) {
    let mut state = state().lock().unwrap_or_else(PoisonError::into_inner);
    state.role = role;
    state.prefill_min_tokens = prefill_min_tokens;
}

/// The prefill split's threshold: a new part shorter than this is not worth a prefill machine.
pub fn prefill_min_tokens() -> u32 {
    state()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .prefill_min_tokens
        .unwrap_or(0)
}

pub fn enabled() -> bool {
    state()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .config
        .is_some()
}

/// The same rule as the POM: letters, digits, '-', '_' and '.', not starting with '.', at most 96 bytes.
pub fn valid_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 96
        && !key.starts_with('.')
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

/// A session file's name in the folder (llama-server takes a bare file name).
pub fn file_name(key: &str, kind: &str) -> String {
    format!("{key}.{kind}")
}

fn path(key: &str, kind: &str) -> Option<PathBuf> {
    let state = state().lock().unwrap_or_else(PoisonError::into_inner);
    let config = state.config.as_ref()?;
    (valid_key(key) && KINDS.contains(&kind)).then(|| config.dir.join(file_name(key, kind)))
}

/// The slot a conversation runs in, and whether that slot already holds it: its own slot, else the
/// least recently used one (which then belongs to this conversation).
pub fn slot_for(key: &str) -> Option<(u32, bool)> {
    let mut state = state().lock().unwrap_or_else(PoisonError::into_inner);
    let slots = state.config.as_ref()?.slots.max(1);
    let now = Instant::now();
    if let Some((&slot, _)) = state.slots.iter().find(|(_, (owner, _))| owner == key) {
        state.slots.insert(slot, (key.to_owned(), now));
        return Some((slot, true));
    }
    let slot = (0..slots)
        .min_by_key(|slot| state.slots.get(slot).map(|(_, used)| *used))
        .unwrap_or(0);
    state.slots.insert(slot, (key.to_owned(), now));
    Some((slot, false))
}

/// Forgets which conversation a slot holds (a failed restore leaves it unknown).
pub fn release_slot(slot: u32) {
    state()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .slots
        .remove(&slot);
}

/// The saved prompt tokens of a conversation.
pub fn saved_ids(key: &str) -> Option<Vec<i64>> {
    let text = fs::read(path(key, "full_ids")?).ok()?;
    serde_json::from_slice(&text).ok()
}

pub fn write_ids(key: &str, ids: &[i64]) -> Result<(), String> {
    let target = path(key, "full_ids").ok_or("sessions are off")?;
    publish(
        &target,
        &serde_json::to_vec(ids).map_err(|e| e.to_string())?,
    )
}

/// How many leading tokens two sequences share.
pub fn common_prefix(a: &[i64], b: &[i64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Whether a saved session is worth restoring for this prompt: its tokens start the prompt, enough of them.
pub fn worth_restoring(saved: &[i64], prompt: &[i64]) -> bool {
    let shared = common_prefix(saved, prompt);
    shared >= MIN_RESTORE_TOKENS && shared * 2 >= saved.len().min(prompt.len())
}

/// `GET /pom/session/offer?key=`: what this machine can give a peer.
pub fn offer(key: &str) -> Option<Value> {
    let fingerprint = state()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .config
        .as_ref()?
        .fingerprint
        .clone();
    let bytes = path(key, "full")
        .and_then(|full| fs::metadata(full).ok())
        .map(|meta| meta.len())
        .unwrap_or(0);
    let tokens = if bytes > 0 {
        saved_ids(key).map(|ids| ids.len()).unwrap_or(0)
    } else {
        0
    };
    Some(json!({
        "tokens": tokens,
        "bytes": bytes,
        "ram_tokens": 0,
        "layers": "whole",
        "config": fingerprint,
        "pending": false,
    }))
}

/// One piece of a session file (`len` 0: to the end), with the file's whole size.
pub fn read_piece(
    key: &str,
    kind: &str,
    offset: u64,
    len: u64,
) -> Result<Option<(Vec<u8>, u64)>, String> {
    let target = path(key, kind).ok_or("invalid session file")?;
    let mut file = match fs::File::open(&target) {
        Ok(file) => file,
        Err(_) => return Ok(None),
    };
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    if offset > size {
        return Err("the offset is past the end of the file".into());
    }
    let want = if len == 0 {
        size - offset
    } else {
        len.min(size - offset)
    };
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    let mut bytes = vec![0; want as usize];
    file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(Some((bytes, size)))
}

/// `PUT /pom/session/chunk`: one piece of a file a peer sends, written beside the file and published
/// (renamed into place) once the last piece is in. Returns whether the file is now whole.
pub fn write_piece(
    key: &str,
    kind: &str,
    offset: u64,
    total: u64,
    bytes: &[u8],
) -> Result<bool, String> {
    if total > MAX_FILE {
        return Err("the file is larger than a session may be".into());
    }
    if offset + bytes.len() as u64 > total {
        return Err("the piece does not fit the file".into());
    }
    let target = path(key, kind).ok_or("invalid session file")?;
    let partial = target.with_extension(format!("{kind}.part"));
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(offset == 0)
        .open(&partial)
        .map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())?;
    if offset + (bytes.len() as u64) < total {
        return Ok(false);
    }
    drop(file);
    fs::rename(&partial, &target).map_err(|e| e.to_string())?;
    // The slot that held this conversation no longer matches the file: the next turn restores it.
    let mut state = state().lock().unwrap_or_else(PoisonError::into_inner);
    state.slots.retain(|_, (owner, _)| owner != key);
    Ok(true)
}

fn publish(target: &Path, bytes: &[u8]) -> Result<(), String> {
    let partial = target.with_extension("tmp");
    fs::write(&partial, bytes).map_err(|e| e.to_string())?;
    fs::rename(&partial, target).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(dir: &Path, slots: u32) {
        configure(Some(Config {
            dir: dir.to_path_buf(),
            slots,
            fingerprint: "m.gguf|q8_0|q8_0".into(),
        }));
    }

    #[test]
    fn sessions_move_in_pieces_and_slots_follow_their_conversation() {
        let dir = std::env::temp_dir().join(format!("avalon-sessions-{}", std::process::id()));
        on(&dir, 2);
        assert!(!valid_key(".x") && !valid_key("a/b") && valid_key("conv-1"));

        // slots: a conversation keeps its slot; a third takes the least recently used one
        assert_eq!(slot_for("a"), Some((0, false)));
        assert_eq!(slot_for("b"), Some((1, false)));
        assert_eq!(slot_for("a"), Some((0, true)));
        assert_eq!(slot_for("c"), Some((1, false)));

        // a file arrives in two pieces and is published whole; its conversation's slot is forgotten
        assert!(!write_piece("c", "full", 0, 6, b"abc").unwrap());
        assert!(read_piece("c", "full", 0, 0).unwrap().is_none());
        assert!(write_piece("c", "full", 3, 6, b"def").unwrap());
        assert_eq!(
            read_piece("c", "full", 2, 3).unwrap().unwrap(),
            (b"cde".to_vec(), 6)
        );
        // the slot it held is free again, so it is the least recently used
        assert_eq!(slot_for("c"), Some((1, false)));
        assert!(write_piece("c", "full", 0, 2, b"abc").is_err());
        assert!(write_piece("c", "delta", 0, 1, b"x").is_err());

        let ids: Vec<i64> = (0..100).collect();
        write_ids("c", &ids).unwrap();
        let offered = offer("c").unwrap();
        assert_eq!(offered["tokens"], 100);
        assert_eq!(offered["bytes"], 6);
        assert_eq!(offered["layers"], "whole");
        assert_eq!(offered["config"], "m.gguf|q8_0|q8_0");

        let mut longer = ids.clone();
        longer.extend(500..600);
        assert!(worth_restoring(&ids, &longer));
        assert!(
            !worth_restoring(&ids[..10], &longer),
            "too short to be worth it"
        );
        assert!(!worth_restoring(&[9, 9, 9], &longer));

        configure(None);
        assert!(!enabled() && offer("c").is_none());
        fs::remove_dir_all(dir).unwrap();
    }
}
