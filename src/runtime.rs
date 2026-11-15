//! The runtime a release pins: the engine is hundreds of megabytes, so the
//! plugin library never embeds it. `scripts/package.sh` builds the library
//! with the URL, SHA-256 and size of each runtime archive the release attaches
//! (`AVALON_RUNTIME_*` for NVIDIA, `AVALON_RUNTIME_HIP_*` for AMD); the first
//! load downloads the one its cards need into the plugin directory, checks it
//! and unpacks it. A development build pins none and reads `build/runtime*`.
//!
//! The node image has `curl`, `sha256sum` and `tar`; the plugin runs them as
//! children (stdin null, output captured), never a crate of its own.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

struct Pin {
    url: &'static str,
    sha256: &'static str,
    size: &'static str,
}

fn pin(hip: bool) -> Option<Pin> {
    let pin = if hip {
        Pin {
            url: option_env!("AVALON_RUNTIME_HIP_URL").unwrap_or(""),
            sha256: option_env!("AVALON_RUNTIME_HIP_SHA256").unwrap_or(""),
            size: option_env!("AVALON_RUNTIME_HIP_SIZE").unwrap_or(""),
        }
    } else {
        Pin {
            url: option_env!("AVALON_RUNTIME_URL").unwrap_or(""),
            sha256: option_env!("AVALON_RUNTIME_SHA256").unwrap_or(""),
            size: option_env!("AVALON_RUNTIME_SIZE").unwrap_or(""),
        }
    };
    (!pin.url.is_empty() && pin.sha256.len() == 64).then_some(pin)
}

/// Whether this build pins a runtime for these cards.
pub fn pinned(hip: bool) -> bool {
    pin(hip).is_some()
}

/// Where the pinned runtime is unpacked: one directory per archive.
pub fn installed_dir(base: &Path, hip: bool) -> Option<PathBuf> {
    let pin = pin(hip)?;
    let name = if hip { "runtime-hip" } else { "runtime" };
    Some(base.join("runtimes").join(&pin.sha256[..16]).join(name))
}

fn run(command: &mut Command) -> Result<String, String> {
    let output = command
        .stdin(Stdio::null())
        .output()
        .map_err(|error| format!("{command:?}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{command:?} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Downloads, verifies and unpacks the pinned runtime; the directory it lives in.
pub fn install(base: &Path, hip: bool) -> Result<PathBuf, String> {
    let pin = pin(hip).ok_or("this build pins no runtime")?;
    let target = installed_dir(base, hip).ok_or("this build pins no runtime")?;
    if crate::launch::is_runtime(&target) {
        return Ok(target);
    }
    let holder = target.parent().ok_or("bad runtime directory")?;
    std::fs::create_dir_all(holder).map_err(|error| error.to_string())?;
    let archive = holder.join("runtime.tar.gz.part");
    run(Command::new("curl")
        .args(["-fsSL", "--retry", "5", "--retry-delay", "3", "-o"])
        .arg(&archive)
        .arg(pin.url))?;
    let size = std::fs::metadata(&archive)
        .map_err(|error| error.to_string())?
        .len();
    if !pin.size.is_empty() && pin.size != size.to_string() {
        let _ = std::fs::remove_file(&archive);
        return Err(format!(
            "the runtime archive has {size} bytes, the release pins {}",
            pin.size
        ));
    }
    let sum = run(Command::new("sha256sum").arg(&archive))?;
    if sum.split_whitespace().next() != Some(pin.sha256) {
        let _ = std::fs::remove_file(&archive);
        return Err("the runtime archive does not match the release's SHA-256".into());
    }
    let unpack = holder.join("unpack.part");
    let _ = std::fs::remove_dir_all(&unpack);
    std::fs::create_dir_all(&unpack).map_err(|error| error.to_string())?;
    run(Command::new("tar")
        .arg("-xzf")
        .arg(&archive)
        .arg("-C")
        .arg(&unpack))?;
    let name = target.file_name().ok_or("bad runtime directory")?;
    let unpacked = unpack.join(name);
    if !crate::launch::is_runtime(&unpacked) {
        return Err(format!(
            "the runtime archive holds no {}/bin/server",
            name.to_string_lossy()
        ));
    }
    let _ = std::fs::remove_dir_all(&target);
    std::fs::rename(&unpacked, &target).map_err(|error| error.to_string())?;
    let _ = std::fs::remove_dir_all(&unpack);
    let _ = std::fs::remove_file(&archive);
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_development_build_pins_nothing() {
        if option_env!("AVALON_RUNTIME_URL").is_none() {
            assert!(!pinned(false));
            assert!(installed_dir(Path::new("/p"), false).is_none());
            assert!(install(Path::new("/p"), false).is_err());
        }
    }
}
