//! The little of a GGUF header the plugin needs: the POM slice keys that say
//! which layers a file holds. Only the header is read, never tensor data.
//! (From pom-plugin-strata's reader.)

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

const MAGIC: &[u8; 4] = b"GGUF";
/// A header with more entries than this is not one this plugin should trust.
const MAX_ENTRIES: u64 = 1 << 20;
const MAX_STRING: u64 = 64 << 20;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Header {
    pub architecture: Option<String>,
    pub block_count: Option<u64>,
    pub split_no: Option<u64>,
    pub split_count: Option<u64>,
    /// Set on a POM slice: the first layer it holds of the original model.
    pub pom_start_layer: Option<u64>,
    /// Set on a POM slice: the last layer it holds of the original model.
    pub pom_end_layer: Option<u64>,
    /// Set on a POM slice: the original model's layer count.
    pub pom_original_block_count: Option<u64>,
    /// Set on a header-only virtual slice: the first file of the whole model its tensors are read from.
    pub external_first: Option<String>,
    pub tensors: Vec<String>,
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn invalid(message: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.to_owned())
}

fn read_string(reader: &mut impl Read) -> std::io::Result<String> {
    let length = read_u64(reader)?;
    if length > MAX_STRING {
        return Err(invalid("string too long"));
    }
    let mut bytes = vec![0; length as usize];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// A scalar value, when it is an integer; strings come back separately.
enum Value {
    Integer(u64),
    Text(String),
    Other,
}

fn skip(reader: &mut impl Read, bytes: u64) -> std::io::Result<()> {
    let copied = std::io::copy(&mut reader.take(bytes), &mut std::io::sink())?;
    if copied == bytes {
        Ok(())
    } else {
        Err(invalid("truncated header"))
    }
}

fn read_value(reader: &mut impl Read, kind: u32) -> std::io::Result<Value> {
    Ok(match kind {
        0 | 7 => {
            let mut byte = [0; 1];
            reader.read_exact(&mut byte)?;
            Value::Integer(u64::from(byte[0]))
        }
        1 => {
            let mut byte = [0; 1];
            reader.read_exact(&mut byte)?;
            Value::Integer(byte[0] as i8 as i64 as u64)
        }
        2 | 3 => {
            let mut bytes = [0; 2];
            reader.read_exact(&mut bytes)?;
            Value::Integer(u64::from(u16::from_le_bytes(bytes)))
        }
        4 | 5 => Value::Integer(u64::from(read_u32(reader)?)),
        6 => {
            skip(reader, 4)?;
            Value::Other
        }
        8 => Value::Text(read_string(reader)?),
        9 => {
            let item = read_u32(reader)?;
            let count = read_u64(reader)?;
            if count > MAX_STRING {
                return Err(invalid("array too long"));
            }
            match item {
                0 | 1 | 7 => skip(reader, count)?,
                2 | 3 => skip(reader, count * 2)?,
                4..=6 => skip(reader, count * 4)?,
                10..=12 => skip(reader, count * 8)?,
                8 => {
                    for _ in 0..count {
                        let length = read_u64(reader)?;
                        if length > MAX_STRING {
                            return Err(invalid("string too long"));
                        }
                        skip(reader, length)?;
                    }
                }
                _ => return Err(invalid("unknown array type")),
            }
            Value::Other
        }
        10 | 11 => Value::Integer(read_u64(reader)?),
        12 => {
            skip(reader, 8)?;
            Value::Other
        }
        _ => return Err(invalid("unknown value type")),
    })
}

pub fn read_header(path: &Path) -> std::io::Result<Header> {
    let mut reader = BufReader::with_capacity(1 << 16, File::open(path)?);
    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("not a GGUF file"));
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return Err(invalid("unsupported GGUF version"));
    }
    let tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;
    if tensor_count > MAX_ENTRIES || kv_count > MAX_ENTRIES {
        return Err(invalid("implausible header"));
    }
    let mut header = Header::default();
    let mut block_counts = Vec::new();
    for _ in 0..kv_count {
        let key = read_string(&mut reader)?;
        let kind = read_u32(&mut reader)?;
        if key == "general.external_data.files" && kind == 9 {
            // an array of strings: the whole model's files
            let item = read_u32(&mut reader)?;
            let count = read_u64(&mut reader)?;
            if item != 8 || count > MAX_ENTRIES {
                return Err(invalid("malformed external data files"));
            }
            for index in 0..count {
                let name = read_string(&mut reader)?;
                if index == 0 {
                    header.external_first = Some(name);
                }
            }
            continue;
        }
        let value = read_value(&mut reader, kind)?;
        match (key.as_str(), value) {
            ("general.architecture", Value::Text(text)) => header.architecture = Some(text),
            ("split.no", Value::Integer(number)) => header.split_no = Some(number),
            ("split.count", Value::Integer(number)) => header.split_count = Some(number),
            ("pom.start_layer", Value::Integer(number)) => header.pom_start_layer = Some(number),
            ("pom.end_layer", Value::Integer(number)) => header.pom_end_layer = Some(number),
            ("pom.original_block_count", Value::Integer(number)) => {
                header.pom_original_block_count = Some(number)
            }
            (key, Value::Integer(number)) if key.ends_with(".block_count") => {
                block_counts.push((key.trim_end_matches(".block_count").to_owned(), number))
            }
            _ => {}
        }
    }
    header.block_count = block_counts
        .iter()
        .find(|(arch, _)| Some(arch.as_str()) == header.architecture.as_deref())
        .map(|(_, count)| *count);
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let dims = read_u32(&mut reader)?;
        if dims > 8 {
            return Err(invalid("implausible tensor rank"));
        }
        skip(&mut reader, u64::from(dims) * 8 + 4 + 8)?;
        header.tensors.push(name);
    }
    Ok(header)
}

/// The layers a POM slice holds of the whole model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stage {
    pub start: u64,
    /// The last layer it holds (inclusive).
    pub end: u64,
    pub total: u64,
}

impl Stage {
    pub fn last(&self) -> bool {
        self.end + 1 >= self.total
    }
}

/// Where a virtual slice reads its tensors: the whole model's first file.
pub fn external_source(path: &Path) -> Option<std::path::PathBuf> {
    let first = read_header(path).ok()?.external_first?;
    let first = std::path::PathBuf::from(first);
    Some(if first.is_absolute() {
        first
    } else {
        path.parent()?.join(first)
    })
}

/// The stage a model file is, when it is a POM slice.
pub fn stage_of(path: &Path) -> Option<Stage> {
    let header = read_header(path).ok()?;
    Some(Stage {
        start: header.pom_start_layer?,
        end: header.pom_end_layer?,
        total: header.pom_original_block_count?,
    })
}
