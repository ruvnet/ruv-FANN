//! Offline checkpoint editing. A structurally valid candidate is NOT a validated model.
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const HEADER_LIMIT: u64 = 16 * 1024 * 1024;
const BUFFER: usize = 1024 * 1024;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub tensor: String,
    pub index: u64,
    pub value: f32,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Patch {
    pub schema: String,
    pub base_sha256: String,
    pub edits: Vec<Edit>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Undo {
    pub offset: u64,
    pub before: Vec<u8>,
    pub after: Vec<u8>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema: String,
    pub base_sha256: String,
    pub candidate_sha256: String,
    pub edits: Vec<Undo>,
    pub behavioral_validation: String,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}
#[derive(Debug, Serialize)]
pub struct Inspection {
    pub data_start: u64,
    pub bytes: u64,
    pub tensors: BTreeMap<String, Tensor>,
}

fn fail<T>(s: &str) -> Result<T> {
    Err(s.into())
}
fn width(s: &str) -> Result<u64> {
    match s {
        "BOOL" | "U8" | "I8" => Ok(1),
        "F16" | "BF16" | "U16" | "I16" => Ok(2),
        "F32" | "U32" | "I32" => Ok(4),
        "F64" | "U64" | "I64" => Ok(8),
        _ => fail("unsupported dtype"),
    }
}
pub fn inspect(f: &mut File) -> Result<Inspection> {
    f.rewind()?;
    let bytes = f.metadata()?.len();
    let mut len = [0; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    if n == 0
        || n > HEADER_LIMIT
        || n % 8 != 0
        || n.checked_add(8).filter(|v| *v <= bytes).is_none()
    {
        return fail("invalid header length");
    }
    let mut raw = vec![0; n as usize];
    f.read_exact(&mut raw)?;
    // Streaming validation avoids mapping an entire multi-GB shard into memory.
    let object: UniqueHeader = serde_json::from_slice(&raw)?;
    let mut tensors = BTreeMap::new();
    for (k, v) in object.0 {
        if k == "__metadata__" {
            let _: BTreeMap<String, String> = serde_json::from_value(v)?;
            continue;
        }
        tensors.insert(k, serde_json::from_value::<Tensor>(v)?);
    }
    let mut ordered: Vec<_> = tensors.values().collect();
    ordered.sort_by_key(|t| (t.data_offsets[0], t.data_offsets[1]));
    let mut cursor = 0;
    for t in ordered {
        if t.shape.len() > 32 {
            return fail("tensor rank exceeds 32");
        }
        let count = t
            .shape
            .iter()
            .try_fold(1u64, |a, b| a.checked_mul(*b))
            .ok_or("shape overflow")?;
        let size = count
            .checked_mul(width(&t.dtype)?)
            .ok_or("tensor size overflow")?;
        if t.data_offsets[0] != cursor || t.data_offsets[1].checked_sub(cursor) != Some(size) {
            return fail("invalid tensor offsets or shape");
        }
        cursor = t.data_offsets[1];
    }
    if cursor != bytes - n - 8 {
        return fail("truncated or trailing tensor data");
    }
    Ok(Inspection {
        data_start: n + 8,
        bytes,
        tensors,
    })
}
struct UniqueHeader(BTreeMap<String, serde_json::Value>);
impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = UniqueHeader;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("unique tensor names")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut m: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut out = BTreeMap::new();
                while let Some((k, v)) = m.next_entry::<String, serde_json::Value>()? {
                    if out.insert(k, v).is_some() {
                        return Err(serde::de::Error::custom("duplicate tensor name"));
                    }
                }
                Ok(UniqueHeader(out))
            }
        }
        d.deserialize_map(V)
    }
}
fn hash(f: &mut File) -> Result<String> {
    f.rewind()?;
    let mut h = Sha256::new();
    let mut buf = vec![0; BUFFER];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn fingerprint(path: &Path) -> Result<String> {
    hash(&mut File::open(path)?)
}
fn decode(dtype: &str, b: &[u8]) -> Result<f32> {
    Ok(match dtype {
        "F32" => f32::from_le_bytes(b.try_into()?),
        "F16" => f16::from_le_bytes(b.try_into()?).to_f32(),
        "BF16" => bf16::from_le_bytes(b.try_into()?).to_f32(),
        _ => return fail("editable dtypes: F32 F16 BF16"),
    })
}
fn encode(dtype: &str, v: f32) -> Result<Vec<u8>> {
    let b = match dtype {
        "F32" => v.to_le_bytes().to_vec(),
        "F16" => f16::from_f32(v).to_le_bytes().to_vec(),
        "BF16" => bf16::from_f32(v).to_le_bytes().to_vec(),
        _ => return fail("unsupported edit dtype"),
    };
    if !decode(dtype, &b)?.is_finite() {
        return fail("nonfinite encoded value");
    }
    Ok(b)
}
fn copy_snapshot(source: &Path, dest: &Path) -> Result<(tempfile::NamedTempFile, String)> {
    let parent = dest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if dest.exists() {
        return fail("destination already exists");
    }
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    let mut f = File::open(source)?;
    let mut h = Sha256::new();
    let mut buf = vec![0; BUFFER];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        temp.write_all(&buf[..n])?;
    }
    Ok((temp, format!("{:x}", h.finalize())))
}
fn commit(mut temp: tempfile::NamedTempFile, dest: &Path) -> Result<()> {
    temp.flush()?;
    temp.as_file().sync_all()?;
    temp.persist_noclobber(dest).map_err(|e| e.error)?;
    #[cfg(unix)]
    File::open(
        dest.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?
    .sync_all()?;
    Ok(())
}
pub fn apply(source: &Path, dest: &Path, p: &Patch) -> Result<Receipt> {
    if p.schema != "ruv-fann.tensor-patch/1" || p.edits.is_empty() || p.edits.len() > 128 {
        return fail("invalid patch schema or edit count");
    }
    // Work only on a private copy. Hash and validate the same bytes that we edit.
    let (mut temp, base) = copy_snapshot(source, dest)?;
    let f = temp.as_file_mut();
    if base != p.base_sha256 {
        return fail("baseline hash mismatch");
    }
    let info = inspect(f)?;
    let mut seen = BTreeSet::new();
    let mut edits = Vec::new();
    for e in &p.edits {
        if !seen.insert((&e.tensor, e.index)) {
            return fail("duplicate edit");
        }
        if !e.value.is_finite() || e.value.abs() > 1_000_000.0 {
            return fail("invalid replacement");
        }
        let t = info.tensors.get(&e.tensor).ok_or("unknown tensor")?;
        let w = width(&t.dtype)?;
        let count = (t.data_offsets[1] - t.data_offsets[0]) / w;
        if e.index >= count {
            return fail("index out of range");
        }
        let offset = info.data_start + t.data_offsets[0] + e.index * w;
        f.seek(SeekFrom::Start(offset))?;
        let mut before = vec![0; w as usize];
        f.read_exact(&mut before)?;
        let old = decode(&t.dtype, &before)?;
        let after = encode(&t.dtype, e.value)?;
        let actual = decode(&t.dtype, &after)?;
        if !old.is_finite() || (actual - old).abs() > 1.0 {
            return fail("nonfinite baseline or delta exceeds 1");
        }
        if after == before {
            return fail("edit is a no-op after dtype rounding");
        }
        edits.push(Undo {
            offset,
            before,
            after,
        });
    }
    for e in &edits {
        f.seek(SeekFrom::Start(e.offset))?;
        f.write_all(&e.after)?;
    }
    inspect(f)?;
    let candidate = hash(f)?;
    commit(temp, dest)?;
    Ok(Receipt {
        schema: "ruv-fann.tensor-receipt/1".into(),
        base_sha256: base,
        candidate_sha256: candidate,
        edits,
        behavioral_validation: "not_run".into(),
    })
}
pub fn rollback(source: &Path, dest: &Path, r: &Receipt) -> Result<()> {
    if r.schema != "ruv-fann.tensor-receipt/1" || r.edits.is_empty() || r.edits.len() > 128 {
        return fail("invalid receipt");
    }
    let (mut temp, candidate_hash) = copy_snapshot(source, dest)?;
    let f = temp.as_file_mut();
    if candidate_hash != r.candidate_sha256 {
        return fail("candidate hash mismatch");
    }
    let info = inspect(f)?;
    let mut ranges = BTreeSet::new();
    for e in &r.edits {
        let n = e.before.len();
        if ![2, 4].contains(&n)
            || n != e.after.len()
            || e.offset < info.data_start
            || e.offset
                .checked_add(n as u64)
                .filter(|x| *x <= info.bytes)
                .is_none()
        {
            return fail("invalid undo range");
        }
        for i in e.offset..e.offset + n as u64 {
            if !ranges.insert(i) {
                return fail("overlapping undo");
            }
        }
        f.seek(SeekFrom::Start(e.offset))?;
        let mut b = vec![0; n];
        f.read_exact(&mut b)?;
        if b != e.after {
            return fail("undo bytes mismatch");
        }
        f.seek(SeekFrom::Start(e.offset))?;
        f.write_all(&e.before)?;
    }
    if hash(f)? != r.base_sha256 {
        return fail("restored hash mismatch");
    }
    inspect(f)?;
    commit(temp, dest)
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return fail("JSON exceeds 1 MiB");
    }
    Ok(serde_json::from_slice(&bytes)?)
}
