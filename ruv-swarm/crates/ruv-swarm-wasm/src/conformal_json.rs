//! Allocation and numeric boundaries for the conformal JSON wire protocol.
use serde::{
    de::{self, SeqAccess, Visitor},
    Deserialize,
};
use std::{
    fmt,
    io::{self, Write},
    marker::PhantomData,
};
pub(super) const INPUT_BYTES: usize = 4 * 1024 * 1024;
pub(super) const OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const SAFE_INTEGER: f64 = 9_007_199_254_740_991.;

pub(super) fn parse<T: serde::de::DeserializeOwned>(input: &str) -> Result<T, String> {
    if input.len() > INPUT_BYTES {
        return Err("Conformal JSON exceeds 4 MiB input budget".into());
    }
    exact_integer_tokens(input)?;
    serde_json::from_str(input).map_err(|e| format!("Invalid conformal JSON: {e}"))
}
// serde_json may parse integer literals beyond u64 as f64. Inspect their raw
// tokens before parsing so a rounded original integer cannot hide a residual.
// Floats containing '.' or an exponent retain the documented Float64 semantics.
fn exact_integer_tokens(input: &str) -> Result<(), String> {
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut quoted = false;
    let mut escape = false;
    while i < bytes.len() {
        let b = bytes[i];
        if quoted {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                quoted = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            quoted = true;
            i += 1;
            continue;
        }
        if b == b'-' || b.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || b".+-eE".contains(&bytes[i])) {
                i += 1;
            }
            let token = &input[start..i];
            if !token.contains(['.', 'e', 'E']) {
                let exact = if token.starts_with('-') {
                    token
                        .parse::<i128>()
                        .ok()
                        .is_some_and(|n| (n as f64) as i128 == n)
                } else {
                    token.parse::<u128>().ok().is_some_and(|n| {
                        let f = n as f64;
                        f < (u128::MAX as f64) && f as u128 == n
                    })
                };
                if !exact {
                    return Err(
                        "Conformal integer token is not exactly representable as Float64".into(),
                    );
                }
            }
            continue;
        }
        i += 1;
    }
    Ok(())
}
pub(super) fn number<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    let n = serde_json::Number::deserialize(d)?;
    if n.as_i64()
        .is_some_and(|x| (x as f64) as i128 != i128::from(x))
        || n.as_u64()
            .is_some_and(|x| (x as f64) as u128 != u128::from(x))
    {
        return Err(de::Error::custom(
            "Conformal integer is not exactly representable as Float64",
        ));
    }
    n.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| de::Error::custom("Conformal numbers must be finite Float64"))
}
pub(super) fn time<'de, D: serde::Deserializer<'de>>(d: D) -> Result<i64, D::Error> {
    let n = number(d)?;
    if n.abs() > SAFE_INTEGER || n.fract() != 0. {
        return Err(de::Error::custom(
            "Conformal timestamps must be JS safe-integer Unix milliseconds",
        ));
    }
    Ok(n as i64)
}
pub(super) fn name<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    struct Name;
    impl Visitor<'_> for Name {
        type Value = String;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a nonempty name of at most 256 UTF-8 bytes")
        }
        fn visit_str<E: de::Error>(self, s: &str) -> Result<String, E> {
            if s.trim().is_empty() || s.len() > 256 {
                return Err(E::custom(
                    "Conformal name exceeds UTF-8 byte budget or is empty",
                ));
            }
            Ok(s.to_owned())
        }
        fn visit_string<E: de::Error>(self, s: String) -> Result<String, E> {
            if s.trim().is_empty() || s.len() > 256 {
                return Err(E::custom(
                    "Conformal name exceeds UTF-8 byte budget or is empty",
                ));
            }
            Ok(s)
        }
    }
    d.deserialize_string(Name)
}
pub(super) fn bounded<'de, D, T, const N: usize>(d: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Bounded<T, const N: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const N: usize> Visitor<'de> for Bounded<T, N> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "an array of at most {N} entries")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<T>, A::Error> {
            let mut out = Vec::new();
            while let Some(value) = seq.next_element()? {
                if out.len() == N {
                    return Err(de::Error::custom("Conformal array entry budget exceeded"));
                }
                out.push(value);
            }
            Ok(out)
        }
    }
    d.deserialize_seq(Bounded::<T, N>(PhantomData))
}
struct BoundedWriter {
    bytes: Vec<u8>,
}
impl Write for BoundedWriter {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(input.len())
            .is_none_or(|n| n > OUTPUT_BYTES)
        {
            return Err(io::Error::other(
                "Conformal serialized output exceeds 8 MiB",
            ));
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn serialize<T: serde::Serialize>(value: &T) -> Result<String, String> {
    let mut writer = BoundedWriter { bytes: Vec::new() };
    serde_json::to_writer(&mut writer, value).map_err(|e| e.to_string())?;
    String::from_utf8(writer.bytes).map_err(|e| e.to_string())
}
