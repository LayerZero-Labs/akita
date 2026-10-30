//! Canonical byte encodings for preprocessing identities and the manifest.
//!
//! Integers are little-endian, byte strings and lists carry a `u64` length,
//! and every hash input starts with its own length-prefixed ASCII tag. The
//! reader is total: every malformed input returns an error and no length is
//! trusted before it is compared with the bytes that remain.

use akita_error::AkitaError;
use akita_params::descriptor_bytes::{push_u32, push_usize};

pub(crate) fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

pub(crate) fn push_u32_le(out: &mut Vec<u8>, value: u32) {
    push_u32(out, value);
}

pub(crate) fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// `bytes(x) = u64(len(x)) || x`.
pub(crate) fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    push_usize(out, bytes.len());
    out.extend_from_slice(bytes);
}

/// `str(s) = bytes(UTF-8 of s)`; `tag(t)` uses the same encoding.
pub(crate) fn push_str(out: &mut Vec<u8>, value: &str) {
    push_bytes(out, value.as_bytes());
}

pub(crate) fn push_count(out: &mut Vec<u8>, count: usize) {
    push_usize(out, count);
}

pub(crate) fn malformed(what: &str) -> AkitaError {
    AkitaError::InvalidSetup(format!("malformed preprocessing manifest: {what}"))
}

/// Bounded reader over one fully buffered encoding.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    pub(crate) fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8], AkitaError> {
        if len > self.bytes.len() {
            return Err(malformed(what));
        }
        let (head, tail) = self.bytes.split_at(len);
        self.bytes = tail;
        Ok(head)
    }

    pub(crate) fn u8(&mut self, what: &str) -> Result<u8, AkitaError> {
        Ok(self.take(1, what)?[0])
    }

    pub(crate) fn u32(&mut self, what: &str) -> Result<u32, AkitaError> {
        let mut out = [0u8; 4];
        out.copy_from_slice(self.take(4, what)?);
        Ok(u32::from_le_bytes(out))
    }

    pub(crate) fn u64(&mut self, what: &str) -> Result<u64, AkitaError> {
        let mut out = [0u8; 8];
        out.copy_from_slice(self.take(8, what)?);
        Ok(u64::from_le_bytes(out))
    }

    pub(crate) fn digest(&mut self, what: &str) -> Result<[u8; 32], AkitaError> {
        let mut out = [0u8; 32];
        out.copy_from_slice(self.take(32, what)?);
        Ok(out)
    }

    /// A `0` or `1` flag; any other byte is invalid.
    pub(crate) fn flag(&mut self, what: &str) -> Result<bool, AkitaError> {
        match self.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(malformed(what)),
        }
    }

    /// A length or count of at most `max` that the remaining input can hold
    /// at `min_item_bytes` bytes per item.
    pub(crate) fn count(
        &mut self,
        max: usize,
        min_item_bytes: usize,
        what: &str,
    ) -> Result<usize, AkitaError> {
        let count = usize::try_from(self.u64(what)?).map_err(|_| malformed(what))?;
        let needed = count
            .checked_mul(min_item_bytes)
            .ok_or_else(|| malformed(what))?;
        if count > max || needed > self.bytes.len() {
            return Err(malformed(what));
        }
        Ok(count)
    }

    pub(crate) fn bytes(&mut self, max: usize, what: &str) -> Result<&'a [u8], AkitaError> {
        let len = self.count(max, 1, what)?;
        self.take(len, what)
    }

    pub(crate) fn str(&mut self, max: usize, what: &str) -> Result<&'a str, AkitaError> {
        core::str::from_utf8(self.bytes(max, what)?).map_err(|_| malformed(what))
    }

    pub(crate) fn finish(self, what: &str) -> Result<(), AkitaError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(malformed(what))
        }
    }
}

/// Lowercase hex, used for digests in file names.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

/// Decode 64 lowercase hex digits into a digest, usable in constants.
///
/// # Panics
///
/// Panics on any other input. In a `const` item the panic is a compile error.
pub const fn digest_from_hex(hex: &str) -> [u8; 32] {
    let bytes = hex.as_bytes();
    assert!(bytes.len() == 64, "digest hex must have 64 digits");
    let mut out = [0u8; 32];
    let mut index = 0;
    while index < 32 {
        out[index] = (hex_digit(bytes[2 * index]) << 4) | hex_digit(bytes[2 * index + 1]);
        index += 1;
    }
    out
}

const fn hex_digit(digit: u8) -> u8 {
    match digit {
        b'0'..=b'9' => digit - b'0',
        b'a'..=b'f' => digit - b'a' + 10,
        _ => panic!("digest hex must use lowercase hex digits"),
    }
}

/// Parse lowercase hex at run time.
pub fn parse_hex(hex: &str) -> Result<Vec<u8>, AkitaError> {
    let bytes = hex.as_bytes();
    if !bytes.len().is_multiple_of(2)
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(AkitaError::InvalidInput(format!(
            "expected lowercase hex, got {hex:?}"
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| (hex_digit(pair[0]) << 4) | hex_digit(pair[1]))
        .collect())
}

/// Parse 64 lowercase hex digits at run time.
pub fn parse_digest_hex(hex: &str) -> Result<[u8; 32], AkitaError> {
    <[u8; 32]>::try_from(parse_hex(hex)?).map_err(|_| {
        AkitaError::InvalidInput(format!("expected 64 lowercase hex digits, got {hex:?}"))
    })
}
