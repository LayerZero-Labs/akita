//! Canonical Golomb-Rice codec for terminal tail `z` segments.

use akita_field::AkitaError;

/// Maximum unary quotient before the bounded escape literal.
pub const GOLOMB_RICE_Q_MAX: u32 = 32;

/// Bit cursor over a byte slice for no-panic decode.
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    pub(crate) fn remaining_bits(&self) -> usize {
        self.bytes.len().saturating_mul(8).saturating_sub(self.bit_pos)
    }

    pub(crate) fn read_bit(&mut self) -> Result<bool, AkitaError> {
        if self.bit_pos >= self.bytes.len().saturating_mul(8) {
            return Err(AkitaError::InvalidProof);
        }
        let byte_idx = self.bit_pos / 8;
        let bit_idx = self.bit_pos % 8;
        self.bit_pos += 1;
        let byte = *self.bytes.get(byte_idx).ok_or(AkitaError::InvalidProof)?;
        Ok((byte >> bit_idx) & 1 == 1)
    }

    pub(crate) fn read_bits(&mut self, count: u32) -> Result<u64, AkitaError> {
        if count == 0 {
            return Ok(0);
        }
        if count > 63 {
            return Err(AkitaError::InvalidProof);
        }
        let needed = count as usize;
        if self.remaining_bits() < needed {
            return Err(AkitaError::InvalidProof);
        }
        let mut out = 0u64;
        for i in 0..needed {
            if self.read_bit()? {
                out |= 1u64 << i;
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn write_bit(&mut self, bit: bool) {
        let byte_idx = self.bit_pos / 8;
        let bit_idx = self.bit_pos % 8;
        if byte_idx >= self.bytes.len() {
            self.bytes.push(0);
        }
        if bit {
            self.bytes[byte_idx] |= 1u8 << bit_idx;
        }
        self.bit_pos += 1;
    }

    fn write_bits(&mut self, value: u64, count: u32) {
        for i in 0..count {
            self.write_bit((value >> i) & 1 == 1);
        }
    }
}

/// Zigzag map a signed integer in `[-2^(W-1), 2^(W-1))` to non-negative `u`.
pub fn zigzag_encode(n: i64, width: u32) -> Result<u64, AkitaError> {
    if width == 0 || width > 63 {
        return Err(AkitaError::InvalidSetup(
            "golomb-rice zigzag width out of range".to_string(),
        ));
    }
    let min = -(1i64 << (width - 1));
    let max = (1i64 << (width - 1)) - 1;
    if n < min || n > max {
        return Err(AkitaError::InvalidProof);
    }
    Ok(((n << 1) ^ (n >> 63)) as u64)
}

/// Inverse of [`zigzag_encode`].
pub fn zigzag_decode(u: u64, width: u32) -> Result<i64, AkitaError> {
    if width == 0 || width > 63 {
        return Err(AkitaError::InvalidSetup(
            "golomb-rice zigzag width out of range".to_string(),
        ));
    }
    let n = ((u >> 1) as i64) ^ (-((u & 1) as i64));
    let min = -(1i64 << (width - 1));
    let max = (1i64 << (width - 1)) - 1;
    if n < min || n > max {
        return Err(AkitaError::InvalidProof);
    }
    Ok(n)
}

/// Rice parameter `k` from public fold-response `sigma`.
#[must_use]
pub fn optimal_rice_k(sigma: u128) -> u32 {
    if sigma <= 1 {
        return 0;
    }
    (u128::BITS - 1 - sigma.leading_zeros()) as u32
}

/// Signed zigzag width for the folded `z` segment from public digit bounds.
#[must_use]
pub fn golomb_rice_zigzag_width_z(num_digits_fold: usize, log_basis: u32) -> u32 {
    let digit_bits = num_digits_fold.saturating_mul(log_basis as usize);
    digit_bits.saturating_add(1).max(1) as u32
}

fn golomb_rice_encode_one_into(
    writer: &mut BitWriter,
    n: i64,
    k: u32,
    w: u32,
) -> Result<(), AkitaError> {
    let u = zigzag_encode(n, w)?;
    let quotient = if k == 0 { u } else { u >> k };
    let remainder = if k == 0 { 0 } else { u & ((1u64 << k) - 1) };
    if quotient >= u64::from(GOLOMB_RICE_Q_MAX) {
        for _ in 0..GOLOMB_RICE_Q_MAX {
            writer.write_bit(true);
        }
        writer.write_bit(false);
        writer.write_bits(u, w);
    } else {
        for _ in 0..quotient {
            writer.write_bit(true);
        }
        writer.write_bit(false);
        writer.write_bits(remainder, k);
    }
    Ok(())
}

fn golomb_rice_decode_one_from(
    reader: &mut BitReader<'_>,
    k: u32,
    w: u32,
) -> Result<i64, AkitaError> {
    let mut quotient = 0u64;
    loop {
        if reader.remaining_bits() == 0 {
            return Err(AkitaError::InvalidProof);
        }
        if reader.read_bit()? {
            quotient += 1;
            if quotient > u64::from(GOLOMB_RICE_Q_MAX) {
                return Err(AkitaError::InvalidProof);
            }
            continue;
        }
        break;
    }
    let u = if quotient >= u64::from(GOLOMB_RICE_Q_MAX) {
        reader.read_bits(w)?
    } else if k == 0 {
        quotient
    } else {
        let remainder = reader.read_bits(k)?;
        (quotient << k) | remainder
    };
    zigzag_decode(u, w)
}

/// Concatenated Golomb-Rice encoding for a fixed-length integer vector.
pub fn golomb_rice_encode_vec(values: &[i64], k: u32, w: u32) -> Result<Vec<u8>, AkitaError> {
    let mut writer = BitWriter::default();
    for &n in values {
        golomb_rice_encode_one_into(&mut writer, n, k, w)?;
    }
    Ok(writer.finish())
}

/// Decode a fixed number of Golomb-Rice integers from `bytes`.
pub fn golomb_rice_decode_vec(
    bytes: &[u8],
    count: usize,
    k: u32,
    w: u32,
) -> Result<Vec<i64>, AkitaError> {
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(golomb_rice_decode_one_from(&mut reader, k, w)?);
    }
    if reader.remaining_bits() > 7 {
        return Err(AkitaError::InvalidProof);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golomb_rice_round_trip_and_canonicality() {
        let k = 3u32;
        let w = 12u32;
        let values = [-100i64, -1, 0, 1, 42, 500];
        let encoded = golomb_rice_encode_vec(&values, k, w).unwrap();
        let decoded = golomb_rice_decode_vec(&encoded, values.len(), k, w).unwrap();
        assert_eq!(decoded, values);
        let reencoded = golomb_rice_encode_vec(&decoded, k, w).unwrap();
        assert_eq!(encoded, reencoded);
    }

    #[test]
    fn golomb_rice_escape_path_round_trip() {
        let k = 0u32;
        let w = 16u32;
        let values = vec![200i64; 1];
        let encoded = golomb_rice_encode_vec(&values, k, w).unwrap();
        let decoded = golomb_rice_decode_vec(&encoded, 1, k, w).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn golomb_rice_decode_rejects_trailing_garbage() {
        let k = 2u32;
        let w = 8u32;
        let mut encoded = golomb_rice_encode_vec(&[3i64, 5i64], k, w).unwrap();
        encoded.push(0xff);
        assert!(golomb_rice_decode_vec(&encoded, 2, k, w).is_err());
    }

    #[test]
    fn golomb_rice_decode_is_total_on_empty_prefix() {
        assert!(golomb_rice_decode_vec(&[], 1, 0, 4).is_err());
    }
}
