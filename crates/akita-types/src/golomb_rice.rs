//! Canonical Golomb-Rice codec for terminal tail `z` segments.
//!
//! Wire format is standard Rice only: unary quotient prefix + stop bit + low-bit remainder.
//! Decode rejects unary runs longer than the cap-derived maximum quotient.

use akita_field::AkitaError;

use crate::instance_descriptor::FoldLinfProtocolBinding;
use crate::tail_golomb_rice_low_bits::{
    cap_rice_low_bits, wire_rice_low_bits, wire_rice_low_bits_from_rule,
};

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

    pub(crate) fn bit_pos(&self) -> usize {
        self.bit_pos
    }

    pub(crate) fn remaining_bits(&self) -> usize {
        self.bytes
            .len()
            .saturating_mul(8)
            .saturating_sub(self.bit_pos)
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

    fn bit_len(&self) -> usize {
        self.bit_pos
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

/// Rice low-bit width from a per-coordinate magnitude scale (e.g. fold `‖z‖_inf` cap).
///
/// Equals `floor(log2(scale))` for `scale > 1`; divisor is `2^rice_low_bits`.
#[must_use]
pub fn rice_low_bits_for_cap(scale: u128) -> u32 {
    if scale <= 1 {
        return 0;
    }
    u128::BITS - 1 - scale.leading_zeros()
}

/// Signed zigzag width for fold-response coefficients bounded by `scale`.
///
/// Mirrors the `[-scale, scale]` envelope priced by
/// [`crate::LevelParams::fold_witness_linf_cap_for_claims`].
#[must_use]
pub fn golomb_rice_zigzag_width(scale: u128) -> u32 {
    if scale == 0 {
        return 1;
    }
    128u32
        .saturating_sub(scale.leading_zeros())
        .saturating_add(1)
        .max(1)
}

/// Average-case tail-`z` planner model bound into [`crate::instance_descriptor::FoldLinfProtocolBinding`].
///
/// Planner model: budget `cap_rice_low_bits + 2` bits per coordinate. Bump when recalibrating
/// [`tail_z_planner_bits_per_coord`].
pub const TAIL_Z_PLANNER_CAP_LOW_BITS_PLUS_TWO: u8 = 1;

/// Average-case planner bit budget per `z` coordinate from cap-derived low-bit width.
///
/// Public `(rice_low_bits, W)` still cover every coefficient in `[-cap, cap]`; this budget
/// prices honest witnesses under [`TAIL_Z_PLANNER_CAP_LOW_BITS_PLUS_TWO`].
#[must_use]
pub fn tail_z_planner_bits_per_coord(cap_rice_low_bits: u32) -> usize {
    (cap_rice_low_bits as usize).saturating_add(2)
}

/// Golomb-Rice bit length for one coordinate at public `(rice_low_bits, zigzag_w)`.
pub fn golomb_rice_bits_for_coord(
    n: i64,
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<usize, AkitaError> {
    let mut writer = BitWriter::default();
    golomb_rice_encode_one_into(&mut writer, n, rice_low_bits, zigzag_w)?;
    Ok(writer.bit_len())
}

/// Total Golomb-Rice bits for a vector at public `(rice_low_bits, zigzag_w)`.
pub fn golomb_rice_total_bits(
    values: &[i64],
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<usize, AkitaError> {
    values.iter().try_fold(0usize, |acc, &n| {
        golomb_rice_bits_for_coord(n, rice_low_bits, zigzag_w)?
            .checked_add(acc)
            .ok_or(AkitaError::InvalidSetup(
                "golomb-rice total bits overflow".to_string(),
            ))
    })
}

/// Witness-sample Rice low-bit width minimizing total Golomb bits (not public).
#[must_use]
pub fn sample_optimal_rice_low_bits(values: &[i64], zigzag_w: u32, low_bits_hi: u32) -> u32 {
    (0..=low_bits_hi)
        .min_by_key(|&rice_low_bits| {
            golomb_rice_total_bits(values, rice_low_bits, zigzag_w).unwrap_or(usize::MAX)
        })
        .unwrap_or(0)
}

/// Payload byte length for each `rice_low_bits` in `0..=low_bits_hi` on a realized `z` sample.
pub fn golomb_rice_low_bits_sweep_payload_bytes(
    values: &[i64],
    zigzag_w: u32,
    low_bits_hi: u32,
) -> Result<Vec<(u32, usize)>, AkitaError> {
    (0..=low_bits_hi)
        .map(|rice_low_bits| {
            let bits = golomb_rice_total_bits(values, rice_low_bits, zigzag_w)?;
            Ok((rice_low_bits, bits.div_ceil(8)))
        })
        .collect()
}

/// Distribution summary for terminal fold-response `z` coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct ZFoldEncodingStats {
    pub coord_count: usize,
    pub witness_linf_cap: u128,
    pub observed_max_abs: u64,
    pub mean_abs: f64,
    pub median_abs: u64,
    pub p90_abs: u64,
    pub p99_abs: u64,
    pub zigzag_w: u32,
    pub rice_low_bits_cap: u32,
    pub rice_low_bits_wire: u32,
    pub rice_low_bits_sample: u32,
    pub bits_per_coord_at_cap: f64,
    pub bits_per_coord_at_wire: f64,
    pub bits_per_coord_at_sample: f64,
    pub bits_per_coord_packed_digits: f64,
    pub total_bits_at_cap: usize,
    pub total_bits_at_wire: usize,
    pub total_bits_at_sample: usize,
    pub total_bits_packed_digits: usize,
    pub actual_payload_bytes: usize,
}

fn percentile_abs(sorted_abs: &[u64], p_num: usize, p_den: usize) -> u64 {
    if sorted_abs.is_empty() {
        return 0;
    }
    let idx = sorted_abs
        .len()
        .saturating_mul(p_num)
        .saturating_div(p_den)
        .min(sorted_abs.len() - 1);
    sorted_abs[idx]
}

/// Analyze realized `z` coefficients against public bounds and Golomb models.
///
/// `values` are centered fold-response ring coefficients (one per `z_coord`).
pub fn analyze_z_fold_golomb_encoding(
    values: &[i64],
    witness_linf_cap: u128,
    zigzag_w: u32,
    depth_fold: usize,
    log_basis: u32,
    actual_payload_bytes: usize,
) -> Result<ZFoldEncodingStats, AkitaError> {
    let rice_low_bits_cap = cap_rice_low_bits(witness_linf_cap);
    let rice_low_bits_wire = wire_rice_low_bits(witness_linf_cap);
    let low_bits_search_hi = rice_low_bits_cap.saturating_add(4);
    let rice_low_bits_sample = sample_optimal_rice_low_bits(values, zigzag_w, low_bits_search_hi);

    let mut abs_vals: Vec<u64> = values.iter().map(|&n| n.unsigned_abs()).collect();
    abs_vals.sort_unstable();
    let observed_max_abs = *abs_vals.last().unwrap_or(&0);
    let sum_abs: u128 = abs_vals.iter().map(|&a| u128::from(a)).sum();
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        sum_abs as f64 / values.len() as f64
    };

    let total_bits_at_cap = golomb_rice_total_bits(values, rice_low_bits_cap, zigzag_w)?;
    let total_bits_at_wire = golomb_rice_total_bits(values, rice_low_bits_wire, zigzag_w)?;
    let total_bits_at_sample = golomb_rice_total_bits(values, rice_low_bits_sample, zigzag_w)?;
    let bits_per_digit_plane = log_basis as usize;
    let total_bits_packed_digits = values
        .len()
        .saturating_mul(depth_fold)
        .saturating_mul(bits_per_digit_plane);
    let n = values.len().max(1) as f64;

    Ok(ZFoldEncodingStats {
        coord_count: values.len(),
        witness_linf_cap,
        observed_max_abs,
        mean_abs,
        median_abs: percentile_abs(&abs_vals, 50, 100),
        p90_abs: percentile_abs(&abs_vals, 90, 100),
        p99_abs: percentile_abs(&abs_vals, 99, 100),
        zigzag_w,
        rice_low_bits_cap,
        rice_low_bits_wire,
        rice_low_bits_sample,
        bits_per_coord_at_cap: total_bits_at_cap as f64 / n,
        bits_per_coord_at_wire: total_bits_at_wire as f64 / n,
        bits_per_coord_at_sample: total_bits_at_sample as f64 / n,
        bits_per_coord_packed_digits: total_bits_packed_digits as f64 / n,
        total_bits_at_cap,
        total_bits_at_wire,
        total_bits_at_sample,
        total_bits_packed_digits,
        actual_payload_bytes,
    })
}

/// Golomb unary quotient for one coefficient at public `(rice_low_bits, zigzag_w)`.
pub fn golomb_rice_quotient_for_coord(
    n: i64,
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<u64, AkitaError> {
    let u = zigzag_encode(n, zigzag_w)?;
    Ok(if rice_low_bits == 0 {
        u
    } else {
        u >> rice_low_bits
    })
}

/// Maximum Golomb quotient among coefficients in `[-cap, cap]` at public `(rice_low_bits, zigzag_w)`.
///
/// Used as the decode unary-run bound for terminal tail `z`: any wire with a longer unary prefix
/// is rejected before reading the remainder.
pub fn golomb_rice_max_quotient_for_cap(
    cap: u128,
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<u64, AkitaError> {
    let cap_i64 = i64::try_from(cap).map_err(|_| {
        AkitaError::InvalidSetup(format!(
            "fold witness linf cap {cap} exceeds i64 for golomb quotient bound"
        ))
    })?;
    golomb_rice_quotient_for_coord(cap_i64, rice_low_bits, zigzag_w)
}

/// Closed-form standard Golomb wire bits for one coefficient.
pub fn golomb_rice_coord_wire_bits(
    n: i64,
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<usize, AkitaError> {
    let quotient = golomb_rice_quotient_for_coord(n, rice_low_bits, zigzag_w)?;
    let unary = quotient as usize + 1;
    Ok(unary.saturating_add(rice_low_bits as usize))
}

/// Total standard Golomb wire bits for a coefficient vector.
pub fn golomb_rice_total_wire_bits(
    values: &[i64],
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<usize, AkitaError> {
    values.iter().try_fold(0usize, |acc, &n| {
        golomb_rice_coord_wire_bits(n, rice_low_bits, zigzag_w)?
            .checked_add(acc)
            .ok_or(AkitaError::InvalidSetup(
                "golomb-rice total wire bits overflow".to_string(),
            ))
    })
}

/// Whether every coefficient lies in `[-cap, cap]`.
pub fn golomb_rice_values_within_cap(values: &[i64], cap: u128) -> Result<(), AkitaError> {
    for &n in values {
        if i128::from(n).unsigned_abs() > cap {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(())
}

fn terminal_z_wire_rice_params(cap: u128) -> Result<(u32, u32), AkitaError> {
    let binding = FoldLinfProtocolBinding::CURRENT;
    let rice_low_bits = wire_rice_low_bits_from_rule(
        cap,
        binding.wire_rice_low_bits_rule_id,
        binding.wire_rice_low_bits_delta,
    )?;
    Ok((rice_low_bits, golomb_rice_zigzag_width(cap)))
}

fn centered_rows_to_i64<const D: usize>(rows: &[[i32; D]]) -> Vec<i64> {
    rows.iter()
        .flat_map(|row| row.iter().map(|&n| i64::from(n)))
        .collect()
}

/// Whether total wire bits fit the planner budget (`cap_rice_low_bits + 2` per coord).
pub fn golomb_rice_values_fit_planner_wire_budget(
    values: &[i64],
    cap: u128,
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<(), AkitaError> {
    let budget_bits = tail_z_planner_bits_per_coord(cap_rice_low_bits(cap))
        .checked_mul(values.len())
        .ok_or(AkitaError::InvalidSetup(
            "terminal z planner bit budget overflow".to_string(),
        ))?;
    let total_bits = golomb_rice_total_wire_bits(values, rice_low_bits, zigzag_w)?;
    if total_bits > budget_bits {
        return Err(AkitaError::InvalidInput(format!(
            "terminal z golomb payload needs {total_bits} bits, planner budget is {budget_bits}"
        )));
    }
    Ok(())
}

/// Whether every centered row coefficient lies in `[-cap, cap]`.
pub fn golomb_rice_rows_encodable_at_wire_low_bits<const D: usize>(
    rows: &[[i32; D]],
    cap: u128,
) -> Result<(), AkitaError> {
    if cap == 0 && rows.iter().any(|row| row.iter().any(|&n| n != 0)) {
        return Err(AkitaError::InvalidInput(
            "golomb-rice encodability check at zero cap".to_string(),
        ));
    }
    golomb_rice_values_within_cap(&centered_rows_to_i64(rows), cap).map_err(|_| {
        AkitaError::InvalidInput(format!("centered coefficient exceeds fold cap {cap}"))
    })
}

/// Whether every centered row is admissible at wire low bits and fits the planner bit budget.
pub fn golomb_rice_rows_admit_terminal_wire<const D: usize>(
    rows: &[[i32; D]],
    cap: u128,
) -> Result<(), AkitaError> {
    golomb_rice_rows_encodable_at_wire_low_bits(rows, cap)?;
    let values = centered_rows_to_i64(rows);
    let (rice_low_bits, zigzag_w) = terminal_z_wire_rice_params(cap)?;
    golomb_rice_values_fit_planner_wire_budget(&values, cap, rice_low_bits, zigzag_w)
}

fn golomb_rice_encode_one_into(
    writer: &mut BitWriter,
    n: i64,
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<(), AkitaError> {
    let u = zigzag_encode(n, zigzag_w)?;
    let quotient = if rice_low_bits == 0 {
        u
    } else {
        u >> rice_low_bits
    };
    let remainder = if rice_low_bits == 0 {
        0
    } else {
        u & ((1u64 << rice_low_bits) - 1)
    };
    for _ in 0..quotient {
        writer.write_bit(true);
    }
    writer.write_bit(false);
    if rice_low_bits == 0 {
        // quotient carries the full zigzag value when rice_low_bits = 0.
    } else {
        writer.write_bits(remainder, rice_low_bits);
    }
    Ok(())
}

fn golomb_rice_decode_one_from(
    reader: &mut BitReader<'_>,
    rice_low_bits: u32,
    zigzag_w: u32,
    max_quotient: u64,
) -> Result<i64, AkitaError> {
    let mut quotient = 0u64;
    loop {
        if reader.remaining_bits() == 0 {
            return Err(AkitaError::InvalidProof);
        }
        if reader.read_bit()? {
            quotient += 1;
            if quotient > max_quotient {
                return Err(AkitaError::InvalidProof);
            }
            continue;
        }
        break;
    }
    let u = if rice_low_bits == 0 {
        quotient
    } else {
        let remainder = reader.read_bits(rice_low_bits)?;
        (quotient << rice_low_bits) | remainder
    };
    zigzag_decode(u, zigzag_w)
}

/// Consume zero padding in the last partial byte, then reject any extra bytes.
fn golomb_rice_consume_canonical_padding(reader: &mut BitReader<'_>) -> Result<(), AkitaError> {
    let bits_after_coords = reader.bit_pos();
    let padding_bits = (8 - (bits_after_coords % 8)) % 8;
    for _ in 0..padding_bits {
        if reader.read_bit()? {
            return Err(AkitaError::InvalidProof);
        }
    }
    if reader.remaining_bits() > 0 {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

/// Concatenated Golomb-Rice encoding for a fixed-length integer vector.
pub fn golomb_rice_encode_vec(
    values: &[i64],
    rice_low_bits: u32,
    zigzag_w: u32,
) -> Result<Vec<u8>, AkitaError> {
    let mut writer = BitWriter::default();
    for &n in values {
        golomb_rice_encode_one_into(&mut writer, n, rice_low_bits, zigzag_w)?;
    }
    Ok(writer.finish())
}

/// Decode a fixed number of Golomb-Rice integers from `bytes`.
///
/// Rejects unary quotients above `max_quotient`, non-zero trailing bits, and any byte padding
/// beyond the minimal length for the encoded bitstream.
pub fn golomb_rice_decode_vec(
    bytes: &[u8],
    count: usize,
    rice_low_bits: u32,
    zigzag_w: u32,
    max_quotient: u64,
) -> Result<Vec<i64>, AkitaError> {
    let mut reader = BitReader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(golomb_rice_decode_one_from(
            &mut reader,
            rice_low_bits,
            zigzag_w,
            max_quotient,
        )?);
    }
    golomb_rice_consume_canonical_padding(&mut reader)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tail_golomb_rice_low_bits::wire_rice_low_bits;

    fn max_quotient_for_values(values: &[i64], rice_low_bits: u32, zigzag_w: u32) -> u64 {
        values
            .iter()
            .map(|&n| golomb_rice_quotient_for_coord(n, rice_low_bits, zigzag_w).expect("quotient"))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn golomb_rice_round_trips_cap_range_at_wire_low_bits() {
        for cap in [504u128, 1008, 1568, 2016] {
            let rice_low_bits = wire_rice_low_bits(cap);
            let zigzag_w = golomb_rice_zigzag_width(cap);
            let max_quotient =
                golomb_rice_max_quotient_for_cap(cap, rice_low_bits, zigzag_w).expect("max q");
            let cap_i64 = cap as i64;
            for n in -cap_i64..=cap_i64 {
                let encoded =
                    golomb_rice_encode_vec(&[n], rice_low_bits, zigzag_w).expect("encode");
                let decoded =
                    golomb_rice_decode_vec(&encoded, 1, rice_low_bits, zigzag_w, max_quotient)
                        .expect("decode");
                assert_eq!(decoded, [n], "cap={cap} n={n}");
            }
        }
    }

    #[test]
    fn rice_low_bits_for_cap_tracks_per_coefficient_scale() {
        let cap = 6_912u128;
        assert_eq!(rice_low_bits_for_cap(cap), 12);
        let level_variance_envelope = 6_189_618u128;
        assert!(
            rice_low_bits_for_cap(cap) < rice_low_bits_for_cap(level_variance_envelope),
            "per-coordinate low bits must use fold cap, not the level variance envelope"
        );
    }

    #[test]
    fn golomb_rice_round_trip_and_canonicality() {
        let rice_low_bits = 3u32;
        let zigzag_w = 12u32;
        let values = [-100i64, -1, 0, 1, 42, 500];
        let max_quotient = max_quotient_for_values(&values, rice_low_bits, zigzag_w);
        let encoded = golomb_rice_encode_vec(&values, rice_low_bits, zigzag_w).unwrap();
        let decoded = golomb_rice_decode_vec(
            &encoded,
            values.len(),
            rice_low_bits,
            zigzag_w,
            max_quotient,
        )
        .unwrap();
        assert_eq!(decoded, values);
        let reencoded = golomb_rice_encode_vec(&decoded, rice_low_bits, zigzag_w).unwrap();
        assert_eq!(encoded, reencoded);
    }

    #[test]
    fn golomb_rice_decode_rejects_trailing_garbage() {
        let rice_low_bits = 2u32;
        let zigzag_w = 8u32;
        let values = [3i64, 5i64];
        let max_quotient = max_quotient_for_values(&values, rice_low_bits, zigzag_w);
        let mut encoded = golomb_rice_encode_vec(&values, rice_low_bits, zigzag_w).unwrap();
        encoded.push(0xff);
        assert!(
            golomb_rice_decode_vec(&encoded, 2, rice_low_bits, zigzag_w, max_quotient).is_err()
        );
    }

    #[test]
    fn golomb_rice_decode_rejects_trailing_zero_byte() {
        let rice_low_bits = 3u32;
        let zigzag_w = 12u32;
        let values = [-1i64, 0, 42];
        let max_quotient = max_quotient_for_values(&values, rice_low_bits, zigzag_w);
        let mut encoded = golomb_rice_encode_vec(&values, rice_low_bits, zigzag_w).unwrap();
        encoded.push(0x00);
        assert!(
            golomb_rice_decode_vec(&encoded, 3, rice_low_bits, zigzag_w, max_quotient).is_err()
        );
    }

    #[test]
    fn golomb_rice_decode_rejects_unary_above_cap_derived_max() {
        let cap = 1008u128;
        let rice_low_bits = wire_rice_low_bits(cap);
        let zigzag_w = golomb_rice_zigzag_width(cap);
        let max_quotient =
            golomb_rice_max_quotient_for_cap(cap, rice_low_bits, zigzag_w).expect("max q");
        assert!(
            max_quotient < 32,
            "test expects cap-derived max below legacy 32"
        );
        let mut writer = BitWriter::default();
        for _ in 0..32 {
            writer.write_bit(true);
        }
        writer.write_bit(false);
        writer.write_bits(0, rice_low_bits);
        let bytes = writer.finish();
        assert!(golomb_rice_decode_vec(&bytes, 1, rice_low_bits, zigzag_w, max_quotient).is_err());
    }

    #[test]
    fn golomb_rice_decode_is_total_on_empty_prefix() {
        assert!(golomb_rice_decode_vec(&[], 1, 0, 4, 0).is_err());
    }

    #[test]
    fn tail_z_planner_cap_low_bits_plus_two_bits_per_coord() {
        assert_eq!(tail_z_planner_bits_per_coord(8), 10);
        assert_eq!(tail_z_planner_bits_per_coord(10), 12);
    }

    #[test]
    fn golomb_rice_rows_encodable_at_wire_low_bits_matches_cap_range() {
        for &cap in &[504u128, 1008] {
            let row = [cap as i32; 4];
            golomb_rice_rows_encodable_at_wire_low_bits(&[row], cap).expect("row encodable");
        }
        assert!(golomb_rice_rows_encodable_at_wire_low_bits(&[[1009i32; 4]], 1008).is_err());
    }

    #[test]
    fn golomb_rice_rows_admit_terminal_wire_rejects_planner_budget_overflow() {
        let cap = 1008u128;
        let row = [[cap as i32; 4]];
        golomb_rice_rows_encodable_at_wire_low_bits(&row, cap).expect("within cap");
        assert!(golomb_rice_rows_admit_terminal_wire(&row, cap).is_err());
    }
}
