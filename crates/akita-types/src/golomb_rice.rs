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

/// Rice parameter `k` from a per-coordinate magnitude scale (e.g. fold `‖z‖_inf` cap).
#[must_use]
pub fn optimal_rice_k(scale: u128) -> u32 {
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

/// Average-case planner model id bound into [`crate::instance_descriptor::FoldLinfProtocolBinding`].
///
/// Bump when recalibrating [`golomb_rice_planner_bits_per_z_coord`].
pub const TAIL_Z_PLANNER_MODEL_ID: u8 = 1;

/// Average-case planner bit budget per `z` coordinate from public Rice `k`.
///
/// Public `(k, W)` still cover every coefficient in `[-cap, cap]`; this budget
/// prices honest witnesses (model v1: `k + 2` bits/coord).
#[must_use]
pub fn golomb_rice_planner_bits_per_z_coord(rice_k: u32) -> usize {
    (rice_k as usize).saturating_add(2)
}

/// Golomb-Rice bit length for one coordinate at public `(k, w)`.
pub fn golomb_rice_bits_for_coord(n: i64, k: u32, w: u32) -> Result<usize, AkitaError> {
    let mut writer = BitWriter::default();
    golomb_rice_encode_one_into(&mut writer, n, k, w)?;
    Ok(writer.bit_len())
}

/// Total Golomb-Rice bits for a vector at public `(k, w)`.
pub fn golomb_rice_total_bits(values: &[i64], k: u32, w: u32) -> Result<usize, AkitaError> {
    values.iter().try_fold(0usize, |acc, &n| {
        golomb_rice_bits_for_coord(n, k, w)?
            .checked_add(acc)
            .ok_or(AkitaError::InvalidSetup(
                "golomb-rice total bits overflow".to_string(),
            ))
    })
}

/// Empirical Rice `k` minimizing total Golomb bits on a realized sample.
#[must_use]
pub fn empirical_optimal_rice_k(values: &[i64], w: u32, k_max: u32) -> u32 {
    (0..=k_max)
        .min_by_key(|&k| golomb_rice_total_bits(values, k, w).unwrap_or(usize::MAX))
        .unwrap_or(0)
}

/// Payload byte length for each Rice `k` in `0..=k_hi` on a realized `z` sample.
pub fn golomb_rice_k_sweep_payload_bytes(
    values: &[i64],
    w: u32,
    k_hi: u32,
) -> Result<Vec<(u32, usize)>, AkitaError> {
    (0..=k_hi)
        .map(|k| {
            let bits = golomb_rice_total_bits(values, k, w)?;
            Ok((k, bits.div_ceil(8)))
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
    /// Security Rice `k` (`optimal_rice_k(cap)`); audit reference for the old cap→k rule.
    pub rice_k_security: u32,
    /// Live Rice `k` on wire ([`crate::tail_golomb_cap_to_k::live_rice_k_for_fold_cap`]).
    pub rice_k_live: u32,
    /// Witness-optimal `k` minimizing total Golomb bits on this sample (not public).
    pub rice_k_empirical: u32,
    pub bits_per_coord_k_security: f64,
    pub bits_per_coord_k_live: f64,
    pub bits_per_coord_k_empirical: f64,
    pub bits_per_coord_packed_digits: f64,
    pub total_bits_k_security: usize,
    pub total_bits_k_live: usize,
    pub total_bits_k_empirical: usize,
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
    let rice_k_security =
        crate::tail_golomb_cap_to_k::security_rice_k_for_fold_cap(witness_linf_cap);
    let rice_k_live = crate::tail_golomb_cap_to_k::live_rice_k_for_fold_cap(witness_linf_cap);
    let k_search_hi = rice_k_security.saturating_add(4);
    let rice_k_empirical = empirical_optimal_rice_k(values, zigzag_w, k_search_hi);

    let mut abs_vals: Vec<u64> = values.iter().map(|&n| n.unsigned_abs()).collect();
    abs_vals.sort_unstable();
    let observed_max_abs = *abs_vals.last().unwrap_or(&0);
    let sum_abs: u128 = abs_vals.iter().map(|&a| u128::from(a)).sum();
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        sum_abs as f64 / values.len() as f64
    };

    let total_bits_k_security = golomb_rice_total_bits(values, rice_k_security, zigzag_w)?;
    let total_bits_k_live = golomb_rice_total_bits(values, rice_k_live, zigzag_w)?;
    let total_bits_k_empirical = golomb_rice_total_bits(values, rice_k_empirical, zigzag_w)?;
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
        rice_k_security,
        rice_k_live,
        rice_k_empirical,
        bits_per_coord_k_security: total_bits_k_security as f64 / n,
        bits_per_coord_k_live: total_bits_k_live as f64 / n,
        bits_per_coord_k_empirical: total_bits_k_empirical as f64 / n,
        bits_per_coord_packed_digits: total_bits_packed_digits as f64 / n,
        total_bits_k_security,
        total_bits_k_live,
        total_bits_k_empirical,
        total_bits_packed_digits,
        actual_payload_bytes,
    })
}

/// Golomb unary quotient for one coefficient at public `(k, w)`.
pub fn golomb_rice_quotient_for_coord(n: i64, k: u32, w: u32) -> Result<u64, AkitaError> {
    let u = zigzag_encode(n, w)?;
    Ok(if k == 0 { u } else { u >> k })
}

/// Whether `n` encodes at `(k, w)` without taking the bounded-unary escape path.
pub fn golomb_rice_coord_encodable_without_escape(
    n: i64,
    k: u32,
    w: u32,
) -> Result<(), AkitaError> {
    let quotient = golomb_rice_quotient_for_coord(n, k, w)?;
    if quotient >= u64::from(GOLOMB_RICE_Q_MAX) {
        return Err(AkitaError::InvalidInput(format!(
            "golomb-rice coefficient {n} needs escape at k={k} (quotient={quotient})"
        )));
    }
    Ok(())
}

/// Whether every centered row encodes at live `(k, W)` derived from fold cap `cap`.
pub fn golomb_rice_rows_encodable_at_live_k<const D: usize>(
    rows: &[[i32; D]],
    cap: u128,
) -> Result<(), AkitaError> {
    if cap == 0 && rows.iter().any(|row| row.iter().any(|&n| n != 0)) {
        return Err(AkitaError::InvalidInput(
            "golomb-rice encodability check at zero cap".to_string(),
        ));
    }
    let k = crate::tail_golomb_cap_to_k::live_rice_k_for_fold_cap(cap);
    let w = golomb_rice_zigzag_width(cap);
    for row in rows {
        for &n in row {
            if i128::from(n).unsigned_abs() > cap {
                return Err(AkitaError::InvalidInput(format!(
                    "centered coefficient {n} exceeds fold grind cap {cap}"
                )));
            }
            golomb_rice_coord_encodable_without_escape(i64::from(n), k, w)?;
        }
    }
    Ok(())
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
        let u = reader.read_bits(w)?;
        if (u >> k) < u64::from(GOLOMB_RICE_Q_MAX) {
            return Err(AkitaError::InvalidProof);
        }
        u
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
    while reader.remaining_bits() > 0 {
        if reader.read_bit()? {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exhaustive_min_sound_k(cap: u128) -> Option<u32> {
        let w = golomb_rice_zigzag_width(cap);
        for k in 0..=20 {
            let mut ok = true;
            for n in -(cap as i64)..=(cap as i64) {
                let encoded = match golomb_rice_encode_vec(&[n], k, w) {
                    Ok(e) => e,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                };
                let decoded = match golomb_rice_decode_vec(&encoded, 1, k, w) {
                    Ok(d) => d,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                };
                if decoded != [n] {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some(k);
            }
        }
        None
    }

    #[test]
    fn min_sound_k_vs_optimal_rice_k_probe() {
        for cap in [504u128, 885, 1008, 1568, 6912] {
            let pub_k = optimal_rice_k(cap);
            let min_k = exhaustive_min_sound_k(cap).expect("min k");
            eprintln!(
                "cap={cap} optimal_rice_k={pub_k} exhaustive_min={min_k} delta={}",
                pub_k.saturating_sub(min_k)
            );
            assert!(pub_k >= min_k, "public k must dominate sound min");
            // `min_k` is always `0` here: escape makes every k decodable. Do not use it as `k_live`.
            assert_eq!(min_k, 0, "cap={cap}");
        }
    }

    #[test]
    fn optimal_rice_k_tracks_per_coefficient_scale() {
        let cap = 6_912u128;
        assert_eq!(optimal_rice_k(cap), 12);
        let level_variance_envelope = 6_189_618u128;
        assert!(
            optimal_rice_k(cap) < optimal_rice_k(level_variance_envelope),
            "per-coordinate k must use fold cap, not the level variance envelope"
        );
    }

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

    fn non_canonical_escape_bytes(u: u64, w: u32) -> Vec<u8> {
        let mut writer = BitWriter::default();
        for _ in 0..GOLOMB_RICE_Q_MAX {
            writer.write_bit(true);
        }
        writer.write_bit(false);
        writer.write_bits(u, w);
        writer.finish()
    }

    #[test]
    fn golomb_rice_decode_rejects_non_canonical_escape() {
        let k = 3u32;
        let w = 12u32;
        let u = 2u64;
        assert!(u >> k < u64::from(GOLOMB_RICE_Q_MAX));
        let bytes = non_canonical_escape_bytes(u, w);
        assert!(golomb_rice_decode_vec(&bytes, 1, k, w).is_err());

        let k0 = 0u32;
        let w0 = 16u32;
        let u0 = 5u64;
        assert!(u0 < u64::from(GOLOMB_RICE_Q_MAX));
        let bytes0 = non_canonical_escape_bytes(u0, w0);
        assert!(golomb_rice_decode_vec(&bytes0, 1, k0, w0).is_err());
    }

    #[test]
    fn golomb_rice_planner_model_v1_is_k_plus_two() {
        assert_eq!(golomb_rice_planner_bits_per_z_coord(8), 10);
        assert_eq!(golomb_rice_planner_bits_per_z_coord(10), 12);
    }

    #[test]
    fn golomb_rice_rows_encodable_at_live_k_matches_cap_range() {
        for &cap in &[504u128, 1008] {
            let k = crate::tail_golomb_cap_to_k::live_rice_k_for_fold_cap(cap);
            let w = golomb_rice_zigzag_width(cap);
            let cap_i64 = cap as i64;
            for n in -cap_i64..=cap_i64 {
                golomb_rice_coord_encodable_without_escape(n, k, w)
                    .unwrap_or_else(|e| panic!("cap={cap} n={n}: {e}"));
            }
            let row = [cap_i64 as i32; 4];
            golomb_rice_rows_encodable_at_live_k(&[row], cap).expect("row encodable");
        }
        assert!(golomb_rice_rows_encodable_at_live_k(&[[1009i32; 4]], 1008).is_err());
    }
}
