//! Sparse ring-element rejection sampler (C-parity oriented).
//!
//! This ports the `polyvec_challenge` rejection sampler from the C reference.

use crate::{CanonicalField, FieldCore, FromSmallInt};
use akita_algebra::ring::CyclotomicRing;
use akita_algebra::SparseChallenge;
use akita_field::HachiError;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake128;
use std::sync::OnceLock;

const MAX_CHALLENGE_SUPPORT: usize = 32 + 8;
const CHALLENGE_OPNORM_BOUND_SQ: f64 = 14.0 * 14.0;
const MAX_CHALLENGE_POLYS: usize = 1 << 12;
const MAX_TEMP_BYTES: usize = 1 << 27; // 128 MiB
const SHAKE128_RATE: usize = 168;
const SINGLE_CHALLENGE_BLOCKS: usize = 2;
const SINGLE_CHALLENGE_BLOCK_BYTES: usize = SINGLE_CHALLENGE_BLOCKS * SHAKE128_RATE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChallengeProfile {
    pub tau1: usize,
    pub tau2: usize,
}

pub(crate) const fn challenge_profile(d: usize) -> ChallengeProfile {
    match d {
        256 => ChallengeProfile { tau1: 23, tau2: 0 },
        128 => ChallengeProfile { tau1: 31, tau2: 0 },
        _ => ChallengeProfile { tau1: 32, tau2: 8 },
    }
}

/// Sample challenge polynomials as signed coefficient arrays.
///
/// Each polynomial has exactly the per-ring-dimension `(tau1, tau2)` support
/// profile in `{±1}` and `{±2}`, all other coefficients 0, and must satisfy
/// operator-norm bound.
///
/// # Errors
///
/// Returns an error if ring parameters are incompatible with the algorithm.
pub fn sample_challenge_coeffs<const D: usize>(
    len: usize,
    seed: &[u8; 16],
    stream_id: u64,
) -> Result<Vec<[i16; D]>, HachiError> {
    validate_challenge_params::<D>()?;
    if len > MAX_CHALLENGE_POLYS {
        return Err(HachiError::InvalidInput(format!(
            "requested too many challenge polynomials: {len} (max {MAX_CHALLENGE_POLYS})"
        )));
    }

    let mut xof = Shake128::default();
    xof.update(seed);
    xof.update(&stream_id.to_le_bytes());
    let mut reader = xof.finalize_xof();

    let mut out = Vec::with_capacity(len);
    let mut remaining = len;

    if remaining == 1 {
        while remaining > 0 {
            let mut buf = [0u8; SINGLE_CHALLENGE_BLOCK_BYTES];
            reader.read(&mut buf);
            let produced = consume_challenge_buffer::<D>(&mut out, remaining, &buf);
            remaining -= produced;
        }
        return Ok(out);
    }

    while remaining >= 10 {
        let bytes = checked_mul(17, SHAKE128_RATE)?;
        ensure_temp_allocation_limit(bytes)?;
        let mut buf = vec![0u8; bytes];
        reader.read(&mut buf);
        let produced = consume_challenge_buffer::<D>(&mut out, 10, &buf);
        remaining -= produced;
    }

    while remaining > 0 {
        let scaled = checked_mul(remaining, 17)?;
        let scaled = checked_add(scaled, 9)?;
        let blocks = scaled / 10;
        let bytes = checked_mul(blocks, SHAKE128_RATE)?;
        ensure_temp_allocation_limit(bytes)?;
        let mut buf = vec![0u8; bytes];
        reader.read(&mut buf);
        let produced = consume_challenge_buffer::<D>(&mut out, remaining, &buf);
        remaining -= produced;
    }

    Ok(out)
}

/// Sample challenge polynomials as dense ring elements.
///
/// # Errors
///
/// Returns an error if parameter checks fail.
pub fn sample_challenges<F, const D: usize>(
    len: usize,
    seed: &[u8; 16],
    stream_id: u64,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let coeffs = sample_challenge_coeffs::<D>(len, seed, stream_id)?;
    Ok(coeffs
        .into_iter()
        .map(|poly| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|i| F::from_i64(poly[i] as i64)))
        })
        .collect())
}

/// Sample challenge polynomials as sparse ring elements.
///
/// # Errors
///
/// Returns an error if parameter checks fail.
pub fn sample_sparse_challenges<const D: usize>(
    len: usize,
    seed: &[u8; 16],
    stream_id: u64,
) -> Result<Vec<SparseChallenge>, HachiError> {
    let profile = challenge_profile(D);
    let support_len = profile.tau1 + profile.tau2;
    Ok(sample_challenge_coeffs::<D>(len, seed, stream_id)?
        .into_iter()
        .map(|poly| {
            let mut positions = Vec::with_capacity(support_len);
            let mut coeffs = Vec::with_capacity(support_len);
            for (idx, coeff) in poly.into_iter().enumerate() {
                if coeff != 0 {
                    positions.push(idx as u32);
                    coeffs.push(coeff);
                }
            }
            SparseChallenge { positions, coeffs }
        })
        .collect())
}

fn validate_challenge_params<const D: usize>() -> Result<(), HachiError> {
    let profile = challenge_profile(D);
    if !D.is_power_of_two() {
        return Err(HachiError::InvalidInput(format!(
            "challenge sampler expects D to be a power of two, got {D}"
        )));
    }
    if D > 256 {
        return Err(HachiError::InvalidInput(format!(
            "challenge sampler expects D <= 256, got {D}"
        )));
    }
    if profile.tau1 + profile.tau2 > D {
        return Err(HachiError::InvalidInput(format!(
            "tau1 + tau2 exceeds ring degree: {} + {} > {D}",
            profile.tau1, profile.tau2
        )));
    }
    Ok(())
}

fn checked_mul(a: usize, b: usize) -> Result<usize, HachiError> {
    a.checked_mul(b)
        .ok_or_else(|| HachiError::InvalidInput("overflow in challenge sampler".to_string()))
}

fn checked_add(a: usize, b: usize) -> Result<usize, HachiError> {
    a.checked_add(b)
        .ok_or_else(|| HachiError::InvalidInput("overflow in challenge sampler".to_string()))
}

fn ensure_temp_allocation_limit(bytes: usize) -> Result<(), HachiError> {
    if bytes > MAX_TEMP_BYTES {
        return Err(HachiError::InvalidInput(format!(
            "challenge sampler temporary allocation too large: {bytes} bytes (max {MAX_TEMP_BYTES})"
        )));
    }
    Ok(())
}

fn consume_challenge_buffer<const D: usize>(
    out: &mut Vec<[i16; D]>,
    target_len: usize,
    buf: &[u8],
) -> usize {
    let profile = challenge_profile(D);
    let support_len = profile.tau1 + profile.tau2;
    let sign_bytes = support_len.div_ceil(8);
    let min_bytes = support_len + sign_bytes;
    let mut produced = 0usize;
    let mut cursor = 0usize;

    while produced < target_len && cursor <= buf.len().saturating_sub(min_bytes) {
        let mut signs = 0u64;
        for k in 0..sign_bytes {
            signs |= (buf[cursor] as u64) << (8 * k);
            cursor += 1;
        }

        let mut poly = [0i16; D];
        let mut k = D - support_len;
        while k < D && cursor < buf.len() {
            let b = (buf[cursor] as usize) & (D - 1);
            cursor += 1;
            if b <= k {
                poly[k] = poly[b];
                let mut value = if k < D - profile.tau2 { 1 } else { 2 };
                if (signs & 1) == 1 {
                    value = -value;
                }
                poly[b] = value;
                signs >>= 1;
                k += 1;
            }
        }

        if k == D && challenge_operator_norm_with_bound::<D>(&poly, CHALLENGE_OPNORM_BOUND_SQ) {
            out.push(poly);
            produced += 1;
        }
    }

    produced
}

struct OpNormTable {
    cos: Vec<f64>,
    sin: Vec<f64>,
}

fn build_opnorm_table(d: usize) -> OpNormTable {
    let mut cos = Vec::with_capacity(d * d);
    let mut sin = Vec::with_capacity(d * d);
    let d_f = d as f64;
    for i in 0..d {
        let theta = ((2 * i + 1) as f64) * std::f64::consts::PI / d_f;
        for j in 0..d {
            let angle = theta * (j as f64);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    OpNormTable { cos, sin }
}

fn opnorm_table<const D: usize>() -> &'static OpNormTable {
    match D {
        1 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(1))
        }
        2 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(2))
        }
        4 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(4))
        }
        8 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(8))
        }
        32 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(32))
        }
        64 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(64))
        }
        128 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(128))
        }
        256 => {
            static TABLE: OnceLock<OpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_opnorm_table(256))
        }
        _ => panic!("unsupported challenge sampler degree {D}"),
    }
}

fn challenge_operator_norm_with_bound<const D: usize>(coeffs: &[i16; D], bound_sq: f64) -> bool {
    let support_limit = challenge_profile(D).tau1 + challenge_profile(D).tau2;
    let table = opnorm_table::<D>();
    let mut support_idx = [0usize; MAX_CHALLENGE_SUPPORT];
    let mut support_coeff = [0.0f64; MAX_CHALLENGE_SUPPORT];
    let mut support_len = 0usize;
    for (idx, &coeff) in coeffs.iter().enumerate() {
        if coeff == 0 {
            continue;
        }
        if support_len == support_limit {
            #[cfg(test)]
            {
                let norm = challenge_operator_norm_dense_reference::<D>(coeffs);
                return norm * norm <= bound_sq;
            }
            #[cfg(not(test))]
            {
                panic!("challenge support exceeded expected sparsity");
            }
        }
        support_idx[support_len] = idx;
        support_coeff[support_len] = coeff as f64;
        support_len += 1;
    }

    for i in 0..D {
        let row_base = i * D;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for idx in 0..support_len {
            let coeff = support_coeff[idx];
            let col = support_idx[idx];
            re += coeff * table.cos[row_base + col];
            im += coeff * table.sin[row_base + col];
        }
        if re * re + im * im > bound_sq {
            return false;
        }
    }

    true
}

#[cfg(test)]
fn challenge_operator_norm_dense_reference<const D: usize>(coeffs: &[i16; D]) -> f64 {
    let mut max_norm = 0.0f64;
    let d_f = D as f64;
    for i in 0..D {
        let theta = ((2 * i + 1) as f64) * std::f64::consts::PI / d_f;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (j, &coeff) in coeffs.iter().enumerate() {
            let angle = theta * (j as f64);
            let c = coeff as f64;
            re += c * angle.cos();
            im += c * angle.sin();
        }
        let norm = (re * re + im * im).sqrt();
        if norm > max_norm {
            max_norm = norm;
        }
    }
    max_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::fields::Fp32;

    type F = Fp32<4294967197>;
    const D: usize = 64;

    const TEST_SEED_A: [u8; 16] = [7u8; 16];
    const TEST_SEED_B: [u8; 16] = [11u8; 16];
    const TEST_SEED_C: [u8; 16] = [5u8; 16];
    const TEST_STREAM_ID_A: u64 = 9;
    const TEST_STREAM_ID_B: u64 = 17;
    const TEST_STREAM_ID_C: u64 = 4;
    const TEST_STREAM_ID_REF: u64 = 7;

    const CHALLENGE_OPNORM_BOUND: f64 = 14.0;

    #[test]
    fn challenge_sampler_is_deterministic() {
        let c1 = sample_challenge_coeffs::<D>(3, &TEST_SEED_A, TEST_STREAM_ID_A).unwrap();
        let c2 = sample_challenge_coeffs::<D>(3, &TEST_SEED_A, TEST_STREAM_ID_A).unwrap();
        assert_eq!(c1, c2);
    }

    #[test]
    fn challenge_sampler_obeys_operator_norm_bound() {
        let samples = sample_challenge_coeffs::<D>(8, &TEST_SEED_B, TEST_STREAM_ID_B).unwrap();
        assert_eq!(samples.len(), 8);
        for poly in &samples {
            let norm = challenge_operator_norm_dense_reference(poly);
            assert!(norm <= CHALLENGE_OPNORM_BOUND);
        }
    }

    #[test]
    fn challenge_sampler_supports_dense_ring_conversion() {
        let dense = sample_challenges::<F, D>(2, &TEST_SEED_C, TEST_STREAM_ID_C).unwrap();
        assert_eq!(dense.len(), 2);
    }

    #[test]
    fn challenge_sampler_matches_transliterated_reference_vector() {
        let seed: [u8; 16] = std::array::from_fn(|i| i as u8);
        let coeffs = sample_challenge_coeffs::<D>(1, &seed, TEST_STREAM_ID_REF).unwrap();
        let got = coeffs[0];
        let expected: [i16; D] = [
            1, 1, 0, 1, 0, 0, 2, -1, 0, 0, 2, 1, 1, -1, -1, 1, -2, 0, 1, 0, -1, -1, 1, 0, 1, -1, 1,
            1, 0, -1, 0, -1, 2, 1, 1, -1, -2, 0, 0, 1, 0, 0, 1, 1, -2, 1, 0, 0, 0, 0, 0, 0, 1, 0,
            -1, -1, 2, -1, 0, 1, -2, 1, 0, 0,
        ];
        assert_eq!(got, expected);
    }
}
