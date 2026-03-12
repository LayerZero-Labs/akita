//! Labrador challenge sampler (C-parity oriented).
//!
//! This ports the `polyvec_challenge` rejection sampler from the C reference.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::labrador::guardrails::{
    checked_add, checked_mul, ensure_power_of_two, ensure_temp_allocation_limit,
    LABRADOR_MAX_CHALLENGE_POLYS,
};
use crate::{CanonicalField, FieldCore, FromSmallInt};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake128;
use std::sync::OnceLock;

/// Number of `±1` coefficients in a challenge polynomial.
pub const LABRADOR_TAU1: usize = 32;
/// Number of `±2` coefficients in a challenge polynomial.
pub const LABRADOR_TAU2: usize = 8;
/// Operator norm bound used by C's challenge rejection sampler.
pub const LABRADOR_CHALLENGE_OPNORM_BOUND: f64 = 14.0;

const SHAKE128_RATE: usize = 168;
const SINGLE_CHALLENGE_BLOCKS: usize = 2;
const SINGLE_CHALLENGE_BLOCK_BYTES: usize = SINGLE_CHALLENGE_BLOCKS * SHAKE128_RATE;

/// Sample Labrador challenge polynomials as signed coefficient arrays.
///
/// The output follows C `polyvec_challenge`: each polynomial has exactly
/// `LABRADOR_TAU1` coefficients in `{±1}`, `LABRADOR_TAU2` coefficients in
/// `{±2}`, all other coefficients 0, and must satisfy operator-norm bound.
///
/// # Errors
///
/// Returns an error if ring parameters are incompatible with the C algorithm.
pub fn sample_labrador_challenge_coeffs<const D: usize>(
    len: usize,
    seed: &[u8; 16],
    nonce: u64,
) -> Result<Vec<[i16; D]>, HachiError> {
    validate_challenge_params::<D>()?;
    if len > LABRADOR_MAX_CHALLENGE_POLYS {
        return Err(HachiError::InvalidInput(format!(
            "requested too many challenge polynomials: {len} (max {LABRADOR_MAX_CHALLENGE_POLYS})"
        )));
    }

    let mut xof = Shake128::default();
    xof.update(seed);
    xof.update(&nonce.to_le_bytes());
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
        let bytes = checked_mul(17, SHAKE128_RATE, "challenge block bytes")?;
        ensure_temp_allocation_limit(bytes, "challenge sampler")?;
        let mut buf = vec![0u8; bytes];
        reader.read(&mut buf);
        let produced = consume_challenge_buffer::<D>(&mut out, 10, &buf);
        remaining -= produced;
    }

    while remaining > 0 {
        let scaled = checked_mul(remaining, 17, "scaled tail blocks numerator")?;
        let scaled = checked_add(scaled, 9, "tail blocks numerator rounding")?;
        let blocks = scaled / 10;
        let bytes = checked_mul(blocks, SHAKE128_RATE, "tail block bytes")?;
        ensure_temp_allocation_limit(bytes, "challenge sampler tail")?;
        let mut buf = vec![0u8; bytes];
        reader.read(&mut buf);
        let produced = consume_challenge_buffer::<D>(&mut out, remaining, &buf);
        remaining -= produced;
    }

    Ok(out)
}

/// Sample Labrador challenge polynomials as dense ring elements.
///
/// # Errors
///
/// Returns an error if parameter checks fail.
pub fn sample_labrador_challenges<F, const D: usize>(
    len: usize,
    seed: &[u8; 16],
    nonce: u64,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let coeffs = sample_labrador_challenge_coeffs::<D>(len, seed, nonce)?;
    Ok(coeffs
        .into_iter()
        .map(|poly| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|i| F::from_i64(poly[i] as i64)))
        })
        .collect())
}

fn validate_challenge_params<const D: usize>() -> Result<(), HachiError> {
    ensure_power_of_two(D, "challenge sampler degree D")?;
    if D > 256 {
        return Err(HachiError::InvalidInput(format!(
            "challenge sampler expects D <= 256, got {D}"
        )));
    }
    if LABRADOR_TAU1 + LABRADOR_TAU2 > D {
        return Err(HachiError::InvalidInput(format!(
            "tau1 + tau2 exceeds ring degree: {LABRADOR_TAU1} + {LABRADOR_TAU2} > {D}"
        )));
    }
    Ok(())
}

fn consume_challenge_buffer<const D: usize>(
    out: &mut Vec<[i16; D]>,
    target_len: usize,
    buf: &[u8],
) -> usize {
    let sign_bytes = (LABRADOR_TAU1 + LABRADOR_TAU2).div_ceil(8);
    let min_bytes = LABRADOR_TAU1 + LABRADOR_TAU2 + sign_bytes;
    let mut produced = 0usize;
    let mut cursor = 0usize;

    while produced < target_len && cursor <= buf.len().saturating_sub(min_bytes) {
        let mut signs = 0u64;
        for k in 0..sign_bytes {
            signs |= (buf[cursor] as u64) << (8 * k);
            cursor += 1;
        }

        let mut poly = [0i16; D];
        let mut k = D - LABRADOR_TAU1 - LABRADOR_TAU2;
        while k < D && cursor < buf.len() {
            let b = (buf[cursor] as usize) & (D - 1);
            cursor += 1;
            if b <= k {
                poly[k] = poly[b];
                let mut value = if k < D - LABRADOR_TAU2 { 1 } else { 2 };
                if (signs & 1) == 1 {
                    value = -value;
                }
                poly[b] = value;
                signs >>= 1;
                k += 1;
            }
        }

        if k == D && challenge_operator_norm::<D>(&poly) <= LABRADOR_CHALLENGE_OPNORM_BOUND {
            out.push(poly);
            produced += 1;
        }
    }

    produced
}

struct ChallengeOpNormTable {
    cos: Vec<f64>,
    sin: Vec<f64>,
}

fn build_challenge_opnorm_table(d: usize) -> ChallengeOpNormTable {
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
    ChallengeOpNormTable { cos, sin }
}

fn challenge_opnorm_table<const D: usize>() -> &'static ChallengeOpNormTable {
    match D {
        1 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(1))
        }
        2 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(2))
        }
        4 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(4))
        }
        8 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(8))
        }
        16 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(16))
        }
        32 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(32))
        }
        64 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(64))
        }
        128 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(128))
        }
        256 => {
            static TABLE: OnceLock<ChallengeOpNormTable> = OnceLock::new();
            TABLE.get_or_init(|| build_challenge_opnorm_table(256))
        }
        _ => panic!("unsupported challenge sampler degree {D}"),
    }
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

fn challenge_operator_norm<const D: usize>(coeffs: &[i16; D]) -> f64 {
    let table = challenge_opnorm_table::<D>();
    let mut support_idx = [0usize; LABRADOR_TAU1 + LABRADOR_TAU2];
    let mut support_coeff = [0.0f64; LABRADOR_TAU1 + LABRADOR_TAU2];
    let mut support_len = 0usize;
    for (idx, &coeff) in coeffs.iter().enumerate() {
        if coeff == 0 {
            continue;
        }
        if support_len == support_idx.len() {
            #[cfg(test)]
            {
                return challenge_operator_norm_dense_reference(coeffs);
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

    let mut max_norm = 0.0f64;
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
    use crate::algebra::fields::Fp32;

    type F = Fp32<4294967197>;
    const D: usize = 64;

    // Fixed test seeds and nonces for deterministic replay.
    const TEST_SEED_A: [u8; 16] = [7u8; 16];
    const TEST_SEED_B: [u8; 16] = [11u8; 16];
    const TEST_SEED_C: [u8; 16] = [5u8; 16];
    const TEST_NONCE_A: u64 = 9;
    const TEST_NONCE_B: u64 = 17;
    const TEST_NONCE_C: u64 = 4;
    const TEST_NONCE_REF: u64 = 7;

    #[test]
    fn challenge_sampler_is_deterministic() {
        let c1 = sample_labrador_challenge_coeffs::<D>(3, &TEST_SEED_A, TEST_NONCE_A).unwrap();
        let c2 = sample_labrador_challenge_coeffs::<D>(3, &TEST_SEED_A, TEST_NONCE_A).unwrap();
        assert_eq!(c1, c2);
    }

    #[test]
    fn challenge_sampler_obeys_operator_norm_bound() {
        let samples = sample_labrador_challenge_coeffs::<D>(8, &TEST_SEED_B, TEST_NONCE_B).unwrap();
        assert_eq!(samples.len(), 8);
        for poly in &samples {
            assert!(challenge_operator_norm(poly) <= LABRADOR_CHALLENGE_OPNORM_BOUND);
        }
    }

    #[test]
    fn challenge_sampler_supports_dense_ring_conversion() {
        let dense = sample_labrador_challenges::<F, D>(2, &TEST_SEED_C, TEST_NONCE_C).unwrap();
        assert_eq!(dense.len(), 2);
    }

    #[test]
    fn challenge_sampler_matches_transliterated_reference_vector() {
        // Captured from the C-reference algorithm semantics (`polyvec_challenge`)
        // for seed = [0,1,2,...,15], nonce = 7, len = 1.
        let seed: [u8; 16] = std::array::from_fn(|i| i as u8);
        let coeffs = sample_labrador_challenge_coeffs::<D>(1, &seed, TEST_NONCE_REF).unwrap();
        let got = coeffs[0];
        let expected: [i16; D] = [
            1, 1, 0, 1, 0, 0, 2, -1, 0, 0, 2, 1, 1, -1, -1, 1, -2, 0, 1, 0, -1, -1, 1, 0, 1, -1, 1,
            1, 0, -1, 0, -1, 2, 1, 1, -1, -2, 0, 0, 1, 0, 0, 1, 1, -2, 1, 0, 0, 0, 0, 0, 0, 1, 0,
            -1, -1, 2, -1, 0, 1, -2, 1, 0, 0,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn sparse_operator_norm_matches_dense_reference() {
        for nonce in [1u64, 3, 7, 9, 17, 29] {
            let polys =
                sample_labrador_challenge_coeffs::<D>(6, &TEST_SEED_A, nonce).expect("sample");
            for poly in polys {
                let sparse = challenge_operator_norm(&poly);
                let dense = challenge_operator_norm_dense_reference(&poly);
                assert_eq!(sparse.to_bits(), dense.to_bits());
                assert_eq!(
                    sparse <= LABRADOR_CHALLENGE_OPNORM_BOUND,
                    dense <= LABRADOR_CHALLENGE_OPNORM_BOUND
                );
            }
        }
    }
}
