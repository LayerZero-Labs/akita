//! Johnson-Lindenstrauss helpers for Labrador reduction.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_JL_NONCE_RETRIES;
use crate::protocol::labrador::types::LabradorWitness;
use crate::protocol::transcript::{labels, Transcript};
use crate::{CanonicalField, FieldCore};
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake128;

const JL_ROWS: usize = 256;
const JL_XOF_DOMAIN: &[u8] = b"hachi/labrador-jl-matrix";

fn expand_jl_seed(seed: &[u8], len: usize) -> Vec<u8> {
    let mut xof = Shake128::default();
    xof.update(JL_XOF_DOMAIN);
    xof.update(seed);
    let mut reader = xof.finalize_xof();
    let mut out = vec![0u8; len];
    reader.read(&mut out);
    out
}

fn jl_row_bytes(cols: usize) -> Result<usize, HachiError> {
    if cols == 0 {
        return Err(HachiError::InvalidInput(
            "JL matrix requires non-zero column count".to_string(),
        ));
    }
    Ok((cols * 2).div_ceil(8))
}

pub(crate) fn replay_nonce_search_seed<F, T>(
    transcript: &mut T,
    jl_nonce: u64,
    cols: usize,
) -> Result<(usize, [u8; 32]), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if !(1..=LABRADOR_MAX_JL_NONCE_RETRIES).contains(&jl_nonce) {
        return Err(HachiError::InvalidInput(format!(
            "JL nonce out of range: {jl_nonce}"
        )));
    }
    let row_bytes = jl_row_bytes(cols)?;
    transcript.append_bytes(labels::ABSORB_LABRADOR_JL_NONCE, &jl_nonce.to_le_bytes());
    let seed_vec = transcript.challenge_bytes(labels::CHALLENGE_LABRADOR_JL_SEED, 32);
    let seed: [u8; 32] = seed_vec
        .try_into()
        .map_err(|_| HachiError::InvalidInput("JL seed length mismatch".to_string()))?;
    Ok((row_bytes, seed))
}

fn centered_from_canonical(
    canonical: u128,
    modulus: u128,
    half_modulus: u128,
) -> Result<i128, HachiError> {
    let magnitude = centered_magnitude(canonical, modulus, half_modulus);
    let magnitude = i128::try_from(magnitude).map_err(|_| {
        HachiError::InvalidInput("JL centered coefficient exceeds i128 range".to_string())
    })?;
    Ok(if canonical > half_modulus {
        -magnitude
    } else {
        magnitude
    })
}

fn centered_magnitude(canonical: u128, modulus: u128, half_modulus: u128) -> u128 {
    if canonical > half_modulus {
        modulus - canonical
    } else {
        canonical
    }
}

enum CenteredWitness<const D: usize> {
    I64 {
        coeffs: Vec<i64>,
    },
    I128 {
        rings: Vec<[i128; D]>,
        sum_abs: u128,
    },
}

impl<const D: usize> CenteredWitness<D> {
    fn ring_len(&self) -> usize {
        match self {
            Self::I64 { coeffs, .. } => coeffs.len() / D,
            Self::I128 { rings, .. } => rings.len(),
        }
    }
}

#[inline]
fn jl_pair_to_sign(pair: u8) -> i8 {
    ((pair == 0b11) as i8) - ((pair == 0b00) as i8)
}

#[inline]
fn jl_pair_at(row: &[u8], col: usize) -> u8 {
    let shift = (col & 0b11) << 1;
    (row[col >> 2] >> shift) & 0b11
}

#[tracing::instrument(skip_all, name = "labrador::center_witness")]
fn center_witness_by_ring<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> Result<CenteredWitness<D>, HachiError> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let total_rings: usize = witness.rows().iter().map(Vec::len).sum();

    let mut requires_i128 = false;
    'detect_width: for row in witness.rows() {
        for ring in row {
            for coeff in ring.coefficients() {
                let canonical = coeff.to_canonical_u128();
                let magnitude = centered_magnitude(canonical, q, half_q);
                if magnitude > i64::MAX as u128 {
                    requires_i128 = true;
                    break 'detect_width;
                }
            }
        }
    }

    if requires_i128 {
        let mut centered = Vec::with_capacity(total_rings);
        let mut sum_abs = 0u128;
        for row in witness.rows() {
            for ring in row {
                let mut coeffs = [0i128; D];
                for (idx, coeff) in ring.coefficients().iter().enumerate() {
                    coeffs[idx] = centered_from_canonical(coeff.to_canonical_u128(), q, half_q)?;
                    sum_abs = sum_abs.saturating_add(coeffs[idx].unsigned_abs());
                }
                centered.push(coeffs);
            }
        }
        Ok(CenteredWitness::I128 {
            rings: centered,
            sum_abs,
        })
    } else {
        let mut centered = Vec::with_capacity(total_rings * D);
        for row in witness.rows() {
            for ring in row {
                for coeff in ring.coefficients() {
                    let centered_i128 =
                        centered_from_canonical(coeff.to_canonical_u128(), q, half_q)?;
                    centered.push(i64::try_from(centered_i128).map_err(|_| {
                        HachiError::InvalidInput(
                            "JL centered coefficient unexpectedly exceeds i64 range".to_string(),
                        )
                    })?);
                }
            }
        }
        Ok(CenteredWitness::I64 { coeffs: centered })
    }
}

/// Packed ternary JL matrix with entries in `{-1, 0, +1}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorJlMatrix {
    cols: usize,
    row_bytes: usize,
    packed_rows: Vec<u8>,
}

impl LabradorJlMatrix {
    /// Number of columns in each JL row.
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn is_well_formed(&self) -> bool {
        self.cols > 0 && self.packed_rows.len() == JL_ROWS * self.row_bytes
    }

    pub(crate) fn row_bytes(&self, row_idx: usize) -> &[u8] {
        debug_assert!(row_idx < JL_ROWS);
        let start = row_idx * self.row_bytes;
        &self.packed_rows[start..start + self.row_bytes]
    }

    #[cfg(test)]
    fn from_sign_rows(signs: Vec<Vec<i8>>) -> Result<Self, HachiError> {
        if signs.len() != JL_ROWS {
            return Err(HachiError::InvalidInput(format!(
                "JL matrix requires exactly {JL_ROWS} rows"
            )));
        }
        let cols = signs.first().map_or(0, Vec::len);
        if cols == 0 {
            return Err(HachiError::InvalidInput(
                "JL matrix requires non-zero column count".to_string(),
            ));
        }
        if signs.iter().any(|row| row.len() != cols) {
            return Err(HachiError::InvalidInput(
                "JL matrix row length mismatch".to_string(),
            ));
        }

        let row_bytes = (cols * 2).div_ceil(8);
        let mut packed_rows = vec![0u8; JL_ROWS * row_bytes];
        for (row_idx, row) in signs.iter().enumerate() {
            let start = row_idx * row_bytes;
            for (col_idx, &sign) in row.iter().enumerate() {
                let pair = match sign {
                    -1 => 0b00,
                    0 => 0b01,
                    1 => 0b11,
                    _ => {
                        return Err(HachiError::InvalidInput(
                            "JL matrix entries must be in {-1, 0, +1}".to_string(),
                        ))
                    }
                };
                packed_rows[start + (col_idx >> 2)] |= pair << ((col_idx & 0b11) << 1);
            }
        }

        Ok(Self {
            cols,
            row_bytes,
            packed_rows,
        })
    }

    #[cfg(test)]
    fn sign_at(&self, row_idx: usize, col_idx: usize) -> Option<i8> {
        if row_idx >= JL_ROWS || col_idx >= self.cols {
            return None;
        }
        Some(jl_pair_to_sign(jl_pair_at(
            self.row_bytes(row_idx),
            col_idx,
        )))
    }

    /// Squeeze a JL matrix directly from the transcript.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is zero.
    #[tracing::instrument(skip_all, name = "labrador::jl_matrix_generate")]
    pub fn generate<F, T>(transcript: &mut T, cols: usize) -> Result<Self, HachiError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        let row_bytes = jl_row_bytes(cols)?;
        let total_bytes = JL_ROWS * row_bytes;
        let seed = transcript.challenge_bytes(labels::CHALLENGE_LABRADOR_JL_SEED, 32);
        let packed_rows = expand_jl_seed(&seed, total_bytes);
        Ok(Self {
            cols,
            row_bytes,
            packed_rows,
        })
    }

    /// Reconstruct the accepted JL matrix from the prover-chosen nonce.
    ///
    /// The prover now performs nonce search on cloned transcript states and
    /// only commits the accepted nonce back into the real transcript. The
    /// verifier therefore absorbs exactly that accepted nonce once.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is zero or `jl_nonce` is out of range.
    #[tracing::instrument(skip_all, name = "labrador::jl_matrix_replay")]
    pub fn replay_nonce_search<F, T>(
        transcript: &mut T,
        jl_nonce: u64,
        cols: usize,
    ) -> Result<Self, HachiError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        let (_row_bytes, seed) = replay_nonce_search_seed::<F, T>(transcript, jl_nonce, cols)?;
        let total_bytes = JL_ROWS * jl_row_bytes(cols)?;
        let packed_rows = expand_jl_seed(&seed, total_bytes);
        Ok(Self {
            cols,
            row_bytes: jl_row_bytes(cols)?,
            packed_rows,
        })
    }
}

/// Project a witness into 256 JL coordinates and return the nonce used.
///
/// Each nonce attempt runs on a cloned transcript. Only the accepted nonce
/// is committed to the real transcript, which keeps verifier replay bounded
/// and prevents rejected attempts from perturbing later challenges.
///
/// # Errors
///
/// Returns an error if the witness is empty or if no valid projection is found
/// within the nonce search limit.
#[tracing::instrument(skip_all, name = "labrador::project")]
pub fn project<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    transcript: &mut T,
) -> Result<([i64; 256], u64, LabradorJlMatrix), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let total_coeffs: usize = witness.rows().iter().map(|row| row.len() * D).sum();
    if total_coeffs == 0 {
        return Err(HachiError::InvalidInput(
            "cannot JL-project empty witness".to_string(),
        ));
    }
    let centered_witness = center_witness_by_ring(witness)?;
    if centered_witness.ring_len() * D != total_coeffs {
        return Err(HachiError::InvalidInput(
            "centered witness length mismatch".to_string(),
        ));
    }

    let witness_norm: u128 = witness.norm();
    let norm_bound = 256u128.saturating_mul(witness_norm);
    let component_bound = next_power_of_two_u64(4.0 * (witness_norm as f64).sqrt());

    for nonce in 1..=LABRADOR_MAX_JL_NONCE_RETRIES {
        let mut nonce_transcript = transcript.clone();
        nonce_transcript.append_bytes(labels::ABSORB_LABRADOR_JL_NONCE, &nonce.to_le_bytes());
        let matrix = LabradorJlMatrix::generate::<F, T>(&mut nonce_transcript, total_coeffs)?;
        if let Some(proj) = project_streaming::<D>(&matrix, &centered_witness, total_coeffs) {
            if proj.iter().any(|&p| p.unsigned_abs() >= component_bound) {
                continue;
            }
            let proj_norm: u128 = proj.iter().fold(0u128, |acc, &p| {
                acc + p.unsigned_abs() as u128 * p.unsigned_abs() as u128
            });
            if proj_norm > norm_bound {
                continue;
            }
            *transcript = nonce_transcript;
            return Ok((proj, nonce, matrix));
        }
    }
    Err(HachiError::InvalidInput(format!(
        "failed JL projection nonce search after {LABRADOR_MAX_JL_NONCE_RETRIES} attempts"
    )))
}

fn next_power_of_two_u64(x: f64) -> u64 {
    if x <= 1.0 {
        return 1;
    }
    let bits = x.log2().ceil() as u32;
    if bits >= 64 {
        return u64::MAX;
    }
    1u64 << bits
}

/// Collapse a JL projection with challenge coefficients.
///
/// Returns the linear target value `sum_i alpha[i] * projection[i]`.
pub fn collapse(projection: &[i64; 256], alpha: &[i64; 256]) -> i64 {
    projection
        .iter()
        .zip(alpha.iter())
        .fold(0i128, |acc, (&p, &a)| acc + (p as i128) * (a as i128))
        .clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Zero out a polynomial constant term for proof transmission.
///
/// Returns the modified polynomial and the removed constant term.
pub fn zero_constant_term_for_proof<F: FieldCore, const D: usize>(
    mut poly: CyclotomicRing<F, D>,
) -> (CyclotomicRing<F, D>, F) {
    let coeffs = poly.coefficients_mut();
    let c0 = coeffs[0];
    coeffs[0] = F::zero();
    (poly, c0)
}

/// Restore a polynomial constant term during verifier-side reduction.
pub fn restore_constant_term<F: FieldCore, const D: usize>(
    mut transmitted: CyclotomicRing<F, D>,
    constant: F,
) -> CyclotomicRing<F, D> {
    transmitted.coefficients_mut()[0] = constant;
    transmitted
}

/// Compute the JL projection by streaming over witness coefficients without
/// materializing the full flattened vector.
#[inline]
fn project_row_i64(row: &[u8], coeffs: &[i64], cols: usize) -> Option<i64> {
    let full_bytes = cols >> 2;
    let remainder = cols & 0b11;
    let mut coeff_idx = 0usize;
    let mut acc = 0i128;

    for &byte in row.iter().take(full_bytes) {
        let pair0 = byte & 0b11;
        let pair1 = (byte >> 2) & 0b11;
        let pair2 = (byte >> 4) & 0b11;
        let pair3 = (byte >> 6) & 0b11;

        acc += (jl_pair_to_sign(pair0) as i128) * (coeffs[coeff_idx] as i128);
        acc += (jl_pair_to_sign(pair1) as i128) * (coeffs[coeff_idx + 1] as i128);
        acc += (jl_pair_to_sign(pair2) as i128) * (coeffs[coeff_idx + 2] as i128);
        acc += (jl_pair_to_sign(pair3) as i128) * (coeffs[coeff_idx + 3] as i128);
        coeff_idx += 4;
    }

    if remainder > 0 {
        let byte = row[full_bytes];
        for lane in 0..remainder {
            let pair = (byte >> (lane << 1)) & 0b11;
            acc += (jl_pair_to_sign(pair) as i128) * (coeffs[coeff_idx] as i128);
            coeff_idx += 1;
        }
    }

    i64::try_from(acc).ok()
}

fn project_row_i128<const D: usize>(
    row: &[u8],
    rings: &[[i128; D]],
    cols: usize,
    use_checked: bool,
) -> Option<i64> {
    let mut acc = 0i128;
    let mut col_idx = 0usize;

    for coeff_chunk in rings {
        for &value in coeff_chunk {
            let pair = jl_pair_at(row, col_idx);
            if use_checked {
                match jl_pair_to_sign(pair) {
                    -1 => acc = acc.checked_sub(value)?,
                    0 => {}
                    1 => acc = acc.checked_add(value)?,
                    _ => unreachable!(),
                }
            } else {
                acc += (jl_pair_to_sign(pair) as i128) * value;
            }
            col_idx += 1;
        }
    }
    debug_assert_eq!(col_idx, cols);
    i64::try_from(acc).ok()
}

#[tracing::instrument(skip_all, name = "labrador::project_streaming")]
fn project_streaming<const D: usize>(
    matrix: &LabradorJlMatrix,
    centered_witness: &CenteredWitness<D>,
    total_coeffs: usize,
) -> Option<[i64; 256]> {
    if !matrix.is_well_formed()
        || matrix.cols() != total_coeffs
        || centered_witness.ring_len() * D != total_coeffs
    {
        return None;
    }
    let results: Vec<Option<i64>> = match centered_witness {
        CenteredWitness::I64 { coeffs } => cfg_into_iter!(0..JL_ROWS)
            .map(|row_idx| project_row_i64(matrix.row_bytes(row_idx), coeffs, total_coeffs))
            .collect(),
        CenteredWitness::I128 { rings, sum_abs } => {
            let use_checked = *sum_abs > i128::MAX as u128;
            cfg_into_iter!(0..JL_ROWS)
                .map(|row_idx| {
                    project_row_i128(matrix.row_bytes(row_idx), rings, total_coeffs, use_checked)
                })
                .collect()
        }
    };
    let mut out = [0i64; 256];
    for (i, val) in results.into_iter().enumerate() {
        out[i] = val?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::{Fp64, Prime128M13M4P0};
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_RECURSION;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    type F128 = Prime128M13M4P0;
    const D: usize = 64;

    fn sample_witness_from_seed_generic<G>(seed: u64) -> LabradorWitness<G, D>
    where
        G: FieldCore + CanonicalField + FromSmallInt,
    {
        let num_rows = 2 + (seed % 3) as usize;
        let row_len = 3 + (seed % 5) as usize;
        let rows: Vec<Vec<CyclotomicRing<G, D>>> = (0..num_rows)
            .map(|r| {
                (0..row_len)
                    .map(|i| {
                        CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                            let mix = seed
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(r as u64 * 997 + i as u64 * 31 + j as u64);
                            G::from_i64(((mix % 11) as i64) - 5)
                        }))
                    })
                    .collect()
            })
            .collect();
        LabradorWitness::new(rows)
    }

    fn sample_witness_from_seed(seed: u64) -> LabradorWitness<F, D> {
        sample_witness_from_seed_generic::<F>(seed)
    }

    fn witness_squared_norm<G: FieldCore + CanonicalField>(
        witness: &LabradorWitness<G, D>,
    ) -> i128 {
        let q = (-G::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let mut norm_sq = 0i128;
        for row in witness.rows() {
            for ring in row {
                for coeff in ring.coefficients() {
                    let c = coeff.to_canonical_u128();
                    let v = if c > half_q {
                        -((q - c) as i128)
                    } else {
                        c as i128
                    };
                    norm_sq += v * v;
                }
            }
        }
        norm_sq
    }

    #[test]
    fn project_is_deterministic_and_replayable() {
        let witness = sample_witness_from_seed(42);
        let mut t1 = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
        let mut t2 = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
        let (p1, n1, _) = project(&witness, &mut t1).unwrap();
        let (p2, n2, _) = project(&witness, &mut t2).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(n1, n2);
    }

    #[test]
    fn project_fp128_is_deterministic_and_replayable() {
        let witness = sample_witness_from_seed_generic::<F128>(42);
        let mut t1 = Blake2bTranscript::<F128>::new(DOMAIN_LABRADOR_RECURSION);
        let mut t2 = Blake2bTranscript::<F128>::new(DOMAIN_LABRADOR_RECURSION);
        let (p1, n1, _) = project(&witness, &mut t1).unwrap();
        let (p2, n2, _) = project(&witness, &mut t2).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(n1, n2);
    }

    #[test]
    fn project_streaming_handles_fp128_centered_values_beyond_i64() {
        let q = (-F128::one()).to_canonical_u128() + 1;
        let large = q / 2 + 17;
        let ring = CyclotomicRing::<F128, D>::from_coefficients(std::array::from_fn(|idx| {
            if idx == 0 || idx == 1 {
                F128::from_canonical_u128_reduced(large)
            } else {
                F128::zero()
            }
        }));
        let witness = LabradorWitness::new(vec![vec![ring]]);
        let centered = center_witness_by_ring(&witness).unwrap();
        let centered_abs = match &centered {
            CenteredWitness::I64 { coeffs, .. } => coeffs[0].unsigned_abs() as u128,
            CenteredWitness::I128 { rings, .. } => rings[0][0].unsigned_abs(),
        };
        assert!(centered_abs > i64::MAX as u128);

        let signs: Vec<Vec<i8>> = (0..JL_ROWS)
            .map(|row_idx| {
                let mut row = vec![0i8; D];
                if row_idx == 0 {
                    row[0] = 1;
                    row[1] = -1;
                }
                row
            })
            .collect();
        let matrix = LabradorJlMatrix::from_sign_rows(signs).unwrap();
        let projection = project_streaming::<D>(&matrix, &centered, D).unwrap();
        assert_eq!(projection[0], 0);
        assert!(projection.iter().skip(1).all(|&v| v == 0));
    }

    #[test]
    fn packed_matrix_roundtrips_manual_signs() {
        let signs: Vec<Vec<i8>> = (0..JL_ROWS)
            .map(|row_idx| {
                (0..7)
                    .map(|col_idx| match (row_idx + col_idx) % 3 {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    })
                    .collect()
            })
            .collect();
        let matrix = LabradorJlMatrix::from_sign_rows(signs.clone()).unwrap();
        for (row_idx, row) in signs.iter().enumerate() {
            for (col_idx, &sign) in row.iter().enumerate() {
                assert_eq!(matrix.sign_at(row_idx, col_idx), Some(sign));
            }
        }
    }

    #[test]
    fn project_norm_bound_over_multiple_witnesses() {
        for seed in 1..=10u64 {
            let witness = sample_witness_from_seed(seed);
            let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
            let (projection, nonce, _) = project(&witness, &mut transcript).unwrap();

            let beta = witness_squared_norm(&witness);
            let p_norm_sq: i128 = projection.iter().map(|&v| (v as i128) * (v as i128)).sum();
            let p_inf: i128 = projection.iter().map(|&v| (v as i128).abs()).max().unwrap();
            let entry_bound = ((128.0 * beta as f64).sqrt()) as i128;

            tracing::debug!(
                seed,
                nonce,
                p_norm_sq,
                p_inf,
                entry_bound,
                beta,
                "JL projection check"
            );
            assert!(
                p_inf <= entry_bound,
                "seed={seed}: ||p||_inf={p_inf} exceeds sqrt(128β)={entry_bound}"
            );
        }
    }

    #[test]
    fn collapse_matches_dot_product() {
        let projection = std::array::from_fn(|i| i as i64 - 10);
        let alpha = std::array::from_fn(|i| (2 * i as i64) - 7);
        let got = collapse(&projection, &alpha);
        let expected = projection
            .iter()
            .zip(alpha.iter())
            .fold(0i64, |acc, (&p, &a)| acc + p * a);
        assert_eq!(got, expected);
    }

    #[test]
    fn lift_zero_and_restore_constant_term() {
        let poly: CyclotomicRing<F, D> =
            CyclotomicRing::from_coefficients(std::array::from_fn(|i| F::from_i64(i as i64 - 5)));
        let (tx, c0) = zero_constant_term_for_proof(poly);
        assert!(tx.coefficients()[0].is_zero());
        let restored = restore_constant_term(tx, c0);
        assert_eq!(restored, poly);
    }
}
