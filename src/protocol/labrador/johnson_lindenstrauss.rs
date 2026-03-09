//! Johnson-Lindenstrauss helpers for Labrador reduction.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_JL_NONCE_RETRIES;
use crate::protocol::labrador::types::LabradorWitness;
use crate::protocol::transcript::{labels, Transcript};
use crate::{CanonicalField, FieldCore};

const JL_ROWS: usize = 256;

/// Binary JL matrix with entries in `{-1, +1}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorJlMatrix {
    /// Matrix entries as `-1/+1`, one inner vec per row.
    pub signs: Vec<Vec<i8>>,
}

impl LabradorJlMatrix {
    /// Number of columns (derived from the first row).
    pub fn cols(&self) -> usize {
        self.signs.first().map_or(0, |r| r.len())
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
        if cols == 0 {
            return Err(HachiError::InvalidInput(
                "JL matrix requires non-zero column count".to_string(),
            ));
        }
        let byte_len_per_row = (cols * 2).div_ceil(8);
        let total_bytes = JL_ROWS * byte_len_per_row;
        let all_bytes = transcript.challenge_bytes(labels::CHALLENGE_LABRADOR_JL_SEED, total_bytes);
        let signs: Vec<Vec<i8>> = all_bytes
            .chunks(byte_len_per_row)
            .map(|bytes| {
                (0..cols)
                    .map(|c| {
                        let bit_offset = c * 2;
                        let byte_idx = bit_offset / 8;
                        let shift = bit_offset % 8;
                        let pair = (bytes[byte_idx] >> shift) & 0b11;
                        match pair {
                            0b00 => -1,
                            0b11 => 1,
                            _ => 0,
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Self { signs })
    }

    /// Replay the prover's nonce loop to reconstruct the JL matrix.
    ///
    /// Absorbs nonces `1..=jl_nonce` and squeezes a matrix for each, returning
    /// the final matrix. Leaves the transcript in the same state the prover
    /// had after `project` returned.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is zero.
    pub fn replay_nonce_search<F, T>(
        transcript: &mut T,
        jl_nonce: u64,
        cols: usize,
    ) -> Result<Self, HachiError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
    {
        let mut matrix = None;
        for nonce in 1..=jl_nonce {
            transcript.append_bytes(labels::ABSORB_LABRADOR_JL_NONCE, &nonce.to_le_bytes());
            matrix = Some(Self::generate::<F, T>(transcript, cols)?);
        }
        matrix.ok_or_else(|| HachiError::InvalidInput("JL nonce must be at least 1".to_string()))
    }
}

/// Project a witness into 256 JL coordinates and return the nonce used.
///
/// Each nonce attempt absorbs the nonce and squeezes the matrix from the
/// transcript. The verifier must replay the same loop from 1 to the
/// returned nonce to keep the transcript in sync.
///
/// # Errors
///
/// Returns an error if the witness is empty or if no valid projection is found
/// within the nonce search limit.
#[tracing::instrument(skip_all, name = "labrador::project")]
pub fn project<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    transcript: &mut T,
) -> Result<([i32; 256], u64, LabradorJlMatrix), HachiError>
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
    for nonce in 1..=LABRADOR_MAX_JL_NONCE_RETRIES {
        transcript.append_bytes(labels::ABSORB_LABRADOR_JL_NONCE, &nonce.to_le_bytes());
        let matrix = LabradorJlMatrix::generate::<F, T>(transcript, total_coeffs)?;
        if let Some(proj) = project_streaming(&matrix, witness) {
            return Ok((proj, nonce, matrix));
        }
    }
    Err(HachiError::InvalidInput(format!(
        "failed JL projection nonce search after {LABRADOR_MAX_JL_NONCE_RETRIES} attempts"
    )))
}

/// Collapse a JL projection with challenge coefficients.
///
/// Returns the linear target value `sum_i alpha[i] * projection[i]`.
pub fn collapse(projection: &[i32; 256], alpha: &[i64; 256]) -> i64 {
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
#[tracing::instrument(skip_all, name = "labrador::project_streaming")]
fn project_streaming<F: FieldCore + CanonicalField, const D: usize>(
    matrix: &LabradorJlMatrix,
    witness: &LabradorWitness<F, D>,
) -> Option<[i32; 256]> {
    if matrix.signs.len() != JL_ROWS {
        return None;
    }
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;

    let results: Vec<Option<i32>> = cfg_iter!(matrix.signs)
        .map(|row| {
            let mut acc = 0i128;
            let mut col = 0;
            for witness_row in witness.rows() {
                for ring in witness_row {
                    for coeff in ring.coefficients() {
                        let c = coeff.to_canonical_u128();
                        let value = if c > half_q {
                            -((q - c) as i128)
                        } else {
                            c as i128
                        };
                        acc += (row[col] as i128) * value;
                        col += 1;
                    }
                }
            }
            if acc < i32::MIN as i128 || acc > i32::MAX as i128 {
                None
            } else {
                Some(acc as i32)
            }
        })
        .collect();
    let mut out = [0i32; 256];
    for (i, val) in results.into_iter().enumerate() {
        out[i] = val?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_witness_from_seed(seed: u64) -> LabradorWitness<F, D> {
        let num_rows = 2 + (seed % 3) as usize;
        let row_len = 3 + (seed % 5) as usize;
        let rows: Vec<Vec<CyclotomicRing<F, D>>> = (0..num_rows)
            .map(|r| {
                (0..row_len)
                    .map(|i| {
                        CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                            let mix = seed
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(r as u64 * 997 + i as u64 * 31 + j as u64);
                            F::from_i64(((mix % 11) as i64) - 5)
                        }))
                    })
                    .collect()
            })
            .collect();
        LabradorWitness::new(rows)
    }

    fn witness_squared_norm(witness: &LabradorWitness<F, D>) -> i128 {
        let q = (-F::one()).to_canonical_u128() + 1;
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
        let mut t1 = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let mut t2 = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let (p1, n1, _) = project(&witness, &mut t1).unwrap();
        let (p2, n2, _) = project(&witness, &mut t2).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(n1, n2);
    }

    #[test]
    fn project_norm_bound_over_multiple_witnesses() {
        for seed in 1..=10u64 {
            let witness = sample_witness_from_seed(seed);
            let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
            let (projection, nonce, _) = project(&witness, &mut transcript).unwrap();

            let beta = witness_squared_norm(&witness);
            let p_norm_sq: i128 = projection.iter().map(|&v| (v as i128) * (v as i128)).sum();
            let p_inf: i128 = projection.iter().map(|&v| (v as i128).abs()).max().unwrap();
            let entry_bound = ((128.0 * beta as f64).sqrt()) as i128;

            println!(
                "seed={seed}: nonce={nonce}, ||p||²={p_norm_sq}, ||p||_inf={p_inf}, \
                 sqrt(128β)={entry_bound}, β={beta}"
            );
            assert!(
                p_inf <= entry_bound,
                "seed={seed}: ||p||_inf={p_inf} exceeds sqrt(128β)={entry_bound}"
            );
        }
    }

    #[test]
    fn collapse_matches_dot_product() {
        let projection = std::array::from_fn(|i| i as i32 - 10);
        let alpha = std::array::from_fn(|i| (2 * i as i64) - 7);
        let got = collapse(&projection, &alpha);
        let expected = projection
            .iter()
            .zip(alpha.iter())
            .fold(0i64, |acc, (&p, &a)| acc + (p as i64) * a);
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
