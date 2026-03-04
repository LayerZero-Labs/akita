//! Johnson-Lindenstrauss helpers for Labrador reduction.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_JL_NONCE_RETRIES;
use crate::protocol::labrador::types::LabradorWitness;
use crate::protocol::prg::{MatrixPrgBackendChoice, MatrixPrgContext};
use crate::{CanonicalField, FieldCore};
use rand_core::RngCore;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;

const JL_ROWS: usize = 256;

/// Binary JL matrix with entries in `{-1, +1}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorJlMatrix {
    /// Number of rows (fixed at 256 in Labrador).
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Matrix entries as `-1/+1`.
    pub signs: Vec<Vec<i8>>,
}

impl LabradorJlMatrix {
    /// Deterministically generate a JL matrix from seed/nonce.
    ///
    /// # Errors
    ///
    /// Returns an error if `cols` is zero.
    pub fn generate(
        seed: &[u8; 16],
        nonce: u64,
        cols: usize,
        backend: MatrixPrgBackendChoice,
    ) -> Result<Self, HachiError> {
        if cols == 0 {
            return Err(HachiError::InvalidInput(
                "JL matrix requires non-zero column count".to_string(),
            ));
        }
        let prg_seed = derive_jl_prg_seed(seed, nonce);
        let byte_len = cols.div_ceil(8);
        let signs: Vec<Vec<i8>> = cfg_into_iter!(0..JL_ROWS)
            .map(|row| {
                let context = MatrixPrgContext {
                    seed: &prg_seed,
                    matrix_label: b"labrador/jl",
                    rows: JL_ROWS,
                    cols,
                    row,
                    col: 0,
                };
                let mut rng = backend.entry_rng(&context);
                let mut bytes = vec![0u8; byte_len];
                rng.fill_bytes(&mut bytes);
                (0..cols)
                    .map(|c| {
                        let bit = (bytes[c / 8] >> (c % 8)) & 1;
                        if bit == 0 {
                            -1
                        } else {
                            1
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            rows: JL_ROWS,
            cols,
            signs,
        })
    }
}

/// Project a witness into 256 JL coordinates and return the nonce used.
///
/// Nonce search starts from `1` and stops at the first projection that fits
/// signed 32-bit coordinates, up to `LABRADOR_MAX_JL_NONCE_RETRIES`.
///
/// # Errors
///
/// Returns an error if the witness is empty or if no valid projection is found
/// within the nonce search limit.
pub fn project<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
    seed: &[u8; 16],
    backend: MatrixPrgBackendChoice,
) -> Result<([i32; 256], u64), HachiError> {
    let vector = flatten_witness_coeffs(witness);
    if vector.is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot JL-project empty witness".to_string(),
        ));
    }
    for nonce in 1..=LABRADOR_MAX_JL_NONCE_RETRIES {
        let matrix = LabradorJlMatrix::generate(seed, nonce, vector.len(), backend)?;
        if let Some(proj) = project_with_matrix(&matrix, &vector) {
            return Ok((proj, nonce));
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

fn derive_jl_prg_seed(seed: &[u8; 16], nonce: u64) -> [u8; 32] {
    let mut xof = Shake256::default();
    xof.update(b"hachi/labrador/jl");
    xof.update(seed);
    xof.update(&nonce.to_le_bytes());
    let mut out = [0u8; 32];
    xof.finalize_xof().read(&mut out);
    out
}

fn flatten_witness_coeffs<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> Vec<i64> {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let per_row: Vec<Vec<i64>> = cfg_iter!(witness.rows())
        .map(|row| {
            row.iter()
                .flat_map(|ring| ring.coefficients().iter())
                .map(|coeff| {
                    let c = coeff.to_canonical_u128();
                    if c > half_q {
                        -((q - c) as i64)
                    } else {
                        c as i64
                    }
                })
                .collect()
        })
        .collect();
    per_row.into_iter().flatten().collect()
}

fn project_with_matrix(matrix: &LabradorJlMatrix, vector: &[i64]) -> Option<[i32; 256]> {
    if matrix.cols != vector.len() || matrix.rows != JL_ROWS {
        return None;
    }
    let results: Vec<Option<i32>> = cfg_iter!(matrix.signs)
        .map(|row| {
            let mut acc = 0i128;
            for (&sign, &value) in row.iter().zip(vector.iter()) {
                acc += (sign as i128) * (value as i128);
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
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_witness() -> LabradorWitness<F, D> {
        let row = |len: usize| -> Vec<CyclotomicRing<F, D>> {
            (0..len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        F::from_i64(((i + j) as i64 % 7) - 3)
                    }))
                })
                .collect()
        };
        LabradorWitness::new(vec![row(4), row(4)])
    }

    #[test]
    fn project_is_deterministic_and_replayable() {
        let witness = sample_witness();
        let seed = [9u8; 16];
        let (p1, n1) = project(&witness, &seed, MatrixPrgBackendChoice::Shake256).unwrap();
        let (p2, n2) = project(&witness, &seed, MatrixPrgBackendChoice::Shake256).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(n1, n2);
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
