//! The [`SparseChallenge`] data type.
//!
//! A [`SparseChallenge`] is a sparse polynomial in `F[X] / (X^D + 1)` represented
//! by its non-zero terms. It is the unified output type for every sampling family
//! in [`crate::SparseChallengeConfig`], so downstream arithmetic can stay uniform
//! regardless of how a challenge was sampled.
//!
//! Production challenges are expected to come from [`crate::sample_sparse_challenges`],
//! which constructs values satisfying the invariants below. Methods on this
//! type check cheap shape/range errors needed for memory safety, but they do
//! not re-validate every sampler invariant on the hot path.
//!
//! This module deliberately depends only on `akita-field`; it does not pull in
//! the transcript layer or the sampler. Most consumers of this crate
//! (`akita-types`, `akita-config`, `akita-planner`, `akita-prover`/
//! `akita-verifier` ring-switching, etc.) only ever touch this type and never
//! run the sampler.

use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};
use std::collections::BTreeMap;

/// Sparse polynomial in `F[X]/(X^D+1)` represented by its non-zero terms.
///
/// Sampler invariants:
/// - `positions.len() == coeffs.len()`
/// - all positions are `< D`
/// - positions are unique
/// - all coeffs are non-zero
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseChallenge {
    /// Coefficient indices (powers of `X`) where the polynomial is non-zero.
    pub positions: Vec<u32>,
    /// Small integer coefficients at the corresponding positions. Stored
    /// as `i8` since every shipping sampling family caps `|coeff| <= 8`.
    pub coeffs: Vec<i8>,
}

/// Sparse polynomial in `Z[X]/(X^D+1)` with wider integer coefficients than
/// [`SparseChallenge`].
///
/// Composing sparse challenges (for example forming `c_{p,q} = α_p · β_q`
/// for tensor-shaped stage-1 folding) can blow past the `i8` envelope of the
/// individual samples. `IntegerChallenge` is the wider-coefficient form used
/// for those composed objects so prover-side digit accumulation can stay in
/// the integer domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerChallenge {
    /// Coefficient indices (powers of `X`) where the polynomial is non-zero.
    pub positions: Vec<u32>,
    /// Integer coefficients at the corresponding positions.
    pub coeffs: Vec<i32>,
}

impl SparseChallenge {
    /// Evaluate this challenge against precomputed scalar powers.
    ///
    /// The small integer coefficients are first embedded into the base field
    /// `F`, then multiplied into `E` with a mixed base-field operation. The
    /// ordinary base-field case is `E = F`.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha_pows` does not have length `D`, or if a
    /// term would index outside the supplied powers. This method assumes the
    /// challenge came from [`crate::sample_sparse_challenges`] and therefore
    /// does not re-check uniqueness of positions on the hot path.
    pub fn eval_at_pows<F, E, const D: usize>(&self, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
            });
        }
        if self.positions.len() != self.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "sparse challenge positions/coeffs length mismatch".to_string(),
            ));
        }

        let mut acc = E::zero();
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            if idx >= D {
                return Err(AkitaError::InvalidInput(format!(
                    "sparse challenge position {idx} out of range for D={D}"
                )));
            }
            if coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "sparse challenge coefficients must be non-zero".to_string(),
                ));
            }
            acc += alpha_pows[idx].mul_base(F::from_i64(coeff as i64));
        }
        Ok(acc)
    }
}

impl IntegerChallenge {
    /// Widen a [`SparseChallenge`] into an [`IntegerChallenge`].
    #[inline]
    #[must_use]
    pub fn from_sparse(challenge: &SparseChallenge) -> Self {
        Self {
            positions: challenge.positions.clone(),
            coeffs: challenge.coeffs.iter().map(|&c| i32::from(c)).collect(),
        }
    }

    /// Narrow to [`SparseChallenge`] when every coefficient fits in `i8`.
    ///
    /// # Errors
    ///
    /// Returns an error if any coefficient is outside the `i8` range.
    pub fn try_to_sparse_i8(&self) -> Result<SparseChallenge, AkitaError> {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&coeff| {
                i8::try_from(coeff).map_err(|_| {
                    AkitaError::InvalidInput(
                        "integer challenge coefficient does not fit in i8".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SparseChallenge {
            positions: self.positions.clone(),
            coeffs,
        })
    }

    /// Multiply two sparse challenges in `Z[X]/(X^D + 1)`.
    ///
    /// # Errors
    ///
    /// Returns an error if either factor is malformed, has a position outside
    /// `0..D`, or if any output coefficient overflows `i32`.
    pub fn tensor_product<const D: usize>(
        left: &SparseChallenge,
        right: &SparseChallenge,
    ) -> Result<Self, AkitaError> {
        validate_sparse::<D>(left)?;
        validate_sparse::<D>(right)?;

        let mut coeffs = BTreeMap::<u32, i32>::new();
        for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
            for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
                let degree = left_pos as usize + right_pos as usize;
                let (pos, sign) = if degree < D {
                    (degree as u32, 1i32)
                } else {
                    ((degree - D) as u32, -1i32)
                };
                let term = i32::from(left_coeff)
                    .checked_mul(i32::from(right_coeff))
                    .and_then(|term| term.checked_mul(sign))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "tensor challenge coefficient overflow".to_string(),
                        )
                    })?;
                let entry = coeffs.entry(pos).or_insert(0);
                *entry = entry.checked_add(term).ok_or_else(|| {
                    AkitaError::InvalidInput("tensor challenge coefficient overflow".to_string())
                })?;
                if *entry == 0 {
                    coeffs.remove(&pos);
                }
            }
        }

        Ok(Self {
            positions: coeffs.keys().copied().collect(),
            coeffs: coeffs.values().copied().collect(),
        })
    }

    /// Evaluate this challenge against precomputed scalar powers, mirroring
    /// [`SparseChallenge::eval_at_pows`].
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha_pows` does not have length `D`, or if a
    /// term would index outside the supplied powers.
    pub fn eval_at_pows<F, E, const D: usize>(&self, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
            });
        }
        if self.positions.len() != self.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "integer challenge positions/coeffs length mismatch".to_string(),
            ));
        }

        let mut acc = E::zero();
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            if idx >= D {
                return Err(AkitaError::InvalidInput(format!(
                    "integer challenge position {idx} out of range for D={D}"
                )));
            }
            if coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "integer challenge coefficients must be non-zero".to_string(),
                ));
            }
            acc += alpha_pows[idx].mul_base(F::from_i64(i64::from(coeff)));
        }
        Ok(acc)
    }
}

fn validate_sparse<const D: usize>(challenge: &SparseChallenge) -> Result<(), AkitaError> {
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidInput(
            "sparse challenge positions/coeffs length mismatch".to_string(),
        ));
    }
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        if pos as usize >= D {
            return Err(AkitaError::InvalidInput(format!(
                "sparse challenge position {pos} out of range for D={D}"
            )));
        }
        if coeff == 0 {
            return Err(AkitaError::InvalidInput(
                "sparse challenge coefficients must be non-zero".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;
    const D: usize = 4;

    fn alpha_pows() -> [F; D] {
        [
            F::from_u64(1),
            F::from_u64(3),
            F::from_u64(9),
            F::from_u64(27),
        ]
    }

    #[test]
    fn eval_at_pows_evaluates_sparse_terms() {
        let challenge = SparseChallenge {
            positions: vec![0, 2],
            coeffs: vec![1, -2],
        };

        let got = challenge.eval_at_pows::<F, F, D>(&alpha_pows()).unwrap();
        let expected = F::from_u64(1) + F::from_i64(-2) * F::from_u64(9);

        assert_eq!(got, expected);
    }

    #[test]
    fn eval_at_pows_rejects_wrong_power_count() {
        let challenge = SparseChallenge {
            positions: vec![0],
            coeffs: vec![1],
        };

        let err = challenge
            .eval_at_pows::<F, F, D>(&alpha_pows()[..D - 1])
            .unwrap_err();

        assert_eq!(
            err,
            AkitaError::InvalidSize {
                expected: D,
                actual: D - 1
            }
        );
    }

    #[test]
    fn eval_at_pows_rejects_out_of_range_position() {
        let challenge = SparseChallenge {
            positions: vec![D as u32],
            coeffs: vec![1],
        };

        let err = challenge
            .eval_at_pows::<F, F, D>(&alpha_pows())
            .unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("out of range")));
    }

    #[test]
    fn eval_at_pows_rejects_mismatched_terms() {
        let challenge = SparseChallenge {
            positions: vec![0, 1],
            coeffs: vec![1],
        };

        let err = challenge
            .eval_at_pows::<F, F, D>(&alpha_pows())
            .unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("length mismatch")));
    }

    #[test]
    fn integer_challenge_from_sparse_round_trips() {
        let sparse = SparseChallenge {
            positions: vec![0, 3],
            coeffs: vec![1, -2],
        };
        let widened = IntegerChallenge::from_sparse(&sparse);
        assert_eq!(widened.positions, sparse.positions);
        assert_eq!(widened.coeffs, vec![1, -2]);

        let narrowed = widened.try_to_sparse_i8().unwrap();
        assert_eq!(narrowed, sparse);
    }

    #[test]
    fn integer_challenge_try_to_sparse_i8_rejects_overflow() {
        let oversized = IntegerChallenge {
            positions: vec![1],
            coeffs: vec![i32::from(i8::MAX) + 1],
        };
        let err = oversized.try_to_sparse_i8().unwrap_err();
        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("does not fit in i8")));
    }

    #[test]
    fn integer_challenge_tensor_product_reduces_negacyclically() {
        let left = SparseChallenge {
            positions: vec![2],
            coeffs: vec![1],
        };
        let right = SparseChallenge {
            positions: vec![D as u32 - 1],
            coeffs: vec![1],
        };
        let product = IntegerChallenge::tensor_product::<D>(&left, &right).unwrap();
        // (2) + (D - 1) = D + 1, which wraps to position 1 with a sign flip.
        assert_eq!(product.positions, vec![1]);
        assert_eq!(product.coeffs, vec![-1]);
    }

    #[test]
    fn integer_challenge_tensor_product_drops_cancellations() {
        let left = SparseChallenge {
            positions: vec![0, 1],
            coeffs: vec![1, 1],
        };
        let right = SparseChallenge {
            positions: vec![0, D as u32 - 1],
            coeffs: vec![1, 1],
        };
        let product = IntegerChallenge::tensor_product::<D>(&left, &right).unwrap();
        // pos 0: 1*1 = 1
        // pos 1: 1*1 = 1
        // pos D-1: 1*1 = 1
        // pos D = wrap to 0 with sign flip: 1*1*(-1) = -1, cancels pos 0 term.
        assert_eq!(product.positions, vec![1, D as u32 - 1]);
        assert_eq!(product.coeffs, vec![1, 1]);
    }

    #[test]
    fn integer_challenge_eval_matches_sparse_when_in_range() {
        let sparse = SparseChallenge {
            positions: vec![0, 2],
            coeffs: vec![1, -2],
        };
        let widened = IntegerChallenge::from_sparse(&sparse);

        let pows = alpha_pows();
        let sparse_eval = sparse.eval_at_pows::<F, F, D>(&pows).unwrap();
        let widened_eval = widened.eval_at_pows::<F, F, D>(&pows).unwrap();
        assert_eq!(sparse_eval, widened_eval);
    }
}
