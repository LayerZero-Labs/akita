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
//! the transcript layer or the sampler.

use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

#[inline]
pub(crate) fn accumulate_small_signed<F, E>(acc: &mut E, value: E, coeff: i64)
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    match coeff {
        1 => *acc += value,
        -1 => *acc -= value,
        2 => {
            *acc += value;
            *acc += value;
        }
        -2 => {
            *acc -= value;
            *acc -= value;
        }
        _ => *acc += value.mul_base(F::from_i64(coeff)),
    }
}

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

impl SparseChallenge {
    /// Validate the sampler invariants for ring dimension `D`.
    ///
    /// # Errors
    ///
    /// Returns an error if positions/coefficients have mismatched lengths, if a
    /// position is outside `0..D`, if a coefficient is zero, or if a position is
    /// repeated.
    pub fn validate<const D: usize>(&self) -> Result<(), AkitaError> {
        self.validate_dyn(D)
    }

    /// Runtime-dimension form of [`Self::validate`].
    ///
    /// # Errors
    ///
    /// Returns an error if positions/coefficients have mismatched lengths, if a
    /// position is outside `0..ring_d`, if a coefficient is zero, or if a
    /// position is repeated.
    pub fn validate_dyn(&self, ring_d: usize) -> Result<(), AkitaError> {
        if self.positions.len() != self.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "sparse challenge positions/coeffs length mismatch".to_string(),
            ));
        }

        let mut seen = vec![false; ring_d];
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            if idx >= ring_d {
                return Err(AkitaError::InvalidInput(format!(
                    "sparse challenge position {pos} out of range for D={ring_d}"
                )));
            }
            if coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "sparse challenge coefficients must be non-zero".to_string(),
                ));
            }
            if seen[idx] {
                return Err(AkitaError::InvalidInput(
                    "sparse challenge positions must be unique".to_string(),
                ));
            }
            seen[idx] = true;
        }
        Ok(())
    }

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
    pub fn eval_at_pows<F, E>(&self, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        F: FieldCore + FromPrimitiveInt,
        E: FieldCore + MulBase<F>,
    {
        let ring_d = alpha_pows.len();
        if self.positions.len() != self.coeffs.len() {
            return Err(AkitaError::InvalidInput(
                "sparse challenge positions/coeffs length mismatch".to_string(),
            ));
        }

        let mut acc = E::zero();
        for (&pos, &coeff) in self.positions.iter().zip(self.coeffs.iter()) {
            let idx = pos as usize;
            if idx >= ring_d {
                return Err(AkitaError::InvalidInput(format!(
                    "sparse challenge position {idx} out of range for D={ring_d}"
                )));
            }
            if coeff == 0 {
                return Err(AkitaError::InvalidInput(
                    "sparse challenge coefficients must be non-zero".to_string(),
                ));
            }
            accumulate_small_signed::<F, E>(&mut acc, alpha_pows[idx], i64::from(coeff));
        }
        Ok(acc)
    }
}

#[cfg(test)]
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

        let got = challenge.eval_at_pows::<F, F>(&alpha_pows()).unwrap();
        let expected = F::from_u64(1) + F::from_i64(-2) * F::from_u64(9);

        assert_eq!(got, expected);
    }

    #[test]
    fn eval_at_pows_rejects_position_beyond_power_count() {
        // The ring dimension is `alpha_pows.len()`; a position that fits the
        // nominal D but not the supplied power table must be rejected.
        let challenge = SparseChallenge {
            positions: vec![D as u32 - 1],
            coeffs: vec![1],
        };

        let err = challenge
            .eval_at_pows::<F, F>(&alpha_pows()[..D - 1])
            .unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn eval_at_pows_rejects_out_of_range_position() {
        let challenge = SparseChallenge {
            positions: vec![D as u32],
            coeffs: vec![1],
        };

        let err = challenge.eval_at_pows::<F, F>(&alpha_pows()).unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("out of range")));
    }

    #[test]
    fn eval_at_pows_rejects_mismatched_terms() {
        let challenge = SparseChallenge {
            positions: vec![0, 1],
            coeffs: vec![1],
        };

        let err = challenge.eval_at_pows::<F, F>(&alpha_pows()).unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("length mismatch")));
    }

    #[test]
    fn validate_rejects_duplicate_positions() {
        let challenge = SparseChallenge {
            positions: vec![1, 1],
            coeffs: vec![1, -1],
        };

        let err = challenge.validate::<D>().unwrap_err();

        assert!(matches!(err, AkitaError::InvalidInput(msg) if msg.contains("unique")));
    }
}
