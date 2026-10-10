//! No-wrap admission for the commitment relation lifted to the proof field.
//!
//! The commitment relation holds modulo the commitment prime `q` and is proved
//! over the proof prime `P`. Its integer form in the commitment ring is
//! `A z - sum_{i < C} c_i T_i = q K`: `A` has canonical entries in
//! `[0, q - 1]`, the response `z` has `m` ring elements of degree `D`, `T_i`
//! are the image columns and `K` is a carry. An identity between enforced
//! ranges that holds modulo `P` holds over the integers when
//!
//! `3 * m * D * (q - 1) * B_z + C * Gamma * B_T + q * B_K < P`.
//!
//! The first term bounds `A z`: a reduced coefficient of a trinomial-ring
//! product collects at most three unreduced coefficients, each a sum of at
//! most `D` products. The second bounds the image term under the challenge
//! multiplication bound `Gamma`; `B_T` is the magnitude the image digits can
//! represent, because the digit range is the only enforced bound on an image
//! coefficient. The third bounds the carry term.
//!
//! Honest carry. An honest image coefficient is a residue of magnitude at most
//! `q - 1`, so `q * |K| <= (q - 1) * (3 * m * D * B_z + C * Gamma)`. With
//! `Gamma = 2w` this gives `|K| <= 3 * (m * D * B_z + C * w)`.

use akita_challenges::BinaryChallengeProfile;
use akita_error::{checked, AkitaError};

use super::{LabiniusRingDegree, LABINIUS_BALANCED_LOG_BASIS};
use crate::sis::{balanced_digit_abs_max, num_digits_for_linf_cap};

/// Balanced base-16 digit count of one image coefficient and the magnitude
/// `B_T` those digits can represent.
///
/// An image coefficient is committed as `sum_l digit_l * 16^l - B_T` with
/// stored digits in `[0, 15]`, so the digits enforce the interval
/// `[-B_T, 7 * (16^k - 1) / 15]`. The count `k` is the least whose positive
/// reach covers the honest value, the canonical residue in
/// `[0, commitment_modulus)`. `B_T = 8 * (16^k - 1) / 15` is the negative
/// reach, the larger of the two, and doubles as the encoding offset.
pub fn labinius_image_digit_bound(commitment_modulus: u32) -> (usize, u128) {
    let digits = num_digits_for_linf_cap(
        u128::from(commitment_modulus.saturating_sub(1)),
        u128::BITS,
        LABINIUS_BALANCED_LOG_BASIS,
    );
    (
        digits,
        balanced_digit_abs_max(LABINIUS_BALANCED_LOG_BASIS, digits),
    )
}

/// Public geometry and enforced response bound of one lifted commitment
/// relation.
#[derive(Debug, Clone, Copy)]
pub struct LabiniusCommitmentLift<'a> {
    /// Fold challenge family, supplying `w` and `Gamma`.
    pub challenge: &'a BinaryChallengeProfile,
    /// Folded columns `C`.
    pub fold_columns: usize,
    /// Ring elements `m` of one column.
    pub matrix_width: usize,
    /// Commitment ring degree `D`.
    pub ring_degree: LabiniusRingDegree,
    /// Commitment prime `q`.
    pub commitment_modulus: u32,
    /// Enforced response coefficient magnitude `B_z`.
    pub response_bound: u128,
}

impl LabiniusCommitmentLift<'_> {
    fn overflow() -> AkitaError {
        AkitaError::InvalidSetup("LaBinius commitment lift bound overflow".into())
    }

    /// `m * D * B_z`.
    fn response_mass(&self) -> Option<u128> {
        let degree = usize::try_from(self.ring_degree.degree()).ok()?;
        u128::try_from(checked::product([self.matrix_width, degree])?)
            .ok()?
            .checked_mul(self.response_bound)
    }

    /// `C * factor`.
    fn column_mass(&self, factor: u64) -> Option<u128> {
        u128::try_from(self.fold_columns)
            .ok()?
            .checked_mul(u128::from(factor))
    }

    /// Honest carry magnitude `3 * (m * D * B_z + C * w)`.
    pub fn honest_carry_bound(&self) -> Result<u128, AkitaError> {
        self.response_mass()
            .zip(self.column_mass(self.challenge.coefficient_l1_bound()))
            .and_then(|(response, challenge)| response.checked_add(challenge)?.checked_mul(3))
            .ok_or_else(Self::overflow)
    }

    /// Admit the enforced carry magnitude `carry_bound` under `proof_prime`:
    /// it covers the honest carry and the lifted relation cannot wrap.
    pub fn check_no_wrap(&self, proof_prime: u128, carry_bound: u128) -> Result<(), AkitaError> {
        if carry_bound < self.honest_carry_bound()? {
            return Err(AkitaError::InvalidSetup(
                "LaBinius commitment carry envelope is below the honest bound".into(),
            ));
        }
        let modulus = u128::from(self.commitment_modulus);
        let (_, image_bound) = labinius_image_digit_bound(self.commitment_modulus);
        let response_term = self
            .response_mass()
            .and_then(|mass| mass.checked_mul(3)?.checked_mul(modulus.checked_sub(1)?));
        let image_term = self
            .column_mass(self.challenge.multiplication_linf_operator_bound())
            .and_then(|mass| mass.checked_mul(image_bound));
        let total = response_term
            .zip(image_term)
            .zip(modulus.checked_mul(carry_bound))
            .and_then(|((response, image), carry)| response.checked_add(image)?.checked_add(carry))
            .ok_or_else(Self::overflow)?;
        if total >= proof_prime {
            return Err(AkitaError::InvalidSetup(
                "LaBinius commitment lift does not satisfy \
                 3*m*D*(q-1)*B_z + C*Gamma*B_T + q*B_K < P"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "commitment_lift_tests.rs"]
mod tests;
