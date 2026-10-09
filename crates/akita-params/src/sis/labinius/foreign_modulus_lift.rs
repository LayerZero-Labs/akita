//! Checked integer admission for a commitment modulus foreign to the opening field.

use akita_challenges::BinaryChallengeProfile;
use akita_error::{checked, AkitaError};

use super::LabiniusRootShape;

/// Provisional policy floor for the reduced-matrix derivation hybrid bias.
/// This is not a derived requirement or a claim about composed security.
pub const LABINIUS_MIN_DERIVATION_BIAS_BITS: u32 = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ForeignModulusLift {
    modulus: u32,
    pub residual_bound: u128,
    pub honest_carry_bound: u128,
    pub carry_len: usize,
    pub bias_bits: u32,
    response_difference_bound: u128,
    gamma: u128,
}

impl ForeignModulusLift {
    pub(super) fn difference_bound(&self, enforced_bits: u32) -> Result<u128, AkitaError> {
        let overflow = || AkitaError::InvalidSetup("LaBinius A-carry envelope overflow".into());
        if !(1..=128).contains(&enforced_bits) {
            return Err(overflow());
        }
        let offset = 1u128.checked_shl(enforced_bits - 1).ok_or_else(overflow)?;
        // The positive endpoint is one smaller than the negative magnitude.
        if offset.checked_sub(1).ok_or_else(overflow)? < self.honest_carry_bound {
            return Err(AkitaError::InvalidSetup(
                "LaBinius A-carry envelope omits an honest endpoint".into(),
            ));
        }
        let diameter = offset
            .checked_sub(1)
            .and_then(|v| v.checked_add(offset))
            .ok_or_else(overflow)?;
        u128::from(self.modulus)
            .checked_mul(diameter)
            .and_then(|v| v.checked_add(self.response_difference_bound))
            .ok_or_else(overflow)
    }

    pub(super) fn check_no_wrap(&self, prime: u128, enforced_bits: u32) -> Result<(), AkitaError> {
        let total = self
            .difference_bound(enforced_bits)?
            .checked_mul(self.gamma)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("LaBinius extraction no-wrap overflow".into())
            })?;
        if total >= prime {
            return Err(AkitaError::InvalidSetup(
                "LaBinius A-carry does not satisfy 4*Gamma_inf*B_nu < P".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn derive_foreign_modulus_lift(
    shape: &LabiniusRootShape,
    modulus: u32,
    challenge: &BinaryChallengeProfile,
) -> Result<ForeignModulusLift, AkitaError> {
    let overflow =
        || AkitaError::InvalidSetup("LaBinius foreign-modulus lift bound overflow".into());
    let q = u128::from(modulus);
    if shape.eta_a.checked_mul(2).ok_or_else(overflow)? >= q {
        return Err(AkitaError::InvalidSetup(
            "LaBinius lift does not satisfy 2*eta_A < q0".into(),
        ));
    }
    let (lower, upper) = shape.response_interval;
    let diameter = upper
        .checked_sub(lower)
        .and_then(|v| u128::try_from(v).ok())
        .ok_or_else(overflow)?;
    let coefficient_count =
        checked::product([shape.ring_elements_per_column, shape.commitment_degree])
            .and_then(|v| u128::try_from(v).ok())
            .ok_or_else(overflow)?;
    let residual_bound = coefficient_count
        .checked_mul(lower.unsigned_abs().max(upper.unsigned_abs()))
        .and_then(|v| {
            u128::try_from(shape.fold_width)
                .ok()?
                .checked_mul(u128::from(challenge.coefficient_l1_bound()))?
                .checked_add(v)
        })
        .and_then(|v| v.checked_mul(q.checked_sub(1)?))
        .ok_or_else(overflow)?;
    if residual_bound.checked_mul(6).ok_or_else(overflow)? >= shape.coefficient_prime.modulus() {
        return Err(AkitaError::InvalidSetup(
            "LaBinius lift does not satisfy 6*H_A < P".into(),
        ));
    }
    let honest_carry_bound = residual_bound.checked_mul(3).ok_or_else(overflow)? / q;
    let carry_len = checked::product([
        usize::try_from(shape.rank_a).map_err(|_| overflow())?,
        shape.commitment_degree,
    ])
    .ok_or_else(overflow)?;
    let coefficients = coefficient_count
        .checked_mul(u128::from(shape.rank_a))
        .ok_or_else(overflow)?;
    let denominator = coefficients.checked_mul(q).ok_or_else(overflow)?;
    let bias_bits = derivation_bias_bits(shape.coefficient_prime.modulus(), denominator)
        .ok_or_else(overflow)?;
    if bias_bits < LABINIUS_MIN_DERIVATION_BIAS_BITS {
        return Err(AkitaError::InvalidSetup(
            "LaBinius reduced derivation bias is below policy floor".into(),
        ));
    }
    let response_difference_bound = coefficient_count
        .checked_mul(q.checked_sub(1).ok_or_else(overflow)?)
        .and_then(|v| v.checked_mul(diameter))
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(overflow)?;
    Ok(ForeignModulusLift {
        modulus,
        residual_bound,
        honest_carry_bound,
        carry_len,
        bias_bits,
        response_difference_bound,
        gamma: u128::from(challenge.multiplication_linf_operator_bound()),
    })
}

// Compare the 130-bit integer 4P with denominator*2^t without overflowing u128.
// The high limb is at most three while the comparison succeeds.
fn derivation_bias_bits(prime: u128, denominator: u128) -> Option<u32> {
    if denominator == 0 {
        return None;
    }
    let threshold = (prime >> 126, prime << 2);
    let mut value = (0u128, denominator);
    let mut bits = None;
    for t in 0..=130 {
        if value > threshold {
            break;
        }
        bits = Some(t);
        let (low, carry) = value.1.overflowing_mul(2);
        value = (value.0.checked_mul(2)?.checked_add(u128::from(carry))?, low);
    }
    bits
}

#[cfg(test)]
#[path = "foreign_modulus_lift_tests.rs"]
mod tests;
