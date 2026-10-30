//! Integer lift admission for the committed binary parity and value carries.

use akita_challenges::FoldChallengeFamily;
use akita_error::{checked, AkitaError};
use akita_params::sis::checked_balanced_digit_representable_bounds;

/// Fixed carry depths for the admitted terminal geometry.
pub const BINARY_PARITY_CARRY_DIGITS: usize = 9;
pub const BINARY_VALUE_CARRY_DIGITS: usize = 6;

/// Checked completeness and exact-lift bounds, using the full malicious U range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryEvaluationBounds {
    pub parity_honest_carry: u128,
    pub value_honest_carry: u128,
    pub parity_carry_bound: u128,
    pub value_carry_bound: u128,
    pub parity_no_wrap_bound: u128,
    pub value_no_wrap_bound: u128,
    pub parity_margin: u128,
    pub value_margin: u128,
}

impl BinaryEvaluationBounds {
    /// Admit scalar rows M and columns C only when both fixed digit depths fit.
    ///
    /// `Hpar=162*M*4+C*w*4`; the reduced parity residual is bounded by
    /// `3*Hpar`. Value verification permits every U digit in [-4,3], even
    /// though honest U is bit-valued. Neither inequality assumes U booleanity.
    pub fn new(
        scalar_rows: usize,
        columns: usize,
        proof_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let bad = || AkitaError::InvalidSetup("binary evaluation lift admission failed".into());
        if !scalar_rows.is_power_of_two() || !columns.is_power_of_two() || scalar_rows < 4 {
            return Err(bad());
        }
        checked::product([scalar_rows, columns, 162]).ok_or_else(bad)?;
        let hpar = (scalar_rows as u128)
            .checked_mul(162 * 4)
            .zip(
                (columns as u128).checked_mul(FoldChallengeFamily::BinaryRoot.weight() as u128 * 4),
            )
            .and_then(|(a, b)| a.checked_add(b))
            .ok_or_else(bad)?;
        let parity_residual = hpar.checked_mul(3).ok_or_else(bad)?;
        let parity_honest_carry = parity_residual / 2;
        let value_honest_carry = (columns as u128)
            .checked_mul(3 * 162)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(bad)?
            / 2;
        let (pk, positive_k) =
            checked_balanced_digit_representable_bounds(3, BINARY_PARITY_CARRY_DIGITS);
        let (pv, positive_v) =
            checked_balanced_digit_representable_bounds(3, BINARY_VALUE_CARRY_DIGITS);
        let parity_carry_bound = pk.ok_or_else(bad)?;
        let value_carry_bound = pv.ok_or_else(bad)?;
        if parity_honest_carry > positive_k.ok_or_else(bad)?
            || value_honest_carry > positive_v.ok_or_else(bad)?
        {
            return Err(bad());
        }
        let parity_no_wrap_bound = parity_carry_bound
            .checked_mul(2)
            .and_then(|x| x.checked_add(parity_residual))
            .ok_or_else(bad)?;
        let value_no_wrap_bound = (columns as u128)
            .checked_mul(3 * 162 * 4)
            .and_then(|x| x.checked_add(1))
            .zip(value_carry_bound.checked_mul(2))
            .and_then(|(a, b)| a.checked_add(b))
            .ok_or_else(bad)?;
        if parity_no_wrap_bound >= proof_modulus || value_no_wrap_bound >= proof_modulus {
            return Err(bad());
        }
        Ok(Self {
            parity_honest_carry,
            value_honest_carry,
            parity_carry_bound,
            value_carry_bound,
            parity_no_wrap_bound,
            value_no_wrap_bound,
            parity_margin: proof_modulus - parity_no_wrap_bound,
            value_margin: proof_modulus - value_no_wrap_bound,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const P: u128 = u128::MAX - 274;

    #[test]
    fn full_malicious_digit_envelopes_match_integer_calculation() {
        let bounds = BinaryEvaluationBounds::new(16_384, 256, P).unwrap();
        assert_eq!(bounds.parity_honest_carry, 15_995_904);
        assert_eq!(bounds.parity_carry_bound, 76_695_844);
        assert_eq!(bounds.value_carry_bound, 149_796);
        assert_eq!(bounds.value_honest_carry, 62_208);
        assert_eq!(bounds.parity_no_wrap_bound, 185_383_496);
        assert_eq!(bounds.value_no_wrap_bound, 797_257);
        assert_eq!(bounds.parity_margin, P - 185_383_496);
        assert_eq!(bounds.value_margin, P - 797_257);
        assert_eq!(
            BinaryEvaluationBounds::new(32_768, 256, P)
                .unwrap()
                .parity_no_wrap_bound,
            217_233_992
        );
        assert!(BinaryEvaluationBounds::new(65_536, 256, P).is_err());
    }

    #[test]
    fn strict_thresholds_and_overflow_reject() {
        let bounds = BinaryEvaluationBounds::new(4, 2, P).unwrap();
        assert_eq!(bounds.parity_honest_carry, 4_440);
        assert_eq!(bounds.value_honest_carry, 486);
        assert_eq!(bounds.parity_no_wrap_bound, 153_400_568);
        assert_eq!(bounds.value_no_wrap_bound, 303_481);
        assert!(BinaryEvaluationBounds::new(4, 2, bounds.parity_no_wrap_bound).is_err());
        assert_eq!(
            BinaryEvaluationBounds::new(4, 2, bounds.parity_no_wrap_bound + 1)
                .unwrap()
                .parity_margin,
            1
        );
        for (rows, columns) in [(0, 2), (3, 2), (4, 0), (usize::MAX, 2), (4, usize::MAX)] {
            assert!(BinaryEvaluationBounds::new(rows, columns, P).is_err());
        }
    }
}
