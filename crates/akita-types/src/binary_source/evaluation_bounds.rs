//! Integer lift admission for the committed binary parity and value carries.
//! The selected profile supplies its base characteristic, including for
//! extension challenges: the extension cardinality is never a lift modulus.

use akita_challenges::FoldChallengeDistribution;
use akita_error::AkitaError;
use akita_params::sis::checked_balanced_digit_representable_bounds;
use jolt_algebra::checked;

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
                (columns as u128)
                    .checked_mul(FoldChallengeDistribution::BinaryRoot.weight() as u128 * 4),
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

    fn expected_envelopes(rows: usize, columns: usize) -> BinaryEvaluationBounds {
        let reach = |digits, digit| (0..digits).map(|i| digit * 8u128.pow(i)).sum::<u128>();
        // Reduce each monomial modulo X^162 + X^81 + 1 and count the
        // unreduced coefficients that can contribute to any output coefficient.
        let mut fan_in = [0u128; 162];
        for power in 0..2 * 162 - 1 {
            if power < 162 {
                fan_in[power] += 1;
            } else if power < 243 {
                fan_in[power - 162] += 1;
                fan_in[power - 81] += 1;
            } else {
                fan_in[power - 243] += 1;
            }
        }
        let reduction = *fan_in.iter().max().unwrap();
        let weight = FoldChallengeDistribution::BinaryRoot.weight() as u128;
        let parity_residual = reduction * (162 * rows as u128 * 4 + columns as u128 * weight * 4);
        let honest_value_residual = reduction * 162 * columns as u128 + 1;
        // Soundness includes every malicious U digit in [-4,3], not just bits.
        let malicious_value_residual = reduction * 162 * columns as u128 * 4 + 1;
        let parity_carry_bound = reach(BINARY_PARITY_CARRY_DIGITS as u32, 4);
        let value_carry_bound = reach(BINARY_VALUE_CARRY_DIGITS as u32, 4);
        let parity_no_wrap_bound = parity_residual + 2 * parity_carry_bound;
        let value_no_wrap_bound = malicious_value_residual + 2 * value_carry_bound;
        BinaryEvaluationBounds {
            parity_honest_carry: parity_residual / 2,
            value_honest_carry: honest_value_residual / 2,
            parity_carry_bound,
            value_carry_bound,
            parity_no_wrap_bound,
            value_no_wrap_bound,
            parity_margin: P - parity_no_wrap_bound,
            value_margin: P - value_no_wrap_bound,
        }
    }

    #[test]
    fn full_malicious_digit_envelopes_match_integer_calculation() {
        for rows in [16_384, 32_768] {
            let expected = expected_envelopes(rows, 256);
            assert_eq!(BinaryEvaluationBounds::new(rows, 256, P).unwrap(), expected);
            let positive = (0..BINARY_PARITY_CARRY_DIGITS)
                .map(|i| 3 * 8u128.pow(i as u32))
                .sum::<u128>();
            assert!(expected.parity_honest_carry <= positive);
        }
        let next = expected_envelopes(65_536, 256);
        let positive = (0..BINARY_PARITY_CARRY_DIGITS)
            .map(|i| 3 * 8u128.pow(i as u32))
            .sum::<u128>();
        assert!(next.parity_honest_carry > positive);
        assert!(BinaryEvaluationBounds::new(65_536, 256, P).is_err());
    }

    #[test]
    fn strict_thresholds_and_overflow_reject() {
        let expected = expected_envelopes(4, 2);
        let bounds = BinaryEvaluationBounds::new(4, 2, P).unwrap();
        assert_eq!(bounds, expected);
        assert!(BinaryEvaluationBounds::new(4, 2, expected.parity_no_wrap_bound).is_err());
        assert_eq!(
            BinaryEvaluationBounds::new(4, 2, expected.parity_no_wrap_bound + 1)
                .unwrap()
                .parity_margin,
            1
        );
        for (rows, columns) in [(0, 2), (3, 2), (4, 0), (usize::MAX, 2), (4, usize::MAX)] {
            assert!(BinaryEvaluationBounds::new(rows, columns, P).is_err());
        }
    }
}
