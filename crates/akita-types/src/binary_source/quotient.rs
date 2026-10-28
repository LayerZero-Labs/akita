//! Exact admission arithmetic for the committed q24 quotient.

use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing, FoldChallengeFamily};
use akita_error::{checked, AkitaError};
use akita_params::sis::{checked_balanced_digit_representable_bounds, num_digits_for_linf_cap};

use super::{
    BINARY_RESPONSE_INTERVAL, BINARY_SOURCE_Q, CANONICAL_IMAGE_WINDOW, ENFORCED_IMAGE_WINDOW,
};

/// Enforced carry range and integer no-wrap admission for `R_A = q K_A`.
///
/// These are arithmetic bounds, not a security certificate. The honest carry
/// uses canonical centered image representatives; the residual uses the full
/// image interval admitted by the balanced digits. The quotient must itself
/// be committed and range checked in the ordinary outgoing witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryQuotientBounds {
    honest_carry_bound: u128,
    digit_count: usize,
    carry_bound: u128,
    residual_bound: u128,
    no_wrap_bound: u128,
    margin: u128,
}

impl BinaryQuotientBounds {
    /// Admit nonempty degree-648 source geometry iff `L_R + q B_KA < P`.
    ///
    /// The exact trinomial product coefficient kernel has row weight 972.
    /// Thus the honest carry is
    /// `ceil(B_A * (972*m*B_Z + C*Gamma) / q)`, while
    /// `L_R = 972*m*B_A*B_Z + C*Gamma*B_T` uses the enforced image range.
    /// Balanced base-eight digits have negative reach `4*(8^d-1)/7` and
    /// positive reach `3*(8^d-1)/7`; both must cover the honest carry.
    pub fn new(
        source_rows: usize,
        columns: usize,
        proof_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let overflow = || AkitaError::InvalidSetup("binary quotient bound overflow".into());
        if source_rows == 0 || columns == 0 {
            return Err(AkitaError::InvalidSetup(
                "binary quotient geometry must be nonempty".into(),
            ));
        }
        // Reject unrepresentable coefficient-table sizes before bound arithmetic.
        checked::product([source_rows, 648]).ok_or_else(overflow)?;
        checked::product([columns, 648]).ok_or_else(overflow)?;
        let challenge = BinaryChallengeProfile::bounded_weight(
            BinaryScalarRing::Cyclotomic243,
            FoldChallengeFamily::BinaryRoot.weight(),
        )?;
        let gamma = u128::from(challenge.multiplication_linf_operator_bound());
        let matrix_bound = CANONICAL_IMAGE_WINDOW.1 as u128;
        let response_bound = u128::from(BINARY_RESPONSE_INTERVAL.0.unsigned_abs());
        let image_bound = u128::from(ENFORCED_IMAGE_WINDOW.0.unsigned_abs());
        let q = u128::from(BINARY_SOURCE_Q);
        let response_mass = 972u128
            .checked_mul(source_rows as u128)
            .and_then(|mass| mass.checked_mul(response_bound))
            .ok_or_else(overflow)?;
        let column_mass = (columns as u128).checked_mul(gamma).ok_or_else(overflow)?;
        let numerator = response_mass
            .checked_add(column_mass)
            .and_then(|mass| mass.checked_mul(matrix_bound))
            .ok_or_else(overflow)?;
        let honest_carry_bound = (numerator / q)
            .checked_add(u128::from(!numerator.is_multiple_of(q)))
            .ok_or_else(overflow)?;
        let digit_count = num_digits_for_linf_cap(honest_carry_bound, u128::BITS, 3);
        let (negative, positive) = checked_balanced_digit_representable_bounds(3, digit_count);
        let carry_bound = negative.ok_or_else(overflow)?;
        if positive.ok_or_else(overflow)? < honest_carry_bound || carry_bound < honest_carry_bound {
            return Err(overflow());
        }
        let residual_bound = response_mass
            .checked_mul(matrix_bound)
            .zip(column_mass.checked_mul(image_bound))
            .and_then(|(response, image)| response.checked_add(image))
            .ok_or_else(overflow)?;
        let no_wrap_bound = q
            .checked_mul(carry_bound)
            .and_then(|carry| residual_bound.checked_add(carry))
            .ok_or_else(overflow)?;
        if no_wrap_bound >= proof_modulus {
            return Err(AkitaError::InvalidSetup(
                "binary quotient requires L_R + q B_KA < P".into(),
            ));
        }
        Ok(Self {
            honest_carry_bound,
            digit_count,
            carry_bound,
            residual_bound,
            no_wrap_bound,
            margin: proof_modulus - no_wrap_bound,
        })
    }

    pub fn honest_carry_bound(&self) -> u128 {
        self.honest_carry_bound
    }
    pub fn digit_count(&self) -> usize {
        self.digit_count
    }
    pub fn carry_bound(&self) -> u128 {
        self.carry_bound
    }
    pub fn residual_bound(&self) -> u128 {
        self.residual_bound
    }
    pub fn no_wrap_bound(&self) -> u128 {
        self.no_wrap_bound
    }
    /// Positive distance from the strict no-wrap threshold to the proof modulus.
    pub fn margin(&self) -> u128 {
        self.margin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u128 = u128::MAX - 274;

    #[test]
    fn small_geometry_matches_independent_integer_calculation() {
        let bounds = BinaryQuotientBounds::new(1, 2, P).unwrap();
        assert_eq!(bounds.honest_carry_bound(), 1_137_332);
        assert_eq!(bounds.digit_count(), 8);
        assert_eq!(bounds.carry_bound(), 9_586_980);
        assert_eq!(bounds.residual_bound(), 16_211_309_718_240);
        assert_eq!(bounds.no_wrap_bound(), 152_858_456_733_060);
        assert_eq!(
            bounds.margin(),
            340_282_366_920_938_463_463_374_454_573_311_478_121
        );
        // Seven digits have insufficient positive reach; eight cover both signs.
        assert!(3 * (8u128.pow(7) - 1) / 7 < bounds.honest_carry_bound());
        assert!(3 * (8u128.pow(8) - 1) / 7 >= bounds.honest_carry_bound());
    }

    #[test]
    fn larger_geometry_matches_independent_integer_calculation() {
        let bounds = BinaryQuotientBounds::new(4096, 256, P).unwrap();
        assert_eq!(bounds.honest_carry_bound(), 4_658_146_490);
        assert_eq!(bounds.digit_count(), 12);
        assert_eq!(bounds.carry_bound(), 39_268_272_420);
        assert_eq!(bounds.residual_bound(), 66_394_525_036_769_280);
        assert_eq!(bounds.no_wrap_bound(), 626_101_272_562_449_060);
        assert_eq!(
            bounds.margin(),
            340_282_366_920_938_463_462_748_506_159_205_762_121
        );
        assert!(3 * (8u128.pow(11) - 1) / 7 < bounds.honest_carry_bound());
        assert!(3 * (8u128.pow(12) - 1) / 7 >= bounds.honest_carry_bound());
    }

    #[test]
    fn no_wrap_is_strict_and_checks_the_carry_envelope() {
        let threshold = 152_858_456_733_060;
        assert!(BinaryQuotientBounds::new(1, 2, threshold).is_err());
        assert!(BinaryQuotientBounds::new(1, 2, threshold - 1).is_err());
        assert_eq!(
            BinaryQuotientBounds::new(1, 2, threshold + 1)
                .unwrap()
                .margin(),
            1
        );
        assert!(BinaryQuotientBounds::new(1, 2, 16_211_309_718_241).is_err());
    }

    #[test]
    fn invalid_geometry_rejects_without_wrapping() {
        for (rows, columns) in [(0, 2), (1, 0), (usize::MAX, 2), (1, usize::MAX)] {
            assert!(BinaryQuotientBounds::new(rows, columns, P).is_err());
        }
        assert!(BinaryQuotientBounds::new(1, 2, 0).is_err());
    }
}
