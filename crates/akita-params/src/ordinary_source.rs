//! Explicit signed source bounds for the experimental mixed native root.

use akita_error::AkitaError;

use crate::sis::{
    checked_balanced_digit_representable_bounds, num_digits_inner, num_digits_open,
    CommittedSourceClass, CommittedSourceContract,
};
use crate::{DecompositionParams, GroupCommitPhaseParams, SisModulusProfileId};

/// Exact signed interval, separate from a commitment's gadget capacity.
///
/// This is an arithmetic contract, not production security admission. The
/// native relation authenticates an offset binary expansion of every source
/// coefficient in addition to its existing A/B/compression preimages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrdinarySourceWidth(u32);

impl OrdinarySourceWidth {
    pub fn new(bits: u32) -> Result<Self, AkitaError> {
        if !(2..=64).contains(&bits) {
            return Err(AkitaError::InvalidSetup(
                "native ordinary signed width must lie in 2..=64".into(),
            ));
        }
        Ok(Self(bits))
    }

    pub fn bits(self) -> u32 {
        self.0
    }

    pub fn offset(self) -> u128 {
        1u128 << (self.0 - 1)
    }

    /// Select the actual commitment field and prove the signed link cannot wrap.
    /// This arithmetic check does not admit a native root or a research schedule.
    pub fn decomposition(
        self,
        modulus: SisModulusProfileId,
    ) -> Result<DecompositionParams, AkitaError> {
        let field_bits = match modulus {
            SisModulusProfileId::Q128Offset275 => 128,
            SisModulusProfileId::Q64Offset59 => 64,
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "unsupported ordinary signed source field".into(),
                ))
            }
        };
        let decomposition = DecompositionParams {
            log_basis: 3,
            log_commit_bound: self.0,
            log_open_bound: Some(field_bits),
        };
        // For any range-checked S digits and negative bits encoding -X,
        // S - X + offset lies in [-negative - offset + 1, positive + offset].
        // Bound the malicious digit envelope, not merely honest source values.
        let (negative, positive) =
            checked_balanced_digit_representable_bounds(3, num_digits_inner(decomposition, true));
        let negative = negative.and_then(|bound| bound.checked_add(self.offset() - 1));
        let positive = positive.and_then(|bound| bound.checked_add(self.offset()));
        if [negative, positive]
            .into_iter()
            .any(|bound| bound.is_none_or(|bound| bound >= modulus.modulus()))
        {
            return Err(AkitaError::InvalidSetup(
                "ordinary signed link can wrap in the selected field".into(),
            ));
        }
        Ok(decomposition)
    }

    pub fn producer_contract(
        self,
        modulus: SisModulusProfileId,
    ) -> Result<CommittedSourceContract, AkitaError> {
        CommittedSourceContract::try_new(
            CommittedSourceClass::BalancedSignedDigit,
            self.decomposition(modulus)?,
        )
    }

    /// Check the same digit-depth primitive used by ordinary source planning.
    /// The matrices retain their existing audited identities; no rank or SIS
    /// bound is weakened to accommodate a wider source.
    pub fn validate_profile(self, profile: &GroupCommitPhaseParams) -> Result<(), AkitaError> {
        let modulus = profile.inner.matrix.sis_modulus_profile();
        let decomposition = self.decomposition(modulus)?;
        profile.validate_frozen_precommit(modulus.field_bits())?;
        if profile.outer.matrix.sis_modulus_profile() != modulus
            || profile.inner.digits.log_basis != 3
            || profile.outer.digits.log_basis != 3
            || profile.inner.digits.num_digits != num_digits_inner(decomposition, true)
            || profile.outer.digits.num_digits != num_digits_open(decomposition)
            || profile.inner.matrix.ring_dimension() > 512
            || profile.outer.matrix.ring_dimension() != 64
        {
            return Err(AkitaError::InvalidSetup(
                "ordinary signed source differs from its frozen prime profile".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_widths_use_canonical_balanced_depth_and_exact_endpoints() {
        for bits in 2..=64 {
            let width = OrdinarySourceWidth::new(bits).unwrap();
            assert_eq!(width.bits(), bits);
            let offset = 1u128 << (bits - 1);
            assert_eq!(width.offset(), offset);
            let decomposition = width
                .decomposition(SisModulusProfileId::Q128Offset275)
                .unwrap();
            let digits = num_digits_inner(decomposition, true);
            let (negative, positive) = width
                .producer_contract(SisModulusProfileId::Q128Offset275)
                .unwrap()
                .declared_bounds();
            assert_eq!(negative, Some(offset));
            assert_eq!(positive, Some(offset - 1));
            let (negative, positive) = crate::sis::balanced_digit_representable_bounds(3, digits);
            assert!(negative >= offset);
            assert!(positive >= offset - 1);
            if digits > 1 {
                let (negative, positive) =
                    crate::sis::balanced_digit_representable_bounds(3, digits - 1);
                assert!(negative < offset || positive < offset - 1);
            }
        }
        for (bits, digits) in [(2, 1), (9, 4), (18, 7), (33, 12), (34, 12), (64, 22)] {
            let width = OrdinarySourceWidth::new(bits).unwrap();
            let decomposition = width
                .decomposition(SisModulusProfileId::Q128Offset275)
                .unwrap();
            assert_eq!(num_digits_inner(decomposition, true), digits);
        }
        for bits in [0, 1, 65, u32::MAX] {
            assert!(OrdinarySourceWidth::new(bits).is_err());
        }
    }

    #[test]
    fn p64_signed_contracts_keep_port_widths_and_reject_wrapping_lifts() {
        let modulus = SisModulusProfileId::Q64Offset59;
        for (bits, digits) in [(2, 1), (9, 4), (18, 7), (33, 12), (62, 21)] {
            let width = OrdinarySourceWidth::new(bits).unwrap();
            let decomposition = width.decomposition(modulus).unwrap();
            assert_eq!(num_digits_inner(decomposition, true), digits);
            assert_eq!(num_digits_open(decomposition), 22);
            assert_eq!(decomposition.log_open_bound, Some(64));
            assert_eq!(
                width.producer_contract(modulus).unwrap().declared_bounds(),
                (Some(1u128 << (bits - 1)), Some((1u128 << (bits - 1)) - 1))
            );
        }
        assert!(OrdinarySourceWidth::new(9)
            .unwrap()
            .decomposition(SisModulusProfileId::Q32Offset99)
            .is_err());
        // These are real integer-link counterexamples, not only oversized
        // metadata: S = p - offset and X = 0 satisfy S-X+offset = 0 mod p,
        // while the integer residual is p. Twenty-two base-eight digits can
        // represent S, so a check using only the honest width would miss it.
        for bits in [63, 64] {
            let width = OrdinarySourceWidth::new(bits).unwrap();
            let s = modulus.modulus() - width.offset();
            let mut remaining = s as i128;
            let mut reconstructed = 0i128;
            let mut power = 1i128;
            for _ in 0..22 {
                let digit = (remaining + 4).rem_euclid(8) - 4;
                assert!((-4..=3).contains(&digit));
                reconstructed += digit * power;
                remaining = (remaining - digit) / 8;
                power *= 8;
            }
            assert_eq!(remaining, 0);
            assert_eq!(reconstructed as u128 + width.offset(), modulus.modulus());
            assert!(matches!(
                width.decomposition(modulus),
                Err(AkitaError::InvalidSetup(message))
                    if message == "ordinary signed link can wrap in the selected field"
            ));
            assert!(width.producer_contract(modulus).is_err());
        }
    }
}
