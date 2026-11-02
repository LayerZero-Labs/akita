//! Explicit signed source bounds for the experimental mixed native root.

use akita_error::AkitaError;

use crate::sis::{
    num_digits_inner, num_digits_open, CommittedSourceClass, CommittedSourceContract,
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

    pub fn decomposition(self) -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: self.0,
            log_open_bound: Some(128),
        }
    }

    pub fn producer_contract(self) -> Result<CommittedSourceContract, AkitaError> {
        CommittedSourceContract::try_new(
            CommittedSourceClass::BalancedSignedDigit,
            self.decomposition(),
        )
    }

    /// Check the same digit-depth primitive used by ordinary source planning.
    /// The matrices retain their existing audited identities; no rank or SIS
    /// bound is weakened to accommodate a wider source.
    pub fn validate_profile(self, profile: &GroupCommitPhaseParams) -> Result<(), AkitaError> {
        profile.validate_frozen_precommit(128)?;
        if profile.inner.matrix.sis_modulus_profile() != SisModulusProfileId::Q128Offset275
            || profile.outer.matrix.sis_modulus_profile() != SisModulusProfileId::Q128Offset275
            || profile.inner.digits.log_basis != 3
            || profile.outer.digits.log_basis != 3
            || profile.inner.digits.num_digits != num_digits_inner(self.decomposition(), true)
            || profile.outer.digits.num_digits != num_digits_open(self.decomposition())
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
        for (bits, digits) in [(2, 1), (9, 4), (18, 7), (33, 12)] {
            let width = OrdinarySourceWidth::new(bits).unwrap();
            assert_eq!(num_digits_inner(width.decomposition(), true), digits);
            let (negative, positive) = width.producer_contract().unwrap().declared_bounds();
            assert_eq!(negative, Some(width.offset()));
            assert_eq!(positive, Some(width.offset() - 1));
            let (negative, positive) = crate::sis::balanced_digit_representable_bounds(3, digits);
            assert!(negative >= width.offset());
            assert!(positive >= width.offset() - 1);
        }
        for bits in [0, 1, 65, u32::MAX] {
            assert!(OrdinarySourceWidth::new(bits).is_err());
        }
    }
}
