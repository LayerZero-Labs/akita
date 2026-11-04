//! Exact parity-injective challenge families for LaBinius-style binary folds.
//!
//! A family contains one signed ternary lift of each admitted binary support.
//! Signs are a deterministic function of the complete canonical support and
//! profile identity, so they do not add challenge entropy.

mod combinatorics;
mod position_sample;
mod profile;
pub(crate) mod root_family;
mod sampler;
mod sign;
mod xof;

use akita_error::AkitaError;
use smallvec::SmallVec;

pub use profile::{
    BinaryChallengeFamily, BinaryChallengeProfile, BinaryScalarRing, BinarySignRule,
};
pub use sampler::BinaryChallengeSampler;
pub(crate) use sampler::BINARY_CHALLENGE_SEED_LEN;

use sign::coefficient_signs;

/// Inline term capacity covering the first degree-162 production candidates
/// and the degree-486 reference weights without per-challenge allocation.
pub const INLINE_BINARY_WEIGHT: usize = 64;

/// One nonzero coefficient in a canonical binary challenge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BinaryChallengeTerm {
    /// Power-basis coefficient position.
    pub position: u16,
    /// Deterministic coefficient, always `-1` or `1`.
    pub coefficient: i8,
}

/// A parity-injective binary-fold challenge in canonical position order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryChallenge {
    terms: SmallVec<[BinaryChallengeTerm; INLINE_BINARY_WEIGHT]>,
}

impl BinaryChallenge {
    pub(crate) fn from_support(
        profile: &BinaryChallengeProfile,
        support: &[u16],
        sign_input: &mut Vec<u8>,
        sign_bytes: &mut Vec<u8>,
    ) -> Self {
        coefficient_signs(profile, support, sign_input, sign_bytes);
        let terms = support
            .iter()
            .copied()
            .zip(sign_bytes.iter().copied())
            .map(|(position, bit)| BinaryChallengeTerm {
                position,
                coefficient: if bit & 1 == 0 { 1 } else { -1 },
            })
            .collect();
        Self { terms }
    }

    /// Convert an already drawn sparse challenge without resampling or changing signs.
    ///
    /// The canonical support order, family bound and deterministic sign rule are
    /// checked against `profile` before the arithmetic kernel can use the terms.
    pub fn try_from_sparse(
        challenge: &crate::SparseChallenge,
        profile: &BinaryChallengeProfile,
    ) -> Result<Self, AkitaError> {
        challenge.validate_dyn(profile.scalar_ring().degree())?;
        let mut terms = SmallVec::new();
        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            terms.push(BinaryChallengeTerm {
                position: u16::try_from(position).map_err(|_| {
                    AkitaError::InvalidInput("binary challenge position is out of range".into())
                })?,
                coefficient,
            });
        }
        let converted = Self { terms };
        converted.validate(profile)?;
        Ok(converted)
    }

    /// Return the canonical nonzero terms.
    #[inline]
    #[must_use]
    pub fn terms(&self) -> &[BinaryChallengeTerm] {
        &self.terms
    }

    /// Return the support weight.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> usize {
        self.terms.len()
    }

    /// Exact coefficient infinity norm.
    #[inline]
    #[must_use]
    pub fn coefficient_linf_norm(&self) -> u8 {
        u8::from(!self.terms.is_empty())
    }

    /// Exact coefficient L1 norm.
    #[inline]
    #[must_use]
    pub fn coefficient_l1_norm(&self) -> u64 {
        self.terms.len() as u64
    }

    /// Exact squared coefficient L2 norm.
    #[inline]
    #[must_use]
    pub fn coefficient_l2_squared_norm(&self) -> u64 {
        self.terms.len() as u64
    }

    /// Canonical little-endian support bitset.
    pub fn canonical_support_encoding(
        &self,
        profile: &BinaryChallengeProfile,
    ) -> Result<Vec<u8>, AkitaError> {
        self.validate(profile)?;
        let mut encoding = vec![0u8; profile.scalar_ring().degree().div_ceil(8)];
        for term in &self.terms {
            let position = usize::from(term.position);
            let byte = encoding.get_mut(position / 8).ok_or_else(|| {
                AkitaError::InvalidInput("binary challenge position is out of range".into())
            })?;
            *byte |= 1 << (position % 8);
        }
        Ok(encoding)
    }

    /// Validate canonical order, profile weight, and deterministic sign replay.
    pub fn validate(&self, profile: &BinaryChallengeProfile) -> Result<(), AkitaError> {
        let degree = profile.scalar_ring().degree();
        let weight = self.terms.len();
        match profile.family() {
            BinaryChallengeFamily::FixedWeight if weight != profile.weight_cap() => {
                return Err(AkitaError::InvalidInput(
                    "binary fixed-weight challenge has the wrong support size".into(),
                ));
            }
            BinaryChallengeFamily::BoundedWeight if weight > profile.weight_cap() => {
                return Err(AkitaError::InvalidInput(
                    "binary bounded-weight challenge exceeds its support cap".into(),
                ));
            }
            _ => {}
        }

        let mut previous = None;
        let mut support = SmallVec::<[u16; INLINE_BINARY_WEIGHT]>::new();
        for term in &self.terms {
            let position = usize::from(term.position);
            if position >= degree {
                return Err(AkitaError::InvalidInput(
                    "binary challenge position is out of range".into(),
                ));
            }
            if previous.is_some_and(|last| term.position <= last) {
                return Err(AkitaError::InvalidInput(
                    "binary challenge positions are not in canonical order".into(),
                ));
            }
            if !matches!(term.coefficient, -1 | 1) {
                return Err(AkitaError::InvalidInput(
                    "binary challenge coefficient is not a deterministic sign".into(),
                ));
            }
            previous = Some(term.position);
            support.push(term.position);
        }

        let mut sign_input = Vec::new();
        let mut sign_bytes = Vec::new();
        coefficient_signs(profile, &support, &mut sign_input, &mut sign_bytes);
        if self
            .terms
            .iter()
            .zip(sign_bytes)
            .any(|(term, bit)| term.coefficient != if bit & 1 == 0 { 1 } else { -1 })
        {
            return Err(AkitaError::InvalidInput(
                "binary challenge signs do not match the profile rule".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_adapter_preserves_terms_and_rejects_noncanonical_challenges() {
        let profile =
            BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 46).unwrap();
        let original = BinaryChallengeSampler::new(profile.clone())
            .sample_from_seed(&[17; 32], 1)
            .unwrap()
            .remove(0);
        let mut sparse = crate::SparseChallenge {
            positions: original
                .terms()
                .iter()
                .map(|term| u32::from(term.position))
                .collect(),
            coeffs: original
                .terms()
                .iter()
                .map(|term| term.coefficient)
                .collect(),
        };
        assert_eq!(
            BinaryChallenge::try_from_sparse(&sparse, &profile).unwrap(),
            original
        );
        sparse.coeffs[0] *= -1;
        assert!(BinaryChallenge::try_from_sparse(&sparse, &profile).is_err());
        sparse.coeffs[0] *= -1;
        sparse.positions[0] = 162;
        assert!(BinaryChallenge::try_from_sparse(&sparse, &profile).is_err());
        sparse.positions[0] = original.terms()[0].position.into();
        sparse.coeffs.pop();
        assert!(BinaryChallenge::try_from_sparse(&sparse, &profile).is_err());
    }

    #[test]
    fn canonical_profile_and_sign_fixture_is_stable() {
        let profile =
            BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 3).unwrap();
        assert_eq!(
            profile.identity_bytes(),
            &[
                1, 0, 0, 0, 0, 162, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 0, 224, 156,
                10,
            ]
        );
        let mut sign_input = Vec::new();
        let mut sign_bytes = Vec::new();
        let challenge = BinaryChallenge::from_support(
            &profile,
            &[0, 81, 161],
            &mut sign_input,
            &mut sign_bytes,
        );
        assert_eq!(
            challenge
                .terms()
                .iter()
                .map(|term| (term.position, term.coefficient))
                .collect::<Vec<_>>(),
            vec![(0, -1), (81, 1), (161, 1)]
        );
        assert_eq!(
            challenge.canonical_support_encoding(&profile).unwrap(),
            vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
        );
    }
}
