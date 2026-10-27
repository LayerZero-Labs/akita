//! Runtime selection of the fold challenge distribution.

use crate::{OperatorNormRejection, SparseChallenge, SparseChallengeConfig};
use akita_error::AkitaError;

/// Extension seam: challenge selection for binary source groups (tracking issue pending).
///
/// The selected family owns its identity, validation and indexed sampling.
/// Serialization preserves the existing signed-sparse configuration representation. Selecting a family is independent of enabling Cargo features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FoldChallengeFamily {
    /// Existing fixed-weight signed-sparse distribution.
    SignedSparse(SparseChallengeConfig),
}

impl serde::Serialize for FoldChallengeFamily {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::SignedSparse(config) => config.serialize(serializer),
        }
    }
}

impl<'de> serde::Deserialize<'de> for FoldChallengeFamily {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        SparseChallengeConfig::deserialize(deserializer).map(Self::SignedSparse)
    }
}

impl From<SparseChallengeConfig> for FoldChallengeFamily {
    fn from(config: SparseChallengeConfig) -> Self {
        Self::SignedSparse(config)
    }
}

impl FoldChallengeFamily {
    /// Select the ordinary ±1-only family.
    pub const fn pm1_only(weight: usize) -> Self {
        Self::SignedSparse(SparseChallengeConfig::pm1_only(weight))
    }

    /// Select the ordinary production family for a ring dimension.
    pub fn production_for_ring_dim(ring_d: usize) -> Option<Self> {
        SparseChallengeConfig::production_for_ring_dim(ring_d).map(Self::SignedSparse)
    }

    /// Inspect a signed-sparse family at a boundary requiring that distribution.
    pub const fn signed_sparse(&self) -> Option<&SparseChallengeConfig> {
        match self {
            Self::SignedSparse(config) => Some(config),
        }
    }

    /// Certified operator-norm policy, if this family supports one.
    pub fn operator_norm_rejection(&self, ring_d: usize) -> Option<OperatorNormRejection> {
        match self {
            Self::SignedSparse(config) => {
                crate::selective_l2_operator_norm_rejection(ring_d, config)
            }
        }
    }

    /// Maximum support weight.
    pub fn weight(&self) -> usize {
        match self {
            Self::SignedSparse(config) => config.weight(),
        }
    }

    /// Maximum coefficient L1 norm.
    pub fn l1_norm(&self) -> usize {
        match self {
            Self::SignedSparse(config) => config.l1_norm(),
        }
    }

    /// Maximum coefficient infinity norm.
    pub fn infinity_norm(&self) -> u32 {
        match self {
            Self::SignedSparse(config) => config.infinity_norm(),
        }
    }

    /// Maximum squared coefficient L2 norm.
    pub fn challenge_l2_sq_max(&self) -> u128 {
        match self {
            Self::SignedSparse(config) => config.challenge_l2_sq_max(),
        }
    }

    /// Canonical family identity bound into the draw context.
    pub fn domain_separator_bytes(&self) -> Vec<u8> {
        match self {
            Self::SignedSparse(config) => config.domain_separator_bytes(),
        }
    }

    /// Validate the family for the selected ring dimension.
    pub fn validate_dyn(&self, ring_d: usize) -> Result<(), &'static str> {
        match self {
            Self::SignedSparse(config) => config.validate_dyn(ring_d),
        }
    }

    /// Validate structure and the ordinary minimum entropy floor.
    pub fn validate_for_ring_dim(&self, ring_d: usize) -> Result<(), &'static str> {
        match self {
            Self::SignedSparse(config) => config.validate_for_ring_dim(ring_d),
        }
    }

    /// Check the per-draw entropy floor for a runtime ring dimension.
    pub fn validate_min_entropy_for_ring_dim(
        &self,
        ring_d: usize,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        match self {
            Self::SignedSparse(config) => {
                config.validate_min_entropy_for_ring_dim(ring_d, required_bits)
            }
        }
    }

    /// Check structure for a statically selected ring dimension.
    pub fn validate<const D: usize>(&self) -> Result<(), &'static str> {
        self.validate_dyn(D)
    }

    /// Raw per-draw entropy in bits.
    pub fn log2_support_bits<const D: usize>(&self) -> f64 {
        match self {
            Self::SignedSparse(config) => config.log2_support_bits::<D>(),
        }
    }

    /// Check a per-draw entropy floor for a static ring dimension.
    pub fn validate_min_entropy<const D: usize>(
        &self,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        match self {
            Self::SignedSparse(config) => config.validate_min_entropy::<D>(required_bits),
        }
    }

    pub(crate) fn validate_draw(
        &self,
        ring_d: usize,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<(), AkitaError> {
        self.validate_dyn(ring_d)
            .map_err(|error| AkitaError::InvalidInput(error.into()))?;
        match self {
            Self::SignedSparse(config) => {
                if ring_d > crate::sampler::MAX_STACK_RING_DIM {
                    return Err(AkitaError::InvalidInput(
                        "ring dimension exceeds supported stack sampler limit".into(),
                    ));
                }
                if let Some(policy) = rejection {
                    policy
                        .validate(ring_d, config)
                        .map_err(|error| AkitaError::InvalidInput(error.into()))?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn sample_indexed(
        &self,
        seed: &[u8; 32],
        ring_d: usize,
        count: usize,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Vec<SparseChallenge>, AkitaError> {
        match self {
            Self::SignedSparse(config) => crate::sampler::sample_indexed_challenges_from_seed(
                seed, ring_d, count, config, rejection,
            ),
        }
    }
}
