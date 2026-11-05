//! The challenge family one fold group draws from (protocol v8), and the
//! runtime selection of its distribution.
//!
//! The family lives here, below `akita-types`, because the schedule prices a
//! fold from it; the draw itself (`akita_types::FoldDraw`) squeezes one
//! challenge message per group and decodes it with this family.

use akita_error::AkitaError;
use jolt_crypto::challenge::{
    OperatorNormRejection, SparseChallengeConfig, SparseChallengeShell, SparseChallengeStream,
};

/// Algebraic domain of one fold-challenge draw.
///
/// Coefficient packing draws in a challenge subring whose dimension must equal
/// the family's ring dimension. Like the family, the domain is bound by the
/// descriptor, not absorbed by the draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldChallengeDrawDomain {
    EvaluationTrace,
    SubringCoefficientPacking { challenge_subring_dimension: usize },
}

/// The challenge family one fold group draws from: its algebraic domain,
/// ring dimension, distribution and operator-norm policy.
///
/// Schedules build it once; the grinding plan prices the fold from it and
/// every draw decodes with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldChallengeFamily {
    domain: FoldChallengeDrawDomain,
    shell: FamilyShell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FamilyShell {
    SignedSparse(SparseChallengeShell),
    /// Decoded from one 32-byte message by indexed SHAKE256 substreams.
    #[cfg(feature = "labinius-challenges")]
    BinaryRoot { ring_d: usize },
}

impl FoldChallengeFamily {
    /// Validate one family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when a coefficient-packing family
    /// disagrees with its subring dimension or carries an operator-norm
    /// policy, and [`AkitaError::InvalidSetup`] for an unsupported shell,
    /// distribution or policy.
    pub fn new(
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        distribution: impl Into<FoldChallengeDistribution>,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, AkitaError> {
        if let FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension,
        } = domain
        {
            if ring_d != challenge_subring_dimension {
                return Err(AkitaError::InvalidInput(
                    "coefficient-packing draw dimension mismatch".into(),
                ));
            }
            if rejection.is_some() {
                return Err(AkitaError::InvalidInput(
                    "coefficient-packing draws require the L-infinity security route".into(),
                ));
            }
        }
        let shell = match distribution.into() {
            FoldChallengeDistribution::SignedSparse(config) => {
                FamilyShell::SignedSparse(SparseChallengeShell::new(ring_d, &config, rejection)?)
            }
            #[cfg(feature = "labinius-challenges")]
            FoldChallengeDistribution::BinaryRoot => {
                if rejection.is_some() {
                    return Err(AkitaError::InvalidSetup(
                        "binary root family has no operator-norm rejection policy".into(),
                    ));
                }
                akita_challenges::root_family::validate(ring_d, 0)
                    .map_err(|error| AkitaError::InvalidSetup(error.into()))?;
                FamilyShell::BinaryRoot { ring_d }
            }
        };
        Ok(Self { domain, shell })
    }

    /// Message layout and decoder for `count` challenges of a signed-sparse
    /// family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `count` is zero, the
    /// message exceeds the message limit, or the family is not signed-sparse.
    pub fn stream(&self, count: usize) -> Result<SparseChallengeStream, AkitaError> {
        match self.shell {
            FamilyShell::SignedSparse(shell) => Ok(shell.stream(count)?),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => Err(AkitaError::InvalidSetup(
                "binary root challenges have no sparse message stream".into(),
            )),
        }
    }

    /// Byte length of the one message that `count` challenges decode from.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `count` is zero or the
    /// message exceeds the message limit.
    pub fn message_len(&self, count: usize) -> Result<usize, AkitaError> {
        match self.shell {
            FamilyShell::SignedSparse(shell) => Ok(shell.stream(count)?.message_len()),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => {
                if count == 0 {
                    return Err(AkitaError::InvalidSetup(
                        "binary root challenge count must be positive".into(),
                    ));
                }
                Ok(akita_challenges::BINARY_CHALLENGE_SEED_LEN)
            }
        }
    }

    /// Guaranteed `log2` size of the set each challenge is uniform on, for
    /// any challenge count: `K` for an unfiltered family, the certified
    /// exponent for an operator-norm one, and the floor of the exact support
    /// entropy for the binary root family.
    #[must_use]
    pub fn set_bits(&self) -> u32 {
        match self.shell {
            FamilyShell::SignedSparse(shell) => shell.set_bits(),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { ring_d } => {
                akita_challenges::root_family::entropy_bits(ring_d).floor() as u32
            }
        }
    }

    #[must_use]
    pub const fn domain(&self) -> FoldChallengeDrawDomain {
        self.domain
    }

    #[must_use]
    pub const fn ring_d(&self) -> usize {
        match self.shell {
            FamilyShell::SignedSparse(shell) => shell.ring_d(),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { ring_d } => ring_d,
        }
    }

    #[must_use]
    pub const fn distribution(&self) -> FoldChallengeDistribution {
        match self.shell {
            FamilyShell::SignedSparse(shell) => {
                FoldChallengeDistribution::SignedSparse(shell.config())
            }
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => FoldChallengeDistribution::BinaryRoot,
        }
    }

    #[must_use]
    pub const fn rejection(&self) -> Option<OperatorNormRejection> {
        match self.shell {
            FamilyShell::SignedSparse(shell) => shell.rejection(),
            #[cfg(feature = "labinius-challenges")]
            FamilyShell::BinaryRoot { .. } => None,
        }
    }
}

/// Extension seam: challenge selection for binary source groups
/// tracked by <https://github.com/LayerZero-Labs/akita/issues/285>.
///
/// The selected distribution owns its identity and validation;
/// [`FoldChallengeFamily`] owns its message layout and decoding.
/// Serialization preserves the existing signed-sparse configuration representation.
/// Selecting a distribution is independent of enabling Cargo features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FoldChallengeDistribution {
    /// Existing fixed-weight signed-sparse distribution.
    SignedSparse(SparseChallengeConfig),
    /// Bounded-weight 46 supports in Phi243, with deterministic Shake256V1 signs.
    #[cfg(feature = "labinius-challenges")]
    BinaryRoot,
}

impl serde::Serialize for FoldChallengeDistribution {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::SignedSparse(config) => {
                #[cfg(feature = "labinius-challenges")]
                if config.count_pm1 == u32::MAX as usize && config.count_pm2 == 1 {
                    return Err(serde::ser::Error::custom(
                        "reserved binary root family encoding",
                    ));
                }
                config.serialize(serializer)
            }
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => SparseChallengeConfig {
                count_pm1: u32::MAX as usize,
                count_pm2: 1,
            }
            .serialize(serializer),
        }
    }
}

impl<'de> serde::Deserialize<'de> for FoldChallengeDistribution {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let config = SparseChallengeConfig::deserialize(deserializer)?;
        // Reserve an impossible supported sparse weight, preserving ordinary
        // JSON and Postcard encodings without an extra enum discriminant.
        #[cfg(feature = "labinius-challenges")]
        if config.count_pm1 == u32::MAX as usize && config.count_pm2 == 1 {
            return Ok(Self::BinaryRoot);
        }
        Ok(Self::SignedSparse(config))
    }
}

impl From<SparseChallengeConfig> for FoldChallengeDistribution {
    fn from(config: SparseChallengeConfig) -> Self {
        Self::SignedSparse(config)
    }
}

impl FoldChallengeDistribution {
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
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => None,
            Self::SignedSparse(config) => Some(config),
        }
    }

    /// Certified operator-norm policy, if this family supports one.
    pub fn operator_norm_rejection(&self, ring_d: usize) -> Option<OperatorNormRejection> {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => None,
            Self::SignedSparse(config) => {
                jolt_crypto::challenge::selective_l2_operator_norm_rejection(ring_d, config)
            }
        }
    }

    /// Maximum support weight.
    pub fn weight(&self) -> usize {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::WEIGHT,
            Self::SignedSparse(config) => config.weight(),
        }
    }

    /// Maximum coefficient L1 norm.
    pub fn l1_norm(&self) -> usize {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::WEIGHT,
            Self::SignedSparse(config) => config.l1_norm(),
        }
    }

    /// Maximum coefficient infinity norm.
    pub fn infinity_norm(&self) -> u32 {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => 1,
            Self::SignedSparse(config) => config.infinity_norm(),
        }
    }

    /// Maximum squared coefficient L2 norm.
    pub fn challenge_l2_sq_max(&self) -> u128 {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::WEIGHT as u128,
            Self::SignedSparse(config) => config.challenge_l2_sq_max(),
        }
    }

    /// Canonical bytes that order semantically distinct distributions.
    ///
    /// This is a schedule-ordering descriptor, not a transcript binding; the
    /// draw context binds the family through `FamilyShell`.
    pub fn ordering_descriptor_bytes(&self) -> Vec<u8> {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::IDENTITY.to_vec(),
            Self::SignedSparse(config) => {
                let mut out = Vec::with_capacity(17);
                out.push(0);
                out.extend_from_slice(&(config.count_pm1 as u64).to_le_bytes());
                out.extend_from_slice(&(config.count_pm2 as u64).to_le_bytes());
                out
            }
        }
    }

    /// Validate the family for the selected ring dimension.
    pub fn validate_dyn(&self, ring_d: usize) -> Result<(), &'static str> {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::validate(ring_d, 0),
            Self::SignedSparse(config) => config.validate_dyn(ring_d),
        }
    }

    /// Validate structure and the ordinary minimum entropy floor.
    pub fn validate_for_ring_dim(&self, ring_d: usize) -> Result<(), &'static str> {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => {
                akita_challenges::root_family::validate(ring_d, jolt_crypto::challenge::MIN_FOLD_CHALLENGE_ENTROPY_BITS)
            }
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
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::validate(ring_d, required_bits),
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
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::entropy_bits(D),
            Self::SignedSparse(config) => config.log2_support_bits::<D>(),
        }
    }

    /// Check a per-draw entropy floor for a static ring dimension.
    pub fn validate_min_entropy<const D: usize>(
        &self,
        required_bits: u32,
    ) -> Result<(), &'static str> {
        match self {
            #[cfg(feature = "labinius-challenges")]
            Self::BinaryRoot => akita_challenges::root_family::validate(D, required_bits),
            Self::SignedSparse(config) => config.validate_min_entropy::<D>(required_bits),
        }
    }
}

#[cfg(all(test, feature = "labinius-challenges"))]
mod tests {
    use super::*;

    #[test]
    fn binary_family_encoding_is_explicit_and_roundtrips() {
        let binary = FoldChallengeDistribution::BinaryRoot;
        let encoded = postcard::to_allocvec(&binary).unwrap();
        assert_eq!(
            postcard::from_bytes::<FoldChallengeDistribution>(&encoded).unwrap(),
            binary
        );
        let json = serde_json::to_string(&binary).unwrap();
        assert_eq!(json, r#"{"count_pm1":4294967295,"count_pm2":1}"#);
        assert_eq!(
            serde_json::from_str::<FoldChallengeDistribution>(&json).unwrap(),
            binary
        );
        let alias = FoldChallengeDistribution::SignedSparse(SparseChallengeConfig {
            count_pm1: u32::MAX as usize,
            count_pm2: 1,
        });
        assert!(postcard::to_allocvec(&alias).is_err());
        assert_eq!(binary.ordering_descriptor_bytes(), akita_challenges::root_family::IDENTITY);
    }
}
