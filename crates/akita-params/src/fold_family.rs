//! The sparse challenge family one fold group draws from (protocol v8).
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

/// The sparse challenge family one fold group draws from: its algebraic
/// domain, ring dimension, shell and operator-norm policy.
///
/// Schedules build it once; the grinding plan prices the fold from it and
/// every draw decodes with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldChallengeFamily {
    domain: FoldChallengeDrawDomain,
    shell: SparseChallengeShell,
}

impl FoldChallengeFamily {
    /// Validate one family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when a coefficient-packing family
    /// disagrees with its subring dimension or carries an operator-norm
    /// policy, and [`AkitaError::InvalidSetup`] for an unsupported shell or
    /// policy.
    pub fn new(
        domain: FoldChallengeDrawDomain,
        ring_d: usize,
        config: SparseChallengeConfig,
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
        Ok(Self {
            domain,
            shell: SparseChallengeShell::new(ring_d, &config, rejection)?,
        })
    }

    /// Message layout and decoder for `count` challenges of this family.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `count` is zero or the
    /// message exceeds the message limit.
    pub fn stream(&self, count: usize) -> Result<SparseChallengeStream, AkitaError> {
        Ok(self.shell.stream(count)?)
    }

    /// Guaranteed `log2` size of the set each challenge is uniform on, for
    /// any challenge count: `K` for an unfiltered family, the certified
    /// exponent for an operator-norm one.
    #[must_use]
    pub const fn set_bits(&self) -> u32 {
        self.shell.set_bits()
    }

    #[must_use]
    pub const fn domain(&self) -> FoldChallengeDrawDomain {
        self.domain
    }

    #[must_use]
    pub const fn ring_d(&self) -> usize {
        self.shell.ring_d()
    }

    #[must_use]
    pub const fn config(&self) -> SparseChallengeConfig {
        self.shell.config()
    }

    #[must_use]
    pub const fn rejection(&self) -> Option<OperatorNormRejection> {
        self.shell.rejection()
    }
}
