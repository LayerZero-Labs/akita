//! Zero-knowledge opening schedules and their hiding randomness.
//!
//! A zero-knowledge schedule carries a [`ZkOpeningSchedule`] whose cut masks
//! levels `0..=cut_level`. At each masked level:
//!
//! - the fold response is one-shot: the grinding plan schedules it with
//!   `nonce_bits = 0`, so no nonce precedes the fold challenges, and an
//!   out-of-range response aborts the proof instead of being retried;
//! - its fold digit count meets [`ZkOpeningSchedule::fold_failure_target`];
//! - its B slices and its D image carry fresh hiding randomness columns, sized
//!   by [`hiding_randomness_columns`]; the randomness hides each image by the
//!   leftover hash lemma once its entropy exceeds the image's bit length by
//!   `2λ` plus a fixed slack.
//!
//! [`ZkOpeningSchedule::admit`] checks all of this in release builds. Catalog
//! admission and the prover both run it; a clear catalog rejects every
//! zero-knowledge schedule, so zero knowledge is not selectable yet.

use crate::sis::{
    BalancedSignedDigitFoldPolicy, CommittedSourceContract, FoldFailureTarget, FoldWitnessNorms,
    HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery, SisModulusProfileId,
};
use crate::{
    CommitmentPayloadMode, CommitmentSliceCount, CommittedGroupParams, FoldSchedule,
    GroupOpenPhaseParams, InnerCommitSecurityRoute, OpeningMethod,
};
use akita_error::AkitaError;
use jolt_algebra::{Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use jolt_crypto::challenge::SparseChallengeConfig;
use jolt_crypto::lattice::required_randomness_columns;

/// The only supported cut: levels 0 and 1 are masked (`k = 2`).
pub const ZK_CUT_LEVEL: u32 = 1;

/// Every masked fold fails with probability at most
/// `2^-ZK_FOLD_SECURITY_BITS / masked_folds`, so all of them together fail
/// with probability at most `2^-ZK_FOLD_SECURITY_BITS`.
pub const ZK_FOLD_SECURITY_BITS: u32 = 136;

/// Failure target of each one-shot fold of an opening masked through
/// `cut_level`.
pub(crate) const fn one_shot_fold_target(cut_level: u32) -> FoldFailureTarget {
    FoldFailureTarget::OneShot {
        security_bits: ZK_FOLD_SECURITY_BITS,
        masked_folds: cut_level + 1,
    }
}

/// The zero-knowledge part of a fold schedule: the cut that masks levels
/// `0..=cut_level`.
///
/// A [`FoldSchedule`] stores it, its descriptor bytes bind it, and its
/// grinding plan makes the masked fold responses one-shot. Deserializing one
/// validates the cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ZkCutEncoding")]
pub struct ZkOpeningSchedule {
    cut_level: u32,
}

/// The serialized form of a [`ZkOpeningSchedule`], before its cut is checked.
#[derive(serde::Deserialize)]
struct ZkCutEncoding {
    cut_level: u32,
}

impl TryFrom<ZkCutEncoding> for ZkOpeningSchedule {
    type Error = AkitaError;

    fn try_from(encoding: ZkCutEncoding) -> Result<Self, AkitaError> {
        let zk = Self {
            cut_level: encoding.cut_level,
        };
        zk.validate_cut()?;
        Ok(zk)
    }
}

impl ZkOpeningSchedule {
    /// The cut masking levels `0..=cut_level`, for tests until zero
    /// knowledge is published.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a cut other than
    /// [`ZK_CUT_LEVEL`].
    #[cfg(any(test, all(feature = "test-support", feature = "zk")))]
    pub fn new(cut_level: u32) -> Result<Self, AkitaError> {
        Self::try_from(ZkCutEncoding { cut_level })
    }

    fn validate_cut(self) -> Result<(), AkitaError> {
        if self.cut_level != ZK_CUT_LEVEL {
            return Err(AkitaError::UnsupportedSchedule(format!(
                "zero-knowledge cut level {} is unsupported; only {ZK_CUT_LEVEL} is",
                self.cut_level
            )));
        }
        Ok(())
    }

    /// The last masked level.
    #[must_use]
    pub const fn cut_level(self) -> u32 {
        self.cut_level
    }

    /// Whether fold `level` is masked.
    #[must_use]
    pub const fn masks(self, level: usize) -> bool {
        level <= self.cut_level as usize
    }

    /// Failure target of each masked fold's one-shot response.
    #[must_use]
    pub const fn fold_failure_target(self) -> FoldFailureTarget {
        one_shot_fold_target(self.cut_level)
    }

    /// The one admission check of a zero-knowledge schedule, in release
    /// builds.
    ///
    /// `root_source` is the contract the root's single committed group is
    /// produced under, and `claim_extension_degree` the degree of the claim
    /// field over the base field. The check requires:
    ///
    /// - `schedule.zk` is a supported cut that precedes the last nonterminal
    ///   fold, whose output is the terminal witness, sent in the clear;
    /// - claims in the base field (`E::DEGREE = 1`): the masked stage checks
    ///   have no extension-field products;
    /// - the root commits exactly one group, and no masked level consumes a
    ///   preceding group (a setup prefix or precommitted group), whose
    ///   response would share the one-shot probe unsized; nor does the level
    ///   after the cut, so the cut level runs no Stage 3;
    /// - the residual commitment, which the level after the cut opens, has
    ///   d_A = d_B;
    /// - every masked level uses the ℓ∞ route (an ℓ2 route reveals the
    ///   response energy and carries a per-attempt ℓ2 cap and an
    ///   operator-norm challenge family, none of which is one-shot sized),
    ///   the quotient relation (not reduced evaluation), the open basis b = 8,
    ///   and one witness chunk; it carries exactly [`b_randomness_columns`]
    ///   for its own group's B slices and [`d_randomness_columns`] for its D
    ///   image, and at least the fold digits its honest-fold policy derives
    ///   for [`fold_failure_target`](Self::fold_failure_target);
    /// - the level after the cut, whose own group is the residual commitment
    ///   to w_k, opens at [`RESIDUAL_RANDOMNESS_LOG_BASIS`] in one outer
    ///   slice with a raw payload, carries exactly [`b_randomness_columns`] B
    ///   columns (r_a), no D randomness and no preceding group;
    /// - every later level, and every preceding group, carries no randomness.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a missing or
    /// unsupported cut and [`AkitaError::InvalidSetup`] naming the first
    /// violating condition otherwise.
    pub fn admit(
        schedule: &FoldSchedule,
        root_source: CommittedSourceContract,
        claim_extension_degree: usize,
    ) -> Result<Self, AkitaError> {
        let zk = schedule.zk.ok_or_else(|| {
            AkitaError::UnsupportedSchedule("schedule has no zero-knowledge cut".into())
        })?;
        zk.validate_cut()?;
        let cut = zk.cut_level as usize;
        let after_cut = schedule.recursive_folds.get(cut).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "zero-knowledge cut level {cut} reaches the terminal witness"
            ))
        })?;
        if claim_extension_degree != 1 {
            return Err(AkitaError::InvalidSetup(
                "zero knowledge needs claims in the base field".into(),
            ));
        }
        if !schedule.root.params.precommitted_groups().is_empty() {
            return Err(AkitaError::InvalidSetup(
                "a masked root must commit exactly one group".into(),
            ));
        }
        if after_cut.params.setup_prefix().is_some() {
            return Err(AkitaError::InvalidSetup(
                "the cut level must not run Stage 3".into(),
            ));
        }
        let residual = after_cut.params.role_dims();
        if residual.d_a() != residual.d_b() {
            return Err(AkitaError::InvalidSetup(
                "the residual commitment needs d_A = d_B".into(),
            ));
        }
        let target = zk.fold_failure_target();
        let levels = std::iter::once(&schedule.root.params)
            .chain(schedule.recursive_folds.iter().map(|fold| &fold.params));
        for (level, params) in levels.enumerate() {
            let invalid = |what: &str| AkitaError::InvalidSetup(format!("level {level} {what}"));
            let own = params.own_group();
            let (expected_b, expected_d) = if zk.masks(level) {
                if !matches!(
                    params.inner().matrix.security_route(),
                    InnerCommitSecurityRoute::Linf(_)
                ) {
                    return Err(invalid("is masked on an L2 route"));
                }
                if params.ring_relation_mode.is_reduced_evaluation() {
                    return Err(invalid("is masked under reduced evaluation"));
                }
                if params.open().digits.log_basis != MASKED_LOG_BASIS {
                    return Err(invalid("is masked with an open basis other than 8"));
                }
                if params.witness_chunk.num_chunks != 1 {
                    return Err(invalid("is masked across several witness chunks"));
                }
                if params.preceding_group_iter().next().is_some() {
                    return Err(invalid("is masked but consumes a preceding group"));
                }
                let sizing = MaskedFoldSizing::new(schedule, level, root_source)?;
                let required = sizing.num_digits_fold(target)?;
                if params.num_digits_fold() < required {
                    return Err(invalid(&format!(
                        "has {} fold digits; its one-shot target needs {required}",
                        params.num_digits_fold()
                    )));
                }
                (b_randomness_columns(own)?, d_randomness_columns(params)?)
            } else if level == cut + 1 {
                // The cut successor: its own group is the residual commitment
                // to w_k, hidden by r_a, which the Σ-protocol consumes at
                // the residual alphabet; the level itself is never folded.
                // It consumes no preceding group: the Stage-3 check above.
                if own.opening.log_basis_open != RESIDUAL_RANDOMNESS_LOG_BASIS {
                    return Err(invalid(
                        "opens the residual commitment at a basis other than 16",
                    ));
                }
                if own.profile.outer_slice_count != CommitmentSliceCount::ONE {
                    return Err(invalid("commits the residual in more than one outer slice"));
                }
                if params.open_randomness_columns != 0 {
                    return Err(invalid("carries D hiding randomness after the cut"));
                }
                if params.payload_mode != CommitmentPayloadMode::Raw {
                    return Err(invalid("sends the residual commitment compressed"));
                }
                if own.profile.randomness_columns != b_randomness_columns(own)? {
                    return Err(invalid("has the wrong residual randomness r_a"));
                }
                continue;
            } else {
                (0, 0)
            };
            if own.profile.randomness_columns != expected_b {
                return Err(invalid("has the wrong B hiding randomness"));
            }
            if params.open_randomness_columns != expected_d {
                return Err(invalid("has the wrong D hiding randomness"));
            }
            if params
                .preceding_group_iter()
                .any(|group| group.profile.randomness_columns != 0)
            {
                return Err(invalid("gives a preceding group hiding randomness"));
            }
        }
        Ok(zk)
    }
}

/// Base-two logarithm of the open basis the masked stage checks assume
/// (b = 8: no Stage-1 product stages).
pub const MASKED_LOG_BASIS: u32 = 3;

/// Base-two logarithm of the alphabet of r_a, the residual commitment's
/// hiding randomness (radix 16). The level after the cut opens at this basis,
/// so [`b_randomness_columns`] sizes r_a, and the backend samples it, at the
/// alphabet the Σ-protocol's `ResidualRandomness` segment declares
/// (`specs/whiteout-akita.md` §4.8.1).
pub const RESIDUAL_RANDOMNESS_LOG_BASIS: u32 = 4;

/// The honest-fold sizing inputs of one masked fold level, read from its
/// scheduled parameters: the root's source contract and group geometry at
/// level 0, and the balanced digits of the previous witness above it.
pub struct MaskedFoldSizing<'a> {
    params: &'a CommittedGroupParams,
    policy: HonestFoldPolicySpec,
    challenge: SparseChallengeConfig,
    num_claims: usize,
    witness_norms: FoldWitnessNorms,
}

impl<'a> MaskedFoldSizing<'a> {
    /// The sizing inputs of fold `level` of `schedule`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a root with
    /// precommitted groups, or a level that is not a nonterminal fold.
    pub fn new(
        schedule: &'a FoldSchedule,
        level: usize,
        root_source: CommittedSourceContract,
    ) -> Result<Self, AkitaError> {
        let field_bits = root_source.decomposition().field_bits();
        let params = match level.checked_sub(1) {
            None => &schedule.root.params,
            Some(index) => {
                &schedule
                    .recursive_folds
                    .get(index)
                    .ok_or_else(|| {
                        AkitaError::UnsupportedSchedule(
                            "a masked level must be a nonterminal fold".into(),
                        )
                    })?
                    .params
            }
        };
        let inner = params.inner().digits;
        let (policy, num_claims, witness_norms) = if level == 0 {
            if !params.precommitted_groups().is_empty() {
                return Err(AkitaError::UnsupportedSchedule(
                    "a masked root must commit exactly one group".into(),
                ));
            }
            let group = params.final_group().profile.group;
            let source_len = jolt_algebra::checked::pow2(group.num_vars())
                .and_then(|len| len.checked_mul(group.num_polynomials()))
                .ok_or_else(|| AkitaError::InvalidSetup("source length overflow".into()))?;
            (
                root_source.class().honest_fold_policy(field_bits),
                group.num_polynomials(),
                root_source
                    .source_norms(inner.log_basis, inner.num_digits, params.d_a(), source_len)?
                    .fold_witness,
            )
        } else {
            (
                HonestFoldPolicySpec::BalancedSignedDigit(
                    BalancedSignedDigitFoldPolicy::universal(field_bits),
                ),
                1,
                FoldWitnessNorms::bounded(inner.log_basis, params.d_a()),
            )
        };
        Ok(Self {
            params,
            policy,
            challenge: params.fold_challenge_config(),
            num_claims,
            witness_norms,
        })
    }

    /// The level's scheduled parameters.
    #[must_use]
    pub const fn params(&self) -> &'a CommittedGroupParams {
        self.params
    }

    /// The level's honest-fold policy.
    #[must_use]
    pub const fn policy(&self) -> &HonestFoldPolicySpec {
        &self.policy
    }

    /// The sizing query for `failure_target`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the response size overflows.
    pub fn query(
        &self,
        failure_target: FoldFailureTarget,
    ) -> Result<HonestFoldSizingQuery<'_>, AkitaError> {
        let params = self.params;
        HonestFoldSizingQuery::new(
            params.d_a(),
            match params.opening_method() {
                OpeningMethod::EvaluationTrace => params.d_a(),
                OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } => challenge_subring_dimension,
            },
            self.num_claims,
            params.blocks(),
            params.witness_chunk.num_chunks,
            params.inner().matrix.input_width(),
            self.witness_norms,
            params.open().digits.log_basis,
            &self.challenge,
            failure_target,
        )
    }

    /// The fold digit count the level's policy derives for `failure_target`.
    pub fn num_digits_fold(&self, failure_target: FoldFailureTarget) -> Result<usize, AkitaError> {
        self.policy.num_digits_fold(self.query(failure_target)?)
    }
}

/// Slack bits added on top of `2λ` when sizing hiding randomness.
const HIDING_SLACK_BITS: usize = 8;

/// Shape of one outer-map image hidden by a randomness vector: one B slice or
/// one level's D image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageShape {
    /// Image rank (output ring elements).
    pub rows: usize,
    /// Ring dimension of the image and of each randomness column.
    pub ring_dim: usize,
    /// Field modulus profile of the image coefficients.
    pub modulus: SisModulusProfileId,
}

/// Balanced digit alphabet `[-2^(log_basis-1), 2^(log_basis-1) - 1]` of the
/// witness that holds the randomness. It is that witness's open basis, the one
/// alphabet the next level's range check proves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DigitAlphabet {
    /// Base-two logarithm of the digit base.
    pub log_basis: u32,
}

/// Number of randomness ring columns that hide one image of `image` shape
/// with digits drawn uniformly from `alphabet`.
///
/// # Errors
///
/// Returns an error for a zero ring dimension, an alphabet outside `1..=8`,
/// or an overflowing size.
pub fn hiding_randomness_columns(
    image: ImageShape,
    alphabet: DigitAlphabet,
) -> Result<usize, AkitaError> {
    let image_coeffs = image
        .rows
        .checked_mul(image.ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("hidden image size overflows usize".into()))?;
    let columns = match image.modulus {
        SisModulusProfileId::Q32Offset99 => required_randomness_columns::<Prime32Offset99>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
        SisModulusProfileId::Q64Offset59 => required_randomness_columns::<Prime64Offset59>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
        SisModulusProfileId::Q128OffsetA7F7 => required_randomness_columns::<Prime128OffsetA7F7>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
        SisModulusProfileId::Q128Offset275 => required_randomness_columns::<Prime128Offset275>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
    }?;
    Ok(columns)
}

/// Hiding-randomness columns of every B slice of `group`'s commitment.
///
/// The randomness hides one slice image and is drawn from the open alphabet
/// of the opening that consumes the commitment, the alphabet that level's
/// range check proves.
///
/// # Errors
///
/// Returns the sizing errors of [`hiding_randomness_columns`].
pub fn b_randomness_columns(group: &GroupOpenPhaseParams) -> Result<usize, AkitaError> {
    let outer = &group.profile.outer.matrix;
    hiding_randomness_columns(
        ImageShape {
            rows: outer.output_rank(),
            ring_dim: outer.ring_dimension(),
            modulus: outer.sis_modulus_profile(),
        },
        DigitAlphabet {
            log_basis: group.opening.log_basis_open,
        },
    )
}

/// Hiding-randomness columns of the D image `params`' level makes, drawn from
/// that level's open alphabet.
///
/// # Errors
///
/// Returns the sizing errors of [`hiding_randomness_columns`].
pub fn d_randomness_columns(params: &CommittedGroupParams) -> Result<usize, AkitaError> {
    let open = params.open();
    hiding_randomness_columns(
        ImageShape {
            rows: open.matrix.output_rank(),
            ring_dim: open.matrix.ring_dimension(),
            modulus: open.matrix.sis_modulus_profile(),
        },
        DigitAlphabet {
            log_basis: open.digits.log_basis,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headline_slice_needs_forty_five_columns() {
        // d_b = 64, rank 1, b = 8 over the 128-bit field: 64 * 128 + 256 + 8
        // bits at 192 bits per column.
        let image = ImageShape {
            rows: 1,
            ring_dim: 64,
            modulus: SisModulusProfileId::Q128OffsetA7F7,
        };
        assert_eq!(
            hiding_randomness_columns(image, DigitAlphabet { log_basis: 3 }).unwrap(),
            45
        );
    }

    #[test]
    fn residual_commitment_needs_ninety_seven_columns() {
        // Three R_256 rows at b = 16: 768 * 128 + 264 bits at 1024 bits per
        // column.
        let image = ImageShape {
            rows: 3,
            ring_dim: 256,
            modulus: SisModulusProfileId::Q128OffsetA7F7,
        };
        assert_eq!(
            hiding_randomness_columns(image, DigitAlphabet { log_basis: 4 }).unwrap(),
            97
        );
    }

    #[test]
    fn rejects_unsupported_alphabets() {
        let image = ImageShape {
            rows: 1,
            ring_dim: 64,
            modulus: SisModulusProfileId::Q64Offset59,
        };
        assert!(hiding_randomness_columns(image, DigitAlphabet { log_basis: 0 }).is_err());
        assert!(hiding_randomness_columns(image, DigitAlphabet { log_basis: 9 }).is_err());
    }
}
