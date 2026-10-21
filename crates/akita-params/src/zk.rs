//! Zero-knowledge opening schedules and their hiding randomness.
//!
//! A zero-knowledge opening runs levels `0..=cut_level` masked. Their fold
//! responses are one-shot: the plan schedules each with `nonce_bits = 0`, so
//! no nonce precedes the fold challenges, and an out-of-range response aborts
//! the proof instead of being retried.
//!
//! [`ZkOpeningSchedule::for_tests`], the only constructor, admits a fold
//! schedule only if every masked level carries at least the digit count its
//! honest-fold policy derives for [`ZkOpeningSchedule::fold_failure_target`].
//! Zero-knowledge mode is not selectable yet.
//!
//! A zero-knowledge schedule also appends fresh randomness columns to every
//! outer (B) slice and every opening (D) image made at a masked level. The
//! randomness hides the image by the leftover hash lemma once its entropy
//! exceeds the image's bit length by `2λ` plus a fixed slack. This module is
//! the one Akita caller of `jolt_crypto`'s sizing rule; planners and schedule
//! validation both size widths through [`hiding_randomness_columns`].

use crate::sis::{
    BalancedSignedDigitFoldPolicy, CommittedSourceContract, FoldFailureTarget, FoldWitnessNorms,
    HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery,
};
use crate::transcript_grinding_plan::derive_grinding_plan;
use crate::{
    ChallengeFieldOrder, CommittedGroupParams, FoldSchedule, GrindingPlan, OpeningClaimsLayout,
    OpeningMethod,
};
use akita_error::AkitaError;
use jolt_algebra::{Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use jolt_crypto::challenge::SparseChallengeConfig;
use jolt_crypto::lattice::required_randomness_columns;

use crate::sis::SisModulusProfileId;
use crate::GroupOpenPhaseParams;

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

/// The schedule an opening runs: clear, or zero knowledge.
#[derive(Clone, Copy, Debug)]
pub enum OpeningSchedule<'a> {
    Clear(&'a FoldSchedule),
    Zk(&'a ZkOpeningSchedule),
}

impl<'a> From<&'a FoldSchedule> for OpeningSchedule<'a> {
    fn from(schedule: &'a FoldSchedule) -> Self {
        Self::Clear(schedule)
    }
}

impl<'a> OpeningSchedule<'a> {
    /// The fold schedule the opening executes.
    #[must_use]
    pub fn fold_schedule(self) -> &'a FoldSchedule {
        match self {
            Self::Clear(schedule) => schedule,
            Self::Zk(schedule) => schedule.schedule(),
        }
    }

    /// The only accepted grinding plan of this schedule.
    pub fn grinding_plan(
        self,
        root_layout: &OpeningClaimsLayout,
        challenge_order: ChallengeFieldOrder,
        claim_extension_degree: usize,
    ) -> Result<GrindingPlan, AkitaError> {
        let (schedule, one_shot_through) = match self {
            Self::Clear(schedule) => (schedule, None),
            Self::Zk(schedule) => (&schedule.schedule, Some(schedule.cut_level)),
        };
        derive_grinding_plan(
            schedule,
            root_layout,
            challenge_order,
            claim_extension_degree,
            one_shot_through,
        )
    }
}

/// An opening schedule whose levels up to `cut_level` are masked.
///
/// The schedule is a clear fold schedule, terminal included: the residual cut
/// that replaces the levels after `cut_level` is not part of this type yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkOpeningSchedule {
    schedule: FoldSchedule,
    cut_level: u32,
}

impl ZkOpeningSchedule {
    /// Wrap a fold schedule with its cut, for tests only.
    ///
    /// `root_source` is the contract the root's single committed group was
    /// produced under.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a cut other than
    /// [`ZK_CUT_LEVEL`], a cut that is not a nonterminal level, a root with
    /// precommitted groups, or a masked level whose fold digit count is below
    /// the one its honest-fold policy derives for the one-shot target.
    #[cfg(any(test, feature = "test-support"))]
    pub fn for_tests(
        schedule: FoldSchedule,
        cut_level: u32,
        root_source: CommittedSourceContract,
    ) -> Result<Self, AkitaError> {
        schedule.validate_structure()?;
        if cut_level != ZK_CUT_LEVEL {
            return Err(AkitaError::UnsupportedSchedule(format!(
                "zero-knowledge cut level {cut_level} is unsupported; only {ZK_CUT_LEVEL} is"
            )));
        }
        if schedule.recursive_folds.len() < cut_level as usize {
            return Err(AkitaError::UnsupportedSchedule(
                "zero-knowledge cut level must precede the terminal fold".into(),
            ));
        }
        let zk = Self {
            schedule,
            cut_level,
        };
        let target = zk.fold_failure_target();
        for level in 0..=cut_level as usize {
            let sizing = MaskedFoldSizing::new(&zk.schedule, level, root_source)?;
            let required = sizing.num_digits_fold(target)?;
            let scheduled = sizing.params.num_digits_fold();
            if scheduled < required {
                return Err(AkitaError::UnsupportedSchedule(format!(
                    "masked level {level} has {scheduled} fold digits; its one-shot target needs {required}"
                )));
            }
        }
        Ok(zk)
    }

    /// The fold schedule, terminal included.
    #[must_use]
    pub fn schedule(&self) -> &FoldSchedule {
        &self.schedule
    }

    /// The last masked level.
    #[must_use]
    pub const fn cut_level(&self) -> u32 {
        self.cut_level
    }

    /// Failure target of each masked fold's one-shot response.
    #[must_use]
    pub const fn fold_failure_target(&self) -> FoldFailureTarget {
        one_shot_fold_target(self.cut_level)
    }
}

/// The honest-fold sizing inputs of one masked fold level, read from its
/// scheduled parameters: the root's source contract and group geometry at
/// level 0, and the balanced digits of the previous witness above it.
pub struct MaskedFoldSizing<'a> {
    pub params: &'a CommittedGroupParams,
    pub policy: HonestFoldPolicySpec,
    pub challenge: SparseChallengeConfig,
    pub num_claims: usize,
    /// Response coefficients across every physical window.
    pub num_fold_coeffs: usize,
    pub witness_norms: FoldWitnessNorms,
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
        let num_fold_coeffs = usize::try_from(params.num_fold_coeffs())
            .ok()
            .and_then(|coeffs| coeffs.checked_mul(params.witness_chunk.num_chunks))
            .ok_or_else(|| AkitaError::InvalidSetup("fold response width overflow".into()))?;
        Ok(Self {
            params,
            policy,
            challenge: params.fold_challenge_config(),
            num_claims,
            num_fold_coeffs,
            witness_norms,
        })
    }

    /// The sizing query for `failure_target`.
    #[must_use]
    pub fn query(&self, failure_target: FoldFailureTarget) -> HonestFoldSizingQuery<'_> {
        let params = self.params;
        let blocks = params.blocks();
        HonestFoldSizingQuery {
            ring_dimension: params.d_a(),
            challenge_dimension: match params.opening_method() {
                OpeningMethod::EvaluationTrace => params.d_a(),
                OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } => challenge_subring_dimension,
            },
            num_claims: self.num_claims,
            num_live_ring_elements_per_claim: blocks.live_ring_elements_per_claim,
            num_positions_per_block: blocks.positions_per_block,
            num_live_blocks: blocks.live_blocks,
            num_chunks: params.witness_chunk.num_chunks,
            num_fold_coeffs: self.num_fold_coeffs,
            witness_norms: self.witness_norms,
            log_basis_response: params.open().digits.log_basis,
            challenge_config: &self.challenge,
            failure_target,
        }
    }

    /// The fold digit count the level's policy derives for `failure_target`.
    pub fn num_digits_fold(&self, failure_target: FoldFailureTarget) -> Result<usize, AkitaError> {
        self.policy.num_digits_fold(self.query(failure_target))
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

/// Recompute every hiding-randomness width of `schedule` and reject any
/// mismatch, in release builds.
///
/// Each level up to `cut_level` must carry exactly [`b_randomness_columns`]
/// for its own group's B slices and [`d_randomness_columns`] for its D
/// image. Both are sized at the level's open alphabet, the one its range
/// check proves. Preceding groups (setup prefixes, committed
/// programs, trusted advice) and every level after the cut carry none, and a
/// hiding level uses one witness chunk. The cut must precede the last
/// nonterminal fold, whose output is the terminal witness.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] naming the first violating level.
pub fn validate_hiding_randomness(
    schedule: &FoldSchedule,
    cut_level: usize,
) -> Result<(), AkitaError> {
    // The last nonterminal fold's output is the terminal witness, sent in the
    // clear, so a cut there would publish that fold's randomness.
    if cut_level >= schedule.recursive_folds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "hiding cut level {cut_level} reaches the terminal witness"
        )));
    }
    let levels = std::iter::once(&schedule.root.params)
        .chain(schedule.recursive_folds.iter().map(|fold| &fold.params));
    for (level, params) in levels.enumerate() {
        let invalid = |what: &str| {
            AkitaError::InvalidSetup(format!("level {level} {what} hiding randomness"))
        };
        let own = params.own_group();
        let (expected_b, expected_d) = if level <= cut_level {
            if params.witness_chunk.num_chunks != 1 {
                return Err(invalid("splits across witness chunks its"));
            }
            (b_randomness_columns(own)?, d_randomness_columns(params)?)
        } else {
            (0, 0)
        };
        if own.profile.randomness_columns != expected_b {
            return Err(invalid("has the wrong B"));
        }
        if params.open_randomness_columns != expected_d {
            return Err(invalid("has the wrong D"));
        }
        if params
            .preceding_group_iter()
            .any(|group| group.profile.randomness_columns != 0)
        {
            return Err(invalid("gives a precommitted group"));
        }
    }
    Ok(())
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
