//! Zero-knowledge opening schedules.
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
use jolt_crypto::challenge::SparseChallengeConfig;

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
