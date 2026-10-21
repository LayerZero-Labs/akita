//! Zero-knowledge opening schedules.
//!
//! A zero-knowledge opening runs levels `0..=cut_level` masked. Their fold
//! responses are one-shot: the plan schedules each with `nonce_bits = 0`, so
//! no nonce precedes the fold challenges, and an out-of-range response aborts
//! the proof instead of being retried. Each of these folds is therefore sized
//! for the failure target [`ZkOpeningSchedule::fold_failure_target`].
//!
//! Zero-knowledge mode is not selectable yet: only
//! [`ZkOpeningSchedule::for_tests`] constructs a schedule.

use crate::sis::FoldFailureTarget;
use crate::transcript_grinding_plan::derive_grinding_plan;
use crate::{ChallengeFieldOrder, FoldSchedule, GrindingPlan, OpeningClaimsLayout};
use akita_error::AkitaError;

/// The only supported cut: levels 0 and 1 are masked (`k = 2`).
pub const ZK_CUT_LEVEL: u32 = 1;

/// Every masked fold fails with probability at most
/// `2^-ZK_FOLD_SECURITY_BITS / masked_folds`, so all of them together fail
/// with probability at most `2^-ZK_FOLD_SECURITY_BITS`.
pub const ZK_FOLD_SECURITY_BITS: u32 = 136;

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
    /// The digit counts are the schedule's own; this constructor does not
    /// resize them for [`fold_failure_target`](Self::fold_failure_target).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a cut other than
    /// [`ZK_CUT_LEVEL`], or a schedule whose cut is not a nonterminal level.
    #[cfg(any(test, feature = "test-support"))]
    pub fn for_tests(schedule: FoldSchedule, cut_level: u32) -> Result<Self, AkitaError> {
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
        Ok(Self {
            schedule,
            cut_level,
        })
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
        FoldFailureTarget::OneShot {
            security_bits: ZK_FOLD_SECURITY_BITS,
            masked_folds: self.cut_level + 1,
        }
    }

    /// The grinding plan: the clear plan of [`schedule`](Self::schedule)
    /// except that the fold responses at levels up to the cut are one-shot.
    pub fn grinding_plan(
        &self,
        root_layout: &OpeningClaimsLayout,
        challenge_order: ChallengeFieldOrder,
        claim_extension_degree: usize,
    ) -> Result<GrindingPlan, AkitaError> {
        derive_grinding_plan(
            &self.schedule,
            root_layout,
            challenge_order,
            claim_extension_degree,
            Some(self.cut_level),
        )
    }
}
