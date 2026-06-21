//! Per-fold grind acceptance targets (`p_grind`) for tail-bound digit sizing.
//!
//! The transcript descriptor still pins the production default from
//! [`crate::FoldLinfProtocolBinding`]; per-level values live in
//! [`super::FoldWitnessLinfCapConfig`] on each [`crate::LevelParams`].

use crate::FoldLinfProtocolBinding;

/// Rational grind acceptance target `p_grind = num / den` used in the
/// union-bound sizing for `t*` (`specs/fold-linf-rejection.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GrindTargetAcceptProb {
    pub num: u32,
    pub den: u32,
}

impl GrindTargetAcceptProb {
    pub const EIGHTH: Self = Self { num: 1, den: 8 };
    pub const SIXTEENTH: Self = Self { num: 1, den: 16 };
    pub const THIRTY_SECOND: Self = Self { num: 1, den: 32 };

    #[inline]
    pub const fn as_rational(self) -> (u128, u128) {
        (self.num as u128, self.den as u128)
    }
}

/// Policy selecting `p_grind` as a function of fold depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GrindTargetAcceptSchedule {
    /// Every fold uses [`FoldLinfProtocolBinding::CURRENT`].
    Uniform,
    /// Folds `0..early_levels` use `early`; all later folds use `late`.
    StepAfterEarlyLevels {
        early_levels: usize,
        early: GrindTargetAcceptProb,
        late: GrindTargetAcceptProb,
    },
}

impl GrindTargetAcceptSchedule {
    pub const PRODUCTION: Self = Self::Uniform;

    /// Experiment: levels 0–1 at `1/8`, level 2+ at `1/16`.
    pub const TWO_EIGHTH_THEN_SIXTEENTH: Self = Self::StepAfterEarlyLevels {
        early_levels: 2,
        early: GrindTargetAcceptProb::EIGHTH,
        late: GrindTargetAcceptProb::SIXTEENTH,
    };

    #[inline]
    #[must_use]
    pub fn at_fold_level(self, fold_level: usize) -> (u128, u128) {
        match self {
            Self::Uniform => FoldLinfProtocolBinding::CURRENT.grind_target_accept_prob(),
            Self::StepAfterEarlyLevels {
                early_levels,
                early,
                late,
            } => {
                if fold_level < early_levels {
                    early.as_rational()
                } else {
                    late.as_rational()
                }
            }
        }
    }
}
