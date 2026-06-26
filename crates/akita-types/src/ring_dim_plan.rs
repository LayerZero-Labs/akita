//! Runtime per-fold ring dimension plan derived from the effective schedule.
//!
//! Infrastructure for runtime ring cutover (waves 4+). Validation runs on every
//! generated schedule table; orchestration wiring lands in the cutover PR.

use std::collections::BTreeSet;

use akita_field::{AkitaError, FieldCore};

use crate::proof::{AkitaExpandedSetup, AkitaSetupSeed};
use crate::schedule::{schedule_num_fold_levels, Schedule};
use crate::setup_geometry::{setup_active_ring_elems_at, SetupRelationShape};

/// Upper bound on fold levels accepted by [`RingDimPlan::from_schedule`].
pub const MAX_FOLD_LEVELS: usize = 16;

/// Ring dimensions supported by runtime dispatch.
pub const SUPPORTED_RING_DIMS: [usize; 4] = [32, 64, 128, 256];

/// Per-fold ring dimensions by protocol role.
///
/// Invariant when nested: `opening | outer | inner` (i.e. `d_d | d_b | d_a`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRingDims {
    /// Fold / ring-switch / inner-commitment ring (`d_a`).
    pub inner: usize,
    /// Outer-commitment ring (`d_b`).
    pub outer: usize,
    /// Opening-commitment ring (`d_d`).
    pub opening: usize,
}

/// Per-level runtime ring geometry for prove / verify orchestration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingLevelContext {
    pub role_dims: CommitmentRingDims,
    /// Fold ring `d_a` (= `role_dims.inner`).
    pub ring_d: usize,
    /// Shape-only setup-product row count at `d_a`.
    pub setup_active_ring_elems: usize,
}

/// Derived view of validated per-level ring dimensions from a schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingDimPlan {
    role_dims: [CommitmentRingDims; MAX_FOLD_LEVELS],
    pub num_folds: usize,
}

impl CommitmentRingDims {
    #[must_use]
    pub const fn uniform(d: usize) -> Self {
        Self {
            inner: d,
            outer: d,
            opening: d,
        }
    }

    #[must_use]
    pub fn nests(self) -> bool {
        self.inner.is_multiple_of(self.outer) && self.outer.is_multiple_of(self.opening)
    }
}

impl RingDimPlan {
    /// Build a validated plan from the effective schedule and setup seed.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when any catalog check fails.
    pub fn from_schedule(schedule: &Schedule, seed: &AkitaSetupSeed) -> Result<Self, AkitaError> {
        let num_folds = schedule_num_fold_levels(schedule);
        if num_folds > MAX_FOLD_LEVELS {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule has {num_folds} fold levels, max supported is {MAX_FOLD_LEVELS}"
            )));
        }
        let mut role_dims = [CommitmentRingDims::uniform(0); MAX_FOLD_LEVELS];
        for (level, slot) in role_dims.iter_mut().take(num_folds).enumerate() {
            let exec = schedule.get_execution_schedule(level)?;
            let lp = &exec.params;
            let dims = CommitmentRingDims::uniform(lp.ring_dimension);
            validate_role_dims(dims)?;
            if !seed.gen_ring_dim.is_multiple_of(dims.inner) {
                return Err(AkitaError::InvalidSetup(format!(
                    "setup gen_ring_dim={} is not divisible by fold ring d_a={}",
                    seed.gen_ring_dim, dims.inner
                )));
            }
            if dims.inner != lp.ring_dimension {
                return Err(AkitaError::InvalidSetup(
                    "schedule ring_dimension disagrees with derived inner role dim".into(),
                ));
            }
            if !exec.current_w_len.is_multiple_of(dims.inner) {
                return Err(AkitaError::InvalidSetup(format!(
                    "witness length {} is not divisible by fold ring d_a={}",
                    exec.current_w_len, dims.inner
                )));
            }
            if !exec.is_terminal {
                let next_ring_d = exec.next_params.ring_dimension;
                if next_ring_d == 0 || !exec.next_w_len.is_multiple_of(next_ring_d) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "next witness length {} is not divisible by next ring dimension {next_ring_d}",
                        exec.next_w_len,
                    )));
                }
            }
            *slot = dims;
        }
        for level in 0..num_folds.saturating_sub(1) {
            let current = role_dims[level];
            let next = role_dims[level + 1];
            let exec = schedule.get_execution_schedule(level)?;
            if current.inner != next.inner && !exec.next_w_len.is_multiple_of(next.inner) {
                return Err(AkitaError::InvalidSetup(format!(
                    "next witness length {} is not divisible by next-level d_a={}",
                    exec.next_w_len, next.inner
                )));
            }
        }
        Ok(Self {
            role_dims,
            num_folds,
        })
    }

    /// Per-role ring dimensions at fold level `level`.
    ///
    /// # Errors
    ///
    /// Returns an error when `level` is out of range.
    pub fn dims_at(&self, level: usize) -> Result<CommitmentRingDims, AkitaError> {
        if level >= self.num_folds {
            return Err(AkitaError::InvalidSetup(format!(
                "ring dim plan has no fold level {level}"
            )));
        }
        let dims = self.role_dims[level];
        validate_role_dims(dims)?;
        Ok(dims)
    }

    /// Fold ring `d_a` at level `level`.
    ///
    /// # Errors
    ///
    /// Returns an error when `level` is out of range or dims fail validation.
    pub fn dim_at(&self, level: usize) -> Result<usize, AkitaError> {
        Ok(self.dims_at(level)?.inner)
    }

    /// Distinct ring dimensions across all roles and fold levels.
    #[must_use]
    pub fn unique_dims(&self) -> Vec<usize> {
        let mut dims = BTreeSet::new();
        for level in 0..self.num_folds {
            if let Ok(role) = self.dims_at(level) {
                dims.insert(role.inner);
                dims.insert(role.outer);
                dims.insert(role.opening);
            }
        }
        dims.into_iter().collect()
    }

    /// Per-level geometry using the live relation shape at `level`.
    ///
    /// # Errors
    ///
    /// Returns an error when level bounds, role nesting, or setup envelope checks fail.
    pub fn context_at<F: FieldCore>(
        &self,
        level: usize,
        schedule: &Schedule,
        expanded: &AkitaExpandedSetup<F>,
        relation_shape: &SetupRelationShape,
    ) -> Result<RingLevelContext, AkitaError> {
        let role_dims = self.dims_at(level)?;
        let exec = schedule.get_execution_schedule(level)?;
        if role_dims.inner != exec.params.ring_dimension {
            return Err(AkitaError::InvalidSetup(
                "ring dim plan disagrees with schedule ring_dimension at level".into(),
            ));
        }
        let setup_active_ring_elems =
            setup_active_ring_elems_at(level, schedule, expanded, relation_shape)?;
        Ok(RingLevelContext {
            role_dims,
            ring_d: role_dims.inner,
            setup_active_ring_elems,
        })
    }
}

/// Validated per-fold ring geometry for prove/verify orchestration.
///
/// Build once at [`validate_schedule_context_at_entry`] and thread inward so inner
/// `prove` / `verify` do not re-validate the schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedScheduleContext {
    pub ring_plan: RingDimPlan,
}

impl ValidatedScheduleContext {
    /// Validate schedule ring geometry against the setup seed.
    ///
    /// # Errors
    ///
    /// Same as [`RingDimPlan::from_schedule`].
    pub fn validate(schedule: &Schedule, seed: &AkitaSetupSeed) -> Result<Self, AkitaError> {
        Ok(Self {
            ring_plan: RingDimPlan::from_schedule(schedule, seed)?,
        })
    }
}

/// Prove or verify entry validation (wave 5a): schedule ring geometry only, no NTT warming.
///
/// Call once the effective schedule is fixed at `batched_prove` / `batched_verify` or the
/// inner `prove` / `verify` orchestration entry.
pub fn validate_ring_dim_plan_at_entry(
    schedule: &Schedule,
    seed: &AkitaSetupSeed,
) -> Result<RingDimPlan, AkitaError> {
    Ok(ValidatedScheduleContext::validate(schedule, seed)?.ring_plan)
}

/// Entry helper returning the full validated context for orchestration threading.
pub fn validate_schedule_context_at_entry(
    schedule: &Schedule,
    seed: &AkitaSetupSeed,
) -> Result<ValidatedScheduleContext, AkitaError> {
    ValidatedScheduleContext::validate(schedule, seed)
}

fn validate_role_dims(dims: CommitmentRingDims) -> Result<(), AkitaError> {
    for d in [dims.inner, dims.outer, dims.opening] {
        if !SUPPORTED_RING_DIMS.contains(&d) {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported ring dimension {d}"
            )));
        }
    }
    if !dims.nests() {
        return Err(AkitaError::InvalidSetup(
            "per-role ring dims must satisfy d_d | d_b | d_a".into(),
        ));
    }
    // PR-perblock-exec: relax to nesting-only once kernels honor distinct role dims.
    if dims.inner != dims.outer || dims.outer != dims.opening {
        return Err(AkitaError::InvalidSetup(
            "per-block execution is not enabled: d_a, d_b, d_d must be equal".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_geometry::SetupRelationShape;
    use crate::{segment_typed_witness_shape, DirectStep, FoldStep, LevelParams, MRowLayout, Step};
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn sample_seed(gen_ring_dim: usize) -> AkitaSetupSeed {
        AkitaSetupSeed {
            max_num_vars: 8,
            max_num_batched_polys: 4,
            gen_ring_dim,
            max_setup_len: 64,
            public_matrix_seed: [0u8; 32],
        }
    }

    fn sample_lp(ring_dimension: usize) -> LevelParams {
        LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            ring_dimension,
            3,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(2, 1, 3, 2, ring_dimension)
        .expect("level params")
    }

    fn terminal_direct_step(lp: &LevelParams, current_w_len: usize) -> DirectStep {
        let shape = segment_typed_witness_shape(lp, 128, 1, 1, 1, 1).expect("terminal shape");
        DirectStep {
            current_w_len,
            witness_shape: shape,
            direct_bytes: 0,
            params: None,
        }
    }

    fn two_level_schedule(d: usize) -> Schedule {
        let lp = sample_lp(d);
        Schedule {
            steps: vec![
                Step::Fold(FoldStep {
                    params: lp.clone(),
                    current_w_len: d * 4,
                    next_w_len: d * 2,
                    level_bytes: 0,
                }),
                Step::Fold(FoldStep {
                    params: lp.clone(),
                    current_w_len: d * 2,
                    next_w_len: d,
                    level_bytes: 0,
                }),
                Step::Direct(terminal_direct_step(&lp, d)),
            ],
            total_bytes: 0,
        }
    }

    #[test]
    fn from_schedule_accepts_uniform_dims() {
        let schedule = two_level_schedule(64);
        let seed = sample_seed(64);
        let plan = RingDimPlan::from_schedule(&schedule, &seed).expect("plan");
        assert_eq!(plan.num_folds, 2);
        assert_eq!(plan.dim_at(0).expect("d0"), 64);
        assert_eq!(plan.unique_dims(), vec![64]);
    }

    #[test]
    fn from_schedule_rejects_non_divisible_envelope() {
        let schedule = two_level_schedule(128);
        let seed = sample_seed(64);
        let err =
            RingDimPlan::from_schedule(&schedule, &seed).expect_err("gen_ring_dim must divide d_a");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn commitment_ring_dims_rejects_non_uniform_triple() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        };
        assert!(dims.nests());
        assert!(validate_role_dims(dims).is_err());
    }

    #[test]
    fn context_at_checks_setup_envelope() {
        let schedule = two_level_schedule(64);
        let seed = sample_seed(64);
        let plan = RingDimPlan::from_schedule(&schedule, &seed).expect("plan");
        let lp = sample_lp(64);
        let shape = SetupRelationShape::from_level_params(&lp, 1, MRowLayout::WithDBlock, 2)
            .expect("shape");
        let shared = crate::derive_public_matrix_flat::<F, 64>(1, &seed.public_matrix_seed);
        let expanded =
            crate::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared);
        let err = plan
            .context_at(0, &schedule, &expanded, &shape)
            .expect_err("tiny envelope");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
