//! Validated schedule context — shape authority for the runtime ring cutover.
//!
//! A [`ValidatedScheduleContext`] wraps a [`Schedule`] and a `gen_ring_dim`
//! (the maximum ring dimension the setup was generated at) and:
//!
//! 1. **Validates** that every fold step's `ring_dimension` divides
//!    `gen_ring_dim`.  Returns [`AkitaError`] on the first violation; never
//!    panics.
//! 2. **Exposes** per-level derived shape so Tier-3/4 slices can read
//!    geometry without recomputing or scattering `ring_dimension * num_blocks`
//!    arithmetic across the codebase.
//!
//! ## What this type is NOT
//!
//! - It does **not** compute `gen_ring_dim` from a `Cfg`.  Computing
//!   `gen_ring_dim = max ring_dimension across the schedule catalog` is the
//!   job of S5 (`AkitaProverSetup`/setup wiring).  Here you provide it.
//! - It does **not** call `dispatch_ring_d!`.  Dispatch happens at kernel
//!   entry in `akita-prover`; this type only carries shape metadata.
//! - It does **not** modify the schedule.  The wrapped schedule is unchanged.

use crate::proof::{AkitaExpandedSetup, AkitaSetupSeed};
use crate::schedule::{schedule_num_fold_levels, FoldStep, Schedule, Step};
use crate::setup_geometry::{setup_active_ring_elems_at, SetupRelationShape};
use akita_field::{AkitaError, FieldCore};

/// Upper bound on fold levels accepted by [`RingDimPlan`].
pub const MAX_FOLD_LEVELS: usize = 16;

/// Ring dimensions supported by runtime dispatch.
pub const SUPPORTED_RING_DIMS: [usize; 4] = [32, 64, 128, 256];

/// Per-fold ring dimensions by protocol role.
///
/// Invariant when nested: `opening | outer | inner` (`d_d | d_b | d_a`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRingDims {
    /// Fold / ring-switch / inner-commitment ring (`d_a`).
    pub inner: usize,
    /// Outer-commitment ring (`d_b`).
    pub outer: usize,
    /// Opening-commitment ring (`d_d`).
    pub opening: usize,
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

    /// Ring dimension for A-role data: the folded witness `z`, A quotient
    /// rows, the consistency row, and fold/ring-switch arithmetic.
    #[must_use]
    pub const fn d_a(self) -> usize {
        self.inner
    }

    /// Ring dimension for B-role data: next-witness digit commitments
    /// (`t_hat`, tiered `u_concat`), COMMIT and B_inner relation rows.
    #[must_use]
    pub const fn d_b(self) -> usize {
        self.outer
    }

    /// Ring dimension for D-role data: opening digits (`e_hat`) and the
    /// D-block relation rows (`v = D * e_hat`).
    #[must_use]
    pub const fn d_d(self) -> usize {
        self.opening
    }

    /// The single dimension shared by all roles, or an error once per-role
    /// dimensions diverge.
    ///
    /// Fused code paths that process several row groups under one ring
    /// dimension must obtain it here — never from a bare
    /// `LevelParams::ring_dimension` read — so that enabling per-role
    /// dimensions turns every not-yet-split fused path into a loud error
    /// instead of silently applying one role's dimension to another role's
    /// rows.
    pub fn uniform_dim(self) -> Result<usize, AkitaError> {
        if self.inner == self.outer && self.outer == self.opening {
            Ok(self.inner)
        } else {
            Err(AkitaError::InvalidSetup(format!(
                "fused ring path requires uniform role dims, got d_a={} d_b={} d_d={}",
                self.inner, self.outer, self.opening
            )))
        }
    }
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
    num_folds: usize,
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
            let Some(Step::Fold(step)) = schedule.steps.get(level) else {
                return Err(AkitaError::InvalidSetup(format!(
                    "schedule is missing fold step at level {level}"
                )));
            };
            let lp = &step.params;
            let dims = CommitmentRingDims::uniform(lp.ring_dimension);
            validate_role_dims(dims)?;
            if !seed.gen_ring_dim.is_multiple_of(dims.inner) {
                return Err(AkitaError::InvalidSetup(format!(
                    "setup gen_ring_dim={} is not divisible by fold ring d_a={}",
                    seed.gen_ring_dim, dims.inner
                )));
            }
            if !step.current_w_len.is_multiple_of(dims.inner) {
                return Err(AkitaError::InvalidSetup(format!(
                    "witness length {} is not divisible by fold ring d_a={}",
                    step.current_w_len, dims.inner
                )));
            }
            if let Some(Step::Fold(next)) = schedule.steps.get(level + 1) {
                let next_ring_d = next.params.ring_dimension;
                if next_ring_d == 0 || !step.next_w_len.is_multiple_of(next_ring_d) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "next witness length {} is not divisible by next ring dimension {next_ring_d}",
                        step.next_w_len,
                    )));
                }
            }
            *slot = dims;
        }
        Ok(Self {
            role_dims,
            num_folds,
        })
    }

    /// Number of fold levels covered by this plan.
    #[must_use]
    pub fn num_folds(&self) -> usize {
        self.num_folds
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
        let mut dims = std::collections::BTreeSet::new();
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

/// Derived shape for one fold level, validated against `gen_ring_dim`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelShape {
    /// Zero-based index of this fold step within the schedule.
    pub level: usize,
    /// Ring dimension for this level (`ring_dimension` in [`crate::LevelParams`]).
    pub ring_dimension: usize,
    /// Total ring elements in the committed witness at this level
    /// (`num_blocks * block_len`).
    pub n_ring_elems: usize,
    /// Total flat field-element count (`n_ring_elems * ring_dimension`).
    ///
    /// This is the length of a flat-coefficient representation of the
    /// committed witness at this level.
    pub flat_field_len: usize,
    /// A-matrix view width in ring elements (`inner_width`), or `0` if the
    /// level's layout has not been fully populated (e.g., root-direct with
    /// `block_len = 0`).
    pub setup_view_dim: usize,
}

impl LevelShape {
    fn from_fold_step(level: usize, step: &FoldStep) -> Result<Self, AkitaError> {
        let params = &step.params;
        let ring_dimension = params.ring_dimension;
        let n_ring_elems = params
            .num_blocks
            .checked_mul(params.block_len)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "level {level}: num_blocks={} * block_len={} overflows usize",
                    params.num_blocks, params.block_len,
                ))
            })?;
        let flat_field_len = n_ring_elems.checked_mul(ring_dimension).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "level {level}: n_ring_elems={n_ring_elems} * ring_dimension={ring_dimension} overflows usize",
            ))
        })?;
        let setup_view_dim = params.inner_width();
        Ok(LevelShape {
            level,
            ring_dimension,
            n_ring_elems,
            flat_field_len,
            setup_view_dim,
        })
    }
}

/// A schedule paired with a validated `gen_ring_dim`.
///
/// Construction validates that every fold level's `ring_dimension` divides
/// `gen_ring_dim`.  After construction the per-level [`LevelShape`] data is
/// cheap to access without re-validation.
///
/// # Example
///
/// ```rust,ignore
/// let ctx = ValidatedScheduleContext::new(&schedule, gen_ring_dim)?;
/// for shape in ctx.level_shapes() {
///     println!("level {} ring_d={} flat_len={}", shape.level, shape.ring_dimension, shape.flat_field_len);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ValidatedScheduleContext<'s> {
    /// The underlying schedule (borrowed, not cloned).
    schedule: &'s Schedule,
    /// Setup generation ring dimension.
    gen_ring_dim: usize,
    /// Validated role-ring plan derived from the effective schedule.
    ring_plan: RingDimPlan,
    /// Per-level shape data, one entry per `Step::Fold` in `schedule.steps`.
    shapes: Vec<LevelShape>,
}

impl<'s> ValidatedScheduleContext<'s> {
    /// Validate `schedule` against `gen_ring_dim` and construct the context.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] if:
    /// - `gen_ring_dim` is zero.
    /// - Any fold level's `ring_dimension` is zero.
    /// - Any fold level's `ring_dimension` does not divide `gen_ring_dim`.
    /// - Any `num_blocks * block_len` or `n_ring_elems * ring_dimension`
    ///   arithmetic overflows.
    pub fn new(schedule: &'s Schedule, gen_ring_dim: usize) -> Result<Self, AkitaError> {
        if gen_ring_dim == 0 {
            return Err(AkitaError::InvalidSetup(
                "gen_ring_dim must be non-zero".to_string(),
            ));
        }
        let seed = AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim,
            max_setup_len: 0,
            public_matrix_seed: [0u8; 32],
        };
        Self::validate(schedule, &seed)
    }

    /// Validate `schedule` against a concrete setup seed.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`], with the seed becoming the source of truth for
    /// the setup generation ring dimension.
    pub fn validate(schedule: &'s Schedule, seed: &AkitaSetupSeed) -> Result<Self, AkitaError> {
        if seed.gen_ring_dim == 0 {
            return Err(AkitaError::InvalidSetup(
                "gen_ring_dim must be non-zero".to_string(),
            ));
        }
        let ring_plan = RingDimPlan::from_schedule(schedule, seed)?;
        let mut shapes = Vec::new();
        let mut fold_index = 0usize;
        for step in &schedule.steps {
            let Step::Fold(fold_step) = step else {
                continue;
            };
            let ring_dimension = fold_step.params.ring_dimension;
            let planned = ring_plan.dim_at(fold_index)?;
            if planned != ring_dimension {
                return Err(AkitaError::InvalidSetup(format!(
                    "fold level {fold_index}: planned ring_dimension={planned} \
                     disagrees with schedule ring_dimension={ring_dimension}",
                )));
            }
            shapes.push(LevelShape::from_fold_step(fold_index, fold_step)?);
            fold_index += 1;
        }
        Ok(ValidatedScheduleContext {
            schedule,
            gen_ring_dim: seed.gen_ring_dim,
            ring_plan,
            shapes,
        })
    }

    /// The `gen_ring_dim` this context was validated against.
    #[inline]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    /// The underlying schedule.
    #[inline]
    pub fn schedule(&self) -> &'s Schedule {
        self.schedule
    }

    /// Per-level shapes, in fold-step order.
    #[inline]
    pub fn level_shapes(&self) -> &[LevelShape] {
        &self.shapes
    }

    /// Validated role-ring plan for this schedule.
    #[inline]
    pub fn ring_plan(&self) -> &RingDimPlan {
        &self.ring_plan
    }

    /// Number of fold levels.
    #[inline]
    pub fn num_fold_levels(&self) -> usize {
        self.shapes.len()
    }

    /// Shape for one fold level by index.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] if `level` is out of range.
    pub fn level_shape(&self, level: usize) -> Result<&LevelShape, AkitaError> {
        self.shapes.get(level).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "schedule has {} fold levels; requested level {level}",
                self.shapes.len(),
            ))
        })
    }
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
    // Per-block distinct role execution is intentionally not active yet. The
    // plan carries role dimensions now so that later slices can relax this
    // check in one place when the kernels honor distinct d_a / d_b / d_d.
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
    use crate::schedule::{DirectStep, FoldStep, Schedule, Step};
    use crate::{CleartextWitnessShape, LevelParams};
    use akita_field::AkitaError;

    /// Build a minimal [`FoldStep`] with the given ring dimension.
    fn make_fold_step(ring_dimension: usize, num_blocks: usize, block_len: usize) -> FoldStep {
        let mut params = LevelParams::log_basis_stub(3);
        params.ring_dimension = ring_dimension;
        params.num_blocks = num_blocks;
        params.block_len = block_len;
        FoldStep {
            params,
            current_w_len: 0,
            next_w_len: 0,
            level_bytes: 0,
        }
    }

    /// Build a trivial terminal `DirectStep`.
    fn make_direct_step() -> DirectStep {
        DirectStep {
            current_w_len: 0,
            witness_shape: CleartextWitnessShape::FieldElements(0),
            direct_bytes: 0,
            params: None,
        }
    }

    /// A uniform-D schedule: all levels share the same ring dimension.
    fn uniform_schedule(ring_dimension: usize, num_levels: usize) -> Schedule {
        let mut steps: Vec<Step> = (0..num_levels)
            .map(|_| Step::Fold(make_fold_step(ring_dimension, 4, 8)))
            .collect();
        steps.push(Step::Direct(make_direct_step()));
        Schedule {
            steps,
            total_bytes: 0,
        }
    }

    /// A hand-built mixed-D schedule: different ring dimensions per fold level.
    fn mixed_d_schedule(dims: &[(usize, usize, usize)]) -> Schedule {
        let mut steps: Vec<Step> = dims
            .iter()
            .map(|&(d, nb, bl)| Step::Fold(make_fold_step(d, nb, bl)))
            .collect();
        steps.push(Step::Direct(make_direct_step()));
        Schedule {
            steps,
            total_bytes: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Acceptance tests
    // -----------------------------------------------------------------------

    #[test]
    fn accepts_uniform_d_schedule_when_d_equals_gen_ring_dim() {
        let sched = uniform_schedule(256, 3);
        let ctx = ValidatedScheduleContext::new(&sched, 256).expect("256|256 must be accepted");
        assert_eq!(ctx.num_fold_levels(), 3);
        assert_eq!(ctx.gen_ring_dim(), 256);
    }

    #[test]
    fn accepts_d_divides_gen_ring_dim() {
        // gen_ring_dim=256, level ring_dimension=64 → 256/64=4, divisible.
        let sched = uniform_schedule(64, 2);
        ValidatedScheduleContext::new(&sched, 256).expect("64|256 must be accepted");
    }

    #[test]
    fn accepts_mixed_d_schedule_when_all_dims_divide_gen_ring_dim() {
        // gen_ring_dim=256; levels use 32, 64, 128, 256 — all divide 256.
        let sched = mixed_d_schedule(&[(32, 4, 8), (64, 4, 4), (128, 2, 4), (256, 2, 2)]);
        let ctx = ValidatedScheduleContext::new(&sched, 256).expect("all dims divide 256");
        assert_eq!(ctx.num_fold_levels(), 4);
    }

    #[test]
    fn level_shape_ring_dimension_matches_params() {
        let sched = uniform_schedule(128, 2);
        let ctx = ValidatedScheduleContext::new(&sched, 256).unwrap();
        for shape in ctx.level_shapes() {
            assert_eq!(shape.ring_dimension, 128);
        }
    }

    #[test]
    fn level_shape_flat_field_len_is_n_ring_times_ring_dim() {
        // make_fold_step(64, 4, 8): num_blocks=4, block_len=8
        // n_ring_elems = 4*8 = 32, flat_field_len = 32*64 = 2048
        let sched = uniform_schedule(64, 1);
        let ctx = ValidatedScheduleContext::new(&sched, 256).unwrap();
        let shape = ctx.level_shape(0).unwrap();
        assert_eq!(shape.n_ring_elems, 32); // 4 * 8
        assert_eq!(shape.flat_field_len, 2048); // 32 * 64
    }

    #[test]
    fn level_shape_by_index_returns_correct_shape() {
        let sched = mixed_d_schedule(&[(32, 4, 8), (64, 4, 4)]);
        let ctx = ValidatedScheduleContext::new(&sched, 256).unwrap();
        let s0 = ctx.level_shape(0).unwrap();
        let s1 = ctx.level_shape(1).unwrap();
        assert_eq!(s0.ring_dimension, 32);
        assert_eq!(s1.ring_dimension, 64);
    }

    #[test]
    fn level_shape_out_of_range_returns_error() {
        let sched = uniform_schedule(64, 1);
        let ctx = ValidatedScheduleContext::new(&sched, 256).unwrap();
        assert!(matches!(
            ctx.level_shape(99),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn schedule_with_no_fold_steps_is_valid() {
        // A root-direct schedule may have no fold steps.
        let sched = Schedule {
            steps: vec![Step::Direct(make_direct_step())],
            total_bytes: 0,
        };
        let ctx = ValidatedScheduleContext::new(&sched, 256).expect("no fold steps is fine");
        assert_eq!(ctx.num_fold_levels(), 0);
    }

    // -----------------------------------------------------------------------
    // Rejection tests
    // -----------------------------------------------------------------------

    #[test]
    fn rejects_zero_gen_ring_dim() {
        let sched = uniform_schedule(64, 1);
        let err =
            ValidatedScheduleContext::new(&sched, 0).expect_err("gen_ring_dim=0 must be rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_does_not_divide_gen_ring_dim() {
        // ring_dimension=96 does not divide gen_ring_dim=256.
        let sched = mixed_d_schedule(&[(96, 4, 4)]);
        let err = ValidatedScheduleContext::new(&sched, 256).expect_err("96 does not divide 256");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_mixed_d_where_one_level_fails_divisibility() {
        // Levels 0,2 divide 256 but level 1 (96) does not.
        let sched = mixed_d_schedule(&[(64, 4, 4), (96, 4, 4), (128, 2, 4)]);
        let err = ValidatedScheduleContext::new(&sched, 256).expect_err("96 does not divide 256");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_zero() {
        let sched = mixed_d_schedule(&[(0, 4, 4)]);
        let err = ValidatedScheduleContext::new(&sched, 256)
            .expect_err("ring_dimension=0 must be rejected");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_larger_than_gen_ring_dim() {
        // ring_dimension=512 does not divide gen_ring_dim=256.
        let sched = mixed_d_schedule(&[(512, 4, 4)]);
        let err = ValidatedScheduleContext::new(&sched, 256).expect_err("512 does not divide 256");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
