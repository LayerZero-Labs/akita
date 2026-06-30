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

use crate::schedule::{FoldStep, Schedule, Step};
use akita_field::AkitaError;

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
        let mut shapes = Vec::new();
        let mut fold_index = 0usize;
        for step in &schedule.steps {
            let Step::Fold(fold_step) = step else {
                continue;
            };
            let ring_dimension = fold_step.params.ring_dimension;
            if ring_dimension == 0 {
                return Err(AkitaError::InvalidSetup(format!(
                    "fold level {fold_index}: ring_dimension must be non-zero",
                )));
            }
            if !gen_ring_dim.is_multiple_of(ring_dimension) {
                return Err(AkitaError::InvalidSetup(format!(
                    "fold level {fold_index}: ring_dimension={ring_dimension} \
                     does not divide gen_ring_dim={gen_ring_dim}",
                )));
            }
            shapes.push(LevelShape::from_fold_step(fold_index, fold_step)?);
            fold_index += 1;
        }
        Ok(ValidatedScheduleContext {
            schedule,
            gen_ring_dim,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{DirectStep, FoldStep, Schedule, Step};
    use crate::{AkitaScheduleLookupKey, CleartextWitnessShape, LevelParams};
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
