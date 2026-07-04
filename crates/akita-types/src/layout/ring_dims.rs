//! Per-level and per-schedule ring dimension planning.
//!
//! [`RingDimPlan`] validates every fold level's role dimensions against the
//! setup seed. Per-level geometry (`n_ring_elems`, `flat_field_len`, …) lives on
//! [`super::LevelParams`].

use crate::proof::AkitaSetupSeed;
use crate::schedule::{schedule_num_fold_levels, Schedule, Step};
use akita_field::AkitaError;

/// Upper bound on fold levels accepted by [`RingDimPlan`].
pub const MAX_FOLD_LEVELS: usize = 16;

/// Ring dimensions supported by runtime dispatch.
pub const SUPPORTED_RING_DIMS: [usize; 4] = [32, 64, 128, 256];

/// Which Ajtai / protocol matrix role a buffer belongs to at one fold level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingRole {
    /// A-role (`d_a`): fold witness, row coefficients, ring-switch geometry.
    Inner,
    /// B-role (`d_b`): sent commitment rows, COMMIT segment of `y`.
    Outer,
    /// D-role (`d_d`): opening digits, D-block rows `v`.
    Opening,
}

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

    /// Ring dimension for `role`.
    #[must_use]
    pub const fn dim_for(self, role: RingRole) -> usize {
        match role {
            RingRole::Inner => self.inner,
            RingRole::Outer => self.outer,
            RingRole::Opening => self.opening,
        }
    }
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
        if seed.gen_ring_dim == 0 {
            return Err(AkitaError::InvalidSetup(
                "gen_ring_dim must be non-zero".to_string(),
            ));
        }
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
            let dims = lp.role_dims;
            validate_role_dims(dims)?;
            validate_role_dims_match_keys(lp)?;
            for (role, d) in [
                (RingRole::Inner, dims.inner),
                (RingRole::Outer, dims.outer),
                (RingRole::Opening, dims.opening),
            ] {
                if !seed.gen_ring_dim.is_multiple_of(d) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "setup gen_ring_dim={} is not divisible by {:?} ring d={d}",
                        seed.gen_ring_dim, role
                    )));
                }
            }
            if !step.current_w_len.is_multiple_of(dims.inner) {
                return Err(AkitaError::InvalidSetup(format!(
                    "witness length {} is not divisible by fold ring d_a={}",
                    step.current_w_len, dims.inner
                )));
            }
            if let Some(Step::Fold(next)) = schedule.steps.get(level + 1) {
                let next_ring_d = next.params.role_dims.inner;
                if next_ring_d == 0 || !step.next_w_len.is_multiple_of(next_ring_d) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "next witness length {} is not divisible by next fold ring d_a={next_ring_d}",
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
}

pub fn validate_role_dims_match_keys(lp: &crate::LevelParams) -> Result<(), AkitaError> {
    let dims = lp.role_dims;
    if lp.ring_dimension != dims.inner {
        return Err(AkitaError::InvalidSetup(format!(
            "ring_dimension={} disagrees with role_dims.d_a={}",
            lp.ring_dimension, dims.inner
        )));
    }
    let a_ring = lp.a_key.sis_table_key().ring_dimension as usize;
    let b_ring = lp.b_key.sis_table_key().ring_dimension as usize;
    let d_ring = lp.d_key.sis_table_key().ring_dimension as usize;
    if a_ring != dims.inner {
        return Err(AkitaError::InvalidSetup(format!(
            "A-key ring dimension {a_ring} disagrees with role_dims.d_a={}",
            dims.inner
        )));
    }
    if b_ring != dims.outer {
        return Err(AkitaError::InvalidSetup(format!(
            "B-key ring dimension {b_ring} disagrees with role_dims.d_b={}",
            dims.outer
        )));
    }
    if d_ring != dims.opening {
        return Err(AkitaError::InvalidSetup(format!(
            "D-key ring dimension {d_ring} disagrees with role_dims.d_d={}",
            dims.opening
        )));
    }
    Ok(())
}

pub fn validate_role_dims(dims: CommitmentRingDims) -> Result<(), AkitaError> {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LevelParams;
    use crate::schedule::{DirectStep, FoldStep, Schedule, Step};
    use crate::sis::SisModulusFamily;
    use crate::CleartextWitnessShape;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::AkitaError;

    fn make_fold_level_params(
        ring_dimension: usize,
        num_blocks: usize,
        block_len: usize,
    ) -> LevelParams {
        let mut params = LevelParams::params_only(
            SisModulusFamily::Q128,
            ring_dimension,
            3,
            1,
            1,
            1,
            SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![1],
            },
        );
        params.num_blocks = num_blocks;
        params.block_len = block_len;
        params.num_digits_commit = 2;
        params.num_digits_open = 2;
        params.stamp_role_dims_from_keys();
        params
    }

    fn make_fold_step(ring_dimension: usize, num_blocks: usize, block_len: usize) -> FoldStep {
        FoldStep {
            params: make_fold_level_params(ring_dimension, num_blocks, block_len),
            current_w_len: 0,
            next_w_len: 0,
            level_bytes: 0,
        }
    }

    fn make_direct_step() -> DirectStep {
        DirectStep {
            current_w_len: 0,
            witness_shape: CleartextWitnessShape::FieldElements(0),
            direct_bytes: 0,
            params: None,
        }
    }

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

    fn seed(gen_ring_dim: usize) -> AkitaSetupSeed {
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim,
            max_setup_len: 0,
            public_matrix_seed: [0u8; 32],
        }
    }

    #[test]
    fn accepts_uniform_d_schedule_when_d_equals_gen_ring_dim() {
        let sched = uniform_schedule(256, 3);
        let plan = RingDimPlan::from_schedule(&sched, &seed(256)).expect("256|256");
        assert_eq!(plan.num_folds(), 3);
    }

    #[test]
    fn accepts_d_divides_gen_ring_dim() {
        let sched = uniform_schedule(64, 2);
        RingDimPlan::from_schedule(&sched, &seed(256)).expect("64|256");
    }

    #[test]
    fn accepts_mixed_d_schedule_when_all_dims_divide_gen_ring_dim() {
        let sched = mixed_d_schedule(&[(32, 4, 8), (64, 4, 4), (128, 2, 4), (256, 2, 2)]);
        let plan = RingDimPlan::from_schedule(&sched, &seed(256)).expect("all divide 256");
        assert_eq!(plan.num_folds(), 4);
    }

    #[test]
    fn level_params_flat_field_len_matches_ring_elems_times_ring_dim() {
        let sched = uniform_schedule(64, 1);
        let Step::Fold(step) = &sched.steps[0] else {
            panic!("expected fold");
        };
        assert_eq!(step.params.n_ring_elems().expect("n_ring"), 32);
        assert_eq!(step.params.flat_field_len().expect("flat"), 2048);
    }

    #[test]
    fn schedule_with_no_fold_steps_is_valid() {
        let sched = Schedule {
            steps: vec![Step::Direct(make_direct_step())],
            total_bytes: 0,
        };
        let plan = RingDimPlan::from_schedule(&sched, &seed(256)).expect("no folds");
        assert_eq!(plan.num_folds(), 0);
    }

    #[test]
    fn rejects_zero_gen_ring_dim() {
        let sched = uniform_schedule(64, 1);
        let err = RingDimPlan::from_schedule(&sched, &seed(0)).expect_err("gen=0");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_does_not_divide_gen_ring_dim() {
        let sched = mixed_d_schedule(&[(96, 4, 4)]);
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("96|256");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_zero() {
        let sched = mixed_d_schedule(&[(0, 4, 4)]);
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("d=0");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    fn fold_step_with_witness_lens(
        ring_dimension: usize,
        current_w_len: usize,
        next_w_len: usize,
    ) -> FoldStep {
        let mut step = make_fold_step(ring_dimension, 4, 8);
        step.current_w_len = current_w_len;
        step.next_w_len = next_w_len;
        step
    }

    #[test]
    fn rejects_witness_length_not_divisible_by_d_a() {
        let sched = Schedule {
            steps: vec![
                Step::Fold(fold_step_with_witness_lens(64, 65, 64)),
                Step::Direct(make_direct_step()),
            ],
            total_bytes: 0,
        };
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("65 % 64");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_next_witness_length_not_divisible_by_next_d_a() {
        let sched = Schedule {
            steps: vec![
                Step::Fold(fold_step_with_witness_lens(64, 64, 65)),
                Step::Fold(make_fold_step(32, 4, 8)),
                Step::Direct(make_direct_step()),
            ],
            total_bytes: 0,
        };
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("next 65 % 32");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_too_many_fold_levels() {
        let steps: Vec<Step> = (0..MAX_FOLD_LEVELS + 1)
            .map(|_| Step::Fold(make_fold_step(64, 4, 8)))
            .chain([Step::Direct(make_direct_step())])
            .collect();
        let sched = Schedule {
            steps,
            total_bytes: 0,
        };
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("> MAX_FOLD_LEVELS");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_role_dims_key_mismatch() {
        let mut step = make_fold_step(64, 4, 8);
        step.params.role_dims = CommitmentRingDims {
            inner: 64,
            outer: 64,
            opening: 32,
        };
        let sched = Schedule {
            steps: vec![Step::Fold(step), Step::Direct(make_direct_step())],
            total_bytes: 0,
        };
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("B-key still 64");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn accepts_nested_per_role_dims_with_matching_keys() {
        use crate::layout::{AjtaiKeyParams, SisModulusFamily};
        use crate::sis::DEFAULT_SIS_SECURITY_BITS;

        let mut params = LevelParams::log_basis_stub(3);
        params.ring_dimension = 128;
        params.num_blocks = 4;
        params.block_len = 8;
        params.num_digits_commit = 2;
        params.num_digits_open = 2;
        params.a_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            1,
            16,
            0,
            128,
        );
        params.b_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            1,
            16,
            0,
            64,
        );
        params.d_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            1,
            16,
            0,
            32,
        );
        params.stamp_role_dims_from_keys();
        let sched = Schedule {
            steps: vec![
                Step::Fold(FoldStep {
                    params,
                    current_w_len: 128,
                    next_w_len: 64,
                    level_bytes: 0,
                }),
                Step::Direct(make_direct_step()),
            ],
            total_bytes: 0,
        };
        let plan = RingDimPlan::from_schedule(&sched, &seed(128)).expect("nested 128|64|32");
        let dims = plan.dims_at(0).expect("level 0");
        assert_eq!(dims.d_a(), 128);
        assert_eq!(dims.d_b(), 64);
        assert_eq!(dims.d_d(), 32);
    }

    #[test]
    fn rejects_ring_dimension_mismatch_with_role_dims() {
        let mut step = make_fold_step(64, 4, 8);
        step.params.ring_dimension = 128;
        let sched = Schedule {
            steps: vec![Step::Fold(step), Step::Direct(make_direct_step())],
            total_bytes: 0,
        };
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("ring_dim != d_a");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_larger_than_gen_ring_dim() {
        let sched = mixed_d_schedule(&[(512, 4, 4)]);
        let err = RingDimPlan::from_schedule(&sched, &seed(256)).expect_err("512|256");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
