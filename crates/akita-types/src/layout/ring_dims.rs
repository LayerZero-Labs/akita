//! Per-level and per-schedule ring dimension validation.
//!
//! [`validate_schedule_ring_dims`] checks every fold level's [`CommitmentRingDims`]
//! against the setup seed. Per-level geometry (`n_ring_elems`, `flat_field_len`, …)
//! lives on [`super::LevelParams`].

use crate::proof::AkitaSetupSeed;
use crate::schedule::Schedule;
use akita_field::AkitaError;

/// Upper bound on fold levels accepted by [`validate_schedule_ring_dims`].
pub const MAX_FOLD_LEVELS: usize = 16;

/// Ring dimensions valid for A-role (`d_a`) sparse fold challenges.
pub const SUPPORTED_CHALLENGE_RING_DIMS: &[usize] =
    akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS;

/// Ring dimensions valid for any commitment matrix role (B/D may use D=16 on fp128).
pub const SUPPORTED_RING_DIMS: [usize; 8] = [16, 32, 64, 128, 256, 512, 1024, 2048];

/// Minimum `d_a` for sparse fold ring challenges (no sampler below this).
pub const MIN_A_ROLE_FOLD_CHALLENGE_RING_D: usize = 64;

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

    /// Ring dimension for B-role data: next-witness digit commitments (`t_hat`),
    /// COMMIT and B_inner relation rows.
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

/// Validate every fold level's per-role ring dimensions against the setup seed.
///
/// Reads [`super::LevelParams::role_dims`] from each scheduled fold step; does
/// not copy them into a separate plan object.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when any catalog, key-consistency,
/// seed-divisibility, or witness-length check fails.
pub fn validate_schedule_ring_dims(
    schedule: &Schedule,
    seed: &AkitaSetupSeed,
) -> Result<(), AkitaError> {
    if seed.gen_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(
            "gen_ring_dim must be non-zero".to_string(),
        ));
    }
    let num_folds = schedule.num_fold_levels();
    if num_folds > MAX_FOLD_LEVELS {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule has {num_folds} fold levels, max supported is {MAX_FOLD_LEVELS}"
        )));
    }
    for level in 0..num_folds {
        let Some(step) = schedule.folds.get(level) else {
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
        if let Some(next) = schedule.folds.get(level + 1) {
            let next_ring_d = next.params.role_dims.inner;
            if next_ring_d == 0 || !step.next_w_len.is_multiple_of(next_ring_d) {
                return Err(AkitaError::InvalidSetup(format!(
                    "next witness length {} is not divisible by next fold ring d_a={next_ring_d}",
                    step.next_w_len,
                )));
            }
        }
    }
    Ok(())
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
    lp.fold_challenge_config
        .validate_for_ring_dim(lp.d_a())
        .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
    Ok(())
}

pub fn validate_role_dims(dims: CommitmentRingDims) -> Result<(), AkitaError> {
    for (role, d) in [
        (RingRole::Inner, dims.inner),
        (RingRole::Outer, dims.outer),
        (RingRole::Opening, dims.opening),
    ] {
        if d == 0 || !d.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(format!(
                "{role:?} ring dimension must be a non-zero power of two, got {d}"
            )));
        }
    }
    if !SUPPORTED_CHALLENGE_RING_DIMS.contains(&dims.inner) {
        return Err(AkitaError::InvalidSetup(format!(
            "A-role ring dimension d_a={} is unsupported for sparse fold challenges (need d_a >= {MIN_A_ROLE_FOLD_CHALLENGE_RING_D})",
            dims.inner
        )));
    }
    for (role, d) in [
        (RingRole::Outer, dims.outer),
        (RingRole::Opening, dims.opening),
    ] {
        if !SUPPORTED_RING_DIMS.contains(&d) {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported {:?} ring dimension {d}",
                role
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
    use crate::schedule::{FoldStep, Schedule, TerminalWitnessPlan};
    use crate::sis::SisModulusProfileId;
    use crate::SegmentTypedWitnessShape;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::AkitaError;

    fn fold_challenge_config_for_ring_dim(ring_dimension: usize) -> SparseChallengeConfig {
        SparseChallengeConfig::production_for_ring_dim(ring_dimension).unwrap_or_else(|| {
            // Rejection/fixture paths outside the production ladder still need a family
            // that clears the 128-bit entropy floor at `ring_dimension`.
            SparseChallengeConfig::pm1_only(ring_dimension.max(31))
        })
    }

    fn make_fold_level_params(
        ring_dimension: usize,
        num_live_blocks: usize,
        num_positions_per_block: usize,
    ) -> LevelParams {
        LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            ring_dimension,
            3,
            1,
            1,
            1,
            fold_challenge_config_for_ring_dim(ring_dimension),
        )
        .with_decomp(
            num_positions_per_block,
            num_live_blocks * num_positions_per_block,
            2,
            2,
        )
        .expect("valid ring-dimension test params")
    }

    fn make_fold_step(
        ring_dimension: usize,
        num_live_blocks: usize,
        num_positions_per_block: usize,
    ) -> FoldStep {
        FoldStep {
            params: make_fold_level_params(
                ring_dimension,
                num_live_blocks,
                num_positions_per_block,
            ),
            current_w_len: 0,
            next_w_len: 0,
            level_bytes: 0,
        }
    }

    fn make_direct_step() -> TerminalWitnessPlan {
        TerminalWitnessPlan {
            current_w_len: 0,
            witness_shape: SegmentTypedWitnessShape {
                layout: crate::TailSegmentLayout {
                    ring_dimension: 64,
                    log_basis: 3,
                    groups: vec![crate::TailSegmentGroupLayout {
                        z_coords: 1,
                        e_field_elems: 0,
                        t_field_elems: 0,
                        z_payload_bytes: 1,
                    }],
                    logical_num_elems: 0,
                },
            },
            terminal_bytes: 0,
        }
    }

    fn uniform_schedule(ring_dimension: usize, num_levels: usize) -> Schedule {
        Schedule {
            folds: (0..num_levels)
                .map(|_| make_fold_step(ring_dimension, 4, 8))
                .collect(),
            terminal: make_direct_step(),
            total_bytes: 0,
        }
    }

    fn mixed_d_schedule(dims: &[(usize, usize, usize)]) -> Schedule {
        Schedule {
            folds: dims
                .iter()
                .map(|&(d, nb, bl)| make_fold_step(d, nb, bl))
                .collect(),
            terminal: make_direct_step(),
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
        validate_schedule_ring_dims(&sched, &seed(256)).expect("256|256");
        assert_eq!(sched.num_fold_levels(), 3);
    }

    #[test]
    fn accepts_d_divides_gen_ring_dim() {
        let sched = uniform_schedule(64, 2);
        validate_schedule_ring_dims(&sched, &seed(256)).expect("64|256");
    }

    #[test]
    fn accepts_mixed_d_schedule_when_all_dims_divide_gen_ring_dim() {
        let sched = mixed_d_schedule(&[(64, 4, 8), (128, 4, 4), (256, 2, 2)]);
        validate_schedule_ring_dims(&sched, &seed(256)).expect("all divide 256");
        assert_eq!(sched.num_fold_levels(), 3);
    }

    #[test]
    fn level_params_flat_field_len_matches_ring_elems_times_ring_dim() {
        let sched = uniform_schedule(64, 1);
        let step = &sched.folds[0];
        assert_eq!(step.params.n_ring_elems().expect("n_ring"), 32);
        assert_eq!(step.params.flat_field_len().expect("flat"), 2048);
    }

    #[test]
    fn schedule_with_no_fold_steps_is_valid() {
        let sched = Schedule {
            folds: Vec::new(),
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        validate_schedule_ring_dims(&sched, &seed(256)).expect("no folds");
        assert_eq!(sched.num_fold_levels(), 0);
    }

    #[test]
    fn rejects_zero_gen_ring_dim() {
        let sched = uniform_schedule(64, 1);
        let err = validate_schedule_ring_dims(&sched, &seed(0)).expect_err("gen=0");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_does_not_divide_gen_ring_dim() {
        let sched = mixed_d_schedule(&[(96, 4, 4)]);
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("96|256");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_zero() {
        let mut fold = make_fold_step(64, 4, 4);
        fold.params.ring_dimension = 0;
        let sched = Schedule {
            folds: vec![fold],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("d=0");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_non_power_of_two_role_dims() {
        let err = validate_role_dims(CommitmentRingDims {
            inner: 128,
            outer: 48,
            opening: 16,
        })
        .expect_err("outer role dim is not a power of two");
        assert!(
            matches!(err, AkitaError::InvalidSetup(message) if message.contains("power of two"))
        );
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
            folds: vec![fold_step_with_witness_lens(64, 65, 64)],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("65 % 64");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_next_witness_length_not_divisible_by_next_d_a() {
        let sched = Schedule {
            folds: vec![
                fold_step_with_witness_lens(64, 64, 65),
                make_fold_step(64, 4, 8),
            ],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("next 65 % 64");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_too_many_fold_levels() {
        let sched = Schedule {
            folds: (0..MAX_FOLD_LEVELS + 1)
                .map(|_| make_fold_step(64, 4, 8))
                .collect(),
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("> MAX_FOLD_LEVELS");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_role_dims_key_mismatch() {
        let mut step = make_fold_step(128, 4, 8);
        step.params.role_dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        };
        let sched = Schedule {
            folds: vec![step],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("B-key still 64");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn accepts_nested_per_role_dims_with_matching_keys() {
        use crate::layout::{AjtaiKeyParams, SisModulusProfileId};
        use crate::sis::DEFAULT_SIS_SECURITY_POLICY;

        let mut params = make_fold_level_params(256, 4, 8);
        params.a_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::A,
            1,
            16,
            0,
            256,
        );
        params.b_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::B,
            1,
            16,
            0,
            128,
        );
        params.d_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::D,
            1,
            16,
            0,
            64,
        );
        params.stamp_role_dims_from_keys();
        let sched = Schedule {
            folds: vec![FoldStep {
                params,
                current_w_len: 256,
                next_w_len: 128,
                level_bytes: 0,
            }],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        validate_schedule_ring_dims(&sched, &seed(256)).expect("nested 256|128|64");
        let step = &sched.folds[0];
        let dims = step.params.role_dims;
        assert_eq!(dims.d_a(), 256);
        assert_eq!(dims.d_b(), 128);
        assert_eq!(dims.d_d(), 64);
    }

    #[test]
    fn accepts_nested_role_dims_with_opening_at_d32() {
        use crate::layout::{AjtaiKeyParams, SisModulusProfileId};
        use crate::sis::DEFAULT_SIS_SECURITY_POLICY;

        let mut params = make_fold_level_params(128, 4, 8);
        params.a_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::A,
            1,
            16,
            0,
            128,
        );
        params.b_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::B,
            1,
            16,
            0,
            64,
        );
        params.d_key = AjtaiKeyParams::new_unchecked(
            DEFAULT_SIS_SECURITY_POLICY,
            crate::sis::SisTableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            crate::sis::SisMatrixRole::D,
            1,
            16,
            0,
            32,
        );
        params.stamp_role_dims_from_keys();
        let sched = Schedule {
            folds: vec![FoldStep {
                params,
                current_w_len: 128,
                next_w_len: 64,
                level_bytes: 0,
            }],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        validate_schedule_ring_dims(&sched, &seed(128)).expect("nested 128|64|32");
        let step = &sched.folds[0];
        let dims = step.params.role_dims;
        assert_eq!(dims.d_a(), 128);
        assert_eq!(dims.d_b(), 64);
        assert_eq!(dims.d_d(), 32);
    }

    #[test]
    fn rejects_inner_ring_dim_32_for_fold_challenge() {
        let mut step = make_fold_step(64, 4, 8);
        step.params.role_dims = CommitmentRingDims {
            inner: 32,
            outer: 32,
            opening: 32,
        };
        let sched = Schedule {
            folds: vec![step],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("d_a=32");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_ring_dimension_mismatch_with_role_dims() {
        let mut step = make_fold_step(64, 4, 8);
        step.params.ring_dimension = 128;
        let sched = Schedule {
            folds: vec![step],
            terminal: make_direct_step(),
            total_bytes: 0,
        };
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("ring_dim != d_a");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }

    #[test]
    fn rejects_level_ring_dimension_larger_than_gen_ring_dim() {
        let sched = mixed_d_schedule(&[(512, 4, 4)]);
        let err = validate_schedule_ring_dims(&sched, &seed(256)).expect_err("512|256");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
