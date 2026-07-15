//! Integration tests for [`akita_types::validate_schedule_ring_dims`] against real
//! generated schedules.
//!
//! These tests exercise the S3 building blocks from Tier 2 of the runtime
//! ring-dimension cutover plan.  They require real schedules from
//! `akita-config` presets and a hand-built mixed-D schedule.
//!
//! Runtime→const-D dispatch is provided by [`akita_types::dispatch_for_field!`] with
//! an explicit [`akita_types::ProtocolDispatchSlot`].
//! Routing/rejection coverage lives in `akita-types` (`dispatch/` unit tests).
//!
//! Gate condition from the plan:
//! `cargo test -p akita-prover dispatch` must pass.

#![allow(missing_docs)]

use akita_challenges::SparseChallengeConfig;
use akita_config::proof_optimized::{fp128, fp64};
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::{
    validate_schedule_ring_dims, AkitaScheduleLookupKey, AkitaSetupSeed, CleartextWitnessShape,
    DirectStep, FoldStep, LevelParams, Schedule, SisModulusProfileId, Step,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_seed(gen_ring_dim: usize) -> AkitaSetupSeed {
    AkitaSetupSeed {
        max_num_vars: 20,
        max_num_batched_polys: 1,
        gen_ring_dim,
        max_setup_len: 1 << 20,
        public_matrix_seed: [0u8; 32],
    }
}

/// Resolve a real schedule from a config preset at the given `num_vars`.
fn real_schedule<Cfg: CommitmentConfig>(num_vars: usize) -> Schedule {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        akita_types::PolynomialGroupLayout::singleton(num_vars),
    ))
    .expect("valid schedule for num_vars")
}

/// Build a minimal `FoldStep` with explicit ring dimension and geometry.
fn make_fold_step(ring_dimension: usize, num_blocks: usize, block_len: usize) -> FoldStep {
    let fold_challenge_config = SparseChallengeConfig::production_for_ring_dim(ring_dimension)
        .unwrap_or_else(|| SparseChallengeConfig::pm1_only(ring_dimension.max(31)));
    let mut params = LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        1,
        1,
        1,
        fold_challenge_config,
    );
    params.role_dims = akita_types::CommitmentRingDims::uniform(ring_dimension);
    params.num_blocks = num_blocks;
    params.block_len = block_len;
    params.num_digits_commit = 2;
    params.num_digits_open = 2;
    params.stamp_role_dims_from_keys();
    FoldStep {
        params,
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

fn assert_fold_level_geometry(sched: &Schedule, level: usize, ring_dimension: usize) {
    let Step::Fold(step) = &sched.steps[level] else {
        panic!("level {level} is not a fold step");
    };
    let lp = &step.params;
    assert_eq!(lp.d_a(), ring_dimension, "level {level} d_a");
    assert_eq!(
        lp.flat_field_len().expect("flat_field_len"),
        lp.n_ring_elems().expect("n_ring_elems") * lp.d_a(),
        "level {level} flat_field_len"
    );
}

// ---------------------------------------------------------------------------
// validate_schedule_ring_dims against REAL schedules (fp64::D64Full, fp128)
// ---------------------------------------------------------------------------

/// For fp64::D64Full, `Cfg::D == 64`, so `gen_ring_dim = 64` and every level
/// must carry `ring_dimension = 64` (uniform-D preset).
#[test]
fn ring_dim_plan_accepts_fp64_d64_schedule_with_gen_ring_dim_64() {
    let sched = real_schedule::<fp64::D64Full>(20);
    let gen_ring_dim = fp64::D64Full::D;
    assert_eq!(gen_ring_dim, 64);
    validate_schedule_ring_dims(&sched, &test_seed(gen_ring_dim))
        .expect("fp64 D64 schedule must be valid for gen_ring_dim=64");
    for level in 0..sched.num_fold_levels() {
        assert_fold_level_geometry(&sched, level, 64);
    }
}

/// For fp128, `Cfg::D == 128`; validate against gen_ring_dim=128.
#[test]
fn ring_dim_plan_accepts_fp128_schedule_with_gen_ring_dim_128() {
    type Cfg = fp128::D128Full;
    let sched = real_schedule::<Cfg>(18);
    let gen_ring_dim = Cfg::D;
    assert_eq!(gen_ring_dim, 128);
    validate_schedule_ring_dims(&sched, &test_seed(gen_ring_dim))
        .expect("fp128 D128 schedule must be valid for gen_ring_dim=128");
    for level in 0..sched.num_fold_levels() {
        assert_fold_level_geometry(&sched, level, 128);
    }
}

/// A fp64::D64Full schedule validated against gen_ring_dim=256 (64 | 256).
#[test]
fn ring_dim_plan_accepts_fp64_d64_schedule_against_larger_gen_ring_dim() {
    type Cfg = fp64::D64Full;
    let sched = real_schedule::<Cfg>(16);
    validate_schedule_ring_dims(&sched, &test_seed(256))
        .expect("D=64 schedule must be valid for gen_ring_dim=256 (64|256)");
    for level in 0..sched.num_fold_levels() {
        assert_fold_level_geometry(&sched, level, 64);
    }
}

/// Passing gen_ring_dim=64 for a D=128 schedule must fail: 128 ∤ 64.
#[test]
fn ring_dim_plan_rejects_fp128_schedule_against_too_small_gen_ring_dim() {
    type Cfg = fp128::D128Full;
    let sched = real_schedule::<Cfg>(16);
    let err = validate_schedule_ring_dims(&sched, &test_seed(64))
        .expect_err("128 does not divide 64; must be rejected");
    assert!(
        matches!(err, AkitaError::InvalidSetup(_)),
        "expected InvalidSetup, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// validate_schedule_ring_dims — hand-built mixed-D schedule
// ---------------------------------------------------------------------------

/// A schedule where each fold level has a *different* A-role ring dimension.
///
/// gen_ring_dim = 256; levels use d_a = 64, 128, 256 (all divide 256; d_a >= 64).
#[test]
fn ring_dim_plan_accepts_mixed_d_schedule_all_divide_gen_ring_dim() {
    let sched = mixed_d_schedule(&[(64, 4, 4), (128, 2, 4), (256, 2, 2)]);
    validate_schedule_ring_dims(&sched, &test_seed(256)).expect("all dims divide 256");
    assert_eq!(sched.num_fold_levels(), 3);

    let expected = [
        (64usize, 16usize, 1024usize),
        (128, 8, 1024),
        (256, 4, 1024),
    ];
    for (level, (d, nr, ff)) in expected.into_iter().enumerate() {
        let Step::Fold(step) = &sched.steps[level] else {
            panic!("level {level} is not a fold step");
        };
        let lp = &step.params;
        assert_eq!(lp.d_a(), d, "level {level} d_a");
        assert_eq!(
            lp.n_ring_elems().expect("n_ring_elems"),
            nr,
            "level {level} n_ring_elems"
        );
        assert_eq!(
            lp.flat_field_len().expect("flat_field_len"),
            ff,
            "level {level} flat_field_len"
        );
    }
}

/// Nested per-role dims: B/D may use D=32 while d_a >= 64.
#[test]
fn ring_dim_plan_accepts_nested_opening_d32() {
    use akita_types::sis::DEFAULT_SIS_SECURITY_POLICY;
    use akita_types::{AjtaiKeyParams, SisMatrixRole, SisModulusProfileId, SisTableDigest};

    let mut step = make_fold_step(128, 4, 8);
    step.params.ring_dimension = 128;
    step.params.a_key = AjtaiKeyParams::new_unchecked(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::A,
        1,
        16,
        0,
        128,
    );
    step.params.b_key = AjtaiKeyParams::new_unchecked(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::B,
        1,
        16,
        0,
        64,
    );
    step.params.d_key = AjtaiKeyParams::new_unchecked(
        DEFAULT_SIS_SECURITY_POLICY,
        SisTableDigest::CURRENT,
        SisModulusProfileId::Q128OffsetA7F7,
        SisMatrixRole::D,
        1,
        16,
        0,
        32,
    );
    step.params.stamp_role_dims_from_keys();
    step.params.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(step.params.d_a()).expect("d_a ladder");
    step.current_w_len = 128;
    let sched = Schedule {
        steps: vec![Step::Fold(step), Step::Direct(make_direct_step())],
        total_bytes: 0,
    };
    validate_schedule_ring_dims(&sched, &test_seed(128)).expect("128|64|32");
    let Step::Fold(step) = &sched.steps[0] else {
        panic!("expected fold");
    };
    let dims = step.params.role_dims;
    assert_eq!(dims.d_a(), 128);
    assert_eq!(dims.d_b(), 64);
    assert_eq!(dims.d_d(), 32);
}

/// A mixed-D schedule where one level's ring_dimension does NOT divide
/// gen_ring_dim — this must be rejected.
#[test]
fn ring_dim_plan_rejects_mixed_d_schedule_one_bad_level() {
    let sched = mixed_d_schedule(&[(64, 4, 4), (96, 4, 4), (128, 2, 4)]);
    let err = validate_schedule_ring_dims(&sched, &test_seed(256))
        .expect_err("96 does not divide 256; must be rejected");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

/// Divisibility holds at the first level but gen_ring_dim < ring_dimension
/// at a later level — must be rejected.
#[test]
fn ring_dim_plan_rejects_level_ring_dim_larger_than_gen_ring_dim() {
    let sched = mixed_d_schedule(&[(64, 4, 4), (512, 2, 2)]);
    let err =
        validate_schedule_ring_dims(&sched, &test_seed(256)).expect_err("512 does not divide 256");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}
