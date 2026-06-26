//! Rejection paths for `RingDimPlan::from_schedule`.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_types::{
    segment_typed_witness_shape, AkitaSetupSeed, CommitmentRingDims, DirectStep, FoldStep,
    LevelParams, RingDimPlan, Schedule, Step,
};

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
        akita_types::SisModulusFamily::Q128,
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
    DirectStep {
        current_w_len,
        witness_shape: segment_typed_witness_shape(lp, 128, 1, 1, 1, 1).expect("terminal shape"),
        direct_bytes: 0,
        params: None,
    }
}

fn one_fold_schedule(lp: LevelParams, current_w_len: usize, next_w_len: usize) -> Schedule {
    Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: lp.clone(),
                current_w_len,
                next_w_len,
                level_bytes: 0,
            }),
            Step::Direct(terminal_direct_step(&lp, next_w_len)),
        ],
        total_bytes: 0,
    }
}

#[test]
fn rejects_unsupported_ring_dimension() {
    let lp = sample_lp(16);
    let schedule = one_fold_schedule(lp, 64, 32);
    let err = RingDimPlan::from_schedule(&schedule, &sample_seed(16)).expect_err("unsupported d");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn rejects_envelope_not_divisible_by_fold_ring() {
    let schedule = one_fold_schedule(sample_lp(128), 512, 256);
    let err = RingDimPlan::from_schedule(&schedule, &sample_seed(64)).expect_err("bad envelope");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn rejects_witness_length_not_divisible_by_ring() {
    let schedule = one_fold_schedule(sample_lp(64), 65, 32);
    let err = RingDimPlan::from_schedule(&schedule, &sample_seed(64)).expect_err("bad witness");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn nesting_helper_accepts_valid_triple() {
    let dims = CommitmentRingDims::uniform(128);
    assert!(dims.nests());
}

#[test]
fn nesting_helper_rejects_non_nested_triple() {
    let dims = CommitmentRingDims {
        inner: 128,
        outer: 96,
        opening: 64,
    };
    assert!(!dims.nests());
}
