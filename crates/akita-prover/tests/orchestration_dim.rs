//! S9a orchestration gates: schedule authority and level-dispatch validation.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::test_support::ring_plan_test_seed;
use akita_config::{effective_batched_schedule, CommitmentConfig};
use akita_field::AkitaError;
use akita_types::{
    validate_role_dispatch, validate_schedule_ring_dims, AkitaScheduleLookupKey, FoldStep,
    LevelParams, OpeningClaimsLayout, RingRole, Schedule, SegmentTypedWitnessShape,
    SisModulusProfileId, TailSegmentGroupLayout, TailSegmentLayout, TerminalWitnessPlan,
};

fn real_schedule<Cfg: CommitmentConfig>(num_vars: usize) -> Schedule {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        akita_types::PolynomialGroupLayout::singleton(num_vars),
    ))
    .expect("valid schedule for num_vars")
}

fn test_level_params(ring_dimension: usize) -> LevelParams {
    LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(ring_dimension)
            .expect("supported test ring dimension"),
    )
    .with_decomp(8, 32, 2, 2)
    .expect("valid test level params")
}

fn make_fold_step(ring_dimension: usize) -> FoldStep {
    FoldStep {
        params: test_level_params(ring_dimension),
        current_w_len: 256,
        next_w_len: 128,
        level_bytes: 0,
    }
}

#[test]
fn batched_schedule_selection_matches_config_preset() {
    type Cfg = fp64::D64Full;
    let nv = 14usize;
    let schedule = real_schedule::<Cfg>(nv);
    let opening_batch = OpeningClaimsLayout::new(nv, 1).expect("opening batch");
    let point = vec![<Cfg as CommitmentConfig>::ExtField::zero(); nv];
    let effective = effective_batched_schedule::<Cfg>(&opening_batch, &point).expect("schedule");
    assert_eq!(effective.folds.len(), schedule.folds.len());
}

#[test]
fn ring_dim_plan_rejects_level_dim_larger_than_gen_ring_dim() {
    let schedule = Schedule {
        folds: vec![make_fold_step(128)],
        terminal: TerminalWitnessPlan {
            current_w_len: 64,
            witness_shape: SegmentTypedWitnessShape {
                layout: TailSegmentLayout {
                    ring_dimension: 64,
                    log_basis: 3,
                    groups: vec![TailSegmentGroupLayout {
                        z_coords: 1,
                        e_field_elems: 64,
                        t_field_elems: 0,
                        z_payload_bytes: 1,
                    }],
                    logical_num_elems: 64,
                },
            },
            terminal_bytes: 0,
        },
        total_bytes: 0,
    };
    let err = validate_schedule_ring_dims(&schedule, &ring_plan_test_seed(64))
        .expect_err("gen_ring_dim=64 cannot host a fold level at ring_dimension=128");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn validate_role_dispatch_rejects_stack_d_mismatch() {
    let params = test_level_params(128);
    let err = validate_role_dispatch::<64>(params.role_dims, RingRole::Inner)
        .expect_err("stack D=64 vs level 128");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_dim_plan_accepts_uniform_d64_preset() {
    type Cfg = fp64::D64Full;
    let schedule = real_schedule::<Cfg>(14);
    validate_schedule_ring_dims(&schedule, &ring_plan_test_seed(Cfg::D))
        .expect("uniform preset envelope");
}

#[test]
fn ring_dim_plan_accepts_fp128_d64_preset() {
    type Cfg = fp128::D64Full;
    let schedule = real_schedule::<Cfg>(13);
    validate_schedule_ring_dims(&schedule, &ring_plan_test_seed(Cfg::D))
        .expect("fp128 uniform preset envelope");
}
