//! S9a orchestration gates: schedule authority and level-dispatch validation.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::test_support::ring_plan_test_seed;
use akita_config::{effective_batched_schedule, CommitmentConfig};
use akita_field::AkitaError;
use akita_types::{
    validate_role_dispatch, AkitaScheduleLookupKey, CleartextWitnessShape, DirectStep, FoldStep,
    LevelParams, OpeningClaimsLayout, RingDimPlan, RingRole, Schedule, Step,
};

fn real_schedule<Cfg: CommitmentConfig>(num_vars: usize) -> Schedule {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        akita_types::PolynomialGroupLayout::singleton(num_vars),
    ))
    .expect("valid schedule for num_vars")
}

fn make_fold_step(ring_dimension: usize) -> FoldStep {
    let mut params = LevelParams::log_basis_stub(3);
    params.ring_dimension = ring_dimension;
    params.role_dims = akita_types::CommitmentRingDims::uniform(ring_dimension);
    params.num_blocks = 4;
    params.block_len = 8;
    FoldStep {
        params,
        current_w_len: 256,
        next_w_len: 128,
        level_bytes: 0,
    }
}

#[test]
fn batched_schedule_selection_matches_config_preset() {
    type Cfg = fp64::D64Full;
    let nv = 10usize;
    let schedule = real_schedule::<Cfg>(nv);
    let opening_batch = OpeningClaimsLayout::new(nv, 1).expect("opening batch");
    let point = vec![<Cfg as CommitmentConfig>::ExtField::zero(); nv];
    let effective = effective_batched_schedule::<Cfg>(&opening_batch, &point).expect("schedule");
    assert_eq!(effective.steps.len(), schedule.steps.len());
}

#[test]
fn ring_dim_plan_rejects_level_dim_larger_than_gen_ring_dim() {
    let schedule = Schedule {
        steps: vec![
            Step::Fold(make_fold_step(128)),
            Step::Direct(DirectStep {
                current_w_len: 64,
                witness_shape: CleartextWitnessShape::FieldElements(64),
                direct_bytes: 0,
                params: None,
            }),
        ],
        total_bytes: 0,
    };
    let err = RingDimPlan::from_schedule(&schedule, &ring_plan_test_seed(64))
        .expect_err("gen_ring_dim=64 cannot host a fold level at ring_dimension=128");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn validate_role_dispatch_rejects_stack_d_mismatch() {
    let mut params = LevelParams::log_basis_stub(3);
    params.ring_dimension = 128;
    params.role_dims = akita_types::CommitmentRingDims::uniform(128);
    let err = validate_role_dispatch::<64>(params.role_dims, RingRole::Inner)
        .expect_err("stack D=64 vs level 128");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_dim_plan_accepts_uniform_d64_preset() {
    type Cfg = fp64::D64Full;
    let schedule = real_schedule::<Cfg>(10);
    RingDimPlan::from_schedule(&schedule, &ring_plan_test_seed(Cfg::D))
        .expect("uniform preset envelope");
}

#[test]
fn ring_dim_plan_accepts_fp128_d64_preset() {
    type Cfg = fp128::D64Full;
    let schedule = real_schedule::<Cfg>(12);
    RingDimPlan::from_schedule(&schedule, &ring_plan_test_seed(Cfg::D))
        .expect("fp128 uniform preset envelope");
}
