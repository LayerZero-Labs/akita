//! Mixed-D and per-role rejection tests for the verifier crate (spec
//! `runtime-ring-cutover.md` Slice 4).

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::test_support::mixed_d_per_level_schedule;
use akita_field::AkitaError;
use akita_field::Prime128OffsetA7F7 as F;
use akita_types::{
    validate_role_dims, validate_role_dispatch, AkitaSetupSeed, CleartextWitnessShape,
    CommitmentRingDims, DirectStep, FoldStep, LevelParams, RingDimPlan, RingRole, RingView,
    Schedule, Step,
};

type Envelope = fp128::D128Full;
type Suffix = fp128::D64Full;

const MIXED_D_SWITCH_FOLD: usize = 2;
const NUM_VARS: usize = 16;

fn mixed_schedule() -> Schedule {
    mixed_d_per_level_schedule::<Envelope, Suffix>(NUM_VARS, 1, MIXED_D_SWITCH_FOLD)
        .expect("mixed-D schedule")
}

#[test]
fn nested_role_dims_reject_non_nesting() {
    let bad = CommitmentRingDims {
        inner: 64,
        outer: 128,
        opening: 32,
    };
    assert!(!bad.nests());
    validate_role_dims(bad).expect_err("non-nesting role dims rejected");
}

#[test]
fn per_role_dispatch_rejects_wrong_stack_d() {
    let dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 32,
    };
    assert!(dims.nests());
    validate_role_dispatch::<64>(dims, RingRole::Inner).expect_err("A-role requires d_a=128");
    validate_role_dispatch::<128>(dims, RingRole::Inner).expect("A-role at 128");
    validate_role_dispatch::<64>(dims, RingRole::Outer).expect("B-role at 64");
    validate_role_dispatch::<32>(dims, RingRole::Opening).expect("D-role at 32");
}

fn test_seed(gen_ring_dim: usize) -> AkitaSetupSeed {
    AkitaSetupSeed {
        max_num_vars: NUM_VARS,
        max_num_batched_polys: 1,
        gen_ring_dim,
        max_setup_len: 1 << 20,
        public_matrix_seed: [0u8; 32],
    }
}

#[test]
fn ring_dim_plan_rejects_fold_dim_above_gen_ring_dim() {
    let schedule = Schedule {
        steps: vec![
            Step::Fold({
                let mut step = FoldStep {
                    params: LevelParams::log_basis_stub(3),
                    current_w_len: 256,
                    next_w_len: 128,
                    level_bytes: 0,
                };
                step.params.ring_dimension = 128;
                step.params.role_dims = CommitmentRingDims::uniform(128);
                step.params.num_blocks = 4;
                step.params.block_len = 8;
                step
            }),
            Step::Direct(DirectStep {
                current_w_len: 64,
                witness_shape: CleartextWitnessShape::FieldElements(64),
                direct_bytes: 0,
                params: None,
            }),
        ],
        total_bytes: 0,
    };
    let err = RingDimPlan::from_schedule(&schedule, &test_seed(64))
        .expect_err("gen_ring_dim=64 cannot host fold d_a=128");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn nested_role_dims_b_role_row_count_differs_from_a_role() {
    let dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 32,
    };
    assert!(dims.nests());
    let coeffs = vec![F::zero(); 128];
    assert_eq!(
        RingView::new(&coeffs, dims.d_b())
            .expect("valid at B-role d_b")
            .num_rings(),
        2
    );
    assert_eq!(
        RingView::new(&coeffs, dims.d_a())
            .expect("valid at A-role d_a")
            .num_rings(),
        1
    );
}

#[test]
fn ring_dim_plan_accepts_mixed_d_per_level_fixture() {
    let schedule = mixed_schedule();
    let seed = akita_types::AkitaSetupSeed {
        max_num_vars: NUM_VARS,
        max_num_batched_polys: 1,
        gen_ring_dim: 128,
        max_setup_len: 1 << 20,
        public_matrix_seed: [0u8; 32],
    };
    let plan = RingDimPlan::from_schedule(&schedule, &seed).expect("mixed-D plan");
    assert_eq!(plan.dim_at(0).expect("level 0"), 128);
    assert_eq!(plan.dim_at(2).expect("level 2"), 64);
}
