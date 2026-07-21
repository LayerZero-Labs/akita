//! Mixed-D and per-role rejection tests for the verifier crate (spec
//! `runtime-ring-cutover.md` Slice 4).

#![allow(missing_docs)]

use akita_field::AkitaError;
use akita_field::Prime128OffsetA7F7 as F;
use akita_types::{
    validate_role_dims, validate_role_dispatch, validate_schedule_ring_dims, AkitaSetupSeed,
    CommitmentRingDims, FoldStep, LevelParams, RingRole, RingView, Schedule, SisModulusProfileId,
    TailSegmentGroupLayout, TailSegmentLayout, TerminalResponseShape, TerminalWitnessPlan,
};

const NUM_VARS: usize = 16;

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
    .with_decomp(8, 32, 2, 2, 2)
    .expect("valid test level params")
}

#[test]
fn ring_dim_plan_rejects_fold_dim_above_gen_ring_dim() {
    let schedule = Schedule {
        folds: vec![FoldStep {
            params: test_level_params(128),
            current_w_len: 256,
            next_w_len: 128,
            level_bytes: 0,
        }],
        terminal: TerminalWitnessPlan {
            current_w_len: 64,
            witness_shape: TerminalResponseShape {
                layout: TailSegmentLayout {
                    ring_dimension: 64,
                    log_basis_open: 3,
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
    let err = validate_schedule_ring_dims(&schedule, &test_seed(64))
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
