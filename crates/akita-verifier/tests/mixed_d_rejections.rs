//! Mixed-ring-dimension rejection tests for the typed fold schedule.

#![allow(missing_docs)]

use akita_field::Prime128OffsetA7F7 as F;
use akita_types::{
    validate_role_dims, validate_role_dispatch, validate_schedule_ring_dims, CommitmentRingDims,
    CommittedGroupParams, FoldParams, FoldSchedule, RingRole, RingView, SisModulusProfileId,
    TailSegmentGroupLayout, TailSegmentLayout, TerminalCommittedGroupParams, TerminalFoldParams,
    TerminalFoldStep, TerminalResponseShape,
};

#[test]
fn role_dims_accept_either_b_d_order_below_a() {
    let dims = CommitmentRingDims {
        inner: 256,
        outer: 64,
        opening: 128,
    };
    validate_role_dims(dims).expect("D may be larger than B");
}

#[test]
fn role_dims_reject_b_larger_than_a() {
    let dims = CommitmentRingDims {
        inner: 64,
        outer: 128,
        opening: 32,
    };
    validate_role_dims(dims).expect_err("B and D dimensions must divide the A-native source");
}

#[test]
fn per_role_dispatch_rejects_wrong_stack_d() {
    let dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 32,
    };
    validate_role_dispatch::<64>(dims, RingRole::Inner).expect_err("A role requires 128");
    validate_role_dispatch::<128>(dims, RingRole::Inner).expect("A role");
    validate_role_dispatch::<64>(dims, RingRole::Outer).expect("B role");
    validate_role_dispatch::<32>(dims, RingRole::Opening).expect("D role");
}

fn params(ring_dimension: usize) -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(ring_dimension)
            .expect("challenge config"),
    )
    .with_decomp(8, 32, 2, 2, 2)
    .expect("test params")
}

#[test]
fn typed_schedule_accepts_root_dimension_independent_of_flat_setup() {
    let root = params(128);
    let terminal_witness = TerminalCommittedGroupParams::from_expanded_group(params(64));
    let schedule = FoldSchedule {
        root: FoldParams {
            params: root.clone(),
            input_witness_len: 256,
            output_witness_len: 64,
        },
        recursive_folds: Vec::new(),
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: terminal_witness,
                sparse_challenge_config:
                    akita_challenges::SparseChallengeConfig::production_for_ring_dim(64)
                        .expect("terminal challenge"),
                response_shape: TerminalResponseShape {
                    layout: TailSegmentLayout {
                        ring_dimension: 64,
                        groups: vec![TailSegmentGroupLayout {
                            z_coords: 64,
                            e_field_elems: 64,
                            t_field_elems: 64,
                            z_linf_cap: Some(1),
                            z_payload_bytes: 1,
                            z_rice_low_bits: 0,
                        }],
                        logical_num_elems: 192,
                    },
                },
            },
            input_witness_len: 64,
        },
    };
    validate_schedule_ring_dims(&schedule)
        .expect("flat setup capacity has no generation ring dimension");
}

#[test]
fn mixed_role_dims_change_flat_row_count() {
    let coeffs = vec![F::zero(); 128];
    assert_eq!(RingView::new(&coeffs, 64).expect("B view").num_rings(), 2);
    assert_eq!(RingView::new(&coeffs, 128).expect("A view").num_rings(), 1);
}
