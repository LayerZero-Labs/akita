//! Mixed-ring-dimension rejection tests for the typed fold schedule.

#![allow(missing_docs)]

use akita_field::{AkitaError, Prime128OffsetA7F7 as F};
use akita_types::{
    validate_role_dims, validate_role_dispatch, validate_schedule_ring_dims, AkitaSetupSeed,
    CommitmentRingDims, CommittedGroupParams, FoldSchedule, RingRole, RingView, RootFinalChallenge,
    RootFinalGroupParams, RootFoldParams, RootFoldStep, RootSource, SisModulusProfileId,
    TailSegmentGroupLayout, TailSegmentLayout, TerminalCommittedGroupParams, TerminalFoldParams,
    TerminalFoldStep, TerminalResponseShape, WitnessPartition,
};

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
fn typed_schedule_rejects_root_dimension_above_setup_dimension() {
    let root = params(128);
    let terminal_witness = TerminalCommittedGroupParams::from_expanded_group(params(64));
    let schedule = FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    source: RootSource::Dense {
                        coefficient_bits: 128,
                    },
                    challenge: RootFinalChallenge::Flat,
                    commitment: root.clone(),
                },
                precommitted_groups: Vec::new(),
                open_commit_matrix: root.open_commit_matrix.clone(),
                sparse_challenge_config: root.fold_challenge_config,
                witness_partition: WitnessPartition::Single,
            },
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
    let seed = AkitaSetupSeed {
        max_num_vars: 16,
        max_num_batched_polys: 1,
        gen_ring_dim: 64,
        max_setup_len: 1 << 20,
        public_matrix_seed: [0; 32],
    };
    assert!(matches!(
        validate_schedule_ring_dims(&schedule, &seed),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn nested_role_dims_change_flat_row_count() {
    let coeffs = vec![F::zero(); 128];
    assert_eq!(RingView::new(&coeffs, 64).expect("B view").num_rings(), 2);
    assert_eq!(RingView::new(&coeffs, 128).expect("A view").num_rings(), 1);
}
