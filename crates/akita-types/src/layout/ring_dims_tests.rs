use super::*;
use crate::{
    CommittedGroupParams, FoldSchedule, RecursiveFoldParams, RecursiveFoldStep, RootFinalChallenge,
    RootFinalGroupParams, RootFoldParams, RootFoldStep, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalCommittedGroupParams, TerminalFoldParams, TerminalFoldStep, TerminalResponseShape,
    WitnessPartition,
};
use akita_challenges::SparseChallengeConfig;

fn committed(ring_dimension: usize) -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(ring_dimension.max(31)),
    )
    .with_decomp(8, 32, 2, 2, 2)
    .expect("ring-dimension test params")
}

fn schedule(root: CommittedGroupParams, terminal: CommittedGroupParams) -> FoldSchedule {
    let terminal_witness = TerminalCommittedGroupParams::from_expanded_group(terminal);
    let ring_dimension = terminal_witness.d_a();
    FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    challenge: RootFinalChallenge::Flat,
                    commitment: root.clone(),
                },
                precommitted_groups: Vec::new(),
                open_commit_matrix: root.open_commit_matrix,
                sparse_challenge_config: root.fold_challenge_config,
                witness_partition: WitnessPartition::Single,
            },
            input_witness_len: root.d_a(),
            output_witness_len: ring_dimension,
        },
        recursive_folds: Vec::new(),
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: terminal_witness,
                sparse_challenge_config: SparseChallengeConfig::pm1_only(ring_dimension.max(31)),
                response_shape: TerminalResponseShape {
                    layout: TailSegmentLayout {
                        ring_dimension,
                        groups: vec![TailSegmentGroupLayout {
                            z_coords: ring_dimension,
                            e_field_elems: ring_dimension,
                            t_field_elems: ring_dimension,
                            z_admission_linf_cap: 1,
                            z_payload_bytes: 1,
                            z_rice_low_bits: 0,
                        }],
                        logical_num_elems: 3 * ring_dimension,
                    },
                },
            },
            input_witness_len: ring_dimension,
        },
    }
}

fn seed(num_field_elements: usize) -> AkitaSetupSeed {
    AkitaSetupSeed {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements,
        public_matrix_id: [0; 32].into(),
    }
}

#[test]
fn accepts_typed_root_and_terminal_ring_dimensions() {
    let schedule = schedule(committed(128), committed(64));
    let required = crate::setup_matrix_field_elements_for_schedule(&schedule).unwrap();
    validate_schedule_ring_dims(&schedule, &seed(required))
        .expect("exact field capacity covers mixed dimensions");
}

#[test]
fn rejects_recursive_shared_d_matrix_mismatch() {
    let root_params = committed(128);
    let recursive_params = committed(64);
    let terminal_params = committed(64);
    let terminal_witness =
        TerminalCommittedGroupParams::from_expanded_group(terminal_params.clone());
    let recursive_input_len = recursive_params.d_a();
    let schedule = FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    challenge: RootFinalChallenge::Flat,
                    commitment: root_params.clone(),
                },
                precommitted_groups: Vec::new(),
                open_commit_matrix: root_params.open_commit_matrix,
                sparse_challenge_config: root_params.fold_challenge_config,
                witness_partition: WitnessPartition::Single,
            },
            input_witness_len: root_params.d_a(),
            output_witness_len: recursive_input_len,
        },
        recursive_folds: vec![RecursiveFoldStep {
            params: RecursiveFoldParams {
                witness: recursive_params.clone(),
                open_commit_matrix: root_params.open_commit_matrix,
                sparse_challenge_config: recursive_params.fold_challenge_config,
                incoming_setup_prefix: None,
                witness_partition: WitnessPartition::Single,
            },
            input_witness_len: recursive_input_len,
            output_witness_len: terminal_witness.d_a(),
        }],
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: terminal_witness,
                sparse_challenge_config: terminal_params.fold_challenge_config,
                response_shape: TerminalResponseShape {
                    layout: TailSegmentLayout {
                        ring_dimension: terminal_params.d_a(),
                        groups: vec![TailSegmentGroupLayout {
                            z_coords: terminal_params.d_a(),
                            e_field_elems: terminal_params.d_a(),
                            t_field_elems: terminal_params.d_a(),
                            z_admission_linf_cap: 1,
                            z_payload_bytes: 1,
                            z_rice_low_bits: 0,
                        }],
                        logical_num_elems: 3 * terminal_params.d_a(),
                    },
                },
            },
            input_witness_len: terminal_params.d_a(),
        },
    };
    let required = crate::setup_matrix_field_elements_for_schedule(&schedule).unwrap();
    let err = validate_schedule_ring_dims(&schedule, &seed(required))
        .expect_err("recursive shared D mismatch must reject");
    assert!(err.to_string().contains("shared D matrix disagrees"));
}

#[test]
fn rejects_undersized_field_capacity() {
    let schedule = schedule(committed(128), committed(64));
    let required = crate::setup_matrix_field_elements_for_schedule(&schedule).unwrap();
    assert!(matches!(
        validate_schedule_ring_dims(&schedule, &seed(required - 1)),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn rejects_non_power_of_two_role_dimension() {
    assert!(matches!(
        validate_role_dims(CommitmentRingDims {
            inner: 128,
            outer: 48,
            opening: 16,
        }),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn accepts_either_b_d_order_below_a() {
    for dims in [
        CommitmentRingDims {
            inner: 128,
            outer: 32,
            opening: 64,
        },
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        },
    ] {
        validate_role_dims(dims).expect("B and D must not be ordered relative to each other");
    }
}

#[test]
fn rejects_b_or_d_larger_than_a() {
    for dims in [
        CommitmentRingDims {
            inner: 64,
            outer: 128,
            opening: 32,
        },
        CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 128,
        },
    ] {
        validate_role_dims(dims).expect_err("A must remain the largest role");
    }
}

#[test]
fn relation_and_witness_common_counts_are_distinct_contracts() {
    let uniform_roles = CommitmentRingDims::uniform(128);
    assert_eq!(uniform_roles.common_relation_coeff_count(), 128);
    assert_eq!(uniform_roles.common_relation_witness_coeff_count(64), 64);

    let mixed_roles = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 32,
    };
    assert_eq!(mixed_roles.common_relation_coeff_count(), 32);
    assert_eq!(mixed_roles.common_relation_witness_coeff_count(16), 16);
}
