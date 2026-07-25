use super::*;
use crate::{
    CommittedGroupParams, FoldSchedule, RootFinalChallenge, RootFinalGroupParams, RootFoldParams,
    RootFoldStep, RootSource, TailSegmentGroupLayout, TailSegmentLayout,
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

fn seed(gen_ring_dim: usize) -> AkitaSetupSeed {
    AkitaSetupSeed {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        gen_ring_dim,
        max_setup_len: 0,
        public_matrix_seed: [0; 32],
    }
}

#[test]
fn accepts_typed_root_and_terminal_ring_dimensions() {
    let schedule = schedule(committed(128), committed(64));
    validate_schedule_ring_dims(&schedule, &seed(256)).expect("128 and 64 divide setup D");
}

#[test]
fn rejects_terminal_dimension_not_dividing_setup_dimension() {
    let schedule = schedule(committed(128), committed(64));
    assert!(matches!(
        validate_schedule_ring_dims(&schedule, &seed(96)),
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
