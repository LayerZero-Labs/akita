use std::{num::NonZeroUsize, sync::Arc};

use super::{select_complete_candidate, CompleteObjectiveBound, CompleteScheduleScore};
use crate::schedule_params::CandidateFoldChain;

fn score(objective: CompleteObjectiveBound, descriptor: u8) -> CompleteScheduleScore {
    CompleteScheduleScore {
        objective,
        legacy_root_output_witness_len: Some(1_000),
        descriptor: vec![descriptor],
    }
}

fn direct(
    proof_bytes: usize,
    setup_field_elements: usize,
    descriptor: u8,
) -> CompleteScheduleScore {
    score(
        CompleteObjectiveBound::Direct {
            proof_bytes,
            setup_field_elements,
        },
        descriptor,
    )
}

fn setup_first(
    first_direct_setup_capacity: usize,
    proof_bytes: usize,
    setup_field_elements: usize,
    descriptor: u8,
) -> CompleteScheduleScore {
    score(
        CompleteObjectiveBound::SetupFirst {
            first_direct_setup_capacity,
            proof_bytes,
            setup_field_elements,
        },
        descriptor,
    )
}

fn padded_setup_envelope_first(
    setup_field_elements: usize,
    first_direct_setup_capacity: usize,
    first_direct_output_witness_len: usize,
    proof_bytes: usize,
    descriptor: u8,
) -> CompleteScheduleScore {
    CompleteScheduleScore {
        objective: CompleteObjectiveBound::PaddedSetupEnvelopeFirst {
            setup_envelope_capacity: akita_types::padded_setup_prefix_len(setup_field_elements),
            first_direct_setup_capacity,
            proof_bytes,
            first_direct_output_witness_len,
        },
        legacy_root_output_witness_len: None,
        descriptor: vec![descriptor],
    }
}

#[test]
fn direct_score_prefers_setup_only_after_proof_ties() {
    let smaller_proof = direct(99, 1_000, 2);
    let smaller_setup = direct(100, 1, 1);
    assert!(smaller_proof < smaller_setup);

    let same_proof_smaller_setup = direct(99, 999, 3);
    assert!(same_proof_smaller_setup < smaller_proof);

    let complete_tie_smaller_descriptor = direct(99, 999, 1);
    assert!(complete_tie_smaller_descriptor < same_proof_smaller_setup);
}

#[test]
fn setup_first_score_uses_total_setup_only_after_primary_coordinates() {
    let smaller_proof = setup_first(16, 99, 1_000, 2);
    let smaller_total_setup = setup_first(16, 100, 1, 1);
    assert!(smaller_proof < smaller_total_setup);

    let same_proof_smaller_total_setup = setup_first(16, 99, 999, 3);
    assert!(same_proof_smaller_total_setup < smaller_proof);
}

#[test]
fn padded_setup_envelope_tolerates_raw_setup_within_one_capacity() {
    let smaller_direct_capacity = padded_setup_envelope_first(5_680_128, 2_097_152, 1_000, 101, 2);
    let smaller_raw_setup = padded_setup_envelope_first(8_388_608, 8_388_608, 1, 99, 1);
    assert_eq!(akita_types::padded_setup_prefix_len(5_680_128), 8_388_608);
    assert!(smaller_direct_capacity < smaller_raw_setup);

    let next_capacity = padded_setup_envelope_first(8_388_609, 1, 1, 1, 1);
    assert!(smaller_direct_capacity < next_capacity);
}

#[test]
fn padded_setup_envelope_prefers_proof_before_first_direct_output() {
    let smaller_output = padded_setup_envelope_first(30, 16, 999, 101, 2);
    let smaller_proof = padded_setup_envelope_first(31, 16, 1_000, 100, 1);
    assert!(smaller_proof < smaller_output);

    let smaller_descriptor = padded_setup_envelope_first(31, 16, 999, 101, 1);
    assert!(smaller_descriptor < smaller_output);
}

#[test]
fn padded_setup_envelope_uses_descriptor_after_output() {
    let smaller_descriptor = padded_setup_envelope_first(30, 16, 1_000, 100, 1);
    let larger_descriptor = padded_setup_envelope_first(30, 16, 1_000, 100, 2);
    assert_eq!(smaller_descriptor.legacy_root_output_witness_len, None);
    assert!(smaller_descriptor < larger_descriptor);
}

#[test]
fn output_witness_precedes_the_canonical_descriptor() {
    let objective = CompleteObjectiveBound::Direct {
        proof_bytes: 100,
        setup_field_elements: 1_000,
    };
    let smaller_output = CompleteScheduleScore {
        objective,
        legacy_root_output_witness_len: Some(999),
        descriptor: vec![2],
    };
    let smaller_descriptor = CompleteScheduleScore {
        objective,
        legacy_root_output_witness_len: Some(1_000),
        descriptor: vec![1],
    };
    assert!(smaller_output < smaller_descriptor);
}

#[test]
fn setup_first_score_compares_padded_capacity_not_natural_length() {
    let natural_9 = super::super::SetupPrefixCapacity::for_natural_len(9);
    let natural_15 = super::super::SetupPrefixCapacity::for_natural_len(15);
    assert_eq!(natural_9, natural_15);

    let better_proof_with_larger_natural_length =
        setup_first(natural_15.field_elements(), 99, 1_000, 2);
    let worse_proof_with_smaller_natural_length =
        setup_first(natural_9.field_elements(), 100, 1, 1);
    assert!(better_proof_with_larger_natural_length < worse_proof_with_smaller_natural_length);
}

#[test]
fn objective_bounds_prune_only_strict_numeric_losses() {
    let incumbent = super::super::CandidateMetrics {
        first_direct_setup_capacity: super::super::SetupPrefixCapacity::for_natural_len(10),
        first_direct_output_witness_len: 1_000,
        cost: super::super::PackedProofCost::new(20, 0).unwrap(),
        setup_field_elements: 30,
    };
    assert!(CompleteObjectiveBound::SetupFirst {
        first_direct_setup_capacity: 16,
        proof_bytes: 21,
        setup_field_elements: 0,
    }
    .is_strictly_worse_than(incumbent));
    assert!(!CompleteObjectiveBound::SetupFirst {
        first_direct_setup_capacity: 8,
        proof_bytes: usize::MAX,
        setup_field_elements: usize::MAX,
    }
    .is_strictly_worse_than(incumbent));
    assert!(!CompleteObjectiveBound::SetupFirst {
        first_direct_setup_capacity: 16,
        proof_bytes: 20,
        setup_field_elements: 31,
    }
    .is_strictly_worse_for_recursive_parent(incumbent));
    assert!(CompleteObjectiveBound::SetupFirst {
        first_direct_setup_capacity: 16,
        proof_bytes: 21,
        setup_field_elements: 0,
    }
    .is_strictly_worse_for_recursive_parent(incumbent));
    assert!(!CompleteObjectiveBound::SetupFirst {
        first_direct_setup_capacity: 16,
        proof_bytes: 20,
        setup_field_elements: usize::MAX,
    }
    .is_strictly_worse_for_recursive_payload(incumbent));
    assert!(CompleteObjectiveBound::SetupFirst {
        first_direct_setup_capacity: 0,
        proof_bytes: 21,
        setup_field_elements: 0,
    }
    .is_strictly_worse_for_recursive_payload(incumbent));

    let padded_envelope_bound = |setup_field_elements, first_direct_setup_capacity, proof_bytes| {
        CompleteObjectiveBound::PaddedSetupEnvelopeFirst {
            setup_envelope_capacity: akita_types::padded_setup_prefix_len(setup_field_elements),
            first_direct_setup_capacity,
            proof_bytes,
            first_direct_output_witness_len: 1_000,
        }
    };
    assert!(padded_envelope_bound(31, 32, 1).is_strictly_worse_than(incumbent));
    assert!(!padded_envelope_bound(31, 8, usize::MAX).is_strictly_worse_than(incumbent));
    assert!(padded_envelope_bound(33, 1, 1).is_strictly_worse_than(incumbent));
    assert!(!padded_envelope_bound(31, 1, 1).setup_envelope_is_strictly_worse_than(incumbent));
    assert!(padded_envelope_bound(33, 1, 1).setup_envelope_is_strictly_worse_than(incumbent));
}

fn complete_candidate(
    proof_bytes: usize,
    setup_field_elements: usize,
    output_witness_len: usize,
) -> super::ScheduleCandidate {
    let challenge = akita_challenges::SparseChallengeConfig::pm1_only(3);
    let mut params = akita_types::CommittedGroupParams::params_only(
        akita_types::SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        4,
        3,
        2,
        challenge,
    )
    .with_decomp(4, 32, 2, 2, 2)
    .expect("candidate parameters");
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix =
        akita_types::sis::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner
                .sis_table_key()
                .expect("L infinity matrix")
                .table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            inner.input_width(),
            4_095,
            inner.ring_dimension(),
        );
    assert_eq!(params.open().digits.log_basis, 3);
    let (terminal_params, linf_cap) =
        akita_types::TerminalFoldParams::try_from_expanded_group(params.clone())
            .expect("terminal parameters");
    let response_shape = akita_types::TerminalResponseShape::derive(&terminal_params, linf_cap)
        .expect("terminal response shape");
    let terminal = akita_schedules::planner_support::CandidateTerminalResponse {
        params: terminal_params,
        sparse_challenge_config: challenge,
        input_witness_len: output_witness_len,
        estimated_direct_payload_bytes: 0,
        response_shape,
        estimated_payload_bytes: 0,
    };
    super::ScheduleCandidate {
        first_direct_setup_field_len: NonZeroUsize::new(1),
        first_direct_output_witness_len: output_witness_len,
        cost: super::super::PackedProofCost::new(proof_bytes, 0).unwrap(),
        setup_field_elements,
        folds: CandidateFoldChain::default().prepend(
            akita_schedules::planner_support::CandidateFoldStep {
                params: Arc::new(params),
                input_witness_len: 256,
                output_witness_len,
                estimated_direct_payload_bytes: proof_bytes,
                estimated_stage3_payload_bytes: 0,
            },
        ),
        terminal: Arc::new(terminal),
    }
}

#[test]
fn actual_policy_can_select_a_noncontractive_complete_candidate() {
    let mut policy = akita_config::policy_of::<akita_config::proof_optimized::fp128::Dense>();
    policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayloadV2;
    let contractive = complete_candidate(101, 64, 1_000);
    let noncontractive = complete_candidate(100, 64, 12_000);
    let input_bits = 256 * policy.decomposition.field_bits() as usize;
    assert!(1_000 * 3 < input_bits);
    assert!(12_000 * 3 >= input_bits);

    for candidates in [
        [&contractive, &noncontractive],
        [&noncontractive, &contractive],
    ] {
        let selected = select_complete_candidate(&policy, candidates, None)
            .expect("complete candidate selection")
            .expect("selected complete candidate");
        assert!(std::ptr::eq(selected, &noncontractive));
    }
}

#[test]
fn exact_proof_tie_selects_the_smaller_setup_envelope() {
    let mut policy = akita_config::policy_of::<akita_config::proof_optimized::fp128::Dense>();
    policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayloadV2;
    let larger_setup = complete_candidate(100, 65, 1_000);
    let smaller_setup = complete_candidate(100, 64, 1_000);

    for candidates in [
        [&larger_setup, &smaller_setup],
        [&smaller_setup, &larger_setup],
    ] {
        let selected = select_complete_candidate(&policy, candidates, None)
            .expect("complete candidate selection")
            .expect("selected complete candidate");
        assert!(std::ptr::eq(selected, &smaller_setup));
    }
}

#[test]
fn exact_numeric_tie_selects_the_smaller_root_output_witness() {
    let mut policy = akita_config::policy_of::<akita_config::proof_optimized::fp128::Dense>();
    policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayloadV2;
    let larger_output = complete_candidate(100, 64, 1_001);
    let smaller_output = complete_candidate(100, 64, 1_000);

    for candidates in [
        [&larger_output, &smaller_output],
        [&smaller_output, &larger_output],
    ] {
        let selected = select_complete_candidate(&policy, candidates, None)
            .expect("complete candidate selection")
            .expect("selected complete candidate");
        assert!(std::ptr::eq(selected, &smaller_output));
    }
}
