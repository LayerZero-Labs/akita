use std::{num::NonZeroUsize, sync::Arc};

use super::{select_complete_candidate, CompleteObjectiveBound, CompleteScheduleScore};
use crate::schedule_params::CandidateFoldChain;

fn score(objective: CompleteObjectiveBound, descriptor: u8) -> CompleteScheduleScore {
    CompleteScheduleScore {
        objective,
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
        proof_bytes: 20,
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
}

fn complete_candidate(proof_bytes: usize, output_witness_len: usize) -> super::ScheduleCandidate {
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
        total_bytes: proof_bytes,
        setup_field_elements: 64,
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
    policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayload;
    let contractive = complete_candidate(101, 1_000);
    let noncontractive = complete_candidate(100, 12_000);
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
