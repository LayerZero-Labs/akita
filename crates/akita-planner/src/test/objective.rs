use super::{CompleteObjectiveBound, CompleteScheduleScore};

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

fn mixed(setup_field_elements: usize, proof_bytes: usize, descriptor: u8) -> CompleteScheduleScore {
    score(
        CompleteObjectiveBound::MixedDimension {
            setup_field_elements,
            proof_bytes,
        },
        descriptor,
    )
}

fn recursive(
    first_direct_setup_capacity: usize,
    proof_bytes: usize,
    setup_field_elements: usize,
    descriptor: u8,
) -> CompleteScheduleScore {
    score(
        CompleteObjectiveBound::RecursiveSetup {
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
fn mixed_dimension_score_prefers_proof_only_after_setup_ties() {
    let smaller_setup = mixed(99, 1_000, 2);
    let smaller_proof = mixed(100, 1, 1);
    assert!(smaller_setup < smaller_proof);

    let same_setup_smaller_proof = mixed(99, 999, 3);
    assert!(same_setup_smaller_proof < smaller_setup);
}

#[test]
fn recursive_score_uses_total_setup_only_after_primary_coordinates() {
    let smaller_proof = recursive(16, 99, 1_000, 2);
    let smaller_total_setup = recursive(16, 100, 1, 1);
    assert!(smaller_proof < smaller_total_setup);

    let same_proof_smaller_total_setup = recursive(16, 99, 999, 3);
    assert!(same_proof_smaller_total_setup < smaller_proof);
}

#[test]
fn recursive_score_compares_padded_capacity_not_natural_length() {
    let natural_9 = super::super::SetupPrefixCapacity::for_natural_len(9);
    let natural_15 = super::super::SetupPrefixCapacity::for_natural_len(15);
    assert_eq!(natural_9, natural_15);

    let better_proof_with_larger_natural_length =
        recursive(natural_15.field_elements(), 99, 1_000, 2);
    let worse_proof_with_smaller_natural_length = recursive(natural_9.field_elements(), 100, 1, 1);
    assert!(better_proof_with_larger_natural_length < worse_proof_with_smaller_natural_length);
}

#[test]
fn objective_bounds_prune_only_strict_numeric_losses() {
    let incumbent = super::super::CandidateMetrics {
        first_direct_setup_capacity: super::super::SetupPrefixCapacity::for_natural_len(10),
        proof_bytes: 20,
        setup_field_elements: 30,
    };
    assert!(CompleteObjectiveBound::MixedDimension {
        setup_field_elements: 31,
        proof_bytes: 0,
    }
    .is_strictly_worse_than(incumbent));
    assert!(!CompleteObjectiveBound::MixedDimension {
        setup_field_elements: 30,
        proof_bytes: 20,
    }
    .is_strictly_worse_than(incumbent));
    assert!(CompleteObjectiveBound::RecursiveSetup {
        first_direct_setup_capacity: 16,
        proof_bytes: 21,
        setup_field_elements: 0,
    }
    .is_strictly_worse_than(incumbent));
    assert!(!CompleteObjectiveBound::RecursiveSetup {
        first_direct_setup_capacity: 8,
        proof_bytes: usize::MAX,
        setup_field_elements: usize::MAX,
    }
    .is_strictly_worse_than(incumbent));
    assert!(!CompleteObjectiveBound::RecursiveSetup {
        first_direct_setup_capacity: 16,
        proof_bytes: 20,
        setup_field_elements: 31,
    }
    .is_strictly_worse_for_recursive_parent(incumbent));
    assert!(CompleteObjectiveBound::RecursiveSetup {
        first_direct_setup_capacity: 16,
        proof_bytes: 21,
        setup_field_elements: 0,
    }
    .is_strictly_worse_for_recursive_parent(incumbent));
}
