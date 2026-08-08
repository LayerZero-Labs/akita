use super::{dominates_mixed_score, offloaded_witness_contracts};
use crate::schedule_params::MixedScore;

#[test]
fn mixed_frontier_keeps_lower_payload_child_until_parent_masks_setup() {
    let lower_setup = MixedScore {
        setup_field_elements: 10,
        proof_bytes: 20,
    };
    let lower_payload = MixedScore {
        setup_field_elements: 15,
        proof_bytes: 10,
    };
    assert!(!dominates_mixed_score(lower_setup, lower_payload));
    assert!(!dominates_mixed_score(lower_payload, lower_setup));

    let parent_setup = 20;
    let lower_setup_complete = MixedScore {
        setup_field_elements: parent_setup.max(lower_setup.setup_field_elements),
        proof_bytes: lower_setup.proof_bytes,
    };
    let lower_payload_complete = MixedScore {
        setup_field_elements: parent_setup.max(lower_payload.setup_field_elements),
        proof_bytes: lower_payload.proof_bytes,
    };
    assert!(lower_payload_complete < lower_setup_complete);
}

#[test]
fn mixed_frontier_keeps_equal_payload_alternatives_for_descriptor_ties() {
    let lower_setup = MixedScore {
        setup_field_elements: 10,
        proof_bytes: 20,
    };
    let higher_setup = MixedScore {
        setup_field_elements: 15,
        proof_bytes: 20,
    };

    assert!(!dominates_mixed_score(lower_setup, higher_setup));
    assert!(!dominates_mixed_score(higher_setup, lower_setup));

    let parent_setup = 20;
    assert_eq!(
        parent_setup.max(lower_setup.setup_field_elements),
        parent_setup.max(higher_setup.setup_field_elements)
    );
}

#[test]
fn offloaded_contraction_accepts_exact_threefold_boundary() {
    assert!(offloaded_witness_contracts(300, 2, 0, 128, 100, 2, 3).unwrap());
    assert!(!offloaded_witness_contracts(299, 2, 0, 128, 100, 2, 3).unwrap());
    assert!(!offloaded_witness_contracts(300, 2, 0, 128, 100, 2, 4).unwrap());
}

#[test]
fn offloaded_contraction_prices_changed_digit_basis() {
    assert!(offloaded_witness_contracts(900, 2, 0, 128, 100, 6, 3).unwrap());
    assert!(!offloaded_witness_contracts(899, 2, 0, 128, 100, 6, 3).unwrap());
}

#[test]
fn offloaded_contraction_includes_full_field_setup_prefix() {
    assert!(offloaded_witness_contracts(100, 2, 100, 128, 1000, 4, 3).unwrap());
    assert!(!offloaded_witness_contracts(100, 2, 90, 128, 1000, 4, 3).unwrap());
}
