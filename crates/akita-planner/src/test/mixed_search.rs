use super::{dominates_score, MixedScore};

#[test]
fn frontier_keeps_lower_payload_child_until_parent_masks_setup() {
    let lower_setup = MixedScore {
        setup_field_elements: 10,
        proof_bytes: 20,
    };
    let lower_payload = MixedScore {
        setup_field_elements: 15,
        proof_bytes: 10,
    };
    assert!(!dominates_score(lower_setup, lower_payload));
    assert!(!dominates_score(lower_payload, lower_setup));

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
fn frontier_keeps_equal_payload_alternatives_for_descriptor_ties() {
    let lower_setup = MixedScore {
        setup_field_elements: 10,
        proof_bytes: 20,
    };
    let higher_setup = MixedScore {
        setup_field_elements: 15,
        proof_bytes: 20,
    };

    assert!(!dominates_score(lower_setup, higher_setup));
    assert!(!dominates_score(higher_setup, lower_setup));

    let parent_setup = 20;
    assert_eq!(
        parent_setup.max(lower_setup.setup_field_elements),
        parent_setup.max(higher_setup.setup_field_elements)
    );
}
