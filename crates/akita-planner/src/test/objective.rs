use super::CompleteScheduleScore;

#[test]
fn direct_score_prefers_setup_only_after_proof_ties() {
    let smaller_proof = CompleteScheduleScore::Direct {
        proof_bytes: 99,
        setup_field_elements: 1_000,
        descriptor: vec![2],
    };
    let smaller_setup = CompleteScheduleScore::Direct {
        proof_bytes: 100,
        setup_field_elements: 1,
        descriptor: vec![1],
    };
    assert!(smaller_proof < smaller_setup);

    let same_proof_smaller_setup = CompleteScheduleScore::Direct {
        proof_bytes: 99,
        setup_field_elements: 999,
        descriptor: vec![3],
    };
    assert!(same_proof_smaller_setup < smaller_proof);

    let complete_tie_smaller_descriptor = CompleteScheduleScore::Direct {
        proof_bytes: 99,
        setup_field_elements: 999,
        descriptor: vec![1],
    };
    assert!(complete_tie_smaller_descriptor < same_proof_smaller_setup);
}

#[test]
fn direct_next_witness_score_prefers_payload_only_after_witness_ties() {
    let smaller_witness = CompleteScheduleScore::DirectNextWitness {
        next_witness_len: 99,
        proof_bytes: 1_000,
        setup_field_elements: 1_000,
        descriptor: vec![2],
    };
    let smaller_payload = CompleteScheduleScore::DirectNextWitness {
        next_witness_len: 100,
        proof_bytes: 1,
        setup_field_elements: 1,
        descriptor: vec![1],
    };
    assert!(smaller_witness < smaller_payload);

    let same_witness_smaller_payload = CompleteScheduleScore::DirectNextWitness {
        next_witness_len: 99,
        proof_bytes: 999,
        setup_field_elements: 1_000,
        descriptor: vec![3],
    };
    assert!(same_witness_smaller_payload < smaller_witness);
}

#[test]
fn mixed_dimension_score_prefers_proof_only_after_setup_ties() {
    let smaller_setup = CompleteScheduleScore::MixedDimension {
        setup_field_elements: 99,
        proof_bytes: 1_000,
        descriptor: vec![2],
    };
    let smaller_proof = CompleteScheduleScore::MixedDimension {
        setup_field_elements: 100,
        proof_bytes: 1,
        descriptor: vec![1],
    };
    assert!(smaller_setup < smaller_proof);

    let same_setup_smaller_proof = CompleteScheduleScore::MixedDimension {
        setup_field_elements: 99,
        proof_bytes: 999,
        descriptor: vec![3],
    };
    assert!(same_setup_smaller_proof < smaller_setup);
}

#[test]
fn recursive_score_uses_total_setup_only_after_primary_coordinates() {
    let smaller_proof = CompleteScheduleScore::RecursiveSetup {
        first_direct_setup_padded_field_len: 10,
        next_witness_len: 20,
        proof_bytes: 99,
        setup_field_elements: 1_000,
        descriptor: vec![2],
    };
    let smaller_total_setup = CompleteScheduleScore::RecursiveSetup {
        first_direct_setup_padded_field_len: 10,
        next_witness_len: 20,
        proof_bytes: 100,
        setup_field_elements: 1,
        descriptor: vec![1],
    };
    assert!(smaller_proof < smaller_total_setup);

    let same_proof_smaller_total_setup = CompleteScheduleScore::RecursiveSetup {
        first_direct_setup_padded_field_len: 10,
        next_witness_len: 20,
        proof_bytes: 99,
        setup_field_elements: 999,
        descriptor: vec![3],
    };
    assert!(same_proof_smaller_total_setup < smaller_proof);
}
