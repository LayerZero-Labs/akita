use super::MixedScore;

#[test]
fn exact_setup_fields_precede_proof_bytes() {
    let generation_dimension = 256;
    let smaller_setup = MixedScore {
        setup_field_elements: generation_dimension + 1,
        proof_bytes: 10_000,
    };
    let larger_setup = MixedScore {
        setup_field_elements: 2 * generation_dimension - 1,
        proof_bytes: 1,
    };

    assert_eq!(
        smaller_setup
            .setup_field_elements
            .div_ceil(generation_dimension),
        larger_setup
            .setup_field_elements
            .div_ceil(generation_dimension)
    );
    assert!(smaller_setup < larger_setup);
}
