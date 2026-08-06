use super::offloaded_witness_contracts;

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
