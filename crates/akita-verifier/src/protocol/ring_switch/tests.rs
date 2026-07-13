use super::*;
use akita_field::Fp32;

type F = Fp32<251>;

#[test]
fn ring_switch_prepare_rejects_invalid_log_basis() {
    let err = validate_log_basis(0).expect_err("invalid log_basis should be rejected");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn pow2_block_summary_rejects_malformed_shapes() {
    let eq_low = vec![F::one(); 2];

    let err = summarize_pow2_block_carries(&eq_low, 0, &[F::one(); 3]).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidInput(_)));

    let err = summarize_pow2_block_carries(&eq_low, 2, &[F::one(); 2]).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidInput(_)));

    let err = summarize_pow2_block_carries(&eq_low[..1], 0, &[F::one(); 2]).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidSize { .. }));
}
