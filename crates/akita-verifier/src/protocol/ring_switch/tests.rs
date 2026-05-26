use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::SisModulusFamily;

type F = Fp32<251>;
const D: usize = 32;

fn stage1_config() -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![1],
    }
}

#[test]
fn ring_switch_prepare_rejects_invalid_log_basis() {
    let lp = LevelParams::params_only(SisModulusFamily::Q32, D, 0, 1, 1, 1, stage1_config());
    let challenges = Challenges::from_sparse(Vec::new(), 0, 0).unwrap();
    let err = match prepare_ring_switch_row_eval::<F, F, D>(
        &challenges,
        F::one(),
        &lp,
        &[],
        &[],
        &[],
        &[],
        &[],
        1,
        MRowLayout::Intermediate,
        0,
        &[],
        &[],
    ) {
        Ok(_) => panic!("invalid log_basis should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_switch_prepare_rejects_zero_num_blocks() {
    let lp = LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config());
    let challenges = Challenges::from_sparse(Vec::new(), 0, 0).unwrap();
    let err = match prepare_ring_switch_row_eval::<F, F, D>(
        &challenges,
        F::one(),
        &lp,
        &[],
        &[],
        &[],
        &[],
        &[],
        1,
        MRowLayout::Intermediate,
        0,
        &[],
        &[],
    ) {
        Ok(_) => panic!("zero num_blocks should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn multiplier_block_summary_rejects_malformed_shapes() {
    let eq_low = vec![F::one(); 2];

    let err = summarize_pow2_multiplier_block_carries(&eq_low, 0, 3, |_| Ok(F::one())).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidInput(_)));

    let err = summarize_pow2_multiplier_block_carries(&eq_low, 2, 2, |_| Ok(F::one())).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidInput(_)));

    let err =
        summarize_pow2_multiplier_block_carries(&eq_low[..1], 0, 2, |_| Ok(F::one())).unwrap_err();
    assert!(matches!(err, AkitaError::InvalidSize { .. }));
}
