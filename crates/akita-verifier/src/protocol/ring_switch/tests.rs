use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::{RingRelationSegmentLayout, SisModulusFamily};

fn dummy_witness_segment_layout() -> RingRelationSegmentLayout {
    RingRelationSegmentLayout {
        offset_e: 0,
        offset_t: 0,
        offset_u: 0,
        offset_z: 0,
        offset_r: 0,
        #[cfg(feature = "zk")]
        b_blinding_offset: 0,
        #[cfg(feature = "zk")]
        d_blinding_offset: 0,
    }
}

type F = Fp32<251>;
const D: usize = 32;

fn stage1_config() -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![1],
    }
}

#[test]
fn ring_switch_prepare_rejects_malformed_level_params() {
    for lp in [
        LevelParams::params_only(SisModulusFamily::Q32, D, 0, 1, 1, 1, stage1_config()),
        LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config()),
    ] {
        let err = malformed_level_params_prepare_err(&lp);
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}

fn malformed_level_params_prepare_err(lp: &LevelParams) -> AkitaError {
    let challenges = Challenges::from_sparse(Vec::new(), 0, 0).unwrap();
    match prepare_ring_switch_row_eval_inner::<F, F, D>(
        &challenges,
        F::one(),
        lp,
        &[],
        &[],
        &[],
        &[],
        &[],
        1,
        MRowLayout::WithDBlock,
        0,
        &[],
        &[],
        dummy_witness_segment_layout(),
    ) {
        Ok(_) => panic!("malformed LevelParams should be rejected"),
        Err(err) => err,
    }
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
