use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::{
    RingMultiplierOpeningPoint, RingOpeningPoint, SisModulusFamily, WitnessChunkLayout,
    WitnessChunkLengths, WitnessLayout,
};

/// Placeholder layout for prepare-path rejection tests. These cases fail before
/// any layout-dependent evaluation runs, so the offsets are not required to
/// match the witness column geometry.
fn reject_test_segment_layout() -> WitnessLayout {
    WitnessLayout {
        blocks_per_chunk: 1,
        chunks: vec![WitnessChunkLayout {
            offset_z: 0,
            offset_e: 0,
            offset_t: 0,
            offset_r: Some(0),
            global_block_base: 0,
        }],
        chunk_lengths: vec![WitnessChunkLengths {
            z_len: 0,
            e_len: 0,
            t_len: 0,
            r_len: Some(0),
        }],
    }
}

type F = Fp32<251>;
const D: usize = 32;

fn fold_challenge_config() -> SparseChallengeConfig {
    SparseChallengeConfig::pm1_only(1)
}

fn reject_test_multiplier_point() -> RingMultiplierOpeningPoint<F> {
    RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
        a: Vec::new(),
        b: Vec::new(),
    })
}

#[test]
fn ring_switch_prepare_rejects_invalid_log_basis() {
    let lp = LevelParams::params_only(
        SisModulusFamily::Q32,
        D,
        0,
        1,
        1,
        1,
        fold_challenge_config(),
    );
    let challenges = Challenges::from_sparse(Vec::new(), 0, 0).unwrap();
    let err = match prepare_ring_switch_row_eval_inner::<F, F, D>(
        &challenges,
        &reject_test_multiplier_point(),
        F::one(),
        &lp,
        &[],
        0,
        &[],
        MRowLayout::WithDBlock,
        reject_test_segment_layout(),
        1,
        0,
    ) {
        Ok(_) => panic!("invalid log_basis should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_switch_prepare_rejects_zero_num_blocks() {
    let lp = LevelParams::params_only(
        SisModulusFamily::Q32,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    );
    let challenges = Challenges::from_sparse(Vec::new(), 0, 0).unwrap();
    let err = match prepare_ring_switch_row_eval_inner::<F, F, D>(
        &challenges,
        &reject_test_multiplier_point(),
        F::one(),
        &lp,
        &[],
        0,
        &[],
        MRowLayout::WithDBlock,
        reject_test_segment_layout(),
        1,
        0,
    ) {
        Ok(_) => panic!("zero num_blocks should be rejected"),
        Err(err) => err,
    };
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
