use super::*;
use akita_algebra::offset_eq::summarize_pow2_block_carries;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::{
    OpeningBatchWitnessGroup, OpeningBatchWitnessLayout, RingMultiplierOpeningPoint,
    RingOpeningPoint, SemanticGroupId, SisModulusFamily,
};

/// Placeholder layout for prepare-path rejection tests. These cases fail before
/// any layout-dependent evaluation runs, so the offsets are not required to
/// match the witness column geometry.
fn reject_test_segment_layout() -> OpeningBatchWitnessLayout {
    OpeningBatchWitnessLayout::new(
        vec![OpeningBatchWitnessGroup {
            id: SemanticGroupId(0),
            num_claims: 1,
            num_blocks: 1,
            block_len: 1,
            depth_open: 1,
            depth_commit: 1,
            depth_fold: 1,
            n_a: 1,
            e_setup_col_offset: 0,
        }],
        vec![SemanticGroupId(0)],
        vec![SemanticGroupId(0)],
        1,
        1,
        1,
    )
    .expect("reject test layout")
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
    let err = match prepare_relation_matrix_evaluator_inner::<F, F, D>(
        &challenges,
        &reject_test_multiplier_point(),
        F::one(),
        &lp,
        &[],
        0,
        &[],
        RelationMatrixRowLayout::WithDBlock,
        reject_test_segment_layout(),
        OpeningBlockLayout::new(1, reject_test_segment_layout().total_len()).unwrap(),
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
    let err = match prepare_relation_matrix_evaluator_inner::<F, F, D>(
        &challenges,
        &reject_test_multiplier_point(),
        F::one(),
        &lp,
        &[],
        0,
        &[],
        RelationMatrixRowLayout::WithDBlock,
        reject_test_segment_layout(),
        OpeningBlockLayout::new(1, reject_test_segment_layout().total_len()).unwrap(),
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
