use super::*;
#[cfg(not(feature = "zk"))]
use akita_challenges::SparseChallenge;
use akita_challenges::SparseChallengeConfig;
use akita_field::Fp32;
use akita_types::{RingRelationSegmentLayout, SisModulusFamily};

fn dummy_witness_segment_layout() -> RingRelationSegmentLayout {
    RingRelationSegmentLayout {
        offset_e: 0,
        offset_t: 0,
        offset_z: 0,
        offset_r: 0,
        #[cfg(feature = "zk")]
        b_blinding_offset: 0,
        #[cfg(feature = "zk")]
        d_blinding_offset: 0,
    }
}
#[cfg(not(feature = "zk"))]
use akita_types::{
    AkitaSetupSeed, CleartextWitnessProof, FlatDigitBlocks, FlatMatrix, PackedDigits,
};

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
    let err = match prepare_ring_switch_row_eval_inner::<F, F, D>(
        &challenges,
        F::one(),
        &lp,
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
        Ok(_) => panic!("invalid log_basis should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_switch_prepare_rejects_zero_num_blocks() {
    let lp = LevelParams::params_only(SisModulusFamily::Q32, D, 2, 1, 1, 1, stage1_config());
    let challenges = Challenges::from_sparse(Vec::new(), 0, 0).unwrap();
    let err = match prepare_ring_switch_row_eval_inner::<F, F, D>(
        &challenges,
        F::one(),
        &lp,
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

#[cfg(not(feature = "zk"))]
mod terminal_direct {
    use super::super::terminal_direct::verify_terminal_direct_relation_rows;
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_prover::protocol::ring_switch::build_terminal_direct_w_coeffs;

    const SMALL_D: usize = 2;

    fn small_params() -> LevelParams {
        LevelParams::params_only(SisModulusFamily::Q32, SMALL_D, 2, 1, 1, 0, stage1_config())
            .with_decomp(0, 0, 1, 1, 1)
            .unwrap()
    }

    fn one() -> CyclotomicRing<F, SMALL_D> {
        CyclotomicRing::one()
    }

    fn zero_blocks() -> FlatDigitBlocks<SMALL_D> {
        FlatDigitBlocks::from_blocks(vec![vec![[0i8; SMALL_D]]])
    }

    fn setup() -> AkitaExpandedSetup<F> {
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupSeed {
                max_num_vars: 1,
                max_num_batched_polys: 1,
                max_num_points: 1,
                gen_ring_dim: SMALL_D,
                max_setup_len: 1,
                public_matrix_seed: [0u8; 32],
            },
            FlatMatrix::from_ring_slice::<SMALL_D>(&[one()]),
        )
    }

    fn challenge() -> Challenges {
        Challenges::from_sparse(
            vec![SparseChallenge {
                positions: vec![0],
                coeffs: vec![1],
            }],
            1,
            1,
        )
        .unwrap()
    }

    fn witness(digits: &[i8]) -> CleartextWitnessProof<F> {
        CleartextWitnessProof::PackedDigits(PackedDigits::from_i8_digits(digits, 2))
    }

    fn opening_point() -> RingOpeningPoint<F> {
        RingOpeningPoint {
            a: vec![F::one()],
            b: vec![F::one()],
        }
    }

    fn verify_with(
        digits: &[i8],
        commitment_rows: &[CyclotomicRing<F, SMALL_D>],
        y_rings: &[CyclotomicRing<F, SMALL_D>],
    ) -> Result<(), AkitaError> {
        let lp = small_params();
        let setup = setup();
        let final_witness = witness(digits);
        let opening = opening_point();
        let multiplier = RingMultiplierOpeningPoint::from_base(&opening);
        verify_terminal_direct_relation_rows::<F, F, SMALL_D>(
            &[opening],
            &[multiplier],
            &[0],
            &challenge(),
            digits.len(),
            &final_witness,
            &setup,
            &lp,
            &[1],
            &[0],
            &[0],
            &[F::one()],
            commitment_rows,
            y_rings,
            1,
        )
    }

    fn valid_zero_witness_digits() -> Vec<i8> {
        let lp = small_params();
        build_terminal_direct_w_coeffs::<F, SMALL_D>(
            &zero_blocks(),
            &zero_blocks(),
            &[[0i32; SMALL_D]],
            &lp,
            1,
        )
        .as_i8_digits()
        .to_vec()
    }

    #[test]
    fn terminal_direct_relation_rows_accept_reduced_identity_relation() {
        let digits = valid_zero_witness_digits();
        let zero = CyclotomicRing::zero();
        verify_with(&digits, &[zero], &[zero]).unwrap();
    }

    #[test]
    fn terminal_direct_relation_rows_reject_public_row_tamper() {
        let digits = valid_zero_witness_digits();
        let err = verify_with(
            &digits,
            &[CyclotomicRing::one()],
            &[CyclotomicRing::zero()],
        )
        .unwrap_err();
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    #[test]
    fn terminal_direct_relation_rows_reject_r_tail_shape() {
        let mut digits = valid_zero_witness_digits();
        digits.extend_from_slice(&[0, 0]);
        let zero = CyclotomicRing::zero();
        let err = verify_with(&digits, &[zero], &[zero]).unwrap_err();
        assert!(matches!(err, AkitaError::InvalidProof));
    }
}
