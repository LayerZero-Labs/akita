use super::*;
use crate::layout::digit_math::compute_num_digits_full_field;
use crate::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
    proof_ring_vec_bytes, root_extension_opening_partials, stage1_tree_stage_shapes,
    sumcheck_rounds, terminal_level_proof_bytes, terminal_level_proof_bytes_for_mode,
    AjtaiKeyParams, AkitaBatchedRootProof, AkitaLevelProof, AkitaStage1Proof,
    AkitaStage1StageProof, AkitaStage2Proof, DirectWitnessProof, DirectWitnessShape, FlatRingVec,
    PackedDigits, SisModulusFamily, TerminalLevelProof,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::FieldCore;
use akita_field::Prime128OffsetA7F7;
use akita_serialization::{AkitaSerialize, Compress};
use akita_sumcheck::EqFactoredUniPoly;
use akita_sumcheck::EXTENSION_OPENING_REDUCTION_DEGREE;
#[cfg(not(feature = "zk"))]
use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};
#[cfg(feature = "zk")]
use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProofMasked, SumcheckProofMasked};

use crate::ExtensionOpeningReductionProof;

type F = Prime128OffsetA7F7;

#[test]
fn root_direct_schedule_uses_field_element_payload() {
    let dummy_commit_params = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        },
    );
    let schedule =
        root_direct_schedule(3, dummy_commit_params.clone()).expect("root-direct schedule");
    assert_eq!(schedule.total_bytes, 0);

    let [Step::Direct(step)] = schedule.steps.as_slice() else {
        panic!("root-direct schedule should contain one direct step");
    };
    assert_eq!(step.current_w_len, 8);
    assert_eq!(step.witness_shape, DirectWitnessShape::FieldElements(8));
    assert_eq!(step.direct_bytes, 0);
    assert_eq!(
        step.terminal_proof_mode,
        TerminalProofMode::RingSwitchSumcheck
    );
    assert_eq!(step.commit_params.as_ref(), Some(&dummy_commit_params));
    assert!(step.level_params.is_none());
}

#[test]
fn terminal_direct_witness_ring_count_excludes_r_hat() {
    let lp = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        },
    )
    .with_decomp(2, 3, 2, 2, 3, 0)
    .expect("level params");
    let field_bits = 128;
    let num_points = 2;
    let num_t_vectors = 3;
    let num_w_vectors = 4;
    let num_public_rows = 5;

    let with_r_hat = w_ring_element_count_with_counts_for_layout_bits_and_quotient(
        field_bits,
        &lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        crate::layout::MRowLayout::Terminal,
        TerminalWitnessQuotient::IncludeRHat,
    )
    .expect("sumcheck-terminal witness size");
    let without_r_hat = w_ring_element_count_with_counts_for_layout_bits_and_quotient(
        field_bits,
        &lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        crate::layout::MRowLayout::Terminal,
        TerminalWitnessQuotient::OmitRHat,
    )
    .expect("direct-terminal witness size");
    let r_rows = lp
        .m_row_count_for(
            num_points,
            num_public_rows,
            crate::layout::MRowLayout::Terminal,
        )
        .expect("terminal row count");
    let expected_r_hat = r_rows * compute_num_digits_full_field(field_bits, lp.log_basis);

    assert_eq!(with_r_hat - without_r_hat, expected_r_hat);

    let err = w_ring_element_count_with_counts_for_layout_bits_and_quotient(
        field_bits,
        &lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        crate::layout::MRowLayout::Intermediate,
        TerminalWitnessQuotient::OmitRHat,
    )
    .expect_err("non-terminal layouts must keep r_hat");
    assert!(err
        .to_string()
        .contains("r_hat omission is only valid for terminal layout"));
}

#[cfg(not(feature = "zk"))]
#[test]
fn terminal_direct_witness_ring_count_ignores_omitted_r_hat_geometry() {
    let lp = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        3,
        0,
        usize::MAX,
        0,
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        },
    );

    assert_eq!(
        w_ring_element_count_with_counts_for_layout_bits_and_quotient(
            128,
            &lp,
            2,
            0,
            0,
            0,
            crate::layout::MRowLayout::Terminal,
            TerminalWitnessQuotient::OmitRHat,
        )
        .expect("direct mode omits r_hat rows before sizing them"),
        0,
    );

    let err = w_ring_element_count_with_counts_for_layout_bits_and_quotient(
        128,
        &lp,
        2,
        0,
        0,
        0,
        crate::layout::MRowLayout::Terminal,
        TerminalWitnessQuotient::IncludeRHat,
    )
    .expect_err("sumcheck mode must still size r_hat rows");
    assert!(err.to_string().contains("M-row count overflow"));
}

#[test]
fn terminal_direct_level_bytes_exclude_stage2_sumcheck() {
    let lp = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        },
    );
    let next_w_len = 64 * 8;
    let num_claims = 3;
    let direct = terminal_level_proof_bytes_for_mode(
        128,
        128,
        &lp,
        next_w_len,
        num_claims,
        TerminalProofMode::DirectRingRelations,
    );
    let sumcheck = terminal_level_proof_bytes_for_mode(
        128,
        128,
        &lp,
        next_w_len,
        num_claims,
        TerminalProofMode::RingSwitchSumcheck,
    );
    let y_bytes = proof_ring_vec_bytes(num_claims, lp.ring_dimension, crate::field_bytes(128));
    let terminal_relation_tag_bytes = 1;
    let stage2_bytes = sumcheck_rounds(lp.ring_dimension, next_w_len) * 3 * crate::field_bytes(128);

    assert_eq!(direct, y_bytes + terminal_relation_tag_bytes);
    assert_eq!(sumcheck, direct + stage2_bytes);
}

#[test]
fn terminal_direct_level_bytes_ignore_sumcheck_geometry() {
    let lp = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        1,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        },
    );
    let direct = terminal_level_proof_bytes_for_mode(
        128,
        128,
        &lp,
        usize::MAX,
        3,
        TerminalProofMode::DirectRingRelations,
    );

    assert_eq!(
        direct,
        proof_ring_vec_bytes(3, 1, crate::field_bytes(128)) + 1
    );
}

#[cfg(not(feature = "zk"))]
fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: (0..rounds)
            .map(|_| CompressedUniPoly {
                coeffs_except_linear_term: vec![F::zero(); degree],
            })
            .collect(),
    }
}

#[cfg(feature = "zk")]
fn dummy_sumcheck_proof_masked<F: FieldCore>(
    rounds: usize,
    degree: usize,
) -> SumcheckProofMasked<F> {
    let compressed_rounds = || {
        (0..rounds)
            .map(|_| CompressedUniPoly {
                coeffs_except_linear_term: vec![F::zero(); degree],
            })
            .collect()
    };
    SumcheckProofMasked {
        masked_round_polys: compressed_rounds(),
    }
}

#[cfg(not(feature = "zk"))]
fn dummy_eq_factored_sumcheck<F: FieldCore>(
    rounds: usize,
    degree: usize,
) -> EqFactoredSumcheckProof<F> {
    EqFactoredSumcheckProof {
        round_polys: (0..rounds)
            .map(|_| EqFactoredUniPoly {
                coeffs_except_linear_term: vec![
                    F::zero();
                    EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(degree)
                ],
            })
            .collect(),
    }
}

#[cfg(feature = "zk")]
fn dummy_eq_factored_sumcheck_proof_masked<F: FieldCore>(
    rounds: usize,
    degree: usize,
) -> EqFactoredSumcheckProofMasked<F> {
    let rounds_for = || {
        (0..rounds)
            .map(|_| EqFactoredUniPoly {
                coeffs_except_linear_term: vec![
                    F::zero();
                    EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(degree)
                ],
            })
            .collect()
    };
    EqFactoredSumcheckProofMasked {
        masked_round_polys: rounds_for(),
    }
}

fn dummy_stage1_proof<F: FieldCore>(rounds: usize, b: usize) -> AkitaStage1Proof<F> {
    AkitaStage1Proof {
        stages: stage1_tree_stage_shapes(rounds, b)
            .into_iter()
            .map(|shape| AkitaStage1StageProof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: dummy_eq_factored_sumcheck_proof_masked(
                    rounds,
                    shape.sumcheck_proof.1,
                ),
                child_claims: vec![F::zero(); shape.child_claims],
            })
            .collect(),
        s_claim: F::zero(),
    }
}

fn exact_level_proof_bytes<F: FieldCore + AkitaSerialize>(
    lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
) -> Result<usize, AkitaError> {
    let current_coeffs = lp
        .d_key
        .row_len()
        .checked_mul(lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let next_commit_coeffs = next_lp
        .b_key
        .row_len()
        .checked_mul(next_lp.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let b = 1usize << lp.log_basis;

    let proof = AkitaLevelProof {
        y_ring: FlatRingVec::from_coeffs(vec![F::zero(); lp.ring_dimension]),
        extension_opening_reduction: None,
        v: FlatRingVec::from_coeffs(vec![F::zero(); current_coeffs]),
        stage1: dummy_stage1_proof(rounds, b),
        stage2: AkitaStage2Proof {
            #[cfg(not(feature = "zk"))]
            sumcheck_proof: dummy_sumcheck(rounds, 3),
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: dummy_sumcheck_proof_masked(rounds, 3),
            next_w_commitment: FlatRingVec::from_coeffs(vec![F::zero(); next_commit_coeffs]),
            #[cfg(not(feature = "zk"))]
            next_w_eval: F::zero(),
            #[cfg(feature = "zk")]
            next_w_eval_masked: F::zero(),
        },
    };
    Ok(proof.serialized_size(Compress::No))
}

#[test]
fn generated_schedule_key_preserves_commitment_group_count() {
    let one_group = AkitaScheduleLookupKey::new_with_points(16, 1, 4, 4, 1);
    let four_groups = AkitaScheduleLookupKey::new_with_points(16, 4, 4, 4, 1);

    assert_ne!(
        generated_schedule_lookup_key(one_group),
        generated_schedule_lookup_key(four_groups),
        "generated schedule lookup must not alias differently grouped commitment shapes"
    );
}

#[test]
fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
    const D: usize = 64;
    let stage1_config = SparseChallengeConfig::Uniform {
        weight: 3,
        nonzero_coeffs: vec![-1, 1],
    };
    let next_lp =
        LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, stage1_config.clone());
    let next_w_len = D * 8;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            log_basis,
            2,
            2,
            2,
            stage1_config.clone(),
        )
        .with_decomp(0, 0, 1, 1, 1, 0)
        .unwrap();
        assert_eq!(
            level_proof_bytes(128, 128, &lp, &lp, &next_lp, next_w_len, 1),
            exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len).unwrap(),
            "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
        );
    }
}

#[test]
fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
    const D: usize = 64;
    let stage1_config = SparseChallengeConfig::Uniform {
        weight: 3,
        nonzero_coeffs: vec![-1, 1],
    };
    let next_w_len = D * 8;
    let num_claims = 3;
    let final_w_num_elems = 1024;
    let final_w_bits = 5;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            log_basis,
            2,
            2,
            2,
            stage1_config.clone(),
        )
        .with_decomp(0, 0, 1, 1, 1, 0)
        .unwrap();
        let rounds = sumcheck_rounds(D, next_w_len);

        let final_witness = DirectWitnessProof::PackedDigits(PackedDigits::from_i8_digits(
            &vec![0i8; final_w_num_elems],
            final_w_bits,
        ));
        let final_witness_bytes_runtime = final_witness.serialized_size(Compress::No);
        let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
            vec![CyclotomicRing::<F, D>::zero(); num_claims],
            None,
            #[cfg(not(feature = "zk"))]
            dummy_sumcheck(rounds, 3),
            #[cfg(feature = "zk")]
            dummy_sumcheck_proof_masked(rounds, 3),
            final_witness,
        );

        // The planner accounts for the final witness separately
        // (`direct_witness_bytes` on the terminal direct step). Subtract
        // it from the serialized terminal level to compare against
        // `terminal_level_proof_bytes`.
        let serialized_without_witness =
            terminal_proof.serialized_size(Compress::No) - final_witness_bytes_runtime;

        assert_eq!(
            terminal_level_proof_bytes(128, 128, &lp, next_w_len, num_claims),
            serialized_without_witness,
            "planned terminal-level bytes should match the serialized terminal body \
             (less final_witness) at log_basis={log_basis}"
        );

        // Sanity-check `direct_witness_bytes` against the runtime
        // packed-digit serialization so any future drift in either
        // accounting path is caught here too.
        assert_eq!(
            direct_witness_bytes(
                128,
                &DirectWitnessShape::PackedDigits((final_w_num_elems, final_w_bits))
            ),
            final_witness_bytes_runtime,
            "direct_witness_bytes should match the serialized packed-digit \
             final witness at log_basis={log_basis}"
        );
    }
}

#[cfg(not(feature = "zk"))]
#[test]
fn planned_direct_terminal_level_bytes_match_terminal_payload_at_all_bases() {
    const D: usize = 64;
    let stage1_config = SparseChallengeConfig::Uniform {
        weight: 3,
        nonzero_coeffs: vec![-1, 1],
    };
    let next_w_len = D * 8;
    let num_claims = 3;
    let final_w_num_elems = 896;
    let final_w_bits = 5;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            log_basis,
            2,
            2,
            2,
            stage1_config.clone(),
        )
        .with_decomp(0, 0, 1, 1, 1, 0)
        .unwrap();

        let final_witness = DirectWitnessProof::PackedDigits(PackedDigits::from_i8_digits(
            &vec![0i8; final_w_num_elems],
            final_w_bits,
        ));
        let final_witness_bytes_runtime = final_witness.serialized_size(Compress::No);
        let terminal_proof =
            TerminalLevelProof::<F, F>::new_direct_with_extension_opening_reduction(
                vec![CyclotomicRing::<F, D>::zero(); num_claims],
                None,
                final_witness,
            );

        let serialized_without_witness =
            terminal_proof.serialized_size(Compress::No) - final_witness_bytes_runtime;

        assert_eq!(
            terminal_level_proof_bytes_for_mode(
                128,
                128,
                &lp,
                next_w_len,
                num_claims,
                TerminalProofMode::DirectRingRelations,
            ),
            serialized_without_witness,
            "planned direct terminal-level bytes should match the serialized terminal body \
             (less final_witness) at log_basis={log_basis}"
        );
    }
}

#[test]
fn planned_batched_root_bytes_match_two_stage_payload_at_all_bases() {
    const D: usize = 64;
    let stage1_config = SparseChallengeConfig::Uniform {
        weight: 3,
        nonzero_coeffs: vec![-1, 1],
    };
    let next_lp =
        LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, stage1_config.clone());
    let next_w_len = D * 8;

    for log_basis in 2..=6 {
        let lp = LevelParams {
            ring_dimension: D,
            log_basis,
            a_key: AjtaiKeyParams::new(SisModulusFamily::Q128, 2, 1, 0, D),
            b_key: AjtaiKeyParams::new(SisModulusFamily::Q128, 2, 1, 0, D),
            d_key: AjtaiKeyParams::new(SisModulusFamily::Q128, 2, 1, 0, D),
            num_blocks: 1,
            block_len: 1,
            m_vars: 0,
            r_vars: 0,
            stage1_config: stage1_config.clone(),
            fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
            num_digits_commit: 1,
            num_digits_open: 1,
            num_digits_fold: 1,
        };
        let rounds = sumcheck_rounds(D, next_w_len);
        let b = 1usize << log_basis;
        let next_commitment = FlatRingVec::from_ring_elems(&vec![
            CyclotomicRing::<F, D>::zero();
            next_lp.b_key.row_len()
        ])
        .into_compact();
        let num_points = 5;
        let root_proof = AkitaBatchedRootProof::new_two_stage::<D>(
            vec![CyclotomicRing::<F, D>::zero(); num_points],
            vec![CyclotomicRing::<F, D>::zero(); lp.d_key.row_len()],
            dummy_stage1_proof(rounds, b),
            #[cfg(not(feature = "zk"))]
            dummy_sumcheck(rounds, 3),
            #[cfg(feature = "zk")]
            dummy_sumcheck_proof_masked(rounds, 3),
            next_commitment,
            F::zero(),
        );

        assert_eq!(
            level_proof_bytes(128, 128, &lp, &lp, &next_lp, next_w_len, num_points),
            root_proof.serialized_size(Compress::No),
            "planned batched root bytes should match the serialized two-stage body at log_basis={log_basis}"
        );
    }
}

#[test]
fn planned_root_extension_reduction_bytes_match_payload() {
    let extension_width = 4;
    let num_claims = 3;
    let opening_vars = 12;
    let partials = root_extension_opening_partials(extension_width, num_claims);
    let reduction = ExtensionOpeningReductionProof {
        partials: vec![F::zero(); partials],
        #[cfg(not(feature = "zk"))]
        sumcheck: dummy_sumcheck(
            opening_vars - extension_width.trailing_zeros() as usize,
            EXTENSION_OPENING_REDUCTION_DEGREE,
        ),
        #[cfg(feature = "zk")]
        sumcheck_proof_masked: dummy_sumcheck_proof_masked(
            opening_vars - extension_width.trailing_zeros() as usize,
            EXTENSION_OPENING_REDUCTION_DEGREE,
        ),
    };
    #[cfg(not(feature = "zk"))]
    let sumcheck_bytes = reduction.sumcheck.serialized_size(Compress::No);
    #[cfg(feature = "zk")]
    let sumcheck_bytes = reduction
        .sumcheck_proof_masked
        .serialized_size(Compress::No);

    assert_eq!(
        extension_opening_reduction_proof_bytes(128, partials, opening_vars, extension_width)
            .unwrap(),
        reduction
            .partials
            .iter()
            .map(|partial| partial.serialized_size(Compress::No))
            .sum::<usize>()
            + sumcheck_bytes,
        "planned root EOR bytes should match the headerless serialized payload"
    );
}
