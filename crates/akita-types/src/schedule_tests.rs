use super::*;
use crate::golomb_rice::golomb_rice_encode_vec;
use crate::tail_golomb_rice_z_params;
use crate::{
    extension_opening_reduction_proof_bytes, level_proof_bytes, segment_typed_witness_bytes,
    stage1_tree_stage_shapes, sumcheck_rounds, AkitaStage1Proof, AkitaStage1StageProof,
    AkitaStage2Proof, ExtensionOpeningReductionProof, FoldLevelProof, NextWitnessBinding,
    RelationMatrixRowLayout, RingVec, SegmentTypedWitness, SegmentTypedWitnessShape,
    SisModulusProfileId, TerminalLevelProof, EXTENSION_OPENING_REDUCTION_DEGREE,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore, Prime128OffsetA7F7};
use akita_serialization::{AkitaSerialize, Compress};
use akita_sumcheck::EqFactoredUniPoly;
use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};

type F = Prime128OffsetA7F7;

#[test]
fn chunked_witness_count_matches_chunk_layout_arithmetic() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    // num_live_blocks = 2^3 = 8, divisible by {1, 2, 4, 8}.
    let lp = LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(4, 32, 2, 2, 2)
    .unwrap();
    let field_bits = 128u32;
    let num_poly = 3usize;

    let single =
        intermediate_w_ring_element_count_with_counts_bits(field_bits, &lp, num_poly, 1).unwrap();
    // num_chunks = 1 must be byte-identical to the single-chunk delegate.
    assert_eq!(
        intermediate_w_ring_element_count_for_chunks(field_bits, &lp, num_poly, 1).unwrap(),
        single
    );

    let z_pre = lp.inner_width() * lp.num_digits_fold(num_poly, field_bits).unwrap();
    for num_chunks in [2usize, 4, 8] {
        let chunked =
            intermediate_w_ring_element_count_for_chunks(field_bits, &lp, num_poly, num_chunks)
                .unwrap();
        // ê/t̂ totals are unchanged (partitioned), and the shared r-tail is
        // a single summed quotient that keeps the single-machine row count
        // (num_commitments = 1). So the ONLY growth is the replicated ẑ:
        // (num_chunks - 1) full-width copies.
        assert_eq!(chunked, single + (num_chunks - 1) * z_pre);
        assert!(chunked > single, "chunked layout must grow vs single chunk");
    }
}

#[test]
fn chunked_witness_count_rejects_invalid_chunk_counts() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    // num_live_blocks = 2^3 = 8.
    let lp = LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(4, 32, 2, 2, 2)
    .unwrap();
    // Non-power-of-two chunk count.
    assert!(matches!(
        intermediate_w_ring_element_count_for_chunks(128, &lp, 1, 6),
        Err(AkitaError::InvalidSetup(_))
    ));
    // num_chunks does not divide num_live_blocks (8 % 16 != 0).
    assert!(matches!(
        intermediate_w_ring_element_count_for_chunks(128, &lp, 1, 16),
        Err(AkitaError::InvalidSetup(_))
    ));
    // Zero chunks.
    assert!(matches!(
        intermediate_w_ring_element_count_for_chunks(128, &lp, 1, 0),
        Err(AkitaError::InvalidSetup(_))
    ));
}

fn segment_typed_final_witness(
    lp: &LevelParams,
    num_claims: usize,
) -> (SegmentTypedWitness<F>, SegmentTypedWitnessShape) {
    let field_bits = F::modulus_bits();
    let shape = SegmentTypedWitnessShape::from_groups(
        lp,
        field_bits,
        [(lp as &dyn crate::LevelParamsLike, num_claims, num_claims, 1)],
    )
    .expect("segment-typed witness shape");
    let layout = shape.layout.clone();
    let group = layout.groups[0];
    let (rice_low_bits, zigzag_w) =
        tail_golomb_rice_z_params(lp, num_claims).expect("golomb z params");
    let z_payload = golomb_rice_encode_vec(&vec![0i64; group.z_coords], rice_low_bits, zigzag_w)
        .expect("encode zero z segment");
    let witness = SegmentTypedWitness {
        layout: layout.clone(),
        z_payloads: vec![z_payload],
        e_fields: RingVec::from_coeffs(vec![F::zero(); group.e_field_elems]),
        t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
    };
    (witness, shape)
}

fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: (0..rounds)
            .map(|_| CompressedUniPoly {
                coeffs_except_linear_term: vec![F::zero(); degree],
            })
            .collect(),
    }
}

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

fn dummy_stage1_proof<F: FieldCore>(rounds: usize, b: usize) -> AkitaStage1Proof<F> {
    AkitaStage1Proof {
        stages: stage1_tree_stage_shapes(rounds, b)
            .into_iter()
            .map(|shape| AkitaStage1StageProof {
                sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                child_claims: vec![F::zero(); shape.child_claims],
            })
            .collect(),
        s_claim: F::zero(),
    }
}

fn exact_level_proof_bytes<F: FieldCore + CanonicalField + AkitaSerialize>(
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
    let b = 1usize << lp.log_basis_open;

    let proof = FoldLevelProof {
        extension_opening_reduction: None,
        v: RingVec::from_coeffs(vec![F::zero(); current_coeffs]),
        fold_grind_nonce: 0,
        stage1: dummy_stage1_proof(rounds, b),
        stage2: AkitaStage2Proof {
            sumcheck_proof: dummy_sumcheck(rounds, 3),
            next_witness_binding: NextWitnessBinding::OuterCommitment(RingVec::from_coeffs(vec![
                F::zero();
                next_commit_coeffs
            ])),
            next_w_eval: F::zero(),
        },
        stage3_sumcheck_proof: None,
    };
    Ok(proof.serialized_size(Compress::No))
}

#[test]
fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let next_lp = LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        2,
        3,
        2,
        fold_challenge_config,
    );
    let next_w_len = D * 8;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    RelationMatrixRowLayout::WithDBlock,
                    Some(crate::NextWitnessBindingPolicy::OuterCommitment),
                )
                .unwrap(),
                exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len).unwrap(),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
    }
}

#[test]
fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let next_w_len = D * 8;
    let num_claims = 3;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();

        let (final_witness, witness_shape) = segment_typed_final_witness(&lp, num_claims);
        let final_witness_bytes_runtime = final_witness.serialized_size(Compress::No);
        let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
            None,
            final_witness,
            0,
        );

        // The planner accounts for the final witness separately
        // (`segment_typed_witness_bytes` on the terminal plan). Subtract
        // it from the serialized terminal level to compare against
        // `terminal_level_proof_bytes`.
        let serialized_without_witness =
            terminal_proof.serialized_size(Compress::No) - final_witness_bytes_runtime;

        assert_eq!(
            level_proof_bytes(
                128,
                128,
                &lp,
                None,
                next_w_len,
                RelationMatrixRowLayout::WithoutCommitmentBlocks,
                None,
            )
            .unwrap(),
            serialized_without_witness,
            "planned terminal-level bytes should match the serialized terminal body \
                 (less final_witness) at log_basis={log_basis}"
        );

        let scheduled_bytes = segment_typed_witness_bytes(128, &witness_shape);
        assert!(
            scheduled_bytes >= final_witness_bytes_runtime,
            "scheduled direct witness budget must cover serialized segment-typed witness \
                 at log_basis={log_basis}"
        );
    }
}

#[test]
fn planned_batched_root_bytes_match_two_stage_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let next_lp = LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        2,
        3,
        2,
        fold_challenge_config,
    );
    let next_w_len = D * 8;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        let rounds = sumcheck_rounds(D, next_w_len);
        let b = 1usize << log_basis;
        let next_commitment = RingVec::from_ring_elems(&vec![
            CyclotomicRing::<F, D>::zero();
            next_lp.b_key.row_len()
        ])
        .into_compact();
        let level_proof = FoldLevelProof::new::<D>(
            vec![CyclotomicRing::<F, D>::zero(); lp.d_key.row_len()],
            dummy_stage1_proof(rounds, b),
            AkitaStage2Proof {
                sumcheck_proof: dummy_sumcheck(rounds, 3),
                next_witness_binding: NextWitnessBinding::OuterCommitment(next_commitment),
                next_w_eval: F::zero(),
            },
        );
        assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    RelationMatrixRowLayout::WithDBlock,
                    Some(crate::NextWitnessBindingPolicy::OuterCommitment),
                )
                .unwrap(),
                level_proof.serialized_size(Compress::No),
                "planned batched root bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
    }
}

#[test]
fn planned_root_extension_reduction_bytes_match_payload() {
    let extension_width = 4usize;
    let num_claims = 3usize;
    let opening_vars = 12usize;
    let partials = extension_width.saturating_mul(num_claims);
    let reduction = ExtensionOpeningReductionProof {
        partials: vec![F::zero(); partials],
        sumcheck: dummy_sumcheck(
            opening_vars - extension_width.trailing_zeros() as usize,
            EXTENSION_OPENING_REDUCTION_DEGREE,
        ),
    };
    let sumcheck_bytes = reduction.sumcheck.serialized_size(Compress::No);

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

#[test]
fn from_layout_accepts_scalar_layout() {
    let layout = OpeningClaimsLayout::new(4, 2).expect("scalar layout");
    let key = AkitaScheduleLookupKey::from_layout::<NoPrecommitSource>(&layout)
        .expect("scalar layout lookup");
    assert_eq!(key.final_group, PolynomialGroupLayout::new(4, 2));
    assert!(key.precommitteds.is_empty());
    assert_eq!(key.num_commitment_groups(), 1);
}

struct NoPrecommitSource;

impl ScheduleKeyPrecommitSource for NoPrecommitSource {
    fn precommitted_group_params(
        _group: PolynomialGroupLayout,
    ) -> Result<PrecommittedGroupParams, AkitaError> {
        Err(AkitaError::InvalidSetup(
            "NoPrecommitSource is only valid for scalar layouts".to_string(),
        ))
    }
}

#[test]
fn validate_rejects_zero_dimensions() {
    assert!(
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(0, 1))
            .validate()
            .is_err()
    );
    assert!(
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(20, 0))
            .validate()
            .is_err()
    );
    assert!(
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(20, 4))
            .validate()
            .is_ok()
    );
}

#[test]
fn group_batch_key_rejects_precommitted_num_vars_above_main() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(24, 1),
            num_live_ring_elements_per_claim: 1usize << 18,
            num_positions_per_block: 16,
            num_live_blocks: 1usize << 14,
            fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
            log_basis_inner: 1,
            log_basis_outer: 2,
            n_a: 3,
            a_coeff_linf_bound: 1,
            n_b: 4,
            b_coeff_linf_bound: 1,
        }],
    };

    let err = multi_group_key
        .validate()
        .expect_err("precommitted groups above the main num_vars must be rejected");
    assert!(matches!(err, AkitaError::InvalidInput(_)));
}

#[test]
fn group_batch_key_rejects_precommitted_num_vars_above_half_main() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(12, 1),
            num_live_ring_elements_per_claim: 64,
            num_positions_per_block: 16,
            num_live_blocks: 4,
            fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
            log_basis_inner: 1,
            log_basis_outer: 2,
            n_a: 3,
            a_coeff_linf_bound: 1,
            n_b: 4,
            b_coeff_linf_bound: 1,
        }],
    };

    multi_group_key
        .validate()
        .expect_err("precommitted groups above half the main key must be rejected");
}

#[test]
fn group_batch_key_allows_mixed_polynomial_counts() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(10, 1),
            num_live_ring_elements_per_claim: 16,
            num_positions_per_block: 4,
            num_live_blocks: 4,
            fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
            log_basis_inner: 1,
            log_basis_outer: 2,
            n_a: 3,
            a_coeff_linf_bound: 1,
            n_b: 4,
            b_coeff_linf_bound: 1,
        }],
    };

    multi_group_key
        .validate()
        .expect("unequal K_g is allowed for a supported precommitted dimension");
    assert_eq!(multi_group_key.num_commitment_groups(), 2);
}

#[test]
fn validate_frozen_precommit_rejects_geometry_mismatch() {
    let layout = PrecommittedGroupParams {
        group: PolynomialGroupLayout::new(20, 1),
        num_live_ring_elements_per_claim: 1,
        num_positions_per_block: 16,
        num_live_blocks: 1,
        fold_challenge_shape: akita_challenges::TensorChallengeShape::Flat,
        log_basis_inner: 1,
        log_basis_outer: 2,
        n_a: 3,
        a_coeff_linf_bound: 1,
        n_b: 4,
        b_coeff_linf_bound: 1,
    };
    let err = layout
        .validate_frozen_precommit(64)
        .expect_err("geometry must match num_vars");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}
