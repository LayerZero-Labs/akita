use super::*;
use crate::golomb_rice::golomb_rice_encode_vec;
use crate::tail_golomb_rice_z_params;
use crate::{
    extension_opening_reduction_proof_bytes, level_proof_bytes, sumcheck_rounds,
    terminal_response_bytes, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
    DigitRangePlan, ExtensionOpeningReductionProof, FoldLevelProof, NextWitnessBinding, RingVec,
    SisModulusProfileId, TailSegmentGroupLayout, TailSegmentLayout, TerminalLevelProof,
    TerminalResponse, TerminalResponseShape, EXTENSION_OPENING_REDUCTION_DEGREE,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore, Prime128OffsetA7F7};
use akita_serialization::{AkitaSerialize, Compress};
use akita_sumcheck::EqFactoredUniPoly;
use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};

type F = Prime128OffsetA7F7;

fn committed_params(ring_dimension: usize) -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(4, 4, 2, 2, 2)
    .expect("schedule validation params")
}

fn recursive_schedule(
    predecessor_ring_dimension: usize,
    successor_ring_dimension: usize,
    offload: bool,
) -> FoldSchedule {
    let predecessor = committed_params(predecessor_ring_dimension);
    let mut successor = committed_params(successor_ring_dimension);
    let incoming_setup_prefix = offload.then(|| {
        let natural_len = crate::SETUP_OFFLOAD_D_SETUP;
        let commitment_params = crate::setup_prefix_precommitted_params(&successor, natural_len)
            .expect("setup-prefix commitment params");
        crate::setup_prefix_slot_id(crate::SETUP_OFFLOAD_D_SETUP, natural_len, commitment_params)
    });
    successor.setup_prefix = incoming_setup_prefix.clone();
    let terminal = TerminalCommittedGroupParams::from_expanded_group(committed_params(
        successor_ring_dimension,
    ));
    let terminal_response_len = 3 * successor_ring_dimension;

    FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    source: RootSource::Dense {
                        coefficient_bits: 128,
                    },
                    challenge: RootFinalChallenge::Flat,
                    commitment: predecessor.clone(),
                },
                precommitted_groups: Vec::new(),
                open_commit_matrix: predecessor.open_commit_matrix.clone(),
                sparse_challenge_config: predecessor.fold_challenge_config,
                witness_partition: WitnessPartition::Single,
            },
            input_witness_len: predecessor_ring_dimension,
            output_witness_len: successor_ring_dimension,
        },
        recursive_folds: vec![RecursiveFoldStep {
            params: RecursiveFoldParams {
                open_commit_matrix: successor.open_commit_matrix.clone(),
                sparse_challenge_config: successor.fold_challenge_config,
                incoming_setup_prefix,
                witness_partition: WitnessPartition::Single,
                witness: successor,
            },
            input_witness_len: successor_ring_dimension,
            output_witness_len: successor_ring_dimension,
        }],
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: terminal,
                sparse_challenge_config: SparseChallengeConfig::pm1_only(3),
                response_shape: TerminalResponseShape {
                    layout: TailSegmentLayout {
                        ring_dimension: successor_ring_dimension,
                        groups: vec![TailSegmentGroupLayout {
                            z_coords: successor_ring_dimension,
                            e_field_elems: successor_ring_dimension,
                            t_field_elems: successor_ring_dimension,
                            z_payload_bytes: 1,
                            z_rice_low_bits: 0,
                        }],
                        logical_num_elems: terminal_response_len,
                    },
                },
            },
            input_witness_len: successor_ring_dimension,
        },
    }
}

#[test]
fn schedule_rejects_setup_prefix_split_authority() {
    let mut schedule = recursive_schedule(64, 64, true);
    schedule.recursive_folds[0].params.witness.setup_prefix = None;

    let err = schedule
        .validate_structure()
        .expect_err("setup-prefix authorities must agree");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn schedule_rejects_offloaded_ring_dimension_transition() {
    let schedule = recursive_schedule(128, 64, true);

    let err = schedule
        .validate_structure()
        .expect_err("offload requires uniform predecessor/successor geometry");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn schedule_accepts_direct_ring_dimension_transition() {
    recursive_schedule(128, 64, false)
        .validate_structure()
        .expect("direct setup contribution supports a ring-dimension transition");
}

#[test]
fn schedule_accepts_offload_at_uniform_successor_dimension() {
    recursive_schedule(64, 64, true)
        .validate_structure()
        .expect("offload supports uniform predecessor/successor geometry");
}

#[test]
fn terminal_projection_preserves_the_fixed_inner_matrix() {
    let sparse = SparseChallengeConfig::pm1_only(3);
    let committed = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        4,
        3,
        2,
        sparse,
    )
    .with_decomp(4, 32, 2, 2, 2)
    .expect("committed params");
    let expected_inner = committed.inner_commit_matrix.clone();

    let (terminal, response_cap) = TerminalCommittedGroupParams::try_from_expanded_group(committed)
        .expect("terminal projection");
    let response_policy = terminal
        .response_linf_policy(&sparse)
        .expect("terminal response bounds");

    assert_eq!(terminal.inner_commit_matrix, expected_inner);
    assert_eq!(response_cap, response_policy.admission_cap);
    assert!(response_policy.admission_cap <= response_policy.certified_capacity);
    assert!(
        response_policy.admission_cap >= response_policy.unconstrained_target.div_ceil(2),
        "terminal capacity must retain at least half of the unconstrained target"
    );
}

#[test]
fn chunked_witness_count_matches_chunk_layout_arithmetic() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    // num_live_blocks = 2^3 = 8, divisible by {1, 2, 4, 8}.
    let lp = CommittedGroupParams::params_only(
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
    let lp = CommittedGroupParams::params_only(
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

fn terminal_response_fixture(
    lp: &CommittedGroupParams,
    num_claims: usize,
) -> (TerminalResponse<F>, TerminalResponseShape) {
    let field_bits = F::modulus_bits();
    let shape = TerminalResponseShape::from_groups(
        lp,
        field_bits,
        [(lp as &dyn crate::LevelParamsLike, num_claims, num_claims, 1)],
    )
    .expect("terminal response shape");
    let layout = shape.layout.clone();
    let group = layout.groups[0];
    let (rice_low_bits, zigzag_w) =
        tail_golomb_rice_z_params(lp, num_claims).expect("golomb z params");
    let z_payload = golomb_rice_encode_vec(&vec![0i64; group.z_coords], rice_low_bits, zigzag_w)
        .expect("encode zero z segment");
    let witness = TerminalResponse {
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
        stages: DigitRangePlan::new(b)
            .expect("test range basis")
            .stage_shapes(rounds)
            .into_iter()
            .map(|shape| AkitaStage1StageProof {
                sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                child_claims: vec![F::zero(); shape.child_claims],
            })
            .collect(),
        range_image_evaluation: F::zero(),
    }
}

fn exact_level_proof_bytes<F: FieldCore + CanonicalField + AkitaSerialize>(
    lp: &CommittedGroupParams,
    next_lp: &CommittedGroupParams,
    output_witness_len: usize,
) -> Result<usize, AkitaError> {
    let current_coeffs = lp
        .open_commit_matrix
        .output_rank()
        .checked_mul(lp.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let next_commit_coeffs = next_lp
        .outer_commit_matrix
        .output_rank()
        .checked_mul(next_lp.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let rounds = sumcheck_rounds(lp.d_a(), output_witness_len);
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
    let next_lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        2,
        3,
        2,
        fold_challenge_config,
    );
    let output_witness_len = D * 8;

    for log_basis in 2..=6 {
        let lp = CommittedGroupParams::params_only(
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
                    output_witness_len,
                    Some(crate::NextWitnessBindingPolicy::OuterCommitment),
                )
                .unwrap(),
                exact_level_proof_bytes::<F>(&lp, &next_lp, output_witness_len).unwrap(),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
    }
}

#[test]
fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let num_claims = 3;

    for log_basis in 2..=6 {
        let lp = CommittedGroupParams::params_only(
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

        let (terminal_response, witness_shape) = terminal_response_fixture(&lp, num_claims);
        let terminal_response_bytes_runtime = terminal_response.serialized_size(Compress::No);
        let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
            None,
            terminal_response,
            0,
        );

        // The planner accounts for the final witness separately
        // (`terminal_response_bytes` on the terminal plan). Subtract
        // it from the serialized terminal level: a direct terminal level
        // carries only the `fold_grind_nonce` (plus any extension-opening
        // reduction, absent from this fixture), matching the planner's
        // terminal-direct accounting.
        let serialized_without_witness =
            terminal_proof.serialized_size(Compress::No) - terminal_response_bytes_runtime;

        assert_eq!(
            crate::FOLD_GRIND_NONCE_BYTES,
            serialized_without_witness,
            "planned terminal-level bytes should match the serialized terminal body \
                 (less terminal_response) at log_basis={log_basis}"
        );

        let scheduled_bytes = terminal_response_bytes(128, &witness_shape);
        assert!(
            scheduled_bytes >= terminal_response_bytes_runtime,
            "scheduled direct witness budget must cover serialized terminal response \
                 at log_basis={log_basis}"
        );
    }
}

#[test]
fn planned_batched_root_bytes_match_two_stage_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let next_lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        2,
        2,
        3,
        2,
        fold_challenge_config,
    );
    let output_witness_len = D * 8;

    for log_basis in 2..=6 {
        let lp = CommittedGroupParams::params_only(
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
        let rounds = sumcheck_rounds(D, output_witness_len);
        let b = 1usize << log_basis;
        let next_commitment =
            RingVec::from_ring_elems(&vec![
                CyclotomicRing::<F, D>::zero();
                next_lp.outer_commit_matrix.output_rank()
            ])
            .into_compact();
        let level_proof = FoldLevelProof::new::<D>(
            vec![CyclotomicRing::<F, D>::zero(); lp.open_commit_matrix.output_rank()],
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
                    output_witness_len,
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
    ) -> Result<PrecommittedGroupDescriptor, AkitaError> {
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
        precommitteds: vec![PrecommittedGroupDescriptor {
            group: PolynomialGroupLayout::new(24, 1),
            num_live_ring_elements_per_claim: 1usize << 18,
            num_positions_per_block: 16,
            num_live_blocks: 1usize << 14,
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
        precommitteds: vec![PrecommittedGroupDescriptor {
            group: PolynomialGroupLayout::new(12, 1),
            num_live_ring_elements_per_claim: 64,
            num_positions_per_block: 16,
            num_live_blocks: 4,
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
        precommitteds: vec![PrecommittedGroupDescriptor {
            group: PolynomialGroupLayout::new(10, 1),
            num_live_ring_elements_per_claim: 16,
            num_positions_per_block: 4,
            num_live_blocks: 4,
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
    let layout = PrecommittedGroupDescriptor {
        group: PolynomialGroupLayout::new(20, 1),
        num_live_ring_elements_per_claim: 1,
        num_positions_per_block: 16,
        num_live_blocks: 1,
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
