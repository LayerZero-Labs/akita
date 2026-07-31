use super::*;

#[test]
fn fold_schedule_estimate_separates_direct_and_stage3_payloads() {
    let estimate = FoldScheduleEstimate {
        estimated_root_direct_payload_bytes: 100,
        estimated_root_stage3_payload_bytes: 11,
        estimated_recursive_direct_payload_bytes: vec![200, 300],
        estimated_recursive_stage3_payload_bytes: vec![22, 0],
        estimated_terminal_direct_payload_bytes: 400,
        estimated_terminal_response_payload_bytes: 350,
        estimated_setup_envelope_ring_elements: 512,
        first_direct_setup_field_len: Some(1_024),
        selected_offload_edges: 2,
    };

    assert_eq!(
        estimate.estimated_direct_proof_payload_bytes().unwrap(),
        1_000
    );
    assert_eq!(estimate.estimated_stage3_payload_bytes().unwrap(), 33);
    assert_eq!(estimate.estimated_proof_payload_bytes().unwrap(), 1_033);
}
use crate::golomb_rice::golomb_rice_encode_vec;
use crate::tail_golomb_rice_z_params;
use crate::{
    extension_opening_reduction_proof_bytes, level_proof_bytes, sumcheck_rounds,
    terminal_response_bytes, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
    DecompositionParams, DigitRangePlan, ExtensionOpeningReductionProof, FoldLevelProof,
    NextWitnessBinding, RingVec, SisModulusProfileId, TailSegmentGroupLayout, TailSegmentLayout,
    TerminalLevelProof, TerminalResponse, TerminalResponseShape,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore, Prime128OffsetA7F7};
use akita_serialization::{AkitaSerialize, Compress};
use akita_sumcheck::EqFactoredUniPoly;
use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};

type F = Prime128OffsetA7F7;

fn committed_params(ring_dimension: usize) -> CommittedGroupParams {
    committed_params_with_geometry(ring_dimension, 4, 4)
}

fn committed_params_with_geometry(
    ring_dimension: usize,
    num_positions_per_block: usize,
    num_live_ring_elements_per_claim: usize,
) -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        ring_dimension,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(
        num_positions_per_block,
        num_live_ring_elements_per_claim,
        2,
        2,
        2,
    )
    .expect("schedule validation params")
}

fn retarget_outer_dimension(
    params: &mut CommittedGroupParams,
    ring_dimension: usize,
) -> Result<(), AkitaError> {
    let outer = &params.outer_commit_matrix;
    let column_scale = outer.ring_dimension() / ring_dimension;
    params.outer_commit_matrix = crate::sis::OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * column_scale,
        outer.coeff_linf_bound(),
        ring_dimension,
    );
    Ok(())
}

fn retarget_open_dimension(
    params: &mut CommittedGroupParams,
    ring_dimension: usize,
) -> Result<(), AkitaError> {
    let open = &params.open_commit_matrix;
    let column_scale = open.ring_dimension() / ring_dimension;
    params.open_commit_matrix = crate::sis::OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width() * column_scale,
        open.coeff_linf_bound(),
        ring_dimension,
    );
    Ok(())
}

fn precommitted_group_params(
    params: &CommittedGroupParams,
    group: PolynomialGroupLayout,
) -> crate::PrecommittedLevelParams {
    crate::PrecommittedLevelParams {
        layout: PrecommittedGroupDescriptor::from_params(group, params),
        inner_commit_matrix: params.inner_commit_matrix.clone(),
        outer_commit_matrix: params.outer_commit_matrix.clone(),
        log_basis_open: params.log_basis_open,
        fold_challenge_config: params.fold_challenge_config,
        num_digits_inner: params.num_digits_inner,
        num_digits_outer: params.num_digits_outer,
        num_digits_open: params.num_digits_open,
        num_digits_fold_one: params.num_digits_fold_one,
    }
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
fn root_source_derivation_distinguishes_dense_and_onehot_bounds() {
    let dense = committed_params(64);
    assert_eq!(
        RootSource::from_commitment(&dense),
        RootSource::Dense {
            coefficient_bits: 128
        }
    );
    assert_eq!(
        RootSource::from_config(
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 128,
                log_open_bound: None,
            },
            256,
        ),
        RootSource::Dense {
            coefficient_bits: 128
        }
    );

    let onehot = dense.with_onehot_chunk_size(256);
    assert_eq!(
        RootSource::from_commitment(&onehot),
        RootSource::OneHot { chunk_size: 256 }
    );
    assert_eq!(
        RootSource::from_config(
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            },
            256,
        ),
        RootSource::OneHot { chunk_size: 256 }
    );
}

#[test]
fn schedule_rejects_root_source_that_disagrees_with_commitment_bounds() {
    let mut schedule = recursive_schedule(64, 64, false);
    schedule.root.params.final_group.source = RootSource::OneHot { chunk_size: 256 };
    assert!(matches!(
        schedule.validate_structure(),
        Err(AkitaError::InvalidSetup(_))
    ));

    schedule.root.params.final_group.source = RootSource::Dense {
        coefficient_bits: 32,
    };
    assert!(matches!(
        schedule.validate_structure(),
        Err(AkitaError::InvalidSetup(_))
    ));
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
fn schedule_rejects_offload_when_producer_projection_misses_prefix_dimension() {
    let schedule = recursive_schedule(128, 64, true);

    let err = schedule
        .validate_structure()
        .expect_err("offload prefix must use the producer setup projection dimension");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn schedule_accepts_stage2_points_within_successor_capacity() {
    recursive_schedule(128, 64, false)
        .validate_structure()
        .expect("successor cubes may be wider than their incoming Stage 2 points");
}

#[test]
fn schedule_rejects_root_stage2_point_wider_than_successor() {
    let mut schedule = recursive_schedule(64, 64, false);
    let narrow_successor = committed_params_with_geometry(64, 1, 1);
    schedule.root.output_witness_len = 128;
    schedule.recursive_folds[0].input_witness_len = 128;
    schedule.recursive_folds[0].params.open_commit_matrix =
        narrow_successor.open_commit_matrix.clone();
    schedule.recursive_folds[0].params.witness = narrow_successor;

    let err = schedule
        .validate_structure()
        .expect_err("the successor cube cannot hold the root Stage 2 point");
    assert!(
        matches!(err, AkitaError::InvalidSetup(message) if message.contains("root fold Stage 2 point"))
    );
}

#[test]
fn schedule_rejects_recursive_stage2_point_wider_than_terminal() {
    let mut schedule = recursive_schedule(64, 64, false);
    let narrow_terminal = committed_params_with_geometry(64, 1, 1);
    schedule.recursive_folds[0].output_witness_len = 128;
    schedule.terminal.input_witness_len = 128;
    schedule.terminal.params.witness =
        TerminalCommittedGroupParams::from_expanded_group(narrow_terminal);

    let err = schedule
        .validate_structure()
        .expect_err("the terminal cube cannot hold the recursive Stage 2 point");
    assert!(
        matches!(err, AkitaError::InvalidSetup(message) if message.contains("recursive fold 0 Stage 2 point"))
    );
}

#[test]
fn schedule_accepts_offload_at_uniform_successor_dimension() {
    recursive_schedule(64, 64, true)
        .validate_structure()
        .expect("offload supports uniform predecessor/successor geometry");
}

#[test]
fn schedule_accepts_mixed_producer_projecting_to_prefix_dimension() {
    let mut schedule = recursive_schedule(128, 64, true);
    let producer = &mut schedule.root.params.final_group.commitment;
    retarget_outer_dimension(producer, 64).expect("retarget producer B role");
    retarget_open_dimension(producer, 64).expect("retarget producer D role");

    schedule
        .validate_structure()
        .expect("mixed A128/B64/D64 producer projects its setup prefix at D64");
}

#[test]
fn schedule_rejects_prefix_commitment_roles_that_miss_consumer_roles() {
    let mut schedule = recursive_schedule(64, 64, true);
    retarget_outer_dimension(&mut schedule.recursive_folds[0].params.witness, 32)
        .expect("retarget consumer B role");

    let err = schedule
        .validate_structure()
        .expect_err("prefix B commitment must match the consumer B role");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn schedule_accepts_exact_multi_group_prefix_from_mixed_producer() {
    let mut schedule = recursive_schedule(128, 64, false);
    let producer = &mut schedule.root.params.final_group.commitment;
    retarget_outer_dimension(producer, 64).expect("retarget producer B role");
    retarget_open_dimension(producer, 64).expect("retarget producer D role");

    let final_group = PolynomialGroupLayout::new(9, 1);
    let singleton_layout =
        OpeningClaimsLayout::from_groups(vec![final_group]).expect("singleton layout");
    let singleton_natural_len = crate::active_setup_field_len(producer, &singleton_layout)
        .expect("singleton setup geometry");

    let precommitted_group = PolynomialGroupLayout::new(9, 1);
    let mut group_params = producer.clone();
    group_params.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(group_params.d_a())
            .expect("precommitted test group uses a production ring dimension");
    let inner = &group_params.inner_commit_matrix;
    group_params.inner_commit_matrix = crate::sis::InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        1,
        inner.ring_dimension(),
    );
    let outer = &group_params.outer_commit_matrix;
    group_params.outer_commit_matrix = crate::sis::OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        1,
        outer.ring_dimension(),
    );
    let precommitted = precommitted_group_params(&group_params, precommitted_group);
    let one_precommitted_d_width = precommitted
        .d_segment_width(producer.role_dims().d_d())
        .expect("precommitted D width");
    let precommitted_group_count = 8;
    producer.precommitted_groups = vec![precommitted; precommitted_group_count];
    let precommitted_d_width = one_precommitted_d_width * precommitted_group_count;

    let open = &producer.open_commit_matrix;
    producer.open_commit_matrix = crate::sis::OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width() + precommitted_d_width,
        open.coeff_linf_bound(),
        open.ring_dimension(),
    );

    let mut groups = vec![precommitted_group; precommitted_group_count];
    groups.push(final_group);
    let opening_layout = OpeningClaimsLayout::from_groups(groups).expect("multi-group layout");
    let natural_len = crate::active_setup_field_len(producer, &opening_layout)
        .expect("multi-group mixed setup geometry");
    assert!(
        natural_len > singleton_natural_len,
        "the exact prefix must include the larger multi-group setup footprint"
    );

    let n_prefix = crate::padded_setup_prefix_len(natural_len);
    let mut consumer = committed_params_with_geometry(64, 16, 64);
    let prefix_ring_slots = n_prefix / crate::SETUP_OFFLOAD_D_SETUP;
    let inner = &consumer.inner_commit_matrix;
    consumer.inner_commit_matrix = crate::sis::InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        prefix_ring_slots * consumer.num_digits_inner,
        inner.coeff_linf_bound(),
        inner.ring_dimension(),
    );
    let commitment_params = crate::setup_prefix_precommitted_params(&consumer, n_prefix)
        .expect("consumer-compatible prefix commitment");
    let prefix =
        crate::setup_prefix_slot_id(crate::SETUP_OFFLOAD_D_SETUP, natural_len, commitment_params);
    schedule.recursive_folds[0].params.witness = consumer.clone();
    schedule.recursive_folds[0].params.open_commit_matrix = consumer.open_commit_matrix.clone();
    schedule.recursive_folds[0].params.incoming_setup_prefix = Some(prefix.clone());
    schedule.recursive_folds[0].params.witness.setup_prefix = Some(prefix);

    schedule
        .validate_structure()
        .expect("mixed multi-group producer offloads its exact D64 setup projection");
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
fn planned_level_bytes_match_non_offloaded_payload_at_all_bases() {
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
                "planned level bytes should match the serialized non-offloaded body at log_basis={log_basis}"
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
fn planned_batched_root_bytes_match_non_offloaded_payload_at_all_bases() {
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
                "planned batched root bytes should match the serialized non-offloaded body at log_basis={log_basis}"
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

fn precommitted_descriptor(num_vars: usize) -> PrecommittedGroupDescriptor {
    PrecommittedGroupDescriptor {
        group: PolynomialGroupLayout::new(num_vars, 1),
        num_live_ring_elements_per_claim: 1usize << (num_vars - 6),
        num_positions_per_block: 16,
        num_live_blocks: 1usize << (num_vars - 10),
        log_basis_inner: 1,
        log_basis_outer: 2,
        inner_ring_dimension: 64,
        outer_ring_dimension: 64,
        n_a: 3,
        a_coeff_linf_bound: 1,
        n_b: 4,
        b_coeff_linf_bound: 1,
    }
}

#[test]
fn group_batch_key_separates_final_source_arity_from_max_opening_arity() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(14, 3),
        precommitteds: vec![precommitted_descriptor(20)],
    };

    multi_group_key
        .validate()
        .expect("commit order must not impose an arity ordering");
    assert_eq!(multi_group_key.final_group.num_vars(), 14);
    assert_eq!(multi_group_key.max_num_vars(), 20);
    assert!(!multi_group_key.fits_setup_capacity(19, 4).unwrap());
    assert!(multi_group_key.fits_setup_capacity(20, 4).unwrap());

    let opening_layout = multi_group_key.opening_layout().expect("opening layout");
    assert_eq!(opening_layout.max_num_vars(), 20);
    assert_eq!(
        opening_layout.groups(),
        &[
            PolynomialGroupLayout::new(20, 1),
            PolynomialGroupLayout::new(14, 3),
        ],
        "opening layout must preserve precommitted-then-final transcript order"
    );
}

#[test]
fn group_batch_key_allows_independent_precommitted_num_vars() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![precommitted_descriptor(12)],
    };

    multi_group_key
        .validate()
        .expect("precommitted group arity is not derived from the final group");
}

#[test]
fn group_batch_key_allows_precommitted_num_vars_equal_to_main() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![precommitted_descriptor(20)],
    };

    multi_group_key
        .validate()
        .expect("precommitted groups may use the final group's full arity");
}

#[test]
fn group_batch_key_allows_mixed_polynomial_counts() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![PrecommittedGroupDescriptor {
            group: PolynomialGroupLayout::new(10, 2),
            num_live_ring_elements_per_claim: 16,
            num_positions_per_block: 4,
            num_live_blocks: 4,
            log_basis_inner: 1,
            log_basis_outer: 2,
            inner_ring_dimension: 64,
            outer_ring_dimension: 64,
            n_a: 3,
            a_coeff_linf_bound: 1,
            n_b: 4,
            b_coeff_linf_bound: 1,
        }],
    };

    multi_group_key
        .validate()
        .expect("a precommitted group may contain multiple polynomials");
    assert_eq!(multi_group_key.num_commitment_groups(), 2);
    assert_eq!(multi_group_key.num_polynomials().unwrap(), 5);
    assert!(!multi_group_key.fits_setup_capacity(20, 4).unwrap());
    assert!(multi_group_key.fits_setup_capacity(20, 5).unwrap());
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
        inner_ring_dimension: 64,
        outer_ring_dimension: 64,
        n_a: 3,
        a_coeff_linf_bound: 1,
        n_b: 4,
        b_coeff_linf_bound: 1,
    };
    let err = layout
        .validate_frozen_precommit()
        .expect_err("geometry must match num_vars");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}
