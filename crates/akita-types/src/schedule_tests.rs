use super::*;
use crate::golomb_rice::golomb_rice_encode_vec;
use crate::proof::{segment_typed_witness_shape_from_groups, SegmentTypedWitness};
use crate::tail_golomb_rice_z_params;
use crate::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_proof_bytes,
    stage1_tree_stage_shapes, sumcheck_rounds, AkitaBatchedRootProof, AkitaIntermediateStage2Proof,
    AkitaLevelProof, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
    CleartextWitnessProof, ExtensionOpeningReductionProof, RelationMatrixRowLayout, RingVec,
    SisModulusFamily, TerminalLevelProof, EXTENSION_OPENING_REDUCTION_DEGREE,
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
    // num_blocks = 2^3 = 8, divisible by {1, 2, 4, 8}.
    let lp = LevelParams::params_only(SisModulusFamily::Q128, D, 3, 2, 2, 2, fold_challenge_config)
        .with_decomp(2, 3, 2, 2, 0)
        .unwrap();
    let field_bits = 128u32;
    let num_poly = 3usize;

    for layout in [
        RelationMatrixRowLayout::WithDBlock,
        RelationMatrixRowLayout::WithoutDBlock,
    ] {
        let single =
            w_ring_element_count_with_counts_for_layout_bits(field_bits, &lp, num_poly, 1, layout)
                .unwrap();
        // num_chunks = 1 must be byte-identical to the single-chunk delegate.
        assert_eq!(
            w_ring_element_count_for_chunks(field_bits, &lp, num_poly, layout, 1).unwrap(),
            single
        );

        let z_pre = lp.inner_width() * lp.num_digits_fold(num_poly, field_bits).unwrap();
        for num_chunks in [2usize, 4, 8] {
            let chunked =
                w_ring_element_count_for_chunks(field_bits, &lp, num_poly, layout, num_chunks)
                    .unwrap();
            // ê/t̂ totals are unchanged (partitioned), and the shared r-tail is
            // a single summed quotient that keeps the single-machine row count
            // (num_commitments = 1). So the ONLY growth is the replicated ẑ:
            // (num_chunks - 1) full-width copies.
            assert_eq!(chunked, single + (num_chunks - 1) * z_pre);
            assert!(chunked > single, "chunked layout must grow vs single chunk");
        }
    }
}

#[test]
fn chunked_witness_count_rejects_invalid_chunk_counts() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    // num_blocks = 2^3 = 8.
    let lp = LevelParams::params_only(SisModulusFamily::Q128, D, 3, 2, 2, 2, fold_challenge_config)
        .with_decomp(2, 3, 2, 2, 0)
        .unwrap();
    // Non-power-of-two chunk count.
    assert!(matches!(
        w_ring_element_count_for_chunks(128, &lp, 1, RelationMatrixRowLayout::WithDBlock, 6),
        Err(AkitaError::InvalidSetup(_))
    ));
    // num_chunks does not divide num_blocks (8 % 16 != 0).
    assert!(matches!(
        w_ring_element_count_for_chunks(128, &lp, 1, RelationMatrixRowLayout::WithDBlock, 16),
        Err(AkitaError::InvalidSetup(_))
    ));
    // Zero chunks.
    assert!(matches!(
        w_ring_element_count_for_chunks(128, &lp, 1, RelationMatrixRowLayout::WithDBlock, 0),
        Err(AkitaError::InvalidSetup(_))
    ));
}

fn segment_typed_final_witness(
    lp: &LevelParams,
    num_claims: usize,
) -> (CleartextWitnessProof<F>, CleartextWitnessShape) {
    let field_bits = F::modulus_bits();
    let shape = segment_typed_witness_shape_from_groups(
        lp,
        field_bits,
        [(lp as &dyn crate::LevelParamsLike, num_claims, num_claims, 1)],
        1,
    )
    .expect("segment-typed witness shape");
    let CleartextWitnessShape::SegmentTyped(ref segment_shape) = shape else {
        panic!("expected segment-typed witness shape");
    };
    let layout = segment_shape.layout.clone();
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
        r_fields: RingVec::from_coeffs(vec![F::zero(); layout.r_field_elems]),
    };
    (CleartextWitnessProof::SegmentTyped(witness), shape)
}

fn grouped_level_params() -> LevelParams {
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let mut params = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("grouped params");
    let precommitted = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("precommitted params");
    params.precommitted_groups = vec![crate::PrecommittedLevelParams {
        layout: PrecommittedGroupParams::from_params(
            PolynomialGroupLayout::new(6, 1),
            &precommitted,
        ),
        a_key: precommitted.a_key.clone(),
        b_key: precommitted.b_key.clone(),
        num_blocks: precommitted.num_blocks,
        block_len: precommitted.block_len,
        num_digits_commit: precommitted.num_digits_commit,
        num_digits_open: precommitted.num_digits_open,
        num_digits_fold_one: precommitted.num_digits_fold_one,
    }];
    params
}

fn scalar_terminal_steps(current_w_len: usize, params: LevelParams) -> (FoldStep, DirectStep) {
    let (_, witness_shape) = segment_typed_final_witness(&params, 1);
    let CleartextWitnessShape::SegmentTyped(shape) = &witness_shape else {
        panic!("expected segment-typed witness");
    };
    let terminal_w_len = shape.layout.logical_num_elems;
    (
        FoldStep {
            params,
            current_w_len,
            next_w_len: terminal_w_len,
            level_bytes: 0,
        },
        DirectStep {
            current_w_len: terminal_w_len,
            witness_shape,
            direct_bytes: 0,
            params: None,
        },
    )
}

#[test]
fn root_direct_schedule_uses_field_element_payload() {
    let dummy_commit_params = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::pm1_only(1),
    );
    let schedule =
        root_direct_schedule(8, dummy_commit_params.clone()).expect("root-direct schedule");
    assert_eq!(schedule.total_bytes, 0);

    let [Step::Direct(step)] = schedule.steps.as_slice() else {
        panic!("root-direct schedule should contain one direct step");
    };
    assert_eq!(step.current_w_len, 8);
    assert_eq!(step.witness_shape, CleartextWitnessShape::FieldElements(8));
    assert_eq!(step.direct_bytes, 0);
    assert_eq!(step.params.as_ref(), Some(&dummy_commit_params));
}

#[test]
fn schedule_structure_accepts_scalar_and_nonterminal_grouped_shapes() {
    let scalar = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("scalar params");
    let (terminal_fold, direct) = scalar_terminal_steps(64, scalar.clone());
    let scalar_schedule = Schedule {
        steps: vec![
            Step::Fold(terminal_fold.clone()),
            Step::Direct(direct.clone()),
        ],
        total_bytes: 0,
    };
    scalar_schedule
        .validate_structure()
        .expect("scalar terminal schedule");

    let grouped_schedule = Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: grouped_level_params(),
                current_w_len: 128,
                next_w_len: terminal_fold.current_w_len,
                level_bytes: 0,
            }),
            Step::Fold(terminal_fold),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    };
    grouped_schedule
        .validate_structure()
        .expect("nonterminal grouped schedule");
}

#[test]
fn schedule_structure_rejects_empty_and_internal_direct_steps() {
    assert!(Schedule {
        steps: Vec::new(),
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());

    let params = grouped_level_params();
    let direct = DirectStep {
        current_w_len: 8,
        witness_shape: CleartextWitnessShape::FieldElements(8),
        direct_bytes: 0,
        params: Some(params),
    };
    assert!(Schedule {
        steps: vec![Step::Direct(direct.clone()), Step::Direct(direct)],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

#[test]
fn root_direct_schedule_rejects_grouped_params() {
    let grouped = grouped_level_params();
    assert!(root_direct_schedule(8, grouped).is_err());
}

#[test]
fn schedule_structure_rejects_grouped_direct_and_terminal_fold() {
    let grouped = grouped_level_params();
    let grouped_direct = Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len: 8,
            witness_shape: CleartextWitnessShape::FieldElements(8),
            direct_bytes: 0,
            params: Some(grouped.clone()),
        })],
        total_bytes: 0,
    };
    assert!(grouped_direct.validate_structure().is_err());

    let scalar = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("scalar params");
    let (_, direct) = scalar_terminal_steps(64, scalar);
    let terminal_w_len = direct.current_w_len;
    assert!(Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: grouped,
                current_w_len: 128,
                next_w_len: terminal_w_len,
                level_bytes: 0,
            }),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

#[test]
fn schedule_structure_rejects_discontinuous_witness_lengths() {
    let scalar = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("scalar params");
    let (mut fold, direct) = scalar_terminal_steps(64, scalar);
    fold.next_w_len = direct.current_w_len + 1;
    assert!(Schedule {
        steps: vec![Step::Fold(fold), Step::Direct(direct)],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

#[test]
fn schedule_structure_rejects_grouped_multi_chunk() {
    let mut grouped = grouped_level_params();
    grouped.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 1,
    };
    let scalar = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("scalar params");
    let (terminal_fold, direct) = scalar_terminal_steps(64, scalar);
    assert!(Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: grouped,
                current_w_len: 128,
                next_w_len: terminal_fold.current_w_len,
                level_bytes: 0,
            }),
            Step::Fold(terminal_fold),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

fn scalar_level_params() -> LevelParams {
    LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("scalar params")
}

fn setup_prefix_only_params() -> LevelParams {
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let precommitted = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        3,
        2,
        2,
        2,
        fold_challenge_config,
    )
    .with_decomp(2, 2, 2, 2, 0)
    .expect("setup prefix params");
    let mut params = scalar_level_params();
    params.setup_prefix = Some(crate::setup_prefix_slot_id(
        64,
        1 << 12,
        crate::PrecommittedLevelParams {
            layout: crate::PrecommittedGroupParams::from_params(
                PolynomialGroupLayout::new(6, 1),
                &precommitted,
            ),
            a_key: precommitted.a_key.clone(),
            b_key: precommitted.b_key.clone(),
            num_blocks: precommitted.num_blocks,
            block_len: precommitted.block_len,
            num_digits_commit: precommitted.num_digits_commit,
            num_digits_open: precommitted.num_digits_open,
            num_digits_fold_one: precommitted.num_digits_fold_one,
        },
    ));
    params
}

fn recursive_setup_offload_schedule() -> Schedule {
    let recursive = {
        let mut params = scalar_level_params();
        params.setup_contribution_mode = SetupContributionMode::Recursive;
        params
    };
    let setup_prefix_successor = setup_prefix_only_params();
    let (terminal_fold, direct) = scalar_terminal_steps(64, scalar_level_params());
    Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: recursive,
                current_w_len: 256,
                next_w_len: 128,
                level_bytes: 0,
            }),
            Step::Fold(FoldStep {
                params: setup_prefix_successor,
                current_w_len: 128,
                next_w_len: terminal_fold.current_w_len,
                level_bytes: 0,
            }),
            Step::Fold(terminal_fold),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    }
}

#[test]
fn schedule_structure_accepts_recursive_setup_offload_chain() {
    recursive_setup_offload_schedule()
        .validate_structure()
        .expect("recursive setup offload chain");
}

#[test]
fn schedule_structure_rejects_recursive_to_direct() {
    let mut schedule = recursive_setup_offload_schedule();
    let terminal = schedule.steps.pop().expect("direct");
    let Step::Direct(direct) = terminal else {
        panic!("expected direct");
    };
    schedule.steps.pop();
    schedule.steps.push(Step::Direct(direct));
    assert!(schedule.validate_structure().is_err());
}

#[test]
fn schedule_structure_rejects_recursive_to_scalar_without_prefix() {
    let recursive = {
        let mut params = scalar_level_params();
        params.setup_contribution_mode = SetupContributionMode::Recursive;
        params
    };
    let (terminal_fold, direct) = scalar_terminal_steps(64, scalar_level_params());
    assert!(Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: recursive,
                current_w_len: 256,
                next_w_len: terminal_fold.current_w_len,
                level_bytes: 0,
            }),
            Step::Fold(terminal_fold),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

#[test]
fn schedule_structure_rejects_direct_to_setup_prefix_successor() {
    let (terminal_fold, direct) = scalar_terminal_steps(64, scalar_level_params());
    assert!(Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: scalar_level_params(),
                current_w_len: 256,
                next_w_len: 128,
                level_bytes: 0,
            }),
            Step::Fold(FoldStep {
                params: setup_prefix_only_params(),
                current_w_len: 128,
                next_w_len: terminal_fold.current_w_len,
                level_bytes: 0,
            }),
            Step::Fold(terminal_fold),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

#[test]
fn schedule_structure_rejects_recursive_successor_with_extra_precommitted() {
    let recursive = {
        let mut params = scalar_level_params();
        params.setup_contribution_mode = SetupContributionMode::Recursive;
        params
    };
    let mut successor = setup_prefix_only_params();
    successor.precommitted_groups = grouped_level_params().precommitted_groups;
    let (terminal_fold, direct) = scalar_terminal_steps(64, scalar_level_params());
    assert!(Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: recursive,
                current_w_len: 256,
                next_w_len: 128,
                level_bytes: 0,
            }),
            Step::Fold(FoldStep {
                params: successor,
                current_w_len: 128,
                next_w_len: terminal_fold.current_w_len,
                level_bytes: 0,
            }),
            Step::Fold(terminal_fold),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

#[test]
fn schedule_structure_rejects_root_fold_with_setup_prefix() {
    let mut root = grouped_level_params();
    root.setup_prefix = setup_prefix_only_params().setup_prefix;
    let scalar = scalar_level_params();
    let (terminal_fold, direct) = scalar_terminal_steps(64, scalar);
    assert!(Schedule {
        steps: vec![
            Step::Fold(FoldStep {
                params: root,
                current_w_len: 256,
                next_w_len: terminal_fold.current_w_len,
                level_bytes: 0,
            }),
            Step::Fold(terminal_fold),
            Step::Direct(direct),
        ],
        total_bytes: 0,
    }
    .validate_structure()
    .is_err());
}

#[test]
fn root_direct_schedule_uses_multi_group_witness_len() {
    let layout = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(2, 1),
        PolynomialGroupLayout::new(3, 2),
        PolynomialGroupLayout::new(4, 1),
    ])
    .expect("multi-group layout");
    let witness_len = layout.root_direct_witness_len().expect("witness len");
    assert_eq!(witness_len, 4 + 16 + 16);

    let dummy_commit_params = LevelParams::params_only(
        crate::SisModulusFamily::Q128,
        64,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::pm1_only(3),
    );
    let schedule =
        root_direct_schedule(witness_len, dummy_commit_params).expect("root-direct schedule");
    let [Step::Direct(step)] = schedule.steps.as_slice() else {
        panic!("root-direct schedule should contain one direct step");
    };
    assert_eq!(step.current_w_len, witness_len);
    assert_eq!(
        step.witness_shape,
        CleartextWitnessShape::FieldElements(witness_len)
    );
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
    let b = 1usize << lp.log_basis;

    let proof = AkitaLevelProof::Intermediate {
        extension_opening_reduction: None,
        v: RingVec::from_coeffs(vec![F::zero(); current_coeffs]),
        fold_grind_nonce: 0,
        stage1: dummy_stage1_proof(rounds, b),
        stage2: AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
            sumcheck_proof: dummy_sumcheck(rounds, 3),
            next_w_commitment: RingVec::from_coeffs(vec![F::zero(); next_commit_coeffs]),
            next_w_eval: F::zero(),
        }),
        stage3_sumcheck_proof: None,
    };
    Ok(proof.serialized_size(Compress::No))
}

#[test]
fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
    const D: usize = 64;
    let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
    let next_lp =
        LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, fold_challenge_config);
    let next_w_len = D * 8;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(0, 0, 1, 1, 0)
        .unwrap();
        assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ),
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
            SisModulusFamily::Q128,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(0, 0, 1, 1, 0)
        .unwrap();
        let rounds = sumcheck_rounds(D, next_w_len);

        let (final_witness, witness_shape) = segment_typed_final_witness(&lp, num_claims);
        let final_witness_bytes_runtime = final_witness.serialized_size(Compress::No);
        let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
            None,
            dummy_sumcheck(rounds, 3),
            final_witness,
            0,
        );

        // The planner accounts for the final witness separately
        // (`direct_witness_bytes` on the terminal direct step). Subtract
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
                num_claims,
                RelationMatrixRowLayout::WithoutDBlock,
            ),
            serialized_without_witness,
            "planned terminal-level bytes should match the serialized terminal body \
                 (less final_witness) at log_basis={log_basis}"
        );

        let scheduled_bytes = direct_witness_bytes(128, &witness_shape);
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
    let next_lp =
        LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, fold_challenge_config);
    let next_w_len = D * 8;

    for log_basis in 2..=6 {
        let lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            D,
            log_basis,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(0, 0, 1, 1, 0)
        .unwrap();
        let rounds = sumcheck_rounds(D, next_w_len);
        let b = 1usize << log_basis;
        let next_commitment = RingVec::from_ring_elems(&vec![
            CyclotomicRing::<F, D>::zero();
            next_lp.b_key.row_len()
        ])
        .into_compact();
        let level_proof = AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
            None,
            vec![CyclotomicRing::<F, D>::zero(); lp.d_key.row_len()],
            dummy_stage1_proof(rounds, b),
            dummy_sumcheck(rounds, 3),
            next_commitment,
            F::zero(),
        );
        let root_proof = AkitaBatchedRootProof::new(level_proof);

        assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ),
                root_proof.serialized_size(Compress::No),
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
fn group_batch_key_allows_precommitted_num_vars_above_main() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(24, 1),
            m_vars: 4,
            r_vars: 2,
            log_basis: 2,
            n_a: 3,
            conservative_n_b: 4,
        }],
    };

    multi_group_key
        .validate()
        .expect("precommitted groups larger than the main group are allowed");
}

#[test]
fn group_batch_key_allows_precommitted_num_vars_above_half_main() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(12, 1),
            m_vars: 4,
            r_vars: 2,
            log_basis: 2,
            n_a: 3,
            conservative_n_b: 4,
        }],
    };

    multi_group_key
        .validate()
        .expect("precommitted groups above half the main key are allowed");
}

#[test]
fn group_batch_key_allows_mixed_polynomial_counts() {
    let multi_group_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(20, 3),
        precommitteds: vec![PrecommittedGroupParams {
            group: PolynomialGroupLayout::new(10, 1),
            m_vars: 12,
            r_vars: 2,
            log_basis: 2,
            n_a: 3,
            conservative_n_b: 4,
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
        m_vars: 4,
        r_vars: 2,
        log_basis: 2,
        n_a: 3,
        conservative_n_b: 4,
    };
    let err = layout
        .validate_frozen_precommit(64, 2)
        .expect_err("geometry must match num_vars");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}
