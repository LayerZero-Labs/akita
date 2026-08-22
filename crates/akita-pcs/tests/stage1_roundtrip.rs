#![allow(missing_docs)]

use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{
    CanonicalBytes, CanonicalField, Ext2, ExtField, FieldCore, FpExt4, FromPrimitiveInt,
    Prime128Offset275, Prime32Offset99, Prime64Offset59, TranscriptChallenge,
};
use akita_prover::DigitRangeProver;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::multilinear_eval;
use akita_transcript::{labels, AkitaTranscript};
use akita_types::{
    AkitaStage1Proof, DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain, GrindingPlan,
    GrindingRun, GrindingSite, ProverGrindingTranscript, SumcheckProtocol, TranscriptNonceStream,
    VerifierGrindingTranscript,
};
use akita_verifier::AkitaStage1Verifier;

type F = Prime128Offset275;

fn stage1_grinding_plan(plan: DigitRangePlan, rounds: usize) -> GrindingPlan {
    let product_stages = plan.product_stage_arities().len();
    let mut runs = Vec::new();
    for stage in 0..=product_stages {
        for round in 0..rounds {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::SumcheckRound {
                        protocol: SumcheckProtocol::Stage1,
                        level: 0,
                        stage: u32::try_from(stage).unwrap(),
                        round: u32::try_from(round).unwrap(),
                    },
                    1,
                    128,
                )
                .unwrap(),
            );
        }
        if stage < product_stages {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::Stage1InterstageBatch {
                        level: 0,
                        stage: u32::try_from(stage).unwrap(),
                    },
                    1,
                    128,
                )
                .unwrap(),
            );
        }
    }
    GrindingPlan::new(runs, 128).unwrap()
}

fn empty_nonce_stream() -> TranscriptNonceStream {
    TranscriptNonceStream::from_bytes(Vec::new(), 0).unwrap()
}

fn sample_stage1_witness(b: usize, live_x_cols: usize, ring_bits: usize) -> Vec<i8> {
    let half = (b / 2) as i16;
    let y_len = 1usize << ring_bits;
    (0..live_x_cols * y_len)
        .map(|idx| {
            (idx as i16 % half)
                .try_into()
                .expect("test digit should fit in i8")
        })
        .collect()
}

fn prove_stage1_case(
    b: usize,
    live_x_cols: usize,
    tau0: Vec<F>,
) -> (AkitaStage1Proof<F>, Vec<F>, DigitRangeEqualityPoint<F>) {
    let col_bits = 3;
    let ring_bits = 1;
    let witness = sample_stage1_witness(b, live_x_cols, ring_bits);
    let equality_point =
        DigitRangeEqualityPoint::from_column_then_ring_challenges(&tau0, col_bits, ring_bits)
            .unwrap();
    let domain = FlatBooleanDomain::new(witness.len(), col_bits + ring_bits).unwrap();

    let prover = DigitRangeProver::new(
        std::sync::Arc::from(witness),
        DigitRangePlan::new(b).unwrap(),
        domain,
        equality_point.clone(),
    )
    .expect("stage1 prover should build");
    let mut prover_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let grinding_plan = stage1_grinding_plan(DigitRangePlan::new(b).unwrap(), tau0.len());
    let mut grinding =
        ProverGrindingTranscript::new(&mut prover_transcript, &grinding_plan).unwrap();
    let (proof, stage1_point) = prover
        .prove(&mut grinding, None, 0)
        .expect("stage1 proof should succeed");
    grinding.finish().unwrap();
    (proof, stage1_point, equality_point)
}

fn assert_stage1_roundtrip(
    b: usize,
    live_x_cols: usize,
    tau0: Vec<F>,
    expected_child_claim_counts: &[usize],
) {
    let (proof, stage1_point, equality_point) = prove_stage1_case(b, live_x_cols, tau0);

    let rounds = equality_point.coordinates().len();
    let plan = DigitRangePlan::new(b).unwrap();
    let verifier = AkitaStage1Verifier::new(equality_point, plan);
    let mut verifier_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let grinding_plan = stage1_grinding_plan(plan, rounds);
    let stream = empty_nonce_stream();
    let mut grinding =
        VerifierGrindingTranscript::new(&mut verifier_transcript, &stream, &grinding_plan).unwrap();
    let verified_point = verifier
        .verify(&proof, &mut grinding, 0)
        .expect("stage1 verification should succeed");
    grinding.finish().unwrap();

    assert_eq!(stage1_point, verified_point);
    assert_eq!(proof.stages.len(), expected_child_claim_counts.len());
    for (stage, &expected_child_claims) in proof.stages.iter().zip(expected_child_claim_counts) {
        assert_eq!(stage.child_claims.len(), expected_child_claims);
    }
}

fn assert_stage1_rejected(
    proof: &AkitaStage1Proof<F>,
    equality_point: DigitRangeEqualityPoint<F>,
    plan: DigitRangePlan,
) {
    let rounds = equality_point.coordinates().len();
    let verifier = AkitaStage1Verifier::new(equality_point, plan);
    let mut transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let grinding_plan = stage1_grinding_plan(plan, rounds);
    let stream = empty_nonce_stream();
    let mut grinding =
        VerifierGrindingTranscript::new(&mut transcript, &stream, &grinding_plan).unwrap();
    assert!(verifier.verify(proof, &mut grinding, 0).is_err());
}

#[test]
fn streaming_high_basis_handles_odd_live_prefix_without_materializing_padding() {
    let point = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(9),
    ];
    for basis in [16, 32, 64] {
        let witness = sample_stage1_witness(basis, 5, 0);
        let equality_point =
            DigitRangeEqualityPoint::from_column_then_ring_challenges(&point, 4, 0).unwrap();
        let domain = FlatBooleanDomain::new(witness.len(), 4).unwrap();
        let plan = DigitRangePlan::new(basis).unwrap();
        let prover = DigitRangeProver::new(
            std::sync::Arc::from(witness),
            plan,
            domain,
            equality_point.clone(),
        )
        .unwrap();
        let mut prover_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let grinding_plan = stage1_grinding_plan(plan, point.len());
        let mut grinding =
            ProverGrindingTranscript::new(&mut prover_transcript, &grinding_plan).unwrap();
        let (proof, expected_point) = prover.prove(&mut grinding, None, 0).unwrap();
        grinding.finish().unwrap();

        let verifier = AkitaStage1Verifier::new(equality_point, plan);
        let mut verifier_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let stream = empty_nonce_stream();
        let mut grinding =
            VerifierGrindingTranscript::new(&mut verifier_transcript, &stream, &grinding_plan)
                .unwrap();
        assert_eq!(
            verifier.verify(&proof, &mut grinding, 0).unwrap(),
            expected_point,
            "basis {basis}"
        );
        grinding.finish().unwrap();
    }
}

fn assert_high_basis_extension_roundtrips<Base, E>()
where
    Base: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
    E: FieldCore
        + ExtField<Base>
        + FromPrimitiveInt
        + HasOptimizedFold
        + HasUnreducedOps
        + AkitaSerialize,
{
    let raw_challenges = [3, 5, 7, 9].map(E::from_u64);
    for basis in [16, 32, 64] {
        let half = i8::try_from(basis / 2).unwrap();
        let witness = (0..5)
            .map(|index| i8::try_from(index % basis).unwrap() - half)
            .collect::<Vec<_>>();
        let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
            &raw_challenges,
            raw_challenges.len(),
            0,
        )
        .unwrap();
        let domain = FlatBooleanDomain::new(witness.len(), raw_challenges.len()).unwrap();
        let plan = DigitRangePlan::new(basis).unwrap();
        let prover = DigitRangeProver::new(
            std::sync::Arc::from(witness),
            plan,
            domain,
            equality_point.clone(),
        )
        .unwrap();
        let mut prover_transcript = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let grinding_plan = stage1_grinding_plan(plan, raw_challenges.len());
        let mut grinding =
            ProverGrindingTranscript::new(&mut prover_transcript, &grinding_plan).unwrap();
        let (proof, expected_point) = prover.prove(&mut grinding, None, 0).unwrap();
        grinding.finish().unwrap();

        let verifier = AkitaStage1Verifier::new(equality_point, plan);
        let mut verifier_transcript = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let stream = empty_nonce_stream();
        let mut grinding =
            VerifierGrindingTranscript::new(&mut verifier_transcript, &stream, &grinding_plan)
                .unwrap();
        assert_eq!(
            verifier.verify(&proof, &mut grinding, 0).unwrap(),
            expected_point,
            "basis {basis}"
        );
        grinding.finish().unwrap();
    }
}

#[test]
fn high_basis_roundtrip_covers_delayed_fp32_extension_accumulation() {
    assert_high_basis_extension_roundtrips::<Prime32Offset99, FpExt4<Prime32Offset99>>();
}

#[test]
fn high_basis_roundtrip_covers_fp64_extension_accumulation() {
    assert_high_basis_extension_roundtrips::<Prime64Offset59, Ext2<Prime64Offset59>>();
}

#[test]
fn high_basis_final_range_image_matches_dense_padding_oracle_at_every_prefix() {
    let raw_challenges = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    for basis in [16, 32, 64] {
        let half = i8::try_from(basis / 2).unwrap();
        for live_len in 1..=8 {
            let witness = (0..live_len)
                .map(|index| i8::try_from((index * 5 + 3) % basis).unwrap() - half)
                .collect::<Vec<_>>();
            let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
                &raw_challenges,
                raw_challenges.len(),
                0,
            )
            .unwrap();
            let domain = FlatBooleanDomain::new(live_len, raw_challenges.len()).unwrap();
            let plan = DigitRangePlan::new(basis).unwrap();
            let prover = DigitRangeProver::new(
                std::sync::Arc::from(witness.as_slice()),
                plan,
                domain,
                equality_point.clone(),
            )
            .unwrap();
            let mut prover_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
            let grinding_plan = stage1_grinding_plan(plan, raw_challenges.len());
            let mut grinding =
                ProverGrindingTranscript::new(&mut prover_transcript, &grinding_plan).unwrap();
            let (proof, stage1_point) = prover.prove(&mut grinding, None, 0).unwrap();
            grinding.finish().unwrap();

            let verifier = AkitaStage1Verifier::new(equality_point, plan);
            let mut verifier_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
            let stream = empty_nonce_stream();
            let mut grinding =
                VerifierGrindingTranscript::new(&mut verifier_transcript, &stream, &grinding_plan)
                    .unwrap();
            assert_eq!(
                verifier.verify(&proof, &mut grinding, 0).unwrap(),
                stage1_point
            );
            grinding.finish().unwrap();

            let mut dense_range_image = witness
                .iter()
                .map(|&digit| {
                    let digit = i64::from(digit);
                    F::from_i64(digit * (digit + 1))
                })
                .collect::<Vec<_>>();
            dense_range_image.resize(8, F::zero());
            assert_eq!(
                proof.range_image_evaluation,
                multilinear_eval(&dense_range_image, &stage1_point).unwrap(),
                "basis={basis}, live_len={live_len}"
            );
        }
    }
}

#[test]
fn stage1_verifier_rejects_every_malformed_plan_shape_without_panicking() {
    for basis in [4, 8, 16, 32, 64] {
        let transcript_point = vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(9),
        ];
        let (proof, _, equality_point) = prove_stage1_case(basis, 6, transcript_point);
        let plan = DigitRangePlan::new(basis).unwrap();

        let mut missing_stage = proof.clone();
        missing_stage.stages.pop();
        assert_stage1_rejected(&missing_stage, equality_point.clone(), plan);

        let mut extra_stage = proof.clone();
        extra_stage.stages.push(extra_stage.stages[0].clone());
        assert_stage1_rejected(&extra_stage, equality_point.clone(), plan);

        for stage_index in 0..proof.stages.len() {
            let mut missing_round = proof.clone();
            missing_round.stages[stage_index]
                .sumcheck_proof
                .round_polys
                .pop();
            assert_stage1_rejected(&missing_round, equality_point.clone(), plan);

            let mut extra_round = proof.clone();
            let extra = extra_round.stages[stage_index].sumcheck_proof.round_polys[0].clone();
            extra_round.stages[stage_index]
                .sumcheck_proof
                .round_polys
                .push(extra);
            assert_stage1_rejected(&extra_round, equality_point.clone(), plan);

            let mut degree_too_low = proof.clone();
            degree_too_low.stages[stage_index]
                .sumcheck_proof
                .round_polys[0]
                .coeffs_except_linear_term
                .pop();
            assert_stage1_rejected(&degree_too_low, equality_point.clone(), plan);

            let mut degree_too_high = proof.clone();
            degree_too_high.stages[stage_index]
                .sumcheck_proof
                .round_polys[0]
                .coeffs_except_linear_term
                .push(F::from_u64(0));
            assert_stage1_rejected(&degree_too_high, equality_point.clone(), plan);

            let mut wrong_child_count = proof.clone();
            if wrong_child_count.stages[stage_index]
                .child_claims
                .is_empty()
            {
                wrong_child_count.stages[stage_index]
                    .child_claims
                    .push(F::from_u64(0));
            } else {
                wrong_child_count.stages[stage_index].child_claims.pop();
            }
            assert_stage1_rejected(&wrong_child_count, equality_point.clone(), plan);
        }

        let short_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
            equality_point.coordinates().get(..3).unwrap(),
            2,
            1,
        )
        .unwrap();
        assert_stage1_rejected(&proof, short_point, plan);
    }
}

#[test]
fn stage1_prover_verifier_roundtrip_covers_compact_and_tree_bases() {
    assert_stage1_roundtrip(
        4,
        5,
        vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(9),
        ],
        &[0],
    );
    assert_stage1_roundtrip(
        8,
        5,
        vec![
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ],
        &[0],
    );
    assert_stage1_roundtrip(
        16,
        6,
        vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(9),
        ],
        &[2, 0],
    );
    assert_stage1_roundtrip(
        32,
        5,
        vec![
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ],
        &[4, 0],
    );
    assert_stage1_roundtrip(
        64,
        5,
        vec![
            F::from_u64(23),
            F::from_u64(29),
            F::from_u64(31),
            F::from_u64(37),
        ],
        &[2, 8, 0],
    );
}
