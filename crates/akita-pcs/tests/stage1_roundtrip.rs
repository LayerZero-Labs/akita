#![allow(missing_docs)]

use akita_field::Prime128Offset275;
use akita_prover::DigitRangeProver;
use akita_transcript::{labels, AkitaTranscript};
use akita_types::{AkitaStage1Proof, DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
use akita_verifier::AkitaStage1Verifier;

type F = Prime128Offset275;

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
    let (proof, stage1_point) = prover
        .prove(&mut prover_transcript)
        .expect("stage1 proof should succeed");
    (proof, stage1_point, equality_point)
}

fn assert_stage1_roundtrip(
    b: usize,
    live_x_cols: usize,
    tau0: Vec<F>,
    expected_child_claim_counts: &[usize],
) {
    let (proof, stage1_point, equality_point) = prove_stage1_case(b, live_x_cols, tau0);

    let verifier = AkitaStage1Verifier::new(equality_point, DigitRangePlan::new(b).unwrap());
    let mut verifier_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let verified_point = verifier
        .verify(&proof, &mut verifier_transcript)
        .expect("stage1 verification should succeed");

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
    let verifier = AkitaStage1Verifier::new(equality_point, plan);
    let mut transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    assert!(verifier.verify(proof, &mut transcript).is_err());
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
