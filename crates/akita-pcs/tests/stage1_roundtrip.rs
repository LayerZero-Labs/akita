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

#[test]
fn stage1_verifier_rejects_malformed_plan_shapes_without_panicking() {
    let basis = 16;
    let transcript_point = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(9),
    ];
    let (proof, _, equality_point) = prove_stage1_case(basis, 6, transcript_point);
    let plan = DigitRangePlan::new(basis).unwrap();

    let mut missing_child = proof.clone();
    missing_child.stages[0].child_claims.pop();
    let verifier = AkitaStage1Verifier::new(equality_point.clone(), plan);
    let mut transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    assert!(verifier.verify(&missing_child, &mut transcript).is_err());

    let mut wrong_degree = proof.clone();
    wrong_degree.stages[0].sumcheck_proof.round_polys[0]
        .coeffs_except_linear_term
        .pop();
    let verifier = AkitaStage1Verifier::new(equality_point.clone(), plan);
    let mut transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    assert!(verifier.verify(&wrong_degree, &mut transcript).is_err());

    let mut extra_stage = proof;
    extra_stage.stages.push(extra_stage.stages[0].clone());
    let verifier = AkitaStage1Verifier::new(equality_point, plan);
    let mut transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    assert!(verifier.verify(&extra_stage, &mut transcript).is_err());
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
