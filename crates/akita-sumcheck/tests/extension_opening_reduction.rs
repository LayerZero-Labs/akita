#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_field::{Ext2, ExtField, FieldCore, Prime128Offset275, Prime64Offset59};
use akita_sumcheck::{
    check_extension_opening_reduction_output, extension_opening_reduction_claim,
    extension_opening_reduction_eval_at_point, prove_extension_opening_reduction, prove_sumcheck,
    tensor_equality_factor_eval_at_point, tensor_equality_factor_evals,
    tensor_logical_claim_from_partials, tensor_packed_witness_evals,
    tensor_partials_from_base_evals, tensor_reduction_claim_from_rows,
    verify_extension_opening_reduction_rounds, verify_sumcheck,
    BatchedExtensionOpeningReductionProver, BatchedExtensionOpeningReductionTerm,
    ExtensionOpeningFactorTerm, ExtensionOpeningReductionFactor, ExtensionOpeningReductionProver,
    ExtensionOpeningReductionVerifier, SumcheckInstanceProver, EXTENSION_OPENING_REDUCTION_DEGREE,
};
use akita_transcript::labels as tr_labels;
use akita_transcript::{Blake2bTranscript, Transcript};

type F = Prime128Offset275;

fn new_transcript() -> Blake2bTranscript<F> {
    <Blake2bTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_round(tr: &mut Blake2bTranscript<F>) -> F {
    tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
}

fn lifted_multilinear_eval<B, E>(evals: &[B], point: &[E]) -> E
where
    B: FieldCore,
    E: ExtField<B>,
{
    let mut layer = evals.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    for &r in point {
        let one_minus_r = E::one() - r;
        let next_len = layer.len() / 2;
        for idx in 0..next_len {
            layer[idx] = layer[2 * idx] * one_minus_r + layer[2 * idx + 1] * r;
        }
        layer.truncate(next_len);
    }
    layer[0]
}

#[test]
fn tensor_partials_recompose_logical_extension_opening() {
    type B = Prime64Offset59;
    type E = Ext2<B>;

    let num_vars = 4;
    let base_evals = (0..(1usize << num_vars))
        .map(|idx| B::from_u64((17 * idx as u64 + 9) % 127))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[B::from_u64(idx as u64 + 3), B::from_u64(5 * idx as u64 + 2)])
        })
        .collect::<Vec<_>>();

    let partials = tensor_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    assert_eq!(
        partials.column_partials.len(),
        <E as ExtField<B>>::EXT_DEGREE
    );
    assert_eq!(partials.row_partials.len(), <E as ExtField<B>>::EXT_DEGREE);

    let logical_claim =
        tensor_logical_claim_from_partials::<B, E>(&point, &partials.column_partials).unwrap();
    assert_eq!(logical_claim, lifted_multilinear_eval(&base_evals, &point));
    akita_sumcheck::check_tensor_extension_opening_claim::<B, E>(
        &point,
        logical_claim,
        &partials.column_partials,
    )
    .unwrap();

    assert!(matches!(
        akita_sumcheck::check_tensor_extension_opening_claim::<B, E>(
            &point,
            logical_claim + E::one(),
            &partials.column_partials,
        ),
        Err(akita_field::AkitaError::InvalidProof)
    ));
}

#[test]
fn tensor_row_reduction_matches_dense_sumcheck_claim() {
    type B = Prime64Offset59;
    type E = Ext2<B>;

    let num_vars = 4;
    let base_evals = (0..(1usize << num_vars))
        .map(|idx| B::from_u64((23 * idx as u64 + 11) % 131))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|idx| {
            E::from_base_slice(&[
                B::from_u64(3 * idx as u64 + 4),
                B::from_u64(7 * idx as u64 + 1),
            ])
        })
        .collect::<Vec<_>>();
    let eta = vec![E::from_base_slice(&[B::from_u64(19), B::from_u64(29)])];

    let packed_witness = tensor_packed_witness_evals::<B, E>(num_vars, &base_evals).unwrap();
    let partials = tensor_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    let row_claim = tensor_reduction_claim_from_rows::<B, E>(&partials.row_partials, &eta).unwrap();
    let factor_evals = tensor_equality_factor_evals::<B, E>(&point[1..], &eta).unwrap();

    assert_eq!(packed_witness.len(), factor_evals.len());
    assert_eq!(
        extension_opening_reduction_claim(&packed_witness, &factor_evals).unwrap(),
        row_claim
    );

    let rho = vec![
        E::from_base_slice(&[B::from_u64(31), B::from_u64(37)]),
        E::from_base_slice(&[B::from_u64(41), B::from_u64(43)]),
        E::from_base_slice(&[B::from_u64(47), B::from_u64(53)]),
    ];
    assert_eq!(
        akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap(),
        tensor_equality_factor_eval_at_point::<B, E>(&point[1..], &eta, &rho).unwrap()
    );
}

#[test]
fn singleton_factor_claim_matches_multilinear_opening() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((11 * i + 4) as u64)).collect();
    let opening_point = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
    let factor = ExtensionOpeningReductionFactor::singleton(opening_point.clone()).unwrap();

    let claim = factor.claim_for_witness(&witness_evals).unwrap();
    let expected = akita_sumcheck::multilinear_eval(&witness_evals, &opening_point).unwrap();
    assert_eq!(claim, expected);

    let rho = vec![F::from_u64(2), F::from_u64(9), F::from_u64(6)];
    let factor_evals = factor.evals();
    let folded_factor = akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap();
    assert_eq!(folded_factor, factor.evaluate(&rho).unwrap());
}

#[test]
fn row_factor_batches_multiple_opening_points() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((5 * i + 8) as u64)).collect();
    let point_a = vec![
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
        F::from_u64(5),
    ];
    let point_b = vec![
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
    ];
    let coeff_a = F::from_u64(19);
    let coeff_b = F::from_u64(23);
    let factor = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(point_a.clone(), coeff_a),
        ExtensionOpeningFactorTerm::new(point_b.clone(), coeff_b),
    ])
    .unwrap();

    assert_eq!(factor.num_vars(), 4);
    assert_eq!(factor.terms().len(), 2);
    let claim = factor.claim_for_witness(&witness_evals).unwrap();
    let expected = coeff_a * akita_sumcheck::multilinear_eval(&witness_evals, &point_a).unwrap()
        + coeff_b * akita_sumcheck::multilinear_eval(&witness_evals, &point_b).unwrap();
    assert_eq!(claim, expected);

    let rho = vec![
        F::from_u64(29),
        F::from_u64(31),
        F::from_u64(37),
        F::from_u64(41),
    ];
    let factor_evals = factor.evals();
    assert_eq!(
        akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap(),
        factor.evaluate(&rho).unwrap()
    );
}

#[test]
fn factor_rejects_malformed_shapes() {
    let err = ExtensionOpeningReductionFactor::<F>::from_terms(Vec::new()).unwrap_err();
    assert!(matches!(err, akita_field::AkitaError::InvalidInput(_)));

    let err = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(vec![F::one(), F::zero()], F::one()),
        ExtensionOpeningFactorTerm::new(vec![F::one()], F::one()),
    ])
    .unwrap_err();
    assert!(matches!(err, akita_field::AkitaError::InvalidSize { .. }));
}

#[test]
fn extension_opening_reduction_proves_witness_factor_claim() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((3 * i + 5) as u64)).collect();
    let factor_evals: Vec<F> = (0..16).map(|i| F::from_u64((7 * i + 11) as u64)).collect();
    let expected_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals).unwrap();

    let mut prover =
        ExtensionOpeningReductionProver::new(witness_evals.clone(), factor_evals.clone()).unwrap();
    assert_eq!(prover.degree_bound(), EXTENSION_OPENING_REDUCTION_DEGREE);
    assert_eq!(prover.input_claim(), expected_claim);

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, sample_round).unwrap();

    let (final_witness, final_factor) = prover.final_witness_and_factor_evals().unwrap();
    assert_eq!(final_claim, final_witness * final_factor);
    assert_eq!(
        final_claim,
        extension_opening_reduction_eval_at_point(&witness_evals, &factor_evals, &challenges)
            .unwrap()
    );

    let verifier = ExtensionOpeningReductionVerifier::new(witness_evals, factor_evals).unwrap();
    let mut verifier_transcript = new_transcript();
    let verified_challenges =
        verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, sample_round)
            .unwrap();
    assert_eq!(verified_challenges, challenges);
}

#[test]
fn batched_extension_opening_reduction_uses_one_common_rho() {
    let witness_a: Vec<F> = (0..16).map(|i| F::from_u64((3 * i + 5) as u64)).collect();
    let factor_a: Vec<F> = (0..16).map(|i| F::from_u64((7 * i + 11) as u64)).collect();
    let witness_b: Vec<F> = (0..16).map(|i| F::from_u64((13 * i + 17) as u64)).collect();
    let factor_b: Vec<F> = (0..16).map(|i| F::from_u64((19 * i + 23) as u64)).collect();
    let coeff_a = F::from_u64(29);
    let coeff_b = F::from_u64(31);
    let expected_claim = coeff_a
        * extension_opening_reduction_claim(&witness_a, &factor_a).unwrap()
        + coeff_b * extension_opening_reduction_claim(&witness_b, &factor_b).unwrap();

    let terms = vec![
        BatchedExtensionOpeningReductionTerm::new(witness_a.clone(), factor_a.clone(), coeff_a)
            .unwrap(),
        BatchedExtensionOpeningReductionTerm::new(witness_b.clone(), factor_b.clone(), coeff_b)
            .unwrap(),
    ];
    let mut prover = BatchedExtensionOpeningReductionProver::new(terms).unwrap();
    assert_eq!(prover.input_claim(), expected_claim);
    assert_eq!(prover.degree_bound(), EXTENSION_OPENING_REDUCTION_DEGREE);

    let mut transcript = new_transcript();
    let (_proof, challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript, sample_round).unwrap();
    let expected_final = prover
        .final_terms()
        .unwrap()
        .into_iter()
        .fold(F::zero(), |acc, (coeff, witness, factor)| {
            acc + coeff * witness * factor
        });
    assert_eq!(final_claim, expected_final);
    assert_eq!(
        final_claim,
        coeff_a
            * extension_opening_reduction_eval_at_point(&witness_a, &factor_a, &challenges)
                .unwrap()
            + coeff_b
                * extension_opening_reduction_eval_at_point(&witness_b, &factor_b, &challenges)
                    .unwrap()
    );
}

#[test]
fn extension_opening_reduction_proves_transparent_factor_claim() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((3 * i + 5) as u64)).collect();
    let factor = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(
            vec![
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
                F::from_u64(5),
            ],
            F::from_u64(7),
        ),
        ExtensionOpeningFactorTerm::new(
            vec![
                F::from_u64(11),
                F::from_u64(13),
                F::from_u64(17),
                F::from_u64(19),
            ],
            F::from_u64(23),
        ),
    ])
    .unwrap();
    let factor_evals = factor.evals();
    let expected_claim = factor.claim_for_witness(&witness_evals).unwrap();

    let mut prover =
        ExtensionOpeningReductionProver::new(witness_evals.clone(), factor_evals.clone()).unwrap();
    assert_eq!(prover.input_claim(), expected_claim);

    let mut prover_transcript = new_transcript();
    let (proof, prover_result) = prove_extension_opening_reduction::<F, _, F, _>(
        &mut prover,
        &mut prover_transcript,
        sample_round,
    )
    .unwrap();
    let (final_witness, final_factor) = prover.final_witness_and_factor_evals().unwrap();
    assert_eq!(
        final_factor,
        factor.evaluate(&prover_result.challenges).unwrap()
    );
    check_extension_opening_reduction_output(
        prover_result.final_claim,
        final_witness,
        final_factor,
    )
    .unwrap();

    let verifier = ExtensionOpeningReductionVerifier::new(witness_evals, factor_evals).unwrap();
    let mut verifier_transcript = new_transcript();
    let verified_challenges =
        verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, sample_round)
            .unwrap();
    assert_eq!(verified_challenges, prover_result.challenges);
}

#[test]
fn detached_verifier_checks_transparent_factor_against_opened_witness() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((17 * i + 3) as u64)).collect();
    let factor = ExtensionOpeningReductionFactor::singleton(vec![
        F::from_u64(2),
        F::from_u64(5),
        F::from_u64(11),
    ])
    .unwrap();
    let factor_evals = factor.evals();
    let input_claim = factor.claim_for_witness(&witness_evals).unwrap();

    let mut prover =
        ExtensionOpeningReductionProver::new(witness_evals.clone(), factor_evals).unwrap();
    let mut prover_transcript = new_transcript();
    let (proof, prover_result) = prove_extension_opening_reduction::<F, _, F, _>(
        &mut prover,
        &mut prover_transcript,
        sample_round,
    )
    .unwrap();

    let mut verifier_transcript = new_transcript();
    let verifier_result = verify_extension_opening_reduction_rounds::<F, _, F, _>(
        &proof,
        input_claim,
        factor.num_vars(),
        &mut verifier_transcript,
        sample_round,
    )
    .unwrap();
    assert_eq!(verifier_result, prover_result);

    let opened_witness =
        akita_sumcheck::multilinear_eval(&witness_evals, &verifier_result.challenges).unwrap();
    let factor_eval = factor.evaluate(&verifier_result.challenges).unwrap();
    check_extension_opening_reduction_output(
        verifier_result.final_claim,
        opened_witness,
        factor_eval,
    )
    .unwrap();

    assert!(matches!(
        check_extension_opening_reduction_output(
            verifier_result.final_claim + F::one(),
            opened_witness,
            factor_eval,
        ),
        Err(akita_field::AkitaError::InvalidProof)
    ));
}

#[test]
fn extension_opening_reduction_rejects_wrong_final_oracle() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((i + 1) as u64)).collect();
    let factor_evals: Vec<F> = (0..8).map(|i| F::from_u64((2 * i + 9) as u64)).collect();

    let mut prover =
        ExtensionOpeningReductionProver::new(witness_evals.clone(), factor_evals).unwrap();
    let mut prover_transcript = new_transcript();
    let (proof, _, _) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, sample_round).unwrap();

    let bad_factor_evals: Vec<F> = (0..8).map(|i| F::from_u64((2 * i + 10) as u64)).collect();
    let verifier = ExtensionOpeningReductionVerifier::new(witness_evals, bad_factor_evals).unwrap();
    let mut verifier_transcript = new_transcript();
    let err =
        verify_sumcheck::<F, _, F, _, _>(&proof, &verifier, &mut verifier_transcript, sample_round)
            .unwrap_err();
    assert!(matches!(err, akita_field::AkitaError::InvalidProof));
}

#[test]
fn extension_opening_reduction_detached_round_verifier_returns_final_claim() {
    let witness_evals: Vec<F> = (0..4).map(|i| F::from_u64((5 * i + 1) as u64)).collect();
    let factor_evals: Vec<F> = (0..4).map(|i| F::from_u64((13 * i + 2) as u64)).collect();
    let mut prover =
        ExtensionOpeningReductionProver::new(witness_evals.clone(), factor_evals.clone()).unwrap();

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut prover_transcript, sample_round).unwrap();

    let mut verifier_transcript = new_transcript();
    verifier_transcript.append_serde(
        tr_labels::ABSORB_SUMCHECK_CLAIM,
        &proof_claim(&witness_evals, &factor_evals),
    );
    let (detached_final_claim, detached_challenges) = proof
        .verify::<F, _, _>(
            proof_claim(&witness_evals, &factor_evals),
            challenges.len(),
            EXTENSION_OPENING_REDUCTION_DEGREE,
            &mut verifier_transcript,
            sample_round,
        )
        .unwrap();

    assert_eq!(detached_challenges, challenges);
    assert_eq!(detached_final_claim, final_claim);
}

#[test]
fn extension_opening_reduction_rejects_malformed_table_lengths() {
    let witness_evals = vec![F::one(), F::from_u64(2), F::from_u64(3)];
    let factor_evals = vec![F::one(), F::from_u64(2), F::from_u64(3)];
    assert!(ExtensionOpeningReductionProver::new(witness_evals, factor_evals).is_err());

    let witness_evals = vec![F::one(), F::from_u64(2)];
    let factor_evals = vec![F::one()];
    assert!(ExtensionOpeningReductionVerifier::new(witness_evals, factor_evals).is_err());
}

fn proof_claim(witness_evals: &[F], factor_evals: &[F]) -> F {
    extension_opening_reduction_claim(witness_evals, factor_evals).unwrap()
}
