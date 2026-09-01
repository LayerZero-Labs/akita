#![allow(missing_docs)]

use akita_algebra::poly::multilinear_eval;
use akita_error::AkitaError;
use akita_prover::protocol::extension_opening_reduction::{
    ExtensionOpeningReductionGroup, ExtensionOpeningReductionProver, ExtensionOpeningReductionTerm,
};
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::labels as tr_labels;
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    check_extension_opening_reduction_output, derive_tensor_extension_opening_claim_from_partials,
    extension_opening_reduction_claim, extension_opening_reduction_eval_at_point,
    tensor_column_partials_from_base_evals, tensor_equality_factor_eval_at_point,
    tensor_equality_factor_evals, tensor_packed_witness_evals, tensor_reduction_claim_from_rows,
    tensor_row_partials_from_columns, ExtensionOpeningFactorTerm, ExtensionOpeningReductionFactor,
    ExtensionOpeningReductionRoundResult, ExtensionOpeningTensorPartials,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};
use jolt_field::{Ext2, ExtField, Field, One, Prime128Offset275, Prime64Offset59, Ring, Zero};

type F = Prime128Offset275;

fn eor_group<E: Field>(
    witness: Vec<E>,
    factor: Vec<E>,
    coeff: E,
) -> Result<ExtensionOpeningReductionGroup<E>, AkitaError> {
    ExtensionOpeningReductionGroup::new(
        vec![ExtensionOpeningReductionTerm::new(witness, coeff)],
        factor,
    )
}

fn new_transcript() -> AkitaTranscript<F> {
    <AkitaTranscript<F> as akita_transcript::TranscriptFactory<F>>::new(
        tr_labels::DOMAIN_AKITA_PROTOCOL,
    )
}

fn sample_round(tr: &mut AkitaTranscript<F>) -> Result<F, AkitaError> {
    Ok(tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND))
}

fn verify_eor_rounds(
    input_claim: F,
    num_rounds: usize,
    proof: &SumcheckProof<F>,
    transcript: &mut AkitaTranscript<F>,
) -> Result<ExtensionOpeningReductionRoundResult<F>, AkitaError> {
    transcript.append_serde(tr_labels::ABSORB_SUMCHECK_CLAIM, &input_claim);
    let (final_claim, challenges) = proof.verify::<F, _, _>(
        input_claim,
        num_rounds,
        EXTENSION_OPENING_REDUCTION_DEGREE,
        transcript,
        sample_round,
    )?;
    Ok(ExtensionOpeningReductionRoundResult {
        final_claim,
        challenges,
    })
}

fn verify_eor_full(
    witness_evals: &[F],
    factor_evals: &[F],
    proof: &SumcheckProof<F>,
) -> Result<Vec<F>, AkitaError> {
    let input_claim = extension_opening_reduction_claim(witness_evals, factor_evals)?;
    let mut transcript = new_transcript();
    let result = verify_eor_rounds(
        input_claim,
        witness_evals.len().trailing_zeros() as usize,
        proof,
        &mut transcript,
    )?;
    let expected =
        extension_opening_reduction_eval_at_point(witness_evals, factor_evals, &result.challenges)?;
    if result.final_claim != expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(result.challenges)
}

fn lifted_multilinear_eval<B, E>(evals: &[B], point: &[E]) -> E
where
    B: Field,
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

    let column_partials =
        tensor_column_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    let row_partials = tensor_row_partials_from_columns::<B, E>(&column_partials).unwrap();
    let partials = ExtensionOpeningTensorPartials {
        column_partials,
        row_partials,
    };
    assert_eq!(partials.column_partials.len(), <E as ExtField<B>>::DEGREE);
    assert_eq!(partials.row_partials.len(), <E as ExtField<B>>::DEGREE);

    let logical_claim = derive_tensor_extension_opening_claim_from_partials::<B, E>(
        &point,
        &partials.column_partials,
    )
    .unwrap();
    assert_eq!(logical_claim, lifted_multilinear_eval(&base_evals, &point));
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
    let column_partials =
        tensor_column_partials_from_base_evals::<B, E>(num_vars, &base_evals, &point).unwrap();
    let row_partials = tensor_row_partials_from_columns::<B, E>(&column_partials).unwrap();
    let partials = ExtensionOpeningTensorPartials {
        column_partials,
        row_partials,
    };
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
    let factor_evals = factor.evals().unwrap();
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
    let factor_evals = factor.evals().unwrap();
    assert_eq!(
        akita_sumcheck::multilinear_eval(&factor_evals, &rho).unwrap(),
        factor.evaluate(&rho).unwrap()
    );
}

#[test]
fn factor_rejects_malformed_shapes() {
    let err = ExtensionOpeningReductionFactor::<F>::from_terms(Vec::new()).unwrap_err();
    assert!(matches!(err, akita_error::AkitaError::InvalidInput(_)));

    let err = ExtensionOpeningReductionFactor::from_terms(vec![
        ExtensionOpeningFactorTerm::new(vec![F::one(), F::zero()], F::one()),
        ExtensionOpeningFactorTerm::new(vec![F::one()], F::one()),
    ])
    .unwrap_err();
    assert!(matches!(err, akita_error::AkitaError::InvalidSize { .. }));
}

#[test]
fn extension_opening_reduction_proves_witness_factor_claim() {
    let witness_evals: Vec<F> = (0..16).map(|i| F::from_u64((3 * i + 5) as u64)).collect();
    let factor_evals: Vec<F> = (0..16).map(|i| F::from_u64((7 * i + 11) as u64)).collect();
    let expected_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals).unwrap();

    let group = eor_group(witness_evals.clone(), factor_evals.clone(), F::one()).unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![group], expected_claim).unwrap();
    assert_eq!(prover.degree_bound(), EXTENSION_OPENING_REDUCTION_DEGREE);
    assert_eq!(prover.input_claim(), expected_claim);

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

    let (final_witness, final_factor) = prover.final_witness_and_factor_evals().unwrap();
    assert_eq!(final_claim, final_witness * final_factor);
    assert_eq!(
        final_claim,
        extension_opening_reduction_eval_at_point(&witness_evals, &factor_evals, &challenges)
            .unwrap()
    );

    let verified_challenges = verify_eor_full(&witness_evals, &factor_evals, &proof).unwrap();
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

    let groups = vec![
        eor_group(witness_a.clone(), factor_a.clone(), coeff_a).unwrap(),
        eor_group(witness_b.clone(), factor_b.clone(), coeff_b).unwrap(),
    ];
    assert_eq!(
        ExtensionOpeningReductionProver::input_claim_from_groups(&groups).unwrap(),
        expected_claim
    );
    let mut prover = ExtensionOpeningReductionProver::new(groups, expected_claim).unwrap();
    assert_eq!(prover.input_claim(), expected_claim);
    assert_eq!(prover.degree_bound(), EXTENSION_OPENING_REDUCTION_DEGREE);

    let mut transcript = new_transcript();
    let (_proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut transcript, sample_round)
        .unwrap();
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
fn shared_dense_factor_preserves_batched_proof() {
    let witness_a = (0..32)
        .map(|index| F::from_u64((3 * index + 5) as u64))
        .collect::<Vec<_>>();
    let witness_b = (0..32)
        .map(|index| F::from_u64((11 * index + 7) as u64))
        .collect::<Vec<_>>();
    let factor = (0..32)
        .map(|index| F::from_u64((17 * index + 13) as u64))
        .collect::<Vec<_>>();
    let coeff_a = F::from_u64(19);
    let coeff_b = F::from_u64(23);
    let input_claim = coeff_a * extension_opening_reduction_claim(&witness_a, &factor).unwrap()
        + coeff_b * extension_opening_reduction_claim(&witness_b, &factor).unwrap();

    let separate_groups = vec![
        eor_group(witness_a.clone(), factor.clone(), coeff_a).unwrap(),
        eor_group(witness_b.clone(), factor.clone(), coeff_b).unwrap(),
    ];
    let shared_group = ExtensionOpeningReductionGroup::new(
        vec![
            ExtensionOpeningReductionTerm::new(witness_a, coeff_a),
            ExtensionOpeningReductionTerm::new(witness_b, coeff_b),
        ],
        factor,
    )
    .unwrap();

    let prove = |groups| {
        let mut prover = ExtensionOpeningReductionProver::new(groups, input_claim).unwrap();
        let mut transcript = new_transcript();
        let result = prover
            .prove::<F, _, _>(&mut transcript, sample_round)
            .unwrap();
        (result, prover.final_terms().unwrap())
    };
    assert_eq!(prove(vec![shared_group]), prove(separate_groups));
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
    let factor_evals = factor.evals().unwrap();
    let expected_claim = factor.claim_for_witness(&witness_evals).unwrap();

    let group = eor_group(witness_evals.clone(), factor_evals.clone(), F::one()).unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![group], expected_claim).unwrap();
    assert_eq!(prover.input_claim(), expected_claim);

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();
    let (final_witness, final_factor) = prover.final_witness_and_factor_evals().unwrap();
    assert_eq!(final_factor, factor.evaluate(&challenges).unwrap());
    check_extension_opening_reduction_output(final_claim, final_witness, final_factor).unwrap();

    let verified_challenges = verify_eor_full(&witness_evals, &factor_evals, &proof).unwrap();
    assert_eq!(verified_challenges, challenges);
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
    let factor_evals = factor.evals().unwrap();
    let input_claim = factor.claim_for_witness(&witness_evals).unwrap();

    let group = eor_group(witness_evals.clone(), factor_evals, F::one()).unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![group], input_claim).unwrap();
    let mut prover_transcript = new_transcript();
    let (proof, _challenges, _final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

    let mut verifier_transcript = new_transcript();
    let verifier_result = verify_eor_rounds(
        input_claim,
        factor.num_vars(),
        &proof,
        &mut verifier_transcript,
    )
    .unwrap();

    let opened_witness = multilinear_eval(&witness_evals, &verifier_result.challenges).unwrap();
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
        Err(akita_error::AkitaError::InvalidProof)
    ));
}

#[test]
fn extension_opening_reduction_rejects_wrong_final_oracle() {
    let witness_evals: Vec<F> = (0..8).map(|i| F::from_u64((i + 1) as u64)).collect();
    let factor_evals: Vec<F> = (0..8).map(|i| F::from_u64((2 * i + 9) as u64)).collect();

    let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals).unwrap();
    let group = eor_group(witness_evals.clone(), factor_evals, F::one()).unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![group], input_claim).unwrap();
    let mut prover_transcript = new_transcript();
    let (proof, _, _) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

    let bad_factor_evals: Vec<F> = (0..8).map(|i| F::from_u64((2 * i + 10) as u64)).collect();
    let err = verify_eor_full(&witness_evals, &bad_factor_evals, &proof).unwrap_err();
    assert!(matches!(err, akita_error::AkitaError::InvalidProof));
}

#[test]
fn extension_opening_reduction_detached_round_verifier_returns_final_claim() {
    let witness_evals: Vec<F> = (0..4).map(|i| F::from_u64((5 * i + 1) as u64)).collect();
    let factor_evals: Vec<F> = (0..4).map(|i| F::from_u64((13 * i + 2) as u64)).collect();
    let input_claim = extension_opening_reduction_claim(&witness_evals, &factor_evals).unwrap();
    let group = eor_group(witness_evals.clone(), factor_evals.clone(), F::one()).unwrap();
    let mut prover = ExtensionOpeningReductionProver::new(vec![group], input_claim).unwrap();

    let mut prover_transcript = new_transcript();
    let (proof, challenges, final_claim) = prover
        .prove::<F, _, _>(&mut prover_transcript, sample_round)
        .unwrap();

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
    assert!(eor_group(witness_evals, factor_evals, F::one()).is_err());

    let witness_evals = vec![F::one(), F::from_u64(2)];
    let factor_evals = vec![F::one()];
    assert!(extension_opening_reduction_claim(&witness_evals, &factor_evals).is_err());
}

fn proof_claim(witness_evals: &[F], factor_evals: &[F]) -> F {
    extension_opening_reduction_claim(witness_evals, factor_evals).unwrap()
}

// ---------------------------------------------------------------------------
// Regression: EOR round messages must honor `SUM_IS_EXACT`.
//
// `accumulate_dense_round` and `fused_fold_witness_and_accumulate` sum
// `mul_unreduced` products and reduce once. That is only sound when the
// field's accumulator is exact w.r.t. per-term `Mul`. For a field that leaves
// `SUM_IS_EXACT` at its conservative `false` default, the
// prover must reduce every product first, or the round coefficients silently
// drift and the prover's claim diverges.
//
// The existing byte-identical tests only cover fields whose flag is `true`
// (exact) or whose accumulator is trivially exact, so they cannot catch a
// regression on the `false` path. These tests drive the public prover API with
// a mock field whose product accumulator deliberately wraps mod 2^64, and
// assert the emitted round messages stay byte-identical to per-term `Mul`.
mod delayed_product_sum_contract {
    use super::*;
    use jolt_field::{AdditiveGroup, One, Ring, Zero};
    use jolt_field::{Fold, Unreduced};
    use std::fmt;
    use std::iter::{Product, Sum};
    use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    type Inner = Prime64Offset59;

    /// `u64` product accumulator that adds modulo `2^64`. Each stored value is a
    /// canonical residue `< p < 2^64`, but summing several near-`p` residues
    /// wraps, so `reduce(Σ mul_unreduced)` diverges from `Σ a*b` — exactly
    /// the hazard `SUM_IS_EXACT = false` exists to flag.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct WrappingU64Accum(u64);

    impl Zero for WrappingU64Accum {
        fn zero() -> Self {
            Self(0)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }
    impl Add for WrappingU64Accum {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0.wrapping_add(rhs.0))
        }
    }
    impl Add<&WrappingU64Accum> for WrappingU64Accum {
        type Output = Self;
        fn add(self, rhs: &Self) -> Self {
            Self(self.0.wrapping_add(rhs.0))
        }
    }
    impl AddAssign for WrappingU64Accum {
        fn add_assign(&mut self, rhs: Self) {
            self.0 = self.0.wrapping_add(rhs.0);
        }
    }
    impl Sub for WrappingU64Accum {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0.wrapping_sub(rhs.0))
        }
    }
    impl Sub<&WrappingU64Accum> for WrappingU64Accum {
        type Output = Self;
        fn sub(self, rhs: &Self) -> Self {
            Self(self.0.wrapping_sub(rhs.0))
        }
    }
    impl SubAssign for WrappingU64Accum {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = self.0.wrapping_sub(rhs.0);
        }
    }
    impl Neg for WrappingU64Accum {
        type Output = Self;
        fn neg(self) -> Self {
            Self(self.0.wrapping_neg())
        }
    }
    impl AdditiveGroup for WrappingU64Accum {}

    /// Field wrapper over `Prime64Offset59` whose only non-standard behavior is
    /// the lossy product accumulator above plus `SUM_IS_EXACT =
    /// false`. All ordinary arithmetic delegates to the exact inner field, so a
    /// per-term `Mul` computation is trivially the ground truth.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
    struct LossyField(Inner);

    impl LossyField {
        fn from_u64(v: u64) -> Self {
            Self(Inner::from_u64(v))
        }
    }
    impl fmt::Display for LossyField {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Zero for LossyField {
        fn zero() -> Self {
            Self(Inner::zero())
        }
        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }
    impl One for LossyField {
        fn one() -> Self {
            Self(Inner::one())
        }
    }
    impl Add for LossyField {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }
    impl Add<&LossyField> for LossyField {
        type Output = Self;
        fn add(self, rhs: &Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }
    impl AddAssign for LossyField {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }
    impl Sub for LossyField {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }
    impl Sub<&LossyField> for LossyField {
        type Output = Self;
        fn sub(self, rhs: &Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }
    impl SubAssign for LossyField {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
        }
    }
    impl Neg for LossyField {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    impl Mul for LossyField {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }
    impl Mul<&LossyField> for LossyField {
        type Output = Self;
        fn mul(self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0)
        }
    }
    impl MulAssign for LossyField {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0;
        }
    }
    impl Sum for LossyField {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), |acc, x| acc + x)
        }
    }
    impl<'a> Sum<&'a LossyField> for LossyField {
        fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), |acc, x| acc + *x)
        }
    }
    impl Product for LossyField {
        fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::one(), |acc, x| acc * x)
        }
    }
    impl<'a> Product<&'a LossyField> for LossyField {
        fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
            iter.fold(Self::one(), |acc, x| acc * *x)
        }
    }
    impl AdditiveGroup for LossyField {}
    impl Ring for LossyField {
        fn from_u64(v: u64) -> Self {
            Self(Inner::from_u64(v))
        }
        fn from_i64(v: i64) -> Self {
            Self(Inner::from_i64(v))
        }
        fn from_u128(v: u128) -> Self {
            Self(Inner::from_u128(v))
        }
        fn from_i128(v: i128) -> Self {
            Self(Inner::from_i128(v))
        }
    }
    impl Field for LossyField {
        fn inverse(&self) -> Option<Self> {
            self.0.inverse().map(Self)
        }
        fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
            Self(Inner::random(rng))
        }
    }

    impl Fold for LossyField {
        type Ctx = Self;
        fn precompute(r: Self) -> Self {
            r
        }
        fn fold_one(r: &Self, even: Self, odd: Self) -> Self {
            even + *r * (odd - even)
        }
    }

    impl Unreduced for LossyField {
        type SmallProduct = WrappingU64Accum;
        type Product = WrappingU64Accum;
        type Wide = Self;

        // Deliberately inexact: the accumulator wraps mod 2^64, so a delayed
        // batch sum diverges from per-term `Mul` once the sum crosses 2^64.
        const SUM_IS_EXACT: bool = false;

        fn mul_u64_unreduced(self, small: u64) -> WrappingU64Accum {
            WrappingU64Accum((self.0 * Inner::from_u64(small)).to_limbs())
        }
        fn mul_unreduced(self, other: Self) -> WrappingU64Accum {
            WrappingU64Accum((self.0 * other.0).to_limbs())
        }
        fn reduce_small_product(accum: WrappingU64Accum) -> Self {
            Self(Inner::from_u64(accum.0))
        }
        fn reduce_product(accum: WrappingU64Accum) -> Self {
            Self(Inner::from_u64(accum.0))
        }
        fn scale_wide(self, small: i32) -> Self::Wide {
            self * Self::from_i32(small)
        }
        fn reduce_wide(wide: Self::Wide) -> Self {
            wide
        }
    }

    /// `p - 1`, the largest canonical residue (~2^64); prime-agnostic via `-1`.
    fn max_residue() -> LossyField {
        -LossyField::one()
    }

    /// Ground-truth degree-2 round message `c + l·X + q·X²` computed entirely
    /// with per-term `Mul`, with `l = claim - 2c - q`.
    fn reference_round_eval(
        witness: &[LossyField],
        factor: &[LossyField],
        claim: LossyField,
        x: LossyField,
    ) -> LossyField {
        let half = witness.len() / 2;
        let mut constant = LossyField::zero();
        let mut quadratic = LossyField::zero();
        for i in 0..half {
            let w0 = witness[2 * i];
            let w1 = witness[2 * i + 1];
            let a0 = factor[2 * i];
            let a1 = factor[2 * i + 1];
            constant += w0 * a0;
            quadratic += (w1 - w0) * (a1 - a0);
        }
        let linear = claim - constant - constant - quadratic;
        constant + linear * x + quadratic * x * x
    }

    /// Exact multilinear fold `even + r·(odd − even)`, matching the prover.
    fn reference_fold(table: &[LossyField], r: LossyField) -> Vec<LossyField> {
        (0..table.len() / 2)
            .map(|i| table[2 * i] + r * (table[2 * i + 1] - table[2 * i]))
            .collect()
    }

    /// Confirm the chosen tables actually trip the lossy accumulator, so the
    /// byte-identicality assertions below would fail if the prover summed wide
    /// products instead of reducing per term.
    fn assert_inputs_are_hazardous(witness: &[LossyField], factor: &[LossyField]) {
        let half = witness.len() / 2;
        let per_term = (0..half).fold(LossyField::zero(), |acc, i| {
            acc + witness[2 * i] * factor[2 * i]
        });
        let delayed = {
            let mut accum = WrappingU64Accum::zero();
            for i in 0..half {
                accum += witness[2 * i].mul_unreduced(factor[2 * i]);
            }
            LossyField::reduce_product(accum)
        };
        assert_ne!(
            per_term, delayed,
            "test inputs must trigger the lossy delayed accumulator"
        );
    }

    // Dense path: round 0 exercises `accumulate_dense_round`; later rounds use
    // the cache filled by `fused_fold_witness_and_accumulate`. Both must reduce per term
    // for this field, so every round message matches the per-term reference.
    #[test]
    fn dense_round_messages_honor_delayed_product_flag() {
        let zero = LossyField::zero();
        let one = LossyField::one();
        let two = LossyField::from_u64(2);
        let max = max_residue();
        // Even slots each multiply to ~2^64; four of them overflow a u64.
        let mut witness = vec![one, two, one, two, one, two, one, two];
        let mut factor = vec![max, zero, max, zero, max, zero, max, zero];
        assert_inputs_are_hazardous(&witness, &factor);

        let input_claim = extension_opening_reduction_claim(&witness, &factor).unwrap();
        let group = eor_group(witness.clone(), factor.clone(), LossyField::one()).unwrap();
        let mut prover = ExtensionOpeningReductionProver::new(vec![group], input_claim).unwrap();
        let mut claim = prover.input_claim();

        let eval_points = [zero, one, two, LossyField::from_u64(3)];
        for round in 0..3 {
            let prover_poly = prover.compute_round_univariate(round, claim);
            for &x in &eval_points {
                assert_eq!(
                    prover_poly.evaluate(&x),
                    reference_round_eval(&witness, &factor, claim, x),
                    "dense round {round} diverged from per-term Mul at x={x:?}"
                );
            }
            let challenge = LossyField::from_u64(7 + round as u64);
            claim = prover_poly.evaluate(&challenge);
            prover.ingest_challenge(round, challenge);
            witness = reference_fold(&witness, challenge);
            factor = reference_fold(&factor, challenge);
        }
    }
}
