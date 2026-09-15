#![allow(missing_docs)]

use akita_algebra::{poly::multilinear_eval, EqPolynomial};
use akita_error::AkitaError;
use akita_prover::RelationRangeImageProver;
use akita_sumcheck::{
    advance_eq_factored_claim, prove_sumcheck, EqFactoredUniPoly, SumcheckInstanceProver,
};
use akita_transcript::{labels, AkitaTranscript, Transcript};
use jolt_field::{One, Prime128Offset275 as F, Ring};

fn sample_round(transcript: &mut AkitaTranscript<F>) -> Result<F, AkitaError> {
    Ok(transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND))
}

#[test]
fn public_relation_range_image_constructor_composes_with_direct_sumcheck() {
    let compact_witness = vec![-1_i8, 0, 1, 2];
    let witness = compact_witness
        .iter()
        .map(|&value| F::from_i64(i64::from(value)))
        .collect::<Vec<_>>();
    let range_image = witness
        .iter()
        .map(|&value| value * (value + F::one()))
        .collect::<Vec<_>>();
    let stage1_point = [F::from_u64(3), F::from_u64(5)];
    let range_image_evaluation = multilinear_eval(&range_image, &stage1_point).unwrap();
    let mut prover = RelationRangeImageProver::new_virtual_only(
        compact_witness,
        &stage1_point,
        range_image_evaluation,
        8,
        2,
        1,
        1,
    )
    .unwrap();
    let mut transcript = AkitaTranscript::prover(b"public-stage2-composition", b"fixture");

    let (proof, challenges, final_claim) =
        prove_sumcheck::<F, _, F, _, _>(&mut prover, &mut transcript, sample_round).unwrap();

    let witness_evaluation = multilinear_eval(&witness, &challenges).unwrap();
    let equality_evaluation = EqPolynomial::mle(&stage1_point, &challenges).unwrap();
    assert_eq!(proof.round_polys.len(), stage1_point.len());
    assert_eq!(prover.num_rounds(), stage1_point.len());
    assert_eq!(
        final_claim,
        equality_evaluation * witness_evaluation * (witness_evaluation + F::one())
    );
}

#[test]
fn normalized_claim_advance_is_public() {
    let claim = F::from_u64(17);
    let tau = F::from_u64(5);
    let challenge = F::from_u64(11);
    let poly = EqFactoredUniPoly {
        coeffs_except_constant_term: vec![F::from_u64(3), F::from_u64(7)],
    };
    let constant = claim - tau * F::from_u64(10);

    assert_eq!(
        advance_eq_factored_claim(claim, tau, &poly, challenge),
        constant + F::from_u64(3) * challenge + F::from_u64(7) * challenge * challenge
    );
}
