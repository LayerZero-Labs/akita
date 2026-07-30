use super::*;
use crate::test_support::EnvelopeFinalGroupConfig;
use akita_types::{AkitaScheduleLookupKey, GroupSource, PolynomialGroupLayout};

type DenseGroupCfg = EnvelopeFinalGroupConfig<Cfg, Cfg>;
type DenseGroupScheme = AkitaCommitmentScheme<DenseGroupCfg>;

#[test]
fn dense_multi_group_root_round_trips() {
    const NUM_VARS: usize = 16;

    let setup = DenseGroupScheme::setup_prover(NUM_VARS, 2).expect("dense grouped setup");
    let prepared = CpuBackend
        .prepare_setup(&setup)
        .expect("prepared dense grouped setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("dense grouped stack");

    let len = 1usize << NUM_VARS;
    let pre_evals = (0..len)
        .map(|index| F::from_u64(index as u64))
        .collect::<Vec<_>>();
    let final_evals = (0..len)
        .map(|index| F::from_u64((3 * index + 7) as u64))
        .collect::<Vec<_>>();
    let pre_poly =
        DensePoly::<F>::from_field_evals(NUM_VARS, D, &pre_evals).expect("precommitted dense poly");
    let final_poly =
        DensePoly::<F>::from_field_evals(NUM_VARS, D, &final_evals).expect("final dense poly");

    let (pre_commitment, pre_hint) = DenseGroupScheme::commit_group(
        &setup,
        std::slice::from_ref(&pre_poly),
        &stack,
        DenseGroupCfg::group_source(),
    )
    .expect("dense precommit");
    let pre_descriptor = pre_commitment.descriptor;
    let (final_commitment, final_hint) = DenseGroupScheme::commit_final_group(
        &setup,
        std::slice::from_ref(&final_poly),
        &stack,
        vec![pre_descriptor],
        DenseGroupCfg::group_source(),
    )
    .expect("dense final commit");

    let lookup_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(NUM_VARS),
        final_source: DenseGroupCfg::group_source(),
        precommitteds: vec![pre_descriptor],
    };
    let schedule = DenseGroupCfg::runtime_schedule(lookup_key).expect("dense multi-group schedule");
    assert!(matches!(
        schedule.root.params.final_group.source.encoding(),
        akita_types::GroupSourceEncoding::Bounded { .. }
    ));
    assert_eq!(schedule.root.params.precommitted_groups.len(), 1);

    let pre_point = (0..NUM_VARS)
        .map(|index| F::from_u64((index + 101) as u64))
        .collect::<Vec<_>>();
    let final_point = (0..NUM_VARS)
        .map(|index| F::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let pre_opening = dense_opening(&pre_evals, &pre_point);
    let final_opening = dense_opening(&final_evals, &final_point);
    let prover_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(pre_point.clone(), vec![pre_opening], pre_commitment.clone())
            .expect("precommitted prover claim"),
        PolynomialGroupClaims::new(
            final_point.clone(),
            vec![final_opening],
            final_commitment.clone(),
        )
        .expect("final prover claim"),
    ])
    .expect("dense grouped prover claims");
    let pre_refs = [&pre_poly];
    let final_refs = [&final_poly];
    let prover_data = ProverOpeningData::new(
        prover_claims,
        vec![pre_hint, final_hint],
        vec![&pre_refs, &final_refs],
    )
    .expect("dense grouped prover data");

    const TRANSCRIPT_DOMAIN: &[u8] = b"test/dense-multi-group";
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = DenseGroupScheme::batched_prove(
        &setup,
        prover_data,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("dense multi-group proof");

    let verifier_setup =
        DenseGroupScheme::setup_verifier(&setup).expect("dense grouped verifier setup");
    let verifier_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(pre_point.clone(), vec![pre_opening], &pre_commitment)
            .expect("precommitted verifier claim"),
        PolynomialGroupClaims::new(final_point.clone(), vec![final_opening], &final_commitment)
            .expect("final verifier claim"),
    ])
    .expect("dense grouped verifier claims");
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    DenseGroupScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims,
        BasisMode::Lagrange,
    )
    .expect("dense multi-group verification");

    let mut wrong_source_commitment = pre_commitment.clone();
    wrong_source_commitment.descriptor.source = GroupSource::bounded(32);
    let wrong_source_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(
            pre_point.clone(),
            vec![pre_opening],
            &wrong_source_commitment,
        )
        .expect("wrong-source precommitted verifier claim"),
        PolynomialGroupClaims::new(final_point.clone(), vec![final_opening], &final_commitment)
            .expect("final verifier claim"),
    ])
    .expect("wrong-source verifier claims");
    let mut wrong_source_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    DenseGroupScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut wrong_source_transcript,
        wrong_source_claims,
        BasisMode::Lagrange,
    )
    .expect_err("changed precommitted source descriptor must reject");

    let mut wrong_final_descriptor = final_commitment.clone();
    wrong_final_descriptor.descriptor.n_b = wrong_final_descriptor
        .descriptor
        .n_b
        .checked_add(1)
        .expect("test n_b increment");
    let wrong_final_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(pre_point.clone(), vec![pre_opening], &pre_commitment)
            .expect("precommitted verifier claim"),
        PolynomialGroupClaims::new(
            final_point.clone(),
            vec![final_opening],
            &wrong_final_descriptor,
        )
        .expect("wrong final verifier claim"),
    ])
    .expect("wrong-final-descriptor verifier claims");
    let mut wrong_final_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    DenseGroupScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut wrong_final_transcript,
        wrong_final_claims,
        BasisMode::Lagrange,
    )
    .expect_err("changed final frozen descriptor must reject");

    let mut wrong_pre_point = pre_point;
    wrong_pre_point[0] += F::one();
    let wrong_point_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(wrong_pre_point, vec![pre_opening], &pre_commitment)
            .expect("wrong-point precommitted verifier claim"),
        PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
            .expect("final verifier claim"),
    ])
    .expect("wrong-point verifier claims");
    let mut wrong_point_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    DenseGroupScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut wrong_point_transcript,
        wrong_point_claims,
        BasisMode::Lagrange,
    )
    .expect_err("wrong group point must reject");
}
