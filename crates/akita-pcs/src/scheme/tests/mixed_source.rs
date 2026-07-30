use super::*;
use akita_types::{AkitaScheduleLookupKey, GroupSource, PolynomialGroupLayout};

type MixedPoly = MultilinearPolynomial<OneHotF, u8>;

#[test]
fn heterogeneous_group_sources_round_trip_with_group_local_points() {
    const ONEHOT_PRE_NV: usize = 14;
    const DENSE_PRE_NV: usize = 15;
    const FINAL_NV: usize = 16;

    let setup = OneHotScheme::setup_prover(FINAL_NV, 4).expect("mixed-source setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("mixed-source stack");

    let onehot_pre = OneHotPoly::<OneHotF, u8>::new(
        16,
        ONEHOT_D,
        (0..((1usize << ONEHOT_PRE_NV) / 16))
            .map(|index| (index % 3 == 0).then_some((index % 16) as u8))
            .collect(),
    )
    .expect("K=16 precommitted polynomial");
    let onehot_pre_wrapped = MixedPoly::onehot(onehot_pre.clone());
    let (onehot_pre_commitment, onehot_pre_hint) = OneHotScheme::commit_group(
        &setup,
        std::slice::from_ref(&onehot_pre_wrapped),
        &stack,
        GroupSource::one_hot(16),
    )
    .expect("K=16 precommit");

    let dense_evals_a = (0..(1usize << DENSE_PRE_NV))
        .map(|index| OneHotF::from_u64((index % 257) as u64))
        .collect::<Vec<_>>();
    let dense_evals_b = (0..(1usize << DENSE_PRE_NV))
        .map(|index| OneHotF::from_u64((index % 509) as u64))
        .collect::<Vec<_>>();
    let dense_a = DensePoly::from_field_evals(DENSE_PRE_NV, ONEHOT_D, &dense_evals_a)
        .expect("first bounded dense polynomial");
    let dense_b = DensePoly::from_field_evals(DENSE_PRE_NV, ONEHOT_D, &dense_evals_b)
        .expect("second bounded dense polynomial");
    let dense_group = [MixedPoly::dense(dense_a), MixedPoly::dense(dense_b)];
    let dense_source = GroupSource::bounded(32);
    let (dense_commitment, dense_hint) =
        OneHotScheme::commit_group(&setup, &dense_group, &stack, dense_source)
            .expect("32-bit dense precommit");

    let final_onehot = OneHotPoly::<OneHotF, u8>::new(
        256,
        ONEHOT_D,
        (0..((1usize << FINAL_NV) / 256))
            .map(|index| Some((17 * index % 256) as u8))
            .collect(),
    )
    .expect("K=256 final polynomial");
    let final_wrapped = MixedPoly::onehot(final_onehot.clone());
    let (final_commitment, final_hint) = OneHotScheme::commit_final_group(
        &setup,
        std::slice::from_ref(&final_wrapped),
        &stack,
        vec![
            onehot_pre_commitment.descriptor,
            dense_commitment.descriptor,
        ],
        GroupSource::one_hot(256),
    )
    .expect("mixed-source final commit");

    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(FINAL_NV, 1),
        final_source: GroupSource::one_hot(256),
        precommitteds: vec![
            onehot_pre_commitment.descriptor,
            dense_commitment.descriptor,
        ],
    };
    let schedule = OneHotCfg::runtime_schedule(key).expect("curated mixed schedule");
    let onehot_pre_params = akita_config::committed_group_params::<OneHotCfg>(
        &PolynomialGroupLayout::new(ONEHOT_PRE_NV, 1),
        GroupSource::one_hot(16),
    )
    .expect("K=16 precommit params");
    let final_params = &schedule.root.params.final_group.commitment;

    let onehot_pre_point = (0..ONEHOT_PRE_NV)
        .map(|index| OneHotF::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let dense_point = (0..DENSE_PRE_NV)
        .map(|index| OneHotF::from_u64((index + 37) as u64))
        .collect::<Vec<_>>();
    let final_point = (0..FINAL_NV)
        .map(|index| OneHotF::from_u64((index + 71) as u64))
        .collect::<Vec<_>>();
    let onehot_pre_opening = opening_from_poly(&onehot_pre, &onehot_pre_point, &onehot_pre_params);
    let dense_openings = vec![
        dense_opening(&dense_evals_a, &dense_point),
        dense_opening(&dense_evals_b, &dense_point),
    ];
    let final_opening = opening_from_poly(&final_onehot, &final_point, final_params);

    let onehot_refs = [&onehot_pre_wrapped];
    let dense_refs = [&dense_group[0], &dense_group[1]];
    let final_refs = [&final_wrapped];
    let prover_data = ProverOpeningData::new(
        OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                onehot_pre_point.clone(),
                vec![onehot_pre_opening],
                onehot_pre_commitment.clone(),
            )
            .expect("K=16 prover claims"),
            PolynomialGroupClaims::new(
                dense_point.clone(),
                dense_openings.clone(),
                dense_commitment.clone(),
            )
            .expect("dense prover claims"),
            PolynomialGroupClaims::new(
                final_point.clone(),
                vec![final_opening],
                final_commitment.clone(),
            )
            .expect("final prover claims"),
        ])
        .expect("mixed prover claims"),
        vec![onehot_pre_hint, dense_hint, final_hint],
        vec![&onehot_refs, &dense_refs, &final_refs],
    )
    .expect("mixed prover opening data");

    const DOMAIN: &[u8] = b"test/heterogeneous-group-sources";
    let mut prover_transcript = AkitaTranscript::new(DOMAIN);
    let proof = OneHotScheme::batched_prove(
        &setup,
        prover_data,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("mixed-source proof");

    let verifier_setup = OneHotScheme::setup_verifier(&setup).expect("verifier setup");
    let verifier_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(
            onehot_pre_point,
            vec![onehot_pre_opening],
            &onehot_pre_commitment,
        )
        .expect("K=16 verifier claims"),
        PolynomialGroupClaims::new(dense_point, dense_openings, &dense_commitment)
            .expect("dense verifier claims"),
        PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
            .expect("final verifier claims"),
    ])
    .expect("mixed verifier claims");
    let mut verifier_transcript = AkitaTranscript::new(DOMAIN);
    OneHotScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims,
        BasisMode::Lagrange,
    )
    .expect("mixed-source verification");
}
