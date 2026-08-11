use super::*;
use akita_prover::MultilinearPolynomial;

#[test]
fn heterogeneous_polynomial_groups_round_trip_with_group_local_points() {
    const ONEHOT_PRE_NV: usize = 14;
    const DENSE_PRE_NV: usize = 15;
    const FINAL_NV: usize = 16;

    let setup = OneHotScheme::setup_prover(FINAL_NV, 4).expect("heterogeneous setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("heterogeneous stack");

    let onehot_pre = OneHotPoly::<OneHotF, u8>::new(
        16,
        ONEHOT_D,
        (0..((1usize << ONEHOT_PRE_NV) / 16))
            .map(|index| (index % 3 == 0).then_some((index % 16) as u8))
            .collect(),
    )
    .expect("K=16 precommitted polynomial");

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

    let final_onehot = OneHotPoly::<OneHotF, u8>::new(
        256,
        ONEHOT_D,
        (0..((1usize << FINAL_NV) / 256))
            .map(|index| Some((17 * index % 256) as u8))
            .collect(),
    )
    .expect("K=256 final polynomial");

    let dense_polys = [dense_a.clone(), dense_b.clone()];
    let onehot_pre_group = [MultilinearPolynomial::onehot(onehot_pre.clone())];
    let dense_group = [
        MultilinearPolynomial::dense(dense_a),
        MultilinearPolynomial::dense(dense_b),
    ];
    let final_group = [MultilinearPolynomial::onehot(final_onehot.clone())];

    let akita_prover::CommitOutput {
        committed_group: onehot_pre_commitment,
        hint: onehot_pre_hint,
    } = OneHotScheme::commit(
        &setup,
        std::slice::from_ref(&onehot_pre),
        &stack,
        akita_prover::GroupPosition::Independent,
    )
    .expect("K=16 precommit");
    let akita_prover::CommitOutput {
        committed_group: dense_commitment,
        hint: dense_hint,
    } = Scheme::commit(
        &setup,
        &dense_polys,
        &stack,
        akita_prover::GroupPosition::Independent,
    )
    .expect("dense precommit");
    let prior_group_profiles = akita_types::PriorGroupProfiles::from_profiles(vec![
        onehot_pre_commitment.profile,
        dense_commitment.profile,
    ]);
    let akita_prover::CommitOutput {
        committed_group: final_commitment,
        hint: final_hint,
    } = OneHotScheme::commit(
        &setup,
        &final_group,
        &stack,
        akita_prover::GroupPosition::Final {
            prior_group_profiles: &prior_group_profiles,
        },
    )
    .expect("heterogeneous final commit");

    let onehot_pre_profile = onehot_pre_commitment.profile;
    let final_profile = final_commitment.profile;

    let onehot_pre_point = (0..ONEHOT_PRE_NV)
        .map(|index| OneHotF::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let dense_point = (0..DENSE_PRE_NV)
        .map(|index| OneHotF::from_u64((index + 37) as u64))
        .collect::<Vec<_>>();
    let final_point = (0..FINAL_NV)
        .map(|index| OneHotF::from_u64((index + 71) as u64))
        .collect::<Vec<_>>();
    let onehot_pre_opening = opening_from_poly(
        &onehot_pre,
        &onehot_pre_point,
        onehot_pre_profile.inner_commit_matrix.ring_dimension(),
        onehot_pre_profile.num_positions_per_block,
        onehot_pre_profile.num_live_blocks,
    );
    let dense_openings = vec![
        dense_opening(&dense_evals_a, &dense_point),
        dense_opening(&dense_evals_b, &dense_point),
    ];
    let final_opening = opening_from_poly(
        &final_onehot,
        &final_point,
        final_profile.inner_commit_matrix.ring_dimension(),
        final_profile.num_positions_per_block,
        final_profile.num_live_blocks,
    );

    let prover_claims = OpeningClaims::from_groups(vec![
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
    .expect("heterogeneous prover claims");
    let onehot_pre_refs = [&onehot_pre_group[0]];
    let dense_refs = [&dense_group[0], &dense_group[1]];
    let final_refs = [&final_group[0]];
    let prover_data = SelectedProverOpeningData::from_committed_claims::<OneHotCfg>(
        prior_group_profiles,
        prover_claims,
        vec![onehot_pre_hint, dense_hint, final_hint],
        vec![&onehot_pre_refs, &dense_refs, &final_refs],
    )
    .expect("heterogeneous prover opening data");
    let selection = prover_data.selection();

    const DOMAIN: &[u8] = b"test/heterogeneous-polynomial-groups";
    let mut prover_transcript = AkitaTranscript::new(DOMAIN);
    let proof = OneHotScheme::batched_prove(
        &setup,
        prover_data,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("heterogeneous proof");
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
    .expect("heterogeneous verifier claims");
    let mut verifier_transcript = AkitaTranscript::new(DOMAIN);
    OneHotScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        GroupBatchStatement::new(selection, verifier_claims).expect("heterogeneous statement"),
        BasisMode::Lagrange,
    )
    .expect("heterogeneous verification");
}
