use super::*;
use akita_types::{AkitaScheduleLookupKey, GroupSource, PolynomialGroupLayout};

struct DownstreamK16Provider;

impl WholeGroupSourceProvider<OneHotF, OneHotPoly<OneHotF, u8>> for DownstreamK16Provider {
    fn planning_source(&self) -> GroupSource {
        GroupSource::registered(
            akita_types::GroupSourceRegistration::new(*b"downstream/k016\0", [16; 16]),
            akita_types::GroupSourceEncoding::SparseBinary { chunk_size: 16 },
        )
    }

    fn validate_group(&self, polynomials: &[OneHotPoly<OneHotF, u8>]) -> Result<(), AkitaError> {
        for poly in polynomials {
            akita_prover::RootPolyMeta::validate_group_source(poly, self.planning_source())?;
        }
        Ok(())
    }
}

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
    let onehot_pre_group = [onehot_pre];
    let downstream_provider = DownstreamK16Provider;
    assert_ne!(
        downstream_provider.planning_source().registration(),
        GroupSource::one_hot(16).registration(),
        "fixture must exercise downstream provider identity in a curated group batch"
    );
    let (onehot_pre_commitment, onehot_pre_hint) =
        OneHotScheme::commit_group(&setup, &onehot_pre_group, &stack, &downstream_provider)
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
    let dense_group = [dense_a, dense_b];
    let dense_provider = DenseGroupProvider::new(32);
    let (dense_commitment, dense_hint) =
        OneHotScheme::commit_group(&setup, &dense_group, &stack, &dense_provider)
            .expect("32-bit dense precommit");

    let final_onehot = OneHotPoly::<OneHotF, u8>::new(
        256,
        ONEHOT_D,
        (0..((1usize << FINAL_NV) / 256))
            .map(|index| Some((17 * index % 256) as u8))
            .collect(),
    )
    .expect("K=256 final polynomial");
    let final_group = [final_onehot];
    let final_provider = OneHotGroupProvider::new(256);
    let (final_commitment, final_hint) = OneHotScheme::commit_final_group(
        &setup,
        &final_group,
        &stack,
        vec![onehot_pre_commitment.profile, dense_commitment.profile],
        &final_provider,
    )
    .expect("mixed-source final commit");

    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(FINAL_NV, 1),
        final_source: GroupSource::one_hot(256),
        precommitteds: vec![onehot_pre_commitment.profile, dense_commitment.profile],
        precommitted_sources: vec![GroupSource::one_hot(16), GroupSource::bounded(32)],
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
    let onehot_pre_opening =
        opening_from_poly(&onehot_pre_group[0], &onehot_pre_point, &onehot_pre_params);
    let dense_openings = vec![
        dense_opening(&dense_evals_a, &dense_point),
        dense_opening(&dense_evals_b, &dense_point),
    ];
    let final_opening = opening_from_poly(&final_group[0], &final_point, final_params);

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
    .expect("mixed prover claims");
    let profiles = batch_profiles::<OneHotCfg>(&prover_claims).expect("mixed profiles");
    let selection = OneHotCfg::select_schedule_for_profiles(&profiles)
        .expect("mixed schedule selection")
        .selection();
    let prover_data = ProverOpeningData::from_prepared_groups(
        selection,
        prover_claims,
        vec![
            ProverGroupInput::new(
                onehot_pre_hint,
                EitherPreparedGroup::Left(
                    downstream_provider
                        .prepare_prover_group(&onehot_pre_group)
                        .expect("prepared downstream K=16 group"),
                ),
            ),
            ProverGroupInput::new(
                dense_hint,
                EitherPreparedGroup::Right(
                    dense_provider
                        .prepare_prover_group(&dense_group)
                        .expect("prepared dense group"),
                ),
            ),
            ProverGroupInput::new(
                final_hint,
                EitherPreparedGroup::Left(
                    final_provider
                        .prepare_prover_group(&final_group)
                        .expect("prepared K=256 group"),
                ),
            ),
        ],
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
        GroupBatchStatement::new(selection, verifier_claims).expect("mixed-source statement"),
        BasisMode::Lagrange,
    )
    .expect("mixed-source verification");
}
