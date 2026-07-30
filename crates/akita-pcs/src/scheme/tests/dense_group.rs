use super::*;

type DenseGroupCfg = Cfg;
type DenseGroupScheme = AkitaCommitmentScheme<DenseGroupCfg>;

struct DownstreamDenseProvider {
    coefficient_bits: u32,
}

impl WholeGroupSourceProvider<F, DensePoly<F>> for DownstreamDenseProvider {
    fn planning_source(&self) -> akita_types::GroupSource {
        akita_types::GroupSource::registered(
            akita_types::GroupSourceRegistration::new(*b"downstream/test\0", [11; 16]),
            akita_types::GroupSourceEncoding::Bounded {
                coefficient_bits: self.coefficient_bits,
            },
        )
    }

    fn validate_group(&self, polynomials: &[DensePoly<F>]) -> Result<(), AkitaError> {
        for poly in polynomials {
            akita_prover::RootPolyMeta::validate_group_source(poly, self.planning_source())?;
        }
        Ok(())
    }
}

#[test]
fn downstream_dense_provider_round_trips() {
    const NUM_VARS: usize = 16;

    let setup = DenseGroupScheme::setup_prover(NUM_VARS, 1).expect("dense provider setup");
    let prepared = CpuBackend
        .prepare_setup(&setup)
        .expect("prepared dense provider setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("dense provider stack");

    let evals = (0..1usize << NUM_VARS)
        .map(|index| F::from_u64((3 * index + 7) as u64))
        .collect::<Vec<_>>();
    let poly =
        DensePoly::<F>::from_field_evals(NUM_VARS, D, &evals).expect("downstream dense poly");
    let provider = DownstreamDenseProvider {
        coefficient_bits: DenseGroupCfg::decomposition().field_bits(),
    };
    assert_ne!(
        provider.planning_source().registration(),
        DenseGroupCfg::group_source().registration(),
        "fixture must exercise a downstream-defined provider identity"
    );

    let (commitment, hint) =
        DenseGroupScheme::commit_group(&setup, std::slice::from_ref(&poly), &stack, &provider)
            .expect("downstream dense commit");
    let point = (0..NUM_VARS)
        .map(|index| F::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let opening = dense_opening(&evals, &point);
    let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![opening],
        commitment.clone(),
    )
    .expect("downstream dense prover claim")])
    .expect("downstream dense prover claims");
    let poly_refs = [&poly];
    let prover_data =
        selected_prover_data::<DenseGroupCfg, _>(prover_claims, vec![hint], vec![&poly_refs])
            .expect("downstream dense prover data");
    let selection = prover_data.selection().expect("dense provider selection");

    const TRANSCRIPT_DOMAIN: &[u8] = b"test/downstream-dense-provider";
    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    let proof = DenseGroupScheme::batched_prove(
        &setup,
        prover_data,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("downstream dense proof");

    let verifier_setup =
        DenseGroupScheme::setup_verifier(&setup).expect("dense provider verifier setup");
    let verifier_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![opening],
        &commitment,
    )
    .expect("downstream dense verifier claim")])
    .expect("downstream dense verifier claims");
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    DenseGroupScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        GroupBatchStatement::new(selection, verifier_claims).expect("dense provider statement"),
        BasisMode::Lagrange,
    )
    .expect("downstream dense verification");

    let mut wrong_profile = commitment.clone();
    wrong_profile.profile.num_live_blocks = wrong_profile.profile.num_live_blocks.saturating_add(1);
    let wrong_profile_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![opening],
        &wrong_profile,
    )
    .expect("wrong-profile dense verifier claim")])
    .expect("wrong-profile dense verifier claims");
    let mut wrong_profile_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    DenseGroupScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut wrong_profile_transcript,
        GroupBatchStatement::new(selection, wrong_profile_claims).expect("wrong-profile statement"),
        BasisMode::Lagrange,
    )
    .expect_err("changed committed profile must reject");

    let mut wrong_point = point;
    wrong_point[0] += F::one();
    let wrong_point_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        wrong_point,
        vec![opening],
        &commitment,
    )
    .expect("wrong-point dense verifier claim")])
    .expect("wrong-point dense verifier claims");
    let mut wrong_point_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_DOMAIN);
    DenseGroupScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut wrong_point_transcript,
        GroupBatchStatement::new(selection, wrong_point_claims).expect("wrong-point statement"),
        BasisMode::Lagrange,
    )
    .expect_err("wrong opening point must reject");
}
