use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_types::{
    derive_public_matrix_prefix, setup_claims::SetupClaim, AkitaExpandedSetup, AkitaSetupDescriptor,
};
use jolt_field::Prime128OffsetA7F7 as F;

fn setup<B: Field + CanonicalEncoding>(len: usize) -> Arc<AkitaExpandedSetup<B>> {
    let seed = [23; 32].into();
    Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 16,
                max_num_batched_polys: 1,
                num_field_elements: len,
                setup_seed: seed,
            },
            derive_public_matrix_prefix(len, &[23; 32].into()),
        ),
    )
}

#[test]
fn single_term_rounds_match_dense_reference() {
    use jolt_field::{Ext2, FpExt4, Prime32Offset99, Prime64Offset59};
    single_term_rounds::<F, F>();
    single_term_rounds::<Prime32Offset99, FpExt4<Prime32Offset99>>();
    single_term_rounds::<Prime64Offset59, Ext2<Prime64Offset59>>();
}

fn single_term_rounds<B, E>()
where
    B: Field + CanonicalEncoding,
    E: FpExtEncoding<B> + ExtField<B> + Ring + MulBaseUnreduced<B>,
{
    use super::super::stage3::product_table::FactoredProductTerm;
    let source = setup::<B>(512);
    let backend = CpuBackend::<B, E>::new(Arc::clone(&source)).unwrap();
    let domain = SetupPolynomialId::new::<B, E>(&source, 512).unwrap();
    let point = (0..9).map(|i| E::from_u64(31 + i)).collect::<Vec<_>>();
    let query = SetupQuery::evaluation(point.clone()).unwrap();
    let (value, prepared) = backend
        .prepare_setup_query(&domain, &query, 8, SetupBatchMemory::Retain)
        .unwrap();
    let statement = SetupBatchStatement::new(
        domain,
        8,
        vec![SetupClaim::new(
            SetupPolynomialId::new::<B, E>(&source, 512).unwrap(),
            query,
            value,
        )
        .unwrap()],
    )
    .unwrap();
    let mut batch = backend
        .begin_setup_batch(&statement, vec![prepared], &[E::one()])
        .unwrap();
    let mut dense = FactoredProductTerm::new_dense(
        source
            .shared_matrix()
            .as_field_slice()
            .iter()
            .copied()
            .map(E::lift_base)
            .collect(),
        EqPolynomial::evals(&point[3..]).unwrap(),
        EqPolynomial::evals(&point[..3]).unwrap(),
    )
    .unwrap();
    let mut claim = value;
    for round in 0..9 {
        let polynomial = batch.round_polynomial(round, claim).unwrap();
        assert_eq!(polynomial, dense.compute_round_univariate(round, claim));
        let challenge = E::from_u64(7 + round as u64 * 19);
        claim = polynomial.evaluate(challenge);
        batch.bind_challenge(round, challenge).unwrap();
        dense.ingest_challenge(round, challenge);
    }
    batch.finish().unwrap();
    assert_eq!(
        batch.setup_evaluation().unwrap(),
        dense.folded_table_value().unwrap()
    );
}

#[test]
fn fused_preparation_scans_once_and_preserves_answers_and_proof_bytes() {
    use jolt_field::{Ext2, FpExt4, Prime32Offset99, Prime64Offset59};
    fused_preparation::<F, F>();
    fused_preparation::<Prime32Offset99, FpExt4<Prime32Offset99>>();
    fused_preparation::<Prime64Offset59, Ext2<Prime64Offset59>>();
}

fn fused_preparation<B, E>()
where
    B: Field + CanonicalEncoding,
    E: FpExtEncoding<B> + ExtField<B> + Ring + MulBaseUnreduced<B>,
{
    let source = setup::<B>(512);
    let backend = CpuBackend::<B, E>::new(Arc::clone(&source)).unwrap();
    let polynomial = SetupPolynomialId::new::<B, E>(&source, 512).unwrap();
    let queries = (0..4)
        .map(|id| {
            SetupQuery::evaluation((0..9).map(|bit| E::from_u64(17 + id * 11 + bit)).collect())
                .unwrap()
        })
        .collect::<Vec<_>>();
    for memory in [SetupBatchMemory::Retain, SetupBatchMemory::Regenerate] {
        let mut claims = Vec::new();
        let mut separate = Vec::new();
        for query in &queries {
            let (value, prepared) = backend
                .prepare_setup_query(&polynomial, query, 8, memory)
                .unwrap();
            claims.push(SetupClaim::new(polynomial.clone(), query.clone(), value).unwrap());
            separate.push(prepared);
        }
        let statement = SetupBatchStatement::new(polynomial.clone(), 8, claims).unwrap();
        let expected = akita_prover::prove_setup_batch::<B, E, _>(
            &backend,
            &statement,
            b"fused-setup-preparation",
            separate,
        )
        .unwrap();
        PREPARATION_SWEEPS.with(|sweeps| sweeps.set(0));
        let fused = backend
            .prepare_setup_queries(&polynomial, &queries, 8, memory)
            .unwrap();
        PREPARATION_SWEEPS.with(|sweeps| assert_eq!(sweeps.get(), 1));
        for (state, claim) in fused.iter().zip(statement.claims()) {
            assert_eq!(state.value(), claim.value());
        }
        let actual = akita_prover::prove_setup_batch::<B, E, _>(
            &backend,
            &statement,
            b"fused-setup-preparation",
            fused,
        )
        .unwrap();
        assert_eq!(actual.argument(), expected.argument());
        PREPARATION_SWEEPS.with(|sweeps| assert_eq!(sweeps.get(), 1));
    }
}

#[test]
#[ignore = "native fused setup preparation benchmark"]
fn profile_fused_setup_preparation() {
    use std::time::Instant;
    let source = setup::<F>(1 << 18);
    let backend = CpuBackend::<F, F>::new(Arc::clone(&source)).unwrap();
    let polynomial = SetupPolynomialId::new::<F, F>(&source, 1 << 18).unwrap();
    for count in [1, 8, 32] {
        let queries = (0..count)
            .map(|id| {
                SetupQuery::evaluation((0..18).map(|bit| F::from_u64(17 + id * 19 + bit)).collect())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for _ in 0..3 {
            let start = Instant::now();
            let separate = queries
                .iter()
                .map(|query| {
                    backend
                        .prepare_setup_query(&polynomial, query, 64, SetupBatchMemory::Retain)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let separate_time = start.elapsed();
            let start = Instant::now();
            let fused = backend
                .prepare_setup_queries(&polynomial, &queries, 64, SetupBatchMemory::Retain)
                .unwrap();
            let fused_time = start.elapsed();
            for ((answer, _), state) in separate.iter().zip(&fused) {
                assert_eq!(*answer, state.value());
            }
            eprintln!("claims={count} separate={separate_time:?} fused={fused_time:?}");
        }
    }
}
