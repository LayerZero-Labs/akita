#![cfg(feature = "external-setup-accumulation")]
use akita_algebra::poly::multilinear_eval;
use akita_cpu_backend::CpuBackend;
use akita_prover::backend::{SetupBatchBackend, SetupBatchKernel};
use akita_prover::prove_setup_batch;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_sumcheck::SumcheckKernel;
use akita_types::{
    derive_public_matrix_prefix,
    setup_claims::{SetupAccumulator, SetupBatchProof, SetupClaim},
    AkitaSetupDescriptor,
};
use akita_types::{
    setup_claims::{SetupBatchMemory, SetupBatchStatement, SetupPolynomialId, SetupQuery},
    AkitaExpandedSetup, FpExtEncoding,
};
use akita_verifier::{decide_setup_accumulator, verify_setup_batch_deferred};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring, Zero};
use jolt_field::{FpExt4, One, Prime128OffsetA7F7 as F, Prime32Offset99};
use std::sync::Arc;

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

fn batch<B, E>(memory: SetupBatchMemory) -> Vec<u8>
where
    B: Field + CanonicalEncoding,
    E: FpExtEncoding<B> + ExtField<B> + Ring + MulBaseUnreduced<B>,
{
    let source = setup::<B>(512);
    let backend = CpuBackend::<B, E>::new(Arc::clone(&source)).unwrap();
    let polynomial = SetupPolynomialId::new::<B, E>(&source, 512).unwrap();
    let mut claims = Vec::new();
    let mut prepared = Vec::new();
    for query_id in 0..3 {
        let point = (0..9)
            .map(|i| E::from_u64(17 + query_id * 13 + i))
            .collect::<Vec<_>>();
        let query = SetupQuery::evaluation(point).unwrap();
        let (value, state) = backend
            .prepare_setup_query(&polynomial, &query, 8, memory)
            .unwrap();
        claims.push(SetupClaim::new(polynomial.clone(), query, value).unwrap());
        prepared.push(state);
    }
    let statement = SetupBatchStatement::new(polynomial.clone(), 8, claims).unwrap();
    let proof =
        prove_setup_batch::<B, E, _>(&backend, &statement, b"setup-batch-test", prepared).unwrap();
    let accumulator =
        verify_setup_batch_deferred::<B, E>(&polynomial, &statement, &proof, b"setup-batch-test")
            .unwrap();
    decide_setup_accumulator(&source, &accumulator).unwrap();
    assert!(
        verify_setup_batch_deferred::<B, E>(&polynomial, &statement, &proof, b"wrong-context")
            .is_err()
    );
    for end in 0..proof.argument().len() {
        let truncated = SetupBatchProof::from_argument(proof.argument()[..end].to_vec());
        assert!(verify_setup_batch_deferred::<B, E>(
            &polynomial,
            &statement,
            &truncated,
            b"setup-batch-test"
        )
        .is_err());
    }
    let mut trailing = proof.argument().to_vec();
    trailing.push(0);
    assert!(verify_setup_batch_deferred::<B, E>(
        &polynomial,
        &statement,
        &SetupBatchProof::from_argument(trailing),
        b"setup-batch-test"
    )
    .is_err());
    // Carry the authenticated reduction endpoint into another batch with a fresh query.
    let inherited = accumulator.into_claim().unwrap();
    let (value, inherited_state) = backend
        .prepare_setup_query(&polynomial, inherited.query(), 8, memory)
        .unwrap();
    assert_eq!(value, inherited.value());
    let query = SetupQuery::evaluation((0..9).map(|i| E::from_u64(41 + i)).collect()).unwrap();
    let (value, state) = backend
        .prepare_setup_query(&polynomial, &query, 8, memory)
        .unwrap();
    let new_claim = SetupClaim::new(polynomial.clone(), query, value).unwrap();
    let next = SetupBatchStatement::new(polynomial.clone(), 8, vec![inherited, new_claim]).unwrap();
    let proof =
        prove_setup_batch::<B, E, _>(&backend, &next, b"next-layer", vec![inherited_state, state])
            .unwrap();
    let accumulator =
        verify_setup_batch_deferred::<B, E>(&polynomial, &next, &proof, b"next-layer").unwrap();
    decide_setup_accumulator(&source, &accumulator).unwrap();
    proof.into_argument()
}

#[test]
fn retained_and_regenerated_batches_cover_base_and_extension_fields() {
    use akita_config::proof_optimized::fp64;
    assert_eq!(
        batch::<fp64::Field, fp64::ExtensionField>(SetupBatchMemory::Retain),
        batch::<fp64::Field, fp64::ExtensionField>(SetupBatchMemory::Regenerate)
    );
    assert_eq!(
        batch::<F, F>(SetupBatchMemory::Retain),
        batch::<F, F>(SetupBatchMemory::Regenerate)
    );
    assert_eq!(
        batch::<Prime32Offset99, FpExt4<Prime32Offset99>>(SetupBatchMemory::Retain),
        batch::<Prime32Offset99, FpExt4<Prime32Offset99>>(SetupBatchMemory::Regenerate)
    );
}

#[test]
fn prefix_embedding_and_accumulator_transport_are_bound_to_domain() {
    let source = setup::<F>(512);
    let small = SetupPolynomialId::new::<F, F>(&source, 128).unwrap();
    let large = SetupPolynomialId::new::<F, F>(&source, 512).unwrap();
    let point = (0..7).map(|i| F::from_u64(9 + i)).collect::<Vec<_>>();
    let value = multilinear_eval(&source.shared_matrix().as_field_slice()[..128], &point).unwrap();
    let accumulator = SetupAccumulator::new(small.clone(), point, value).unwrap();
    let mut bytes = Vec::new();
    accumulator.serialize_compressed(&mut bytes).unwrap();
    assert_eq!(bytes.len(), accumulator.compressed_size());
    let decoded = SetupAccumulator::<F>::deserialize_compressed_exact(&bytes, &small).unwrap();
    assert_eq!(decoded, accumulator);
    assert!(SetupAccumulator::<F>::deserialize_compressed_exact(&bytes, &large).is_err());
    let embedded = decoded.into_claim_for(large.clone()).unwrap();
    let backend = CpuBackend::<F, F>::new(Arc::clone(&source)).unwrap();
    let (got, _) = backend
        .prepare_setup_query(&large, embedded.query(), 8, SetupBatchMemory::Regenerate)
        .unwrap();
    assert_eq!(got, value);
}

#[test]
fn false_setup_answers_are_rejected_by_final_discharge() {
    let source = setup::<F>(512);
    let mut malicious = (*source).clone();
    malicious.shared_matrix = akita_params::FlatMatrix::from_flat_data(
        source
            .shared_matrix()
            .as_field_slice()
            .iter()
            .map(|&v| v + F::one())
            .collect(),
    );
    let backend = CpuBackend::<F, F>::new(Arc::new(malicious)).unwrap();
    let domain = SetupPolynomialId::new::<F, F>(&source, 512).unwrap();
    let query = SetupQuery::evaluation(vec![F::from_u64(7); 9]).unwrap();
    let (value, prepared) = backend
        .prepare_setup_query(&domain, &query, 8, SetupBatchMemory::Retain)
        .unwrap();
    let statement = SetupBatchStatement::new(
        domain.clone(),
        8,
        vec![SetupClaim::new(domain.clone(), query, value).unwrap()],
    )
    .unwrap();
    let proof =
        prove_setup_batch::<F, F, _>(&backend, &statement, b"untrusted-answer", vec![prepared])
            .unwrap();
    let accumulator =
        verify_setup_batch_deferred::<F, F>(&domain, &statement, &proof, b"untrusted-answer")
            .unwrap();
    assert!(decide_setup_accumulator(&source, &accumulator).is_err());
}

#[test]
fn preparation_rejects_cross_owner_query_and_round_reuse() {
    let source = setup::<F>(512);
    let first = CpuBackend::<F, F>::new(Arc::clone(&source)).unwrap();
    let second = CpuBackend::<F, F>::new(Arc::clone(&source)).unwrap();
    let domain = SetupPolynomialId::new::<F, F>(&source, 512).unwrap();
    let query = SetupQuery::evaluation(vec![F::from_u64(9); 9]).unwrap();
    let (value, prepared) = first
        .prepare_setup_query(&domain, &query, 8, SetupBatchMemory::Retain)
        .unwrap();
    let statement = SetupBatchStatement::new(
        domain.clone(),
        8,
        vec![SetupClaim::new(domain, query, value).unwrap()],
    )
    .unwrap();
    assert!(second
        .begin_setup_batch(&statement, vec![prepared], &[F::one()])
        .is_err());
    let (_, prepared) = first
        .prepare_setup_query(
            statement.polynomial(),
            statement.claims()[0].query(),
            8,
            SetupBatchMemory::Retain,
        )
        .unwrap();
    let mut kernel = first
        .begin_setup_batch(&statement, vec![prepared], &[F::one()])
        .unwrap();
    assert!(kernel.bind_challenge(0, F::one()).is_err());
    assert!(kernel.finish().is_err());
    kernel.round_polynomial(0, value).unwrap();
    assert!(kernel.round_polynomial(0, value).is_err());
}

/// Opt-in native microbenchmark: includes advice preparation and compares equivalent separate reductions.
#[test]
#[ignore = "native setup batch microbenchmark"]
fn profile_setup_batch_preparation_and_shared_rounds() {
    use std::time::Instant;
    let source = setup::<F>(1 << 16);
    let backend = CpuBackend::<F, F>::new(Arc::clone(&source)).unwrap();
    let domain = SetupPolynomialId::new::<F, F>(&source, 1 << 16).unwrap();
    for count in [1, 8, 32] {
        for memory in [SetupBatchMemory::Retain, SetupBatchMemory::Regenerate] {
            for together in [false, true] {
                let start = Instant::now();
                let mut claims = Vec::new();
                let mut preparations = Vec::new();
                let mut proof_bytes = 0;
                for query_id in 0..count {
                    let query = SetupQuery::evaluation(
                        (0..16)
                            .map(|i| F::from_u64(17 + query_id * 19 + i))
                            .collect(),
                    )
                    .unwrap();
                    let (value, prepared) = backend
                        .prepare_setup_query(&domain, &query, 64, memory)
                        .unwrap();
                    let claim = SetupClaim::new(domain.clone(), query, value).unwrap();
                    if together {
                        claims.push(claim);
                        preparations.push(prepared);
                    } else {
                        let statement =
                            SetupBatchStatement::new(domain.clone(), 64, vec![claim]).unwrap();
                        proof_bytes += prove_setup_batch::<F, F, _>(
                            &backend,
                            &statement,
                            b"profile-separate",
                            vec![prepared],
                        )
                        .unwrap()
                        .argument()
                        .len();
                    }
                }
                if together {
                    let statement = SetupBatchStatement::new(domain.clone(), 64, claims).unwrap();
                    proof_bytes = prove_setup_batch::<F, F, _>(
                        &backend,
                        &statement,
                        b"profile-together",
                        preparations,
                    )
                    .unwrap()
                    .argument()
                    .len();
                }
                eprintln!("claims={count} memory={memory:?} together={together} elapsed={:?} proof_bytes={proof_bytes}",start.elapsed());
            }
        }
    }
}

#[test]
fn zero_terminal_weight_and_zero_round_domains_need_no_division() {
    let source = setup::<F>(16);
    let backend = CpuBackend::<F, F>::new(Arc::clone(&source)).unwrap();
    let domain = SetupPolynomialId::new::<F, F>(&source, 16).unwrap();
    let query = SetupQuery::evaluation(vec![F::zero(); 4]).unwrap();
    assert!(backend
        .prepare_setup_query(&domain, &query, 1, SetupBatchMemory::Retain)
        .is_err());
    let (value, prepared) = backend
        .prepare_setup_query(&domain, &query, 4, SetupBatchMemory::Regenerate)
        .unwrap();
    let statement = SetupBatchStatement::new(
        domain.clone(),
        4,
        vec![SetupClaim::new(domain, query, value).unwrap()],
    )
    .unwrap();
    let mut kernel = backend
        .begin_setup_batch(&statement, vec![prepared], &[F::one()])
        .unwrap();
    let mut claim = value;
    for round in 0..4 {
        let polynomial = kernel.round_polynomial(round, claim).unwrap();
        assert_eq!(
            polynomial.evaluate(F::zero()) + polynomial.evaluate(F::one()),
            claim
        );
        claim = polynomial.evaluate(F::one());
        kernel.bind_challenge(round, F::one()).unwrap();
    }
    kernel.finish().unwrap();
    assert_eq!(claim, F::zero());
    let expected = source.shared_matrix().as_field_slice()[15];
    assert_eq!(kernel.setup_evaluation().unwrap(), expected);
    let scalar_domain = SetupPolynomialId::new::<F, F>(&source, 1).unwrap();
    let query = SetupQuery::evaluation(Vec::new()).unwrap();
    let (value, prepared) = backend
        .prepare_setup_query(&scalar_domain, &query, 1, SetupBatchMemory::Retain)
        .unwrap();
    let statement = SetupBatchStatement::new(
        scalar_domain.clone(),
        1,
        vec![SetupClaim::new(scalar_domain.clone(), query, value).unwrap()],
    )
    .unwrap();
    let proof =
        prove_setup_batch::<F, F, _>(&backend, &statement, b"zero-round", vec![prepared]).unwrap();
    let accumulator =
        verify_setup_batch_deferred::<F, F>(&scalar_domain, &statement, &proof, b"zero-round")
            .unwrap();
    decide_setup_accumulator(&source, &accumulator).unwrap();
}
