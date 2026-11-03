use super::*;
use crate::opaque::{ProofContext, ProofScope};
use jolt_field::{One, Prime128OffsetA7F7 as F, Zero};

fn proof_scope<B: jolt_field::Field, E>(
    backend: &CpuBackend<B, E>,
) -> ProofScope<'_, CpuBackend<B, E>> {
    let scope = backend.owner().begin_test_scope(vec![1]).unwrap();
    ProofScope::admitted(
        backend,
        crate::opaque::lifecycle::CpuProofSessionHandle::new(Arc::clone(backend.owner()), scope),
    )
}

fn session(
    backend: &CpuBackend<F, F>,
    proof: &crate::opaque::CpuProofSessionHandle,
) -> (F, CpuStage3Session<F, F>) {
    let setup = SetupProductSumcheck::new(
        SetupProductSource::Table((1..=4).map(F::from_u64).collect()),
        2,
        vec![F::one(); 2],
        vec![F::one(); 2],
    )
    .unwrap();
    let binding = backend
        .binding(
            proof,
            &ProofContext::new(
                backend.owner_id(),
                backend.owner().setup_digest(),
                proof.scope_id(),
                0,
            ),
        )
        .unwrap();
    let lease = binding.scope_lease().clone();
    (
        setup.input_claim(),
        CpuStage3Session {
            binding,
            lease,
            setup: Stage3Setup::Bound(setup),
        },
    )
}

#[test]
fn stage3_enforces_round_bind_and_completion_order() {
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let scope = proof_scope(&backend);
    assert!(backend
        .finish_stage3(session(&backend, scope.session()).1)
        .is_err());
    let (mut claim, mut state) = session(&backend, scope.session());
    assert!(backend
        .bind_stage3_challenge(&mut state, 0, F::one())
        .is_err());
    assert!(backend
        .stage3_round_polynomial(&mut state, 1, claim)
        .is_err());
    assert!(backend
        .stage3_round_polynomial(&mut state, 0, F::zero())
        .is_err());
    let challenges = [F::from_u64(3), F::from_u64(5)];
    for (round, challenge) in challenges.into_iter().enumerate() {
        let polynomial = backend
            .stage3_round_polynomial(&mut state, round, claim)
            .unwrap();
        assert!(backend
            .stage3_round_polynomial(&mut state, round, claim)
            .is_err());
        assert!(backend
            .bind_stage3_challenge(&mut state, round + 1, challenge)
            .is_err());
        backend.trim_caches().unwrap();
        backend
            .bind_stage3_challenge(&mut state, round, challenge)
            .unwrap();
        assert!(backend
            .bind_stage3_challenge(&mut state, round, challenge)
            .is_err());
        claim = polynomial.evaluate(challenge);
    }
    assert!(backend
        .stage3_round_polynomial(&mut state, 2, claim)
        .is_err());
    let evaluation = backend.finish_stage3(state).unwrap();
    assert_eq!(
        evaluation,
        akita_algebra::poly::multilinear_eval(
            &(1..=4).map(F::from_u64).collect::<Vec<_>>(),
            &challenges,
        )
        .unwrap()
    );
    scope.finish().unwrap();
}

#[test]
fn stage3_scopes_are_independent_and_owner_bound() {
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let foreign = CpuBackend::<F, F>::for_arithmetic_tests();
    let first = proof_scope(&backend);
    let second = proof_scope(&backend);
    let (claim, mut a) = session(&backend, first.session());
    let (_, mut b) = session(&backend, second.session());
    assert!(foreign.stage3_round_polynomial(&mut a, 0, claim).is_err());
    drop(first);
    assert!(backend.stage3_round_polynomial(&mut a, 0, claim).is_err());
    assert!(backend.stage3_round_polynomial(&mut b, 0, claim).is_ok());
    second.finish().unwrap();
    assert!(backend.bind_stage3_challenge(&mut b, 0, F::one()).is_err());
}

#[test]
fn stage3_retains_setup_across_cache_eviction() {
    use crate::opaque::ComputeBackendSetup;
    type Base = jolt_field::Prime64Offset59;
    type Extension =
        <akita_config::proof_optimized::fp64::OneHot as akita_config::CommitmentConfig>::ExtField;
    let setup = crate::AkitaProverSetup::<Base>::generate_with_capacity(
        2,
        1,
        akita_params::SetupMatrixCapacity {
            num_field_elements: 64,
        },
    )
    .unwrap();
    let expected = setup.expanded.shared_matrix().as_field_slice()[..4].to_vec();
    let allocation = std::sync::Arc::downgrade(&setup.expanded);
    let backend = CpuBackend::<Base, Extension>::new(setup.expanded.clone()).unwrap();
    let scope = proof_scope(&backend);
    let product = SetupProductSumcheck::new(
        SetupProductSource::Expanded(setup.expanded.clone()),
        2,
        vec![Extension::one(); 2],
        vec![Extension::one(); 2],
    )
    .unwrap();
    let binding = backend
        .binding(
            scope.session(),
            &ProofContext::new(
                backend.owner_id(),
                backend.owner().setup_digest(),
                scope.session().scope_id(),
                0,
            ),
        )
        .unwrap();
    let lease = binding.scope_lease().clone();
    let mut claim = product.input_claim();
    let mut state = CpuStage3Session {
        binding,
        lease,
        setup: Stage3Setup::Bound(product),
    };
    drop(setup);
    backend
        .ensure_ntt_slot(
            backend.prepared().unwrap(),
            akita_types::NttCacheKey {
                ring_d: 64,
                num_ring_elements: 1,
                domain: akita_types::NttTransformDomain::Negacyclic,
            },
        )
        .unwrap();
    let challenges = [
        Extension::lift_base(Base::from_u64(3)),
        Extension::lift_base(Base::from_u64(5)),
    ];
    for (round, challenge) in challenges.into_iter().enumerate() {
        let polynomial = backend
            .stage3_round_polynomial(&mut state, round, claim)
            .unwrap();
        claim = polynomial.evaluate(challenge);
        let released = backend.trim_caches().unwrap();
        if round == 0 {
            assert!(released > 0);
        }
        assert!(allocation.upgrade().is_some());
        backend
            .bind_stage3_challenge(&mut state, round, challenge)
            .unwrap();
    }
    assert_eq!(
        backend.finish_stage3(state).unwrap(),
        akita_algebra::poly::multilinear_eval(
            &expected
                .into_iter()
                .map(Extension::lift_base)
                .collect::<Vec<_>>(),
            &challenges,
        )
        .unwrap()
    );
    scope.finish().unwrap();
    drop(backend);
    assert!(allocation.upgrade().is_none());
}
