use super::*;
use crate::opaque::{CpuProofSessionHandle, ProofScope};
use jolt_field::{One, Prime128OffsetA7F7 as F, Zero};

#[test]
fn stage3_alternate_table_uses_ordinary_rounds_and_terminal_value() {
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let scope_id = backend.owner().begin_test_scope(vec![1]).unwrap();
    let scope = ProofScope::admitted(
        &backend,
        CpuProofSessionHandle::new(Arc::clone(backend.owner()), scope_id),
    );
    let coefficients: Arc<[F]> = (1..=8).map(F::from_u64).collect::<Vec<_>>().into();
    let index_factor = vec![F::from_u64(7), F::from_u64(11)];
    let coefficient_factor = vec![F::from_u64(3), F::from_u64(5), F::zero(), F::one()];
    let mut reference = RectangularSetupProductTerm::new(
        SetupProductSource::Table(coefficients.to_vec()),
        2,
        index_factor.clone(),
        coefficient_factor.clone(),
    )
    .unwrap();
    let (mut claim, mut session) = backend
        .begin_factored_stage3(FactoredStage3Request {
            session: scope.session(),
            level: 0,
            coefficients,
            index_factor,
            coefficient_factor,
        })
        .unwrap();
    assert_eq!(claim, reference.input_claim());
    for round in 0..reference.num_rounds() {
        let polynomial = backend
            .stage3_round_polynomial(&mut session, round, claim)
            .unwrap();
        assert_eq!(polynomial, reference.compute_round_univariate());
        let challenge = F::from_u64(13 + round as u64);
        backend
            .bind_stage3_challenge(&mut session, round, challenge)
            .unwrap();
        reference.ingest_challenge(round, challenge);
        claim = polynomial.evaluate(challenge);
    }
    assert_eq!(
        backend.finish_stage3(session).unwrap(),
        reference.folded_table_value().unwrap()
    );
}

#[test]
fn stage3_alternate_table_rejects_invalid_dimensions() {
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let scope_id = backend.owner().begin_test_scope(vec![1]).unwrap();
    let scope = ProofScope::admitted(
        &backend,
        CpuProofSessionHandle::new(Arc::clone(backend.owner()), scope_id),
    );
    for (index_len, coefficient_len, source_len) in
        [(0, 2, 0), (3, 2, 6), (2, 0, 0), (2, 2, 3), (2, 2, 5)]
    {
        let result = backend.begin_factored_stage3(FactoredStage3Request {
            session: scope.session(),
            level: 0,
            coefficients: vec![F::one(); source_len].into(),
            index_factor: vec![F::one(); index_len],
            coefficient_factor: vec![F::one(); coefficient_len],
        });
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
}
