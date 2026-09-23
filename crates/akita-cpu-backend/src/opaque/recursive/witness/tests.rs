use super::*;
use crate::opaque::consumer_kernels::RelationWitnessSession;
use crate::opaque::eor::ExtensionOpeningSession;
use crate::opaque::{CpuBackend, RootOpeningSource, RootPolyShape};
use crate::sources::packed_digits::PackedSignedDigits;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use jolt_field::{One, Prime128OffsetA7F7 as F, Ring, Zero};

#[test]
fn suffix_batch_fold_rejects_mixed_extents_and_count_mismatch() {
    use crate::opaque::{
        CpuBackend, DecomposeFoldBatchPlan, OpeningBatchKernel, RootOpeningSource,
    };
    use akita_challenges::SparseChallenge;
    use akita_error::AkitaError;

    const D: usize = 64;
    let witnesses = [
        RecursiveWitnessFlat::from_i8_digits(vec![1; D]),
        RecursiveWitnessFlat::from_i8_digits(vec![1; 2 * D]),
    ];
    let challenges = vec![
        SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        2
    ];
    let backend = CpuBackend::for_arithmetic_tests();
    let run = |refs: &[&RecursiveWitnessFlat]| {
        OpeningBatchKernel::decompose_fold_batch(
            &backend,
            None,
            <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_batch(refs).unwrap(),
            DecomposeFoldBatchPlan::Sparse {
                challenges: &challenges,
                num_positions_per_block: 1,
                num_digits: 1,
                log_basis: 1,
            },
        )
    };

    assert!(matches!(
        run(&[&witnesses[0], &witnesses[1]]),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        run(&[&witnesses[0], &witnesses[0], &witnesses[0]]),
        Err(AkitaError::InvalidInput(_))
    ));
}

fn stage1_session() -> super::super::digit_range::DigitRangeSession<F> {
    let domain = akita_types::FlatBooleanDomain::new(4, 2).unwrap();
    let equality = akita_types::DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &[F::from_u64(3), F::from_u64(5)],
        1,
        1,
    )
    .unwrap();
    let prover = super::super::DigitRangeProver::from_packed_digits(
        PackedSignedDigits::from_i8_digits_auto(vec![-2, -1, 0, 1]),
        akita_types::DigitRangePlan::new(16).unwrap(),
        domain,
        equality,
    )
    .unwrap();
    super::super::digit_range::DigitRangeSession::new(prover, None).unwrap()
}

#[test]
fn stage1_session_rejects_skipped_repeated_reordered_and_mismatched_rounds() {
    let step = crate::opaque::Stage1Step::Product(0);

    let mut session = stage1_session();
    assert!(session.round_polynomial(step, 1, F::zero()).is_err());
    assert!(session
        .round_polynomial(crate::opaque::Stage1Step::RangeLeaf, 0, F::zero())
        .is_err());
    assert!(session.round_polynomial(step, 0, F::one()).is_err());
    assert!(session.bind_challenge(step, 0, F::one()).is_err());
    assert!(session.finish().is_err());

    let mut session = stage1_session();
    session.round_polynomial(step, 0, F::zero()).unwrap();
    assert!(session.round_polynomial(step, 0, F::zero()).is_err());
    assert!(session.bind_challenge(step, 1, F::one()).is_err());

    let mut session = stage1_session();
    session.round_polynomial(step, 0, F::zero()).unwrap();
    assert!(session
        .bind_challenge(crate::opaque::Stage1Step::Product(1), 0, F::one())
        .is_err());
}

fn two_value_eor_session() -> Box<dyn ExtensionOpeningSession<F>> {
    let term = crate::opaque::recursive::opening::ExtensionOpeningReductionTerm::new(
        vec![F::one(), F::from_u64(2)],
        F::one(),
    );
    let group = crate::opaque::recursive::opening::ExtensionOpeningReductionGroup::new(
        vec![term],
        vec![F::one(), F::one()],
    )
    .unwrap();
    cpu_extension_opening_session(group, F::from_u64(3)).unwrap()
}

#[test]
fn eor_session_rejects_round_and_challenge_misuse() {
    let mut session = two_value_eor_session();
    assert!(session.round_polynomial(1, F::from_u64(3)).is_err());
    assert!(session.bind_challenge(0, F::one()).is_err());

    let mut session = two_value_eor_session();
    session.round_polynomial(0, F::from_u64(3)).unwrap();
    assert!(session.round_polynomial(0, F::from_u64(3)).is_err());
    assert!(session.bind_challenge(1, F::one()).is_err());
    session.bind_challenge(0, F::from_u64(7)).unwrap();
    assert!(session.bind_challenge(0, F::one()).is_err());

    assert!(session.round_polynomial(1, F::zero()).is_err());
    assert!(session.finish().is_ok());

    assert!(two_value_eor_session().finish().is_err());
}

fn two_round_relation_session() -> ConsumerStage2Session<F> {
    let point = [F::from_u64(3), F::from_u64(5)];
    let witness = [-1_i8, 0, 1, 2];
    let values = witness.map(|value| F::from_i64(i64::from(value)));
    let range_image = values.map(|value| value * (value + F::one()));
    let evaluation = akita_algebra::poly::multilinear_eval(&range_image, &point).unwrap();
    let prover = super::super::relation_range_image::RelationRangeImageProver::new_virtual_only(
        witness.to_vec(),
        &point,
        evaluation,
        8,
        2,
        1,
        1,
    )
    .unwrap();
    let claim = akita_sumcheck::SumcheckInstanceProver::input_claim(&prover);
    ConsumerStage2Session::for_test(prover, claim)
}

#[test]
fn relation_session_rejects_round_and_challenge_misuse() {
    let mut session = two_round_relation_session();
    let claim = session.input_claim();
    assert!(session.round_polynomial(1, claim).is_err());
    assert!(session.round_polynomial(0, claim + F::one()).is_err());
    assert!(session.bind_challenge(0, F::one()).is_err());

    let mut session = two_round_relation_session();
    let claim = session.input_claim();
    session.round_polynomial(0, claim).unwrap();
    assert!(session.round_polynomial(0, claim).is_err());
    assert!(session.bind_challenge(1, F::one()).is_err());
    session.bind_challenge(0, F::from_u64(7)).unwrap();
    assert!(session.bind_challenge(0, F::one()).is_err());
    assert!(session.finish().is_err());

    let mut completed = two_round_relation_session();
    for round in 0..completed.num_rounds() {
        let claim = completed.input_claim();
        completed.round_polynomial(round, claim).unwrap();
        completed
            .bind_challenge(round, F::from_u64(round as u64 + 7))
            .unwrap();
    }
    let final_claim = completed.input_claim();
    assert!(completed
        .round_polynomial(completed.num_rounds(), final_claim)
        .is_err());
    assert!(completed.finish().is_ok());
}

#[test]
fn public_stage2_dispatch_rejects_exhausted_round_and_preserves_finish() {
    use crate::opaque::{OpaqueStage2Kernel, ProofContext, ProofScope};

    let backend = CpuBackend::for_arithmetic_tests();
    let scope_id = backend.owner().begin_test_scope(vec![1]).unwrap();
    let scope = ProofScope::admitted(
        &backend,
        crate::opaque::lifecycle::CpuProofSessionHandle::new(
            std::sync::Arc::clone(backend.owner()),
            scope_id,
        ),
    );
    let context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        scope.session().scope_id(),
        0,
    );
    let mut session = two_round_relation_session();
    let binding = backend.binding(&context).unwrap();
    session.set_operation_binding(binding, backend.binding_lease(&binding).unwrap());
    let rounds = OpaqueStage2Kernel::<F, F>::stage2_num_rounds(&backend, &session).unwrap();
    let mut claim = OpaqueStage2Kernel::<F, F>::stage2_input_claim(&backend, &session).unwrap();
    for round in 0..rounds {
        let polynomial = OpaqueStage2Kernel::<F, F>::stage2_round_polynomial(
            &backend,
            &mut session,
            round,
            claim,
        )
        .unwrap();
        let challenge = F::from_u64(round as u64 + 7);
        claim = polynomial.evaluate(&challenge);
        OpaqueStage2Kernel::<F, F>::bind_stage2_challenge(&backend, &mut session, round, challenge)
            .unwrap();
    }
    assert!(OpaqueStage2Kernel::<F, F>::stage2_round_polynomial(
        &backend,
        &mut session,
        rounds,
        claim,
    )
    .is_err());
    assert!(OpaqueStage2Kernel::<F, F>::finish_stage2(&backend, session).is_ok());
    scope.finish().unwrap();
}

#[test]
fn suffix_opening_views_share_packed_digit_storage() {
    const D: usize = 16;
    let digits: Vec<i8> = (0..64).map(|idx| (idx % 5) as i8 - 2).collect();
    let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
    let opening: SuffixWitnessView<'_, F, D> = witness.opening_view().expect("opening view");
    assert_eq!(
        opening.padded_ring_elems,
        <RecursiveWitnessFlat as RootPolyShape<F, D>>::num_ring_elems(&witness)
    );
    for (index, chunk) in digits.chunks_exact(D).enumerate() {
        assert_eq!(opening.ring_elem(index).unwrap().as_slice(), chunk);
    }

    let polys = [&witness];
    let batch = <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_batch(&polys)
        .expect("opening batch");
    assert_eq!(batch.polys.len(), 1);
}

#[test]
fn recursive_owner_keeps_only_exact_packed_digits() {
    const D: usize = 64;
    let witness = RecursiveWitnessFlat::from_i8_digits(vec![-4, -1, 0, 3, 2]);
    assert_eq!(witness.digits.bit_width(), 3);
    assert_eq!(witness.digits.encoded_bytes().len(), 2);

    let aligned = witness
        .align_for_commitment_ring_dim(D)
        .expect("commitment alignment");
    assert_eq!(aligned.committed_coeff_len().unwrap(), D);
    assert_eq!(aligned.digits.encoded_bytes().len(), 2);
    assert_eq!(aligned.to_i8_digits(), [-4, -1, 0, 3, 2]);
}

#[test]
fn commitment_padding_does_not_create_live_blocks() {
    const D: usize = 64;
    let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 70 * D])
        .align_for_commitment_ring_dim(D)
        .expect("commitment alignment");
    let view = witness.view::<F, D>().expect("aligned view");

    assert_eq!(view.live_ring_elems, 70);
    assert!(view.padded_ring_elems >= view.live_ring_elems);
    assert_eq!(view.num_live_blocks(10).expect("live blocks"), 7);
}

#[test]
fn opening_view_uses_commitment_domain_after_commitment_padding() {
    const D: usize = 64;
    let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 70 * D])
        .align_for_commitment_ring_dim(D)
        .expect("commitment alignment");

    let opened: SuffixWitnessView<'_, F, D> = witness.opening_view().expect("opening view");

    assert_eq!(witness.live_coeff_len(), 70 * D);
    assert_eq!(witness.digits.len(), witness.live_coeff_len());
    assert_eq!(opened.live_coeff_len, witness.live_coeff_len());
    assert_eq!(opened.digits.len(), witness.committed_coeff_len().unwrap());
    assert_eq!(opened.live_ring_elems, 70);
    assert_eq!(
        opened.padded_ring_elems * D,
        witness.committed_coeff_len().unwrap()
    );
    assert_eq!(opened.padded_ring_elems * D, 1 << 13);
    assert_eq!(opened.num_vars(), 13);
    for index in 0..opened.padded_ring_elems {
        let expected = if index < 70 { [1; D] } else { [0; D] };
        assert_eq!(opened.ring_elem(index), Some(expected));
    }
    assert!(opened.ring_elem(opened.padded_ring_elems).is_none());
}

#[test]
fn logical_rows_are_contiguous_for_partial_final_fold() {
    let digits: Vec<i8> = (0..20).collect();
    let w = RecursiveWitnessFlat::from_i8_digits(digits);
    let view = w.view::<jolt_field::Prime128OffsetA7F7, 2>().expect("view");
    let num_live_blocks = 4;
    let num_positions_per_block = (w.live_coeff_len() / 2).div_ceil(num_live_blocks);

    let row = |block_idx: usize| -> Vec<[i8; 2]> {
        (0..num_positions_per_block)
            .filter_map(|col_idx| view.block_elem(block_idx, col_idx, num_positions_per_block))
            .collect()
    };

    assert_eq!(row(0), vec![[0, 1], [2, 3], [4, 5]]);
    assert_eq!(row(1), vec![[6, 7], [8, 9], [10, 11]]);
    assert_eq!(row(2), vec![[12, 13], [14, 15], [16, 17]]);
    assert_eq!(row(3), vec![[18, 19]]);
}

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 4;
    let digits = vec![1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12];
    let w = RecursiveWitnessFlat::from_i8_digits(digits);
    let view = w.view::<F, D>().expect("view");
    let scalars = vec![ring::<D>(10), ring::<D>(20)];
    let got = view.fold_blocks_ring(&scalars, 2);

    let expected = (0..2)
        .map(|block_idx| {
            (0..2).fold(CyclotomicRing::<F, D>::zero(), |acc, col_idx| {
                let Some(digits) = view.block_elem(block_idx, col_idx, 2) else {
                    return acc;
                };
                let coeff = CyclotomicRing::from_coefficients(digits.map(F::from_i8));
                acc + coeff * scalars[col_idx]
            })
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[test]
fn fused_evaluation_uses_physical_order_with_partial_final_fold() {
    const D: usize = 4;
    let digits = (0..24).map(|idx| idx as i8 - 12).collect();
    let w = RecursiveWitnessFlat::from_i8_digits(digits);
    let view = w.view::<F, D>().expect("view");
    let num_positions_per_block = 4;
    let live_block_weights = vec![F::from_u64(2), F::from_u64(5)];
    let position_weights = vec![
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
    ];

    let expected_folded = view.fold_blocks(&position_weights, num_positions_per_block);
    let expected_eval = expected_folded
        .iter()
        .zip(live_block_weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
            acc + f_i.scale(s_i)
        });
    let (eval, folded) = view
        .evaluate_and_fold(
            &live_block_weights,
            &position_weights,
            num_positions_per_block,
        )
        .unwrap();

    assert_eq!(folded, expected_folded);
    assert_eq!(eval, expected_eval);
}

#[test]
fn fused_ring_evaluation_uses_physical_order_with_partial_final_fold() {
    const D: usize = 4;
    let digits = (0..24).map(|idx| idx as i8 - 12).collect();
    let w = RecursiveWitnessFlat::from_i8_digits(digits);
    let view = w.view::<F, D>().expect("view");
    let num_positions_per_block = 4;
    let live_block_weights = vec![ring::<D>(2), ring::<D>(5)];
    let position_weights = vec![ring::<D>(7), ring::<D>(11), ring::<D>(13), ring::<D>(17)];

    let expected_folded = view.fold_blocks_ring(&position_weights, num_positions_per_block);
    let expected_eval = expected_folded
        .iter()
        .zip(live_block_weights.iter())
        .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
            acc + (*f_i * *s_i)
        });
    let (eval, folded) = view
        .evaluate_and_fold_ring(
            &live_block_weights,
            &position_weights,
            num_positions_per_block,
        )
        .unwrap();

    assert_eq!(folded, expected_folded);
    assert_eq!(eval, expected_eval);
}

#[test]
fn suffix_witness_decompose_fold_is_deterministic() {
    const D: usize = 16;
    let digits = (0..48).map(|idx| (idx % 7) as i8 - 3).collect();
    let w = RecursiveWitnessFlat::from_i8_digits(digits);
    let view = w.view::<F, D>().expect("view");
    let challenges = vec![
        SparseChallenge {
            positions: vec![0, 2].into(),
            coeffs: vec![1, -1].into(),
        },
        SparseChallenge {
            positions: vec![1, 3].into(),
            coeffs: vec![2, 1].into(),
        },
    ];

    let once = view.decompose_fold(&challenges, 2, 1, 0).unwrap();
    let twice = view.decompose_fold(&challenges, 2, 1, 0).unwrap();
    assert_eq!(once, twice);
}

#[test]
fn backend_rejects_foreign_and_expired_sessions_independently() {
    use crate::opaque::{OpaqueStage1Kernel, OpaqueStage2Kernel, ProofContext, ProofScope};
    let backend = CpuBackend::for_arithmetic_tests();
    let scope_id = backend.owner().begin_test_scope(vec![1]).unwrap();
    let scope = ProofScope::admitted(
        &backend,
        crate::opaque::lifecycle::CpuProofSessionHandle::new(
            std::sync::Arc::clone(backend.owner()),
            scope_id,
        ),
    );
    let context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        scope.session().scope_id(),
        0,
    );
    let binding = backend.binding(&context).unwrap();
    let lease = backend.binding_lease(&binding).unwrap();
    let mut stage2 = two_round_relation_session();
    stage2.set_operation_binding(binding, lease.clone());
    let claim = OpaqueStage2Kernel::<F, F>::stage2_input_claim(&backend, &stage2).unwrap();
    assert_eq!(
        OpaqueStage2Kernel::<F, F>::stage2_num_rounds(&backend, &stage2).unwrap(),
        2
    );

    let foreign = CpuBackend::for_arithmetic_tests();
    let foreign_id = foreign.owner().begin_test_scope(vec![1]).unwrap();
    let foreign_scope = ProofScope::admitted(
        &foreign,
        crate::opaque::lifecycle::CpuProofSessionHandle::new(
            std::sync::Arc::clone(foreign.owner()),
            foreign_id,
        ),
    );
    let foreign_context = ProofContext::new(
        foreign.owner_id(),
        foreign.owner().setup_digest(),
        foreign_scope.session().scope_id(),
        0,
    );
    let second_id = backend.owner().begin_test_scope(vec![1]).unwrap();
    let second_scope = ProofScope::admitted(
        &backend,
        crate::opaque::lifecycle::CpuProofSessionHandle::new(
            std::sync::Arc::clone(backend.owner()),
            second_id,
        ),
    );
    let second_context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        second_scope.session().scope_id(),
        0,
    );
    let second_binding = backend.binding(&second_context).unwrap();
    let second_lease = backend.binding_lease(&second_binding).unwrap();
    let mut independent = two_round_relation_session();
    independent.set_operation_binding(second_binding, second_lease);
    let invalid_bindings = [
        foreign.binding(&foreign_context).unwrap(),
        binding.for_level_operation(99, binding.operation_id()),
        binding.with_group(Some(1)),
    ];
    for invalid in invalid_bindings {
        stage2.set_operation_binding(invalid, lease.clone());
        assert!(OpaqueStage2Kernel::<F, F>::stage2_input_claim(&backend, &stage2).is_err());
        assert!(OpaqueStage2Kernel::<F, F>::stage2_num_rounds(&backend, &stage2).is_err());
        assert!(OpaqueStage2Kernel::<F, F>::stage2_round_polynomial(
            &backend,
            &mut stage2,
            0,
            claim
        )
        .is_err());
        assert!(stage2.pending.is_none());
    }
    stage2.set_operation_binding(binding, lease.clone());
    OpaqueStage2Kernel::<F, F>::stage2_round_polynomial(&backend, &mut stage2, 0, claim).unwrap();
    let mut stage1 = CpuStage1SessionHandle {
        binding,
        lease,
        session_state: stage1_session(),
    };
    OpaqueStage1Kernel::<F, F>::stage1_round_polynomial(
        &backend,
        &mut stage1,
        crate::opaque::Stage1Step::Product(0),
        0,
        F::zero(),
    )
    .unwrap();
    scope.finish().unwrap();
    assert!(OpaqueStage2Kernel::<F, F>::stage2_input_claim(&backend, &stage2).is_err());
    assert!(OpaqueStage2Kernel::<F, F>::stage2_num_rounds(&backend, &stage2).is_err());
    assert!(
        OpaqueStage2Kernel::<F, F>::bind_stage2_challenge(&backend, &mut stage2, 0, F::one())
            .is_err()
    );
    assert!(OpaqueStage2Kernel::<F, F>::finish_stage2(&backend, stage2).is_err());
    assert!(OpaqueStage1Kernel::<F, F>::bind_stage1_challenge(
        &backend,
        &mut stage1,
        crate::opaque::Stage1Step::Product(0),
        0,
        F::one(),
    )
    .is_err());
    assert!(OpaqueStage1Kernel::<F, F>::finish_stage1(&backend, stage1).is_err());
    assert!(OpaqueStage2Kernel::<F, F>::stage2_input_claim(&backend, &independent).is_ok());
    drop(second_scope);
    assert!(OpaqueStage2Kernel::<F, F>::stage2_input_claim(&backend, &independent).is_err());
}
