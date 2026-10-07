use super::*;
use crate::opaque::CpuBackend;
use crate::opaque::OpeningBatchKernel;
use akita_challenges::{Challenges, SparseChallenge};
use jolt_field::{One, Ring};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

type F = jolt_field::Prime128Offset275;
const D: usize = 4;
static BATCH_CALLS: AtomicUsize = AtomicUsize::new(0);
static SOURCE_TRAVERSALS: AtomicUsize = AtomicUsize::new(0);
static RECORDING_LOCK: Mutex<()> = Mutex::new(());

struct RecordingBatch {
    centered: Vec<[i32; D]>,
}

impl OpeningBatchKernel<RecordingBatch, F, D> for CpuBackend<F, F> {
    fn evaluate_and_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: RecordingBatch,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<Vec<OpeningFoldOutput<F, D>>, AkitaError> {
        BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
        SOURCE_TRAVERSALS.fetch_add(1, Ordering::Relaxed);
        plan.validate::<D>(2)?;
        Ok(source
            .centered
            .iter()
            .map(|row| {
                let eval = akita_algebra::CyclotomicRing::from_coefficients(
                    [F::from_u64(row[0] as u64); D],
                );
                OpeningFoldOutput {
                    eval,
                    folded: vec![eval; row[1] as usize],
                }
            })
            .collect())
    }

    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: RecordingBatch,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
        SOURCE_TRAVERSALS.fetch_add(1, Ordering::Relaxed);
        match plan {
            DecomposeFoldBatchPlan::Sparse { .. } => {
                let centered = source.centered.first().copied().unwrap_or([0; D]);
                Ok(CpuFoldResponses::sparse(
                    DecomposeFoldWitness::from_centered_rows::<D>(vec![centered]),
                ))
            }
            DecomposeFoldBatchPlan::SparseChunked { chunk_ranges, .. } => {
                CpuFoldResponses::chunked::<D>(
                    chunk_ranges
                        .iter()
                        .enumerate()
                        .map(|(index, _)| {
                            DecomposeFoldWitness::from_centered_rows::<D>(vec![source
                                .centered
                                .get(index)
                                .copied()
                                .unwrap_or([0; D])])
                        })
                        .collect(),
                )
            }
        }
    }
}

fn fold_endpoint_probe(
    centered: Vec<[i32; D]>,
    chunked: bool,
) -> FoldProbeOutcome<CpuAcceptedFold<F>> {
    let blocks = if chunked { 2 } else { 1 };
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: Vec::new().into(),
                coeffs: Vec::new().into(),
            };
            blocks
        ],
        blocks,
        1,
    )
    .unwrap();
    let ranges = akita_params::dyadic_block_ranges(blocks, blocks).unwrap();
    let geometry = if chunked {
        FoldProbeGeometry::SparseChunked {
            chunk_ranges: &ranges,
        }
    } else {
        FoldProbeGeometry::Sparse
    };
    let plan = ValidatedFoldProbePlan::new::<D>(
        &challenges,
        1,
        blocks,
        geometry,
        1,
        1,
        1,
        akita_params::OpeningMethod::EvaluationTrace,
        ValidatedFoldAcceptancePlan::new(128, 127, None),
    )
    .unwrap();
    FoldResponseKernel::probe(
        &CpuBackend::<F, F>::for_arithmetic_tests(),
        None,
        RecordingBatch { centered },
        &plan,
    )
    .unwrap()
}

fn assert_fold_endpoint_acceptance(centered: Vec<[i32; D]>, chunked: bool, accepted: bool) {
    assert_eq!(
        matches!(
            fold_endpoint_probe(centered, chunked),
            FoldProbeOutcome::Accepted { .. }
        ),
        accepted
    );
}

#[test]
fn sparse_and_chunked_admission_use_sign_aware_digit_endpoints() {
    let _guard = RECORDING_LOCK.lock().unwrap();
    for chunked in [false, true] {
        let repeat = if chunked { 2 } else { 1 };
        assert_fold_endpoint_acceptance(vec![[-128; D]; repeat], chunked, true);
        assert_fold_endpoint_acceptance(vec![[127; D]; repeat], chunked, true);
        assert_fold_endpoint_acceptance(vec![[-128, 127, -1, 0]; repeat], chunked, true);
        assert_fold_endpoint_acceptance(vec![[-129, 0, 0, 0]; repeat], chunked, false);
        assert_fold_endpoint_acceptance(vec![[128, 0, 0, 0]; repeat], chunked, false);
    }
}

#[test]
fn chunked_probe_dispatches_and_traverses_once() {
    let _guard = RECORDING_LOCK.lock().unwrap();
    for (blocks, chunks) in [(8, 2), (8, 4), (8, 8), (2, 8)] {
        BATCH_CALLS.store(0, Ordering::Relaxed);
        SOURCE_TRAVERSALS.store(0, Ordering::Relaxed);
        let challenges = Challenges::from_sparse(
            vec![
                SparseChallenge {
                    positions: Vec::new().into(),
                    coeffs: Vec::new().into(),
                };
                blocks
            ],
            blocks,
            1,
        )
        .unwrap();
        let ranges = akita_params::dyadic_block_ranges(blocks, chunks).unwrap();
        let plan = ValidatedFoldProbePlan::new::<D>(
            &challenges,
            1,
            blocks,
            FoldProbeGeometry::SparseChunked {
                chunk_ranges: &ranges,
            },
            1,
            1,
            1,
            akita_params::OpeningMethod::EvaluationTrace,
            crate::opaque::ValidatedFoldAcceptancePlan::new(u128::MAX, u128::MAX, None),
        )
        .unwrap();
        let outcome = FoldResponseKernel::probe(
            &CpuBackend::<F, F>::for_arithmetic_tests(),
            None,
            RecordingBatch {
                centered: Vec::new(),
            },
            &plan,
        )
        .unwrap();
        let FoldProbeOutcome::Accepted { fold_handle, .. } = outcome else {
            panic!("expected admitted fold");
        };
        assert!(fold_handle.validate_challenges(&challenges).is_ok());
        let mut changed = challenges.as_slice().to_vec();
        changed[0] = SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        let changed = Challenges::from_sparse(changed, blocks, 1).unwrap();
        assert!(fold_handle.validate_challenges(&changed).is_err());
        assert_eq!(BATCH_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(SOURCE_TRAVERSALS.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn chunk_admission_precedes_cancelling_aggregation() {
    let _guard = RECORDING_LOCK.lock().unwrap();
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: Vec::new().into(),
                coeffs: Vec::new().into(),
            };
            2
        ],
        2,
        1,
    )
    .unwrap();
    let ranges = akita_params::dyadic_block_ranges(2, 2).unwrap();
    let plan = ValidatedFoldProbePlan::new::<D>(
        &challenges,
        1,
        2,
        FoldProbeGeometry::SparseChunked {
            chunk_ranges: &ranges,
        },
        1,
        1,
        1,
        akita_params::OpeningMethod::EvaluationTrace,
        crate::opaque::ValidatedFoldAcceptancePlan::new(10, 10, None),
    )
    .unwrap();
    let outcome = FoldResponseKernel::probe(
        &CpuBackend::<F, F>::for_arithmetic_tests(),
        None,
        RecordingBatch {
            centered: vec![[20; D], [-20; D]],
        },
        &plan,
    )
    .unwrap();
    assert!(matches!(outcome, FoldProbeOutcome::Rejected));
}

#[test]
fn chunk_aggregation_sums_exactly_and_accepts_empty_chunks() {
    let chunks = vec![
        DecomposeFoldWitness::from_centered_rows::<D>(vec![[1, 2, 3, 4]]),
        DecomposeFoldWitness::from_centered_rows::<D>(vec![[0; D]]),
        DecomposeFoldWitness::from_centered_rows::<D>(vec![[4, 3, 2, 1]]),
    ];
    let responses = CpuFoldResponses::chunked::<D>(chunks).unwrap();
    assert_eq!(responses.global.centered_coeffs_flat(), &[5, 5, 5, 5]);
}

#[test]
fn opening_bindings_reject_foreign_scopes_levels_groups_and_computations() {
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let proof = backend.owner().begin_test_scope(vec![2, 1]).unwrap();
    let session =
        crate::opaque::CpuProofSessionHandle::new(std::sync::Arc::clone(backend.owner()), proof);
    let context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        session.scope_id(),
        0,
    )
    .for_group(0);
    let opening = backend.binding(&session, &context).unwrap();
    let second_opening = backend.binding(&session, &context).unwrap();
    assert!(opening.validate_lineage(&second_opening).is_ok());
    assert!(opening.validate_computation(&second_opening).is_err());
    assert!(opening
        .validate_computation(&opening.clone().with_group(Some(1)))
        .is_err());
    assert!(opening
        .validate_lineage(&opening.for_level_operation(1, opening.operation_id()))
        .is_err());
    let foreign = CpuBackend::<F, F>::for_arithmetic_tests();
    assert!(foreign.validate_binding(&opening).is_err());
    let independent = backend.owner().begin_test_scope(vec![2, 1]).unwrap();
    let independent_session = crate::opaque::CpuProofSessionHandle::new(
        std::sync::Arc::clone(backend.owner()),
        independent,
    );
    let independent_context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        independent_session.scope_id(),
        0,
    )
    .for_group(0);
    let independent_opening = backend
        .binding(&independent_session, &independent_context)
        .unwrap();
    assert!(opening.validate_lineage(&independent_opening).is_err());
    backend.finish_scope(&session).unwrap();
    assert!(backend.validate_binding(&opening).is_err());
    assert!(backend.validate_binding(&independent_opening).is_ok());
    backend.abort_scope_best_effort(&independent_session);
    assert!(backend.validate_binding(&independent_opening).is_err());
}

#[test]
fn accepted_fold_rejects_substituted_challenges_and_opening_computation() {
    let _guard = RECORDING_LOCK.lock().unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let proof = backend.owner().begin_test_scope(vec![1]).unwrap();
    let session =
        crate::opaque::CpuProofSessionHandle::new(std::sync::Arc::clone(backend.owner()), proof);
    let context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        session.scope_id(),
        0,
    )
    .for_group(0);
    let first_opening = backend.binding(&session, &context).unwrap();
    let other_opening = backend.binding(&session, &context).unwrap();
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![0].into(),
                coeffs: vec![1].into(),
            };
            2
        ],
        2,
        1,
    )
    .unwrap();
    let ranges = akita_params::dyadic_block_ranges(2, 2).unwrap();
    let plan = ValidatedFoldProbePlan::new::<D>(
        &challenges,
        1,
        2,
        FoldProbeGeometry::SparseChunked {
            chunk_ranges: &ranges,
        },
        1,
        1,
        1,
        akita_params::OpeningMethod::EvaluationTrace,
        ValidatedFoldAcceptancePlan::new(u128::MAX, u128::MAX, None),
    )
    .unwrap();
    let accepted = FoldResponseKernel::probe(
        &backend,
        None,
        RecordingBatch {
            centered: vec![[1; D], [2; D]],
        },
        &plan,
    )
    .unwrap();
    let FoldProbeOutcome::Accepted {
        mut fold_handle, ..
    } = accepted
    else {
        panic!("expected accepted fold")
    };
    fold_handle.bind(first_opening.clone());
    fold_handle.validate_challenges(&challenges).unwrap();
    fold_handle
        .binding()
        .validate_computation(&first_opening)
        .unwrap();
    assert!(fold_handle
        .binding()
        .validate_computation(&other_opening)
        .is_err());
    let substituted = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![1].into(),
                coeffs: vec![1].into(),
            };
            2
        ],
        2,
        1,
    )
    .unwrap();
    assert!(fold_handle.validate_challenges(&substituted).is_err());
    backend.finish_scope(&session).unwrap();
    assert!(backend.validate_binding(&fold_handle.binding()).is_err());
}

impl RootPolyShape<F, D> for RecordingBatch {
    fn num_ring_elems(&self) -> usize {
        2
    }
}
impl RootOpeningSource<F, D> for RecordingBatch {
    type OpeningView<'a> = ();
    type OpeningBatchView<'a> = Self;
    fn opening_view(&self) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "singleton evaluation must not be called".into(),
        ))
    }
    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self, AkitaError> {
        Ok(Self {
            centered: polys
                .iter()
                .flat_map(|poly| poly.centered.iter().copied())
                .collect(),
        })
    }
}

#[test]
fn opening_evaluation_batches_once_in_source_order_and_rejects_bad_shapes() {
    let _guard = RECORDING_LOCK.lock().unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let first = RecordingBatch {
        centered: vec![[10, 2, 0, 0]],
    };
    let second = RecordingBatch {
        centered: vec![[20, 2, 0, 0]],
    };
    let missing = RecordingBatch {
        centered: Vec::new(),
    };
    let short = RecordingBatch {
        centered: vec![[30, 1, 0, 0]],
    };
    let extra = RecordingBatch {
        centered: vec![[40, 2, 0, 0]; 2],
    };
    for (polys, valid) in [
        ([&second, &first], true),
        ([&first, &missing], false),
        ([&first, &short], false),
        ([&extra, &first], false),
    ] {
        BATCH_CALLS.store(0, Ordering::Relaxed);
        SOURCE_TRAVERSALS.store(0, Ordering::Relaxed);
        let result = super::source::prepare_and_evaluate_opening_group::<F, F, RecordingBatch, _, D>(
            &backend,
            None,
            &polys,
            &[F::one(); 3],
            akita_params::BasisMode::Lagrange,
            1,
            2,
            2,
        );
        assert_eq!(result.is_ok(), valid);
        if let Ok((_, (evals, folded))) = result {
            let expected = [20, 10].map(|value| {
                akita_algebra::CyclotomicRing::from_coefficients([F::from_u64(value); D])
            });
            assert_eq!(evals, expected);
            assert_eq!(folded, expected.map(|eval| vec![eval; 2]));
        }
        assert_eq!(BATCH_CALLS.load(Ordering::Relaxed), 1);
        assert_eq!(SOURCE_TRAVERSALS.load(Ordering::Relaxed), 1);
    }
}
