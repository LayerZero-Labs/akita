use super::*;
use crate::opaque::CpuBackend;
use crate::opaque::OpeningBatchKernel;
use akita_challenges::{Challenges, SparseChallenge};
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

impl OpeningBatchKernel<RecordingBatch, F, D> for CpuBackend {
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: RecordingBatch,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
        SOURCE_TRAVERSALS.fetch_add(1, Ordering::Relaxed);
        let DecomposeFoldBatchPlan::SparseChunked { chunk_ranges, .. } = plan else {
            return Err(AkitaError::InvalidInput(
                "expected chunked recording plan".into(),
            ));
        };
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
        let ranges = akita_types::dyadic_block_ranges(blocks, chunks).unwrap();
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
            akita_types::OpeningMethod::EvaluationTrace,
            crate::opaque::ValidatedFoldAcceptancePlan::new(u128::MAX, u128::MAX, None),
        )
        .unwrap();
        let outcome = FoldResponseKernel::probe(
            &CpuBackend::for_arithmetic_tests(),
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
    let ranges = akita_types::dyadic_block_ranges(2, 2).unwrap();
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
        akita_types::OpeningMethod::EvaluationTrace,
        crate::opaque::ValidatedFoldAcceptancePlan::new(10, 10, None),
    )
    .unwrap();
    let outcome = FoldResponseKernel::probe(
        &CpuBackend::for_arithmetic_tests(),
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
    let backend = CpuBackend::for_arithmetic_tests();
    let scope = backend.owner().begin_test_scope(vec![2, 1]).unwrap();
    let context = ProofContext::new(backend.owner_id(), backend.owner().setup_digest(), scope, 0)
        .for_group(0);
    let opening = backend.binding(&context).unwrap();
    let second_opening = backend.binding(&context).unwrap();
    assert!(opening.validate_lineage(&second_opening).is_ok());
    assert!(opening.validate_computation(&second_opening).is_err());
    assert!(opening
        .validate_computation(&opening.with_group(Some(1)))
        .is_err());
    assert!(opening
        .validate_lineage(&opening.for_level_operation(1, opening.operation_id()))
        .is_err());
    let foreign = CpuBackend::for_arithmetic_tests();
    assert!(foreign.validate_binding(&opening).is_err());
    let independent = backend.owner().begin_test_scope(vec![2, 1]).unwrap();
    let independent_context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        independent,
        0,
    )
    .for_group(0);
    let independent_opening = backend.binding(&independent_context).unwrap();
    assert!(opening.validate_lineage(&independent_opening).is_err());
    backend.owner().finish_scope(scope).unwrap();
    assert!(backend.validate_binding(&opening).is_err());
    assert!(backend.validate_binding(&independent_opening).is_ok());
    backend.owner().abort_scope(independent);
    assert!(backend.validate_binding(&independent_opening).is_err());
}

#[test]
fn accepted_fold_rejects_substituted_challenges_and_opening_computation() {
    let _guard = RECORDING_LOCK.lock().unwrap();
    let backend = CpuBackend::for_arithmetic_tests();
    let scope = backend.owner().begin_test_scope(vec![1]).unwrap();
    let context = ProofContext::new(backend.owner_id(), backend.owner().setup_digest(), scope, 0)
        .for_group(0);
    let first_opening = backend.binding(&context).unwrap();
    let other_opening = backend.binding(&context).unwrap();
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
    let ranges = akita_types::dyadic_block_ranges(2, 2).unwrap();
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
        akita_types::OpeningMethod::EvaluationTrace,
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
    let FoldProbeOutcome::Accepted { mut fold_handle } = accepted else {
        panic!("expected accepted fold")
    };
    fold_handle.bind(first_opening);
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
    backend.owner().finish_scope(scope).unwrap();
    assert!(backend.validate_binding(&fold_handle.binding()).is_err());
}
