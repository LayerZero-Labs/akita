use super::*;
use crate::commitment::{
    BackendKindId, BackendStateRef, CommitmentExecutionPlan, CommitmentNttRequirement,
    CommitmentNttRoute, CommitmentRequestCapabilities, CommitmentResourceControl, CommitmentSource,
    CommitmentStateBinding, CompressionOperationCapabilities, CompressionState, DenseType,
    FusedInnerOuterOperation, InnerCommitOperation, InnerCommitOutput, InnerImage,
    InnerImageExportOperation, InnerImageInput, InnerRelationState, NoRetainedStatePolicy,
    OuterCommitOperation, OuterCompressionState, PolynomialType, PortableCommitmentState,
    PortableCompressionState, PortableCompressionStateExport, PortableStatePolicy,
    PreparedCommitmentResources, PreparedCompression, PreparedFusedCommitment,
    PreparedInnerCommitment, ResidentStatePolicy, ResolvedCommitSource, StageDimensionCapabilities,
    StageResources, TerminalBindingState, UncompressedCommitPlan, UncompressedCommitmentOutput,
};
use crate::compute::{
    ComputeBackendSetup, CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation,
    CpuOuterCommitOperation, CpuPreparedSetup, NttCacheOwnerId, NttOperationCluster,
    RoutedNttRequirement,
};
use crate::{AkitaProverSetup, DensePoly};
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{
    CommittedGroupParams, RingVec, SetupMatrixCapacity, SisModulusProfileId, TerminalFoldParams,
};
use jolt_field::Prime64Offset59;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

type F = Prime64Offset59;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FusedEvent {
    SubmitFused,
    DeviceAComplete,
    DeviceBBegin,
    HostInnerRows,
    HostResult,
    InnerStateConsumed,
    CompressionStateConsumed,
}

struct RecordingFused<'a> {
    inner: Arc<CpuInnerCommitOperation<'a, F>>,
    outer: CpuOuterCommitOperation<'a, F>,
    events: Arc<Mutex<Vec<FusedEvent>>>,
    attempts_per_call: Arc<AtomicUsize>,
}

struct RecordingInner<'a> {
    operation: Arc<CpuInnerCommitOperation<'a, F>>,
    calls: Arc<AtomicUsize>,
}

struct CountingResources<'a> {
    inner: PreparedCommitmentResources<'a, F, CpuBackend>,
    ensures: Arc<AtomicUsize>,
    routes: Arc<Mutex<Vec<CommitmentNttRoute>>>,
    folds: Arc<Mutex<Vec<usize>>>,
}

impl CommitmentResourceControl<F> for CountingResources<'_> {
    fn setup_descriptor(&self) -> &akita_types::AkitaSetupDescriptor {
        self.inner.setup_descriptor()
    }

    fn ensure_ntt_slot(&self, requirement: CommitmentNttRequirement) -> Result<(), AkitaError> {
        self.ensures.fetch_add(1, Ordering::SeqCst);
        self.routes.lock().unwrap().push(requirement.route());
        self.folds.lock().unwrap().push(requirement.fold_level());
        self.inner.ensure_ntt_slot(requirement)
    }

    fn requirement_is_cached(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<bool, AkitaError> {
        self.routes.lock().unwrap().push(requirement.route());
        self.folds.lock().unwrap().push(requirement.fold_level());
        self.inner.requirement_is_cached(requirement)
    }

    fn cache_owner_id(&self) -> NttCacheOwnerId {
        self.inner.cache_owner_id()
    }

    fn release_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        self.inner.release_built_ntt_slots()
    }

    fn compression_cache_bytes(&self) -> Option<usize> {
        self.inner.compression_cache_bytes()
    }
}

fn counting_resources<'a>(
    backend: &'a CpuBackend,
    prepared: &'a CpuPreparedSetup<F>,
    expanded: &'a akita_types::AkitaExpandedSetup<F>,
    ensures: Arc<AtomicUsize>,
    routes: Arc<Mutex<Vec<CommitmentNttRoute>>>,
    folds: Arc<Mutex<Vec<usize>>>,
) -> StageResources<'a, F> {
    StageResources::controlled(CountingResources {
        inner: PreparedCommitmentResources::new(backend, prepared, expanded).unwrap(),
        ensures,
        routes,
        folds,
    })
}

impl InnerCommitOperation<F> for RecordingInner<'_> {
    fn commit_inner(
        &self,
        state: &CommitmentStateBinding,
        plan: &crate::compute::CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.operation.commit_inner(state, plan, sources)
    }
}

struct RecordingInnerConsumer {
    inner: Arc<dyn InnerImageExportOperation<F>>,
    events: Arc<Mutex<Vec<FusedEvent>>>,
}

impl InnerImageExportOperation<F> for RecordingInnerConsumer {
    fn export_inner_rows(
        &self,
        plan: &crate::compute::CommitInnerPlan,
        image: &BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        self.events
            .lock()
            .map_err(|_| AkitaError::InvalidInput("fused event recorder is poisoned".into()))?
            .push(FusedEvent::InnerStateConsumed);
        self.inner.export_inner_rows(plan, image)
    }
}

struct RecordingCompressionConsumer {
    inner: Arc<dyn PortableCompressionStateExport<F>>,
    events: Arc<Mutex<Vec<FusedEvent>>>,
}

impl PortableCompressionStateExport<F> for RecordingCompressionConsumer {
    fn export_compression_state(
        &self,
        state: &BackendStateRef<CompressionState>,
    ) -> Result<PortableCompressionState<F>, AkitaError> {
        self.events
            .lock()
            .map_err(|_| AkitaError::InvalidInput("fused event recorder is poisoned".into()))?
            .push(FusedEvent::CompressionStateConsumed);
        self.inner.export_compression_state(state)
    }
}

impl RecordingFused<'_> {
    fn record(&self, event: FusedEvent) -> Result<(), AkitaError> {
        self.events
            .lock()
            .map_err(|_| AkitaError::InvalidInput("fused event recorder is poisoned".into()))?
            .push(event);
        Ok(())
    }
}

impl FusedInnerOuterOperation<F> for RecordingFused<'_> {
    fn commit_inner_outer(
        &self,
        state: &CommitmentStateBinding,
        plan: &UncompressedCommitPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<UncompressedCommitmentOutput<F>, AkitaError> {
        for _ in 0..self.attempts_per_call.load(Ordering::SeqCst) {
            self.record(FusedEvent::SubmitFused)?;
        }
        let inner = self.inner.commit_inner(state, plan.inner(), sources)?;
        self.record(FusedEvent::DeviceAComplete)?;
        self.record(FusedEvent::DeviceBBegin)?;
        let u = self
            .outer
            .commit_outer(plan, InnerImageInput::Owned(inner.image()))?;
        let output = UncompressedCommitmentOutput::new(inner.into_image(), u, plan)?;
        self.record(FusedEvent::HostResult)?;
        Ok(output)
    }
}

#[test]
fn explicitly_selected_fused_route_has_one_submission_and_cpu_parity() {
    struct SplitContext;
    struct FusedContext;
    struct FusedCommand;
    struct SplitBackend;
    struct FusedBackend;

    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .unwrap();
    let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 128 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::DEFAULT;
    let prepared = backend.prepare_setup(&setup).unwrap();
    let standard_types = vec![PolynomialType::Dense(DenseType::Coefficients)];
    let split = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        standard_types.clone(),
        PortableStatePolicy,
    )
    .unwrap();

    let mut builder = CommitmentExecutorBuilder::new(setup.expanded.as_ref(), ResidentStatePolicy);
    let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
    let split_inner_calls = Arc::new(AtomicUsize::new(0));
    let split_inner = Arc::new(RecordingInner {
        operation: inner.clone(),
        calls: split_inner_calls.clone(),
    });
    let compression = Arc::new(
        CpuCompressionOperation::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
    );
    let events = Arc::new(Mutex::new(Vec::new()));
    let attempts_per_call = Arc::new(AtomicUsize::new(1));
    let fused = Arc::new(RecordingFused {
        inner: inner.clone(),
        outer: CpuOuterCommitOperation::new(&backend, &prepared, inner.as_ref()),
        events: events.clone(),
        attempts_per_call: attempts_per_call.clone(),
    });
    let fused_inner_consumer = Arc::new(RecordingInnerConsumer {
        inner: inner.portable_exporter(),
        events: events.clone(),
    });
    let fused_compression_consumer = Arc::new(RecordingCompressionConsumer {
        inner: compression.portable_exporter(),
        events: events.clone(),
    });
    let split_inner_ensures = Arc::new(AtomicUsize::new(0));
    let compression_ensures = Arc::new(AtomicUsize::new(0));
    let fused_ensures = Arc::new(AtomicUsize::new(0));
    let split_inner_routes = Arc::new(Mutex::new(Vec::new()));
    let split_inner_folds = Arc::new(Mutex::new(Vec::new()));
    let split_instance = builder.issue_backend_instance();
    let fused_instance = builder.issue_backend_instance();
    let inner_context = builder
        .operation_context(
            split_instance,
            "split-inner",
            counting_resources(
                &backend,
                &prepared,
                setup.expanded.as_ref(),
                split_inner_ensures.clone(),
                split_inner_routes.clone(),
                split_inner_folds.clone(),
            ),
        )
        .unwrap();
    let compression_context = builder
        .operation_context(
            split_instance,
            "cpu-compression",
            counting_resources(
                &backend,
                &prepared,
                setup.expanded.as_ref(),
                compression_ensures.clone(),
                Arc::new(Mutex::new(Vec::new())),
                Arc::new(Mutex::new(Vec::new())),
            ),
        )
        .unwrap();
    let fused_context = builder
        .operation_context(
            fused_instance,
            "recording-fused",
            counting_resources(
                &backend,
                &prepared,
                setup.expanded.as_ref(),
                fused_ensures.clone(),
                Arc::new(Mutex::new(Vec::new())),
                Arc::new(Mutex::new(Vec::new())),
            ),
        )
        .unwrap();
    builder
        .register_inner(
            PreparedInnerCommitment::new(
                split_inner,
                inner.owner().clone(),
                inner_context,
                CommitmentRequestCapabilities::split::<SplitContext>(
                    BackendKindId::of::<SplitBackend>("split-cpu").unwrap(),
                    standard_types.clone(),
                ),
                StageDimensionCapabilities::cpu_role::<F>(akita_types::RingRole::Inner),
                Some(inner.portable_exporter()),
            )
            .unwrap(),
        )
        .unwrap();
    builder
        .register_compression(PreparedCompression::new(
            compression.clone(),
            compression.owner().clone(),
            compression_context,
            CompressionOperationCapabilities::cpu::<F>(),
            Some(fused_compression_consumer),
        ))
        .unwrap();
    builder
        .register_fused(
            PreparedFusedCommitment::new(
                fused,
                inner.owner().clone(),
                fused_context,
                CommitmentRequestCapabilities::fused::<FusedContext, FusedCommand>(
                    BackendKindId::of::<FusedBackend>("recording-fused").unwrap(),
                    standard_types,
                ),
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                Some(fused_inner_consumer),
            )
            .unwrap(),
        )
        .unwrap();
    let fused_executor = builder.build().unwrap();
    let source = DensePoly::from_field_evals(9, vec![F::default(); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    let inner_requirement = plan
        .inner_ntt_requirement(PolynomialType::Dense(DenseType::Coefficients))
        .unwrap()
        .unwrap();
    let routed = |fold_level, route| RoutedNttRequirement {
        fold_level,
        cluster: NttOperationCluster::Commit,
        commitment_stage: Some(inner_requirement.stage()),
        commitment_route: Some(route),
        key: inner_requirement.key(),
        routing_extent: inner_requirement.routing_extent(),
    };
    fused_executor
        .prewarm_routed_requirement(routed(0, CommitmentNttRoute::InnerOuter))
        .unwrap();
    assert_eq!(fused_ensures.load(Ordering::SeqCst), 1);
    assert_eq!(split_inner_ensures.load(Ordering::SeqCst), 0);
    fused_executor
        .prewarm_routed_requirement(routed(7, CommitmentNttRoute::InnerOnly))
        .unwrap();
    assert_eq!(split_inner_ensures.load(Ordering::SeqCst), 1);
    assert!(split_inner_routes
        .lock()
        .unwrap()
        .iter()
        .all(|route| *route == CommitmentNttRoute::InnerOnly));
    assert_eq!(split_inner_folds.lock().unwrap().as_slice(), [7]);
    split_inner_routes.lock().unwrap().clear();
    assert!(fused_executor
        .retained_routed_requirement(routed(7, CommitmentNttRoute::InnerOnly))
        .unwrap()
        .is_some());
    assert_eq!(
        split_inner_routes.lock().unwrap().as_slice(),
        [CommitmentNttRoute::InnerOnly]
    );
    assert_eq!(split_inner_folds.lock().unwrap().as_slice(), [7, 7]);
    fused_ensures.store(0, Ordering::SeqCst);
    split_inner_ensures.store(0, Ordering::SeqCst);

    fused_executor.prewarm_request(&plan, &sources).unwrap();
    assert_eq!(fused_ensures.load(Ordering::SeqCst), 2);
    assert_eq!(split_inner_ensures.load(Ordering::SeqCst), 0);
    assert_eq!(compression_ensures.load(Ordering::SeqCst), 0);

    let expected_u = split.execute_uncompressed_stages(&plan, &sources).unwrap();
    let actual_u = fused_executor
        .execute_uncompressed_stages(&plan, &sources)
        .unwrap();
    assert_eq!(actual_u.u(), expected_u.u());
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [
            FusedEvent::SubmitFused,
            FusedEvent::DeviceAComplete,
            FusedEvent::DeviceBBegin,
            FusedEvent::HostResult,
        ]
    );
    assert!(!events.lock().unwrap().contains(&FusedEvent::HostInnerRows));

    events.lock().unwrap().clear();
    let expected = split.execute_full(&plan, &sources).unwrap();
    let actual = fused_executor.execute_full(&plan, &sources).unwrap();
    assert_eq!(actual.terminal_payload(), expected.terminal_payload());
    let expected_inner = expected.prover_state().inner_relation_material().unwrap();
    let retained_before_freeze = actual.prover_state().retained_bytes().unwrap();
    actual.prover_state().terminal_t_fields_message().unwrap();
    assert_eq!(
        actual.prover_state().retained_bytes().unwrap(),
        retained_before_freeze
    );
    let actual_inner = actual.prover_state().inner_relation_material().unwrap();
    assert_eq!(
        actual_inner.ring_dimension(),
        expected_inner.ring_dimension()
    );
    assert_eq!(actual_inner.rows(), expected_inner.rows());
    let portable = actual.prover_state().portable_hint().unwrap();
    assert_eq!(portable.inner_rows(), actual_inner.rows());
    let compression_plan = plan.compression().unwrap();
    let expected_compression = expected
        .prover_state()
        .outer_compression_material(compression_plan, params.ring_relation_mode)
        .unwrap();
    let actual_compression = actual
        .prover_state()
        .outer_compression_material(compression_plan, params.ring_relation_mode)
        .unwrap();
    match (expected_compression, actual_compression) {
        (
            PortableCompressionState::QuotientLift {
                witness: expected_witness,
                quotients: expected_quotients,
            },
            PortableCompressionState::QuotientLift {
                witness: actual_witness,
                quotients: actual_quotients,
            },
        ) => {
            assert_eq!(actual_witness, expected_witness);
            assert_eq!(actual_quotients, expected_quotients);
        }
        _ => panic!("root fixture must retain quotient-lift compression state"),
    }
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [
            FusedEvent::SubmitFused,
            FusedEvent::DeviceAComplete,
            FusedEvent::DeviceBBegin,
            FusedEvent::HostResult,
            FusedEvent::InnerStateConsumed,
            FusedEvent::CompressionStateConsumed,
            FusedEvent::CompressionStateConsumed,
        ]
    );

    events.lock().unwrap().clear();
    let terminal_plan =
        CommitmentExecutionPlan::for_terminal(&TerminalFoldParams::from_expanded_group(params))
            .unwrap();
    split_inner_ensures.store(0, Ordering::SeqCst);
    fused_ensures.store(0, Ordering::SeqCst);
    split_inner_routes.lock().unwrap().clear();
    fused_executor
        .prewarm_request(&terminal_plan, &sources)
        .unwrap();
    assert_eq!(split_inner_ensures.load(Ordering::SeqCst), 1);
    assert_eq!(fused_ensures.load(Ordering::SeqCst), 0);
    assert_eq!(
        split_inner_routes.lock().unwrap().as_slice(),
        [CommitmentNttRoute::InnerOnly, CommitmentNttRoute::InnerOnly]
    );
    let expected_inner_state = split.execute_inner(&terminal_plan, &sources).unwrap();
    let actual_inner_state = fused_executor
        .execute_inner(&terminal_plan, &sources)
        .unwrap();
    let expected_terminal_inner = expected_inner_state.inner_relation_material().unwrap();
    let actual_terminal_inner = actual_inner_state.inner_relation_material().unwrap();
    assert_eq!(
        actual_terminal_inner.ring_dimension(),
        expected_terminal_inner.ring_dimension()
    );
    assert_eq!(actual_terminal_inner.rows(), expected_terminal_inner.rows());
    assert_eq!(split_inner_calls.load(Ordering::SeqCst), 1);
    assert!(events.lock().unwrap().is_empty());

    attempts_per_call.store(2, Ordering::SeqCst);
    fused_executor
        .execute_uncompressed_stages(&plan, &sources)
        .unwrap();
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| **event == FusedEvent::SubmitFused)
            .count(),
        2
    );
}

#[test]
fn fused_only_executor_needs_no_split_registration_or_inner_exporter() {
    struct FusedContext;
    struct FusedCommand;
    struct FusedBackend;

    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(1),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .unwrap();
    let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 128 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::DEFAULT;
    let prepared = backend.prepare_setup(&setup).unwrap();
    let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
    let events = Arc::new(Mutex::new(Vec::new()));
    let fused = Arc::new(RecordingFused {
        inner: inner.clone(),
        outer: CpuOuterCommitOperation::new(&backend, &prepared, inner.as_ref()),
        events: events.clone(),
        attempts_per_call: Arc::new(AtomicUsize::new(1)),
    });
    let compression = Arc::new(
        CpuCompressionOperation::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
    );
    let mut builder =
        CommitmentExecutorBuilder::new(setup.expanded.as_ref(), NoRetainedStatePolicy);
    let instance = builder.issue_backend_instance();
    let resources = StageResources::controlled(
        PreparedCommitmentResources::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
    );
    let fused_context = builder
        .operation_context(instance, "fused-only", resources.clone())
        .unwrap();
    let compression_context = builder
        .operation_context(instance, "cpu-compression", resources)
        .unwrap();
    builder
        .register_fused(
            PreparedFusedCommitment::new(
                fused,
                inner.owner().clone(),
                fused_context,
                CommitmentRequestCapabilities::fused::<FusedContext, FusedCommand>(
                    BackendKindId::of::<FusedBackend>("fused-only").unwrap(),
                    vec![PolynomialType::Dense(DenseType::Coefficients)],
                ),
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    builder
        .register_compression(PreparedCompression::new(
            compression.clone(),
            compression.owner().clone(),
            compression_context,
            CompressionOperationCapabilities::cpu::<F>(),
            None,
        ))
        .unwrap();
    let executor = builder.build().unwrap();
    let source = DensePoly::from_field_evals(9, vec![F::default(); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    executor.execute_full(&plan, &sources).unwrap();
    assert_eq!(
        events.lock().unwrap().as_slice(),
        [
            FusedEvent::SubmitFused,
            FusedEvent::DeviceAComplete,
            FusedEvent::DeviceBBegin,
            FusedEvent::HostResult,
        ]
    );
}
