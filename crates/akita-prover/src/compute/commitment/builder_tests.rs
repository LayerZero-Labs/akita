use super::*;
use crate::compute::{
    CommitmentExecutionPlan, CommitmentNttRequirement, CommitmentResourceControl, CommitmentSource,
    CommitmentStateBinding, CompressionStageOutput, ComputeBackendSetup, CpuBackend,
    CpuCompressionOperation, CpuInnerCommitOperation, CpuOuterCommitOperation, InnerCommitOutput,
    InnerImageInput, NttCacheOwnerId, PreparedCommitmentResources, ResolvedCommitSource,
    StageResources,
};
use crate::{AkitaProverSetup, DensePoly};
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{
    CommittedGroupParams, CompressionChainPlan, RingRelationMode, RingVec, SetupMatrixCapacity,
    SisModulusProfileId,
};
use jolt_field::Prime64Offset59;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type F = Prime64Offset59;

struct UnusedInner;

impl InnerCommitOperation<F> for UnusedInner {
    fn commit_inner(
        &self,
        _state: &CommitmentStateBinding,
        _plan: &crate::compute::CommitInnerPlan,
        _sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        Err(AkitaError::InvalidInput("unused test operation".into()))
    }
}

struct UnusedOuter;

impl OuterCommitOperation<F> for UnusedOuter {
    fn commit_outer(
        &self,
        _plan: &crate::compute::UncompressedCommitPlan,
        _inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError> {
        Err(AkitaError::InvalidInput("unused test operation".into()))
    }
}

struct UnusedCompression;

impl CompressionOperation<F> for UnusedCompression {
    fn compress(
        &self,
        _state: &CommitmentStateBinding,
        _plan: &CompressionChainPlan,
        _relation_mode: RingRelationMode,
        _u: RingVec<F>,
    ) -> Result<CompressionStageOutput<F>, AkitaError> {
        Err(AkitaError::InvalidInput("unused test operation".into()))
    }
}

struct RecordingCpuOuter<'a> {
    operation: CpuOuterCommitOperation<'a, F>,
    calls: Arc<AtomicUsize>,
}

impl OuterCommitOperation<F> for RecordingCpuOuter<'_> {
    fn commit_outer(
        &self,
        plan: &crate::compute::UncompressedCommitPlan,
        inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.operation.commit_outer(plan, inner)
    }
}

struct RecordingResources {
    setup: akita_types::AkitaSetupDescriptor,
    identity: Arc<AtomicUsize>,
    ensures: Arc<AtomicUsize>,
    releases: Arc<AtomicUsize>,
    planned_bytes: usize,
    released_bytes: usize,
}

impl CommitmentResourceControl<F> for RecordingResources {
    fn setup_descriptor(&self) -> &akita_types::AkitaSetupDescriptor {
        &self.setup
    }

    fn ensure_ntt_slot(&self, _requirement: CommitmentNttRequirement) -> Result<(), AkitaError> {
        self.ensures.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn requirement_is_cached(
        &self,
        _requirement: CommitmentNttRequirement,
    ) -> Result<bool, AkitaError> {
        Ok(true)
    }

    fn cache_owner_id(&self) -> NttCacheOwnerId {
        NttCacheOwnerId::from_owner(self.identity.as_ref())
    }

    fn planned_ntt_cache_entry_bytes(
        &self,
        _requirement: CommitmentNttRequirement,
    ) -> Result<usize, AkitaError> {
        Ok(self.planned_bytes)
    }

    fn release_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        self.releases.fetch_add(1, Ordering::SeqCst);
        Ok(self.released_bytes)
    }
}

#[test]
fn distinct_resource_owners_are_planned_prewarmed_and_released_independently() {
    struct TestBackend;
    struct TestContext;

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
            num_field_elements: 4 * 64,
        },
    )
    .unwrap();
    let mut builder = CommitmentExecutorBuilder::new::<TestContext>(
        setup.expanded.as_ref(),
        BackendKindId::of::<TestBackend>("test").unwrap(),
        vec![PolynomialType::Dense(super::super::DenseType::Coefficients)],
        super::super::NoRetainedStatePolicy,
    );
    let image_owner = StateOwnerCapability::<InnerImage>::new();
    let inner_ensures = Arc::new(AtomicUsize::new(0));
    let outer_ensures = Arc::new(AtomicUsize::new(0));
    let compression_ensures = Arc::new(AtomicUsize::new(0));
    let inner_releases = Arc::new(AtomicUsize::new(0));
    let outer_releases = Arc::new(AtomicUsize::new(0));
    let compression_releases = Arc::new(AtomicUsize::new(0));
    let resources = |ensures: Arc<AtomicUsize>,
                     releases: Arc<AtomicUsize>,
                     planned_bytes,
                     released_bytes| RecordingResources {
        setup: setup.expanded.descriptor().clone(),
        identity: Arc::new(AtomicUsize::new(0)),
        ensures,
        releases,
        planned_bytes,
        released_bytes,
    };
    let inner_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "inner",
            StageResources::controlled(resources(
                inner_ensures.clone(),
                inner_releases.clone(),
                101,
                11,
            )),
        )
        .unwrap();
    let outer_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "outer",
            StageResources::controlled(resources(
                outer_ensures.clone(),
                outer_releases.clone(),
                202,
                13,
            )),
        )
        .unwrap();
    let compression_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "compression",
            StageResources::controlled(resources(
                compression_ensures.clone(),
                compression_releases.clone(),
                303,
                17,
            )),
        )
        .unwrap();
    builder
        .register_inner(
            Arc::new(UnusedInner),
            image_owner.clone(),
            inner_context,
            StageDimensionCapabilities::new(vec![64]).unwrap(),
            None,
        )
        .unwrap();
    builder
        .register_outer(
            Arc::new(UnusedOuter),
            image_owner,
            outer_context,
            StageDimensionCapabilities::new(vec![64]).unwrap(),
        )
        .unwrap();
    let compression_dimensions = plan
        .compression()
        .unwrap()
        .maps()
        .iter()
        .map(|map| map.ring_dimension())
        .collect();
    builder
        .register_compression(
            Arc::new(UnusedCompression),
            compression_context,
            CompressionOperationCapabilities::new(
                StageDimensionCapabilities::new(compression_dimensions).unwrap(),
                vec![RingRelationMode::QuotientLift],
            )
            .unwrap(),
            None,
        )
        .unwrap();
    let executor = builder.build().unwrap();
    let source = DensePoly::from_field_evals(9, vec![F::default(); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    let mut bytes: Vec<_> = executor
        .planned_request_ntt_cache_metrics(&plan, &sources)
        .unwrap()
        .into_iter()
        .map(|metric| metric.cache_bytes)
        .collect();
    bytes.sort_unstable();
    assert_eq!(bytes, vec![101, 202]);

    executor.prewarm_request(&plan, &sources).unwrap();
    assert_eq!(inner_ensures.load(Ordering::SeqCst), 1);
    assert_eq!(outer_ensures.load(Ordering::SeqCst), 1);
    assert_eq!(compression_ensures.load(Ordering::SeqCst), 0);

    assert_eq!(executor.release_built_ntt_slots().unwrap(), 41);
    assert_eq!(inner_releases.load(Ordering::SeqCst), 1);
    assert_eq!(outer_releases.load(Ordering::SeqCst), 1);
    assert_eq!(compression_releases.load(Ordering::SeqCst), 1);
}

#[test]
fn mixed_outer_with_cpu_compression_matches_the_all_cpu_route() {
    struct MixedBackend;
    struct MixedContext;

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
    let standard_types = vec![PolynomialType::Dense(super::super::DenseType::Coefficients)];
    let all_cpu = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        standard_types.clone(),
        super::super::NoRetainedStatePolicy,
    )
    .unwrap();

    let mut builder = CommitmentExecutorBuilder::new::<MixedContext>(
        setup.expanded.as_ref(),
        BackendKindId::of::<MixedBackend>("mixed").unwrap(),
        standard_types,
        super::super::NoRetainedStatePolicy,
    );
    let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
    let outer = RecordingCpuOuter {
        operation: CpuOuterCommitOperation::new(&backend, &prepared, inner.as_ref()),
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let outer_calls = outer.calls.clone();
    let compression = Arc::new(
        CpuCompressionOperation::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
    );
    let resources = StageResources::controlled(
        PreparedCommitmentResources::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
    );
    let cpu_instance = builder.issue_backend_instance();
    let mixed_outer_instance = builder.issue_backend_instance();
    let inner_context = builder
        .operation_context(cpu_instance, "cpu-inner", resources.clone())
        .unwrap();
    let outer_context = builder
        .operation_context(
            mixed_outer_instance,
            "recording-cpu-outer",
            resources.clone(),
        )
        .unwrap();
    let compression_context = builder
        .operation_context(cpu_instance, "cpu-compression", resources)
        .unwrap();
    builder
        .register_inner(
            inner.clone(),
            inner.owner().clone(),
            inner_context,
            StageDimensionCapabilities::cpu_role::<F>(akita_types::RingRole::Inner),
            Some(inner.portable_exporter()),
        )
        .unwrap();
    builder
        .register_outer(
            Arc::new(outer),
            StateOwnerCapability::new(),
            outer_context,
            StageDimensionCapabilities::cpu_role::<F>(akita_types::RingRole::Outer),
        )
        .unwrap();
    builder
        .register_compression(
            compression.clone(),
            compression_context,
            CompressionOperationCapabilities::cpu::<F>(),
            Some(compression.portable_exporter()),
        )
        .unwrap();
    let mixed = builder.build().unwrap();
    let source = DensePoly::from_field_evals(9, vec![F::default(); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    let expected = all_cpu.execute_full(&plan, &sources).unwrap();
    let actual = mixed.execute_full(&plan, &sources).unwrap();
    assert_eq!(actual.terminal_payload(), expected.terminal_payload());
    assert_eq!(outer_calls.load(Ordering::SeqCst), 1);
}
