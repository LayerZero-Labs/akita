use super::*;
use crate::commitment::external::CpuBackendKind;
use crate::commitment::{
    AvailablePolynomialTypes, CommitSourceClass, CommitSourceDescriptor, CompressionOperation,
    CompressionState, DenseCoefficientSource, DenseRepresentation, DenseType,
    NoRetainedStatePolicy, OuterCommitOperation, PolynomialRepresentation, PolynomialTypeSelection,
    PortableStatePolicy, StateOwnerCapability,
};
use crate::{AkitaProverSetup, DensePoly};
use akita_params::FoldChallengeDistribution;
use akita_params::{
    CommittedGroupParams, SetupMatrixCapacity, SisModulusProfileId, TerminalFoldParams,
};
use jolt_algebra::{Prime64Offset59, Ring};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type F = Prime64Offset59;

fn plan() -> CommitmentExecutionPlan {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        1,
        1,
        1,
        FoldChallengeDistribution::pm1_only(1),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .unwrap();
    let terminal = TerminalFoldParams::from_expanded_group(params);
    CommitmentExecutionPlan::for_terminal(&terminal, 0).unwrap()
}

fn split_capabilities() -> CommitmentRequestCapabilities {
    CommitmentRequestCapabilities::split::<CpuInnerContext>(
        BackendKindId::of::<CpuBackendKind>("cpu").unwrap(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
    )
}

struct CountingDense {
    poly: DensePoly<F>,
    materializations: AtomicUsize,
}

impl DenseCoefficientSource<F> for CountingDense {
    fn coefficients(&self) -> &[F] {
        self.poly.field_coeffs()
    }
}

impl CommitmentSource<F> for CountingDense {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(9, 512, 512, CommitSourceClass::Dense, "counting-dense")
    }

    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        Ok((0, 1))
    }

    fn available_polynomial_types(
        &self,
        _plan: &crate::opaque::CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        AvailablePolynomialTypes::new(vec![PolynomialType::Dense(DenseType::Coefficients)])
    }

    fn represent_as(
        &self,
        _selected: PolynomialTypeSelection,
        _plan: &crate::opaque::CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        self.materializations.fetch_add(1, Ordering::SeqCst);
        Ok(PolynomialRepresentation::Dense(
            DenseRepresentation::Coefficients(self),
        ))
    }
}

#[test]
fn cpu_executor_prepares_resources_before_source_materialization() {
    let plan = plan();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 3 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .unwrap();
    let source = CountingDense {
        poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
        materializations: AtomicUsize::new(0),
    };
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    let error = match executor.execute_inner(&plan, &sources) {
        Err(error) => error,
        Ok(_) => panic!("undersized setup must fail before materialization"),
    };
    assert!(matches!(
        error,
        AkitaError::InvalidSetup(message) if message.contains("setup field elements")
    ));
    assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
}

#[test]
fn unsupported_inner_dimension_is_rejected_before_materialization() {
    let plan = plan();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 128 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let mut executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .unwrap();
    executor.inner_mut().unwrap().dimensions = StageDimensionCapabilities::new(vec![128]).unwrap();
    let source = CountingDense {
        poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
        materializations: AtomicUsize::new(0),
    };
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    assert!(executor.execute_inner(&plan, &sources).is_err());
    assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
}

#[test]
fn unsupported_compression_mode_is_rejected_before_materialization() {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        1,
        1,
        1,
        FoldChallengeDistribution::pm1_only(1),
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
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let mut executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .unwrap();
    executor.compression.as_mut().unwrap().capabilities = CompressionOperationCapabilities::new(
        StageDimensionCapabilities::new(vec![64, 32]).unwrap(),
        vec![akita_params::RingRelationMode::ReducedEvaluation],
    )
    .unwrap();
    let source = CountingDense {
        poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
        materializations: AtomicUsize::new(0),
    };
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    assert!(executor.execute_full(&plan, &sources).is_err());
    assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
}

#[test]
fn cpu_executor_reports_prewarms_and_releases_shared_resources() {
    let plan = plan();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .unwrap();
    let source = CountingDense {
        poly: DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap(),
        materializations: AtomicUsize::new(0),
    };
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    executor.prewarm_request(&plan, &sources).unwrap();
    assert_eq!(source.materializations.load(Ordering::SeqCst), 0);
    let cached = prepared.shared_ntt_cache_bytes();
    assert!(cached > 0);
    assert_eq!(backend.release_built_ntt_slots(&prepared).unwrap(), cached);
    assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
}

struct RecordingOuter {
    host_calls: Arc<AtomicUsize>,
}

impl OuterCommitOperation<F> for RecordingOuter {
    fn commit_outer(
        &self,
        plan: &super::super::UncompressedCommitPlan,
        inner: super::super::InnerImageInput<'_, F>,
        _randomness: &[i8],
    ) -> Result<akita_types::RingVec<F>, AkitaError> {
        if !matches!(inner, super::super::InnerImageInput::HostRows(_)) {
            return Err(AkitaError::InvalidInput(
                "recording outer expected an explicit host transfer".into(),
            ));
        }
        self.host_calls.fetch_add(1, Ordering::SeqCst);
        akita_types::RingVec::from_coeffs_with_ring_dim(
            vec![F::from_u64(0); plan.outer().output_coefficient_len()?],
            plan.outer().ring_dimension(),
        )
    }
}

struct RecordingCompression {
    calls: Arc<AtomicUsize>,
    owner: StateOwnerCapability<CompressionState>,
}

impl CompressionOperation<F> for RecordingCompression {
    fn compress(
        &self,
        binding: &CommitmentStateBinding,
        plan: &akita_params::CompressionChainPlan,
        relation_mode: akita_params::RingRelationMode,
        _u: akita_types::RingVec<F>,
    ) -> Result<super::super::CompressionStageOutput<F>, AkitaError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let map = plan
            .maps()
            .last()
            .ok_or_else(|| AkitaError::InvalidSetup("empty compression chain".into()))?;
        let terminal = akita_types::RingVec::from_coeffs_with_ring_dim(
            vec![F::from_u64(0); map.output_coefficients()],
            map.ring_dimension(),
        )?;
        let state = self.owner.bind(binding.clone(), 0, ());
        super::super::CompressionStageOutput::new(terminal, state, plan, relation_mode)
    }
}

#[test]
fn builder_routes_independent_outer_and_compression_operations() {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        1,
        1,
        1,
        FoldChallengeDistribution::pm1_only(1),
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
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let mut builder = CommitmentExecutorBuilder::new(setup.expanded.as_ref(), ResidentStatePolicy);
    let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
    let inner_owner = inner.owner().clone();
    let inner_resources = StageResources::controlled(
        PreparedCommitmentResources::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
    );
    let inner_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "cpu-inner",
            inner_resources,
        )
        .unwrap();
    builder
        .register_inner(
            PreparedInnerCommitment::new(
                inner.clone(),
                inner_owner,
                inner_context,
                split_capabilities(),
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                Some(inner.portable_exporter()),
            )
            .unwrap(),
        )
        .unwrap();
    let host_calls = Arc::new(AtomicUsize::new(0));
    let outer_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "recording-outer",
            StageResources::none(),
        )
        .unwrap();
    builder
        .register_outer(PreparedOuterCommitment::new(
            Arc::new(RecordingOuter {
                host_calls: host_calls.clone(),
            }),
            StateOwnerCapability::new(),
            outer_context,
            StageDimensionCapabilities::new(vec![64]).unwrap(),
        ))
        .unwrap();
    let compression_calls = Arc::new(AtomicUsize::new(0));
    let compression_owner = StateOwnerCapability::new();
    let compression_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "recording-compression",
            StageResources::none(),
        )
        .unwrap();
    builder
        .register_compression(PreparedCompression::new(
            Arc::new(RecordingCompression {
                calls: compression_calls.clone(),
                owner: compression_owner.clone(),
            }),
            compression_owner,
            compression_context,
            CompressionOperationCapabilities::new(
                StageDimensionCapabilities::new(vec![64, 32]).unwrap(),
                vec![akita_params::RingRelationMode::QuotientLift],
            )
            .unwrap(),
            None,
        ))
        .unwrap();
    let executor = builder.build().unwrap();
    let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&poly];

    let output = executor.execute_full(&plan, &sources).unwrap();
    assert_eq!(host_calls.load(Ordering::SeqCst), 1);
    assert_eq!(compression_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        output.terminal_payload().coeff_len(),
        plan.compression().unwrap().terminal_coefficients()
    );
}

#[test]
fn builder_rejects_cross_owner_route_without_exporter() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let mut builder = CommitmentExecutorBuilder::new(setup.expanded.as_ref(), ResidentStatePolicy);
    let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
    let inner_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "cpu-inner",
            StageResources::none(),
        )
        .unwrap();
    builder
        .register_inner(
            PreparedInnerCommitment::new(
                inner.clone(),
                inner.owner().clone(),
                inner_context,
                split_capabilities(),
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let outer_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "recording-outer",
            StageResources::none(),
        )
        .unwrap();
    builder
        .register_outer(PreparedOuterCommitment::new(
            Arc::new(RecordingOuter {
                host_calls: Arc::new(AtomicUsize::new(0)),
            }),
            StateOwnerCapability::new(),
            outer_context,
            StageDimensionCapabilities::new(vec![64]).unwrap(),
        ))
        .unwrap();
    let compression_owner = StateOwnerCapability::new();
    let compression_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "recording-compression",
            StageResources::none(),
        )
        .unwrap();
    builder
        .register_compression(PreparedCompression::new(
            Arc::new(RecordingCompression {
                calls: Arc::new(AtomicUsize::new(0)),
                owner: compression_owner.clone(),
            }),
            compression_owner,
            compression_context,
            CompressionOperationCapabilities::new(
                StageDimensionCapabilities::new(vec![64, 32]).unwrap(),
                vec![akita_params::RingRelationMode::QuotientLift],
            )
            .unwrap(),
            None,
        ))
        .unwrap();

    assert!(builder.build().is_err());
}

#[test]
fn cpu_executor_inner_path_binds_setup_plan_and_source() {
    let plan = plan();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .unwrap();
    assert_eq!(
        executor.inner().unwrap().stage.backend_instance,
        executor.outer().unwrap().stage.backend_instance
    );
    assert_eq!(
        executor.outer().unwrap().stage.backend_instance,
        executor
            .compression
            .as_ref()
            .unwrap()
            .stage
            .backend_instance
    );
    assert_ne!(
        executor.inner().unwrap().stage.operation_id,
        executor.outer().unwrap().stage.operation_id
    );
    assert_ne!(
        executor.outer().unwrap().stage.operation_id,
        executor.compression.as_ref().unwrap().stage.operation_id
    );
    assert_eq!(
        executor.inner().unwrap().stage.resources.cache_owner_id(),
        executor.outer().unwrap().stage.resources.cache_owner_id()
    );
    assert_eq!(
        executor.outer().unwrap().stage.resources.cache_owner_id(),
        executor
            .compression
            .as_ref()
            .unwrap()
            .stage
            .resources
            .cache_owner_id()
    );
    let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
    let output = executor.execute_inner_stages(&plan, &sources).unwrap();
    let rows = executor
        .inner()
        .unwrap()
        .exporter
        .as_ref()
        .unwrap()
        .export_inner_rows(plan.inner(), output.image())
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].ring_dim(), 64);
    drop(output);

    let output = executor.execute_inner(&plan, &sources).unwrap();
    assert_eq!(output.binding().relation_mode(), None);
    drop(output);
}

#[test]
fn cpu_executor_rejects_prepared_setup_from_another_descriptor() {
    let setup_a = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * 64,
        },
    )
    .unwrap();
    let setup_b = AkitaProverSetup::<F>::generate_with_capacity(
        10,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * 64,
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared_a = backend.prepare_setup(&setup_a).unwrap();

    assert!(CommitmentExecutor::cpu(
        &backend,
        &prepared_a,
        setup_b.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .is_err());
}

#[test]
fn cpu_executor_split_path_keeps_inner_resident_and_returns_checked_u() {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        1,
        1,
        1,
        FoldChallengeDistribution::pm1_only(1),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .unwrap();
    let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
    let compression_capacity = plan
        .compression()
        .unwrap()
        .maps()
        .iter()
        .map(|map| map.input_width() * map.ring_dimension())
        .max()
        .unwrap();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: compression_capacity.max(4 * 64),
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .unwrap();
    let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
    let output = executor
        .execute_uncompressed_stages(&plan, &sources, None)
        .unwrap();
    let uncompressed = plan.uncompressed().unwrap();

    assert_eq!(output.u().ring_dim(), uncompressed.outer().ring_dimension());
    assert_eq!(
        output.u().coeff_len(),
        uncompressed.outer().output_coefficient_len().unwrap()
    );
    assert_eq!(output.image().binding().inner_plan(), uncompressed.inner());
}

#[test]
fn cpu_executor_full_path_retains_mode_bound_state() {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        1,
        1,
        1,
        FoldChallengeDistribution::pm1_only(1),
    )
    .with_decomp(4, 8, 1, 2, 2)
    .unwrap();
    let plan = CommitmentExecutionPlan::for_root(&params.own_group().profile).unwrap();
    let compression_capacity = plan
        .compression()
        .unwrap()
        .maps()
        .iter()
        .map(|map| map.input_width() * map.ring_dimension())
        .max()
        .unwrap();
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: compression_capacity.max(4 * 64),
        },
    )
    .unwrap();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let prepared = backend.prepare_setup(&setup).unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .unwrap();
    let poly = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&poly];
    let output = executor.execute_full(&plan, &sources).unwrap();

    assert_eq!(
        output.prover_state().binding().relation_mode(),
        plan.relation_mode()
    );
    assert_eq!(
        output.terminal_payload().coeff_len(),
        plan.compression().unwrap().terminal_coefficients()
    );
    drop(output);

    let no_state_executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        NoRetainedStatePolicy,
    )
    .unwrap();
    let output = no_state_executor.execute_full(&plan, &sources).unwrap();
    let expected_payload = output.terminal_payload().clone();
    assert_eq!(output.prover_state(), &());

    let portable_executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        setup.expanded.as_ref(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        PortableStatePolicy,
    )
    .unwrap();
    let inner = portable_executor
        .execute_inner_stages(&plan, &sources)
        .unwrap();
    let expected_rows = portable_executor
        .inner()
        .unwrap()
        .exporter
        .as_ref()
        .unwrap()
        .export_inner_rows(plan.inner(), inner.image())
        .unwrap();
    drop(inner);
    let output = portable_executor.execute_full(&plan, &sources).unwrap();
    let hint = output.prover_state();

    assert_eq!(output.terminal_payload(), &expected_payload);
    assert_eq!(hint.inner_rows().unwrap().as_ref(), expected_rows);
    assert_eq!(
        hint.outer_compression_witness(plan.compression().unwrap())
            .unwrap()
            .plan(),
        plan.compression().unwrap()
    );
    assert_eq!(
        hint.outer_compression_quotients(plan.compression().unwrap())
            .unwrap()
            .len(),
        plan.compression().unwrap().maps().len()
    );
}
