use super::*;
use crate::commitment::{
    AvailablePolynomialTypes, BackendKindId, CommitSourceClass, CommitSourceDescriptor,
    CommitmentExecutionPlan, CommitmentRequestCapabilities, CommitmentSource,
    CommitmentStateBinding, CompressionOperationCapabilities, ExternalFusedInnerCommitmentEncoder,
    ExternalInnerCommitmentCapability, ExternalInnerCommitmentInput,
    ExternalInnerCommitmentOperation, ExternalOperationIdentity, FusedInnerOuterOperation,
    InnerImage, NoRetainedStatePolicy, PolynomialRepresentation, PolynomialType,
    PolynomialTypeSelection, PreparedCompression, PreparedExternalInnerCommitment,
    PreparedFusedCommitment, PreparedInnerCommitment, ResolvedCommitSource,
    StageDimensionCapabilities, StageResources, StateOwnerCapability, UncompressedCommitPlan,
    UncompressedCommitmentOutput,
};
use crate::compute::{
    ComputeBackendSetup, CpuBackend, CpuCompressionOperation, CpuInnerCommitOperation,
};
use crate::{AkitaProverSetup, CommitInnerWitness};
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{CommittedGroupParams, RingVec, SetupMatrixCapacity, SisModulusProfileId};
use jolt_field::Prime64Offset59;
use std::any::{Any, TypeId};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type F = Prime64Offset59;

struct ExternalBackend;
struct ExternalFamily {
    value: u8,
}
struct ExternalAlgorithm;
struct ExternalContext;

#[derive(Default)]
struct FusedCommand {
    values: Vec<u8>,
}

struct ExternalOperation;

impl ExternalInnerCommitmentOperation<F> for ExternalOperation {
    fn identity(&self) -> ExternalOperationIdentity {
        ExternalOperationIdentity::of::<ExternalFamily, ExternalAlgorithm, ExternalContext>()
    }

    fn commit_group(
        &self,
        _plan: &crate::compute::CommitInnerPlan,
        _sources: &[ExternalInnerCommitmentInput<'_>],
        _context: &dyn Any,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "ordinary external operation must not run in a fused request".into(),
        ))
    }
}

struct ExternalEncoder;

impl ExternalFusedInnerCommitmentEncoder for ExternalEncoder {
    fn command_context_type_id(&self) -> TypeId {
        TypeId::of::<FusedCommand>()
    }

    fn encode(
        &self,
        _plan: &crate::compute::CommitInnerPlan,
        source: &ExternalInnerCommitmentInput<'_>,
        command: &mut dyn Any,
    ) -> Result<(), AkitaError> {
        let command = command.downcast_mut::<FusedCommand>().ok_or_else(|| {
            AkitaError::InvalidInput("external encoder received the wrong fused command".into())
        })?;
        command
            .values
            .push(source.payload::<ExternalFamily>()?.value);
        Ok(())
    }
}

struct ExternalOnlySource {
    payload: ExternalFamily,
    operation: ExternalOperation,
    encoder: ExternalEncoder,
}

impl CommitmentSource<F> for ExternalOnlySource {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(9, 512, 512, CommitSourceClass::Dense, "external-fused-test")
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
        _plan: &crate::compute::CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        AvailablePolynomialTypes::new(Vec::new())
    }

    fn represent_as(
        &self,
        _selected: PolynomialTypeSelection,
        _plan: &crate::compute::CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external-only source has no standard representation".into(),
        ))
    }

    fn external_inner_commitment_capability(
        &self,
        backend: BackendKindId,
        _plan: &crate::compute::CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        let expected = BackendKindId::of::<ExternalBackend>("external-fused")?;
        Ok((backend == expected).then(|| {
            ExternalInnerCommitmentCapability::new_fused::<
                ExternalFamily,
                ExternalAlgorithm,
                ExternalContext,
                FusedCommand,
            >(expected, "external-fused")
            .unwrap()
        }))
    }

    fn prepare_external_inner_commitment(
        &self,
        selected: ExternalInnerCommitmentCapability,
        _plan: &crate::compute::CommitInnerPlan,
    ) -> Result<PreparedExternalInnerCommitment<'_, F>, AkitaError> {
        PreparedExternalInnerCommitment::new(
            selected,
            &self.payload,
            &self.operation,
            Some(&self.encoder),
        )
    }
}

struct EncodingFused {
    owner: StateOwnerCapability<InnerImage>,
    submissions: Arc<AtomicUsize>,
    encoded_values: Arc<AtomicUsize>,
}

impl FusedInnerOuterOperation<F> for EncodingFused {
    fn commit_inner_outer(
        &self,
        binding: &CommitmentStateBinding,
        plan: &UncompressedCommitPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<UncompressedCommitmentOutput<F>, AkitaError> {
        self.submissions.fetch_add(1, Ordering::SeqCst);
        let mut command = FusedCommand::default();
        for source in sources {
            let external = source.external().ok_or_else(|| {
                AkitaError::InvalidInput("fused external test received a standard source".into())
            })?;
            external
                .fused_encoder()
                .ok_or_else(|| {
                    AkitaError::InvalidInput("fused source has no command encoder".into())
                })?
                .encode(plan.inner(), external.input(), &mut command)?;
        }
        self.encoded_values
            .fetch_add(command.values.len(), Ordering::SeqCst);
        let image = self.owner.bind(binding.clone(), 0, ());
        let u = RingVec::from_coeffs_with_ring_dim(
            vec![F::default(); plan.outer().output_coefficient_len()?],
            plan.outer().ring_dimension(),
        )?;
        UncompressedCommitmentOutput::new(image, u, plan)
    }
}

#[test]
fn fused_external_encoder_appends_without_ordinary_submission() {
    struct SplitBackend;
    struct SplitContext;

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
    let mut builder =
        CommitmentExecutorBuilder::new(setup.expanded.as_ref(), NoRetainedStatePolicy);
    let inner = Arc::new(CpuInnerCommitOperation::new(&backend, &prepared));
    let compression = Arc::new(
        CpuCompressionOperation::new(&backend, &prepared, setup.expanded.as_ref()).unwrap(),
    );
    let no_resources = |name| {
        builder
            .operation_context(
                builder.issue_backend_instance(),
                name,
                StageResources::none(),
            )
            .unwrap()
    };
    let inner_context = no_resources("split-inner");
    let compression_context = no_resources("compression");
    let fused_context = no_resources("fused");
    builder
        .register_inner(
            PreparedInnerCommitment::new(
                inner.clone(),
                inner.owner().clone(),
                inner_context,
                CommitmentRequestCapabilities::split::<SplitContext>(
                    BackendKindId::of::<SplitBackend>("split").unwrap(),
                    vec![PolynomialType::Dense(super::super::DenseType::Coefficients)],
                ),
                StageDimensionCapabilities::new(vec![64]).unwrap(),
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
            Some(compression.portable_exporter()),
        ))
        .unwrap();
    let fused_owner = StateOwnerCapability::new();
    let submissions = Arc::new(AtomicUsize::new(0));
    let encoded_values = Arc::new(AtomicUsize::new(0));
    builder
        .register_fused(
            PreparedFusedCommitment::new(
                Arc::new(EncodingFused {
                    owner: fused_owner.clone(),
                    submissions: submissions.clone(),
                    encoded_values: encoded_values.clone(),
                }),
                fused_owner,
                fused_context,
                CommitmentRequestCapabilities::fused::<ExternalContext, FusedCommand>(
                    BackendKindId::of::<ExternalBackend>("external-fused").unwrap(),
                    Vec::new(),
                ),
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let executor = builder.build().unwrap();
    let source = ExternalOnlySource {
        payload: ExternalFamily { value: 7 },
        operation: ExternalOperation,
        encoder: ExternalEncoder,
    };
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    let output = executor
        .execute_uncompressed_stages(&plan, &sources)
        .unwrap();
    assert_eq!(
        output.u().ring_dim(),
        plan.uncompressed().unwrap().outer().ring_dimension()
    );
    assert_eq!(submissions.load(Ordering::SeqCst), 1);
    assert_eq!(encoded_values.load(Ordering::SeqCst), 1);
}
