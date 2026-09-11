use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_prover::{
    AkitaProverSetup, BackendKindId, CommitmentExecutionPlan, CommitmentExecutorBuilder,
    CommitmentRequestCapabilities, CommitmentSource, CommitmentStateBinding, CompressionOperation,
    CompressionOperationCapabilities, CompressionStageOutput, CompressionState, DensePoly,
    DenseType, FusedInnerOuterOperation, InnerImage, NoRetainedStatePolicy, PolynomialType,
    PreparedCompression, PreparedFusedCommitment, ResolvedCommitSource, StageDimensionCapabilities,
    StageResources, StateOwnerCapability, TerminalTFieldsMessage, UncompressedCommitPlan,
    UncompressedCommitmentOutput,
};
use akita_types::{
    CommittedGroupParams, RingRelationMode, RingVec, SetupMatrixCapacity, SisModulusProfileId,
};
use jolt_field::{Prime64Offset59, Ring};
use std::sync::Arc;

type F = Prime64Offset59;

struct ExternalKind;
struct ExternalContext;
struct ExternalCommand;

struct ExternalFused {
    owner: StateOwnerCapability<InnerImage>,
}

impl FusedInnerOuterOperation<F> for ExternalFused {
    fn commit_inner_outer(
        &self,
        binding: &CommitmentStateBinding,
        plan: &UncompressedCommitPlan,
        _sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<UncompressedCommitmentOutput<F>, AkitaError> {
        let image = self.owner.bind(binding.clone(), 0, ());
        let u = RingVec::from_coeffs_with_ring_dim(
            vec![F::default(); plan.outer().output_coefficient_len()?],
            plan.outer().ring_dimension(),
        )?;
        UncompressedCommitmentOutput::new(image, u, plan)
    }
}

struct ExternalCompression {
    owner: StateOwnerCapability<CompressionState>,
}

impl CompressionOperation<F> for ExternalCompression {
    fn compress(
        &self,
        binding: &CommitmentStateBinding,
        plan: &akita_types::CompressionChainPlan,
        relation_mode: RingRelationMode,
        _u: RingVec<F>,
    ) -> Result<CompressionStageOutput<F>, AkitaError> {
        let terminal = plan.maps().last().ok_or_else(|| {
            AkitaError::InvalidSetup("external test received an empty compression plan".into())
        })?;
        let payload = RingVec::from_coeffs_with_ring_dim(
            vec![F::default(); terminal.output_coefficients()],
            terminal.ring_dimension(),
        )?;
        CompressionStageOutput::new(
            payload,
            self.owner.bind(binding.clone(), 0, ()),
            plan,
            relation_mode,
        )
    }
}

#[test]
fn downstream_crate_can_compose_fused_and_compression_backends() {
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
    let mut builder =
        CommitmentExecutorBuilder::new(setup.expanded.as_ref(), NoRetainedStatePolicy);
    let fused_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "external-fused",
            StageResources::none(),
        )
        .unwrap();
    let fused_owner = StateOwnerCapability::new();
    builder
        .register_fused(
            PreparedFusedCommitment::new(
                Arc::new(ExternalFused {
                    owner: fused_owner.clone(),
                }),
                fused_owner,
                fused_context,
                CommitmentRequestCapabilities::fused::<ExternalContext, ExternalCommand>(
                    BackendKindId::of::<ExternalKind>("external").unwrap(),
                    vec![PolynomialType::Dense(DenseType::Coefficients)],
                ),
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let compression_dimensions = plan
        .compression()
        .unwrap()
        .maps()
        .iter()
        .map(|map| map.ring_dimension())
        .collect();
    let compression_context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "external-compression",
            StageResources::none(),
        )
        .unwrap();
    let compression_owner = StateOwnerCapability::new();
    builder
        .register_compression(PreparedCompression::new(
            Arc::new(ExternalCompression {
                owner: compression_owner.clone(),
            }),
            compression_owner,
            compression_context,
            CompressionOperationCapabilities::new(
                StageDimensionCapabilities::new(compression_dimensions).unwrap(),
                vec![RingRelationMode::QuotientLift],
            )
            .unwrap(),
            None,
        ))
        .unwrap();
    let executor = builder.build().unwrap();
    let polynomial = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&polynomial];

    let output = executor.execute_full(&plan, &sources).unwrap();
    assert_eq!(
        output.terminal_payload().coeff_len(),
        plan.compression().unwrap().terminal_coefficients()
    );
    TerminalTFieldsMessage::from_row(output.terminal_payload()).unwrap();
}
