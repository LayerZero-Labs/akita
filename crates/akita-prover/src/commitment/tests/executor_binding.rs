use super::*;
use crate::commitment::{
    CommitmentSource, DenseType, InnerCommitOperation, InnerImage, NoRetainedStatePolicy,
    ResolvedCommitSource, StateOwnerCapability,
};
use crate::{AkitaProverSetup, DensePoly};
use akita_challenges::SparseChallengeConfig;
use akita_types::{
    CommittedGroupParams, SetupMatrixCapacity, SisModulusProfileId, TerminalFoldParams,
};
use jolt_field::{Prime64Offset59, Ring};
use std::sync::Arc;

type F = Prime64Offset59;

struct ReboundInner {
    owner: StateOwnerCapability<InnerImage>,
    rebound: bool,
}

impl InnerCommitOperation<F> for ReboundInner {
    fn commit_inner(
        &self,
        binding: &CommitmentStateBinding,
        plan: &crate::compute::CommitInnerPlan,
        _sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        let output_binding = if self.rebound {
            CommitmentStateBinding::new(
                binding.setup().clone(),
                *plan,
                binding.source_count(),
                binding.relation_mode(),
                binding.compression_plan().cloned(),
            )?
        } else {
            binding.clone()
        };
        Ok(InnerCommitOutput::new(self.owner.bind(
            output_binding,
            0,
            (),
        )))
    }
}

#[test]
fn executor_rejects_same_shape_rebinding_and_foreign_state_owners() {
    struct Backend;
    struct InnerContext;

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
    let plan =
        CommitmentExecutionPlan::for_terminal(&TerminalFoldParams::from_expanded_group(params))
            .unwrap();
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
    let context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "rebound-inner",
            StageResources::none(),
        )
        .unwrap();
    let operation = Arc::new(ReboundInner {
        owner: StateOwnerCapability::new(),
        rebound: true,
    });
    let capabilities = CommitmentRequestCapabilities::split::<InnerContext>(
        BackendKindId::of::<Backend>("rebound").unwrap(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
    );
    builder
        .register_inner(
            PreparedInnerCommitment::new(
                operation.clone(),
                operation.owner.clone(),
                context,
                capabilities,
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let executor = builder.build().unwrap();
    let consumer_error = executor
        .preflight_prover_state_consumers(&plan)
        .expect_err("missing inner state consumer must fail preflight");
    assert!(matches!(
        consumer_error,
        AkitaError::InvalidInput(message) if message.contains("inner-relation state operation")
    ));
    let source = DensePoly::from_field_evals(9, vec![F::from_u64(1); 512]).unwrap();
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];

    let error = match executor.execute_inner(&plan, &sources) {
        Err(error) => error,
        Ok(_) => panic!("rebound state must be rejected"),
    };
    assert!(
        matches!(
            &error,
            AkitaError::InvalidInput(message) if message.contains("different commitment request")
        ),
        "unexpected error: {error:?}"
    );

    let mut builder =
        CommitmentExecutorBuilder::new(setup.expanded.as_ref(), NoRetainedStatePolicy);
    let context = builder
        .operation_context(
            builder.issue_backend_instance(),
            "wrong-owner-inner",
            StageResources::none(),
        )
        .unwrap();
    let operation = Arc::new(ReboundInner {
        owner: StateOwnerCapability::new(),
        rebound: false,
    });
    let capabilities = CommitmentRequestCapabilities::split::<InnerContext>(
        BackendKindId::of::<Backend>("wrong-owner").unwrap(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
    );
    builder
        .register_inner(
            PreparedInnerCommitment::new(
                operation,
                StateOwnerCapability::new(),
                context,
                capabilities,
                StageDimensionCapabilities::new(vec![64]).unwrap(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let error = match builder.build().unwrap().execute_inner(&plan, &sources) {
        Err(error) => error,
        Ok(_) => panic!("foreign state owner must be rejected"),
    };
    assert!(matches!(
        error,
        AkitaError::InvalidInput(message) if message.contains("different prepared implementation")
    ));
}
