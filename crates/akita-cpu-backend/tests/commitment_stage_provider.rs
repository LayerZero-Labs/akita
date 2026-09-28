#![allow(missing_docs)]

//! A downstream crate can implement and install a commitment stage provider
//! using only the `commitment_backend` extension contracts.

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_cpu_backend::commitment_backend::{
    BackendInstanceId, BackendKindId, BackendStateRef, CommitInnerPlan, CommitmentOperationContext,
    CommitmentRequestCapabilities, CommitmentStageProvider, CommitmentStages,
    CommitmentStateBinding, DenseRepresentation, DenseType, InnerCommitOperation,
    InnerCommitOutput, InnerImage, InnerImageExportOperation, InnerImageInput,
    OuterCommitOperation, PolynomialRepresentation, PolynomialType, PreparedInnerCommitment,
    PreparedOuterCommitment, ResolvedCommitSource, StageDimensionCapabilities, StageResources,
    StateOwnerCapability, UncompressedCommitPlan,
};
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_types::{AkitaExpandedSetup, RingVec};
use jolt_field::Ring;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type Cfg = fp128::Dense;
type F = <Cfg as CommitmentConfig>::Field;
type E = <Cfg as CommitmentConfig>::ExtField;

const NUM_VARS: usize = 14;

/// Device stand-in that declines every plan before any source is materialized.
struct DecliningInner {
    owner: StateOwnerCapability<InnerImage>,
}

impl InnerCommitOperation<F> for DecliningInner {
    fn supports_plan(&self, _plan: &CommitInnerPlan) -> bool {
        false
    }

    fn commit_inner(
        &self,
        binding: &CommitmentStateBinding,
        _plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError> {
        for source in sources {
            match source.representation() {
                Some(PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(dense))) => {
                    let _ = dense.coefficients();
                }
                _ => {
                    return Err(AkitaError::InvalidInput(
                        "declining inner stage accepts only dense coefficients".into(),
                    ))
                }
            }
        }
        Ok(InnerCommitOutput::new(self.owner.bind(
            binding.clone(),
            0,
            (),
        )))
    }
}

impl InnerImageExportOperation<F> for DecliningInner {
    fn export_inner_rows(
        &self,
        _plan: &CommitInnerPlan,
        image: &BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        self.owner.value::<()>(image)?;
        Err(AkitaError::InvalidInput(
            "declining inner stage retains no rows".into(),
        ))
    }
}

struct DecliningOuter;

impl OuterCommitOperation<F> for DecliningOuter {
    fn supports_plan(&self, _plan: &UncompressedCommitPlan) -> bool {
        false
    }

    fn commit_outer(
        &self,
        _plan: &UncompressedCommitPlan,
        _inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "declining outer stage has no arithmetic".into(),
        ))
    }
}

#[derive(Default)]
struct DecliningProvider {
    requests: AtomicUsize,
}

impl CommitmentStageProvider<F> for DecliningProvider {
    fn stages<'a>(
        &'a self,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<CommitmentStages<'a, F>, AkitaError> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        let owner = StateOwnerCapability::new();
        let inner = Arc::new(DecliningInner {
            owner: owner.clone(),
        });
        let instance = BackendInstanceId::issue();
        let dimensions = StageDimensionCapabilities::new(vec![64, 128, 256, 512, 1024])?;
        Ok(CommitmentStages {
            inner: Some(PreparedInnerCommitment::new(
                inner.clone(),
                owner.clone(),
                CommitmentOperationContext::new(
                    expanded.descriptor(),
                    instance,
                    "declining-inner",
                    StageResources::none(),
                )?,
                CommitmentRequestCapabilities::split::<DecliningProvider>(
                    BackendKindId::of::<DecliningProvider>("declining")?,
                    vec![PolynomialType::Dense(DenseType::Coefficients)],
                ),
                dimensions.clone(),
                Some(inner),
            )?),
            outer: Some(PreparedOuterCommitment::new(
                Arc::new(DecliningOuter),
                owner,
                CommitmentOperationContext::new(
                    expanded.descriptor(),
                    instance,
                    "declining-outer",
                    StageResources::none(),
                )?,
                dimensions,
            )),
        })
    }
}

#[test]
fn downstream_provider_that_declines_keeps_cpu_commitment() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
            let capacity =
                akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, NUM_VARS, 1)
                    .unwrap()
                    .matrix_capacity();
            let setup =
                AkitaProverSetup::<F>::generate_with_capacity(NUM_VARS, 1, capacity).unwrap();
            let evals = (0..1usize << NUM_VARS)
                .map(|index| F::from_u64(index as u64 + 1))
                .collect::<Vec<_>>();
            let poly = DensePoly::<F>::from_field_evals(NUM_VARS, &evals).unwrap();
            let commit = |backend: &CpuBackend<F, E>| {
                backend
                    .commit(
                        &catalog,
                        &backend.import_source(vec![poly.clone()]).unwrap(),
                        GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .unwrap()
                    .committed_group
            };

            let expected = commit(&CpuBackend::new(setup.expanded.clone()).unwrap());
            let provider = Arc::new(DecliningProvider::default());
            let installed: Arc<dyn CommitmentStageProvider<F>> = provider.clone();
            let backend = CpuBackend::<F, E>::new(setup.expanded.clone())
                .unwrap()
                .with_commitment_stage_provider(installed);

            assert_eq!(commit(&backend), expected);
            assert_eq!(provider.requests.load(Ordering::SeqCst), 1);
        })
        .unwrap()
        .join()
        .unwrap();
}
