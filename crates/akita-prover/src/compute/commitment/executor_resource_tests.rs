use super::*;
use crate::compute::{
    BackendInstanceId, CommitmentNttRequirement, CommitmentOperationContext,
    CommitmentResourceControl, DenseType, NttCacheOwnerId,
};
use crate::AkitaProverSetup;
use akita_challenges::SparseChallengeConfig;
use akita_types::{CommittedGroupParams, SetupMatrixCapacity, SisModulusProfileId};
use jolt_field::Prime64Offset59;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

type F = Prime64Offset59;

struct StreamedResources {
    setup: akita_types::AkitaSetupDescriptor,
    ensures: Arc<AtomicUsize>,
}

impl CommitmentResourceControl<F> for StreamedResources {
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
        Ok(false)
    }

    fn cache_owner_id(&self) -> NttCacheOwnerId {
        NttCacheOwnerId::from_owner(self.ensures.as_ref())
    }

    fn planned_ntt_cache_entry_bytes(
        &self,
        _requirement: CommitmentNttRequirement,
    ) -> Result<usize, AkitaError> {
        Ok(0)
    }

    fn release_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        Ok(0)
    }
}

#[test]
fn stage_registration_skips_streamed_slots() {
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        9,
        1,
        SetupMatrixCapacity {
            num_field_elements: 4 * 64,
        },
    )
    .unwrap();
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
    let terminal = akita_types::TerminalFoldParams::from_expanded_group(params);
    let plan = CommitmentExecutionPlan::for_terminal(&terminal).unwrap();
    let requirement = plan
        .inner_ntt_requirement(PolynomialType::Dense(DenseType::Coefficients))
        .unwrap()
        .unwrap();
    let ensures = Arc::new(AtomicUsize::new(0));
    let registration = super::super::prepared::PreparedStage::new(
        Arc::new(()),
        CommitmentOperationContext {
            backend_instance: BackendInstanceId::issue(),
            name: "streamed-inner",
            resources: StageResources::controlled(StreamedResources {
                setup: setup.expanded.descriptor().clone(),
                ensures: ensures.clone(),
            }),
        },
    );

    registration.ensure_ntt_slot(requirement).unwrap();
    assert_eq!(ensures.load(Ordering::SeqCst), 0);
}
