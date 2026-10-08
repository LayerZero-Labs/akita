//! Per-fold instance selection; typed executors retain all kernel state.
use super::registry::*;
pub use super::registry::{BackendId, BackendRegistry, FixedFoldRoute, FoldExecutionPolicy};
use super::{root, suffix, FoldPublicState, SuffixProverState};
use crate::backend::*;
use crate::SelectedProverOpeningData;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::BasisMode;
use akita_serialization::AkitaSerialize;
use akita_types::{AkitaSetupDescriptor, Commitment, FpExtEncoding, OpeningClaims, ProverGrinding};
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Fold, MulBaseUnreduced, PseudoMersenne, Ring,
    Unreduced,
};
use std::any::{Any, TypeId};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PROOF: AtomicU64 = AtomicU64::new(1);

impl<Cfg, B> Executor<Cfg> for TypedExecutor<'_, Cfg, B>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
{
    fn identity(&self) -> BackendInstanceIdentity {
        self.backend.instance_identity()
    }
    fn backend_type_id(&self) -> TypeId {
        TypeId::of::<B>()
    }
    fn root(
        &self,
        id: BackendId,
        registry: &BackendRegistry<'_, Cfg>,
        policy: &mut dyn FoldExecutionPolicy,
        request: &ProofRequest<'_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        claims: &OpeningClaims<'_, Cfg::ExtField, Commitment<Cfg::Field>>,
        handles: &dyn Any,
        basis: BasisMode,
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError> {
        let handles = handles
            .downcast_ref::<Vec<B::CommitmentHandle>>()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "root commitment handle family differs from selected backend".into(),
                )
            })?;
        let session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("root executor re-entry".into()))?;
        let mut handoff = FoldHandoff {
            registry,
            policy,
            request,
            current: id,
            adopted: None,
        };
        let state = root::prove_root::<Cfg, B>(
            request.setup,
            self.prefixes,
            self.backend,
            claims,
            request.layout,
            handles,
            request.schedule,
            session
                .as_ref()
                .ok_or_else(|| AkitaError::Internal("root executor has no session".into()))?,
            grinding,
            basis,
            &mut handoff,
        )?
        .next_state;
        handoff.complete(self.backend, state)
    }
    fn validate_prefixes(
        &self,
        request: &ProofRequest<'_, Cfg>,
        requirements: &FoldExecutionRequirements<'_>,
    ) -> Result<(), AkitaError> {
        let schedule = requirements.schedule();
        let current = if requirements.level() == 0 {
            Some(&schedule.root.params)
        } else {
            schedule
                .recursive_folds
                .get(requirements.level() - 1)
                .map(|f| &f.params)
        };
        let successor = schedule
            .recursive_folds
            .get(requirements.level())
            .map(|f| &f.params);
        for params in current.into_iter().chain(successor) {
            if let Some(id) = params.setup_prefix().and_then(|p| p.slot_id()) {
                let local = self.prefixes.get(&id).ok_or_else(|| {
                    AkitaError::InvalidSetup("executor lacks its setup-prefix handle".into())
                })?;
                let public = request.prefixes.get(&id).ok_or_else(|| {
                    AkitaError::InvalidSetup("admitted setup prefix is missing".into())
                })?;
                if &local.public != *public {
                    return Err(AkitaError::InvalidSetup(
                        "executor setup prefix differs from the admitted commitment".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    fn prepare_executor(&self, request: &ProofRequest<'_, Cfg>) -> Result<(), AkitaError> {
        let mut session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        *session = Some(self.backend.prepare_executor(
            request.setup,
            request.schedules,
            request.schedule,
            request.layout,
        )?);
        Ok(())
    }
    fn begin_fold(&self, request: &ProofRequest<'_, Cfg>, level: usize) -> Result<(), AkitaError> {
        let requirements = FoldExecutionRequirements::new(request.schedule, request.layout, level)?;
        self.validate_prefixes(request, &requirements)?;
        let session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        self.backend.begin_fold(
            session
                .as_ref()
                .ok_or_else(|| AkitaError::Internal("executor has not been prepared".into()))?,
            &requirements,
        )
    }
    fn import_packet(
        &self,
        id: BackendId,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
        packet: Box<dyn Any + Send>,
    ) -> Result<ResidentSuccessor, AkitaError> {
        let session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during import".into())
        })?;
        let packet = packet.downcast::<B::ImportPacket>().map_err(|_| {
            AkitaError::InvalidInput("destination import packet family mismatch".into())
        })?;
        let imported = self.backend.import_successor(
            session
                .as_ref()
                .ok_or_else(|| AkitaError::InvalidInput("destination was not admitted".into()))?,
            plan,
            *packet,
        )?;
        let (witness, material) = imported.into_parts();
        Ok(ResidentSuccessor::new(id, witness, material))
    }
    fn recursive(
        &self,
        id: BackendId,
        registry: &BackendRegistry<'_, Cfg>,
        policy: &mut dyn FoldExecutionPolicy,
        request: &ProofRequest<'_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        public: FoldPublicState<Cfg::Field, Cfg::ExtField>,
        resident: ResidentSuccessor,
        index: usize,
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError> {
        let session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during fold".into())
        })?;
        let (witness_handle, commitment_material) =
            resident.into_pair::<B::WitnessHandle, B::CommitmentMaterialHandle>(id)?;
        let mut handoff = FoldHandoff {
            registry,
            policy,
            request,
            current: id,
            adopted: None,
        };
        let state = suffix::prove_recursive_step::<Cfg, B>(
            request.setup,
            self.prefixes,
            self.backend,
            grinding,
            SuffixProverState {
                public,
                witness_handle,
                commitment_material,
            },
            request.schedule,
            session
                .as_ref()
                .ok_or_else(|| AkitaError::InvalidInput("executor was not admitted".into()))?,
            index,
            &mut handoff,
        )?;
        handoff.complete(self.backend, state)
    }
    fn terminal(
        &self,
        id: BackendId,
        request: &ProofRequest<'_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        public: FoldPublicState<Cfg::Field, Cfg::ExtField>,
        resident: ResidentSuccessor,
    ) -> Result<(), AkitaError> {
        let session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during terminal fold".into())
        })?;
        let (witness_handle, commitment_material) =
            resident.into_pair::<B::WitnessHandle, B::CommitmentMaterialHandle>(id)?;
        suffix::prove_terminal_suffix::<Cfg::Field, Cfg::ExtField, B>(
            self.backend,
            grinding,
            public.level,
            SuffixProverState {
                public,
                witness_handle,
                commitment_material,
            },
            &request.schedule.terminal,
            session.as_ref().ok_or_else(|| {
                AkitaError::InvalidInput("terminal executor was not admitted".into())
            })?,
        )
    }
    fn finish(&self) -> Result<(), AkitaError> {
        let mut session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during completion".into())
        })?;
        if let Some(active) = session.as_ref() {
            self.backend.finish_scope(active)?;
        }
        session.take();
        Ok(())
    }
    fn abort(&self) {
        if let Ok(mut session) = self.session.try_borrow_mut() {
            if let Some(active) = session.take() {
                self.backend.abort_scope_best_effort(&active);
            }
        }
    }
}

/// Prove with one transcript and route-selected registry entries for every fold.
pub fn batched_prove<'a, Cfg, H>(
    expanded: &AkitaSetupDescriptor,
    schedules: &TrustedScheduleCatalog<Cfg>,
    registry: &BackendRegistry<'_, Cfg>,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
    transcript_session: &[u8],
    basis: BasisMode,
    policy: &mut dyn FoldExecutionPolicy,
) -> Result<Vec<u8>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + 'static,
    H: CommitmentHandleMetadata + 'static,
{
    let (resolved, claims, grinding_plan, descriptor_bytes) =
        root::resolve_root::<Cfg, H>(expanded, schedules, opening, basis)?;
    let requirements =
        FoldExecutionRequirements::new(resolved.schedule(), claims.opening_layout(), 0)?;
    let root_id = policy.choose_backend(None, &requirements)?;
    let root_executor = registry.slot(root_id)?;
    if registry.active.replace(true) {
        return Err(AkitaError::InvalidInput(
            "backend registry already has an active proof".into(),
        ));
    }
    let _guard = RegistryProofGuard(registry);
    let identity = NEXT_PROOF
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| AkitaError::Internal("proof identity exhausted".into()))?;
    let request = ProofRequest {
        setup: expanded,
        schedules,
        schedule: resolved.schedule(),
        layout: claims.opening_layout(),
        prefixes: &registry.public_prefixes,
        identity,
    };
    for executor in &registry.slots {
        executor.prepare_executor(&request)?;
    }
    root_executor.begin_fold(&request, 0)?;
    let channel = akita_transcript::new_prover_channel(transcript_session, &descriptor_bytes)?;
    let mut grinding = ProverGrinding::new(channel, &grinding_plan);
    let (mut public, mut resident) = root_executor.root(
        root_id,
        registry,
        policy,
        &request,
        &mut grinding,
        claims.opening_claims(),
        &claims.groups,
        basis,
    )?;
    for index in 0..request.schedule.recursive_folds.len() {
        let owner = resident.owner;
        (public, resident) = registry.slot(owner)?.recursive(
            owner,
            registry,
            policy,
            &request,
            &mut grinding,
            public,
            resident,
            index,
        )?;
    }
    let owner = resident.owner;
    let levels = public.level + 1;
    registry
        .slot(owner)?
        .terminal(owner, &request, &mut grinding, public, resident)?;
    if levels != request.schedule.num_fold_levels() {
        return Err(AkitaError::Internal(
            "proved fold count differs from schedule".into(),
        ));
    }
    let proof = grinding.finish()?;
    for slot in &registry.slots {
        slot.finish()?;
    }
    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_dispatch_checks_both_instance_and_handle_family() {
        let owner = BackendId {
            registry: 1,
            slot: 0,
        };
        let foreign = BackendId {
            registry: 1,
            slot: 1,
        };
        assert!(ResidentSuccessor::new(owner, 1u8, 2u8)
            .into_pair::<u8, u8>(foreign)
            .is_err());
        assert!(ResidentSuccessor::new(owner, 1u8, 2u8)
            .into_pair::<u16, u16>(owner)
            .is_err());
        assert_eq!(
            ResidentSuccessor::new(owner, 1u8, 2u8)
                .into_pair::<u8, u8>(owner)
                .unwrap(),
            (1, 2)
        );
    }
}
