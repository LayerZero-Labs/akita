//! Per-fold instance selection; typed executors retain all kernel state.
use super::registry::*;
pub use super::registry::{BackendId, BackendRegistry, FixedFoldRoute, FoldExecutionPolicy};
use super::{root, suffix, FoldPublicState, SuffixProverState};
use crate::backend::*;
use crate::{ProverOpeningData, SelectedProverOpeningData};
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
        handoff: &mut FoldHandoff<'_, '_, '_, Cfg>,
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
        let request = handoff.request;
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
                .ok_or_else(|| AkitaError::Internal("root executor has no session".into()))?
                .session(),
            grinding,
            basis,
            handoff,
        )?
        .next_state;
        handoff.complete(self.backend, state)
    }
    fn prepare_executor(&self, request: &ProofRequest<'_, Cfg>) -> Result<(), AkitaError> {
        let mut session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        let prepared = self.backend.prepare_executor(
            request.setup,
            request.schedules,
            request.schedule,
            request.layout,
        )?;
        *session = Some(ProofScope::admitted(self.backend, prepared));
        Ok(())
    }
    fn begin_fold(&self, request: &ProofRequest<'_, Cfg>, level: usize) -> Result<(), AkitaError> {
        let requirements = FoldExecutionRequirements::new(request.schedule, request.layout, level)?;
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
                self.prefixes.get(&id).ok_or_else(|| {
                    AkitaError::InvalidSetup("executor lacks its setup-prefix handle".into())
                })?;
            }
        }
        let session = self
            .session
            .try_borrow_mut()
            .map_err(|_| AkitaError::InvalidInput("backend executor re-entry".into()))?;
        self.backend.begin_fold(
            session
                .as_ref()
                .ok_or_else(|| AkitaError::Internal("executor has not been prepared".into()))?
                .session(),
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
                .ok_or_else(|| AkitaError::InvalidInput("destination was not admitted".into()))?
                .session(),
            plan,
            *packet,
        )?;
        let (witness, material) = imported.into_parts();
        Ok(ResidentSuccessor::new(id, witness, material))
    }
    fn recursive(
        &self,
        handoff: &mut FoldHandoff<'_, '_, '_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        public: FoldPublicState<Cfg::Field, Cfg::ExtField>,
        resident: ResidentSuccessor,
        index: usize,
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError> {
        let session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during fold".into())
        })?;
        let (witness_handle, commitment_material) =
            resident.into_pair::<B::WitnessHandle, B::CommitmentMaterialHandle>(handoff.current)?;
        let request = handoff.request;
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
                .ok_or_else(|| AkitaError::InvalidInput("executor was not admitted".into()))?
                .session(),
            index,
            handoff,
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
            session
                .as_ref()
                .ok_or_else(|| {
                    AkitaError::InvalidInput("terminal executor was not admitted".into())
                })?
                .session(),
        )
    }
    fn finish(&self) -> Result<(), AkitaError> {
        let mut session = self.session.try_borrow_mut().map_err(|_| {
            AkitaError::InvalidInput("backend executor re-entry during completion".into())
        })?;
        if let Some(scope) = session.take() {
            scope.finish()?;
        }
        Ok(())
    }
    fn abort(&self) {
        if let Ok(mut session) = self.session.try_borrow_mut() {
            session.take();
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
    let channel = akita_transcript::new_prover_channel(transcript_session, &descriptor_bytes)?;
    let mut grinding = ProverGrinding::new(channel, &grinding_plan);
    prove_resolved::<Cfg, H>(
        expanded,
        schedules,
        registry,
        &claims,
        resolved.schedule(),
        basis,
        policy,
        &mut grinding,
    )?;
    grinding.finish()
}

/// Append a multi-group opening to a caller-owned Spongefish session.
///
/// Applications must create the session with Akita's `new_prover_channel` or
/// `new_verifier_channel`, binding their application instance bytes; arbitrary raw
/// spongefish states cannot be adopted.
///
/// Absorbs the native protocol version and complete setup/catalog/profile
/// descriptor before any opening challenge. Completion checks this opening's
/// grinding plan and leaves the caller's session available for its next message.
#[allow(clippy::too_many_arguments)]
pub fn batched_prove_in_session<'a, Cfg, H>(
    expanded: &AkitaSetupDescriptor,
    schedules: &TrustedScheduleCatalog<Cfg>,
    registry: &BackendRegistry<'_, Cfg>,
    opening: SelectedProverOpeningData<'a, Cfg::ExtField, H, Cfg::Field>,
    session: &mut akita_transcript::ProverSession,
    basis: BasisMode,
    policy: &mut dyn FoldExecutionPolicy,
) -> Result<(), AkitaError>
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
    akita_transcript::absorb_subprotocol_context(
        session,
        b"akita-pcs",
        akita_transcript::PROTOCOL_VERSION,
        &descriptor_bytes,
    )?;
    let mut grinding = ProverGrinding::new_borrowed(session, &grinding_plan);
    prove_resolved::<Cfg, H>(
        expanded,
        schedules,
        registry,
        &claims,
        resolved.schedule(),
        basis,
        policy,
        &mut grinding,
    )?;
    grinding.finish_opening()
}

#[allow(clippy::too_many_arguments)]
fn prove_resolved<Cfg, H>(
    expanded: &AkitaSetupDescriptor,
    schedules: &TrustedScheduleCatalog<Cfg>,
    registry: &BackendRegistry<'_, Cfg>,
    claims: &ProverOpeningData<'_, Cfg::ExtField, H, Cfg::Field>,
    schedule: &akita_params::FoldSchedule,
    basis: BasisMode,
    policy: &mut dyn FoldExecutionPolicy,
    grinding: &mut ProverGrinding<'_>,
) -> Result<(), AkitaError>
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
    if registry.active.replace(true) {
        return Err(AkitaError::InvalidInput(
            "backend registry already has an active proof".into(),
        ));
    }
    let _guard = RegistryProofGuard(registry);
    let mut owners = Vec::with_capacity(schedule.num_fold_levels());
    let mut selected = vec![false; registry.slots.len()];
    for level in 0..schedule.num_fold_levels() {
        let requirements =
            FoldExecutionRequirements::new(schedule, claims.opening_layout(), level)?;
        let current = owners.last().copied();
        let owner = policy.choose_backend(current, &requirements)?;
        registry.slot(owner)?;
        if let Some(previous) = current.filter(|previous| *previous != owner) {
            registry.bridge(previous, owner)?;
        }
        selected[owner.slot] = true;
        owners.push(owner);
    }
    let root_id = *owners
        .first()
        .ok_or_else(|| AkitaError::Internal("proof schedule has no fold levels".into()))?;
    let root_executor = registry.slot(root_id)?;
    let identity = NEXT_PROOF
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| AkitaError::Internal("proof identity exhausted".into()))?;
    let request = ProofRequest {
        setup: expanded,
        schedules,
        schedule,
        layout: claims.opening_layout(),
        identity,
    };
    for (executor, selected) in registry.slots.iter().zip(&selected) {
        if *selected {
            executor.prepare_executor(&request)?;
        }
    }
    for (level, owner) in owners.iter().enumerate() {
        registry.slot(*owner)?.begin_fold(&request, level)?;
    }
    let mut handoff = FoldHandoff {
        registry,
        request: &request,
        current: root_id,
        next: *owners
            .get(1)
            .ok_or_else(|| AkitaError::Internal("proof schedule has no successor fold".into()))?,
        adopted: None,
    };
    let (mut public, mut resident) = root_executor.root(
        &mut handoff,
        grinding,
        claims.opening_claims(),
        &claims.groups,
        basis,
    )?;
    for index in 0..request.schedule.recursive_folds.len() {
        let owner = resident.owner;
        handoff.current = owner;
        handoff.next = *owners
            .get(index + 2)
            .ok_or_else(|| AkitaError::Internal("resolved route has no successor fold".into()))?;
        (public, resident) =
            registry
                .slot(owner)?
                .recursive(&mut handoff, grinding, public, resident, index)?;
    }
    let owner = resident.owner;
    let levels = public.level + 1;
    registry
        .slot(owner)?
        .terminal(owner, &request, grinding, public, resident)?;
    if levels != request.schedule.num_fold_levels() {
        return Err(AkitaError::Internal(
            "proved fold count differs from schedule".into(),
        ));
    }
    for (slot, selected) in registry.slots.iter().zip(selected) {
        if selected {
            slot.finish()?;
        }
    }
    Ok(())
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
