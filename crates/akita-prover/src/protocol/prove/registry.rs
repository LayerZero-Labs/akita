//! Backend registry, bridges, and fold handoff routing.
use super::{FoldPublicState, SuffixProverState};
use crate::backend::*;
use crate::protocol::ring_switch::NextWitnessState;
use crate::SetupPrefixProverRegistry;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{BasisMode, FoldSchedule, OpeningClaimsLayout, SetupPrefixSlotId};
use akita_serialization::AkitaSerialize;
use akita_types::{
    AkitaSetupDescriptor, Commitment, FpExtEncoding, OpeningClaims, ProverGrinding,
    SetupPrefixVerifierSlot,
};
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Fold, MulBaseUnreduced, PseudoMersenne, Ring,
    Unreduced,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::{BTreeMap, HashMap},
};

static NEXT_REGISTRY: AtomicU64 = AtomicU64::new(1);

/// Identifies one instance in one registry. Foreign registry IDs are rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendId {
    pub(super) registry: u64,
    pub(super) slot: usize,
}

/// Caller-selected registry entries for every fold, including the root.
pub trait FoldExecutionPolicy {
    /// `current` is absent at level 0 and identifies the producer at later levels.
    fn choose_backend(
        &mut self,
        current: Option<BackendId>,
        requirements: &FoldExecutionRequirements<'_>,
    ) -> Result<BackendId, AkitaError>;
}

/// Fixed registry IDs indexed by fold level, starting with the root at level 0.
pub struct FixedFoldRoute {
    route: Vec<BackendId>,
}
impl FixedFoldRoute {
    pub fn new(route: Vec<BackendId>) -> Self {
        Self { route }
    }
}
impl FoldExecutionPolicy for FixedFoldRoute {
    fn choose_backend(
        &mut self,
        _: Option<BackendId>,
        requirements: &FoldExecutionRequirements<'_>,
    ) -> Result<BackendId, AkitaError> {
        self.route
            .get(requirements.level())
            .copied()
            .ok_or_else(|| {
                AkitaError::InvalidInput("fixed backend route is shorter than the schedule".into())
            })
    }
}

pub(super) struct ResidentSuccessor {
    pub(super) owner: BackendId,
    pair: Box<dyn Any + Send>,
}
impl ResidentSuccessor {
    pub(super) fn new<W: Send + 'static, M: Send + 'static>(
        owner: BackendId,
        witness: W,
        material: M,
    ) -> Self {
        Self {
            owner,
            pair: Box::new((witness, material)),
        }
    }
    pub(super) fn into_pair<W: Send + 'static, M: Send + 'static>(
        self,
        owner: BackendId,
    ) -> Result<(W, M), AkitaError> {
        if self.owner != owner {
            return Err(AkitaError::InvalidInput(
                "resident successor belongs to another instance".into(),
            ));
        }
        self.pair
            .downcast::<(W, M)>()
            .map(|pair| *pair)
            .map_err(|_| {
                AkitaError::InvalidInput("resident successor handle family mismatch".into())
            })
    }
}

pub(super) struct ProofRequest<'p, Cfg: CommitmentConfig> {
    pub(super) setup: &'p AkitaSetupDescriptor,
    pub(super) schedules: &'p TrustedScheduleCatalog<Cfg>,
    pub(super) schedule: &'p FoldSchedule,
    pub(super) layout: &'p OpeningClaimsLayout,
    pub(super) prefixes: BTreeMap<SetupPrefixSlotId, &'p SetupPrefixVerifierSlot<Cfg::Field>>,
    pub(super) identity: u64,
}

pub(super) type ExecutionContinuation<Cfg> = (
    FoldPublicState<<Cfg as CommitmentConfig>::Field, <Cfg as CommitmentConfig>::ExtField>,
    ResidentSuccessor,
);

pub(super) trait Executor<Cfg: CommitmentConfig> {
    fn identity(&self) -> BackendInstanceIdentity;
    fn backend_type_id(&self) -> TypeId;
    fn public_prefixes(&self) -> BTreeMap<SetupPrefixSlotId, &SetupPrefixVerifierSlot<Cfg::Field>>;
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError>;
    fn validate_prefixes(
        &self,
        request: &ProofRequest<'_, Cfg>,
        requirements: &FoldExecutionRequirements<'_>,
    ) -> Result<(), AkitaError>;
    fn prepare_executor(&self, request: &ProofRequest<'_, Cfg>) -> Result<(), AkitaError>;
    fn begin_fold(&self, request: &ProofRequest<'_, Cfg>, level: usize) -> Result<(), AkitaError>;
    fn import_packet(
        &self,
        id: BackendId,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
        packet: Box<dyn Any + Send>,
    ) -> Result<ResidentSuccessor, AkitaError>;
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError>;
    fn terminal(
        &self,
        id: BackendId,
        request: &ProofRequest<'_, Cfg>,
        grinding: &mut ProverGrinding<'_>,
        public: FoldPublicState<Cfg::Field, Cfg::ExtField>,
        resident: ResidentSuccessor,
    ) -> Result<(), AkitaError>;
    fn finish(&self) -> Result<(), AkitaError>;
    fn abort(&self);
}

impl<Cfg: CommitmentConfig> dyn Executor<Cfg> + '_ {
    fn import_as<B>(
        &self,
        id: BackendId,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
        packet: B::ImportPacket,
    ) -> Result<ResidentSuccessor, AkitaError>
    where
        B: SuccessorImportKernel<Cfg::Field, Cfg::ExtField>,
    {
        self.import_packet(id, plan, Box::new(packet))
    }
}

pub(super) struct TypedExecutor<
    'a,
    Cfg: CommitmentConfig,
    B: ProverBackend<Cfg::Field, Cfg::ExtField>,
> {
    pub(super) backend: &'a B,
    pub(super) prefixes: &'a SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    pub(super) session: RefCell<Option<B::ProofSessionHandle>>,
}

/// Reusable instances and setup resources; all sessions are prepared before proving.
pub struct BackendRegistry<'a, Cfg: CommitmentConfig> {
    identity: u64,
    pub(super) slots: Vec<Box<dyn Executor<Cfg> + 'a>>,
    bridges: HashMap<(TypeId, TypeId), Box<dyn ErasedBridge<Cfg>>>,
    pub(super) active: std::cell::Cell<bool>,
}

impl<'a, Cfg> BackendRegistry<'a, Cfg>
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
{
    pub fn new() -> Result<Self, AkitaError> {
        let identity = NEXT_REGISTRY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| AkitaError::Internal("backend registry identity exhausted".into()))?;
        Ok(Self {
            identity,
            slots: Vec::new(),
            bridges: HashMap::new(),
            active: std::cell::Cell::new(false),
        })
    }
    /// Register a directed edge. Distinct instances of the same type need an edge too.
    pub fn register_bridge<A, B>(&mut self) -> Result<(), AkitaError>
    where
        A: SuccessorExportKernel<Cfg::Field, Cfg::ExtField> + 'static,
        B: SuccessorImportKernel<Cfg::Field, Cfg::ExtField> + 'static,
        Edge<A, B>: SuccessorBridge<A, B, Cfg::Field, Cfg::ExtField>,
    {
        let key = (TypeId::of::<A>(), TypeId::of::<B>());
        if self.bridges.contains_key(&key) {
            return Err(AkitaError::InvalidInput(
                "backend bridge is already registered".into(),
            ));
        }
        self.bridges.insert(key, Box::new(Edge::<A, B>::default()));
        Ok(())
    }
    pub fn register<B>(
        &mut self,
        backend: &'a B,
        prefixes: &'a SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    ) -> Result<BackendId, AkitaError>
    where
        B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
    {
        if self
            .slots
            .iter()
            .any(|slot| slot.identity() == backend.instance_identity())
        {
            return Err(AkitaError::InvalidInput(
                "backend instance is already registered".into(),
            ));
        }
        let id = BackendId {
            registry: self.identity,
            slot: self.slots.len(),
        };
        let executor = Box::new(TypedExecutor {
            backend,
            prefixes,
            session: RefCell::new(None),
        });
        self.slots.push(executor);
        Ok(id)
    }
}

impl<Cfg: CommitmentConfig> BackendRegistry<'_, Cfg> {
    pub(super) fn slot(&self, id: BackendId) -> Result<&dyn Executor<Cfg>, AkitaError> {
        if id.registry != self.identity {
            return Err(AkitaError::InvalidInput(
                "backend ID belongs to another registry".into(),
            ));
        }
        self.slots
            .get(id.slot)
            .map(|slot| slot.as_ref())
            .ok_or_else(|| AkitaError::InvalidInput("backend ID is not registered".into()))
    }
}

/// Type-erased producer export; borrows stay on the active fold session.
trait ErasedExporter<Cfg: CommitmentConfig> {
    fn export_packet(
        &self,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
    ) -> Result<Box<dyn Any + Send>, AkitaError>;
}

impl<Cfg: CommitmentConfig> dyn ErasedExporter<Cfg> + '_ {
    fn export_as<A>(
        &self,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
    ) -> Result<A::ExportPacket, AkitaError>
    where
        A: SuccessorExportKernel<Cfg::Field, Cfg::ExtField>,
    {
        self.export_packet(plan)?
            .downcast::<A::ExportPacket>()
            .map(|packet| *packet)
            .map_err(|_| AkitaError::InvalidInput("bridge export packet family mismatch".into()))
    }
}

struct TypedExporter<'a, Cfg, B>
where
    Cfg: CommitmentConfig,
    B: SuccessorExportKernel<Cfg::Field, Cfg::ExtField>,
{
    backend: &'a B,
    session: &'a B::ProofSessionHandle,
    witness: &'a B::WitnessHandle,
    material: &'a B::CommitmentMaterialHandle,
}

impl<Cfg, B> ErasedExporter<Cfg> for TypedExporter<'_, Cfg, B>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding,
    B: SuccessorExportKernel<Cfg::Field, Cfg::ExtField>,
{
    fn export_packet(
        &self,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
    ) -> Result<Box<dyn Any + Send>, AkitaError> {
        Ok(Box::new(self.backend.export_successor(
            self.session,
            self.witness,
            self.material,
            plan,
        )?))
    }
}

trait ErasedBridge<Cfg: CommitmentConfig> {
    fn transfer(
        &self,
        src: &dyn ErasedExporter<Cfg>,
        dst: &dyn Executor<Cfg>,
        id: BackendId,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
    ) -> Result<ResidentSuccessor, AkitaError>;
}

impl<Cfg, A, B> ErasedBridge<Cfg> for Edge<A, B>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding,
    A: SuccessorExportKernel<Cfg::Field, Cfg::ExtField>,
    B: SuccessorImportKernel<Cfg::Field, Cfg::ExtField>,
    Self: SuccessorBridge<A, B, Cfg::Field, Cfg::ExtField>,
{
    fn transfer(
        &self,
        src: &dyn ErasedExporter<Cfg>,
        dst: &dyn Executor<Cfg>,
        id: BackendId,
        plan: &ValidatedSuccessorHandoffPlan<'_, Cfg::Field>,
    ) -> Result<ResidentSuccessor, AkitaError> {
        let export = src.export_as::<A>(plan)?;
        let packet = Self::convert(export, plan)?;
        plan.validate_export(packet.descriptor())?;
        dst.import_as::<B>(id, plan, packet)
    }
}

pub(super) struct FoldHandoff<'r, 'a, 'p, Cfg: CommitmentConfig> {
    pub(super) registry: &'r BackendRegistry<'a, Cfg>,
    pub(super) policy: &'r mut dyn FoldExecutionPolicy,
    pub(super) request: &'r ProofRequest<'p, Cfg>,
    pub(super) current: BackendId,
    pub(super) adopted: Option<ResidentSuccessor>,
}

pub(super) struct RegistryProofGuard<'r, 'a, Cfg: CommitmentConfig>(
    pub(super) &'r BackendRegistry<'a, Cfg>,
);
impl<Cfg: CommitmentConfig> Drop for RegistryProofGuard<'_, '_, Cfg> {
    fn drop(&mut self) {
        for slot in &self.0.slots {
            slot.abort();
        }
        self.0.active.set(false);
    }
}

impl<Cfg: CommitmentConfig> FoldHandoff<'_, '_, '_, Cfg> {
    pub(super) fn complete<B: ProverBackend<Cfg::Field, Cfg::ExtField>>(
        &mut self,
        backend: &B,
        state: SuffixProverState<
            Cfg::Field,
            Cfg::ExtField,
            B::CommitmentMaterialHandle,
            B::WitnessHandle,
        >,
    ) -> Result<ExecutionContinuation<Cfg>, AkitaError> {
        let SuffixProverState {
            public,
            witness_handle,
            commitment_material,
        } = state;
        let resident = match self.adopted.take() {
            Some(imported) => {
                backend.release_witness_handle(witness_handle)?;
                drop(commitment_material);
                imported
            }
            None => ResidentSuccessor::new(self.current, witness_handle, commitment_material),
        };
        Ok((public, resident))
    }

    /// Route the committed successor to the next fold owner before producer sumchecks.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn handoff_successor<B>(
        &mut self,
        backend: &B,
        session: &B::ProofSessionHandle,
        level: usize,
        producer: &akita_params::CommittedGroupParams,
        relation: &akita_types::RingRelationInstance<Cfg::Field>,
        commitment: &ValidatedRecursiveWitnessCommitPlan,
        binding: &NextWitnessState<Cfg::Field>,
        witness: &B::WitnessHandle,
        material: &B::CommitmentMaterialHandle,
    ) -> Result<(), AkitaError>
    where
        Cfg::Field: CanonicalEncoding,
        B: ProverBackend<Cfg::Field, Cfg::ExtField> + 'static,
    {
        let requirements =
            FoldExecutionRequirements::new(self.request.schedule, self.request.layout, level + 1)?;
        let selected = self
            .policy
            .choose_backend(Some(self.current), &requirements)?;
        let destination = self.registry.slot(selected)?;
        if selected == self.current {
            destination.validate_prefixes(self.request, &requirements)?;
            backend.begin_fold(session, &requirements)?;
            return Ok(());
        }
        destination.begin_fold(self.request, level + 1)?;
        let layout = relation.segment_layout(producer, None)?;
        if layout.live_coeff_len() != commitment.logical_len() {
            return Err(AkitaError::Internal(
                "handoff layout differs from the committed witness".into(),
            ));
        }
        let plan = ValidatedSuccessorHandoffPlan {
            handoff: (u128::from(self.request.identity) << 64) | (level as u128),
            setup: self.request.setup,
            schedule: self.request.schedule,
            producer: level,
            layout,
            producer_parameters: producer,
            commitment,
            binding: match binding {
                NextWitnessState::OuterPayload(rows) => SuccessorPublicBinding::Outer(rows),
                NextWitnessState::TerminalInnerState(message) => {
                    SuccessorPublicBinding::Terminal(message.fields())
                }
            },
        };
        let bridge = self
            .registry
            .bridges
            .get(&(TypeId::of::<B>(), destination.backend_type_id()))
            .ok_or_else(|| {
                AkitaError::InvalidInput("no bridge registered for selected backend types".into())
            })?;
        let src = TypedExporter::<Cfg, B> {
            backend,
            session,
            witness,
            material,
        };
        self.adopted = Some(bridge.transfer(&src, destination, selected, &plan)?);
        Ok(())
    }
}
