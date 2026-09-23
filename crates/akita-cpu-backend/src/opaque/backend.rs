//! CPU backend ownership, setup identity, and proof-session lifecycle.

use crate::arithmetic::CpuPreparedSetup;
use crate::opaque::{BackendIdentity, OperationBinding};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_prover::backend::ProofContext;
use akita_serialization::Valid;
use akita_types::{AkitaExpandedSetup, FoldSchedule, OpeningClaimsLayout, SetupPrefixSlotId};
use jolt_field::{CanonicalEncoding, Field};
use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};

type SetupPrefixCacheValue = Arc<dyn Any + Send + Sync>;

enum SetupPrefixCacheState {
    Computing,
    Ready(SetupPrefixCacheValue),
    Failed,
}

struct SetupPrefixCacheCell {
    state: Mutex<SetupPrefixCacheState>,
    ready: Condvar,
}

impl SetupPrefixCacheCell {
    fn computing() -> Self {
        Self {
            state: Mutex::new(SetupPrefixCacheState::Computing),
            ready: Condvar::new(),
        }
    }
}

/// Reusable owner of CPU setup resources and independent proof scopes.
///
/// Prepared-opening handles are linear protocol capabilities and cannot be
/// cloned, even though their immutable backing allocations may be shared
/// internally.
///
/// ```compile_fail
/// use akita_cpu_backend::CpuBackend;
/// use akita_prover::ProverHandleFamily;
/// use jolt_field::Prime128OffsetA7F7;
///
/// type Opening = <CpuBackend as ProverHandleFamily<
///     Prime128OffsetA7F7,
///     Prime128OffsetA7F7,
/// >>::PreparedOpeningHandle;
///
/// fn require_clone<T: Clone>() {}
/// require_clone::<Opening>();
/// ```
pub struct CpuBackend {
    identity: Arc<BackendIdentity>,
    prepared: Box<dyn CpuSetupResources>,
    configuration: TypeId,
    schedules: Box<dyn CpuConfiguration>,
    max_cached_ring_switch_elements: usize,
    commit_scratch_bytes_per_worker: usize,
    /// Derived setup-prefix material, memoized for the life of this backend.
    ///
    /// A prefix commitment is a pure function of the owned setup and the slot
    /// id, both of which are fixed here, so deriving it more than once is
    /// wasted work. Values are type-erased because the backend is erased over
    /// its field; `validate_config` has already pinned the field before any
    /// lookup, so the downcast is exact.
    setup_prefix_cache: Mutex<BTreeMap<SetupPrefixSlotId, Arc<SetupPrefixCacheCell>>>,
}

trait CpuSetupResources: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn trim_caches(&self) -> Result<usize, AkitaError>;
}

trait CpuConfiguration: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn validate_extension(&self, extension: TypeId) -> Result<(), AkitaError>;
    fn validate_schedule(
        &self,
        extension: TypeId,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<(), AkitaError>;
}

impl<Cfg: CommitmentConfig> CpuConfiguration for TrustedScheduleCatalog<Cfg> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn validate_extension(&self, extension: TypeId) -> Result<(), AkitaError> {
        if extension != TypeId::of::<Cfg::ExtField>() {
            return Err(AkitaError::InvalidInput(
                "proof extension field differs from backend configuration".into(),
            ));
        }
        Ok(())
    }

    fn validate_schedule(
        &self,
        extension: TypeId,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        self.validate_extension(extension)?;
        let row = self
            .catalog()
            .rows()
            .find(|row| row.schedule() == plan)
            .ok_or_else(|| {
                AkitaError::UnsupportedSchedule(
                    "proof schedule is absent from the backend's trusted catalog".into(),
                )
            })?;
        row.validate_opening_layout(layout)
    }
}

impl<F: Field + CanonicalEncoding> CpuSetupResources for CpuPreparedSetup<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn trim_caches(&self) -> Result<usize, AkitaError> {
        self.drop_built_ntt_slots()
    }
}

impl core::fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuBackend")
            .field("backend_id", &self.identity.backend_id())
            .finish_non_exhaustive()
    }
}

impl CpuBackend {
    pub(crate) fn validate_extension<E: 'static>(&self) -> Result<(), AkitaError> {
        self.schedules.validate_extension(TypeId::of::<E>())
    }

    /// Default maximum cached extent for a ring-switch NTT operation.
    pub const DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS: usize = 1 << 21;

    /// Default temporary sparse commitment memory per worker.
    pub const DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER: usize = 8 << 20;

    /// Own a setup and its immutable trusted configuration.
    pub fn new<Cfg: CommitmentConfig>(
        expanded: Arc<AkitaExpandedSetup<Cfg::Field>>,
        schedules: &TrustedScheduleCatalog<Cfg>,
    ) -> Result<Self, AkitaError> {
        Self::with_resource_limits::<Cfg>(
            expanded,
            schedules,
            Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS,
            Self::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
        )
    }

    /// Create a CPU backend with explicit resource limits.
    pub fn with_resource_limits<Cfg: CommitmentConfig>(
        expanded: Arc<AkitaExpandedSetup<Cfg::Field>>,
        schedules: &TrustedScheduleCatalog<Cfg>,
        max_cached_ring_switch_elements: usize,
        commit_scratch_bytes_per_worker: usize,
    ) -> Result<Self, akita_error::AkitaError> {
        if commit_scratch_bytes_per_worker == 0 {
            return Err(akita_error::AkitaError::InvalidSetup(
                "CPU commitment scratch bytes per worker must be nonzero".into(),
            ));
        }
        expanded
            .descriptor
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let digest = akita_types::setup_seed_digest(&expanded.descriptor.setup_seed)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        Ok(Self {
            identity: BackendIdentity::new(digest)?,
            prepared: Box::new(CpuPreparedSetup::new(expanded)),
            configuration: TypeId::of::<Cfg>(),
            schedules: Box::new(schedules.clone()),
            max_cached_ring_switch_elements,
            commit_scratch_bytes_per_worker,
            setup_prefix_cache: Mutex::new(BTreeMap::new()),
        })
    }

    /// Return memoized setup-prefix material, deriving it once per slot.
    pub(crate) fn memoized_setup_prefix<T, Derive>(
        &self,
        id: &SetupPrefixSlotId,
        derive: Derive,
    ) -> Result<Arc<T>, AkitaError>
    where
        T: Any + Send + Sync,
        Derive: FnOnce() -> Result<T, AkitaError>,
    {
        let mut derive = Some(derive);
        loop {
            let (cell, derive_here) = {
                let mut cache = self.setup_prefix_cache.lock().map_err(|_| {
                    AkitaError::InvalidSetup("setup prefix cache lock poisoned".into())
                })?;
                match cache.get(id) {
                    Some(cell) => (Arc::clone(cell), false),
                    None => {
                        let cell = Arc::new(SetupPrefixCacheCell::computing());
                        cache.insert(id.clone(), Arc::clone(&cell));
                        (cell, true)
                    }
                }
            };

            if derive_here {
                let derive = derive.take().ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "setup prefix cache derivation was already consumed".into(),
                    )
                })?;
                match derive() {
                    Ok(value) => {
                        let value = Arc::new(value);
                        let erased: SetupPrefixCacheValue = value.clone();
                        let mut state = cell.state.lock().map_err(|_| {
                            AkitaError::InvalidSetup("setup prefix slot lock poisoned".into())
                        })?;
                        *state = SetupPrefixCacheState::Ready(erased);
                        cell.ready.notify_all();
                        return Ok(value);
                    }
                    Err(error) => {
                        let mut state = cell.state.lock().map_err(|_| {
                            AkitaError::InvalidSetup("setup prefix slot lock poisoned".into())
                        })?;
                        *state = SetupPrefixCacheState::Failed;
                        let mut cache = self.setup_prefix_cache.lock().map_err(|_| {
                            AkitaError::InvalidSetup("setup prefix cache lock poisoned".into())
                        })?;
                        if cache
                            .get(id)
                            .is_some_and(|current| Arc::ptr_eq(current, &cell))
                        {
                            cache.remove(id);
                        }
                        cell.ready.notify_all();
                        return Err(error);
                    }
                }
            }

            let mut state = cell
                .state
                .lock()
                .map_err(|_| AkitaError::InvalidSetup("setup prefix slot lock poisoned".into()))?;
            loop {
                match &*state {
                    SetupPrefixCacheState::Computing => {
                        state = cell.ready.wait(state).map_err(|_| {
                            AkitaError::InvalidSetup("setup prefix slot lock poisoned".into())
                        })?;
                    }
                    SetupPrefixCacheState::Ready(value) => {
                        return Arc::clone(value).downcast::<T>().map_err(|_| {
                            AkitaError::InvalidInput(
                                "setup prefix material belongs to another field".into(),
                            )
                        });
                    }
                    SetupPrefixCacheState::Failed => break,
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn setup_prefix_cache_len(&self) -> Result<usize, AkitaError> {
        self.setup_prefix_cache
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| AkitaError::InvalidSetup("setup prefix cache lock poisoned".into()))
    }

    pub(crate) fn validate_config<Cfg: CommitmentConfig>(&self) -> Result<(), AkitaError> {
        if self.configuration != TypeId::of::<Cfg>() {
            return Err(AkitaError::InvalidInput(
                "configuration belongs to another backend".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn schedules<Cfg: CommitmentConfig>(
        &self,
    ) -> Result<&TrustedScheduleCatalog<Cfg>, AkitaError> {
        self.validate_config::<Cfg>()?;
        self.schedules
            .as_any()
            .downcast_ref()
            .ok_or_else(|| AkitaError::InvalidSetup("backend configuration type mismatch".into()))
    }

    pub(crate) fn validate_proof_configuration<E: 'static>(
        &self,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        self.schedules
            .validate_schedule(TypeId::of::<E>(), plan, layout)
    }

    pub(crate) fn prepared<F: Field + CanonicalEncoding>(
        &self,
    ) -> Result<&CpuPreparedSetup<F>, AkitaError> {
        self.prepared.as_any().downcast_ref().ok_or_else(|| {
            AkitaError::InvalidInput("field belongs to another backend configuration".into())
        })
    }

    pub(crate) fn owner(&self) -> &Arc<BackendIdentity> {
        &self.identity
    }
    pub(crate) fn owner_id(&self) -> u64 {
        self.identity.backend_id()
    }
    pub(crate) fn validate_context(&self, context: &ProofContext) -> Result<(), AkitaError> {
        self.identity.validate_context(context)
    }
    pub(crate) fn binding(&self, context: &ProofContext) -> Result<OperationBinding, AkitaError> {
        self.validate_context(context)?;
        Ok(OperationBinding::new(
            self.owner_id(),
            context.scope_id(),
            self.identity.setup_digest(),
            context.fold_level(),
            self.identity.next_operation_id()?,
        )
        .with_group(context.group_index()))
    }
    pub(crate) fn validate_binding(&self, binding: &OperationBinding) -> Result<(), AkitaError> {
        self.binding_lease(binding).map(drop)
    }
    pub(crate) fn binding_lease(
        &self,
        binding: &OperationBinding,
    ) -> Result<crate::opaque::ScopeLease, AkitaError> {
        let lease = self.identity.scope_lease(binding.scope_id())?;
        self.validate_leased_binding(binding, &lease)?;
        Ok(lease)
    }
    pub(crate) fn validate_leased_binding(
        &self,
        binding: &OperationBinding,
        lease: &crate::opaque::ScopeLease,
    ) -> Result<(), AkitaError> {
        binding.validate_owner(
            self.owner_id(),
            self.identity.setup_digest(),
            binding.fold_level(),
        )?;
        lease.validate(
            binding.scope_id(),
            binding.fold_level(),
            binding.group_index(),
        )
    }
    pub(crate) fn next_binding(
        &self,
        parent: OperationBinding,
    ) -> Result<OperationBinding, AkitaError> {
        self.validate_binding(&parent)?;
        Ok(parent.for_operation(self.identity.next_operation_id()?))
    }
    pub(crate) fn next_level_binding(
        &self,
        parent: OperationBinding,
    ) -> Result<OperationBinding, AkitaError> {
        self.validate_binding(&parent)?;
        let level = parent
            .fold_level()
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidInput("fold level overflow".into()))?;
        let next = parent
            .for_level_operation(level, self.identity.next_operation_id()?)
            .with_group(None);
        self.validate_binding(&next)?;
        Ok(next)
    }
    /// Release idle shared setup transforms. Active operations retain their resources.
    pub fn trim_caches(&self) -> Result<usize, AkitaError> {
        self.prepared.trim_caches()
    }

    /// Largest ring-switch operation extent retained as an NTT cache.
    pub const fn max_cached_ring_switch_elements(&self) -> usize {
        self.max_cached_ring_switch_elements
    }

    /// Temporary sparse commitment memory allowed per worker.
    pub const fn commit_scratch_bytes_per_worker(&self) -> usize {
        self.commit_scratch_bytes_per_worker
    }

    #[inline]
    pub(crate) fn ntt_operation_uses_cache(
        &self,
        cluster: crate::arithmetic::requirements::NttOperationCluster,
        num_ring_elements: usize,
    ) -> bool {
        let cached = cluster != crate::arithmetic::requirements::NttOperationCluster::RingSwitch
            || num_ring_elements <= self.max_cached_ring_switch_elements;
        tracing::debug!(
            ?cluster,
            num_ring_elements,
            max_cached_ring_switch_elements = self.max_cached_ring_switch_elements,
            cached,
            "CPU NTT execution policy"
        );
        cached
    }
}

#[cfg(test)]
struct ArithmeticTestResources;
#[cfg(test)]
impl CpuConfiguration for () {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn validate_extension(&self, _: TypeId) -> Result<(), AkitaError> {
        Ok(())
    }
    fn validate_schedule(
        &self,
        _: TypeId,
        _: &FoldSchedule,
        _: &OpeningClaimsLayout,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidSetup(
            "arithmetic fixture has no trusted proof configuration".into(),
        ))
    }
}
#[cfg(test)]
impl CpuSetupResources for ArithmeticTestResources {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn trim_caches(&self) -> Result<usize, AkitaError> {
        Ok(0)
    }
}
#[cfg(test)]
impl CpuBackend {
    /// Unit-test arithmetic route. It cannot import sources or admit proofs.
    pub(crate) fn for_arithmetic_tests() -> Self {
        Self::with_test_resource_limits(
            Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS,
            Self::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
        )
        .expect("valid arithmetic fixture")
    }
    pub(crate) fn with_test_resource_limits(
        max_cached_ring_switch_elements: usize,
        commit_scratch_bytes_per_worker: usize,
    ) -> Result<Self, AkitaError> {
        if commit_scratch_bytes_per_worker == 0 {
            return Err(AkitaError::InvalidSetup("zero test scratch budget".into()));
        }
        Ok(Self {
            identity: BackendIdentity::new([0; 32])?,
            prepared: Box::new(ArithmeticTestResources),
            configuration: TypeId::of::<()>(),
            schedules: Box::new(()),
            max_cached_ring_switch_elements,
            commit_scratch_bytes_per_worker,
            setup_prefix_cache: Mutex::new(BTreeMap::new()),
        })
    }
    pub(crate) fn for_test_setup<F: Field + CanonicalEncoding>(
        expanded: Arc<AkitaExpandedSetup<F>>,
    ) -> Result<Self, AkitaError> {
        expanded
            .descriptor
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let digest = akita_types::setup_seed_digest(&expanded.descriptor.setup_seed)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        Ok(Self {
            identity: BackendIdentity::new(digest)?,
            prepared: Box::new(CpuPreparedSetup::new(expanded)),
            configuration: TypeId::of::<()>(),
            schedules: Box::new(()),
            max_cached_ring_switch_elements: Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS,
            commit_scratch_bytes_per_worker: Self::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
            setup_prefix_cache: Mutex::new(BTreeMap::new()),
        })
    }
}
