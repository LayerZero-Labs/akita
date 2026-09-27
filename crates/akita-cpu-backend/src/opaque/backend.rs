//! CPU backend ownership, setup identity, and proof-session lifecycle.

use crate::arithmetic::CpuPreparedSetup;
use crate::opaque::owned_prefix::CachedSetupPrefix;
use crate::opaque::{BackendIdentity, CpuProofSessionHandle, OperationBinding};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_prover::backend::ProofContext;
use akita_serialization::Valid;
use akita_types::{AkitaExpandedSetup, FoldSchedule, OpeningClaimsLayout, SetupPrefixSlotId};
use core::marker::PhantomData;
use jolt_field::{CanonicalEncoding, Field};
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};

enum SetupPrefixCacheState<T> {
    Computing,
    Ready(Arc<T>),
    Failed,
}

struct SetupPrefixCacheCell<T> {
    state: Mutex<SetupPrefixCacheState<T>>,
    ready: Condvar,
}

impl<T> SetupPrefixCacheCell<T> {
    fn computing() -> Self {
        Self {
            state: Mutex::new(SetupPrefixCacheState::Computing),
            ready: Condvar::new(),
        }
    }
}

pub(super) struct SetupPrefixCache<T> {
    entries: Mutex<BTreeMap<SetupPrefixSlotId, Arc<SetupPrefixCacheCell<T>>>>,
}

impl<T> Default for SetupPrefixCache<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(BTreeMap::new()),
        }
    }
}

impl<T: Send + Sync> SetupPrefixCache<T> {
    pub(super) fn memoized<Derive>(
        &self,
        id: &SetupPrefixSlotId,
        derive: Derive,
    ) -> Result<Arc<T>, AkitaError>
    where
        Derive: FnOnce() -> Result<T, AkitaError>,
    {
        let mut derive = Some(derive);
        loop {
            let (cell, derive_here) = {
                let mut cache = self.entries.lock().map_err(|_| {
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
                        let mut state = cell.state.lock().map_err(|_| {
                            AkitaError::InvalidSetup("setup prefix slot lock poisoned".into())
                        })?;
                        *state = SetupPrefixCacheState::Ready(Arc::clone(&value));
                        cell.ready.notify_all();
                        return Ok(value);
                    }
                    Err(error) => {
                        let mut state = cell.state.lock().map_err(|_| {
                            AkitaError::InvalidSetup("setup prefix slot lock poisoned".into())
                        })?;
                        *state = SetupPrefixCacheState::Failed;
                        let mut cache = self.entries.lock().map_err(|_| {
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
                    SetupPrefixCacheState::Ready(value) => return Ok(Arc::clone(value)),
                    SetupPrefixCacheState::Failed => break,
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> Result<usize, AkitaError> {
        self.entries
            .lock()
            .map(|cache| cache.len())
            .map_err(|_| AkitaError::InvalidSetup("setup prefix cache lock poisoned".into()))
    }
}

/// Reusable owner of CPU setup resources and independent proof scopes.
///
/// The backend is typed by its base field `F` and extension field `E` only.
/// It stores no schedule catalog: commitment takes the producer family's
/// catalog as an argument, and proof admission takes the consumer's catalog,
/// so one backend can commit groups from several schedule families that share
/// `F` and `E`.
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
/// type Opening = <CpuBackend<Prime128OffsetA7F7, Prime128OffsetA7F7> as ProverHandleFamily<
///     Prime128OffsetA7F7,
///     Prime128OffsetA7F7,
/// >>::PreparedOpeningHandle;
///
/// fn require_clone<T: Clone>() {}
/// require_clone::<Opening>();
/// ```
pub struct CpuBackend<F: Field, E> {
    identity: Arc<BackendIdentity>,
    prepared: Option<CpuPreparedSetup<F>>,
    max_cached_ring_switch_elements: usize,
    /// Derived setup-prefix material, memoized for the life of this backend.
    ///
    /// A prefix commitment is a pure function of the owned setup and the slot
    /// id, both of which are fixed here, so deriving it more than once is
    /// wasted work.
    setup_prefix_cache: SetupPrefixCache<CachedSetupPrefix<F>>,
    extension: PhantomData<fn() -> E>,
}

impl<F: Field, E> core::fmt::Debug for CpuBackend<F, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuBackend")
            .field("backend_id", &self.identity.backend_id())
            .finish_non_exhaustive()
    }
}

impl<F: Field, E> CpuBackend<F, E> {
    /// Default maximum cached extent for a ring-switch NTT operation.
    pub const DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS: usize = 1 << 21;

    /// Own a setup.
    pub fn new(expanded: Arc<AkitaExpandedSetup<F>>) -> Result<Self, AkitaError>
    where
        F: CanonicalEncoding,
    {
        Self::with_ring_switch_cache_limit(expanded, Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS)
    }

    /// Create a CPU backend with a ring-switch cache limit.
    ///
    /// Zero streams every supported operation; `usize::MAX` retains all of
    /// them. Commitment scratch is sized automatically for each operation.
    pub fn with_ring_switch_cache_limit(
        expanded: Arc<AkitaExpandedSetup<F>>,
        max_cached_ring_switch_elements: usize,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalEncoding,
    {
        expanded
            .descriptor
            .check()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let digest = akita_types::setup_seed_digest(&expanded.descriptor.setup_seed)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        Ok(Self {
            identity: BackendIdentity::new(digest)?,
            prepared: Some(CpuPreparedSetup::new(expanded)),
            max_cached_ring_switch_elements,
            setup_prefix_cache: SetupPrefixCache::default(),
            extension: PhantomData,
        })
    }

    /// Return memoized setup-prefix material, deriving it once per slot.
    pub(super) fn memoized_setup_prefix<Derive>(
        &self,
        id: &SetupPrefixSlotId,
        derive: Derive,
    ) -> Result<Arc<CachedSetupPrefix<F>>, AkitaError>
    where
        Derive: FnOnce() -> Result<CachedSetupPrefix<F>, AkitaError>,
    {
        self.setup_prefix_cache.memoized(id, derive)
    }

    #[cfg(test)]
    pub(crate) fn setup_prefix_cache_len(&self) -> Result<usize, AkitaError> {
        self.setup_prefix_cache.len()
    }

    /// Require the proof schedule to be a row of `schedules` and validate the
    /// opening layout against that row.
    pub(crate) fn validate_proof_configuration<Cfg>(
        schedules: &TrustedScheduleCatalog<Cfg>,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<(), AkitaError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
        let row = schedules
            .catalog()
            .rows()
            .find(|row| row.schedule() == plan)
            .ok_or_else(|| {
                AkitaError::UnsupportedSchedule(
                    "proof schedule is absent from the trusted catalog".into(),
                )
            })?;
        row.validate_opening_layout(layout)
    }

    pub(crate) fn prepared(&self) -> Result<&CpuPreparedSetup<F>, AkitaError> {
        self.prepared
            .as_ref()
            .ok_or_else(|| AkitaError::InvalidSetup("test backend has no owned setup".into()))
    }

    pub(crate) fn owner(&self) -> &Arc<BackendIdentity> {
        &self.identity
    }
    pub(crate) fn owner_id(&self) -> u64 {
        self.identity.backend_id()
    }
    pub(crate) fn validate_context(
        &self,
        session: &CpuProofSessionHandle,
        context: &ProofContext,
    ) -> Result<(), AkitaError> {
        let lease = session.validate_owner(&self.identity)?;
        lease.validate_context(context)
    }
    pub(crate) fn binding(
        &self,
        session: &CpuProofSessionHandle,
        context: &ProofContext,
    ) -> Result<OperationBinding, AkitaError> {
        self.validate_context(session, context)?;
        Ok(OperationBinding::new(
            self.owner_id(),
            context.scope_id(),
            self.identity.setup_digest(),
            context.fold_level(),
            self.identity.next_operation_id()?,
            session.scope_lease(),
        )
        .with_group(context.group_index()))
    }
    pub(crate) fn validate_binding(&self, binding: &OperationBinding) -> Result<(), AkitaError> {
        self.validate_leased_binding(binding, binding.scope_lease())
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
        lease.validate_owner(&self.identity)?;
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
    pub fn trim_caches(&self) -> Result<usize, AkitaError>
    where
        F: CanonicalEncoding,
    {
        #[cfg(test)]
        if self.prepared.is_none() {
            return Ok(0);
        }
        self.prepared()?.drop_built_ntt_slots()
    }

    /// Largest ring-switch operation extent retained as an NTT cache.
    pub const fn max_cached_ring_switch_elements(&self) -> usize {
        self.max_cached_ring_switch_elements
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
impl<F: Field, E> CpuBackend<F, E> {
    /// Unit-test arithmetic route. It owns no setup, so it cannot import
    /// sources or admit proofs.
    pub(crate) fn for_arithmetic_tests() -> Self {
        Self::with_test_ring_switch_cache_limit(Self::DEFAULT_MAX_CACHED_RING_SWITCH_ELEMENTS)
            .expect("valid arithmetic fixture")
    }
    pub(crate) fn with_test_ring_switch_cache_limit(
        max_cached_ring_switch_elements: usize,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            identity: BackendIdentity::new([0; 32])?,
            prepared: None,
            max_cached_ring_switch_elements,
            setup_prefix_cache: SetupPrefixCache::default(),
            extension: PhantomData,
        })
    }
}
