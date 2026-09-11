use crate::compute::CommitInnerPlan;
use akita_error::AkitaError;
use akita_types::{AkitaSetupDescriptor, CompressionChainPlan, RingRelationMode};
use std::any::Any;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_OWNER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_INVOCATION_ID: AtomicU64 = AtomicU64::new(1);

/// Semantic marker for a resident inner commitment image.
pub enum InnerImage {}

/// Semantic marker for retained commitment-compression state.
pub enum CompressionState {}

/// Immutable setup and request facts attached to resident stage state.
#[derive(Clone, PartialEq, Eq)]
pub struct CommitmentStateBinding {
    invocation: u64,
    setup: AkitaSetupDescriptor,
    inner_plan: CommitInnerPlan,
    source_count: usize,
    relation_mode: Option<RingRelationMode>,
    compression_plan: Option<CompressionChainPlan>,
}

impl CommitmentStateBinding {
    /// Construct a checked state binding for one commitment request.
    pub fn new(
        setup: AkitaSetupDescriptor,
        inner_plan: CommitInnerPlan,
        source_count: usize,
        relation_mode: Option<RingRelationMode>,
        compression_plan: Option<CompressionChainPlan>,
    ) -> Result<Self, AkitaError> {
        if source_count == 0
            || inner_plan.ring_dimension == 0
            || !inner_plan.ring_dimension.is_power_of_two()
            || inner_plan.num_live_blocks == 0
            || inner_plan.n_a == 0
            || inner_plan.num_positions_per_block == 0
            || inner_plan.num_digits_inner == 0
            || inner_plan.log_basis_inner == 0
        {
            return Err(AkitaError::InvalidInput(
                "commitment state binding has invalid source or inner-stage geometry".into(),
            ));
        }
        Ok(Self {
            invocation: NEXT_INVOCATION_ID.fetch_add(1, Ordering::Relaxed),
            setup,
            inner_plan,
            source_count,
            relation_mode,
            compression_plan,
        })
    }

    pub const fn setup(&self) -> &AkitaSetupDescriptor {
        &self.setup
    }

    pub const fn inner_plan(&self) -> &CommitInnerPlan {
        &self.inner_plan
    }

    pub const fn source_count(&self) -> usize {
        self.source_count
    }

    pub const fn relation_mode(&self) -> Option<RingRelationMode> {
        self.relation_mode
    }

    pub const fn compression_plan(&self) -> Option<&CompressionChainPlan> {
        self.compression_plan.as_ref()
    }
}

impl std::fmt::Debug for CommitmentStateBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommitmentStateBinding")
            .field("invocation", &"opaque")
            .field("inner_plan", &self.inner_plan)
            .field("source_count", &self.source_count)
            .field("relation_mode", &self.relation_mode)
            .field("compression_plan", &self.compression_plan)
            .finish_non_exhaustive()
    }
}

struct OwnedBackendState {
    owner: u64,
    binding: CommitmentStateBinding,
    retained_bytes: usize,
    value: Box<dyn Any + Send + Sync>,
}

/// Checked state that directly owns a backend value.
///
/// The erased value may be CPU witness data, a device allocation lease, or a
/// remote lease. Dropping the last reference drops the value and runs its
/// ordinary backend-defined cleanup.
pub struct BackendStateRef<K> {
    state: Arc<OwnedBackendState>,
    marker: PhantomData<fn() -> K>,
}

impl<K> Clone for BackendStateRef<K> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            marker: PhantomData,
        }
    }
}

impl<K> std::fmt::Debug for BackendStateRef<K> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("BackendStateRef(..)")
    }
}

impl<K> BackendStateRef<K> {
    pub fn binding(&self) -> &CommitmentStateBinding {
        &self.state.binding
    }

    pub fn retained_bytes(&self) -> usize {
        self.state.retained_bytes
    }
}

/// Identity and construction authority for one backend state representation.
///
/// A backend keeps this beside its operation. Akita does not allocate a slot
/// or keep a second table containing the backend value.
pub struct StateOwnerCapability<K> {
    owner: u64,
    marker: PhantomData<fn() -> K>,
}

impl<K> Clone for StateOwnerCapability<K> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner,
            marker: PhantomData,
        }
    }
}

impl<K> std::fmt::Debug for StateOwnerCapability<K> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("StateOwnerCapability(..)")
    }
}

impl<K> StateOwnerCapability<K> {
    pub fn new() -> Self {
        Self {
            owner: NEXT_OWNER_ID.fetch_add(1, Ordering::Relaxed),
            marker: PhantomData,
        }
    }

    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        self.owner == other.owner
    }

    pub(crate) fn owns(&self, state: &BackendStateRef<K>) -> bool {
        self.owner == state.state.owner
    }

    /// Bind and directly own one concrete backend value.
    pub fn bind<T>(
        &self,
        binding: CommitmentStateBinding,
        retained_bytes: usize,
        value: T,
    ) -> BackendStateRef<K>
    where
        T: Any + Send + Sync,
    {
        BackendStateRef {
            state: Arc::new(OwnedBackendState {
                owner: self.owner,
                binding,
                retained_bytes,
                value: Box::new(value),
            }),
            marker: PhantomData,
        }
    }

    /// Borrow this owner's concrete value after checking ownership and type.
    pub fn value<'a, T>(&self, state: &'a BackendStateRef<K>) -> Result<&'a T, AkitaError>
    where
        T: Any + Send + Sync,
    {
        if state.state.owner != self.owner {
            return Err(AkitaError::InvalidInput(
                "backend state belongs to a different owner".into(),
            ));
        }
        state.state.value.downcast_ref::<T>().ok_or_else(|| {
            AkitaError::InvalidInput("backend state has the wrong concrete representation".into())
        })
    }

    /// Consume a uniquely held state value without copying it.
    ///
    /// Shared state is returned intact so an exporter can use its borrowed
    /// fallback. Ownership and concrete type are always checked first.
    pub fn try_unwrap<T>(
        &self,
        state: BackendStateRef<K>,
    ) -> Result<Result<T, BackendStateRef<K>>, AkitaError>
    where
        T: Any + Send + Sync,
    {
        if state.state.owner != self.owner {
            return Err(AkitaError::InvalidInput(
                "backend state belongs to a different owner".into(),
            ));
        }
        match Arc::try_unwrap(state.state) {
            Ok(owned) => owned
                .value
                .downcast::<T>()
                .map(|value| Ok(*value))
                .map_err(|_| {
                    AkitaError::InvalidInput(
                        "backend state has the wrong concrete representation".into(),
                    )
                }),
            Err(shared) => Ok(Err(BackendStateRef {
                state: shared,
                marker: PhantomData,
            })),
        }
    }
}

impl<K> Default for StateOwnerCapability<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::AkitaSetupSeed;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn binding() -> CommitmentStateBinding {
        CommitmentStateBinding::new(
            AkitaSetupDescriptor {
                max_num_vars: 8,
                max_num_batched_polys: 1,
                num_field_elements: 64,
                setup_seed: AkitaSetupSeed::shake256_paged_v1([7; 32]),
            },
            CommitInnerPlan {
                ring_dimension: 64,
                num_live_blocks: 1,
                n_a: 1,
                num_positions_per_block: 1,
                num_digits_inner: 1,
                log_basis_inner: 1,
            },
            1,
            None,
            None,
        )
        .unwrap()
    }

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn owned_state_drops_its_backend_value_after_the_last_lease() {
        let owner = StateOwnerCapability::<InnerImage>::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let state = owner.bind(binding(), 123, DropCounter(drops.clone()));
        let cloned = state.clone();
        assert_eq!(state.retained_bytes(), 123);
        assert!(owner.value::<DropCounter>(&state).is_ok());
        drop(state);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(cloned);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn foreign_owner_and_wrong_value_type_are_rejected() {
        let owner = StateOwnerCapability::<InnerImage>::new();
        let foreign = StateOwnerCapability::<InnerImage>::new();
        let state = owner.bind(binding(), 0, 7_u64);
        assert!(foreign.value::<u64>(&state).is_err());
        assert!(owner.value::<u32>(&state).is_err());
    }

    #[test]
    fn consuming_extraction_moves_unique_state_and_preserves_shared_state() {
        let owner = StateOwnerCapability::<InnerImage>::new();
        let unique = owner.bind(binding(), 0, vec![1_u64, 2, 3]);
        assert_eq!(
            owner.try_unwrap::<Vec<u64>>(unique).unwrap().unwrap(),
            [1, 2, 3]
        );

        let shared = owner.bind(binding(), 0, vec![4_u64, 5]);
        let lease = shared.clone();
        let returned = owner
            .try_unwrap::<Vec<u64>>(shared)
            .unwrap()
            .expect_err("shared state must be returned intact");
        assert_eq!(owner.value::<Vec<u64>>(&returned).unwrap(), &[4, 5]);
        assert_eq!(owner.value::<Vec<u64>>(&lease).unwrap(), &[4, 5]);
    }
}
