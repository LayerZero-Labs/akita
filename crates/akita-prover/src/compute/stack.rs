//! Prover compute stack: per-fold stack selection and per-cluster routing.
//!
//! Two orthogonal axes:
//!
//! 1. **Per-fold stack** ([`LevelProveStacks`]): which [`ProverComputeStack`]
//!    runs fold `level`. `batched_prove` / `prove` take `&impl LevelProveStacks`;
//!    passing `&stack` is the degenerate case (same stack at every level).
//!    Tiered hardware provers use [`TieredProveStacks`] or a custom impl.
//!
//! 2. **Per-cluster context** (inside one stack): commit, opening, tensor, and
//!    ring-switch each hold a validated [`OperationCtx`]. Protocol internals route
//!    kernels to the matching cluster (for example `commit_w` uses
//!    `stack.commit()`, `ring_switch_build_w` uses `stack.ring_switch()`).
//!
//! Commit entry points call `stack.commit()` and `stack.tensor()` directly.
//! Prove entry points call `stacks.prove_stack_at_level(level)` once per fold,
//! then dispatch through the cluster accessors on that stack.

use crate::compute::backend::{ComputeBackendSetup, NttCacheOwnerId};
use crate::compute::requirements::{
    NttExecutionRequirements, NttOperationCluster, RoutedNttRequirement,
};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::AkitaExpandedSetup;
use std::marker::PhantomData;

/// A single operation context: a backend plus its validated prepared setup.
///
/// Construction validates the prepared setup against explicit expanded-setup
/// metadata, so a kernel may assume its context was validated. The fields are
/// private to keep that invariant: an `OperationCtx` cannot exist without going
/// through a validating constructor.
pub struct OperationCtx<'a, F, B>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
    backend: &'a B,
    prepared: &'a B::PreparedSetup,
    _field: PhantomData<fn() -> F>,
}

impl<'a, F, B> OperationCtx<'a, F, B>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
    /// Build an operation context, validating `prepared` against `expanded`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] (via
    /// [`ComputeBackendSetup::validate_prepared_setup`]) when `prepared` was not
    /// built from `expanded`.
    pub fn new(
        backend: &'a B,
        prepared: &'a B::PreparedSetup,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        backend.validate_prepared_setup(prepared, expanded)?;
        Ok(Self {
            backend,
            prepared,
            _field: PhantomData,
        })
    }

    /// Borrowed backend for this operation cluster.
    pub fn backend(&self) -> &'a B {
        self.backend
    }

    /// Borrowed prepared setup for this operation cluster.
    pub fn prepared(&self) -> &'a B::PreparedSetup {
        self.prepared
    }

    fn ensure_ntt(&self, requirement: RoutedNttRequirement) -> Result<(), AkitaError> {
        self.backend.ensure_ntt_slot(self.prepared, requirement.key)
    }

    fn retained_ntt_owner(
        &self,
        requirement: RoutedNttRequirement,
    ) -> Result<Option<NttCacheOwnerId>, AkitaError> {
        if !self
            .backend
            .ntt_requirement_is_cached(self.prepared, requirement)?
        {
            return Ok(None);
        }
        Ok(Some(self.backend.ntt_cache_owner_id(self.prepared)))
    }

    fn planned_ntt(
        &self,
        requirement: RoutedNttRequirement,
    ) -> Result<Option<(NttCacheOwnerId, usize)>, AkitaError> {
        if !self
            .backend
            .ntt_requirement_is_cached(self.prepared, requirement)?
        {
            return Ok(None);
        }
        Ok(Some((
            self.backend.ntt_cache_owner_id(self.prepared),
            self.backend
                .planned_ntt_cache_entry_bytes(self.prepared, requirement.key)?,
        )))
    }

    fn release_ntt_if_new(
        &self,
        released_owners: &mut Vec<NttCacheOwnerId>,
    ) -> Result<usize, AkitaError> {
        let owner = self.backend.ntt_cache_owner_id(self.prepared);
        if released_owners.contains(&owner) {
            return Ok(0);
        }
        let freed = self.backend.release_built_ntt_slots(self.prepared)?;
        released_owners.push(owner);
        Ok(freed)
    }
}

/// One fold-level prover stack with four operation clusters.
///
/// A single proof may use different stacks at different fold levels via
/// [`LevelProveStacks`]. Within one stack, each cluster (commit / opening /
/// tensor / ring-switch) may still use a different backend and prepared setup.
/// [`UniformProverStack`] is the degenerate case where all four clusters share
/// one backend ([`ProverComputeStack::uniform`]).
pub struct ProverComputeStack<'a, F, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    commit: OperationCtx<'a, F, C>,
    opening: OperationCtx<'a, F, O>,
    tensor: OperationCtx<'a, F, T>,
    ring_switch: OperationCtx<'a, F, R>,
}

impl<'a, F, C, O, T, R> ProverComputeStack<'a, F, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    /// Drop releasable NTT slots across all four clusters and return the total
    /// number of bytes freed. Physically shared cache owners are released once.
    ///
    /// Slots rebuild on next use. Callers should release only at a lifecycle
    /// boundary they own; proof execution retains shared prepared state by
    /// default. Active readers remain valid. Release does not cancel cache
    /// construction already in progress, so callers that require an empty
    /// cache must prevent concurrent construction at this boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when a backend cannot update its cache state or the
    /// released-byte total overflows.
    pub fn release_built_ntt_slots(&self) -> Result<usize, AkitaError> {
        let mut released_owners = Vec::with_capacity(4);
        let mut freed = 0usize;
        for released in [
            self.commit.release_ntt_if_new(&mut released_owners)?,
            self.opening.release_ntt_if_new(&mut released_owners)?,
            self.tensor.release_ntt_if_new(&mut released_owners)?,
            self.ring_switch.release_ntt_if_new(&mut released_owners)?,
        ] {
            freed = freed.checked_add(released).ok_or_else(|| {
                AkitaError::InvalidSetup("released NTT cache bytes overflow".into())
            })?;
        }
        Ok(freed)
    }

    /// Build a heterogeneous prover stack, validating every contained context
    /// against the same expanded setup before any transcript work.
    ///
    /// # Errors
    ///
    /// Returns an error if any cluster's prepared setup fails validation.
    pub fn new(
        commit: (&'a C, &'a C::PreparedSetup),
        opening: (&'a O, &'a O::PreparedSetup),
        tensor: (&'a T, &'a T::PreparedSetup),
        ring_switch: (&'a R, &'a R::PreparedSetup),
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            commit: OperationCtx::new(commit.0, commit.1, expanded)?,
            opening: OperationCtx::new(opening.0, opening.1, expanded)?,
            tensor: OperationCtx::new(tensor.0, tensor.1, expanded)?,
            ring_switch: OperationCtx::new(ring_switch.0, ring_switch.1, expanded)?,
        })
    }

    /// Commit operation context.
    pub fn commit(&self) -> &OperationCtx<'a, F, C> {
        &self.commit
    }

    /// Opening / decompose-fold operation context.
    pub fn opening(&self) -> &OperationCtx<'a, F, O> {
        &self.opening
    }

    /// Tensor projection operation context.
    pub fn tensor(&self) -> &OperationCtx<'a, F, T> {
        &self.tensor
    }

    /// Ring-switch operation context.
    pub fn ring_switch(&self) -> &OperationCtx<'a, F, R> {
        &self.ring_switch
    }

    fn prewarm_requirement(&self, requirement: RoutedNttRequirement) -> Result<(), AkitaError> {
        match requirement.cluster {
            NttOperationCluster::Commit => self.commit.ensure_ntt(requirement),
            NttOperationCluster::Opening => self.opening.ensure_ntt(requirement),
            NttOperationCluster::Tensor => self.tensor.ensure_ntt(requirement),
            NttOperationCluster::RingSwitch => self.ring_switch.ensure_ntt(requirement),
        }
    }

    fn retained_requirement_owner(
        &self,
        requirement: RoutedNttRequirement,
    ) -> Result<Option<NttCacheOwnerId>, AkitaError> {
        match requirement.cluster {
            NttOperationCluster::Commit => self.commit.retained_ntt_owner(requirement),
            NttOperationCluster::Opening => self.opening.retained_ntt_owner(requirement),
            NttOperationCluster::Tensor => self.tensor.retained_ntt_owner(requirement),
            NttOperationCluster::RingSwitch => self.ring_switch.retained_ntt_owner(requirement),
        }
    }

    fn planned_requirement(
        &self,
        requirement: RoutedNttRequirement,
    ) -> Result<Option<(NttCacheOwnerId, usize)>, AkitaError> {
        match requirement.cluster {
            NttOperationCluster::Commit => self.commit.planned_ntt(requirement),
            NttOperationCluster::Opening => self.opening.planned_ntt(requirement),
            NttOperationCluster::Tensor => self.tensor.planned_ntt(requirement),
            NttOperationCluster::RingSwitch => self.ring_switch.planned_ntt(requirement),
        }
    }
}

/// Single-backend degenerate [`ProverComputeStack`] (all four clusters share `B`).
pub type UniformProverStack<'a, F, B> = ProverComputeStack<'a, F, B, B, B, B>;

/// Per-fold selection of a [`ProverComputeStack`] during proving.
///
/// `prove_fold` and suffix preparation call `prove_stack_at_level(level)` before
/// routing work to commit / opening / tensor / ring-switch clusters on that
/// stack.
///
/// **Uniform case:** [`UniformProverStack`] fixes all four associated cluster
/// types to one backend `B` (what `batched_prove(..., &stack, ...)` uses today).
///
/// **Heterogeneous case:** each associated type may differ; protocol internals
/// route kernels through the matching cluster on the returned stack.
///
/// **Tiered case:** [`TieredProveStacks`] maps fold ranges to distinct stacks
/// (for example multi-GPU folds 0–1, single-GPU 2–3, CPU thereafter). Every
/// tier must share the same `(Commit, Opening, Tensor, RingSwitch)` type tuple;
/// tiers differ only in backend handles and prepared setups.
///
/// **Facade alternative:** a single backend type that dispatches on `level`
/// internally also works; this trait is not required when the backend owns tier
/// selection.
pub trait LevelProveStacks<'a, F>
where
    F: FieldCore + CanonicalField,
{
    /// Commit cluster backend for stacks returned by this selector.
    type Commit: ComputeBackendSetup<F>;
    /// Opening cluster backend for stacks returned by this selector.
    type Opening: ComputeBackendSetup<F>;
    /// Tensor cluster backend for stacks returned by this selector.
    type Tensor: ComputeBackendSetup<F>;
    /// Ring-switch cluster backend for stacks returned by this selector.
    type RingSwitch: ComputeBackendSetup<F>;

    /// Stack whose operation clusters should execute fold `level`.
    fn prove_stack_at_level(
        &self,
        level: usize,
    ) -> &ProverComputeStack<'a, F, Self::Commit, Self::Opening, Self::Tensor, Self::RingSwitch>;

    /// Optional lifecycle hook after the root fold and before the recursive
    /// suffix. The default retains every prepared NTT cache.
    ///
    /// A downstream stack selector may override this when it owns an isolated
    /// root cache and intentionally wants to release it at this boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when the selected lifecycle action fails.
    fn after_root_fold(&self) -> Result<(), AkitaError> {
        Ok(())
    }
}

/// Explicit stack policy that releases every NTT cache owner reachable from
/// the root stack after the root fold.
///
/// This restores the memory-minimizing root/suffix boundary for applications
/// that choose it. Do not use it when the root stack shares prepared cache
/// owners with concurrent work whose warm state must be retained. It is also
/// the caller's responsibility to prevent concurrent cache construction if the
/// suffix must begin with an empty cache.
pub struct ReleaseRootNttAfterFold<S> {
    stacks: S,
}

impl<S> ReleaseRootNttAfterFold<S> {
    /// Wrap a stack selector with explicit root NTT release behavior.
    pub fn new(stacks: S) -> Self {
        Self { stacks }
    }

    /// Recover the wrapped stack selector.
    pub fn into_inner(self) -> S {
        self.stacks
    }
}

impl<'a, F, C, O, T, R, S> LevelProveStacks<'a, F> for ReleaseRootNttAfterFold<S>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F> + 'a,
    O: ComputeBackendSetup<F> + 'a,
    T: ComputeBackendSetup<F> + 'a,
    R: ComputeBackendSetup<F> + 'a,
    S: LevelProveStacks<'a, F, Commit = C, Opening = O, Tensor = T, RingSwitch = R>,
    C::PreparedSetup: 'a,
    O::PreparedSetup: 'a,
    T::PreparedSetup: 'a,
    R::PreparedSetup: 'a,
{
    type Commit = C;
    type Opening = O;
    type Tensor = T;
    type RingSwitch = R;

    fn prove_stack_at_level(&self, level: usize) -> &ProverComputeStack<'a, F, C, O, T, R> {
        self.stacks.prove_stack_at_level(level)
    }

    fn after_root_fold(&self) -> Result<(), AkitaError> {
        self.stacks
            .prove_stack_at_level(0)
            .release_built_ntt_slots()
            .map(|_| ())
    }
}

/// Prewarm the retained part of an exact execution plan.
///
/// Each routed backend applies the same cache-retention policy as its runtime
/// kernel. Streamed operations remain in the logical requirement plan but do
/// not allocate a prepared slot.
pub fn prewarm_ntt_requirements<'a, F, S>(
    stacks: &S,
    requirements: &NttExecutionRequirements,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + 'a,
    S: LevelProveStacks<'a, F> + ?Sized + 'a,
{
    let mut retained = Vec::<(NttCacheOwnerId, RoutedNttRequirement)>::new();
    for requirement in requirements.entries() {
        let stack = stacks.prove_stack_at_level(requirement.fold_level);
        let Some(owner) = stack.retained_requirement_owner(*requirement)? else {
            continue;
        };
        if let Some((_, existing)) = retained.iter_mut().find(|(existing_owner, existing)| {
            *existing_owner == owner
                && existing.key.ring_d == requirement.key.ring_d
                && existing.key.domain == requirement.key.domain
        }) {
            if requirement.key.num_ring_elements > existing.key.num_ring_elements {
                *existing = *requirement;
            }
        } else {
            retained.push((owner, *requirement));
        }
    }
    retained.sort_by_key(|(_, requirement)| std::cmp::Reverse(requirement.key.num_ring_elements));
    for (_, requirement) in retained {
        stacks
            .prove_stack_at_level(requirement.fold_level)
            .prewarm_requirement(requirement)?;
    }
    Ok(())
}

/// Planned cache state for one physical prepared owner after max-joining routes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedNttCacheOwnerMetric {
    /// Process-local physical cache identity. Never serialized or transcript-bound.
    pub owner_id: NttCacheOwnerId,
    /// Max-joined exact keys resident on this owner.
    pub keys: Vec<akita_types::NttCacheKey>,
    /// Backend-reported resident bytes for `keys`.
    pub cache_bytes: usize,
}

/// Report planned bytes after routing and physical prepared-owner aliasing.
pub fn planned_ntt_cache_metrics<'a, F, S>(
    stacks: &S,
    requirements: &NttExecutionRequirements,
) -> Result<Vec<PlannedNttCacheOwnerMetric>, AkitaError>
where
    F: FieldCore + CanonicalField + 'a,
    S: LevelProveStacks<'a, F> + ?Sized + 'a,
{
    let mut owners = Vec::<PlannedNttCacheOwnerMetric>::new();
    let mut entry_bytes = Vec::<Vec<usize>>::new();
    for requirement in requirements.entries() {
        let stack = stacks.prove_stack_at_level(requirement.fold_level);
        let Some((owner_id, bytes)) = stack.planned_requirement(*requirement)? else {
            continue;
        };
        let owner_index = owners
            .iter()
            .position(|owner| owner.owner_id == owner_id)
            .unwrap_or_else(|| {
                owners.push(PlannedNttCacheOwnerMetric {
                    owner_id,
                    keys: Vec::new(),
                    cache_bytes: 0,
                });
                entry_bytes.push(Vec::new());
                owners.len() - 1
            });
        let key_index = owners[owner_index].keys.iter().position(|key| {
            key.ring_d == requirement.key.ring_d && key.domain == requirement.key.domain
        });
        match key_index {
            Some(index) => {
                let current = owners[owner_index].keys[index];
                if requirement.key.num_ring_elements > current.num_ring_elements {
                    owners[owner_index].keys[index] = requirement.key;
                    entry_bytes[owner_index][index] = bytes;
                } else if requirement.key.num_ring_elements == current.num_ring_elements
                    && entry_bytes[owner_index][index] != bytes
                {
                    return Err(AkitaError::InvalidSetup(
                        "aliased NTT cache backends disagree on planned bytes".into(),
                    ));
                }
            }
            None => {
                owners[owner_index].keys.push(requirement.key);
                entry_bytes[owner_index].push(bytes);
            }
        }
    }
    for (owner, bytes) in owners.iter_mut().zip(entry_bytes) {
        owner.cache_bytes = bytes.into_iter().try_fold(0usize, |total, entry| {
            total
                .checked_add(entry)
                .ok_or_else(|| AkitaError::InvalidSetup("planned NTT bytes overflow".into()))
        })?;
        owner.keys.sort_by_key(|key| {
            let domain = match key.domain {
                akita_types::NttTransformDomain::Negacyclic => 0,
                akita_types::NttTransformDomain::Cyclic => 1,
            };
            (key.ring_d, domain)
        });
    }
    Ok(owners)
}

impl<'a, F, C, O, T, R> LevelProveStacks<'a, F> for ProverComputeStack<'a, F, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    type Commit = C;
    type Opening = O;
    type Tensor = T;
    type RingSwitch = R;

    fn prove_stack_at_level(&self, _level: usize) -> &Self {
        self
    }
}

impl<'a, F, C, O, T, R, S> LevelProveStacks<'a, F> for &S
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
    S: LevelProveStacks<'a, F, Commit = C, Opening = O, Tensor = T, RingSwitch = R> + ?Sized,
{
    type Commit = C;
    type Opening = O;
    type Tensor = T;
    type RingSwitch = R;

    fn prove_stack_at_level(&self, level: usize) -> &ProverComputeStack<'a, F, C, O, T, R> {
        (*self).prove_stack_at_level(level)
    }

    fn after_root_fold(&self) -> Result<(), AkitaError> {
        <S as LevelProveStacks<'a, F>>::after_root_fold(*self)
    }
}

/// Tiered fold boundaries for [`LevelProveStacks`].
///
/// `tier_max_level[i]` is the last fold level (inclusive) handled by `stacks[i]`.
/// The final tier should use `usize::MAX` so every remaining fold maps to it.
///
/// # Example
///
/// Folds 0–1 on `multi_gpu`, 2–3 on `single_gpu`, 4+ on `cpu`:
///
/// ```ignore
/// let stacks = [multi_gpu, single_gpu, cpu];
/// let tiered = TieredProveStacks::new(&stacks, &[1, 3, usize::MAX])?;
/// batched_prove(..., &tiered, ...)?;
/// ```
pub struct TieredProveStacks<'a, F, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    stacks: &'a [ProverComputeStack<'a, F, C, O, T, R>],
    tier_max_level: &'a [usize],
}

impl<'a, F, C, O, T, R> TieredProveStacks<'a, F, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    /// Build a tier table. `stacks.len()` must equal `tier_max_level.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tier table is empty or `tier_max_level` is not
    /// strictly increasing.
    pub fn new(
        stacks: &'a [ProverComputeStack<'a, F, C, O, T, R>],
        tier_max_level: &'a [usize],
    ) -> Result<Self, AkitaError> {
        if stacks.is_empty() {
            return Err(AkitaError::InvalidInput(
                "tiered prove stacks require at least one stack".to_string(),
            ));
        }
        if tier_max_level.len() != stacks.len() {
            return Err(AkitaError::InvalidInput(
                "tiered prove stacks length mismatch".to_string(),
            ));
        }
        for window in tier_max_level.windows(2) {
            if window[0] >= window[1] {
                return Err(AkitaError::InvalidInput(
                    "tier_max_level must be strictly increasing".to_string(),
                ));
            }
        }
        Ok(Self {
            stacks,
            tier_max_level,
        })
    }

    fn tier_index_for_level(&self, level: usize) -> usize {
        self.tier_max_level
            .iter()
            .position(|max_level| level <= *max_level)
            .unwrap_or(self.stacks.len() - 1)
    }
}

impl<'a, F, C, O, T, R> LevelProveStacks<'a, F> for TieredProveStacks<'a, F, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    type Commit = C;
    type Opening = O;
    type Tensor = T;
    type RingSwitch = R;

    fn prove_stack_at_level(&self, level: usize) -> &ProverComputeStack<'a, F, C, O, T, R> {
        &self.stacks[self.tier_index_for_level(level)]
    }
}

impl<'a, F, B> ProverComputeStack<'a, F, B, B, B, B>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
    /// Build a CPU-only / single-backend stack where every operation cluster
    /// shares one backend and prepared setup. Validates the prepared setup once
    /// per cluster against `expanded`.
    ///
    /// # Errors
    ///
    /// Returns an error if the prepared setup fails validation.
    pub fn uniform(
        backend: &'a B,
        prepared: &'a B::PreparedSetup,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        Self::new(
            (backend, prepared),
            (backend, prepared),
            (backend, prepared),
            (backend, prepared),
            expanded,
        )
    }
}

#[cfg(test)]
mod tests;
