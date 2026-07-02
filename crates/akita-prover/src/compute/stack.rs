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

use crate::compute::backend::ComputeBackendSetup;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{AkitaExpandedSetup, NttCacheKey};
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

    /// Warm the full-envelope NTT slot for `ring_d` on this cluster's prepared setup.
    ///
    /// # Errors
    ///
    /// Returns an error when the envelope key cannot be derived or cache build fails.
    pub fn ensure_envelope_ntt(
        &self,
        expanded: &AkitaExpandedSetup<F>,
        ring_d: usize,
    ) -> Result<(), AkitaError> {
        let key = NttCacheKey::from_envelope(expanded, ring_d)?;
        self.backend.ensure_ntt_slot(self.prepared, key)
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

    /// Warm full-envelope NTT slots for every cluster at fold ring degree `ring_d`.
    ///
    /// # Errors
    ///
    /// Returns an error when any cluster fails envelope key derivation or cache build.
    pub fn ensure_fold_level_envelope_ntt(
        &self,
        expanded: &AkitaExpandedSetup<F>,
        ring_d: usize,
    ) -> Result<(), AkitaError> {
        self.commit.ensure_envelope_ntt(expanded, ring_d)?;
        self.opening.ensure_envelope_ntt(expanded, ring_d)?;
        self.tensor.ensure_envelope_ntt(expanded, ring_d)?;
        self.ring_switch.ensure_envelope_ntt(expanded, ring_d)
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
mod tests {
    use super::*;
    use crate::AkitaProverSetup;
    use crate::CpuBackend;
    use akita_field::{AkitaError, Fp64};
    use akita_types::SetupMatrixEnvelope;

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn test_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope { max_setup_len }
    }

    #[test]
    fn operation_ctx_rejects_mismatched_expanded_setup() {
        let setup_a = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(4096))
            .expect("setup a");
        let setup_b = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(8192))
            .expect("setup b");
        assert_ne!(setup_a.expanded.seed(), setup_b.expanded.seed());

        let prepared_a = CpuBackend.prepare_setup(&setup_a).expect("prepared a");
        assert!(matches!(
            OperationCtx::new(&CpuBackend, &prepared_a, setup_b.expanded.as_ref()),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn operation_ctx_accepts_matching_expanded_setup() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(4096))
            .expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
        OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("matching expanded metadata should validate");
    }

    use crate::compute::{CommitCluster, RingSwitchCluster};

    fn assert_distinct_backend_types<C: 'static, R: 'static>() {
        fn type_id<T: 'static>() -> std::any::TypeId {
            std::any::TypeId::of::<T>()
        }
        assert_ne!(type_id::<C>(), type_id::<R>());
    }

    type TestUniformStack<'a> = UniformProverStack<'a, F, CpuBackend>;
    type TestHeterogeneousStack<'a> =
        ProverComputeStack<'a, F, CommitCluster, CpuBackend, CpuBackend, RingSwitchCluster>;

    #[test]
    fn heterogeneous_stack_accepts_distinct_operation_clusters() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(4096))
            .expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
        let commit_backend = CommitCluster;
        let ring_backend = RingSwitchCluster;
        let stack: TestHeterogeneousStack<'_> = ProverComputeStack::new(
            (&commit_backend, &prepared),
            (&CpuBackend, &prepared),
            (&CpuBackend, &prepared),
            (&ring_backend, &prepared),
            setup.expanded.as_ref(),
        )
        .expect("heterogeneous stack");
        assert_distinct_backend_types::<CommitCluster, RingSwitchCluster>();
        assert_eq!(
            stack.commit().backend() as *const _,
            &commit_backend as *const _
        );
        assert_eq!(
            stack.ring_switch().backend() as *const _,
            &ring_backend as *const _
        );
    }

    #[test]
    fn heterogeneous_stack_implements_level_prove_stacks() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(4096))
            .expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
        let commit_backend = CommitCluster;
        let ring_backend = RingSwitchCluster;
        let stack: TestHeterogeneousStack<'_> = ProverComputeStack::new(
            (&commit_backend, &prepared),
            (&CpuBackend, &prepared),
            (&CpuBackend, &prepared),
            (&ring_backend, &prepared),
            setup.expanded.as_ref(),
        )
        .expect("heterogeneous stack");
        let selected: &TestHeterogeneousStack<'_> =
            LevelProveStacks::prove_stack_at_level(&stack, 0);
        assert_eq!(
            selected.commit().backend() as *const _,
            stack.commit().backend() as *const _
        );
    }

    #[test]
    fn tiered_prove_stacks_rejects_empty_table() {
        let result =
            TieredProveStacks::<F, CpuBackend, CpuBackend, CpuBackend, CpuBackend>::new(&[], &[]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }

    #[test]
    fn tiered_prove_stacks_rejects_length_mismatch() {
        let setup = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(4096))
            .expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
        let stack = TestUniformStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
        let stacks = [stack];
        let result = TieredProveStacks::new(&stacks, &[1, 2]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }

    #[test]
    fn tiered_prove_stacks_rejects_non_increasing_bounds() {
        let setup_a = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(4096))
            .expect("setup a");
        let setup_b = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(8192))
            .expect("setup b");
        let prepared_a = CpuBackend.prepare_setup(&setup_a).expect("prepared a");
        let prepared_b = CpuBackend.prepare_setup(&setup_b).expect("prepared b");
        let stack_a =
            TestUniformStack::uniform(&CpuBackend, &prepared_a, setup_a.expanded.as_ref())
                .expect("stack a");
        let stack_b =
            TestUniformStack::uniform(&CpuBackend, &prepared_b, setup_b.expanded.as_ref())
                .expect("stack b");
        let stacks = [stack_a, stack_b];
        let result = TieredProveStacks::new(&stacks, &[2, 1]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }

    #[test]
    fn tiered_prove_stacks_selects_by_fold_level() {
        let setup_a = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(4096))
            .expect("setup a");
        let setup_b = AkitaProverSetup::<F>::generate_with_capacity(8, 1, D, test_envelope(8192))
            .expect("setup b");
        let prepared_a = CpuBackend.prepare_setup(&setup_a).expect("prepared a");
        let prepared_b = CpuBackend.prepare_setup(&setup_b).expect("prepared b");
        let stack_a =
            TestUniformStack::uniform(&CpuBackend, &prepared_a, setup_a.expanded.as_ref())
                .expect("stack a");
        let stack_b =
            TestUniformStack::uniform(&CpuBackend, &prepared_b, setup_b.expanded.as_ref())
                .expect("stack b");
        let stacks = [stack_a, stack_b];
        let tiered = TieredProveStacks::new(&stacks, &[1, usize::MAX]).expect("tiered");
        assert!(std::ptr::eq(
            tiered.prove_stack_at_level(0),
            tiered.prove_stack_at_level(1),
        ));
        assert!(!std::ptr::eq(
            tiered.prove_stack_at_level(0),
            tiered.prove_stack_at_level(2),
        ));
    }
}
