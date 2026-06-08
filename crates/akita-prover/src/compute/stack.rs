use crate::compute::backend::ComputeBackendSetup;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::AkitaExpandedSetup;
use std::marker::PhantomData;

/// A single operation context: a backend plus its validated prepared setup.
///
/// Construction validates the prepared setup against explicit expanded-setup
/// metadata, so a kernel may assume its context was validated. The fields are
/// private to keep that invariant: an `OperationCtx` cannot exist without going
/// through a validating constructor.
pub struct OperationCtx<'a, F, B, const D: usize>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F>,
{
    backend: &'a B,
    prepared: &'a B::PreparedSetup<D>,
    _field: PhantomData<fn() -> F>,
}

impl<'a, F, B, const D: usize> OperationCtx<'a, F, B, D>
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
        prepared: &'a B::PreparedSetup<D>,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<Self, AkitaError> {
        backend.validate_prepared_setup::<D>(prepared, expanded)?;
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
    pub fn prepared(&self) -> &'a B::PreparedSetup<D> {
        self.prepared
    }
}

/// Per-operation-cluster prover compute stack.
///
/// Each cluster (commit / opening / tensor / ring-switch) carries its own
/// backend plus prepared context, so a proof may mix backends across clusters.
/// A CPU-only prover is the degenerate case where every cluster shares one
/// backend and prepared setup ([`ProverComputeStack::uniform`]).
pub struct ProverComputeStack<'a, F, const D: usize, C, O, T, R>
where
    F: FieldCore + CanonicalField,
    C: ComputeBackendSetup<F>,
    O: ComputeBackendSetup<F>,
    T: ComputeBackendSetup<F>,
    R: ComputeBackendSetup<F>,
{
    commit: OperationCtx<'a, F, C, D>,
    opening: OperationCtx<'a, F, O, D>,
    tensor: OperationCtx<'a, F, T, D>,
    ring_switch: OperationCtx<'a, F, R, D>,
}

impl<'a, F, const D: usize, C, O, T, R> ProverComputeStack<'a, F, D, C, O, T, R>
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
        commit: (&'a C, &'a C::PreparedSetup<D>),
        opening: (&'a O, &'a O::PreparedSetup<D>),
        tensor: (&'a T, &'a T::PreparedSetup<D>),
        ring_switch: (&'a R, &'a R::PreparedSetup<D>),
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
    pub fn commit(&self) -> &OperationCtx<'a, F, C, D> {
        &self.commit
    }

    /// Opening / decompose-fold operation context.
    pub fn opening(&self) -> &OperationCtx<'a, F, O, D> {
        &self.opening
    }

    /// Tensor projection operation context.
    pub fn tensor(&self) -> &OperationCtx<'a, F, T, D> {
        &self.tensor
    }

    /// Ring-switch operation context.
    pub fn ring_switch(&self) -> &OperationCtx<'a, F, R, D> {
        &self.ring_switch
    }
}

/// Single-backend degenerate [`ProverComputeStack`] (all four clusters share `B`).
pub type UniformProverStack<'a, F, B, const D: usize> = ProverComputeStack<'a, F, D, B, B, B, B>;

impl<'a, F, B, const D: usize> ProverComputeStack<'a, F, D, B, B, B, B>
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
        prepared: &'a B::PreparedSetup<D>,
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
    use akita_field::Fp64;
    use akita_types::SetupMatrixEnvelope;

    type F = Fp64<4294967197>;
    const D: usize = 32;

    fn test_envelope(max_setup_len: usize) -> SetupMatrixEnvelope {
        SetupMatrixEnvelope {
            max_setup_len,
            #[cfg(feature = "zk")]
            max_zk_b_len: 0,
            #[cfg(feature = "zk")]
            max_zk_d_len: 0,
        }
    }

    #[test]
    fn operation_ctx_rejects_mismatched_expanded_setup() {
        let setup_a =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, test_envelope(4096))
                .expect("setup a");
        let setup_b =
            AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, test_envelope(8192))
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
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(8, 1, 1, test_envelope(4096))
            .expect("setup");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared");
        OperationCtx::new(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("matching expanded metadata should validate");
    }
}
