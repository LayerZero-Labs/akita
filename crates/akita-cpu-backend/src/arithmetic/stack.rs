//! Private arithmetic context for an owning backend and validated setup cache.
use crate::opaque::ComputeBackendSetup;
use akita_error::AkitaError;
use akita_types::AkitaExpandedSetup;
use jolt_field::{CanonicalEncoding, Field};
use std::marker::PhantomData;
/// A single operation context: a backend plus its validated prepared setup.
///
/// Construction validates the prepared setup against explicit expanded-setup
/// metadata, so a kernel may assume its context was validated. The fields are
/// private to keep that invariant: an `OperationCtx` cannot exist without going
/// through a validating constructor.
pub(crate) struct OperationCtx<'a, F, B>
where
    F: Field + CanonicalEncoding,
    B: ComputeBackendSetup<F>,
{
    backend: &'a B,
    prepared: &'a B::PreparedSetup,
    _field: PhantomData<fn() -> F>,
}

impl<'a, F, B> OperationCtx<'a, F, B>
where
    F: Field + CanonicalEncoding,
    B: ComputeBackendSetup<F>,
{
    /// Build an operation context, validating `prepared` against `expanded`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] (via
    /// [`ComputeBackendSetup::validate_prepared_setup`]) when `prepared` was not
    /// built from `expanded`.
    pub(crate) fn new(
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
    pub(crate) fn backend(&self) -> &'a B {
        self.backend
    }

    /// Borrowed prepared setup for this operation cluster.
    pub(crate) fn prepared(&self) -> &'a B::PreparedSetup {
        self.prepared
    }
}
