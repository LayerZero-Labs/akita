//! Typed borrowed group carriers for prover execution.

use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::FpExtEncoding;
use jolt_field::Unreduced;
use jolt_field::{AdditiveGroup, CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Homogeneous polynomial storage for one prepared prover group.
///
/// Root orchestration is generic over this coarse group carrier. Applications
/// that need multiple polynomial representations can use one application-owned
/// enum as `P`; Akita does not recursively compose provider or group wrappers.
#[derive(Debug, Clone)]
pub struct PreparedProverGroup<'a, P> {
    polys: Vec<&'a P>,
}

impl<'a, P> PreparedProverGroup<'a, P> {
    /// Borrow every polynomial in one homogeneous group.
    pub fn new(polys: &'a [P]) -> Result<Self, AkitaError> {
        if polys.is_empty() {
            return Err(AkitaError::InvalidInput(
                "prepared prover group must be nonempty".to_string(),
            ));
        }
        Ok(Self {
            polys: polys.iter().collect(),
        })
    }

    /// Preserve an existing borrowed polynomial-reference slice as one group.
    pub fn from_refs(polys: &'a [&'a P]) -> Result<Self, AkitaError> {
        if polys.is_empty() {
            return Err(AkitaError::InvalidInput(
                "prepared prover group must be nonempty".to_string(),
            ));
        }
        Ok(Self {
            polys: polys.to_vec(),
        })
    }

    pub(crate) fn polynomial_refs(&self) -> &[&'a P] {
        &self.polys
    }

    pub(crate) fn from_ref_vec(polys: Vec<&'a P>) -> Result<Self, AkitaError> {
        if polys.is_empty() {
            return Err(AkitaError::InvalidInput(
                "prepared prover group must be nonempty".to_string(),
            ));
        }
        Ok(Self { polys })
    }
}

/// Capability marker for a prepared whole-group carrier and prover backends.
///
/// This is implemented automatically for [`PreparedProverGroup`]. Applications
/// choose the concrete polynomial type inside the carrier; low-level kernels
/// remain statically dispatched.
#[allow(private_bounds)]
pub trait PreparedGroupProveOps<F, E, O>:
    crate::protocol::core::RootProverGroupOpening<F, E, O> + Clone
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
{
}

impl<F, E, O, G> PreparedGroupProveOps<F, E, O> for G
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
    G: crate::protocol::core::RootProverGroupOpening<F, E, O> + Clone,
{
}
