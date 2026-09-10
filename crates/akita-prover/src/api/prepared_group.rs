//! Typed borrowed group carriers for prover execution.

use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::FpExtEncoding;
use jolt_field::Unreduced;
use jolt_field::{AdditiveGroup, CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use std::sync::Arc;

/// Homogeneous polynomial storage for one prepared prover group.
///
/// Root orchestration is generic over this coarse group carrier. Use
/// [`ErasedPreparedProverGroup`] to combine groups with different concrete
/// polynomial types in one opening batch.
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

/// Type-erased borrowed polynomial group for heterogeneous opening batches.
///
/// Every polynomial inside one value still has one concrete type. Erasure is
/// applied once to the complete group so dense and one-hot groups can be
/// proved together without a per-polynomial sum type.
pub struct ErasedPreparedProverGroup<'a, F, E, O>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
{
    inner: Arc<dyn crate::protocol::core::RootProverGroupOpening<F, E, O> + 'a>,
}

impl<'a, F, E, O> Clone for ErasedPreparedProverGroup<'a, F, E, O>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<'a, F, E, O> ErasedPreparedProverGroup<'a, F, E, O>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
{
    /// Prepare and erase one nonempty homogeneous polynomial group.
    pub fn from_refs<P>(polys: &'a [&'a P]) -> Result<Self, AkitaError>
    where
        P: crate::compute::RuntimeRootProvePoly<F>,
        O: crate::compute::RuntimeOpeningProveBackendFor<F, P>
            + crate::compute::RuntimeCoefficientPackingBackendFor<F, P, E>,
    {
        Ok(Self {
            inner: Arc::new(PreparedProverGroup::from_refs(polys)?),
        })
    }
}

impl<F, E, O> crate::protocol::core::RootProverGroupMeta<F>
    for ErasedPreparedProverGroup<'_, F, E, O>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
{
    fn num_polynomials(&self) -> usize {
        self.inner.num_polynomials()
    }

    fn num_vars(&self) -> Result<usize, AkitaError> {
        self.inner.num_vars()
    }

    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
        self.inner.exact_integer_coeff_l2_sq()
    }
}

impl<F, E, O> crate::protocol::core::RootProverGroupOpening<F, E, O>
    for ErasedPreparedProverGroup<'_, F, E, O>
where
    F: Field + CanonicalEncoding + Ring + Unreduced + AkitaSerialize + 'static,
    <F as Unreduced>::Wide: From<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
{
    fn prepare_opening(
        &self,
        ctx: &crate::compute::OperationCtx<'_, F, O>,
        ring_dimension: usize,
        protocol_point: &[E],
        basis: akita_types::BasisMode,
        num_positions_per_block: usize,
        num_live_blocks: usize,
        alpha_bits: usize,
        opening_method: akita_types::OpeningMethod,
    ) -> Result<crate::protocol::core::PreparedGroupOpening<F, E>, AkitaError> {
        self.inner.prepare_opening(
            ctx,
            ring_dimension,
            protocol_point,
            basis,
            num_positions_per_block,
            num_live_blocks,
            alpha_bits,
            opening_method,
        )
    }

    fn probe_fold(
        &self,
        ctx: &crate::compute::OperationCtx<'_, F, O>,
        challenges: &crate::protocol::fold_grind::GroupFoldChallenges,
        root_params: &akita_types::CommittedGroupParams,
        params: &akita_types::GroupOpenPhaseParams,
    ) -> Result<crate::protocol::fold_grind::FoldProbeOutput<F>, AkitaError> {
        self.inner.probe_fold(ctx, challenges, root_params, params)
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
