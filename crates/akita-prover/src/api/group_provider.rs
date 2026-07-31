//! Prover-only whole-group source providers.
//!
//! Public commitments bind an exact [`akita_types::CommittedGroupProfile`].
//! They do not identify the application representation that produced the
//! committed polynomials. That representation boundary lives here: a provider
//! validates one whole group and returns a typed borrowed [`PreparedGroup`]
//! before config/catalog selection or commitment kernels run.

use crate::compute::RootPolyMeta;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    MulBaseUnreduced,
};
use akita_serialization::AkitaSerialize;
use akita_types::FpExtEncoding;
use akita_types::GroupSource;

/// A provider-validated, borrowed polynomial group.
///
/// The polynomial type remains concrete, so commitment and tensor kernels
/// monomorphize exactly as they do for a direct `&[P]` input. The erased source
/// model is retained only for prover-side catalog selection; it is not part of
/// the committed profile or verifier input.
#[derive(Debug)]
pub struct PreparedGroup<'a, P> {
    polys: &'a [P],
    planning_source: GroupSource,
}

impl<'a, P> PreparedGroup<'a, P> {
    /// Borrow the provider-validated polynomials.
    #[inline]
    pub fn polynomials(&self) -> &'a [P] {
        self.polys
    }

    #[inline]
    pub(crate) fn planning_source(&self) -> GroupSource {
        self.planning_source
    }

    /// Carry this already validated group into opening-time root orchestration.
    pub fn into_prover_group(self) -> PreparedProverGroup<'a, P> {
        PreparedProverGroup {
            polys: self.polys.iter().collect(),
        }
    }
}

/// Homogeneous polynomial storage for one prepared prover group.
///
/// Root orchestration is generic over this group carrier rather than over one
/// polynomial type for the whole batch. [`EitherPreparedGroup`] composes
/// carriers with unrelated polynomial types while each operation still enters
/// a monomorphized kernel over the concrete variant.
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

/// One of two unrelated prepared whole-group representations.
///
/// Nest this type to combine more than two source representations. Dispatch is
/// once per whole-group operation; polynomial and backend kernels remain
/// monomorphized within the selected variant.
#[derive(Debug, Clone)]
pub enum EitherPreparedGroup<L, R> {
    /// First whole-group representation.
    Left(L),
    /// Second whole-group representation.
    Right(R),
}

/// Capability marker for a prepared whole-group carrier and prover backends.
///
/// This is implemented automatically for [`PreparedProverGroup`] and
/// compositions of [`EitherPreparedGroup`]. Applications choose arbitrary
/// concrete polynomial types inside those carriers; low-level kernels remain
/// statically dispatched.
#[allow(private_bounds)]
pub trait PreparedGroupProveOps<F, E, O, TS>:
    crate::protocol::core::RootProverGroupOpening<F, E, O>
    + crate::protocol::core::RootProverGroupTensor<F, E, TS>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
    TS: crate::compute::ComputeBackendSetup<F>,
{
}

impl<F, E, O, TS, G> PreparedGroupProveOps<F, E, O, TS> for G
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F> + AkitaSerialize,
    O: crate::compute::ComputeBackendSetup<F> + crate::compute::DigitRowsComputeBackend<F>,
    TS: crate::compute::ComputeBackendSetup<F>,
    G: crate::protocol::core::RootProverGroupOpening<F, E, O>
        + crate::protocol::core::RootProverGroupTensor<F, E, TS>,
{
}

/// Open prover-side boundary for one complete commitment group.
///
/// Implementations may enforce application-specific invariants across the
/// entire group. [`Self::planning_source`] supplies only the honest-prover
/// model used to select a generated row; verifier acceptance is determined by
/// that row's exact v1 profile and parameters.
pub trait WholeGroupSourceProvider<F, P>: Send + Sync
where
    F: CanonicalField,
    P: RootPolyMeta<F>,
{
    /// Erased honest-prover model used for generated-row lookup.
    fn planning_source(&self) -> GroupSource;

    /// Validate every representation invariant owned by this provider.
    fn validate_group(&self, polynomials: &[P]) -> Result<(), AkitaError>;

    /// Validate and prepare a whole group for monomorphized commitment kernels.
    fn prepare_group<'a>(&self, polynomials: &'a [P]) -> Result<PreparedGroup<'a, P>, AkitaError> {
        if polynomials.is_empty() {
            return Err(AkitaError::InvalidInput(
                "source provider requires a nonempty polynomial group".to_string(),
            ));
        }
        let planning_source = self.planning_source();
        planning_source.validate(F::modulus_bits())?;
        self.validate_group(polynomials)?;
        Ok(PreparedGroup {
            polys: polynomials,
            planning_source,
        })
    }

    /// Validate and prepare a whole group for opening-time prover kernels.
    fn prepare_prover_group<'a>(
        &self,
        polynomials: &'a [P],
    ) -> Result<PreparedProverGroup<'a, P>, AkitaError> {
        self.prepare_group(polynomials)
            .map(PreparedGroup::into_prover_group)
    }
}

fn validate_builtin_group<F, P>(polynomials: &[P], source: GroupSource) -> Result<(), AkitaError>
where
    F: CanonicalField,
    P: RootPolyMeta<F>,
{
    for (poly_idx, poly) in polynomials.iter().enumerate() {
        poly.validate_group_source(source).map_err(|err| {
            AkitaError::InvalidInput(format!(
                "polynomial {poly_idx} does not satisfy the whole-group provider: {err}"
            ))
        })?;
    }
    Ok(())
}

/// Built-in provider for dense bounded-coefficient polynomial groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseGroupProvider {
    coefficient_bits: u32,
}

impl DenseGroupProvider {
    /// Construct a dense provider with the declared coefficient bound.
    #[must_use]
    pub const fn new(coefficient_bits: u32) -> Self {
        Self { coefficient_bits }
    }

    /// Declared dense coefficient bit bound.
    #[must_use]
    pub const fn coefficient_bits(self) -> u32 {
        self.coefficient_bits
    }
}

impl<F, P> WholeGroupSourceProvider<F, P> for DenseGroupProvider
where
    F: CanonicalField,
    P: RootPolyMeta<F>,
{
    fn planning_source(&self) -> GroupSource {
        GroupSource::bounded(self.coefficient_bits)
    }

    fn validate_group(&self, polynomials: &[P]) -> Result<(), AkitaError> {
        validate_builtin_group::<F, P>(polynomials, GroupSource::bounded(self.coefficient_bits))
    }
}

/// Built-in provider for sparse-binary one-hot polynomial groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneHotGroupProvider {
    chunk_size: usize,
}

impl OneHotGroupProvider {
    /// Construct a one-hot provider with exact chunk size `K`.
    #[must_use]
    pub const fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    /// Exact one-hot chunk size `K`.
    #[must_use]
    pub const fn chunk_size(self) -> usize {
        self.chunk_size
    }
}

impl<F, P> WholeGroupSourceProvider<F, P> for OneHotGroupProvider
where
    F: CanonicalField,
    P: RootPolyMeta<F>,
{
    fn planning_source(&self) -> GroupSource {
        GroupSource::one_hot(self.chunk_size)
    }

    fn validate_group(&self, polynomials: &[P]) -> Result<(), AkitaError> {
        validate_builtin_group::<F, P>(polynomials, GroupSource::one_hot(self.chunk_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime32Offset99;
    use akita_types::GroupSourceEncoding;

    type F = Prime32Offset99;

    #[derive(Clone)]
    struct MockPoly {
        expected: GroupSourceEncoding,
    }

    impl RootPolyMeta<F> for MockPoly {
        fn num_ring_elems(&self) -> usize {
            1
        }

        fn num_vars(&self) -> usize {
            5
        }

        fn validate_group_source(&self, source: GroupSource) -> Result<(), AkitaError> {
            if source.encoding() != self.expected {
                return Err(AkitaError::InvalidInput(
                    "mock representation/source mismatch".to_string(),
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn builtins_prepare_complete_typed_groups() {
        let dense = [MockPoly {
            expected: GroupSource::bounded(32).encoding(),
        }];
        let prepared = DenseGroupProvider::new(32)
            .prepare_group(&dense)
            .expect("dense group");
        assert_eq!(prepared.polynomials().len(), 1);
        assert_eq!(
            prepared.planning_source().encoding(),
            GroupSource::bounded(32).encoding()
        );

        let one_hot = [MockPoly {
            expected: GroupSource::one_hot(16).encoding(),
        }];
        OneHotGroupProvider::new(16)
            .prepare_group(&one_hot)
            .expect("one-hot group");
    }

    #[test]
    fn provider_rejects_empty_or_mismatched_groups() {
        let empty: [MockPoly; 0] = [];
        assert!(DenseGroupProvider::new(32).prepare_group(&empty).is_err());

        let dense = [MockPoly {
            expected: GroupSource::bounded(32).encoding(),
        }];
        assert!(OneHotGroupProvider::new(16).prepare_group(&dense).is_err());
    }
}
