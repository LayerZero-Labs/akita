use super::{
    BackendKindId, CommitmentRequestCapabilities, ExternalInnerCommitmentCapability,
    PreparedExternalInnerCommitment,
};
use crate::opaque::CommitInnerPlan;
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, Field};

use compiler::checked_logical_len;

/// Admission class of a commitment source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CommitSourceClass {
    /// General field coefficients represented by balanced signed digits.
    Dense,
    /// Bounded signed coefficients in Akita's packed representation.
    ShortNorm,
    /// At most one unit coefficient in every fixed-size chunk.
    OneHot {
        /// Number of logical coefficients in one one-hot chunk.
        chunk_size: usize,
    },
}

#[cfg(test)]
#[path = "tests/source_bounds.rs"]
mod bounds_tests;

/// O(1) structural metadata used before representation materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommitSourceDescriptor {
    num_vars: usize,
    total_coefficient_len: usize,
    live_coefficient_len: usize,
    class: CommitSourceClass,
    family_name: &'static str,
}

impl CommitSourceDescriptor {
    /// Construct a descriptor after checking its extents.
    pub(crate) fn new(
        num_vars: usize,
        total_coefficient_len: usize,
        live_coefficient_len: usize,
        class: CommitSourceClass,
        family_name: &'static str,
    ) -> Result<Self, AkitaError> {
        let logical_len = checked_logical_len(num_vars)?;
        if live_coefficient_len == 0
            || live_coefficient_len > logical_len
            || logical_len > total_coefficient_len
        {
            return Err(AkitaError::InvalidInput(format!(
                "commit source extents live={live_coefficient_len}, logical={logical_len}, total={total_coefficient_len} are inconsistent"
            )));
        }
        if !matches!(class, CommitSourceClass::ShortNorm) && live_coefficient_len != logical_len {
            return Err(AkitaError::InvalidInput(
                "dense and one-hot commitment sources must cover their complete logical domain"
                    .into(),
            ));
        }
        if matches!(class, CommitSourceClass::OneHot { chunk_size: 0 }) {
            return Err(AkitaError::InvalidInput(
                "one-hot commitment source requires a nonzero chunk size".into(),
            ));
        }
        if family_name.is_empty() {
            return Err(AkitaError::InvalidInput(
                "commit source family name must not be empty".into(),
            ));
        }
        Ok(Self {
            num_vars,
            total_coefficient_len,
            live_coefficient_len,
            class,
            family_name,
        })
    }

    /// Logical multilinear variable count.
    pub(crate) const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Physical coefficient extent, including commitment padding.
    pub(crate) const fn total_coefficient_len(&self) -> usize {
        self.total_coefficient_len
    }

    /// Source-owned live coefficient extent.
    pub(crate) const fn live_coefficient_len(&self) -> usize {
        self.live_coefficient_len
    }

    /// Admission class used by the selected schedule contract.
    pub(crate) const fn class(&self) -> CommitSourceClass {
        self.class
    }
}

/// Physical dense representation offered by a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DenseType {
    /// Borrowed canonical field coefficients.
    Coefficients,
    /// Borrowed exact-plan balanced digit planes.
    PredecomposedDigits,
}

/// Packed short-norm source representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShortNormType {
    signed_bit_width: u8,
}

impl ShortNormType {
    /// Describe a checked packed signed width.
    pub(crate) fn new(signed_bit_width: u8) -> Result<Self, AkitaError> {
        if !(1..=8).contains(&signed_bit_width) {
            return Err(AkitaError::InvalidInput(format!(
                "packed short-norm signed width {signed_bit_width} is outside 1..=8"
            )));
        }
        Ok(Self { signed_bit_width })
    }
}

/// Width of the source-owned one-hot position indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum OneHotIndexWidth {
    /// Eight-bit positions.
    U8,
    /// Sixteen-bit positions.
    U16,
    /// Thirty-two-bit positions.
    U32,
    /// Native-word positions.
    Usize,
}

/// Semantic one-hot representation descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OneHotType {
    chunk_size: usize,
    index_width: OneHotIndexWidth,
}

impl OneHotType {
    /// Construct a checked one-hot type.
    pub(crate) fn new(
        chunk_size: usize,
        index_width: OneHotIndexWidth,
    ) -> Result<Self, AkitaError> {
        if chunk_size == 0 {
            return Err(AkitaError::InvalidInput(
                "one-hot polynomial type requires a nonzero chunk size".into(),
            ));
        }
        Ok(Self {
            chunk_size,
            index_width,
        })
    }

    /// Stored position-index width.
    pub(crate) const fn index_width(self) -> OneHotIndexWidth {
        self.index_width
    }
}

/// Akita-owned semantic polynomial types understood by commitment operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PolynomialType {
    /// General dense coefficients or exact-plan cached digit planes.
    Dense(DenseType),
    /// Packed bounded signed coefficients.
    ShortNorm(ShortNormType),
    /// Structurally one-hot coefficients.
    OneHot(OneHotType),
}

/// Side-effect-free representation offerings for one source and plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AvailablePolynomialTypes {
    entries: Vec<PolynomialType>,
}

impl AvailablePolynomialTypes {
    /// Construct a duplicate-free offering list.
    ///
    /// An empty list is valid for a source that supports only an external
    /// inner commitment operation.
    pub(crate) fn new(entries: Vec<PolynomialType>) -> Result<Self, AkitaError> {
        if entries
            .iter()
            .enumerate()
            .any(|(index, entry)| entries[..index].contains(entry))
        {
            return Err(AkitaError::InvalidInput(
                "commit source advertised a duplicate polynomial representation".into(),
            ));
        }
        Ok(Self { entries })
    }

    /// Offered representations in source preference order.
    pub(crate) fn as_slice(&self) -> &[PolynomialType] {
        &self.entries
    }

    /// Issue an opaque selection for one advertised entry.
    #[cfg(test)]
    pub(crate) fn select(&self, index: usize) -> Result<PolynomialTypeSelection, AkitaError> {
        let selected = self.entries.get(index).copied().ok_or_else(|| {
            AkitaError::InvalidInput("polynomial representation selection is out of range".into())
        })?;
        Ok(PolynomialTypeSelection { selected })
    }
}

/// Opaque representation choice issued after capability intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PolynomialTypeSelection {
    selected: PolynomialType,
}

mod compiler;
pub(crate) use compiler::{compile_commitment_request, CompiledCommitmentRequest};

impl PolynomialTypeSelection {
    /// Selected semantic and physical representation.
    pub(crate) const fn polynomial_type(self) -> PolynomialType {
        self.selected
    }
}

/// Borrowed source of canonical dense coefficients.
pub(crate) trait DenseCoefficientSource<F: Field>: Send + Sync {
    /// Physical coefficient buffer in canonical source order.
    fn coefficients(&self) -> &[F];
}

/// Borrowed dense source representation.
pub(crate) enum DenseRepresentation<'a, F: Field> {
    /// Canonical field coefficients.
    Coefficients(&'a dyn DenseCoefficientSource<F>),
    /// Exact-plan balanced digit planes in `[ring][digit][coefficient]` order.
    PredecomposedDigits(PredecomposedDigitPlanes<'a>),
}

/// Checked borrowed dense digit planes.
pub(crate) struct PredecomposedDigitPlanes<'a> {
    /// Flat signed digits in `[ring][digit][coefficient]` order.
    pub bytes: &'a [i8],
    /// Ring dimension of each digit plane.
    pub ring_dimension: usize,
    /// Number of digit planes per ring.
    pub num_digits: usize,
    /// Logarithm of the balanced decomposition basis.
    pub log_basis: u32,
    /// Logical number of source rings.
    pub logical_ring_count: usize,
}

/// Borrowed packed bounded signed coefficients.
pub(crate) struct ShortNormRepresentation<'a> {
    /// Encoded two's-complement payload, excluding safe-load padding.
    encoded_bytes: &'a [u8],
    /// Number of source-owned live coefficients.
    live_coefficient_len: usize,
    /// Commitment-aligned logical coefficient extent.
    physical_coefficient_len: usize,
    /// Stored two's-complement bit width.
    signed_bit_width: u8,
    packed_view: crate::sources::packed_digits::PackedSignedDigitView<'a>,
}

impl<'a> ShortNormRepresentation<'a> {
    /// Construct a portable borrowed packed representation.
    #[cfg(test)]
    pub(crate) fn new(
        encoded_bytes: &'a [u8],
        live_coefficient_len: usize,
        stored_coefficient_len: usize,
        physical_coefficient_len: usize,
        signed_bit_width: u8,
        negative_abs_max: u8,
        positive_max: u8,
    ) -> Result<Self, AkitaError> {
        ShortNormType::new(signed_bit_width)?;
        if live_coefficient_len == 0
            || live_coefficient_len > stored_coefficient_len
            || stored_coefficient_len > physical_coefficient_len
        {
            return Err(AkitaError::InvalidInput(
                "packed short-norm extents are inconsistent".into(),
            ));
        }
        let stored_bits = checked::product([stored_coefficient_len, usize::from(signed_bit_width)])
            .ok_or_else(|| {
                AkitaError::InvalidInput("packed short-norm bit length overflow".into())
            })?;
        let stored_bytes = checked::div_ceil(stored_bits, 8).ok_or_else(|| {
            AkitaError::InvalidInput("packed short-norm byte length overflow".into())
        })?;
        if encoded_bytes.len() != stored_bytes {
            return Err(AkitaError::InvalidSize {
                expected: stored_bytes,
                actual: encoded_bytes.len(),
            });
        }
        let packed_view = crate::sources::packed_digits::PackedSignedDigitView::from_encoded(
            encoded_bytes,
            live_coefficient_len,
            stored_coefficient_len,
            physical_coefficient_len,
            signed_bit_width,
            negative_abs_max,
            positive_max,
        )?;
        Ok(Self {
            encoded_bytes,
            live_coefficient_len,
            physical_coefficient_len,
            signed_bit_width,
            packed_view,
        })
    }

    /// Borrow Akita-owned packed digits whose exact bounds were established
    /// while their immutable storage was built.
    pub(crate) fn from_validated_packed(
        digits: &'a crate::sources::packed_digits::PackedSignedDigits,
        live_coefficient_len: usize,
        physical_coefficient_len: usize,
    ) -> Result<Self, AkitaError> {
        let stored_coefficient_len = digits.len();
        if live_coefficient_len == 0
            || live_coefficient_len > stored_coefficient_len
            || stored_coefficient_len > physical_coefficient_len
        {
            return Err(AkitaError::InvalidInput(
                "packed short-norm extents are inconsistent".into(),
            ));
        }
        Ok(Self {
            encoded_bytes: digits.encoded_bytes(),
            live_coefficient_len,
            physical_coefficient_len,
            signed_bit_width: digits.bit_width(),
            packed_view: digits.zero_padded(physical_coefficient_len)?,
        })
    }

    /// Encoded two's-complement payload, excluding safe-load padding.
    pub(crate) const fn encoded_bytes(&self) -> &'a [u8] {
        self.encoded_bytes
    }

    /// Number of source-owned live coefficients.
    pub(crate) const fn live_coefficient_len(&self) -> usize {
        self.live_coefficient_len
    }

    /// Commitment-aligned logical coefficient extent.
    pub(crate) const fn physical_coefficient_len(&self) -> usize {
        self.physical_coefficient_len
    }

    /// Stored two's-complement bit width.
    pub(crate) const fn signed_bit_width(&self) -> u8 {
        self.signed_bit_width
    }

    pub(crate) const fn packed_view(
        &self,
    ) -> crate::sources::packed_digits::PackedSignedDigitView<'a> {
        self.packed_view
    }
}

/// Borrowed one-hot chunk indices at their stored width.
/// `None` denotes an all-zero chunk.
pub(crate) enum UnitPositionSlice<'a> {
    /// Eight-bit positions.
    U8(&'a [Option<u8>]),
    /// Sixteen-bit positions.
    U16(&'a [Option<u16>]),
    /// Thirty-two-bit positions.
    U32(&'a [Option<u32>]),
    /// Native-word positions.
    Usize(&'a [Option<usize>]),
}

impl UnitPositionSlice<'_> {
    /// Number of represented one-hot chunks.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::Usize(values) => values.len(),
        }
    }

    fn contains_out_of_range_position(&self, chunk_size: usize) -> bool {
        match self {
            Self::U8(values) => values
                .iter()
                .flatten()
                .any(|position| usize::from(*position) >= chunk_size),
            Self::U16(values) => values
                .iter()
                .flatten()
                .any(|position| usize::from(*position) >= chunk_size),
            Self::U32(values) => values
                .iter()
                .flatten()
                .any(|position| usize::try_from(*position).map_or(true, |p| p >= chunk_size)),
            Self::Usize(values) => values
                .iter()
                .flatten()
                .any(|position| *position >= chunk_size),
        }
    }
}

/// Borrowed one-hot source representation.
pub(crate) struct OneHotRepresentation<'a> {
    /// Complete source-owned position slice, preserving index width.
    pub positions: UnitPositionSlice<'a>,
    /// Logical coefficients per one-hot chunk.
    pub chunk_size: usize,
    /// Logical multilinear variable count.
    pub num_vars: usize,
}

/// Materialized representation selected by request compilation.
pub(crate) enum PolynomialRepresentation<'a, F: Field> {
    /// Dense coefficients or predecomposed digit planes.
    Dense(DenseRepresentation<'a, F>),
    /// Packed bounded signed coefficients.
    ShortNorm(ShortNormRepresentation<'a>),
    /// Structurally one-hot positions.
    OneHot(OneHotRepresentation<'a>),
}

impl<F: Field> PolynomialRepresentation<'_, F> {
    fn polynomial_type(&self) -> Result<PolynomialType, AkitaError> {
        match self {
            Self::Dense(DenseRepresentation::Coefficients(_)) => {
                Ok(PolynomialType::Dense(DenseType::Coefficients))
            }
            Self::Dense(DenseRepresentation::PredecomposedDigits(_)) => {
                Ok(PolynomialType::Dense(DenseType::PredecomposedDigits))
            }
            Self::ShortNorm(representation) => Ok(PolynomialType::ShortNorm(ShortNormType::new(
                representation.signed_bit_width(),
            )?)),
            Self::OneHot(representation) => {
                let width = match representation.positions {
                    UnitPositionSlice::U8(_) => OneHotIndexWidth::U8,
                    UnitPositionSlice::U16(_) => OneHotIndexWidth::U16,
                    UnitPositionSlice::U32(_) => OneHotIndexWidth::U32,
                    UnitPositionSlice::Usize(_) => OneHotIndexWidth::Usize,
                };
                Ok(PolynomialType::OneHot(OneHotType::new(
                    representation.chunk_size,
                    width,
                )?))
            }
        }
    }
}

/// Ring-dimension-free source contract for commitment execution.
pub(crate) trait CommitmentSource<F: Field>: Send + Sync {
    /// Return O(1) source metadata without materializing a representation.
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError>;

    /// Return exact centered negative and positive reach for admission.
    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding;

    /// Declare available lossless representations without building lazy state.
    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError>;

    /// Borrow or materialize only the representation selected by compilation.
    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError>;

    /// Declare a source-owned operation for the requested backend family.
    fn external_inner_commitment_capability(
        &self,
        _backend: BackendKindId,
        _plan: &CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        Ok(None)
    }

    /// Prepare only the external operation selected by request compilation.
    fn prepare_external_inner_commitment(
        &self,
        _selected: ExternalInnerCommitmentCapability,
        _plan: &CommitInnerPlan,
    ) -> Result<PreparedExternalInnerCommitment<'_, F>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "source did not provide the selected external inner commitment capability".into(),
        ))
    }
}

impl<F: Field, S: CommitmentSource<F> + ?Sized> CommitmentSource<F> for &S {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        (**self).descriptor()
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        (**self).committed_centered_reach(modulus, centering_threshold)
    }

    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        (**self).available_polynomial_types(plan)
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        (**self).represent_as(selected, plan)
    }

    fn external_inner_commitment_capability(
        &self,
        backend: BackendKindId,
        plan: &CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        (**self).external_inner_commitment_capability(backend, plan)
    }

    fn prepare_external_inner_commitment(
        &self,
        selected: ExternalInnerCommitmentCapability,
        plan: &CommitInnerPlan,
    ) -> Result<PreparedExternalInnerCommitment<'_, F>, AkitaError> {
        (**self).prepare_external_inner_commitment(selected, plan)
    }
}

enum ResolvedCommitSourcePath<'a, F: Field> {
    Standard {
        selected: PolynomialType,
        representation: PolynomialRepresentation<'a, F>,
    },
    External(PreparedExternalInnerCommitment<'a, F>),
}

/// One admitted source with its compiled representation choice.
pub(crate) struct ResolvedCommitSource<'a, F: Field> {
    descriptor: CommitSourceDescriptor,
    inner_plan: CommitInnerPlan,
    path: ResolvedCommitSourcePath<'a, F>,
}

impl<'a, F: Field> ResolvedCommitSource<'a, F> {
    /// Validated source metadata.
    pub(crate) const fn descriptor(&self) -> &CommitSourceDescriptor {
        &self.descriptor
    }

    /// Compiled polynomial representation type.
    pub(crate) const fn selected_type(&self) -> Option<PolynomialType> {
        match self.path {
            ResolvedCommitSourcePath::Standard { selected, .. } => Some(selected),
            ResolvedCommitSourcePath::External(_) => None,
        }
    }

    /// Borrow the selected representation.
    pub(crate) const fn representation(&self) -> Option<&PolynomialRepresentation<'a, F>> {
        match &self.path {
            ResolvedCommitSourcePath::Standard { representation, .. } => Some(representation),
            ResolvedCommitSourcePath::External(_) => None,
        }
    }

    /// Borrow the prepared external path when one was selected.
    pub(crate) const fn external(&self) -> Option<&PreparedExternalInnerCommitment<'a, F>> {
        match &self.path {
            ResolvedCommitSourcePath::Standard { .. } => None,
            ResolvedCommitSourcePath::External(prepared) => Some(prepared),
        }
    }

    pub(crate) fn validate_plan(&self, plan: &CommitInnerPlan) -> Result<(), AkitaError> {
        if self.inner_plan != *plan {
            return Err(AkitaError::InvalidInput(
                "resolved commitment source was compiled for a different inner plan".into(),
            ));
        }
        Ok(())
    }
}

mod builtins;

#[cfg(test)]
mod tests;
