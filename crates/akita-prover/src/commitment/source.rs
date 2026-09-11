use super::{
    BackendKindId, CommitmentRequestCapabilities, ExternalInnerCommitmentCapability,
    PreparedExternalInnerCommitment,
};
use crate::compute::CommitInnerPlan;
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, Field};

/// Admission class of a commitment source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommitSourceClass {
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
pub struct CommitSourceDescriptor {
    num_vars: usize,
    total_coefficient_len: usize,
    live_coefficient_len: usize,
    class: CommitSourceClass,
    family_name: &'static str,
}

impl CommitSourceDescriptor {
    /// Construct a descriptor after checking its extents.
    pub fn new(
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
    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Physical coefficient extent, including commitment padding.
    pub const fn total_coefficient_len(&self) -> usize {
        self.total_coefficient_len
    }

    /// Source-owned live coefficient extent.
    pub const fn live_coefficient_len(&self) -> usize {
        self.live_coefficient_len
    }

    /// Admission class used by the selected schedule contract.
    pub const fn class(&self) -> CommitSourceClass {
        self.class
    }

    /// Stable diagnostic family name. This is never a type identity.
    pub const fn family_name(&self) -> &'static str {
        self.family_name
    }
}

/// Physical dense representation offered by a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseType {
    /// Borrowed canonical field coefficients.
    Coefficients,
    /// Borrowed exact-plan balanced digit planes.
    PredecomposedDigits,
}

/// Packed short-norm source representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShortNormType {
    signed_bit_width: u8,
}

impl ShortNormType {
    /// Describe a checked packed signed width.
    pub fn new(signed_bit_width: u8) -> Result<Self, AkitaError> {
        if !(1..=8).contains(&signed_bit_width) {
            return Err(AkitaError::InvalidInput(format!(
                "packed short-norm signed width {signed_bit_width} is outside 1..=8"
            )));
        }
        Ok(Self { signed_bit_width })
    }

    /// Stored two's-complement bit width.
    pub const fn signed_bit_width(self) -> u8 {
        self.signed_bit_width
    }
}

/// Width of the source-owned one-hot position indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OneHotIndexWidth {
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
pub struct OneHotType {
    chunk_size: usize,
    index_width: OneHotIndexWidth,
}

impl OneHotType {
    /// Construct a checked one-hot type.
    pub fn new(chunk_size: usize, index_width: OneHotIndexWidth) -> Result<Self, AkitaError> {
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

    /// Logical coefficient count per source chunk.
    pub const fn chunk_size(self) -> usize {
        self.chunk_size
    }

    /// Stored position-index width.
    pub const fn index_width(self) -> OneHotIndexWidth {
        self.index_width
    }
}

/// Akita-owned semantic polynomial types understood by commitment operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolynomialType {
    /// General dense coefficients or exact-plan cached digit planes.
    Dense(DenseType),
    /// Packed bounded signed coefficients.
    ShortNorm(ShortNormType),
    /// Structurally one-hot coefficients.
    OneHot(OneHotType),
}

/// Side-effect-free representation offerings for one source and plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailablePolynomialTypes {
    entries: Vec<PolynomialType>,
}

impl AvailablePolynomialTypes {
    /// Construct a duplicate-free offering list.
    ///
    /// An empty list is valid for a source that supports only an external
    /// inner commitment operation.
    pub fn new(entries: Vec<PolynomialType>) -> Result<Self, AkitaError> {
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
    pub fn as_slice(&self) -> &[PolynomialType] {
        &self.entries
    }

    /// Issue an opaque selection for one advertised entry.
    pub fn select(&self, index: usize) -> Result<PolynomialTypeSelection, AkitaError> {
        let selected = self.entries.get(index).copied().ok_or_else(|| {
            AkitaError::InvalidInput("polynomial representation selection is out of range".into())
        })?;
        Ok(PolynomialTypeSelection { selected })
    }
}

/// Opaque representation choice issued after capability intersection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolynomialTypeSelection {
    selected: PolynomialType,
}

#[derive(Clone, Copy)]
enum PendingCommitGroupPath {
    Standard(PolynomialType),
    External(ExternalInnerCommitmentCapability),
}

/// Validated source-path selections that have not materialized source data.
pub struct CompiledCommitmentRequest<'a, F: Field> {
    plan: CommitInnerPlan,
    sources: &'a [&'a dyn CommitmentSource<F>],
    descriptors: Vec<CommitSourceDescriptor>,
    path: PendingCommitGroupPath,
}

impl<'a, F: Field> CompiledCommitmentRequest<'a, F> {
    /// Selected standard types; external paths own their resource requirements.
    pub fn selected_polynomial_types(&self) -> impl Iterator<Item = PolynomialType> + '_ {
        let selected = match self.path {
            PendingCommitGroupPath::Standard(selected) => Some(selected),
            PendingCommitGroupPath::External(_) => None,
        };
        selected.into_iter()
    }

    /// Materialize only the representations selected during request compilation.
    pub fn materialize(self) -> Result<Vec<ResolvedCommitSource<'a, F>>, AkitaError> {
        self.sources
            .iter()
            .zip(self.descriptors)
            .map(|(source, descriptor)| {
                let path = match self.path {
                    PendingCommitGroupPath::Standard(selected) => {
                        let selection = PolynomialTypeSelection { selected };
                        let selected = selection.polynomial_type();
                        let representation = source.represent_as(selection, &self.plan)?;
                        validate_materialized_representation(
                            &descriptor,
                            &self.plan,
                            selected,
                            &representation,
                        )?;
                        ResolvedCommitSourcePath::Standard {
                            selected,
                            representation,
                        }
                    }
                    PendingCommitGroupPath::External(capability) => {
                        let prepared =
                            source.prepare_external_inner_commitment(capability, &self.plan)?;
                        if prepared.capability() != capability {
                            return Err(AkitaError::InvalidInput(
                                "source prepared a different external commitment capability".into(),
                            ));
                        }
                        ResolvedCommitSourcePath::External(prepared)
                    }
                };
                Ok(ResolvedCommitSource {
                    descriptor,
                    inner_plan: self.plan,
                    path,
                })
            })
            .collect()
    }
}

impl PolynomialTypeSelection {
    /// Selected semantic and physical representation.
    pub const fn polynomial_type(self) -> PolynomialType {
        self.selected
    }
}

/// Borrowed source of canonical dense coefficients.
pub trait DenseCoefficientSource<F: Field>: Send + Sync {
    /// Physical coefficient buffer in canonical source order.
    fn coefficients(&self) -> &[F];
}

/// Borrowed dense source representation.
pub enum DenseRepresentation<'a, F: Field> {
    /// Canonical field coefficients.
    Coefficients(&'a dyn DenseCoefficientSource<F>),
    /// Exact-plan balanced digit planes in `[ring][digit][coefficient]` order.
    PredecomposedDigits(PredecomposedDigitPlanes<'a>),
}

/// Checked borrowed dense digit planes.
pub struct PredecomposedDigitPlanes<'a> {
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
pub struct ShortNormRepresentation<'a> {
    /// Encoded two's-complement payload, excluding safe-load padding.
    pub encoded_bytes: &'a [u8],
    /// Number of source-owned live coefficients.
    pub live_coefficient_len: usize,
    /// Commitment-aligned logical coefficient extent.
    pub physical_coefficient_len: usize,
    /// Stored two's-complement bit width.
    pub signed_bit_width: u8,
    /// Exact largest negative magnitude.
    pub negative_abs_max: u8,
    /// Exact largest positive value.
    pub positive_max: u8,
    pub(crate) packed_view: Option<crate::backend::packed_digits::PackedSignedDigitView<'a>>,
}

impl<'a> ShortNormRepresentation<'a> {
    /// Construct a portable borrowed packed representation.
    pub fn new(
        encoded_bytes: &'a [u8],
        live_coefficient_len: usize,
        physical_coefficient_len: usize,
        signed_bit_width: u8,
        negative_abs_max: u8,
        positive_max: u8,
    ) -> Result<Self, AkitaError> {
        ShortNormType::new(signed_bit_width)?;
        if live_coefficient_len == 0 || live_coefficient_len > physical_coefficient_len {
            return Err(AkitaError::InvalidInput(
                "packed short-norm extents are inconsistent".into(),
            ));
        }
        let minimum_bits = checked::product([live_coefficient_len, usize::from(signed_bit_width)])
            .ok_or_else(|| {
                AkitaError::InvalidInput("packed short-norm bit length overflow".into())
            })?;
        let maximum_bits =
            checked::product([physical_coefficient_len, usize::from(signed_bit_width)])
                .ok_or_else(|| {
                    AkitaError::InvalidInput("packed short-norm bit length overflow".into())
                })?;
        let minimum_bytes = checked::div_ceil(minimum_bits, 8).ok_or_else(|| {
            AkitaError::InvalidInput("packed short-norm byte length overflow".into())
        })?;
        let maximum_bytes = checked::div_ceil(maximum_bits, 8).ok_or_else(|| {
            AkitaError::InvalidInput("packed short-norm byte length overflow".into())
        })?;
        if !(minimum_bytes..=maximum_bytes).contains(&encoded_bytes.len()) {
            return Err(AkitaError::InvalidSize {
                expected: maximum_bytes,
                actual: encoded_bytes.len(),
            });
        }
        let packed_view = Some(
            crate::backend::packed_digits::PackedSignedDigitView::from_encoded(
                encoded_bytes,
                live_coefficient_len,
                physical_coefficient_len,
                signed_bit_width,
                negative_abs_max,
                positive_max,
            )?,
        );
        Ok(Self {
            encoded_bytes,
            live_coefficient_len,
            physical_coefficient_len,
            signed_bit_width,
            negative_abs_max,
            positive_max,
            packed_view,
        })
    }
}

/// Borrowed one-hot chunk indices at their stored width.
/// `None` denotes an all-zero chunk.
pub enum UnitPositionSlice<'a> {
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
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::Usize(values) => values.len(),
        }
    }

    /// Whether no chunks are represented.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
pub struct OneHotRepresentation<'a> {
    /// Complete source-owned position slice, preserving index width.
    pub positions: UnitPositionSlice<'a>,
    /// Logical coefficients per one-hot chunk.
    pub chunk_size: usize,
    /// Logical multilinear variable count.
    pub num_vars: usize,
}

/// Materialized representation selected by request compilation.
pub enum PolynomialRepresentation<'a, F: Field> {
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
                representation.signed_bit_width,
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
pub trait CommitmentSource<F: Field>: Send + Sync {
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

enum ResolvedCommitSourcePath<'a, F: Field> {
    Standard {
        selected: PolynomialType,
        representation: PolynomialRepresentation<'a, F>,
    },
    External(PreparedExternalInnerCommitment<'a, F>),
}

/// One admitted source with its compiled representation choice.
pub struct ResolvedCommitSource<'a, F: Field> {
    descriptor: CommitSourceDescriptor,
    inner_plan: CommitInnerPlan,
    path: ResolvedCommitSourcePath<'a, F>,
}

impl<'a, F: Field> ResolvedCommitSource<'a, F> {
    /// Validated source metadata.
    pub const fn descriptor(&self) -> &CommitSourceDescriptor {
        &self.descriptor
    }

    /// Compiled polynomial representation type.
    pub const fn selected_type(&self) -> Option<PolynomialType> {
        match self.path {
            ResolvedCommitSourcePath::Standard { selected, .. } => Some(selected),
            ResolvedCommitSourcePath::External(_) => None,
        }
    }

    /// Borrow the selected representation.
    pub const fn representation(&self) -> Option<&PolynomialRepresentation<'a, F>> {
        match &self.path {
            ResolvedCommitSourcePath::Standard { representation, .. } => Some(representation),
            ResolvedCommitSourcePath::External(_) => None,
        }
    }

    /// Borrow the prepared external path when one was selected.
    pub const fn external(&self) -> Option<&PreparedExternalInnerCommitment<'a, F>> {
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

fn validate_materialized_representation<F: Field>(
    descriptor: &CommitSourceDescriptor,
    plan: &CommitInnerPlan,
    selected: PolynomialType,
    representation: &PolynomialRepresentation<'_, F>,
) -> Result<(), AkitaError> {
    if representation.polynomial_type()? != selected {
        return Err(AkitaError::InvalidInput(
            "commit source materialized a representation other than the compiled selection".into(),
        ));
    }
    match representation {
        PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(source)) => {
            if source.coefficients().len() != descriptor.total_coefficient_len() {
                return Err(AkitaError::InvalidSize {
                    expected: descriptor.total_coefficient_len(),
                    actual: source.coefficients().len(),
                });
            }
        }
        PolynomialRepresentation::Dense(DenseRepresentation::PredecomposedDigits(planes)) => {
            let logical_len = checked_logical_len(descriptor.num_vars())?;
            let expected_rings = logical_len.div_ceil(plan.ring_dimension);
            let expected_bytes =
                checked::product([expected_rings, plan.num_digits_inner, plan.ring_dimension])
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("dense digit-plane length overflow".into())
                    })?;
            if planes.ring_dimension != plan.ring_dimension
                || planes.num_digits != plan.num_digits_inner
                || planes.log_basis != plan.log_basis_inner
                || planes.logical_ring_count != expected_rings
                || planes.bytes.len() != expected_bytes
            {
                return Err(AkitaError::InvalidInput(
                    "predecomposed dense planes do not match the compiled inner plan".into(),
                ));
            }
        }
        PolynomialRepresentation::ShortNorm(representation) => {
            if representation.live_coefficient_len != descriptor.live_coefficient_len()
                || representation.physical_coefficient_len != descriptor.total_coefficient_len()
                || representation.encoded_bytes.is_empty()
            {
                return Err(AkitaError::InvalidInput(
                    "packed short-norm representation disagrees with its source descriptor".into(),
                ));
            }
        }
        PolynomialRepresentation::OneHot(representation) => {
            let logical_len = checked_logical_len(representation.num_vars)?;
            if representation.chunk_size == 0
                || !logical_len.is_multiple_of(representation.chunk_size)
            {
                return Err(AkitaError::InvalidInput(
                    "one-hot chunk size must exactly divide the logical coefficient count".into(),
                ));
            }
            let expected_positions = logical_len / representation.chunk_size;
            if representation.num_vars != descriptor.num_vars()
                || descriptor.class()
                    != (CommitSourceClass::OneHot {
                        chunk_size: representation.chunk_size,
                    })
                || representation.positions.len() != expected_positions
            {
                return Err(AkitaError::InvalidInput(
                    "one-hot representation disagrees with its source descriptor".into(),
                ));
            }
            if representation
                .positions
                .contains_out_of_range_position(representation.chunk_size)
            {
                return Err(AkitaError::InvalidInput(
                    "one-hot representation contains a position outside its chunk".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Compile source paths for one checked inner plan.
///
/// Discovery and capability intersection complete for the entire source group
/// before any standard or external representation is materialized.
pub fn compile_commitment_request<'a, F: Field>(
    plan: &CommitInnerPlan,
    sources: &'a [&'a dyn CommitmentSource<F>],
    capabilities: &CommitmentRequestCapabilities,
) -> Result<CompiledCommitmentRequest<'a, F>, AkitaError> {
    if sources.is_empty() {
        return Err(AkitaError::InvalidInput(
            "commitment request requires at least one source".into(),
        ));
    }

    let mut candidates = Vec::with_capacity(sources.len());
    let mut expected_num_vars = None;
    for source in sources {
        let descriptor = source.descriptor()?;
        validate_plan_extents(&descriptor, plan)?;
        if expected_num_vars
            .replace(descriptor.num_vars())
            .is_some_and(|expected| expected != descriptor.num_vars())
        {
            return Err(AkitaError::InvalidInput(
                "all commitment sources must have the same num_vars".into(),
            ));
        }
        let available = source.available_polynomial_types(plan)?;
        let external = source.external_inner_commitment_capability(capabilities.backend, plan)?;
        if let Some(external) = external {
            if external.backend() != capabilities.backend
                || external.context_type_id() != capabilities.external_context
                || (capabilities.fused_command_context.is_some()
                    && external.fused_command_context_type_id()
                        != capabilities.fused_command_context)
            {
                return Err(AkitaError::InvalidInput(
                    "external commitment capability is incoherent with the selected operation"
                        .into(),
                ));
            }
        }
        candidates.push((descriptor, available, external));
    }

    let homogeneous_external = candidates
        .first()
        .and_then(|(_, _, external)| *external)
        .filter(|selected| {
            candidates
                .iter()
                .all(|(_, _, external)| *external == Some(*selected))
        });
    let path = if let Some(external) = homogeneous_external {
        PendingCommitGroupPath::External(external)
    } else {
        let common =
            if capabilities.standard_types.is_empty() && capabilities.accepts_any_standard_type {
                candidates.first().and_then(|(_, available, _)| {
                    available.as_slice().iter().copied().find(|offered| {
                        candidates
                            .iter()
                            .all(|(_, available, _)| available.as_slice().contains(offered))
                    })
                })
            } else {
                capabilities
                    .standard_types
                    .iter()
                    .copied()
                    .find(|supported| {
                        candidates
                            .iter()
                            .all(|(_, available, _)| available.as_slice().contains(supported))
                    })
            };
        let Some(selected) = common else {
            return Err(AkitaError::InvalidInput(
                "commitment source group has neither one common external path nor one common standard representation for the selected inner operation".into(),
            ));
        };
        PendingCommitGroupPath::Standard(selected)
    };

    let descriptors = candidates
        .into_iter()
        .map(|(descriptor, _, _)| descriptor)
        .collect();

    Ok(CompiledCommitmentRequest {
        plan: *plan,
        sources,
        descriptors,
        path,
    })
}

fn checked_logical_len(num_vars: usize) -> Result<usize, AkitaError> {
    let shift = u32::try_from(num_vars).map_err(|_| {
        AkitaError::InvalidInput(format!("commit source arity {num_vars} exceeds u32"))
    })?;
    1usize.checked_shl(shift).ok_or_else(|| {
        AkitaError::InvalidInput(format!("commit source arity 2^{num_vars} overflows usize"))
    })
}

fn validate_plan_extents(
    descriptor: &CommitSourceDescriptor,
    plan: &CommitInnerPlan,
) -> Result<(), AkitaError> {
    if plan.ring_dimension == 0
        || !plan.ring_dimension.is_power_of_two()
        || plan.num_positions_per_block == 0
        || !plan.num_positions_per_block.is_power_of_two()
        || plan.num_digits_inner == 0
        || plan.n_a == 0
    {
        return Err(AkitaError::InvalidInput(
            "inner commitment plan has invalid ring, block, digit, or row geometry".into(),
        ));
    }
    let _logical_len = checked_logical_len(descriptor.num_vars)?;
    let ring_count = descriptor
        .live_coefficient_len
        .div_ceil(plan.ring_dimension);
    let physical_len = ring_count
        .checked_mul(plan.ring_dimension)
        .ok_or_else(|| AkitaError::InvalidInput("commit source physical extent overflow".into()))?;
    let live_blocks = ring_count.div_ceil(plan.num_positions_per_block);
    if physical_len > descriptor.total_coefficient_len || live_blocks != plan.num_live_blocks {
        return Err(AkitaError::InvalidInput(format!(
            "commit source does not match plan extents: physical={physical_len}/{}, live_blocks={live_blocks}/{}",
            descriptor.total_coefficient_len, plan.num_live_blocks
        )));
    }
    Ok(())
}

impl<F: Field> DenseCoefficientSource<F> for crate::DensePoly<F> {
    fn coefficients(&self) -> &[F] {
        self.field_coeffs()
    }
}

impl<F: Field> CommitmentSource<F> for crate::DensePoly<F> {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        let num_vars = crate::compute::RootPolyMeta::<F>::num_vars(self);
        CommitSourceDescriptor::new(
            num_vars,
            self.field_coeffs().len(),
            checked_logical_len(num_vars)?,
            CommitSourceClass::Dense,
            "akita_dense",
        )
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        Ok(crate::compute::centered_reach_of_field_coeffs(
            self.field_coeffs(),
            modulus,
            centering_threshold,
        ))
    }

    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        validate_plan_extents(&<Self as CommitmentSource<F>>::descriptor(self)?, plan)?;
        let mut available = Vec::with_capacity(2);
        if self
            .cached_digit_parts(
                plan.ring_dimension,
                plan.num_digits_inner,
                plan.log_basis_inner,
            )
            .is_some()
        {
            available.push(PolynomialType::Dense(DenseType::PredecomposedDigits));
        }
        available.push(PolynomialType::Dense(DenseType::Coefficients));
        AvailablePolynomialTypes::new(available)
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        validate_plan_extents(&self.descriptor()?, plan)?;
        match selected.polynomial_type() {
            PolynomialType::Dense(DenseType::Coefficients) => Ok(PolynomialRepresentation::Dense(
                DenseRepresentation::Coefficients(self),
            )),
            PolynomialType::Dense(DenseType::PredecomposedDigits) => {
                let bytes = self
                    .cached_digit_parts(
                        plan.ring_dimension,
                        plan.num_digits_inner,
                        plan.log_basis_inner,
                    )
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "selected dense digit cache is no longer available".into(),
                        )
                    })?;
                let logical_len = checked_logical_len(self.descriptor()?.num_vars())?;
                Ok(PolynomialRepresentation::Dense(
                    DenseRepresentation::PredecomposedDigits(PredecomposedDigitPlanes {
                        bytes,
                        ring_dimension: plan.ring_dimension,
                        num_digits: plan.num_digits_inner,
                        log_basis: plan.log_basis_inner,
                        logical_ring_count: logical_len.div_ceil(plan.ring_dimension),
                    }),
                ))
            }
            _ => Err(AkitaError::InvalidInput(
                "dense source received an unadvertised representation selection".into(),
            )),
        }
    }
}

macro_rules! impl_onehot_commitment_source {
    ($index:ty, $width:ident, $variant:ident) => {
        impl<F: Field> CommitmentSource<F> for crate::OneHotPoly<F, $index> {
            fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
                let num_vars = crate::compute::RootPolyMeta::<F>::num_vars(self);
                let logical_len = checked_logical_len(num_vars)?;
                CommitSourceDescriptor::new(
                    num_vars,
                    logical_len,
                    logical_len,
                    CommitSourceClass::OneHot {
                        chunk_size: self.onehot_k(),
                    },
                    "akita_onehot",
                )
            }

            fn committed_centered_reach(
                &self,
                _modulus: u128,
                _centering_threshold: u128,
            ) -> Result<(u128, u128), AkitaError>
            where
                F: CanonicalEncoding,
            {
                Ok((0, 1))
            }

            fn available_polynomial_types(
                &self,
                plan: &CommitInnerPlan,
            ) -> Result<AvailablePolynomialTypes, AkitaError> {
                validate_plan_extents(&self.descriptor()?, plan)?;
                self.validate_ring_dimension(plan.ring_dimension)?;
                AvailablePolynomialTypes::new(vec![PolynomialType::OneHot(OneHotType::new(
                    self.onehot_k(),
                    OneHotIndexWidth::$width,
                )?)])
            }

            fn represent_as(
                &self,
                selected: PolynomialTypeSelection,
                plan: &CommitInnerPlan,
            ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
                let available = self.available_polynomial_types(plan)?;
                if available.as_slice().first().copied() != Some(selected.polynomial_type()) {
                    return Err(AkitaError::InvalidInput(
                        "one-hot source received an unadvertised representation selection".into(),
                    ));
                }
                Ok(PolynomialRepresentation::OneHot(OneHotRepresentation {
                    positions: UnitPositionSlice::$variant(self.indices()),
                    chunk_size: self.onehot_k(),
                    num_vars: crate::compute::RootPolyMeta::<F>::num_vars(self),
                }))
            }
        }
    };
}

impl_onehot_commitment_source!(u8, U8, U8);
impl_onehot_commitment_source!(u16, U16, U16);
impl_onehot_commitment_source!(u32, U32, U32);
impl_onehot_commitment_source!(usize, Usize, Usize);

impl<F: Field> CommitmentSource<F> for crate::RecursiveWitnessFlat {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        let logical_len = self
            .live_coeff_len()
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(|| {
                AkitaError::InvalidInput("recursive witness logical extent overflows usize".into())
            })?;
        let num_vars = logical_len.trailing_zeros() as usize;
        CommitSourceDescriptor::new(
            num_vars,
            self.commitment_physical_len()?,
            self.live_coeff_len(),
            CommitSourceClass::ShortNorm,
            "akita_recursive_packed",
        )
    }

    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        let (_, _, negative_abs_max, positive_max) = self.packed_representation_parts();
        Ok((u128::from(negative_abs_max), u128::from(positive_max)))
    }

    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        validate_plan_extents(&<Self as CommitmentSource<F>>::descriptor(self)?, plan)?;
        let (_, signed_bit_width, _, _) = self.packed_representation_parts();
        AvailablePolynomialTypes::new(vec![PolynomialType::ShortNorm(ShortNormType::new(
            signed_bit_width,
        )?)])
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        let available = <Self as CommitmentSource<F>>::available_polynomial_types(self, plan)?;
        if available.as_slice().first().copied() != Some(selected.polynomial_type()) {
            return Err(AkitaError::InvalidInput(
                "short-norm source received an unadvertised representation selection".into(),
            ));
        }
        let (encoded_bytes, signed_bit_width, negative_abs_max, positive_max) =
            self.packed_representation_parts();
        let physical_coefficient_len =
            <Self as CommitmentSource<F>>::descriptor(self)?.total_coefficient_len();
        let mut representation = ShortNormRepresentation::new(
            encoded_bytes,
            self.live_coeff_len(),
            physical_coefficient_len,
            signed_bit_width,
            negative_abs_max,
            positive_max,
        )?;
        representation.packed_view = Some(self.packed_commitment_view(physical_coefficient_len)?);
        Ok(PolynomialRepresentation::ShortNorm(representation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{
        ExternalInnerCommitmentInput, ExternalInnerCommitmentOperation, ExternalOperationIdentity,
    };
    use crate::{DensePoly, OneHotPoly};
    use jolt_field::{Fp64, Ring};
    use std::sync::atomic::{AtomicUsize, Ordering};

    type F = Fp64<4294967197>;

    fn plan() -> CommitInnerPlan {
        CommitInnerPlan {
            ring_dimension: 64,
            num_live_blocks: 1,
            n_a: 2,
            num_positions_per_block: 1,
            num_digits_inner: 11,
            log_basis_inner: 3,
        }
    }

    #[test]
    fn dense_source_borrows_coefficients_without_copying() {
        let poly = DensePoly::<F>::from_field_evals(6, vec![F::from_u64(1); 64]).unwrap();
        let descriptor = poly.descriptor().unwrap();
        assert_eq!(descriptor.num_vars(), 6);
        assert_eq!(descriptor.live_coefficient_len(), 64);
        assert_eq!(descriptor.class(), CommitSourceClass::Dense);

        let available = poly.available_polynomial_types(&plan()).unwrap();
        assert_eq!(
            available.as_slice(),
            &[PolynomialType::Dense(DenseType::Coefficients)]
        );
        let selected = available.select(0).unwrap();
        let PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(source)) =
            poly.represent_as(selected, &plan()).unwrap()
        else {
            panic!("dense source selected the wrong physical representation");
        };
        assert!(std::ptr::eq(
            source.coefficients().as_ptr(),
            poly.field_coeffs().as_ptr()
        ));
    }

    macro_rules! assert_onehot_width {
        ($index:ty, $width:ident, $variant:ident) => {{
            let poly = OneHotPoly::<F, $index>::new(8, vec![None; 8]).unwrap();
            let available = poly.available_polynomial_types(&plan()).unwrap();
            assert_eq!(
                available.as_slice(),
                &[PolynomialType::OneHot(
                    OneHotType::new(8, OneHotIndexWidth::$width).unwrap()
                )]
            );
            let selected = available.select(0).unwrap();
            let PolynomialRepresentation::OneHot(representation) =
                poly.represent_as(selected, &plan()).unwrap()
            else {
                panic!("one-hot source selected the wrong representation");
            };
            let UnitPositionSlice::$variant(positions) = representation.positions else {
                panic!("one-hot source widened its stored index type");
            };
            assert_eq!(positions, &[None; 8]);
        }};
    }

    #[test]
    fn onehot_source_preserves_every_stored_index_width_and_none() {
        assert_onehot_width!(u8, U8, U8);
        assert_onehot_width!(u16, U16, U16);
        assert_onehot_width!(u32, U32, U32);
        assert_onehot_width!(usize, Usize, Usize);
    }

    #[test]
    fn source_rejects_plan_extent_mismatch_before_materialization() {
        let poly = DensePoly::<F>::from_field_evals(6, vec![F::from_u64(1); 64]).unwrap();
        let mut wrong = plan();
        wrong.num_live_blocks = 2;
        assert!(matches!(
            poly.available_polynomial_types(&wrong),
            Err(AkitaError::InvalidInput(_))
        ));
    }

    #[test]
    fn recursive_source_exposes_packed_short_norm_without_decoding() {
        let witness = crate::RecursiveWitnessFlat::from_i8_digits(vec![-2, 0, 3, 1])
            .align_for_commitment_ring_dim(64)
            .unwrap();
        let recursive_plan = CommitInnerPlan {
            ring_dimension: 64,
            num_live_blocks: 1,
            n_a: 2,
            num_positions_per_block: 1,
            num_digits_inner: 2,
            log_basis_inner: 3,
        };
        let available =
            <crate::RecursiveWitnessFlat as CommitmentSource<F>>::available_polynomial_types(
                &witness,
                &recursive_plan,
            )
            .unwrap();
        let selected = available.select(0).unwrap();
        let PolynomialRepresentation::ShortNorm(representation) =
            <crate::RecursiveWitnessFlat as CommitmentSource<F>>::represent_as(
                &witness,
                selected,
                &recursive_plan,
            )
            .unwrap()
        else {
            panic!("recursive witness selected the wrong representation");
        };
        assert_eq!(representation.live_coefficient_len, 4);
        assert_eq!(representation.physical_coefficient_len, 64);
        assert_eq!(representation.negative_abs_max, 2);
        assert_eq!(representation.positive_max, 3);
        assert!(!representation.encoded_bytes.is_empty());
    }

    struct CountingSource {
        onehot: bool,
        coefficients: Vec<F>,
        positions: Vec<Option<u8>>,
        materializations: AtomicUsize,
    }

    impl DenseCoefficientSource<F> for CountingSource {
        fn coefficients(&self) -> &[F] {
            &self.coefficients
        }
    }

    impl CommitmentSource<F> for CountingSource {
        fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
            CommitSourceDescriptor::new(
                6,
                64,
                64,
                if self.onehot {
                    CommitSourceClass::OneHot { chunk_size: 8 }
                } else {
                    CommitSourceClass::Dense
                },
                "counting_test_source",
            )
        }

        fn committed_centered_reach(
            &self,
            _modulus: u128,
            _centering_threshold: u128,
        ) -> Result<(u128, u128), AkitaError> {
            Ok((0, 1))
        }

        fn available_polynomial_types(
            &self,
            _plan: &CommitInnerPlan,
        ) -> Result<AvailablePolynomialTypes, AkitaError> {
            AvailablePolynomialTypes::new(vec![if self.onehot {
                PolynomialType::OneHot(OneHotType::new(8, OneHotIndexWidth::U8)?)
            } else {
                PolynomialType::Dense(DenseType::Coefficients)
            }])
        }

        fn represent_as(
            &self,
            selected: PolynomialTypeSelection,
            _plan: &CommitInnerPlan,
        ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
            self.materializations.fetch_add(1, Ordering::SeqCst);
            match selected.polynomial_type() {
                PolynomialType::Dense(DenseType::Coefficients) if !self.onehot => Ok(
                    PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(self)),
                ),
                PolynomialType::OneHot(_) if self.onehot => {
                    Ok(PolynomialRepresentation::OneHot(OneHotRepresentation {
                        positions: UnitPositionSlice::U8(&self.positions),
                        chunk_size: 8,
                        num_vars: 6,
                    }))
                }
                _ => Err(AkitaError::InvalidInput(
                    "counting source received wrong selection".into(),
                )),
            }
        }
    }

    #[test]
    fn request_compilation_finishes_discovery_before_materialization() {
        let dense = CountingSource {
            onehot: false,
            coefficients: vec![F::from_u64(1); 64],
            positions: Vec::new(),
            materializations: AtomicUsize::new(0),
        };
        let unsupported_onehot = CountingSource {
            onehot: true,
            coefficients: Vec::new(),
            positions: vec![None; 8],
            materializations: AtomicUsize::new(0),
        };
        let sources: [&dyn CommitmentSource<F>; 2] = [&dense, &unsupported_onehot];
        struct TestBackend;
        let capabilities = CommitmentRequestCapabilities::split::<()>(
            BackendKindId::of::<TestBackend>("test").unwrap(),
            vec![PolynomialType::Dense(DenseType::Coefficients)],
        );
        assert!(matches!(
            compile_commitment_request(&plan(), &sources, &capabilities,),
            Err(AkitaError::InvalidInput(_))
        ));
        assert_eq!(dense.materializations.load(Ordering::SeqCst), 0);
        assert_eq!(
            unsupported_onehot.materializations.load(Ordering::SeqCst),
            0
        );

        let supported: [&dyn CommitmentSource<F>; 1] = [&dense];
        let compiled = compile_commitment_request(&plan(), &supported, &capabilities).unwrap();
        assert_eq!(dense.materializations.load(Ordering::SeqCst), 0);
        assert_eq!(compiled.materialize().unwrap().len(), 1);
        assert_eq!(dense.materializations.load(Ordering::SeqCst), 1);
    }

    struct OfferingSource {
        offers: Vec<PolynomialType>,
    }

    impl CommitmentSource<F> for OfferingSource {
        fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
            CommitSourceDescriptor::new(6, 64, 64, CommitSourceClass::Dense, "offering_source")
        }

        fn committed_centered_reach(
            &self,
            _modulus: u128,
            _centering_threshold: u128,
        ) -> Result<(u128, u128), AkitaError> {
            Ok((0, 1))
        }

        fn available_polynomial_types(
            &self,
            _plan: &CommitInnerPlan,
        ) -> Result<AvailablePolynomialTypes, AkitaError> {
            AvailablePolynomialTypes::new(self.offers.clone())
        }

        fn represent_as(
            &self,
            _selected: PolynomialTypeSelection,
            _plan: &CommitInnerPlan,
        ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
            unreachable!("selection tests do not materialize")
        }
    }

    #[test]
    fn request_selects_one_common_representation_for_the_group() {
        let coefficients = PolynomialType::Dense(DenseType::Coefficients);
        let digits = PolynomialType::Dense(DenseType::PredecomposedDigits);
        let first = OfferingSource {
            offers: vec![digits, coefficients],
        };
        let second = OfferingSource {
            offers: vec![coefficients, digits],
        };
        let sources: [&dyn CommitmentSource<F>; 2] = [&first, &second];
        struct TestBackend;
        let backend = BackendKindId::of::<TestBackend>("test").unwrap();

        let preferred =
            CommitmentRequestCapabilities::split::<()>(backend, vec![coefficients, digits]);
        let compiled = compile_commitment_request(&plan(), &sources, &preferred).unwrap();
        assert_eq!(
            compiled.selected_polynomial_types().collect::<Vec<_>>(),
            vec![coefficients]
        );

        let mut accept_any = CommitmentRequestCapabilities::split::<()>(backend, Vec::new());
        accept_any.accept_any_standard_type();
        let compiled = compile_commitment_request(&plan(), &sources, &accept_any).unwrap();
        assert_eq!(
            compiled.selected_polynomial_types().collect::<Vec<_>>(),
            vec![digits]
        );
    }

    #[test]
    fn onehot_materialization_rejects_positions_outside_the_chunk() {
        let source = CountingSource {
            onehot: true,
            coefficients: Vec::new(),
            positions: vec![Some(0), Some(7), None, Some(8), None, None, None, None],
            materializations: AtomicUsize::new(0),
        };
        let sources: [&dyn CommitmentSource<F>; 1] = [&source];
        struct TestBackend;
        let capabilities = CommitmentRequestCapabilities::split::<()>(
            BackendKindId::of::<TestBackend>("test").unwrap(),
            vec![PolynomialType::OneHot(
                OneHotType::new(8, OneHotIndexWidth::U8).unwrap(),
            )],
        );

        let error = compile_commitment_request(&plan(), &sources, &capabilities)
            .unwrap()
            .materialize()
            .err()
            .expect("out-of-range one-hot position must be rejected");
        assert!(
            matches!(error, AkitaError::InvalidInput(message) if message.contains("outside its chunk"))
        );
    }

    struct ExternalBackend;
    struct ExternalFamily(u8);
    struct ExternalAlgorithm;
    struct ExternalContext;
    struct FusedCommand;

    struct ExternalOperation;

    impl ExternalInnerCommitmentOperation<F> for ExternalOperation {
        fn identity(&self) -> ExternalOperationIdentity {
            ExternalOperationIdentity::of::<ExternalFamily, ExternalAlgorithm, ExternalContext>()
        }

        fn commit_group(
            &self,
            _plan: &CommitInnerPlan,
            _sources: &[ExternalInnerCommitmentInput<'_>],
            _context: &dyn std::any::Any,
        ) -> Result<Vec<crate::CommitInnerWitness<F>>, AkitaError> {
            Ok(Vec::new())
        }
    }

    struct ExternalOnlySource {
        payload: ExternalFamily,
        operation: ExternalOperation,
        preparations: AtomicUsize,
    }

    impl CommitmentSource<F> for ExternalOnlySource {
        fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
            CommitSourceDescriptor::new(6, 64, 64, CommitSourceClass::Dense, "external_only_test")
        }

        fn committed_centered_reach(
            &self,
            _modulus: u128,
            _centering_threshold: u128,
        ) -> Result<(u128, u128), AkitaError> {
            Ok((0, 1))
        }

        fn available_polynomial_types(
            &self,
            _plan: &CommitInnerPlan,
        ) -> Result<AvailablePolynomialTypes, AkitaError> {
            AvailablePolynomialTypes::new(Vec::new())
        }

        fn represent_as(
            &self,
            _selected: PolynomialTypeSelection,
            _plan: &CommitInnerPlan,
        ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
            Err(AkitaError::InvalidInput(
                "external-only source has no standard representation".into(),
            ))
        }

        fn external_inner_commitment_capability(
            &self,
            backend: BackendKindId,
            _plan: &CommitInnerPlan,
        ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
            let expected = BackendKindId::of::<ExternalBackend>("external-test")?;
            Ok((backend == expected).then(|| {
                ExternalInnerCommitmentCapability::new::<
                    ExternalFamily,
                    ExternalAlgorithm,
                    ExternalContext,
                >(expected, "external-test-algorithm")
                .unwrap()
            }))
        }

        fn prepare_external_inner_commitment(
            &self,
            selected: ExternalInnerCommitmentCapability,
            _plan: &CommitInnerPlan,
        ) -> Result<PreparedExternalInnerCommitment<'_, F>, AkitaError> {
            self.preparations.fetch_add(1, Ordering::SeqCst);
            PreparedExternalInnerCommitment::new(selected, &self.payload, &self.operation, None)
        }
    }

    #[test]
    fn external_only_source_is_checked_and_selected() {
        let source = ExternalOnlySource {
            payload: ExternalFamily(7),
            operation: ExternalOperation,
            preparations: AtomicUsize::new(0),
        };
        let sources: [&dyn CommitmentSource<F>; 1] = [&source];
        let backend = BackendKindId::of::<ExternalBackend>("external-test").unwrap();
        let split = CommitmentRequestCapabilities::split::<ExternalContext>(backend, Vec::new());
        let resolved = compile_commitment_request(&plan(), &sources, &split)
            .unwrap()
            .materialize()
            .unwrap();
        let external = resolved[0].external().unwrap();
        assert_eq!(external.input().payload::<ExternalFamily>().unwrap().0, 7);
        assert_eq!(source.preparations.load(Ordering::SeqCst), 1);

        let fused = CommitmentRequestCapabilities::fused::<ExternalContext, FusedCommand>(
            backend,
            Vec::new(),
        );
        assert!(compile_commitment_request(&plan(), &sources, &fused,).is_err());
        assert_eq!(source.preparations.load(Ordering::SeqCst), 1);
    }
}
