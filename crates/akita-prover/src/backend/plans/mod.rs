use akita_challenges::Challenges;
use akita_error::AkitaError;
use jolt_field::Field;
use std::marker::PhantomData;
use std::ops::Range;

mod relation;
pub use relation::{
    EvaluationTraceDescription, PhysicalL2WeightRequest, RelationWeightRequest,
    Stage2OpeningDescription, ValidatedRelationSessionPlan,
};

/// Canonical response geometry for one validated fold probe.
#[derive(Debug, Clone, Copy)]
pub enum FoldProbeGeometry<'a> {
    /// One ordinary, unpartitioned response.
    Sparse,
    /// One response per canonical dyadic block range.
    SparseChunked { chunk_ranges: &'a [Range<usize>] },
}

impl FoldProbeGeometry<'_> {
    /// Canonical ranges for a chunked probe, or `None` for an ordinary probe.
    pub fn chunk_ranges(&self) -> Option<&[Range<usize>]> {
        match self {
            Self::Sparse => None,
            Self::SparseChunked { chunk_ranges } => Some(chunk_ranges),
        }
    }
}

/// Schedule-derived fold-response admission policy.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedFoldAcceptancePlan {
    digit_negative_abs_bound: u128,
    digit_positive_bound: u128,
    response_l2_sq_cap: Option<u128>,
}

impl ValidatedFoldAcceptancePlan {
    /// Public admission bounds, checked against the admitted schedule by each backend.
    pub fn new(
        digit_negative_abs_bound: u128,
        digit_positive_bound: u128,
        response_l2_sq_cap: Option<u128>,
    ) -> Self {
        Self {
            digit_negative_abs_bound,
            digit_positive_bound,
            response_l2_sq_cap,
        }
    }

    /// Largest admitted absolute value on the negative side.
    pub const fn digit_negative_abs_bound(&self) -> u128 {
        self.digit_negative_abs_bound
    }

    /// Largest admitted value on the positive side.
    pub const fn digit_positive_bound(&self) -> u128 {
        self.digit_positive_bound
    }

    /// Optional squared-L2 admission cap.
    pub const fn response_l2_sq_cap(&self) -> Option<u128> {
        self.response_l2_sq_cap
    }
}

/// Akita-validated, context-free inputs for one backend fold probe.
///
/// Construction is crate-private so external kernels can inspect, but cannot
/// forge, challenge geometry or admission policy.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedFoldProbePlan<'a> {
    ring_dimension: usize,
    challenges: &'a Challenges,
    geometry: FoldProbeGeometry<'a>,
    num_positions_per_block: usize,
    num_digits: usize,
    log_basis: u32,
    opening_method: akita_types::OpeningMethod,
    acceptance: ValidatedFoldAcceptancePlan,
}

impl<'a> ValidatedFoldProbePlan<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<const D: usize>(
        challenges: &'a Challenges,
        source_claims: usize,
        expected_live_blocks: usize,
        geometry: FoldProbeGeometry<'a>,
        num_positions_per_block: usize,
        num_digits: usize,
        log_basis: u32,
        opening_method: akita_types::OpeningMethod,
        acceptance: ValidatedFoldAcceptancePlan,
    ) -> Result<Self, AkitaError> {
        if challenges.is_empty()
            || source_claims == 0
            || challenges.num_claims() != source_claims
            || challenges.num_live_blocks_per_claim() != expected_live_blocks
            || num_positions_per_block == 0
            || num_digits == 0
        {
            return Err(AkitaError::InvalidInput(
                "fold probe plan has malformed source or response geometry".into(),
            ));
        }
        for challenge in challenges.as_slice() {
            challenge.validate::<D>()?;
        }
        num_positions_per_block
            .checked_mul(num_digits)
            .and_then(|rows| rows.checked_mul(D))
            .ok_or_else(|| AkitaError::InvalidInput("fold response size overflow".into()))?;
        if let FoldProbeGeometry::SparseChunked { chunk_ranges } = geometry {
            if chunk_ranges.len() < 2
                || chunk_ranges
                    != akita_types::dyadic_block_ranges(expected_live_blocks, chunk_ranges.len())?
            {
                return Err(AkitaError::InvalidInput(
                    "chunked fold probe requires the canonical dyadic ranges".into(),
                ));
            }
        }
        Ok(Self {
            ring_dimension: D,
            challenges,
            geometry,
            num_positions_per_block,
            num_digits,
            log_basis,
            opening_method,
            acceptance,
        })
    }

    /// Full, unwindowed typed challenge batch.
    pub const fn challenges(&self) -> &Challenges {
        self.challenges
    }

    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    /// Validated response geometry.
    pub const fn geometry(&self) -> FoldProbeGeometry<'a> {
        self.geometry
    }

    pub const fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }

    pub const fn num_digits(&self) -> usize {
        self.num_digits
    }

    pub const fn log_basis(&self) -> u32 {
        self.log_basis
    }

    pub const fn opening_method(&self) -> akita_types::OpeningMethod {
        self.opening_method
    }

    pub const fn acceptance(&self) -> &ValidatedFoldAcceptancePlan {
        &self.acceptance
    }
}

/// Validated public geometry for consumer-owned recursive-witness construction.
#[derive(Clone, Copy)]
pub struct ValidatedRecursiveWitnessPlan<'a, F: Field> {
    expected_logical_len: usize,
    commitment_ring_dimension: usize,
    _field: PhantomData<&'a F>,
}

/// Validated public inputs for one consumer-owned Stage 1 session.
#[derive(Clone)]
pub struct ValidatedStage1Plan<E: Field> {
    digit_range: akita_types::DigitRangePlan,
    domain: akita_types::FlatBooleanDomain,
    equality: akita_types::DigitRangeEqualityPoint<E>,
    physical: Option<akita_types::PhysicalResponsePlan>,
    witness_len: usize,
}

impl<E: Field> ValidatedStage1Plan<E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        digit_range: akita_types::DigitRangePlan,
        domain: akita_types::FlatBooleanDomain,
        equality: akita_types::DigitRangeEqualityPoint<E>,
        physical: Option<akita_types::PhysicalResponsePlan>,
        witness_len: usize,
    ) -> Self {
        Self {
            digit_range,
            domain,
            equality,
            physical,
            witness_len,
        }
    }

    pub const fn digit_range(&self) -> akita_types::DigitRangePlan {
        self.digit_range
    }
    pub const fn domain(&self) -> akita_types::FlatBooleanDomain {
        self.domain
    }
    pub fn equality(&self) -> akita_types::DigitRangeEqualityPoint<E> {
        self.equality.clone()
    }
    pub fn physical(&self) -> Option<akita_types::PhysicalResponsePlan> {
        self.physical.clone()
    }
    pub const fn witness_len(&self) -> usize {
        self.witness_len
    }
}

impl<'a, F: Field> ValidatedRecursiveWitnessPlan<'a, F> {
    pub(crate) const fn new(expected_logical_len: usize, commitment_ring_dimension: usize) -> Self {
        Self {
            expected_logical_len,
            commitment_ring_dimension,

            _field: PhantomData,
        }
    }

    pub const fn logical_len(&self) -> usize {
        self.expected_logical_len
    }

    pub const fn commitment_ring_dimension(&self) -> usize {
        self.commitment_ring_dimension
    }
}

/// Public successor commitment parameters; execution policy belongs to the backend.
#[derive(Debug, Clone)]
pub enum WitnessCommitmentParameters {
    Recursive(akita_types::CommittedGroupParams),
    Terminal(akita_types::TerminalFoldParams),
}

/// Validated public geometry for committing an opaque recursive witness.
#[derive(Debug, Clone)]
pub struct ValidatedRecursiveWitnessCommitPlan {
    logical_len: usize,
    padded_len: usize,
    ring_dimension: usize,

    parameters: WitnessCommitmentParameters,
    source_encoding: Option<akita_types::CommittedSourceEncoding>,
    binding: akita_types::NextWitnessBindingPolicy,
}

impl ValidatedRecursiveWitnessCommitPlan {
    pub(crate) const fn new(
        logical_len: usize,
        padded_len: usize,
        ring_dimension: usize,

        parameters: WitnessCommitmentParameters,
        source_encoding: Option<akita_types::CommittedSourceEncoding>,
        binding: akita_types::NextWitnessBindingPolicy,
    ) -> Self {
        Self {
            logical_len,
            padded_len,
            ring_dimension,

            parameters,
            source_encoding,
            binding,
        }
    }
    pub const fn logical_len(&self) -> usize {
        self.logical_len
    }
    pub const fn padded_len(&self) -> usize {
        self.padded_len
    }
    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    pub const fn parameters(&self) -> &WitnessCommitmentParameters {
        &self.parameters
    }
    pub const fn source_encoding(&self) -> Option<akita_types::CommittedSourceEncoding> {
        self.source_encoding
    }
    pub const fn binding(&self) -> akita_types::NextWitnessBindingPolicy {
        self.binding
    }
}

/// Validated native opening geometry for one opaque recursive witness.
#[derive(Clone, Copy)]
pub struct ValidatedRecursiveGroupOpeningPlan<'a, E: Field> {
    point: &'a [E],
    basis: akita_types::BasisMode,
    ring_dimension: usize,
    positions_per_block: usize,
    live_blocks: usize,
    alpha_bits: usize,
    opening_method: akita_types::OpeningMethod,
    witness_len: usize,
}

impl<'a, E: Field> ValidatedRecursiveGroupOpeningPlan<'a, E> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        point: &'a [E],
        basis: akita_types::BasisMode,
        ring_dimension: usize,
        positions_per_block: usize,
        live_blocks: usize,
        alpha_bits: usize,
        opening_method: akita_types::OpeningMethod,
        witness_len: usize,
    ) -> Self {
        Self {
            point,
            basis,
            ring_dimension,
            positions_per_block,
            live_blocks,
            alpha_bits,
            opening_method,
            witness_len,
        }
    }
    pub const fn point(&self) -> &'a [E] {
        self.point
    }
    pub const fn basis(&self) -> akita_types::BasisMode {
        self.basis
    }
    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    pub const fn positions_per_block(&self) -> usize {
        self.positions_per_block
    }
    pub const fn live_blocks(&self) -> usize {
        self.live_blocks
    }
    pub const fn alpha_bits(&self) -> usize {
        self.alpha_bits
    }
    pub const fn opening_method(&self) -> akita_types::OpeningMethod {
        self.opening_method
    }
    pub const fn witness_len(&self) -> usize {
        self.witness_len
    }
}

/// Validated public geometry for preparing the opaque Stage 1/2 relation handle.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedRelationWitnessPlan {
    witness_len: usize,
    coefficient_count: usize,
    extension_degree: usize,
    opening_source_len: usize,
}

impl ValidatedRelationWitnessPlan {
    pub(crate) const fn new(
        witness_len: usize,
        coefficient_count: usize,
        extension_degree: usize,
        opening_source_len: usize,
    ) -> Self {
        Self {
            witness_len,
            coefficient_count,
            extension_degree,
            opening_source_len,
        }
    }
    pub const fn witness_len(self) -> usize {
        self.witness_len
    }
    pub const fn coefficient_count(self) -> usize {
        self.coefficient_count
    }
    pub const fn extension_degree(self) -> usize {
        self.extension_degree
    }
    pub const fn opening_source_len(self) -> usize {
        self.opening_source_len
    }
}

/// Akita-validated inputs for one terminal fold probe.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedTerminalFoldProbePlan<'a> {
    ring_dimension: usize,
    challenges: &'a Challenges,
    num_positions_per_block: usize,
    num_digits: usize,
    log_basis: u32,
    coordinate_count: usize,
    linf_cap: Option<u128>,
    l2_sq_cap: Option<u128>,
    rice_low_bits: u32,
    payload_bytes: usize,
}

impl<'a> ValidatedTerminalFoldProbePlan<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<const D: usize>(
        challenges: &'a Challenges,
        num_positions_per_block: usize,
        num_digits: usize,
        log_basis: u32,
        coordinate_count: usize,
        linf_cap: Option<u128>,
        l2_sq_cap: Option<u128>,
        rice_low_bits: u32,
        payload_bytes: usize,
    ) -> Result<Self, AkitaError> {
        if challenges.num_claims() != 1
            || challenges.num_live_blocks_per_claim() == 0
            || num_positions_per_block == 0
            || num_digits == 0
            || coordinate_count == 0
            || payload_bytes == 0
        {
            return Err(AkitaError::InvalidInput(
                "terminal fold probe has malformed geometry".into(),
            ));
        }
        for challenge in challenges.as_slice() {
            challenge.validate::<D>()?;
        }
        Ok(Self {
            ring_dimension: D,
            challenges,
            num_positions_per_block,
            num_digits,
            log_basis,
            coordinate_count,
            linf_cap,
            l2_sq_cap,
            rice_low_bits,
            payload_bytes,
        })
    }

    pub const fn challenges(&self) -> &Challenges {
        self.challenges
    }
    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    pub const fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }
    pub const fn num_digits(&self) -> usize {
        self.num_digits
    }
    pub const fn log_basis(&self) -> u32 {
        self.log_basis
    }
    pub const fn coordinate_count(&self) -> usize {
        self.coordinate_count
    }
    pub const fn linf_cap(&self) -> Option<u128> {
        self.linf_cap
    }
    pub const fn l2_sq_cap(&self) -> Option<u128> {
        self.l2_sq_cap
    }
    pub const fn rice_low_bits(&self) -> u32 {
        self.rice_low_bits
    }
    pub const fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }
}

/// Akita-validated canonical wire policy for an accepted terminal Z handle.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedTerminalZEncodingPlan {
    ring_dimension: usize,
    coordinate_count: usize,
    linf_cap: Option<u128>,
    rice_low_bits: u32,
    payload_bytes: usize,
}

impl ValidatedTerminalZEncodingPlan {
    pub(crate) const fn from_probe(plan: &ValidatedTerminalFoldProbePlan<'_>) -> Self {
        Self {
            ring_dimension: plan.ring_dimension,
            coordinate_count: plan.coordinate_count,
            linf_cap: plan.linf_cap,
            rice_low_bits: plan.rice_low_bits,
            payload_bytes: plan.payload_bytes,
        }
    }

    pub const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    pub const fn coordinate_count(self) -> usize {
        self.coordinate_count
    }
    pub const fn linf_cap(self) -> Option<u128> {
        self.linf_cap
    }
    pub const fn rice_low_bits(self) -> u32 {
        self.rice_low_bits
    }
    pub const fn payload_bytes(self) -> usize {
        self.payload_bytes
    }
}
