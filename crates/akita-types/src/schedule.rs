//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::layout::params::{
    append_fold_linf_cap_config_descriptor_bytes, append_schedule_sparse_challenge_descriptor_bytes,
};
use crate::sis::FoldWitnessLinfCapConfig;
use crate::{
    CommittedGroupParams, InnerCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams,
    PolynomialGroupLayout, SetupContributionMode, TerminalResponseShape,
};
use akita_field::{AkitaError, CanonicalField};

/// Public inputs that deterministically select one level's active Akita params.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleInputs {
    /// Root polynomial variable count.
    pub num_vars: usize,
    /// Fold level, where `0` is the original polynomial.
    pub level: usize,
    /// Current witness length in field elements before this level runs.
    pub input_witness_len: usize,
}

/// Transcript binding used for one fold's outgoing witness state.
///
/// This is schedule-owned because the same intermediate proof body may either
/// recurse through an outer commitment or hand its witness to the final
/// suffix fold as a public inner `t` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextWitnessBindingPolicy {
    /// Bind `u = B * decompose(t)` and recurse through another committed fold.
    OuterCommitment,
    /// Bind canonical inner-state `t` bytes for the following suffix-terminal
    /// fold. No outer `u` is present on this edge.
    TerminalInnerState,
}

/// Root layout metadata frozen when a standalone commitment group is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommittedGroupProfile {
    /// Version of the canonical committed-profile encoding.
    pub version: u8,
    /// Per-group root schedule entry shape.
    pub group: PolynomialGroupLayout,
    /// Exact number of live source ring elements per claim (`N`).
    pub num_live_ring_elements_per_claim: usize,
    /// Number of positions per block (`M`), power-of-two in the current Boolean layout.
    pub num_positions_per_block: usize,
    /// Exact number of live blocks (`B = ceil(N / M)`).
    pub num_live_blocks: usize,
    /// Gadget basis selected for the standalone A/source digits.
    pub log_basis_inner: u32,
    /// Exact gadget depth used by the standalone A/source relation.
    pub num_digits_inner: usize,
    /// Complete audited A/source matrix identity.
    pub inner_commit_matrix: InnerCommitMatrixParams,
    /// Gadget basis selected for the standalone B/`t_hat` digits.
    pub log_basis_outer: u32,
    /// Exact gadget depth used by the standalone B/`t_hat` relation.
    pub num_digits_outer: usize,
    /// Complete audited B/commitment matrix identity.
    pub outer_commit_matrix: OuterCommitMatrixParams,
}

impl CommittedGroupProfile {
    /// Current committed-profile format.
    pub const VERSION: u8 = 1;

    /// Build frozen group metadata from the concrete commit params.
    pub fn from_params(group: PolynomialGroupLayout, params: &CommittedGroupParams) -> Self {
        Self {
            version: Self::VERSION,
            group,
            num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
            num_positions_per_block: params.num_positions_per_block,
            num_live_blocks: params.num_live_blocks,
            log_basis_inner: params.log_basis_inner,
            num_digits_inner: params.num_digits_inner,
            inner_commit_matrix: params.inner_commit_matrix,
            log_basis_outer: params.log_basis_outer,
            num_digits_outer: params.num_digits_outer,
            outer_commit_matrix: params.outer_commit_matrix,
        }
    }

    /// Canonical versioned bytes used for catalog and schedule-key identity.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(self.version);
        push_usize(bytes, self.group.num_vars());
        push_usize(bytes, self.group.num_polynomials());
        push_usize(bytes, self.num_live_ring_elements_per_claim);
        push_usize(bytes, self.num_positions_per_block);
        push_usize(bytes, self.num_live_blocks);
        push_u32(bytes, self.log_basis_inner);
        push_usize(bytes, self.num_digits_inner);
        self.inner_commit_matrix.append_descriptor_bytes(bytes);
        push_u32(bytes, self.log_basis_outer);
        push_usize(bytes, self.num_digits_outer);
        self.outer_commit_matrix.append_descriptor_bytes(bytes);
    }

    /// Validate that this layout is a well-formed standalone commitment group.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.group.validate()?;
        if self.version != Self::VERSION {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported committed-group profile version {}",
                self.version
            )));
        }
        self.inner_commit_matrix.validate()?;
        self.outer_commit_matrix.validate()?;
        let inner_ring_dimension = self.inner_commit_matrix.ring_dimension();
        let outer_ring_dimension = self.outer_commit_matrix.ring_dimension();
        if !inner_ring_dimension.is_power_of_two()
            || !outer_ring_dimension.is_power_of_two()
            || !inner_ring_dimension.is_multiple_of(outer_ring_dimension)
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted group requires power-of-two A/B dimensions with A divisible by B"
                    .to_string(),
            ));
        }
        if self.log_basis_inner == 0 || self.num_digits_inner == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero inner basis and digit depth".to_string(),
            ));
        }
        if self.log_basis_outer == 0 || self.num_digits_outer == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero outer basis and digit depth".to_string(),
            ));
        }
        if self.inner_commit_matrix.sis_modulus_profile().field_bits() != field_bits
            || self.outer_commit_matrix.sis_modulus_profile().field_bits() != field_bits
        {
            return Err(AkitaError::InvalidSetup(
                "committed-group matrix modulus profile does not match the field".to_string(),
            ));
        }
        let expected_a_width = self
            .num_positions_per_block
            .checked_mul(self.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("committed-group A width overflow".into()))?;
        let projection_ratio = inner_ring_dimension / outer_ring_dimension;
        let expected_b_width = self
            .inner_commit_matrix
            .output_rank()
            .checked_mul(self.num_digits_outer)
            .and_then(|width| width.checked_mul(self.num_live_blocks))
            .and_then(|width| width.checked_mul(self.group.num_polynomials()))
            .and_then(|width| width.checked_mul(projection_ratio))
            .ok_or_else(|| AkitaError::InvalidSetup("committed-group B width overflow".into()))?;
        if self.inner_commit_matrix.input_width() != expected_a_width
            || self.outer_commit_matrix.input_width() != expected_b_width
        {
            return Err(AkitaError::InvalidSetup(
                "committed-group A/B matrix widths do not match frozen geometry".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate that frozen exact block geometry matches `group.num_vars`.
    pub fn validate_root_geometry(&self) -> Result<(), AkitaError> {
        let inner_ring_dimension = self.inner_commit_matrix.ring_dimension();
        let alpha = inner_ring_dimension.trailing_zeros() as usize;
        let Some(source_field_len) = self
            .num_live_ring_elements_per_claim
            .checked_mul(inner_ring_dimension)
        else {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout geometry overflow".to_string(),
            ));
        };
        let num_vars = u32::try_from(self.group.num_vars()).map_err(|_| {
            AkitaError::InvalidSetup("commitment group variable count exceeds u32".to_string())
        })?;
        let expected_field_len = 1usize.checked_shl(num_vars).ok_or_else(|| {
            AkitaError::InvalidSetup("commitment group field length overflow".to_string())
        })?;
        if source_field_len != expected_field_len
            || self.num_positions_per_block == 0
            || !self.num_positions_per_block.is_power_of_two()
            || self.num_live_blocks
                != self
                    .num_live_ring_elements_per_claim
                    .div_ceil(self.num_positions_per_block)
        {
            return Err(AkitaError::InvalidSetup(format!(
                "precommitted group geometry does not match group.num_vars: \
                 N={} L={} F={} alpha={} group.num_vars={}",
                self.num_live_ring_elements_per_claim,
                self.num_positions_per_block,
                self.num_live_blocks,
                alpha,
                self.group.num_vars()
            )));
        }
        Ok(())
    }

    /// Validate metadata frozen by a precommitted group at precommit time.
    pub fn validate_frozen_precommit(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.validate(field_bits)?;
        self.validate_root_geometry()?;
        Ok(())
    }
}

/// Canonical runtime schedule lookup key.
///
/// Scalar same-point openings use an empty `precommitteds` vector and store the
/// sole group in `final_group`. Multi-group roots list earlier groups in
/// `precommitteds` and the final group in `final_group`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Final group shape for the multi-group root commitment.
    pub final_group: PolynomialGroupLayout,
    /// Exact coefficient class requested for the final group.
    pub final_source: GroupSource,
    /// Previously committed groups in caller-supplied transcript order.
    pub precommitteds: Vec<CommittedGroupProfile>,
    /// Planner-only honest source models for `precommitteds`.
    ///
    /// Runtime catalog resolution may leave this empty because generated rows
    /// carry their already-certified exact consuming parameters.
    pub precommitted_sources: Vec<GroupSource>,
}

impl AkitaScheduleLookupKey {
    /// Scalar root-opening context with no precommitted groups.
    pub fn single(final_group: PolynomialGroupLayout, final_source: GroupSource) -> Self {
        Self {
            final_group,
            final_source,
            precommitteds: Vec::new(),
            precommitted_sources: Vec::new(),
        }
    }

    /// Canonical ordered bytes used for schedule-key identity tests and caches.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_usize(&mut bytes, self.final_group.num_vars());
        push_usize(&mut bytes, self.final_group.num_polynomials());
        self.final_source.append_descriptor_bytes(&mut bytes);
        push_usize(&mut bytes, self.precommitteds.len());
        for descriptor in &self.precommitteds {
            descriptor.append_descriptor_bytes(&mut bytes);
        }
        push_usize(&mut bytes, self.precommitted_sources.len());
        for source in &self.precommitted_sources {
            source.append_descriptor_bytes(&mut bytes);
        }
        bytes
    }

    /// Build a multi-group opening layout from this schedule lookup key.
    pub fn opening_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        let mut groups: Vec<PolynomialGroupLayout> = self
            .precommitteds
            .iter()
            .map(|layout| layout.group)
            .collect();
        groups.push(self.final_group);
        OpeningClaimsLayout::from_groups(groups)
    }

    /// Number of commitment groups in this schedule key.
    pub fn num_commitment_groups(&self) -> usize {
        self.precommitteds.len() + 1
    }

    /// Maximum opening arity across the final and precommitted groups.
    ///
    /// This is the shared opening-point/EOR domain. It is intentionally
    /// distinct from `final_group.num_vars()`, which remains the source arity
    /// used to size the final commitment and root witness.
    pub fn max_num_vars(&self) -> usize {
        self.precommitteds
            .iter()
            .map(|descriptor| descriptor.group.num_vars())
            .fold(self.final_group.num_vars(), usize::max)
    }

    /// Total number of polynomials across the final and precommitted groups.
    pub fn num_polynomials(&self) -> Result<usize, AkitaError> {
        let mut total = self.final_group.num_polynomials();
        for layout in &self.precommitteds {
            total = total
                .checked_add(layout.group.num_polynomials())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "multi-group root polynomial count overflow".to_string(),
                    )
                })?;
        }
        Ok(total)
    }

    /// Whether the complete opening key fits a setup's public capacity.
    pub fn fits_setup_capacity(
        &self,
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<bool, AkitaError> {
        Ok(self.max_num_vars() <= max_num_vars && self.num_polynomials()? <= max_num_batched_polys)
    }

    /// Validate per-group metadata.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.final_group.validate()?;
        self.final_source.validate(field_bits)?;
        if !self.precommitted_sources.is_empty()
            && self.precommitted_sources.len() != self.precommitteds.len()
        {
            return Err(AkitaError::InvalidSetup(
                "planner precommitted source count does not match committed profiles".to_string(),
            ));
        }
        if self.final_group.num_vars() == 0 {
            return Err(AkitaError::InvalidSetup(
                "schedule lookup key dimensions must be at least 1".to_string(),
            ));
        }
        for layout in &self.precommitteds {
            layout.group.validate()?;
            layout.validate(field_bits)?;
        }
        for source in &self.precommitted_sources {
            source.validate(field_bits)?;
        }
        Ok(())
    }
}

/// Exact ordered committed profiles used to resolve a verifier-approved row.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommittedGroupBatchProfile {
    /// Final/new commitment group.
    pub final_group: CommittedGroupProfile,
    /// Earlier commitments in transcript order.
    pub precommitteds: Vec<CommittedGroupProfile>,
}

impl CommittedGroupBatchProfile {
    /// Build the corresponding public opening layout.
    pub fn opening_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        let mut groups = self
            .precommitteds
            .iter()
            .map(|profile| profile.group)
            .collect::<Vec<_>>();
        groups.push(self.final_group.group);
        OpeningClaimsLayout::from_groups(groups)
    }

    /// Validate all frozen profiles.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.final_group.validate_frozen_precommit(field_bits)?;
        for profile in &self.precommitteds {
            profile.validate_frozen_precommit(field_bits)?;
        }
        Ok(())
    }
}

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::sis::compute_num_digits_field_width(field_bits, log_basis)
}

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
#[inline]
pub fn detect_field_modulus<F: CanonicalField>() -> u128 {
    crate::dispatch::field_modulus::<F>()
}

/// Total ring elements in an intermediate recursive witness polynomial.
/// Terminal witnesses are quotient-free and must be sized from their
/// [`crate::TerminalResponseShape`] instead.
pub fn intermediate_w_ring_element_count_with_counts<F: CanonicalField>(
    lp: &CommittedGroupParams,
    num_polynomials: usize,
    num_z_segments: usize,
) -> Result<usize, AkitaError> {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    intermediate_w_ring_element_count_with_counts_bits(
        field_bits,
        lp,
        num_polynomials,
        num_z_segments,
    )
}

/// Non-generic variant of [`intermediate_w_ring_element_count_with_counts`] for
/// callers that already know the effective field bit width. The planner
/// search uses this to keep its API free of a base-field type parameter.
pub fn intermediate_w_ring_element_count_with_counts_bits(
    field_bits: u32,
    lp: &CommittedGroupParams,
    num_polynomials: usize,
    num_z_segments: usize,
) -> Result<usize, AkitaError> {
    lp.require_scalar_level("intermediate_w_ring_element_count_with_counts_bits")?;
    let e_hat_count = num_polynomials
        .checked_mul(lp.num_live_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_polynomials
        .checked_mul(lp.num_live_blocks)
        .and_then(|n| n.checked_mul(lp.inner_commit_matrix.output_rank()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let num_digits_fold = lp.num_digits_fold(num_polynomials, field_bits)?;
    let z_pre_count = num_z_segments
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    let r_rows = lp.relation_matrix_row_count(1)?;
    let r_count = r_rows
        .checked_mul(crate::sis::compute_num_digits_field_width(
            field_bits,
            lp.log_basis_open,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("witness r-tail width overflow".to_string()))?;

    e_hat_count
        .checked_add(t_hat_count)
        .and_then(|n| n.checked_add(z_pre_count))
        .and_then(|n| n.checked_add(r_count))
        .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
}

/// Witness ring-element count for a chunked (multi-chunk) or single-chunk layout.
///
/// `num_chunks == 1` delegates to
/// [`intermediate_w_ring_element_count_with_counts_bits`] with `num_public_rows = 1`,
/// so it is byte-identical to the historical single-chunk pricing.
///
/// `num_chunks > 1` prices the multi-chunk witness layout used by the distributed
/// prover: `num_chunks` chunks each holding a partitioned slice of `ê`/`t̂` plus a
/// **replicated full-width** `ẑ`, followed by a single shared `r`-tail. The
/// per-node relations stack *horizontally* (`M = [M_0 | … | M_{num_chunks-1}]`),
/// sharing the same row blocks (concatenation adds columns, not rows) and summing
/// the partial commitments `u_j` into one `u`, so the quotient `r = Σ_j r_j` keeps
/// the **single-machine shape** — its row count is priced with `num_commitments =
/// 1`, unchanged from the single-chunk layout. The **only** extra cost over the
/// single-chunk layout is `(num_chunks - 1) · z_chunk` ring elements (the
/// replicated `ẑ`).
///
/// The exact `ê`/`t̂` live-block prefix is partitioned without padding. Its
/// total width and the shared `r` tail therefore stay unchanged.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `num_chunks == 0`, `num_chunks > 1`
/// is not a power of two, there are fewer live blocks than chunks, or
/// any width product overflows. Never panics — verifier-reachable through the runtime DP fallback.
pub fn intermediate_w_ring_element_count_for_chunks(
    field_bits: u32,
    lp: &CommittedGroupParams,
    num_polynomials: usize,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    if num_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "intermediate_w_ring_element_count_for_chunks: num_chunks must be >= 1".to_string(),
        ));
    }
    if num_chunks == 1 {
        return intermediate_w_ring_element_count_with_counts_bits(
            field_bits,
            lp,
            num_polynomials,
            1,
        );
    }
    if !num_chunks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "intermediate_w_ring_element_count_for_chunks: num_chunks must be a power of two"
                .to_string(),
        ));
    }
    if lp.num_live_blocks < num_chunks {
        return Err(AkitaError::InvalidSetup(format!(
            "intermediate_w_ring_element_count_for_chunks: num_live_blocks={} smaller than num_chunks={num_chunks}",
            lp.num_live_blocks
        )));
    }
    let overflow = || AkitaError::InvalidSetup("chunked witness width overflow".to_string());
    let single =
        intermediate_w_ring_element_count_with_counts_bits(field_bits, lp, num_polynomials, 1)?;
    let num_digits_fold = lp.num_digits_fold(num_polynomials, field_bits)?;
    let z_chunk = lp
        .inner_width()
        .checked_mul(num_digits_fold)
        .ok_or_else(overflow)?;
    num_chunks
        .checked_sub(1)
        .and_then(|copies| copies.checked_mul(z_chunk))
        .and_then(|extra| single.checked_add(extra))
        .ok_or_else(overflow)
}

/// Canonical identity of one registered polynomial-source family.
///
/// The fixed-width identifier and parameters are transcript-safe value data.
/// They erase the source's Rust type without erasing its public protocol
/// identity. Applications can define new registrations without extending an
/// Akita enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupSourceRegistration {
    type_id: [u8; 16],
    parameters: [u8; 16],
}

impl GroupSourceRegistration {
    /// Build a fixed-width source registration.
    pub const fn new(type_id: [u8; 16], parameters: [u8; 16]) -> Self {
        Self {
            type_id,
            parameters,
        }
    }

    /// Stable source-family identifier.
    pub const fn type_id(self) -> [u8; 16] {
        self.type_id
    }

    /// Canonical registration parameters.
    pub const fn parameters(self) -> [u8; 16] {
        self.parameters
    }
}

/// Certified source encoding understood by the current Akita protocol.
///
/// Registrations are open-world. Encodings are the smaller, versioned
/// protocol vocabulary whose decomposition and fold-norm rules the verifier
/// enforces. A new storage format can register against an existing encoding;
/// adding a new encoding requires a protocol change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GroupSourceEncoding {
    /// Balanced decomposition of centered coefficients with this public bound.
    Bounded { coefficient_bits: u32 },
    /// Binary coefficients with at most one nonzero per logical chunk.
    SparseBinary { chunk_size: usize },
}

impl GroupSourceEncoding {
    /// Encoded byte length in canonical protocol serialization.
    pub const SERIALIZED_SIZE: usize = 1 + 8;
}

/// Erased, self-describing source contract for one commitment group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupSource {
    registration: GroupSourceRegistration,
    encoding: GroupSourceEncoding,
}

/// Compile-time registration that erases into a canonical [`GroupSource`].
///
/// A concrete polynomial backend separately validates that its values belong
/// to the returned registration before commitment or proving.
pub trait RegisteredGroupSource {
    /// Stable source-family identifier and canonical public parameters.
    fn registration(&self) -> GroupSourceRegistration;

    /// Certified protocol encoding used for decomposition and security sizing.
    fn encoding(&self) -> GroupSourceEncoding;

    /// Erase this registration into the public group-source contract.
    fn group_source(&self) -> GroupSource {
        GroupSource::registered(self.registration(), self.encoding())
    }
}

impl GroupSource {
    /// Built-in bounded-dense registration identifier.
    pub const BOUNDED_REGISTRATION_ID: [u8; 16] = *b"akita/dense/v1\0\0";
    /// Built-in one-hot registration identifier.
    pub const ONE_HOT_REGISTRATION_ID: [u8; 16] = *b"akita/onehot/v1\0";
    /// Encoded byte length in canonical commitment serialization.
    pub const SERIALIZED_SIZE: usize = 16 + 16 + GroupSourceEncoding::SERIALIZED_SIZE;

    /// Build an open-world registered source contract.
    pub const fn registered(
        registration: GroupSourceRegistration,
        encoding: GroupSourceEncoding,
    ) -> Self {
        Self {
            registration,
            encoding,
        }
    }

    /// Built-in bounded-dense source contract.
    pub const fn bounded(coefficient_bits: u32) -> Self {
        Self::registered(
            GroupSourceRegistration::new(Self::BOUNDED_REGISTRATION_ID, [0; 16]),
            GroupSourceEncoding::Bounded { coefficient_bits },
        )
    }

    /// Built-in one-hot source contract.
    pub const fn one_hot(chunk_size: usize) -> Self {
        Self::registered(
            GroupSourceRegistration::new(Self::ONE_HOT_REGISTRATION_ID, [0; 16]),
            GroupSourceEncoding::SparseBinary { chunk_size },
        )
    }

    /// Canonical built-in registration for a certified protocol encoding.
    ///
    /// Provider registrations erase to the same profile encoding at the public
    /// commitment boundary.
    pub const fn from_encoding(encoding: GroupSourceEncoding) -> Self {
        match encoding {
            GroupSourceEncoding::Bounded { coefficient_bits } => Self::bounded(coefficient_bits),
            GroupSourceEncoding::SparseBinary { chunk_size } => Self::one_hot(chunk_size),
        }
    }

    /// Canonical registration identity and parameters.
    pub const fn registration(self) -> GroupSourceRegistration {
        self.registration
    }

    /// Certified protocol encoding.
    pub const fn encoding(self) -> GroupSourceEncoding {
        self.encoding
    }

    /// Validate a checked group-local source contract.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        if self.registration.type_id == [0; 16] {
            return Err(AkitaError::InvalidSetup(
                "group source registration identifier must be nonzero".into(),
            ));
        }
        match self.encoding {
            GroupSourceEncoding::Bounded { coefficient_bits }
                if coefficient_bits == 0 || coefficient_bits > field_bits =>
            {
                Err(AkitaError::InvalidSetup(format!(
                    "dense source coefficient_bits={coefficient_bits} must be in 1..={field_bits}"
                )))
            }
            GroupSourceEncoding::SparseBinary { chunk_size: 0 } => Err(AkitaError::InvalidSetup(
                "one-hot source chunk_size must be nonzero".into(),
            )),
            GroupSourceEncoding::SparseBinary { chunk_size } if !chunk_size.is_power_of_two() => {
                Err(AkitaError::InvalidSetup(format!(
                    "one-hot source chunk_size={chunk_size} must be a power of two"
                )))
            }
            _ => Ok(()),
        }
    }

    /// Validate the source representation against its commitment ring.
    pub fn validate_for_ring_dimension(
        &self,
        field_bits: u32,
        ring_dimension: usize,
    ) -> Result<(), AkitaError> {
        self.validate(field_bits)?;
        if ring_dimension == 0 {
            return Err(AkitaError::InvalidSetup(
                "source ring dimension must be nonzero".to_string(),
            ));
        }
        if let GroupSourceEncoding::SparseBinary { chunk_size } = self.encoding {
            if !(chunk_size.is_multiple_of(ring_dimension)
                || ring_dimension.is_multiple_of(chunk_size))
            {
                return Err(AkitaError::InvalidSetup(format!(
                    "one-hot chunk_size={chunk_size} and ring dimension {ring_dimension} must divide one another"
                )));
            }
        }
        Ok(())
    }

    /// Apply this source contract to the preset's field/opening decomposition.
    pub fn decomposition(self, base: crate::DecompositionParams) -> crate::DecompositionParams {
        crate::DecompositionParams {
            log_commit_bound: match self.encoding {
                GroupSourceEncoding::Bounded { coefficient_bits } => coefficient_bits,
                GroupSourceEncoding::SparseBinary { .. } => 1,
            },
            log_open_bound: Some(base.field_bits()),
            ..base
        }
    }

    /// Sparse-binary chunk size used by fold-norm pricing.
    pub const fn sparse_chunk_size(self) -> Option<usize> {
        match self.encoding {
            GroupSourceEncoding::Bounded { .. } => None,
            GroupSourceEncoding::SparseBinary { chunk_size } => Some(chunk_size),
        }
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        match self.encoding {
            GroupSourceEncoding::Bounded { coefficient_bits } => {
                bytes.push(0);
                push_u32(bytes, coefficient_bits);
            }
            GroupSourceEncoding::SparseBinary { chunk_size } => {
                bytes.push(1);
                push_usize(bytes, chunk_size);
            }
        }
    }

    /// Derive the root-source contract encoded by committed-group parameters.
    pub fn from_commitment(commitment: &CommittedGroupParams) -> Self {
        commitment.source
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootFinalChallenge {
    Flat,
    Tensor { fold_low_len: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessPartition {
    Single,
    Distributed { num_chunks: usize },
}

impl WitnessPartition {
    pub fn num_chunks(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::Distributed { num_chunks } => *num_chunks,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootFinalGroupParams {
    pub challenge: RootFinalChallenge,
    pub commitment: CommittedGroupParams,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootPrecommittedGroupParams {
    pub descriptor: CommittedGroupProfile,
    pub commitment: crate::PrecommittedLevelParams,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootFoldParams {
    pub final_group: RootFinalGroupParams,
    pub precommitted_groups: Vec<RootPrecommittedGroupParams>,
    pub open_commit_matrix: crate::OpenCommitMatrixParams,
    pub sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub witness_partition: WitnessPartition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveFoldParams {
    pub witness: CommittedGroupParams,
    pub open_commit_matrix: crate::OpenCommitMatrixParams,
    pub sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub incoming_setup_prefix: Option<crate::SetupPrefixSlotId>,
    pub witness_partition: WitnessPartition,
}

impl RecursiveFoldParams {
    /// Setup-contribution mode of the fold that produces this recursive
    /// witness. Presence of this consumer-owned prefix is the sole authority.
    pub fn predecessor_setup_contribution_mode(&self) -> SetupContributionMode {
        if self.incoming_setup_prefix.is_some() {
            SetupContributionMode::Recursive
        } else {
            SetupContributionMode::Direct
        }
    }
}

/// Exact terminal committed-witness parameters.
///
/// The terminal relation binds only the source decomposition through the
/// inner commitment matrix. It has no outer/open commitment matrix and no
/// outer/open response decomposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalCommittedGroupParams {
    pub log_basis_inner: u32,
    pub inner_commit_matrix: InnerCommitMatrixParams,
    pub num_live_ring_elements_per_claim: usize,
    pub num_positions_per_block: usize,
    pub num_live_blocks: usize,
    pub num_digits_inner: usize,
    pub fold_linf_cap_config: FoldWitnessLinfCapConfig,
}

/// Minimum fraction of the unconstrained terminal-response target that a
/// fixed inner matrix must admit. This is a planner completeness heuristic,
/// not a security assumption: security always uses the matrix's exact
/// SIS-certified capacity.
pub const TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM: u128 = 1;
pub const TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN: u128 = 2;

/// Derived terminal-response norm policy for one fixed inner matrix.
///
/// The three values have deliberately different meanings:
///
/// - `unconstrained_target` is the pre-digit-snap Rademacher/worst-case target
///   for an honest response. The terminal path has no response digits, so a
///   digit boundary must not reduce this value.
/// - `certified_capacity` is the largest raw response norm secured by the
///   fixed inner matrix, obtained by inverting its checked SIS-table capacity.
/// - `admission_cap` is the verifier's actual raw-response limit. It is the
///   certified capacity restricted to the current signed response
///   representation.
///
/// A candidate is usable only when `admission_cap >= ceil(target / 2)`. The
/// half-target rule is intentionally empirical: applying the conservative
/// coordinate union bound again at a reduced cap is vacuous for production
/// tails. It affects honest-prover viability only. The exact SIS capacity
/// remains the security authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TerminalResponseLinfPolicy {
    pub unconstrained_target: u128,
    pub certified_capacity: u128,
    pub admission_cap: u128,
}

impl TerminalCommittedGroupParams {
    pub fn from_expanded_group(params: CommittedGroupParams) -> Self {
        Self {
            log_basis_inner: params.log_basis_inner,
            inner_commit_matrix: params.inner_commit_matrix,
            num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
            num_positions_per_block: params.num_positions_per_block,
            num_live_blocks: params.num_live_blocks,
            num_digits_inner: params.num_digits_inner,
            fold_linf_cap_config: params.fold_linf_cap_config,
        }
    }

    /// Project an ordinary scalar group into terminal parameters and validate
    /// the directly checked response bound against its fixed inner matrix.
    pub fn try_from_expanded_group(
        params: CommittedGroupParams,
    ) -> Result<(Self, u128), AkitaError> {
        let sparse = params.fold_challenge_config;
        let terminal = Self::from_expanded_group(params);
        let response_policy = terminal.response_linf_policy(&sparse)?;
        Ok((terminal, response_policy.admission_cap))
    }

    #[inline]
    pub fn d_a(&self) -> usize {
        self.inner_commit_matrix.ring_dimension()
    }

    #[inline]
    pub fn inner_width(&self) -> usize {
        self.inner_commit_matrix.input_width()
    }

    /// Logical opening-point width for the witness entering the terminal fold.
    pub fn recursive_opening_num_vars(&self) -> Result<usize, AkitaError> {
        crate::layout::params::recursive_opening_num_vars_for_geometry(
            self.d_a(),
            self.num_positions_per_block,
            self.num_live_blocks,
        )
    }

    /// Derive the terminal target, fixed-matrix capacity, and admission cap.
    ///
    /// This is the single source used by planning, encoding, grinding, and
    /// verifier admission. The fixed matrix is never resized here. Candidates
    /// retaining less than half of the unconstrained target are rejected as
    /// impractical, while every admitted response remains secured by the exact
    /// matrix capacity.
    pub fn response_linf_policy(
        &self,
        sparse: &akita_challenges::SparseChallengeConfig,
    ) -> Result<TerminalResponseLinfPolicy, AkitaError> {
        let challenge = crate::sis::FoldChallengeNorms::new(
            sparse,
            akita_challenges::TensorChallengeShape::Flat,
        );
        let witness = crate::sis::FoldWitnessNorms::bounded(self.log_basis_inner, self.d_a());
        let (unconstrained_target, _) = crate::sis::fold_witness_unsnapped_linf_cap(
            self.num_live_blocks,
            1,
            challenge,
            witness,
            &self.fold_linf_cap_config,
        )?;
        let collision_capacity = self
            .inner_commit_matrix
            .max_secure_collision_linf()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("terminal A has no collision capacity".into())
            })?;
        let certified_capacity = crate::sis::max_response_linf_for_role_a_collision(
            collision_capacity,
            challenge.l1_norm,
            self.inner_commit_matrix
                .sis_modulus_profile()
                .ring_subfield_embedding_norm_bound(),
        )
        .filter(|value| *value > 0)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal A cannot certify a response".into()))?;
        // Terminal NTT kernels currently consume signed i16 coefficients.
        // This representation limit is independent of the SIS capacity.
        let admission_cap = certified_capacity.min(i16::MAX as u128);
        let minimum_usable_cap = unconstrained_target
            .checked_mul(TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal target ratio overflow".into()))?
            .div_ceil(TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN);
        if admission_cap < minimum_usable_cap {
            return Err(AkitaError::InvalidSetup(format!(
                "terminal response capacity {admission_cap} retains less than \
                     {TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM}/\
                     {TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN} of target \
                     {unconstrained_target}"
            )));
        }
        Ok(TerminalResponseLinfPolicy {
            unconstrained_target,
            certified_capacity,
            admission_cap,
        })
    }

    /// Validate the terminal Fiat–Shamir grind nonce under the same bound
    /// policy used to derive the response wire.
    pub fn validate_fold_grind_nonce(
        &self,
        sparse: &akita_challenges::SparseChallengeConfig,
        nonce: u32,
    ) -> Result<(), AkitaError> {
        let admission_cap = self.response_linf_policy(sparse)?.admission_cap;
        crate::sis::FoldWitnessGrindContract {
            policy: self.fold_linf_cap_config.policy,
            witness_linf_cap: admission_cap,
        }
        .validate_nonce(
            nonce,
            crate::FoldLinfProtocolBinding::CURRENT.max_grind_attempts,
        )
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_u32(bytes, self.log_basis_inner);
        self.inner_commit_matrix.append_descriptor_bytes(bytes);
        push_usize(bytes, self.num_live_ring_elements_per_claim);
        push_usize(bytes, self.num_positions_per_block);
        push_usize(bytes, self.num_live_blocks);
        push_usize(bytes, self.num_digits_inner);
        append_fold_linf_cap_config_descriptor_bytes(bytes, &self.fold_linf_cap_config);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalFoldParams {
    pub witness: TerminalCommittedGroupParams,
    pub sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub response_shape: TerminalResponseShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootFoldStep {
    pub params: RootFoldParams,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveFoldStep {
    pub params: RecursiveFoldParams,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalFoldStep {
    pub params: TerminalFoldParams,
    pub input_witness_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldSchedule {
    pub root: RootFoldStep,
    pub recursive_folds: Vec<RecursiveFoldStep>,
    pub terminal: TerminalFoldStep,
}

impl FoldSchedule {
    pub fn num_fold_levels(&self) -> usize {
        self.recursive_folds.len() + 2
    }

    pub fn root_fold(&self) -> &RootFoldStep {
        &self.root
    }

    pub fn root_fold_mut(&mut self) -> &mut RootFoldStep {
        &mut self.root
    }

    pub fn validate_structure(&self) -> Result<(), AkitaError> {
        if self.root.input_witness_len == 0 || self.root.output_witness_len == 0 {
            return Err(AkitaError::InvalidSetup(
                "root fold witness lengths must be nonzero".to_string(),
            ));
        }
        let first_successor_len = self
            .recursive_folds
            .first()
            .map_or(self.terminal.input_witness_len, |step| {
                step.input_witness_len
            });
        if self.root.output_witness_len != first_successor_len {
            return Err(AkitaError::InvalidSetup(
                "root output witness length does not match its successor".to_string(),
            ));
        }
        let mut predecessor = &self.root.params.final_group.commitment;
        let root_shared_d = predecessor.role_dims().d_d();
        let mut predecessor_setup_d = self.root.params.precommitted_groups.iter().try_fold(
            predecessor.role_dims().common_relation_coeff_count(),
            |common, group| {
                let dims = group.commitment.role_dims(root_shared_d);
                crate::validate_role_dims(dims)?;
                Ok::<_, AkitaError>(common.min(dims.common_relation_coeff_count()))
            },
        )?;
        for (index, step) in self.recursive_folds.iter().enumerate() {
            if step.input_witness_len == 0 || step.output_witness_len == 0 {
                return Err(AkitaError::InvalidSetup(
                    "recursive fold witness lengths must be nonzero".to_string(),
                ));
            }
            if step.params.witness.setup_prefix != step.params.incoming_setup_prefix {
                return Err(AkitaError::InvalidSetup(format!(
                    "recursive fold {index} setup-prefix mirror disagrees with its successor edge"
                )));
            }
            if let Some(prefix) = &step.params.incoming_setup_prefix {
                let producer_dims = predecessor.role_dims();
                let consumer_dims = step.params.witness.role_dims();
                crate::validate_role_dims(producer_dims)?;
                crate::validate_role_dims(consumer_dims)?;
                let prefix_inner_d = prefix
                    .commitment_params
                    .layout
                    .inner_commit_matrix
                    .ring_dimension();
                let prefix_outer_d = prefix
                    .commitment_params
                    .layout
                    .outer_commit_matrix
                    .ring_dimension();
                if prefix.d_setup != predecessor_setup_d
                    || prefix_inner_d != consumer_dims.d_a()
                    || prefix_outer_d != consumer_dims.d_b()
                {
                    return Err(AkitaError::InvalidSetup(format!(
                        "recursive fold {index} setup offload geometry disagrees: \
                         producer roles {producer_dims:?} project at D{}, prefix source is D{}, \
                         prefix commitment roles are A{prefix_inner_d}/B{prefix_outer_d}, and \
                         consumer roles are {consumer_dims:?}",
                        predecessor_setup_d, prefix.d_setup,
                    )));
                }
            }
            let successor_len = self
                .recursive_folds
                .get(index + 1)
                .map_or(self.terminal.input_witness_len, |next| {
                    next.input_witness_len
                });
            if step.output_witness_len != successor_len {
                return Err(AkitaError::InvalidSetup(format!(
                    "recursive fold {index} output witness length does not match its successor"
                )));
            }
            predecessor = &step.params.witness;
            predecessor_setup_d = predecessor.role_dims().common_relation_coeff_count();
        }
        if self.terminal.input_witness_len == 0
            || self.terminal.params.response_shape.logical_num_elems() == 0
        {
            return Err(AkitaError::InvalidSetup(
                "terminal fold and response lengths must be nonzero".to_string(),
            ));
        }
        Ok(())
    }

    pub fn initial_witness_len(&self) -> usize {
        self.root.input_witness_len
    }

    /// Canonical byte encoding used to order semantically distinct schedules.
    ///
    /// This is an ordering descriptor, not a wire encoding or transcript
    /// commitment. It includes every schedule field that can affect proving or
    /// verification.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(1);
        match self.root.params.final_group.challenge {
            RootFinalChallenge::Flat => bytes.push(0),
            RootFinalChallenge::Tensor { fold_low_len } => {
                bytes.push(1);
                push_usize(bytes, fold_low_len);
            }
        }
        self.root
            .params
            .final_group
            .commitment
            .append_descriptor_bytes(bytes);
        push_usize(bytes, self.root.params.precommitted_groups.len());
        for group in &self.root.params.precommitted_groups {
            group.descriptor.append_descriptor_bytes(bytes);
            group.commitment.append_descriptor_bytes(bytes);
        }
        self.root
            .params
            .open_commit_matrix
            .append_descriptor_bytes(bytes);
        append_schedule_sparse_challenge_descriptor_bytes(
            bytes,
            &self.root.params.sparse_challenge_config,
        );
        append_witness_partition_descriptor_bytes(bytes, &self.root.params.witness_partition);
        push_usize(bytes, self.root.input_witness_len);
        push_usize(bytes, self.root.output_witness_len);
        push_usize(bytes, self.recursive_folds.len());
        for fold in &self.recursive_folds {
            fold.params.witness.append_descriptor_bytes(bytes);
            fold.params
                .open_commit_matrix
                .append_descriptor_bytes(bytes);
            append_schedule_sparse_challenge_descriptor_bytes(
                bytes,
                &fold.params.sparse_challenge_config,
            );
            match &fold.params.incoming_setup_prefix {
                None => bytes.push(0),
                Some(prefix) => {
                    bytes.push(1);
                    prefix.append_descriptor_bytes(bytes);
                }
            }
            append_witness_partition_descriptor_bytes(bytes, &fold.params.witness_partition);
            push_usize(bytes, fold.input_witness_len);
            push_usize(bytes, fold.output_witness_len);
        }
        bytes.push(3);
        self.terminal.params.witness.append_descriptor_bytes(bytes);
        append_schedule_sparse_challenge_descriptor_bytes(
            bytes,
            &self.terminal.params.sparse_challenge_config,
        );
        self.terminal
            .params
            .response_shape
            .append_descriptor_bytes(bytes);
        push_usize(bytes, self.terminal.input_witness_len);
    }
}

fn append_witness_partition_descriptor_bytes(bytes: &mut Vec<u8>, partition: &WitnessPartition) {
    match partition {
        WitnessPartition::Single => bytes.push(0),
        WitnessPartition::Distributed { num_chunks } => {
            bytes.push(1);
            push_usize(bytes, *num_chunks);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldScheduleEstimate {
    pub estimated_root_direct_payload_bytes: usize,
    pub estimated_root_stage3_payload_bytes: usize,
    pub estimated_recursive_direct_payload_bytes: Vec<usize>,
    pub estimated_recursive_stage3_payload_bytes: Vec<usize>,
    pub estimated_terminal_direct_payload_bytes: usize,
    pub estimated_terminal_response_payload_bytes: usize,
    /// Maximum setup-matrix envelope, in ring elements at the active level's
    /// inner ring dimension.
    pub estimated_setup_envelope_ring_elements: usize,
    /// Natural (unpadded) setup length at the first direct edge, when the
    /// recursive setup planner is active.
    pub first_direct_setup_field_len: Option<usize>,
    /// Number of recursive successors that consume an offloaded setup prefix.
    pub selected_offload_edges: usize,
}

impl FoldScheduleEstimate {
    pub fn estimated_direct_proof_payload_bytes(&self) -> Result<usize, AkitaError> {
        self.estimated_recursive_direct_payload_bytes
            .iter()
            .try_fold(self.estimated_root_direct_payload_bytes, |sum, value| {
                sum.checked_add(*value).ok_or_else(|| {
                    AkitaError::InvalidSetup("fold schedule estimate overflow".to_string())
                })
            })?
            .checked_add(self.estimated_terminal_direct_payload_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("fold schedule estimate overflow".to_string()))
    }

    pub fn estimated_stage3_payload_bytes(&self) -> Result<usize, AkitaError> {
        self.estimated_recursive_stage3_payload_bytes
            .iter()
            .try_fold(self.estimated_root_stage3_payload_bytes, |sum, value| {
                sum.checked_add(*value).ok_or_else(|| {
                    AkitaError::InvalidSetup("fold schedule estimate overflow".to_string())
                })
            })
    }

    pub fn estimated_proof_payload_bytes(&self) -> Result<usize, AkitaError> {
        self.estimated_direct_proof_payload_bytes()?
            .checked_add(self.estimated_stage3_payload_bytes()?)
            .ok_or_else(|| AkitaError::InvalidSetup("fold schedule estimate overflow".to_string()))
    }
}

#[derive(Clone, Debug)]
pub struct PlannedFoldSchedule {
    pub schedule: FoldSchedule,
    pub estimate: FoldScheduleEstimate,
}

/// Witness length entering the root fold, in field elements.
pub fn root_input_witness_len(lp: &CommittedGroupParams) -> usize {
    lp.num_live_blocks
        .checked_mul(lp.num_positions_per_block)
        .and_then(|len| len.checked_mul(lp.d_a()))
        .unwrap_or(0)
}
#[cfg(test)]
#[path = "schedule_tests.rs"]
mod tests;
