//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::layout::params::append_schedule_sparse_challenge_descriptor_bytes;
use crate::{
    CommittedGroupParams, InnerCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams,
    PolynomialGroupLayout, RelationAddressGeometry, SetupContributionMode, TerminalResponseShape,
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
    /// Bind the terminal compressed commitment payload and recurse.
    OuterPayload,
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
    pub const VERSION: u8 = 2;

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
/// Single-group openings use an empty `precommitteds` vector and store the
/// sole group in `final_group`. Multi-group roots list earlier groups in
/// `precommitteds` and the final group in `final_group`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Final group shape for the multi-group root commitment.
    pub final_group: PolynomialGroupLayout,
    /// Previously committed groups in caller-supplied transcript order.
    pub precommitteds: Vec<CommittedGroupProfile>,
}

impl AkitaScheduleLookupKey {
    /// Scalar root-opening context with no precommitted groups.
    pub fn single(final_group: PolynomialGroupLayout) -> Self {
        Self {
            final_group,
            precommitteds: Vec::new(),
        }
    }

    /// Canonical ordered bytes used for schedule-key identity tests and caches.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_usize(&mut bytes, self.final_group.num_vars());
        push_usize(&mut bytes, self.final_group.num_polynomials());
        push_usize(&mut bytes, self.precommitteds.len());
        for descriptor in &self.precommitteds {
            descriptor.append_descriptor_bytes(&mut bytes);
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
    /// This is the maximum group-local opening/EOR domain. It is intentionally
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
        if self.final_group.num_vars() == 0 {
            return Err(AkitaError::InvalidSetup(
                "schedule lookup key dimensions must be at least 1".to_string(),
            ));
        }
        for layout in &self.precommitteds {
            layout.group.validate()?;
            layout.validate(field_bits)?;
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
    let num_digits_fold = lp.num_digits_fold();
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
    let num_digits_fold = lp.num_digits_fold();
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
}

/// Minimum fraction of the unconstrained terminal-response target that a
/// fixed inner matrix must admit. This is a planner completeness heuristic,
/// not a security assumption: security always uses the matrix's exact
/// SIS-certified capacity.
pub const TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM: u128 = 1;
pub const TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN: u128 = 2;

impl TerminalCommittedGroupParams {
    pub fn from_expanded_group(params: CommittedGroupParams) -> Self {
        Self {
            log_basis_inner: params.log_basis_inner,
            inner_commit_matrix: params.inner_commit_matrix,
            num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
            num_positions_per_block: params.num_positions_per_block,
            num_live_blocks: params.num_live_blocks,
            num_digits_inner: params.num_digits_inner,
        }
    }

    /// Project an ordinary scalar group into terminal parameters and validate
    /// the directly checked response bound against its fixed inner matrix.
    pub fn try_from_expanded_group(
        params: CommittedGroupParams,
    ) -> Result<(Self, u128), AkitaError> {
        let sparse = params.fold_challenge_config;
        let num_fold_coeffs = usize::try_from(params.num_fold_coeffs()).map_err(|_| {
            AkitaError::InvalidSetup("terminal fold coefficient count exceeds usize".into())
        })?;
        let cap_config =
            crate::sis::FoldWitnessLinfCapConfig::for_fold_coeffs(&sparse, num_fold_coeffs)?;
        let challenge = crate::sis::FoldChallengeNorms::new(&sparse);
        let witness = crate::sis::FoldWitnessNorms::bounded(params.log_basis_inner, params.d_a());
        let (unconstrained_target, _) = crate::sis::fold_witness_unsnapped_linf_cap(
            params.num_live_blocks,
            1,
            challenge,
            witness,
            &cap_config,
        )?;
        let terminal = Self::from_expanded_group(params);
        let admission_cap = terminal.certified_response_linf_cap(&sparse)?;
        let minimum_usable_cap = unconstrained_target
            .checked_mul(TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal target ratio overflow".into()))?
            .div_ceil(TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN);
        if admission_cap < minimum_usable_cap {
            return Err(AkitaError::InvalidSetup(format!(
                "terminal response capacity {admission_cap} retains less than \
                 {TERMINAL_RESPONSE_MIN_TARGET_RETAIN_NUM}/\
                 {TERMINAL_RESPONSE_MIN_TARGET_RETAIN_DEN} of target {unconstrained_target}"
            )));
        }
        Ok((terminal, admission_cap))
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

    /// Largest raw response admitted by this fixed inner matrix and the signed
    /// terminal coefficient representation.
    pub fn certified_response_linf_cap(
        &self,
        sparse: &akita_challenges::SparseChallengeConfig,
    ) -> Result<u128, AkitaError> {
        let challenge = crate::sis::FoldChallengeNorms::new(sparse);
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
        Ok(certified_capacity.min(i16::MAX as u128))
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_u32(bytes, self.log_basis_inner);
        self.inner_commit_matrix.append_descriptor_bytes(bytes);
        push_usize(bytes, self.num_live_ring_elements_per_claim);
        push_usize(bytes, self.num_positions_per_block);
        push_usize(bytes, self.num_live_blocks);
        push_usize(bytes, self.num_digits_inner);
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
        if !self
            .root
            .params
            .final_group
            .commitment
            .payload_mode
            .is_compressed()
        {
            return Err(AkitaError::InvalidSetup(
                "root fold payload must be compressed".into(),
            ));
        }
        let mut payload_phase = crate::CommitmentPayloadPhase::CompressedPrefix;
        for (index, step) in self.recursive_folds.iter().enumerate() {
            let consumes_setup_prefix = step.params.witness.setup_prefix.is_some();
            if payload_phase == crate::CommitmentPayloadPhase::RawSuffix && consumes_setup_prefix {
                return Err(AkitaError::InvalidSetup(format!(
                    "recursive fold {index} cannot resume compression by consuming a setup prefix after the raw suffix"
                )));
            }
            if !payload_phase
                .candidate_modes(index + 1, consumes_setup_prefix)
                .contains(&step.params.witness.payload_mode)
            {
                return Err(AkitaError::InvalidSetup(format!(
                    "recursive fold {index} payload mode disagrees with the compression cutover policy"
                )));
            }
            payload_phase = payload_phase.after(step.params.witness.payload_mode);
        }
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
        let (first_successor_d, first_successor_opening_num_vars) =
            self.recursive_folds.first().map_or_else(
                || {
                    Ok((
                        self.terminal.params.witness.d_a(),
                        self.terminal.params.witness.recursive_opening_num_vars()?,
                    ))
                },
                |step| {
                    Ok((
                        step.params.witness.d_a(),
                        step.params.witness.recursive_opening_num_vars()?,
                    ))
                },
            )?;
        validate_stage2_successor_capacity(
            "root fold",
            &self.root.params.final_group.commitment,
            self.root.output_witness_len,
            first_successor_d,
            first_successor_opening_num_vars,
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
                let n_prefix = prefix.n_prefix()?;
                if prefix.natural_len == 0
                    || prefix.natural_len > n_prefix
                    || prefix.d_setup() == 0
                    || !n_prefix.is_multiple_of(prefix.d_setup())
                {
                    return Err(AkitaError::InvalidSetup(format!(
                        "recursive fold {index} setup-prefix geometry is invalid"
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
            let (successor_d, successor_opening_num_vars) =
                self.recursive_folds.get(index + 1).map_or_else(
                    || {
                        Ok((
                            self.terminal.params.witness.d_a(),
                            self.terminal.params.witness.recursive_opening_num_vars()?,
                        ))
                    },
                    |next| {
                        Ok((
                            next.params.witness.d_a(),
                            next.params.witness.recursive_opening_num_vars()?,
                        ))
                    },
                )?;
            validate_stage2_successor_capacity(
                &format!("recursive fold {index}"),
                &step.params.witness,
                step.output_witness_len,
                successor_d,
                successor_opening_num_vars,
            )?;
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

fn validate_stage2_successor_capacity(
    predecessor_name: &str,
    predecessor: &CommittedGroupParams,
    output_witness_len: usize,
    successor_ring_dimension: usize,
    successor_opening_num_vars: usize,
) -> Result<(), AkitaError> {
    // Stage 2 owns the predecessor-derived point. A successor may expose a
    // wider scheduled cube; preparation derives that wider representation by
    // zero-extension. The schedule must reject only points that do not fit.
    if successor_ring_dimension == 0 || !successor_ring_dimension.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "{predecessor_name} successor ring dimension {successor_ring_dimension} is invalid"
        )));
    }
    let shared_d = predecessor.role_dims().d_d();
    let precommitted_role_dims = predecessor
        .precommitted_group_iter()
        .map(|group| group.role_dims(shared_d))
        .collect::<Vec<_>>();
    let geometry = RelationAddressGeometry::new_for_groups(
        predecessor.role_dims(),
        &precommitted_role_dims,
        successor_ring_dimension,
        output_witness_len,
    )?;
    let stage2_num_vars = geometry.relation_point_variable_count();
    if stage2_num_vars > successor_opening_num_vars {
        return Err(AkitaError::InvalidSetup(format!(
            "{predecessor_name} Stage 2 point has {stage2_num_vars} variables, exceeding \
             successor opening capacity {successor_opening_num_vars}"
        )));
    }
    Ok(())
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
    /// Maximum flat setup-matrix capacity required by the schedule.
    pub estimated_num_setup_field_elements: usize,
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
