//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::layout::params::append_schedule_sparse_challenge_descriptor_bytes;
use crate::sis::FoldWitnessLinfCapConfig;
use crate::{
    CommittedGroupParams, InnerCommitMatrixParams, OpeningClaimsLayout, PolynomialGroupLayout,
    SetupContributionMode, TerminalResponseShape,
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
pub struct PrecommittedGroupDescriptor {
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
    /// Gadget basis selected for the standalone B/`t_hat` digits.
    pub log_basis_outer: u32,
    /// Conservative A-role row count frozen at precommit time.
    pub n_a: usize,
    /// Conservative A-role collision bucket frozen at precommit time.
    pub a_coeff_linf_bound: u128,
    /// Conservative B-role row count frozen at precommit time.
    pub n_b: usize,
    /// Conservative B-role collision bucket frozen at precommit time.
    pub b_coeff_linf_bound: u128,
}

impl PrecommittedGroupDescriptor {
    /// Build frozen group metadata from the concrete commit params.
    pub fn from_params(group: PolynomialGroupLayout, params: &CommittedGroupParams) -> Self {
        Self {
            group,
            num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
            num_positions_per_block: params.num_positions_per_block,
            num_live_blocks: params.num_live_blocks,
            log_basis_inner: params.log_basis_inner,
            log_basis_outer: params.log_basis_outer,
            n_a: params.inner_commit_matrix.output_rank(),
            a_coeff_linf_bound: params.inner_commit_matrix.coeff_linf_bound(),
            n_b: params.outer_commit_matrix.output_rank(),
            b_coeff_linf_bound: params.outer_commit_matrix.coeff_linf_bound(),
        }
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.group.num_vars());
        push_usize(bytes, self.group.num_polynomials());
        push_usize(bytes, self.num_live_ring_elements_per_claim);
        push_usize(bytes, self.num_positions_per_block);
        push_usize(bytes, self.num_live_blocks);
        push_u32(bytes, self.log_basis_inner);
        push_u32(bytes, self.log_basis_outer);
        push_usize(bytes, self.n_a);
        crate::descriptor_bytes::push_u128(bytes, self.a_coeff_linf_bound);
        push_usize(bytes, self.n_b);
        crate::descriptor_bytes::push_u128(bytes, self.b_coeff_linf_bound);
    }

    /// Validate that this layout is a well-formed standalone commitment group.
    pub fn validate(&self) -> Result<(), AkitaError> {
        self.group.validate()?;
        if self.group.num_polynomials() != 1 {
            return Err(AkitaError::InvalidSetup(format!(
                "precommitted groups must contain exactly one polynomial, got {}",
                self.group.num_polynomials()
            )));
        }
        if self.n_a == 0
            || self.n_b == 0
            || self.a_coeff_linf_bound == 0
            || self.b_coeff_linf_bound == 0
        {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero conservative A/B rows and bounds"
                    .to_string(),
            ));
        }
        if self.log_basis_inner == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero log_basis_inner".to_string(),
            ));
        }
        if self.log_basis_outer == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero log_basis_outer".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate that frozen exact block geometry matches `group.num_vars`.
    pub fn validate_root_geometry(&self, ring_dimension: usize) -> Result<(), AkitaError> {
        let alpha = ring_dimension.trailing_zeros() as usize;
        let Some(source_field_len) = self
            .num_live_ring_elements_per_claim
            .checked_mul(ring_dimension)
        else {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout geometry overflow".to_string(),
            ));
        };
        let expected_field_len = 1usize
            .checked_shl(self.group.num_vars() as u32)
            .ok_or_else(|| {
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
    pub fn validate_frozen_precommit(&self, ring_dimension: usize) -> Result<(), AkitaError> {
        self.validate()?;
        self.validate_root_geometry(ring_dimension)?;
        Ok(())
    }
}

/// Freezes conservative root-commit metadata for each precommitted group when
/// building a schedule lookup key from an opening layout.
pub trait ScheduleKeyPrecommitSource {
    /// Resolve frozen standalone-commit params for one precommitted group.
    fn precommitted_group_params(
        group: PolynomialGroupLayout,
    ) -> Result<PrecommittedGroupDescriptor, AkitaError>;
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
    /// Previously committed groups in caller-supplied transcript order.
    pub precommitteds: Vec<PrecommittedGroupDescriptor>,
}

impl AkitaScheduleLookupKey {
    /// Scalar root-opening context with no precommitted groups.
    pub fn single(final_group: PolynomialGroupLayout) -> Self {
        Self {
            final_group,
            precommitteds: Vec::new(),
        }
    }

    /// Build the canonical schedule lookup key for `layout`.
    ///
    /// Scalar layouts leave `precommitteds` empty. Grouped layouts freeze each
    /// earlier group through `S` (production uses the conservative commit
    /// adapter wired by `akita-config`'s `opening_schedule_key`).
    pub fn from_layout<S: ScheduleKeyPrecommitSource>(
        layout: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        layout.check()?;
        let precommitteds = if layout.num_groups() == 1 {
            Vec::new()
        } else {
            layout
                .root_precommitted_group_layouts()?
                .iter()
                .copied()
                .map(S::precommitted_group_params)
                .collect::<Result<Vec<_>, _>>()?
        };
        let key = Self {
            final_group: layout.root_final_group_layout()?,
            precommitteds,
        };
        key.validate()?;
        Ok(key)
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

    /// Validate per-group metadata.
    pub fn validate(&self) -> Result<(), AkitaError> {
        self.final_group.validate()?;
        if self.final_group.num_vars() == 0 {
            return Err(AkitaError::InvalidSetup(
                "schedule lookup key dimensions must be at least 1".to_string(),
            ));
        }
        for layout in &self.precommitteds {
            layout.group.validate()?;
            if layout.group.num_vars() > self.final_group.num_vars() / 2 {
                return Err(AkitaError::InvalidInput(
                    "multi-group root requires precommitted groups to have at most half the final num_vars"
                        .to_string(),
                ));
            }
            layout.validate()?;
        }
        Ok(())
    }
}

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::sis::compute_num_digits_full_field(field_bits, log_basis)
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
        .checked_mul(crate::sis::compute_num_digits_full_field(
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RootSource {
    Dense { coefficient_bits: u32 },
    OneHot { chunk_size: usize },
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
    pub source: RootSource,
    pub challenge: RootFinalChallenge,
    pub commitment: CommittedGroupParams,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootPrecommittedGroupParams {
    pub descriptor: PrecommittedGroupDescriptor,
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

    /// Project an ordinary scalar group into terminal parameters and resize
    /// its fixed inner matrix for the unsnapped raw-response collision bound.
    pub fn try_from_expanded_group(
        params: CommittedGroupParams,
    ) -> Result<(Self, u128), AkitaError> {
        let sparse = params.fold_challenge_config;
        let mut terminal = Self::from_expanded_group(params);
        let challenge = crate::sis::FoldChallengeNorms::new(
            &sparse,
            akita_challenges::TensorChallengeShape::Flat,
        );
        let witness =
            crate::sis::FoldWitnessNorms::new(terminal.log_basis_inner, terminal.d_a(), 1, false);
        let (honest_cap, _) = crate::sis::fold_witness_unsnapped_linf_cap(
            terminal.num_live_blocks,
            1,
            challenge,
            witness,
            &terminal.fold_linf_cap_config,
        )?;
        let old_key = terminal.inner_commit_matrix.sis_table_key();
        let collision = crate::sis::role_a_collision_inf_norm_for_response_bound(
            challenge.l1_norm,
            old_key.modulus_profile.ring_subfield_embedding_norm_bound(),
            honest_cap,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("terminal A collision overflow".into()))?;
        let collision_bucket = crate::sis::ceil_supported_linf_bound(
            old_key.policy,
            old_key.table_digest,
            old_key.modulus_profile,
            crate::sis::SisMatrixRole::Inner,
            old_key.ring_dimension,
            collision,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup("terminal A collision exceeds the SIS table".into())
        })?;
        terminal.inner_commit_matrix = crate::sis::InnerCommitMatrixParams::try_new_with_min_rank(
            crate::sis::SisTableKey {
                coeff_linf_bound: collision_bucket,
                ..old_key
            },
            terminal.inner_width(),
        )?;
        terminal.response_linf_bounds(&sparse)?;
        Ok((terminal, honest_cap))
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

    /// Derive the honest terminal response cap and the independently
    /// certified A-collision limit from the exact typed terminal parameters.
    /// This is the single protocol source used by planning, encoding, and
    /// verifier admission.
    pub fn response_linf_bounds(
        &self,
        sparse: &akita_challenges::SparseChallengeConfig,
    ) -> Result<(u128, u128), AkitaError> {
        let challenge = crate::sis::FoldChallengeNorms::new(
            sparse,
            akita_challenges::TensorChallengeShape::Flat,
        );
        let witness = crate::sis::FoldWitnessNorms::new(self.log_basis_inner, self.d_a(), 1, false);
        let (honest_cap, _) = crate::sis::fold_witness_unsnapped_linf_cap(
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
        let security_cap = crate::sis::max_response_linf_for_role_a_collision(
            collision_capacity,
            challenge.l1_norm,
            self.inner_commit_matrix
                .sis_modulus_profile()
                .ring_subfield_embedding_norm_bound(),
        )
        .filter(|value| *value > 0)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal A cannot certify a response".into()))?;
        if honest_cap > security_cap {
            return Err(AkitaError::InvalidSetup(
                "terminal honest response cap exceeds A collision capacity".into(),
            ));
        }
        Ok((honest_cap, security_cap))
    }

    /// Validate the terminal Fiat–Shamir grind nonce under the same bound
    /// policy used to derive the response wire.
    pub fn validate_fold_grind_nonce(
        &self,
        sparse: &akita_challenges::SparseChallengeConfig,
        nonce: u32,
    ) -> Result<(), AkitaError> {
        let (honest_cap, _) = self.response_linf_bounds(sparse)?;
        crate::sis::FoldWitnessGrindContract {
            policy: self.fold_linf_cap_config.policy,
            witness_linf_cap: honest_cap,
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
        for (index, step) in self.recursive_folds.iter().enumerate() {
            if step.input_witness_len == 0 || step.output_witness_len == 0 {
                return Err(AkitaError::InvalidSetup(
                    "recursive fold witness lengths must be nonzero".to_string(),
                ));
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

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(1);
        match self.root.params.final_group.source {
            RootSource::Dense { coefficient_bits } => {
                bytes.push(0);
                push_u32(bytes, coefficient_bits);
            }
            RootSource::OneHot { chunk_size } => {
                bytes.push(1);
                push_usize(bytes, chunk_size);
            }
        }
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
    pub estimated_recursive_direct_payload_bytes: Vec<usize>,
    pub estimated_terminal_direct_payload_bytes: usize,
    pub estimated_terminal_response_payload_bytes: usize,
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
