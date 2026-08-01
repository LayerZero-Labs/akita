//! Unified per-level parameters for the Akita protocol.
//!
//! `CommittedGroupParams` merges ring dimension, matrix ranks, challenge config,
//! block geometry, and digit depths into a single struct that fully
//! describes one recursion level.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::{AkitaError, CanonicalField};

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::layout::ring_dims::CommitmentRingDims;
use crate::opening_claims::OpeningClaimsLayout;
use crate::proof::{RelationAddressGeometry, SetupPrefixSlotId};

pub use crate::sis::{
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams, SisModulusProfileId,
};

pub(crate) fn recursive_opening_num_vars_for_geometry(
    d_a: usize,
    num_positions_per_block: usize,
    num_live_blocks: usize,
) -> Result<usize, AkitaError> {
    if d_a == 0
        || !d_a.is_power_of_two()
        || num_positions_per_block == 0
        || !num_positions_per_block.is_power_of_two()
        || num_live_blocks == 0
    {
        return Err(AkitaError::InvalidSetup(
            "invalid recursive opening geometry".to_string(),
        ));
    }
    (d_a.trailing_zeros() as usize)
        .checked_add(num_positions_per_block.trailing_zeros() as usize)
        .and_then(|bits| {
            num_live_blocks
                .checked_next_power_of_two()
                .and_then(|blocks| bits.checked_add(blocks.trailing_zeros() as usize))
        })
        .ok_or_else(|| AkitaError::InvalidSetup("recursive opening num_vars overflow".to_string()))
}

mod descriptor;
mod precommitted;
pub(crate) use descriptor::append_sparse_challenge_descriptor_bytes as append_schedule_sparse_challenge_descriptor_bytes;
use descriptor::append_tensor_challenge_shape_descriptor_bytes;
pub use precommitted::{LevelParamsLike, PrecommittedLevelParams};

/// Gadget basis used by opening-digit segments in the shared D product.
///
/// A grouped root concatenates the main group's `e_hat` with every
/// precommitted group's fresh `e_hat`; all fresh opening digits use the root
/// opening basis.
#[must_use]
pub fn shared_d_digit_log_basis(
    main_log_basis: u32,
    _precommitted_groups: &[PrecommittedLevelParams],
) -> u32 {
    main_log_basis
}

/// Unified per-level parameters for one Akita recursion level.
///
/// Combines ring dimension, Ajtai matrix descriptions, block geometry,
/// sparse-challenge configuration, and digit decomposition depths into a
/// single authoritative struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedGroupParams {
    /// Base-2 logarithm of the A/source gadget decomposition base.
    pub log_basis_inner: u32,
    /// Base-2 logarithm of the B/`t_hat` gadget decomposition base.
    pub log_basis_outer: u32,
    /// Base-2 logarithm of the D/`e_hat` gadget decomposition base.
    pub log_basis_open: u32,
    /// Inner Ajtai matrix (A): output rank `n_a`, input width `inner_width`.
    pub inner_commit_matrix: InnerCommitMatrixParams,
    /// Outer commitment matrix (B): output rank `n_b`, input width `outer_width`.
    pub outer_commit_matrix: OuterCommitMatrixParams,
    /// Opening matrix (D): output rank `n_d`, input width `d_matrix_width`.
    pub open_commit_matrix: OpenCommitMatrixParams,
    /// Exact number of live source ring elements per claim (`N`).
    pub num_live_ring_elements_per_claim: usize,
    /// Number of positions per block (`M`), power-of-two in the current Boolean layout.
    pub num_positions_per_block: usize,
    /// Exact number of live blocks (`B = ceil(N / M)`).
    pub num_live_blocks: usize,
    pub fold_challenge_config: SparseChallengeConfig,
    /// Shape of the stage-1 fold-round challenge vector at this level.
    ///
    /// Defaults to [`TensorChallengeShape::Flat`]. Tensor presets set selected
    /// levels to [`TensorChallengeShape::Tensor`] during schedule construction.
    pub fold_challenge_shape: TensorChallengeShape,
    /// Gadget decomposition depth for A/source coefficients.
    pub num_digits_inner: usize,
    /// Gadget decomposition depth for B/`t_hat` values.
    pub num_digits_outer: usize,
    /// Gadget decomposition depth for D/opening evaluations.
    pub num_digits_open: usize,
    /// Exact folded-witness digit depth selected by this schedule row.
    pub num_digits_fold: usize,
    /// Multi-chunk witness layout for this level (default: single-chunk).
    ///
    /// The planner populates this from `policy.witness_chunk` and the level's
    /// position in the fold recursion; the verifier consumes it as the source of
    /// truth for the per-level witness column layout. `ChunkedWitnessCfg::default()`
    /// (single chunk) is byte-identical to the historical layout.
    pub witness_chunk: crate::witness::ChunkedWitnessCfg,
    /// Precommitted group-local params for a multi-group root. Empty for scalar
    /// levels; when non-empty, the top-level fields describe the final/new
    /// group and `open_commit_matrix` describes the shared D matrix over all group `w_hat`
    /// segments.
    pub precommitted_groups: Vec<PrecommittedLevelParams>,
    /// Derived runtime mirror of the successor-owned setup-prefix edge.
    ///
    /// [`crate::RecursiveFoldParams::incoming_setup_prefix`] is authoritative;
    /// [`crate::FoldSchedule::validate_structure`] rejects disagreement before
    /// prover or verifier execution.
    pub setup_prefix: Option<SetupPrefixSlotId>,
}

impl CommittedGroupParams {
    /// Largest gadget basis accepted by this level's shared D product.
    #[must_use]
    pub fn shared_d_digit_log_basis(&self) -> u32 {
        shared_d_digit_log_basis(self.log_basis_open, &self.precommitted_groups)
    }

    /// Per-role ring dimensions derived from the three matrix objects.
    #[must_use]
    pub fn role_dims(&self) -> CommitmentRingDims {
        CommitmentRingDims {
            inner: self.inner_commit_matrix.ring_dimension(),
            outer: self.outer_commit_matrix.ring_dimension(),
            opening: self.open_commit_matrix.ring_dimension(),
        }
    }

    /// A-role ring dimension (`d_a`); alias of [`CommitmentRingDims::d_a`] on [`Self::role_dims`].
    #[inline]
    #[must_use]
    pub fn d_a(&self) -> usize {
        self.inner_commit_matrix.ring_dimension()
    }

    /// Build a params-only `CommittedGroupParams` with zeroed layout fields.
    ///
    /// Only ring dimension, matrix row counts, log_basis, and fold_challenge_config
    /// are populated. Column counts, block geometry, and digit depths are
    /// zeroed. Call `with_layout` to fill them from a derived layout.
    pub fn params_only(
        sis_modulus_profile: SisModulusProfileId,
        ring_dimension: usize,
        log_basis: u32,
        n_a: usize,
        n_b: usize,
        n_d: usize,
        fold_challenge_config: SparseChallengeConfig,
    ) -> Self {
        Self {
            log_basis_inner: log_basis,
            log_basis_outer: log_basis,
            log_basis_open: log_basis,
            inner_commit_matrix: InnerCommitMatrixParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                crate::sis::SisTableDigest::CURRENT,
                sis_modulus_profile,
                n_a,
                0,
                0,
                ring_dimension,
            ),
            outer_commit_matrix: OuterCommitMatrixParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                crate::sis::SisTableDigest::CURRENT,
                sis_modulus_profile,
                n_b,
                0,
                0,
                ring_dimension,
            ),
            open_commit_matrix: OpenCommitMatrixParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                crate::sis::SisTableDigest::CURRENT,
                sis_modulus_profile,
                n_d,
                0,
                0,
                ring_dimension,
            ),
            num_live_ring_elements_per_claim: 0,
            num_positions_per_block: 0,
            num_live_blocks: 0,
            fold_challenge_config,
            fold_challenge_shape: TensorChallengeShape::Flat,
            num_digits_inner: 0,
            num_digits_outer: 0,
            num_digits_open: 0,
            num_digits_fold: 1,
            witness_chunk: crate::witness::ChunkedWitnessCfg::default_non_chunked(),
            precommitted_groups: Vec::new(),
            setup_prefix: None,
        }
    }

    /// True when this level carries multi-group-root metadata.
    #[inline]
    pub fn has_precommitted_groups(&self) -> bool {
        self.precommitted_group_count() != 0
    }

    #[inline]
    pub fn precommitted_group_count(&self) -> usize {
        self.setup_prefix
            .as_ref()
            .map_or(0usize, |_| 1usize)
            .saturating_add(self.precommitted_groups.len())
    }

    #[inline]
    pub fn precommitted_group_params(
        &self,
        group_index: usize,
    ) -> Option<&PrecommittedLevelParams> {
        if let Some(setup_prefix) = &self.setup_prefix {
            if group_index == 0 {
                return Some(&setup_prefix.commitment_params);
            }
            return self.precommitted_groups.get(group_index - 1);
        }
        self.precommitted_groups.get(group_index)
    }

    #[inline]
    pub fn precommitted_group_iter(&self) -> impl Iterator<Item = &PrecommittedLevelParams> {
        self.setup_prefix
            .as_ref()
            .map(|setup_prefix| &setup_prefix.commitment_params)
            .into_iter()
            .chain(self.precommitted_groups.iter())
    }

    /// Reject multi-group-root params at scalar-only call sites.
    pub fn require_scalar_level(&self, context: &str) -> Result<(), AkitaError> {
        if self.has_precommitted_groups() {
            return Err(AkitaError::InvalidSetup(format!(
                "{context} requires scalar root level params"
            )));
        }
        Ok(())
    }

    /// Worst-case L1 mass of the fold-round challenge.
    #[inline]
    pub fn challenge_l1_mass(&self) -> usize {
        self.fold_challenge_shape
            .effective_l1_mass(&self.fold_challenge_config)
    }

    /// Effective fold-round challenge L∞ norm `||c||_inf` at this level,
    /// accounting for the challenge shape (flat vs tensor).
    #[inline]
    pub fn challenge_infinity_norm(&self) -> usize {
        self.fold_challenge_shape
            .effective_infinity_norm(&self.fold_challenge_config)
    }

    /// Effective per-block worst-case `‖c‖_2²` upper bound at this fold level.
    #[inline]
    pub fn challenge_l2_sq_max(&self) -> u128 {
        self.fold_challenge_shape
            .effective_l2_sq_max(&self.fold_challenge_config)
    }

    /// Fold-challenge coefficient count `inner_width · D`.
    #[inline]
    pub fn num_fold_coeffs(&self) -> u128 {
        (self.inner_width() as u128).saturating_mul(self.d_a() as u128)
    }

    /// Validate the shared fold nonce from schedule-owned challenge policies.
    ///
    /// This verifier boundary deliberately does not reconstruct an honest
    /// source model or an honest folded-response cap. Those values guide the
    /// prover's search only; nonce admission is fixed by each selected
    /// group's challenge family, shape, and native A dimension.
    pub fn validate_fold_grind_nonce(
        &self,
        opening_batch: &OpeningClaimsLayout,
        max_grind_attempts: u32,
        fold_grind_nonce: u32,
    ) -> Result<(), AkitaError> {
        self.validate_opening_batch(opening_batch)?;
        if max_grind_attempts == 0 {
            return Err(AkitaError::InvalidSetup(
                "fold grind attempt budget must be positive".to_string(),
            ));
        }
        if fold_grind_nonce >= max_grind_attempts {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Exact scheduled gadget decomposition depth for the folded witness.
    #[inline]
    pub fn num_digits_fold(&self) -> usize {
        self.num_digits_fold
    }

    /// Maximum terminal folded-response norm certified by a group's fixed A matrix.
    ///
    /// This inverts the checked-in SIS table for the matrix's exact width and
    /// rank, then applies the complete A-role weak-binding formula. It performs
    /// no online lattice estimation and does not use the honest-response cap.
    pub fn terminal_response_linf_limit_for_params(
        &self,
        params: &(impl LevelParamsLike + ?Sized),
    ) -> Result<u128, AkitaError> {
        let inner_commit_matrix = params.inner_commit_matrix_params();
        if inner_commit_matrix.sis_table_key().role != crate::sis::SisMatrixRole::Inner {
            return Err(AkitaError::InvalidSetup(
                "terminal response requires an A-role inner matrix".to_string(),
            ));
        }
        let collision_capacity =
            inner_commit_matrix
                .max_secure_collision_linf()
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "terminal inner matrix has no supported SIS collision capacity".to_string(),
                    )
                })?;
        let challenge = crate::sis::FoldChallengeNorms::new(
            &self.fold_challenge_config,
            params.fold_challenge_shape(),
        );
        crate::sis::max_response_linf_for_role_a_collision(
            collision_capacity,
            challenge.l1_norm,
            inner_commit_matrix
                .sis_modulus_profile()
                .ring_subfield_embedding_norm_bound(),
        )
        .filter(|&limit| limit > 0)
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "terminal inner matrix cannot certify a nonzero folded response".to_string(),
            )
        })
    }

    /// Number of Boolean coordinates in the block-index domain.
    #[inline]
    pub fn block_index_bits(&self) -> usize {
        self.num_live_blocks
            .checked_next_power_of_two()
            .map_or(0, |capacity| capacity.trailing_zeros() as usize)
    }

    /// Number of Boolean coordinates in one block-position slice.
    #[inline]
    pub fn position_index_bits(&self) -> usize {
        self.num_positions_per_block.trailing_zeros() as usize
    }

    /// Boolean block-index domain size (`next_power_of_two(B)`).
    #[inline]
    pub fn block_index_domain_size(&self) -> Result<usize, AkitaError> {
        self.num_live_blocks
            .checked_next_power_of_two()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("block-index domain size overflows usize".to_string())
            })
    }

    /// Validate the exact source/block geometry before it reaches allocation.
    pub fn validate_block_geometry(&self) -> Result<(), AkitaError> {
        if self.num_live_ring_elements_per_claim == 0
            || self.num_positions_per_block == 0
            || !self.num_positions_per_block.is_power_of_two()
            || self.num_live_blocks == 0
        {
            return Err(AkitaError::InvalidSetup(
                "invalid digit-innermost block geometry".to_string(),
            ));
        }
        let expected = self
            .num_live_ring_elements_per_claim
            .div_ceil(self.num_positions_per_block);
        if self.num_live_blocks != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "num_live_blocks={} does not equal ceil(num_live_ring_elements_per_claim={} / num_positions_per_block={})={expected}",
                self.num_live_blocks,
                self.num_live_ring_elements_per_claim,
                self.num_positions_per_block,
            )));
        }
        self.block_index_domain_size()?;
        Ok(())
    }

    /// Width of inner matrix A (column count of the A-key).
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.inner_commit_matrix.input_width()
    }

    /// Exact live source ring elements in one claim.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on overflow.
    pub fn n_ring_elems(&self) -> Result<usize, AkitaError> {
        self.validate_block_geometry()?;
        Ok(self.num_live_ring_elements_per_claim)
    }

    /// Total flat field-element count (`n_ring_elems * d_a`).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on overflow.
    pub fn flat_field_len(&self) -> Result<usize, AkitaError> {
        let n_ring_elems = self.n_ring_elems()?;
        n_ring_elems.checked_mul(self.d_a()).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "n_ring_elems={n_ring_elems} * d_a={} overflows usize",
                self.d_a(),
            ))
        })
    }

    /// Append the descriptor digest encoding for this parameter set.
    ///
    /// Kept next to [`CommittedGroupParams`] so protocol-affecting field changes are
    /// reviewed with their Fiat-Shamir binding.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_u32(bytes, self.log_basis_inner);
        push_u32(bytes, self.log_basis_outer);
        push_u32(bytes, self.log_basis_open);
        self.inner_commit_matrix.append_descriptor_bytes(bytes);
        self.outer_commit_matrix.append_descriptor_bytes(bytes);
        self.open_commit_matrix.append_descriptor_bytes(bytes);
        push_usize(bytes, self.num_live_ring_elements_per_claim);
        push_usize(bytes, self.num_positions_per_block);
        push_usize(bytes, self.num_live_blocks);
        append_schedule_sparse_challenge_descriptor_bytes(bytes, &self.fold_challenge_config);
        append_tensor_challenge_shape_descriptor_bytes(bytes, self.fold_challenge_shape);
        push_usize(bytes, self.num_digits_inner);
        push_usize(bytes, self.num_digits_outer);
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.num_digits_fold);
        // Chunk binding is appended only when the level is chunked, so
        // single-chunk descriptors stay byte-for-byte identical to the historical
        // layout (the flag-off no-op invariant). When chunked, bind the chunk
        // count and activated-level count into the Fiat-Shamir digest.
        if self.witness_chunk.num_chunks != 1 {
            self.witness_chunk.append_descriptor_bytes(bytes);
        }

        if !self.precommitted_groups.is_empty() {
            push_usize(bytes, self.precommitted_groups.len());
            for group in &self.precommitted_groups {
                group.append_descriptor_bytes(bytes);
            }
        }
        if let Some(setup_prefix) = &self.setup_prefix {
            bytes.push(1);
            setup_prefix.append_descriptor_bytes(bytes);
        } else {
            bytes.push(0);
        }
    }

    /// Width of outer matrix B (column count of the B-key).
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.outer_commit_matrix.input_width()
    }

    /// Width of prover matrix D (column count of the D-key).
    #[inline]
    pub fn d_matrix_width(&self) -> usize {
        self.open_commit_matrix.input_width()
    }

    /// Total outer variable count (`block_index_bits + position_index_bits`).
    #[inline]
    pub fn outer_vars(&self) -> usize {
        self.block_index_bits() + self.position_index_bits()
    }

    /// Logical opening-point variable count for recursive fold levels.
    ///
    /// Uses the direct `[position bits | fold bits]` source split plus the
    /// inner `log2(d_a)` coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if the summed dimension overflows `usize`.
    pub fn recursive_opening_num_vars(&self) -> Result<usize, AkitaError> {
        self.validate_block_geometry()?;
        recursive_opening_num_vars_for_geometry(
            self.d_a(),
            self.num_positions_per_block,
            self.num_live_blocks,
        )
    }

    // ---- Canonical relation-matrix row layout offsets (single source of truth) ----
    //
    // Scalar row layout: consistency (1) | A (n_a) | B (n_b · nc) | D.
    // Multi-group row layout: [consistency_g | A_g | B_g]_g | D.
    // Public-output rows bind through the fused trace term, not the M-matrix.
    // Every row-offset site (prover quotient/`generate_relation_rhs`, setup-contribution
    // `prepare`, the relation claim, the verifier ring-switch row eval) must
    // derive its block starts from these helpers rather than recompute inline.

    #[inline]
    fn relation_matrix_row_overflow() -> AkitaError {
        AkitaError::InvalidSetup("relation-matrix row count overflow".to_string())
    }

    /// Absolute start row of the A block (immediately after the consistency row).
    #[inline]
    pub fn a_start(&self) -> usize {
        1
    }

    /// Absolute start row of the B block.
    #[inline]
    pub fn b_start(&self) -> Result<usize, AkitaError> {
        self.a_start()
            .checked_add(self.inner_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)
    }

    /// Absolute start row of the D block.
    #[inline]
    pub fn d_start(&self, num_commitments: usize) -> Result<usize, AkitaError> {
        let b_rows = self
            .outer_commit_matrix
            .output_rank()
            .checked_mul(num_commitments)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        self.b_start()?
            .checked_add(b_rows)
            .ok_or_else(Self::relation_matrix_row_overflow)
    }

    /// Number of commitment groups in this opening batch (`precommitted + final`).
    #[inline]
    fn group_count(&self) -> usize {
        self.precommitted_group_count() + 1
    }

    /// Build the canonical root opening layout around one final group.
    pub fn opening_layout_for_final_group(
        &self,
        final_group: crate::PolynomialGroupLayout,
    ) -> Result<OpeningClaimsLayout, AkitaError> {
        let precommitted = self
            .precommitted_group_iter()
            .map(|group| group.layout.group)
            .collect::<Vec<_>>();
        OpeningClaimsLayout::from_root_groups(&precommitted, final_group)
    }

    pub fn validate_opening_batch(
        &self,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        opening_batch.check()?;
        if self.log_basis_open < self.log_basis_inner || self.log_basis_open < self.log_basis_outer
        {
            return Err(AkitaError::InvalidSetup(
                "certified opening basis must dominate level inner/outer bases".to_string(),
            ));
        }
        if opening_batch.num_groups() != self.group_count() {
            return Err(AkitaError::InvalidSetup(
                "opening group count does not match level params".to_string(),
            ));
        }
        for group_index in 0..self.precommitted_group_count() {
            let group_params = self
                .precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            group_params.validate()?;
            if group_params.log_basis_open != self.log_basis_open {
                return Err(AkitaError::InvalidSetup(
                    "all opening groups must use the batch-shared opening basis".to_string(),
                ));
            }
            let group_layout = opening_batch.group_layout(group_index)?;
            if *group_layout != group_params.layout.group {
                return Err(AkitaError::InvalidSetup(
                    "precommitted group layout does not match level params".to_string(),
                ));
            }
        }
        opening_batch.root_final_group_index()
    }

    /// Resolve one opening group's A/B dimensions with this level's shared D.
    ///
    /// The final group owns the level-level A/B matrices. Precommitted groups
    /// own their own A/B matrices; none owns a separate D matrix.
    pub fn group_role_dims(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<CommitmentRingDims, AkitaError> {
        let final_group_index = self.validate_opening_batch(opening_batch)?;
        let dims = if group_index == final_group_index {
            self.role_dims()
        } else {
            self.precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .role_dims(self.open_commit_matrix.ring_dimension())
        };
        dims.validate_role_projection()?;
        Ok(dims)
    }

    /// Ring dimension of the batch-owned recursive-witness carrier.
    ///
    /// Every group keeps its native A dimension for fold and relation
    /// arithmetic. The physical witness carrier uses the largest group A
    /// dimension so support is independent of caller group order.
    #[must_use]
    pub fn relation_witness_carrier_ring_dimension(&self) -> usize {
        self.precommitted_group_iter()
            .map(|group| group.layout.inner_commit_matrix.ring_dimension())
            .fold(self.d_a(), usize::max)
    }

    /// Resolve flat relation-address geometry across every opening group's
    /// native A/B dimensions and this level's shared D dimension.
    pub fn relation_address_geometry(
        &self,
        opening_batch: &OpeningClaimsLayout,
        outgoing_witness_ring_dimension: usize,
        live_witness_coeff_len: usize,
    ) -> Result<RelationAddressGeometry, AkitaError> {
        self.validate_opening_batch(opening_batch)?;
        let group_role_dims = (0..opening_batch.num_groups())
            .map(|group_index| self.group_role_dims(opening_batch, group_index))
            .collect::<Result<Vec<_>, _>>()?;
        RelationAddressGeometry::new_for_groups(
            self.role_dims(),
            &group_role_dims,
            outgoing_witness_ring_dimension,
            live_witness_coeff_len,
        )
    }

    /// Sent commitment row count for one opening group.
    pub fn group_commitment_rows(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<usize, AkitaError> {
        let final_group_index = self.validate_opening_batch(opening_batch)?;
        if group_index == final_group_index {
            return Ok(self.outer_commit_matrix.output_rank());
        }
        self.precommitted_group_params(group_index)
            .map(|group| group.layout.outer_commit_matrix.output_rank())
            .ok_or(AkitaError::InvalidProof)
    }

    /// Group-local parameter view for folded opening work.
    pub fn group_params<'a>(
        &'a self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<&'a dyn LevelParamsLike, AkitaError> {
        let final_group_index = self.validate_opening_batch(opening_batch)?;
        if group_index == final_group_index {
            return Ok(self);
        }
        self.precommitted_group_params(group_index)
            .map(|group| group as &dyn LevelParamsLike)
            .ok_or(AkitaError::InvalidProof)
    }

    fn multi_group_relation_matrix_row_count_for(
        &self,
        num_commitments: usize,
    ) -> Result<usize, AkitaError> {
        if num_commitments != self.group_count() {
            return Err(AkitaError::InvalidSetup(
                "multi-group relation rows require the real group count".to_string(),
            ));
        }

        let mut rows = 1usize
            .checked_add(self.inner_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        rows = rows
            .checked_add(self.outer_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        for group in self.precommitted_group_iter() {
            rows = rows
                .checked_add(1)
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            rows = rows
                .checked_add(group.layout.inner_commit_matrix.output_rank())
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            rows = rows
                .checked_add(group.layout.outer_commit_matrix.output_rank())
                .ok_or_else(Self::relation_matrix_row_overflow)?;
        }
        rows.checked_add(self.open_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)
    }

    /// Absolute start row of one group's A block in the multi-group root layout
    /// (`consistency_final | A_final | B_final |
    ///   [consistency_pre | A_pre | B_pre]* | D`).
    fn group_a_start(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<usize, AkitaError> {
        let final_group_index = self.validate_opening_batch(opening_batch)?;
        if group_index > final_group_index {
            return Err(AkitaError::InvalidProof);
        }
        if group_index == final_group_index {
            return Ok(self.a_start());
        }

        let mut start = self
            .a_start()
            .checked_add(self.inner_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        start = start
            .checked_add(self.outer_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        for prior_index in 0..group_index {
            let prior = self
                .precommitted_group_params(prior_index)
                .ok_or(AkitaError::InvalidProof)?;
            start = start
                .checked_add(1)
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            start = start
                .checked_add(prior.layout.inner_commit_matrix.output_rank())
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            start = start
                .checked_add(prior.layout.outer_commit_matrix.output_rank())
                .ok_or_else(Self::relation_matrix_row_overflow)?;
        }
        start
            .checked_add(1)
            .ok_or_else(Self::relation_matrix_row_overflow)
    }

    /// M-row index of one opening group's native consistency equation.
    pub fn consistency_row_index(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<usize, AkitaError> {
        self.group_a_start(opening_batch, group_index)?
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)
    }

    fn group_a_rows(
        &self,
        group_index: usize,
        final_group_index: usize,
    ) -> Result<usize, AkitaError> {
        if group_index == final_group_index {
            Ok(self.inner_commit_matrix.output_rank())
        } else {
            Ok(self
                .precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .layout
                .inner_commit_matrix
                .output_rank())
        }
    }

    fn group_b_rows(
        &self,
        group_index: usize,
        final_group_index: usize,
    ) -> Result<usize, AkitaError> {
        if group_index == final_group_index {
            Ok(self.outer_commit_matrix.output_rank())
        } else {
            Ok(self
                .precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .layout
                .outer_commit_matrix
                .output_rank())
        }
    }

    /// M-row range for one commitment group.
    pub fn commitment_row_range(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<std::ops::Range<usize>, AkitaError> {
        let final_group_index = self.validate_opening_batch(opening_batch)?;
        let a_start = self.group_a_start(opening_batch, group_index)?;
        let n_a = self.group_a_rows(group_index, final_group_index)?;
        let n_b = self.group_b_rows(group_index, final_group_index)?;
        let start = a_start
            .checked_add(n_a)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        let end = start
            .checked_add(n_b)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        Ok(start..end)
    }

    /// M-row range for one opening group's A block.
    pub fn a_row_range(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<std::ops::Range<usize>, AkitaError> {
        let final_group_index = self.validate_opening_batch(opening_batch)?;
        let start = self.group_a_start(opening_batch, group_index)?;
        let rows = self.group_a_rows(group_index, final_group_index)?;
        let end = start
            .checked_add(rows)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        Ok(start..end)
    }

    /// Exact live next-witness length in field elements for scalar or
    /// multi-group folds.
    pub fn output_witness_len<F: CanonicalField>(
        &self,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        opening_batch.check()?;
        self.witness_chunk.validate()?;
        self.validate_opening_batch(opening_batch)?;
        let witness_layout = crate::WitnessLayout::new(
            self,
            opening_batch,
            self.witness_chunk.num_chunks,
            crate::r_decomp_levels::<F>(self.log_basis_open),
        )?;
        Ok(witness_layout.live_coeff_len())
    }

    /// Row count for an explicit relation-matrix row layout.
    ///
    /// Scalar layout: `consistency (1) | A (n_a) | B (n_b · num_commitments)
    /// | optional D (n_d)`.
    ///
    /// Grouped-root layout: `[consistency_g | A_g | B_g]_g | optional D`,
    /// in canonical root group order. Public openings bind through the fused
    /// trace term, not M rows.
    ///
    /// Terminal folds use a separate direct-response protocol and therefore
    /// never construct this relation matrix.
    #[inline]
    pub fn relation_matrix_row_count(&self, num_commitments: usize) -> Result<usize, AkitaError> {
        if self.has_precommitted_groups() {
            return self.multi_group_relation_matrix_row_count_for(num_commitments);
        }
        self.require_scalar_level("relation_matrix_row_count_for")?;
        let after_a = self
            .a_start()
            .checked_add(self.inner_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        let commitment_rows = self
            .outer_commit_matrix
            .output_rank()
            .checked_mul(num_commitments)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        let after_commitment = after_a
            .checked_add(commitment_rows)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        after_commitment
            .checked_add(self.open_commit_matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)
    }

    /// Logical row index of the shared EvaluationTrace row (last padded row).
    ///
    /// Physical quotient rows occupy `0..relation_matrix_row_count`; EvaluationTrace
    /// sits at `relation_matrix_row_count` and is absent from the physical M matrix.
    pub fn evaluation_trace_row_index(
        &self,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        opening_batch.check()?;
        if self.has_precommitted_groups() {
            self.validate_opening_batch(opening_batch)?;
        } else {
            self.require_scalar_level(
                "CommittedGroupParams::evaluation_trace_row_index_for_layout",
            )?;
        }
        self.relation_matrix_row_count(opening_batch.num_groups())
    }

    /// Boolean variables needed to index the padded row space
    /// (`next_power_of_two(evaluation_trace_row + 1).trailing_zeros()`).
    pub fn relation_row_index_num_vars(
        &self,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        let total_rows = self
            .evaluation_trace_row_index(opening_batch)?
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidSetup("relation-row count overflow".to_string()))?;
        let padded = total_rows.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup("relation-row index width overflow".to_string())
        })?;
        Ok(padded.trailing_zeros() as usize)
    }

    /// Fill in layout-derived fields from exact digit-innermost geometry.
    ///
    /// Takes a params-only `CommittedGroupParams` (with zeroed layout fields) and
    /// `num_positions_per_block` is `M`, power-of-two in the current Boolean layout, and
    /// `num_live_ring_elements_per_claim` is the exact live `N`. The exact live block
    /// count `B` is derived as `ceil(N / M)`.
    ///
    /// # Errors
    ///
    /// Returns an error when parameters are invalid or derived widths overflow.
    pub fn with_decomp(
        &self,
        num_positions_per_block: usize,
        num_live_ring_elements_per_claim: usize,
        num_digits_inner: usize,
        num_digits_outer: usize,
        num_digits_open: usize,
    ) -> Result<Self, AkitaError> {
        if num_live_ring_elements_per_claim == 0
            || num_positions_per_block == 0
            || !num_positions_per_block.is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "with_decomp requires positive N and power-of-two M".to_string(),
            ));
        }
        let num_live_blocks = num_live_ring_elements_per_claim.div_ceil(num_positions_per_block);
        num_live_blocks.checked_next_power_of_two().ok_or_else(|| {
            AkitaError::InvalidSetup("block-index domain size overflows usize".to_string())
        })?;
        let inner_width = num_positions_per_block
            .checked_mul(num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = self
            .inner_commit_matrix
            .output_rank()
            .checked_mul(num_digits_outer)
            .and_then(|x| x.checked_mul(num_live_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("outer width overflow".to_string()))?;
        let d_matrix_width = num_digits_open
            .checked_mul(num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
        let rebuilt = Self {
            log_basis_inner: self.log_basis_inner,
            log_basis_outer: self.log_basis_outer,
            log_basis_open: self.log_basis_open,
            inner_commit_matrix: InnerCommitMatrixParams::new_unchecked(
                self.inner_commit_matrix.security_policy(),
                self.inner_commit_matrix.sis_table_key().table_digest,
                self.inner_commit_matrix.sis_modulus_profile(),
                self.inner_commit_matrix.output_rank,
                inner_width,
                self.inner_commit_matrix.coeff_linf_bound(),
                self.inner_commit_matrix.ring_dimension(),
            ),
            outer_commit_matrix: OuterCommitMatrixParams::new_unchecked(
                self.outer_commit_matrix.security_policy(),
                self.outer_commit_matrix.sis_table_key().table_digest,
                self.outer_commit_matrix.sis_modulus_profile(),
                self.outer_commit_matrix.output_rank,
                outer_width,
                self.outer_commit_matrix.coeff_linf_bound(),
                self.outer_commit_matrix.ring_dimension(),
            ),
            open_commit_matrix: OpenCommitMatrixParams::new_unchecked(
                self.open_commit_matrix.security_policy(),
                self.open_commit_matrix.sis_table_key().table_digest,
                self.open_commit_matrix.sis_modulus_profile(),
                self.open_commit_matrix.output_rank,
                d_matrix_width,
                self.open_commit_matrix.coeff_linf_bound(),
                self.open_commit_matrix.ring_dimension(),
            ),
            num_live_ring_elements_per_claim,
            num_positions_per_block,
            num_live_blocks,
            fold_challenge_config: self.fold_challenge_config,
            fold_challenge_shape: self.fold_challenge_shape,
            num_digits_inner,
            num_digits_outer,
            num_digits_open,
            num_digits_fold: self.num_digits_fold,
            // `with_decomp` recomputes only the A/B/D widths; the chunk layout is
            // a property of the witness this level commits, so preserve it.
            witness_chunk: self.witness_chunk,
            precommitted_groups: self.precommitted_groups.clone(),
            setup_prefix: self.setup_prefix.clone(),
        };
        rebuilt.validate_exact_fold_plan()
    }

    /// Build a new `CommittedGroupParams` that keeps rank/ring/SIS-bucket info
    /// from `self` but replaces all layout-derived fields with those
    /// from `other`.
    ///
    /// "Layout-derived fields" are the matrix input widths, `num_live_blocks`,
    /// `num_positions_per_block`,
    /// `position_index_bits`, `block_index_bits`, and the commit/open digit counts. The audited
    /// coefficient-L∞ SIS bucket is not a layout field: it is the bucket the
    /// output rank was sized against, so it is preserved from `self`,
    /// matching the placement of the output rank and `sis_modulus_profile`. Pulling the
    /// bucket from `other` would lose the audited value when the layout
    /// argument was constructed via [`CommittedGroupParams::params_only`] or threaded
    /// through [`Self::with_decomp`], and would let the SIS audit at
    /// role-specific commit-matrix constructors short-circuit silently.
    pub fn with_layout(&self, other: &CommittedGroupParams) -> Result<Self, AkitaError> {
        Self {
            log_basis_inner: other.log_basis_inner,
            log_basis_outer: other.log_basis_outer,
            log_basis_open: other.log_basis_open,
            inner_commit_matrix: InnerCommitMatrixParams::new_unchecked(
                self.inner_commit_matrix.security_policy(),
                self.inner_commit_matrix.sis_table_key().table_digest,
                self.inner_commit_matrix.sis_modulus_profile(),
                self.inner_commit_matrix.output_rank,
                other.inner_commit_matrix.input_width,
                self.inner_commit_matrix.coeff_linf_bound(),
                self.inner_commit_matrix.ring_dimension(),
            ),
            outer_commit_matrix: OuterCommitMatrixParams::new_unchecked(
                self.outer_commit_matrix.security_policy(),
                self.outer_commit_matrix.sis_table_key().table_digest,
                self.outer_commit_matrix.sis_modulus_profile(),
                self.outer_commit_matrix.output_rank,
                other.outer_commit_matrix.input_width,
                self.outer_commit_matrix.coeff_linf_bound(),
                self.outer_commit_matrix.ring_dimension(),
            ),
            open_commit_matrix: OpenCommitMatrixParams::new_unchecked(
                self.open_commit_matrix.security_policy(),
                self.open_commit_matrix.sis_table_key().table_digest,
                self.open_commit_matrix.sis_modulus_profile(),
                self.open_commit_matrix.output_rank,
                other.open_commit_matrix.input_width,
                self.open_commit_matrix.coeff_linf_bound(),
                self.open_commit_matrix.ring_dimension(),
            ),
            num_live_ring_elements_per_claim: other.num_live_ring_elements_per_claim,
            num_positions_per_block: other.num_positions_per_block,
            num_live_blocks: other.num_live_blocks,
            fold_challenge_config: self.fold_challenge_config,
            fold_challenge_shape: other.fold_challenge_shape,
            num_digits_inner: other.num_digits_inner,
            num_digits_outer: other.num_digits_outer,
            num_digits_open: other.num_digits_open,
            num_digits_fold: other.num_digits_fold,
            // The chunk layout is a property of the committed witness, sized with
            // the ranks, so it stays with `self` like the SIS buckets.
            witness_chunk: self.witness_chunk,
            precommitted_groups: self.precommitted_groups.clone(),
            setup_prefix: self.setup_prefix.clone(),
        }
        .validate_exact_fold_plan()
    }

    fn validate_exact_fold_plan(self) -> Result<Self, AkitaError> {
        if self.num_digits_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "exact fold plan must have nonzero digit depth".into(),
            ));
        }
        Ok(self)
    }
}

#[cfg(test)]
#[path = "params/tests/mod.rs"]
mod tests;
