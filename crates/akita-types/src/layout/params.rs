//! Unified per-level parameters for the Akita protocol.
//!
//! `CommittedGroupParams` merges ring dimension, matrix ranks, challenge config,
//! block geometry, and digit depths into a single struct that fully
//! describes one recursion level.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_field::CanonicalField;

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::layout::ring_dims::CommitmentRingDims;
use crate::opening_claims::OpeningClaimsLayout;
use crate::proof::{
    CompressionRelationAddressGeometry, RelationAddressGeometry, RelationRowFamily,
};

pub use crate::sis::{
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams, SisModulusProfileId,
};

fn compression_relation_row_count(
    num_commitments: usize,
    base_rows: usize,
) -> Result<usize, AkitaError> {
    let compression_rows = num_commitments
        .checked_add(1)
        .and_then(|chains| chains.checked_mul(crate::COMPRESSION_MAP_COUNT))
        .ok_or_else(CommittedGroupParams::relation_matrix_row_overflow)?;
    base_rows
        .checked_add(compression_rows)
        .ok_or_else(CommittedGroupParams::relation_matrix_row_overflow)
}

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
        .checked_add(crate::BlockGeometry::position_index_bits_for(
            num_positions_per_block,
        ))
        .and_then(|bits| {
            crate::BlockGeometry::checked_block_index_bits_for(num_live_blocks)
                .and_then(|blocks| bits.checked_add(blocks))
        })
        .ok_or_else(|| AkitaError::InvalidSetup("recursive opening num_vars overflow".to_string()))
}

mod descriptor;
mod groups;
mod own_group;
mod precommitted;
pub(crate) use descriptor::append_sparse_challenge_descriptor_bytes as append_schedule_sparse_challenge_descriptor_bytes;
use groups::FoldGroups;
pub use precommitted::{
    opening_d_segment_width, GroupOpenPhaseParams, GroupOpeningPlan, OpeningFamily, OpeningMethod,
    PrecommittedGroupAdmissionPolicy,
};

/// Gadget basis used by opening-digit segments in the shared D product.
///
/// A grouped root concatenates the main group's `e_hat` with every
/// precommitted group's fresh `e_hat`; all fresh opening digits use the root
/// opening basis.
#[must_use]
pub fn shared_d_digit_log_basis(
    main_log_basis: u32,
    _precommitted_groups: &[GroupOpenPhaseParams],
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
    /// Every group this fold consumes or commits, in canonical order: an
    /// incoming setup prefix, then the frozen precommitted groups, then this
    /// fold's own new group last.
    ///
    /// The single place a fold's groups live. Its own group used to be a dozen
    /// flat fields alongside this list, which is why `final_group` had to
    /// materialise one and why three callers each invented a layout for it.
    groups: FoldGroups,
    /// Shared D matrix over every group's `w_hat` segment.
    ///
    /// The matrix is fold-owned, with one owner rather than one copy per group.
    /// The opening *digits* are not here: each group carries its own
    /// in its `GroupOpeningPlan`, so putting them on the fold too would recreate
    /// exactly the mirror this spec exists to delete.
    pub open_matrix: OpenCommitMatrixParams,
    /// Public B/D payload encoding selected for this fold level.
    pub payload_mode: crate::CommitmentPayloadMode,
    /// Physical source encoding authenticated by A and B.
    pub source_encoding: crate::CommittedSourceEncoding,
    /// Multi-chunk witness layout this level commits under.
    pub witness_chunk: crate::witness::ChunkedWitnessCfg,
}

impl CommittedGroupParams {
    /// Build fold parameters with checked, canonically ordered group storage.
    pub fn try_new(
        groups: Vec<GroupOpenPhaseParams>,
        open_matrix: OpenCommitMatrixParams,
        payload_mode: crate::CommitmentPayloadMode,
        source_encoding: crate::CommittedSourceEncoding,
        witness_chunk: crate::witness::ChunkedWitnessCfg,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            groups: FoldGroups::try_from_vec(groups)?,
            open_matrix,
            payload_mode,
            source_encoding,
            witness_chunk,
        })
    }

    /// Validate the canonical group topology before reading group geometry.
    pub fn validate_group_topology(&self) -> Result<(), AkitaError> {
        self.groups.validate_topology()
    }

    /// Canonical byte encoding used to order semantically distinct level candidates.
    ///
    /// This is an ordering descriptor, not a wire encoding or transcript commitment.
    #[must_use]
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    /// Checked wire geometry for this level's final-group B image.
    pub fn outer_payload_geometry(&self) -> Result<crate::CommitmentPayloadGeometry, AkitaError> {
        let logical_rows = self
            .outer_slice_count()
            .logical_output_rows(self.outer().matrix.output_rank())?;
        crate::CommitmentPayloadGeometry::for_mode(
            self.payload_mode,
            self.outer().matrix.sis_modulus_profile(),
            logical_rows,
            self.role_dims().d_b(),
        )
    }

    /// Checked wire geometry for this level's shared D image.
    pub fn opening_payload_geometry(&self) -> Result<crate::CommitmentPayloadGeometry, AkitaError> {
        crate::CommitmentPayloadGeometry::for_mode(
            self.payload_mode,
            self.open().matrix.sis_modulus_profile(),
            self.open().matrix.output_rank(),
            self.role_dims().d_d(),
        )
    }

    /// Whether every B/D image at this level fits the compression policy cap.
    pub fn compression_sources_supported(&self) -> Result<bool, AkitaError> {
        if !self.payload_mode.is_compressed() {
            return Ok(true);
        }
        let final_outer = self.outer_slice_count().complete_source_coefficients(
            self.outer().matrix.output_rank(),
            self.role_dims().d_b(),
        )?;
        if crate::CompressionChainPlan::try_for_complete_source(
            self.outer().matrix.sis_modulus_profile(),
            final_outer,
        )?
        .is_none()
        {
            return Ok(false);
        }
        for group in self.precommitted_group_iter() {
            let source = group
                .profile
                .outer_slice_count
                .complete_source_coefficients(
                    group.profile.outer.matrix.output_rank(),
                    group.profile.outer.matrix.ring_dimension(),
                )?;
            if crate::CompressionChainPlan::try_for_complete_source(
                group.profile.outer.matrix.sis_modulus_profile(),
                source,
            )?
            .is_none()
            {
                return Ok(false);
            }
        }
        let opening = self
            .open()
            .matrix
            .output_rank()
            .checked_mul(self.role_dims().d_d())
            .ok_or_else(|| AkitaError::InvalidSetup("D compression shape overflow".into()))?;
        Ok(crate::CompressionChainPlan::try_for_complete_source(
            self.open().matrix.sis_modulus_profile(),
            opening,
        )?
        .is_some())
    }

    /// Largest gadget basis accepted by this level's shared D product.
    #[must_use]
    pub fn shared_d_digit_log_basis(&self) -> u32 {
        shared_d_digit_log_basis(self.open().digits.log_basis, self.groups.as_slice())
    }

    /// Per-role ring dimensions derived from the three matrix objects.
    #[must_use]
    pub fn role_dims(&self) -> CommitmentRingDims {
        CommitmentRingDims {
            inner: self.inner().matrix.ring_dimension(),
            outer: self.outer().matrix.ring_dimension(),
            opening: self.open().matrix.ring_dimension(),
        }
    }

    /// A-role ring dimension (`d_a`); alias of [`CommitmentRingDims::d_a`] on [`Self::role_dims`].
    #[inline]
    #[must_use]
    pub fn d_a(&self) -> usize {
        self.inner().matrix.ring_dimension()
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
            open_matrix: OpenCommitMatrixParams::new_unchecked(
                crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                crate::sis::SisTableDigest::CURRENT,
                sis_modulus_profile,
                n_d,
                0,
                0,
                ring_dimension,
            ),
            payload_mode: crate::CommitmentPayloadMode::Compressed,
            source_encoding: crate::CommittedSourceEncoding::CanonicalCoefficientTable,
            witness_chunk: crate::witness::ChunkedWitnessCfg::default_non_chunked(),
            // A zeroed shell whose only group is its own; callers fill the
            // geometry through `with_decomp`, as they always did.
            groups: FoldGroups::singleton(GroupOpenPhaseParams {
                profile: crate::GroupCommitPhaseParams {
                    version: crate::GroupCommitPhaseParams::VERSION,
                    group: crate::PolynomialGroupLayout::singleton(0),
                    blocks: crate::BlockGeometry::new(0, 0, 0),
                    outer_slice_count: crate::CommitmentSliceCount::ONE,
                    inner: crate::RoleParams::new(
                        crate::GadgetDigits::new(log_basis, 0),
                        InnerCommitMatrixParams::new_unchecked(
                            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                            crate::sis::SisTableDigest::CURRENT,
                            sis_modulus_profile,
                            n_a,
                            0,
                            0,
                            ring_dimension,
                        ),
                    ),
                    outer: crate::RoleParams::new(
                        crate::GadgetDigits::new(log_basis, 0),
                        OuterCommitMatrixParams::new_unchecked(
                            crate::sis::DEFAULT_SIS_SECURITY_POLICY,
                            crate::sis::SisTableDigest::CURRENT,
                            sis_modulus_profile,
                            n_b,
                            0,
                            0,
                            ring_dimension,
                        ),
                    ),
                },
                opening: GroupOpeningPlan {
                    opening_method: OpeningMethod::EvaluationTrace,
                    fold_challenge_config,
                    log_basis_open: log_basis,
                    num_digits_open: 0,
                    num_digits_fold: 1,
                },
                setup_natural_len: None,
            }),
        }
    }

    /// True when this level carries multi-group-root metadata.
    #[inline]
    pub fn has_precommitted_groups(&self) -> bool {
        self.precommitted_group_count() != 0
    }

    #[inline]
    pub fn precommitted_group_count(&self) -> usize {
        self.groups.preceding().len()
    }

    #[inline]
    pub fn precommitted_group_params(&self, group_index: usize) -> Option<&GroupOpenPhaseParams> {
        // The own group is the last entry and is not a precommitted group, so
        // bound the index before indexing the list.
        if group_index >= self.precommitted_group_count() {
            return None;
        }
        self.groups.preceding_group(group_index)
    }

    pub fn setup_prefix(&self) -> Option<&GroupOpenPhaseParams> {
        self.groups.setup_prefix()
    }

    /// The frozen precommitted groups, without any incoming prefix.
    ///
    /// A slice, not an iterator: the prefix is always at index zero, so the
    /// precommitted groups are a contiguous tail and every caller that indexed
    /// or measured the old `Vec` keeps working.
    #[must_use]
    pub fn precommitted_groups(&self) -> &[GroupOpenPhaseParams] {
        self.groups.precommitted()
    }

    #[cfg(test)]
    pub(crate) fn preceding_group_mut_for_test(
        &mut self,
        group_index: usize,
    ) -> Option<&mut GroupOpenPhaseParams> {
        self.groups.preceding_group_mut(group_index)
    }

    /// Add one precommitted group, keeping the fold's own group last.
    pub fn insert_precommitted_group(
        &mut self,
        group: GroupOpenPhaseParams,
    ) -> Result<(), AkitaError> {
        self.groups.insert_precommitted(group)
    }

    /// Replace this fold's precommitted groups, keeping any incoming prefix.
    pub fn set_precommitted_groups(
        &mut self,
        groups: Vec<GroupOpenPhaseParams>,
    ) -> Result<(), AkitaError> {
        self.groups.replace_precommitted(groups)
    }

    /// Replace this fold's incoming setup prefix.
    ///
    /// Keeps the prefix at index zero so canonical order survives the edit.
    pub fn set_setup_prefix(
        &mut self,
        prefix: Option<GroupOpenPhaseParams>,
    ) -> Result<(), AkitaError> {
        self.groups.replace_setup_prefix(prefix)
    }

    pub fn precommitted_group_iter(&self) -> impl Iterator<Item = &GroupOpenPhaseParams> {
        // Excludes the own group at the end, but *includes* an incoming prefix:
        // this iterator has always yielded `[prefix?, precommitted...]`, and the
        // opening batch is built from it, so dropping the prefix here shrinks the
        // batch by one and makes it disagree with `group_count()`.
        self.groups.preceding().iter()
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
        self.fold_challenge_config().l1_norm()
    }

    /// Effective fold-round challenge L∞ norm `||c||_inf` at this level.
    #[inline]
    pub fn challenge_infinity_norm(&self) -> usize {
        self.fold_challenge_config().infinity_norm() as usize
    }

    /// Effective per-block worst-case `‖c‖_2²` upper bound at this fold level.
    #[inline]
    pub fn challenge_l2_sq_max(&self) -> u128 {
        self.fold_challenge_config().challenge_l2_sq_max()
    }

    /// Fold-challenge coefficient count `inner_width · D`.
    #[inline]
    pub fn num_fold_coeffs(&self) -> u128 {
        (self.inner_width() as u128).saturating_mul(self.d_a() as u128)
    }

    /// Validate the shared fold nonce against the protocol-wide attempt cap.
    ///
    /// This verifier boundary deliberately does not reconstruct an honest
    /// source model or an honest folded-response cap. Those values guide the
    /// prover's search only.
    pub fn validate_fold_grind_nonce(
        &self,
        opening_batch: &OpeningClaimsLayout,
        fold_grind_nonce: u32,
    ) -> Result<(), AkitaError> {
        self.validate_opening_batch(opening_batch)?;
        crate::FoldLinfProtocolBinding::CURRENT.validate_grind_nonce(fold_grind_nonce)
    }

    /// Exact scheduled gadget decomposition depth for the folded witness.
    #[inline]
    pub fn num_digits_fold(&self) -> usize {
        self.own_group().opening.num_digits_fold
    }

    /// This fold's block triple.
    ///
    /// Block triple of this fold's own new group.
    #[inline]
    #[must_use]
    pub fn blocks(&self) -> crate::BlockGeometry {
        self.own_group().profile.blocks
    }

    /// Number of Boolean coordinates in the block-index domain.
    #[inline]
    pub fn block_index_bits(&self) -> usize {
        self.blocks().block_index_bits()
    }

    /// Number of Boolean coordinates in one block-position slice.
    #[inline]
    pub fn position_index_bits(&self) -> usize {
        self.blocks().position_index_bits()
    }

    /// Boolean block-index domain size (`next_power_of_two(B)`).
    #[inline]
    pub fn block_index_domain_size(&self) -> Result<usize, AkitaError> {
        self.blocks().block_index_domain_size()
    }

    /// Validate the exact source/block geometry before it reaches allocation.
    pub fn validate_block_geometry(&self) -> Result<(), AkitaError> {
        self.blocks().validate()
    }

    /// Validate the exact A/B geometry executed by one commitment request.
    ///
    /// This binds the concrete polynomial arity and fold level to the same B
    /// width, slice policy, and complete-source compression cap used for SIS
    /// pricing and descriptor construction.
    pub fn validate_commitment_request(
        &self,
        fold_level: usize,
        num_polynomials: usize,
    ) -> Result<crate::CommitmentSliceGeometry, AkitaError> {
        self.validate_group_topology()?;
        let own_profile = &self.own_group().profile;
        own_profile.validate(own_profile.inner.matrix.sis_modulus_profile().field_bits())?;
        own_profile.validate_root_geometry()?;
        if num_polynomials == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment request requires at least one polynomial".into(),
            ));
        }
        if self.own_group().profile.group.num_polynomials() != num_polynomials {
            return Err(AkitaError::InvalidSetup(
                "stored own-group arity disagrees with the commitment request".into(),
            ));
        }
        self.source_encoding.validate(self.d_a())?;
        self.validate_block_geometry()?;
        self.outer_slice_count().validate_for_commitment(
            fold_level,
            self.payload_mode,
            self.blocks().live_blocks,
        )?;
        let expected_a_width = self
            .blocks()
            .positions_per_block
            .checked_mul(self.inner().digits.num_digits)
            .ok_or_else(|| AkitaError::InvalidSetup("commitment A width overflow".into()))?;
        if self.inner().matrix.input_width() != expected_a_width {
            return Err(AkitaError::InvalidSetup(
                "commitment A matrix width disagrees with request geometry".into(),
            ));
        }
        let geometry = crate::CommitmentSliceGeometry::try_new(
            self.outer_slice_count(),
            self.blocks().live_blocks,
            num_polynomials,
            self.inner().matrix.output_rank(),
            self.outer().digits.num_digits,
            self.role_dims().d_a(),
            self.role_dims().d_b(),
        )?;
        if self.outer().matrix.input_width() != geometry.physical_input_width() {
            return Err(AkitaError::InvalidSetup(
                "commitment B matrix width disagrees with sliced request geometry".into(),
            ));
        }
        if self.payload_mode.is_compressed() {
            let source_coefficients = geometry
                .logical_output_rows(self.outer().matrix.output_rank())?
                .checked_mul(self.role_dims().d_b())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("commitment B source size overflow".into())
                })?;
            if crate::CompressionChainPlan::try_for_complete_source(
                self.outer().matrix.sis_modulus_profile(),
                source_coefficients,
            )?
            .is_none()
            {
                return Err(AkitaError::InvalidSetup(
                    "commitment B source exceeds the compression cap".into(),
                ));
            }
        }
        Ok(geometry)
    }

    /// Polynomial arity encoded by the exact physical B width.
    pub fn commitment_polynomial_count(&self) -> Result<usize, AkitaError> {
        let one_polynomial_width = crate::CommitmentSliceGeometry::try_new(
            self.outer_slice_count(),
            self.blocks().live_blocks,
            1,
            self.inner().matrix.output_rank(),
            self.outer().digits.num_digits,
            self.role_dims().d_a(),
            self.role_dims().d_b(),
        )?
        .physical_input_width();
        self.outer()
            .matrix
            .input_width()
            .checked_div(one_polynomial_width)
            .filter(|count| {
                *count != 0 && self.outer().matrix.input_width() == *count * one_polynomial_width
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "commitment B width does not encode an exact polynomial count".into(),
                )
            })
    }

    /// Width of inner matrix A (column count of the A-key).
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.inner().matrix.input_width()
    }

    /// Exact live source ring elements in one claim.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] on overflow.
    pub fn n_ring_elems(&self) -> Result<usize, AkitaError> {
        self.validate_block_geometry()?;
        Ok(self.blocks().live_ring_elements_per_claim)
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
        self.append_descriptor_bytes_with_payload_mode(bytes, self.payload_mode);
    }

    pub(crate) fn append_descriptor_bytes_with_payload_mode(
        &self,
        bytes: &mut Vec<u8>,
        payload_mode: crate::CommitmentPayloadMode,
    ) {
        bytes.push(payload_mode.tag());
        self.source_encoding.append_descriptor_bytes(bytes);
        self.opening_method().append_descriptor_bytes(bytes);
        push_u32(bytes, self.inner().digits.log_basis);
        push_u32(bytes, self.outer().digits.log_basis);
        push_u32(bytes, self.open().digits.log_basis);
        self.inner().matrix.append_descriptor_bytes(bytes);
        self.outer().matrix.append_descriptor_bytes(bytes);
        self.open().matrix.append_descriptor_bytes(bytes);
        push_usize(bytes, self.blocks().live_ring_elements_per_claim);
        push_usize(bytes, self.blocks().positions_per_block);
        push_usize(bytes, self.blocks().live_blocks);
        self.outer_slice_count().append_descriptor_bytes(bytes);
        append_schedule_sparse_challenge_descriptor_bytes(bytes, &self.fold_challenge_config());
        push_usize(bytes, self.inner().digits.num_digits);
        push_usize(bytes, self.outer().digits.num_digits);
        push_usize(bytes, self.open().digits.num_digits);
        push_usize(bytes, self.num_digits_fold());
        // Chunk binding is appended only when the level is chunked, so
        // single-chunk descriptors stay byte-for-byte identical to the historical
        // layout (the flag-off no-op invariant). When chunked, bind the chunk
        // count and activated-level count into the Fiat-Shamir digest.
        if self.witness_chunk.num_chunks != 1 {
            self.witness_chunk.append_descriptor_bytes(bytes);
        }

        if !self.precommitted_groups().is_empty() {
            push_usize(bytes, self.precommitted_groups().len());
            for group in self.precommitted_groups() {
                group.append_descriptor_bytes(bytes);
            }
        }
        if let Some(setup_prefix) = self.setup_prefix() {
            bytes.push(1);
            setup_prefix.append_setup_prefix_descriptor_bytes(bytes);
        } else {
            bytes.push(0);
        }
    }

    /// Width of outer matrix B (column count of the B-key).
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.outer().matrix.input_width()
    }

    /// Width of prover matrix D (column count of the D-key).
    #[inline]
    pub fn d_matrix_width(&self) -> usize {
        self.open().matrix.input_width()
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
            self.blocks().positions_per_block,
            self.blocks().live_blocks,
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
            .checked_add(self.inner().matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)
    }

    /// Absolute start row of the D block.
    #[inline]
    pub fn d_start(&self, num_commitments: usize) -> Result<usize, AkitaError> {
        let b_rows = self
            .outer()
            .matrix
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
            .map(|group| group.profile.group)
            .collect::<Vec<_>>();
        OpeningClaimsLayout::from_root_groups(&precommitted, final_group)
    }

    pub(crate) fn validate_opening_batch_geometry(
        &self,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        self.validate_group_topology()?;
        opening_batch.check()?;
        if self.open().digits.log_basis < self.outer().digits.log_basis {
            return Err(AkitaError::InvalidSetup(
                "certified opening basis must dominate the level outer basis".to_string(),
            ));
        }
        if opening_batch.num_groups() != self.group_count() {
            return Err(AkitaError::InvalidSetup(
                "opening group count does not match level params".to_string(),
            ));
        }
        // The catalog-validation path may carry a placeholder final-group
        // layout. The stored own profile remains authoritative and is validated
        // above; exact layout equality is enforced wherever the batch has a
        // concrete final layout.
        for group_index in 0..self.precommitted_group_count() {
            let group_params = self
                .precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            group_params.validate()?;
            if group_params.opening.log_basis_open != self.open().digits.log_basis {
                return Err(AkitaError::InvalidSetup(
                    "all opening groups must use the batch-shared opening basis".to_string(),
                ));
            }
            let group_layout = opening_batch.group_layout(group_index)?;
            if *group_layout != group_params.profile.group {
                return Err(AkitaError::InvalidSetup(
                    "precommitted group layout does not match level params".to_string(),
                ));
            }
        }
        opening_batch.root_final_group_index()
    }

    pub fn validate_opening_batch(
        &self,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        self.validate_opening_batch_geometry(opening_batch)
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
                .role_dims(self.open().matrix.ring_dimension())
        };
        dims.validate_role_projection()?;
        Ok(dims)
    }

    /// Resolve one opening group's structural A/B/D dimensions without
    /// requiring that its opening method is executable by the caller.
    ///
    /// Consumers must still apply their own method admission before executing
    /// method-specific algebra.
    pub fn group_role_dims_geometry(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<CommitmentRingDims, AkitaError> {
        let final_group_index = self.validate_opening_batch_geometry(opening_batch)?;
        let dims = if group_index == final_group_index {
            self.role_dims()
        } else {
            self.precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .role_dims(self.open().matrix.ring_dimension())
        };
        dims.validate_role_projection()?;
        Ok(dims)
    }

    /// Resolve flat relation-address geometry across every opening group's
    /// native A/B dimensions and this level's shared D dimension.
    pub fn relation_address_geometry(
        &self,
        opening_batch: &OpeningClaimsLayout,
        extension_degree: usize,
        outgoing_witness_ring_dimension: usize,
        live_witness_coeff_len: usize,
    ) -> Result<RelationAddressGeometry, AkitaError> {
        self.validate_opening_batch(opening_batch)?;
        let relation_geometry =
            crate::RelationWitnessGeometry::for_level(self, opening_batch, extension_degree)?;
        RelationAddressGeometry::for_relation(
            &relation_geometry,
            outgoing_witness_ring_dimension,
            live_witness_coeff_len,
        )
    }

    /// Resolve the independent compact address geometry for F/H rows.
    pub fn compression_relation_address_geometry(
        &self,
        opening_batch: &OpeningClaimsLayout,
        extension_degree: usize,
        outgoing_witness_ring_dimension: usize,
        live_witness_coeff_len: usize,
    ) -> Result<CompressionRelationAddressGeometry, AkitaError> {
        let relation_geometry =
            crate::RelationWitnessGeometry::for_level(self, opening_batch, extension_degree)?;
        let compression_row_dims = relation_geometry
            .rhs_layout()
            .row_families()?
            .into_iter()
            .filter_map(|row| {
                matches!(
                    row,
                    RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
                )
                .then_some(row.geometry().polynomial_modulus_dimension())
            })
            .collect::<Vec<_>>();
        CompressionRelationAddressGeometry::new(
            &compression_row_dims,
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
            return self
                .outer_slice_count()
                .logical_output_rows(self.outer().matrix.output_rank());
        }
        let group = self
            .precommitted_group_params(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        group
            .profile
            .outer_slice_count
            .logical_output_rows(group.profile.outer.matrix.output_rank())
    }

    /// This fold's own new group.
    ///
    /// Takes no layout because the fold stores its own authoritative layout.
    ///
    /// Cheap: `GroupOpenPhaseParams` is `Copy` and all of its fields already
    /// were.
    pub fn final_group(&self) -> crate::GroupOpenPhaseParams {
        *self.own_group()
    }

    /// This fold's final group, for a scalar (single-polynomial) fold.
    ///
    /// Derives the layout from geometry the fold already carries: a scalar fold
    /// has one polynomial, and `N * d_a == 2^num_vars` is the invariant
    /// `validate_root_geometry` enforces. A grouped fold must supply its layout
    /// explicitly through [`Self::final_group`], because `num_polynomials` is
    /// not derivable from the fold alone.
    pub fn final_group_scalar(&self) -> Result<crate::GroupOpenPhaseParams, AkitaError> {
        let source_len = self
            .blocks()
            .live_ring_elements_per_claim
            .checked_mul(self.d_a())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("scalar final-group source length overflow".to_string())
            })?;
        if !source_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "scalar final-group source length is not a power of two".to_string(),
            ));
        }
        Ok(self.final_group())
    }

    /// Physical source encoding of the group at `group_index`.
    ///
    /// The fold's own new witness carries this fold's encoding; every earlier
    /// group is canonical by admission, because
    /// `GroupCommitPhaseParams::try_from_params` refuses to freeze a
    /// non-canonical standalone profile. This is the single owner that replaces
    /// the group accessor's hard-coded constant.
    #[must_use]
    pub fn source_encoding_of(&self, group_index: usize) -> crate::CommittedSourceEncoding {
        if group_index == self.precommitted_group_count() {
            self.source_encoding
        } else {
            crate::CommittedSourceEncoding::CanonicalCoefficientTable
        }
    }

    /// Every group this fold opens, in canonical transcript order.
    ///
    /// An incoming setup prefix is group 0, earlier precommitted groups follow,
    /// and the fold's own final/new group is last. This is the one ordering the
    /// schedule commits to; see `validate_nonterminal_opening_execution`.
    pub fn groups(&self) -> &[crate::GroupOpenPhaseParams] {
        self.groups.as_slice()
    }

    /// One group of this fold's opening batch, as a concrete group.
    ///
    /// Formerly returned `&dyn LevelParamsLike`, because the final group was the
    /// fold itself while the others were `GroupOpenPhaseParams` and callers had
    /// to be prevented from caring which. Both are now the same type, so the
    /// erasure is gone: the return is a value, `GroupOpenPhaseParams` being
    /// `Copy`.
    pub fn group_params(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<crate::GroupOpenPhaseParams, AkitaError> {
        self.validate_opening_batch(opening_batch)?;
        self.groups
            .as_slice()
            .get(group_index)
            .copied()
            .ok_or(AkitaError::InvalidProof)
    }

    /// Resolve one group's structurally validated parameters without admitting
    /// its opening method for execution.
    ///
    /// Construction code uses this boundary while a new opening method is
    /// being prepared. Execution paths must use [`Self::group_params`], which
    /// additionally enforces the currently supported method set.
    /// As [`Self::group_params`], but validating geometry only.
    pub fn group_params_geometry(
        &self,
        opening_batch: &OpeningClaimsLayout,
        group_index: usize,
    ) -> Result<crate::GroupOpenPhaseParams, AkitaError> {
        self.validate_opening_batch_geometry(opening_batch)?;
        self.groups
            .as_slice()
            .get(group_index)
            .copied()
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
            .checked_add(self.inner().matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        let final_b_rows = self
            .outer_slice_count()
            .logical_output_rows(self.outer().matrix.output_rank())?;
        rows = rows
            .checked_add(final_b_rows)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        for group in self.precommitted_group_iter() {
            rows = rows
                .checked_add(1)
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            rows = rows
                .checked_add(group.profile.inner.matrix.output_rank())
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            let group_b_rows = group
                .profile
                .outer_slice_count
                .logical_output_rows(group.profile.outer.matrix.output_rank())?;
            rows = rows
                .checked_add(group_b_rows)
                .ok_or_else(Self::relation_matrix_row_overflow)?;
        }
        let base = rows
            .checked_add(self.open().matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        if self.payload_mode.is_compressed() {
            compression_relation_row_count(num_commitments, base)
        } else {
            Ok(base)
        }
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
            .checked_add(self.inner().matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        start = start
            .checked_add(
                self.outer_slice_count()
                    .logical_output_rows(self.outer().matrix.output_rank())?,
            )
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        for prior_index in 0..group_index {
            let prior = self
                .precommitted_group_params(prior_index)
                .ok_or(AkitaError::InvalidProof)?;
            start = start
                .checked_add(1)
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            start = start
                .checked_add(prior.profile.inner.matrix.output_rank())
                .ok_or_else(Self::relation_matrix_row_overflow)?;
            start = start
                .checked_add(
                    prior
                        .profile
                        .outer_slice_count
                        .logical_output_rows(prior.profile.outer.matrix.output_rank())?,
                )
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
            Ok(self.inner().matrix.output_rank())
        } else {
            Ok(self
                .precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .profile
                .inner
                .matrix
                .output_rank())
        }
    }

    fn group_b_rows(
        &self,
        group_index: usize,
        final_group_index: usize,
    ) -> Result<usize, AkitaError> {
        if group_index == final_group_index {
            self.outer_slice_count()
                .logical_output_rows(self.outer().matrix.output_rank())
        } else {
            let group = self
                .precommitted_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            group
                .profile
                .outer_slice_count
                .logical_output_rows(group.profile.outer.matrix.output_rank())
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
        extension_degree: usize,
    ) -> Result<usize, AkitaError> {
        self.output_witness_len_for_field_bits(F::modulus_bits(), extension_degree, opening_batch)
    }

    /// Exact live next-witness length using an explicit base-field bit width.
    ///
    /// Generated schedule replay uses the catalog-bound field width without
    /// monomorphizing on a concrete field type.
    pub fn output_witness_len_for_field_bits(
        &self,
        field_bits: u32,
        extension_degree: usize,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<usize, AkitaError> {
        opening_batch.check()?;
        self.witness_chunk.validate()?;
        let relation_geometry =
            crate::RelationWitnessGeometry::for_level(self, opening_batch, extension_degree)?;
        let witness_layout = crate::WitnessLayout::new(
            self,
            opening_batch,
            &relation_geometry,
            self.witness_chunk.num_chunks,
            crate::sis::compute_num_digits_field_width(field_bits, self.open().digits.log_basis),
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
            .checked_add(self.inner().matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        let logical_b_rows = self
            .outer_slice_count()
            .logical_output_rows(self.outer().matrix.output_rank())?;
        let commitment_rows = logical_b_rows
            .checked_mul(num_commitments)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        let after_commitment = after_a
            .checked_add(commitment_rows)
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        let base = after_commitment
            .checked_add(self.open().matrix.output_rank())
            .ok_or_else(Self::relation_matrix_row_overflow)?;
        if self.payload_mode.is_compressed() {
            compression_relation_row_count(num_commitments, base)
        } else {
            Ok(base)
        }
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
        crate::BlockGeometry::checked_block_index_domain_size_for(num_live_blocks).ok_or_else(
            || AkitaError::InvalidSetup("block-index domain size overflows usize".to_string()),
        )?;
        let inner_width = num_positions_per_block
            .checked_mul(num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".to_string()))?;
        let outer_width = crate::CommitmentSliceGeometry::try_new(
            self.outer_slice_count(),
            num_live_blocks,
            1,
            self.inner().matrix.output_rank(),
            num_digits_outer,
            self.inner().matrix.ring_dimension(),
            self.outer().matrix.ring_dimension(),
        )?
        .physical_input_width();
        let d_matrix_width = num_digits_open
            .checked_mul(num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("D-matrix width overflow".to_string()))?;
        let rebuilt = Self {
            payload_mode: self.payload_mode,
            source_encoding: self.source_encoding,
            witness_chunk: self.witness_chunk,
            open_matrix: OpenCommitMatrixParams::new_unchecked(
                self.open_matrix.security_policy(),
                self.open_matrix.sis_table_key().table_digest,
                self.open_matrix.sis_modulus_profile(),
                self.open_matrix.output_rank,
                d_matrix_width,
                self.open_matrix.coeff_linf_bound(),
                self.open_matrix.ring_dimension(),
            ),
            // Rebuild only this fold's own group; the earlier entries are frozen.
            groups: {
                let mut groups = self.groups.clone();
                let own = groups.own_mut();
                own.profile.blocks = crate::BlockGeometry::new(
                    num_live_ring_elements_per_claim,
                    num_positions_per_block,
                    num_live_blocks,
                );
                own.profile.inner = crate::RoleParams::new(
                    crate::GadgetDigits::new(own.profile.inner.digits.log_basis, num_digits_inner),
                    own.profile.inner.matrix.try_with_input_width(inner_width)?,
                );
                own.profile.outer = crate::RoleParams::new(
                    crate::GadgetDigits::new(own.profile.outer.digits.log_basis, num_digits_outer),
                    OuterCommitMatrixParams::new_unchecked(
                        own.profile.outer.matrix.security_policy(),
                        own.profile.outer.matrix.sis_table_key().table_digest,
                        own.profile.outer.matrix.sis_modulus_profile(),
                        own.profile.outer.matrix.output_rank,
                        outer_width,
                        own.profile.outer.matrix.coeff_linf_bound(),
                        own.profile.outer.matrix.ring_dimension(),
                    ),
                );
                own.opening.num_digits_open = num_digits_open;
                groups
            },
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
            payload_mode: other.payload_mode,
            source_encoding: other.source_encoding,
            // The chunk layout is a property of the committed witness, sized with
            // the ranks, so it stays with `self` like the SIS buckets.
            witness_chunk: self.witness_chunk,
            open_matrix: OpenCommitMatrixParams::new_unchecked(
                self.open_matrix.security_policy(),
                self.open_matrix.sis_table_key().table_digest,
                self.open_matrix.sis_modulus_profile(),
                self.open_matrix.output_rank,
                other.open_matrix.input_width,
                self.open_matrix.coeff_linf_bound(),
                self.open_matrix.ring_dimension(),
            ),
            // Layout-derived parts of this fold's own group come from `other`;
            // ranks and audited buckets stay with `self`. Earlier entries are
            // frozen groups and are carried across untouched.
            groups: {
                let mut groups = self.groups.clone();
                let own = groups.own_mut();
                own.profile.group = other.group();
                own.profile.blocks = other.blocks();
                own.profile.outer_slice_count = other.outer_slice_count();
                own.profile.inner = crate::RoleParams::new(
                    crate::GadgetDigits::new(
                        other.inner().digits.log_basis,
                        other.inner().digits.num_digits,
                    ),
                    self.inner()
                        .matrix
                        .try_with_input_width(other.inner().matrix.input_width)?,
                );
                own.profile.outer = crate::RoleParams::new(
                    crate::GadgetDigits::new(
                        other.outer().digits.log_basis,
                        other.outer().digits.num_digits,
                    ),
                    OuterCommitMatrixParams::new_unchecked(
                        self.outer().matrix.security_policy(),
                        self.outer().matrix.sis_table_key().table_digest,
                        self.outer().matrix.sis_modulus_profile(),
                        self.outer().matrix.output_rank,
                        other.outer().matrix.input_width,
                        self.outer().matrix.coeff_linf_bound(),
                        self.outer().matrix.ring_dimension(),
                    ),
                );
                own.opening.opening_method = other.opening_method();
                own.opening.log_basis_open = other.own_group().opening.log_basis_open;
                own.opening.num_digits_open = other.own_group().opening.num_digits_open;
                own.opening.num_digits_fold = other.num_digits_fold();
                own.opening.fold_challenge_config = self.fold_challenge_config();
                groups
            },
        }
        .validate_exact_fold_plan()
    }

    fn validate_exact_fold_plan(self) -> Result<Self, AkitaError> {
        self.validate_group_topology()?;
        if self.num_digits_fold() == 0 {
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
