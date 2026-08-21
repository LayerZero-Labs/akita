use super::*;

impl CommittedGroupParams {
    /// Reject multi-group-root params at scalar-only call sites.
    pub fn require_scalar_level(&self, context: &str) -> Result<(), AkitaError> {
        if self.has_preceding_groups() {
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
        crate::TranscriptGrindingBinding::validate_fold_response_nonce(fold_grind_nonce)
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
        let geometry = own_profile.derive_slice_geometry()?;
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
}
