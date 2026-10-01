use super::*;

impl CommittedGroupParams {
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
            ring_relation_mode: crate::RingRelationMode::QuotientLift,
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
            ring_relation_mode: self.ring_relation_mode,
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
            ring_relation_mode: other.ring_relation_mode,
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
