use crate::opaque::CommitInnerPlan;
use akita_error::{checked, AkitaError};
use akita_types::{
    validate_setup_prefix_domain, CommitmentPayloadMode, CommitmentSliceGeometry,
    CommittedGroupParams, CompressionChainPlan, GroupCommitPhaseParams, RingRelationMode,
    SetupPrefixSlotId, SisModulusProfileId, TerminalFoldParams,
};

/// Checked B-stage arithmetic plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OuterCommitPlan {
    /// Number of rows produced by each physical B application.
    n_b: usize,
    /// Runtime B ring dimension.
    ring_dimension: usize,
    /// Number of balanced outer digits.
    num_digits_outer: usize,
    /// Logarithm of the outer gadget basis.
    log_basis_outer: u32,
    /// Canonical slicing and padding geometry.
    geometry: CommitmentSliceGeometry,
}

impl OuterCommitPlan {
    /// Construct a checked B-stage plan.
    pub(crate) fn new(
        n_b: usize,
        ring_dimension: usize,
        num_digits_outer: usize,
        log_basis_outer: u32,
        geometry: CommitmentSliceGeometry,
    ) -> Result<Self, AkitaError> {
        if n_b == 0
            || ring_dimension == 0
            || !ring_dimension.is_power_of_two()
            || num_digits_outer == 0
            || log_basis_outer == 0
        {
            return Err(AkitaError::InvalidSetup(
                "outer commitment plan requires nonzero power-of-two ring geometry and digit parameters"
                    .into(),
            ));
        }
        checked::product([geometry.slice_count().get(), n_b, ring_dimension]).ok_or_else(|| {
            AkitaError::InvalidSetup("outer commitment output length overflow".into())
        })?;
        Ok(Self {
            n_b,
            ring_dimension,
            num_digits_outer,
            log_basis_outer,
            geometry,
        })
    }

    /// Exact coefficient length of the canonical stacked B image.
    pub(crate) fn output_coefficient_len(&self) -> Result<usize, AkitaError> {
        checked::product([
            self.geometry.slice_count().get(),
            self.n_b,
            self.ring_dimension,
        ])
        .ok_or_else(|| AkitaError::InvalidSetup("outer commitment output length overflow".into()))
    }

    /// Number of B-matrix output rows per physical slice.
    pub(crate) const fn n_b(&self) -> usize {
        self.n_b
    }

    /// Runtime B-ring dimension.
    pub(crate) const fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }

    /// Number of outer balanced digits.
    pub(crate) const fn num_digits_outer(&self) -> usize {
        self.num_digits_outer
    }

    /// Logarithm of the outer decomposition basis.
    pub(crate) const fn log_basis_outer(&self) -> u32 {
        self.log_basis_outer
    }

    /// Canonical source slicing and padding geometry.
    pub(crate) const fn geometry(&self) -> &CommitmentSliceGeometry {
        &self.geometry
    }
}

/// Shared checked plan for inner-plus-outer execution.
#[derive(Debug, Clone)]
pub(crate) struct UncompressedCommitPlan {
    inner: CommitInnerPlan,
    outer: OuterCommitPlan,
}

impl UncompressedCommitPlan {
    /// A-stage view.
    pub(crate) const fn inner(&self) -> &CommitInnerPlan {
        &self.inner
    }

    /// B-stage view.
    pub(crate) const fn outer(&self) -> &OuterCommitPlan {
        &self.outer
    }
}

/// Protocol mode represented by a commitment execution plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitmentExecutionMode {
    /// Root or setup-prefix A/B commitment followed by compression.
    Full,
    /// Recursive A/B commitment without compression.
    Uncompressed,
    /// Terminal A-only commitment.
    InnerOnly,
}

/// Canonical checked commitment arithmetic plan.
#[derive(Debug, Clone)]
pub(crate) struct CommitmentExecutionPlan {
    fold_level: usize,
    mode: CommitmentExecutionMode,
    uncompressed: Option<UncompressedCommitPlan>,
    inner_only: CommitInnerPlan,
    compression: Option<CompressionChainPlan>,
    relation_mode: Option<RingRelationMode>,
    inner_modulus_profile: SisModulusProfileId,
}

impl CommitmentExecutionPlan {
    /// Largest shared-setup prefix touched by this exact execution mode.
    pub(crate) fn max_setup_field_elements(&self) -> Result<usize, AkitaError> {
        let inner = self.inner();
        let inner_width = checked::product([inner.num_positions_per_block, inner.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("commitment A width overflow".into()))?;
        let outer = self.uncompressed().map(|plan| {
            let outer = plan.outer();
            akita_types::CommitmentSetupMatrixShape {
                rows: outer.n_b(),
                columns: outer.geometry().physical_input_width(),
                ring_dimension: outer.ring_dimension(),
            }
        });
        akita_types::commitment_execution_setup_field_elements(
            akita_types::CommitmentSetupMatrixShape {
                rows: inner.n_a,
                columns: inner_width,
                ring_dimension: inner.ring_dimension,
            },
            outer,
            self.compression(),
        )
    }

    /// Build a standalone root plan, owned by fold-zero resources, from an
    /// already-resolved frozen commitment profile.
    pub(crate) fn for_root(profile: &GroupCommitPhaseParams) -> Result<Self, AkitaError> {
        Self::from_profile(
            0,
            profile,
            CommitmentPayloadMode::Compressed,
            RingRelationMode::QuotientLift,
        )
    }

    /// Build the material-consumption plan for one group in a fold relation.
    ///
    /// Unlike [`Self::for_root`], this preserves the consuming level's payload
    /// and ring-relation modes. Those modes determine whether compression
    /// material exists and which form it takes.
    pub(crate) fn for_relation_group(
        params: &CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        group_index: usize,
        fold_level: usize,
    ) -> Result<Self, AkitaError> {
        let group = params.group_params(opening_batch, group_index)?;
        Self::from_profile(
            fold_level,
            &group.profile,
            params.payload_mode,
            params.ring_relation_mode,
        )
    }

    /// Build a recursive plan for the predecessor fold whose stack performs
    /// this commitment.
    pub(crate) fn for_recursive(
        params: &CommittedGroupParams,
        owner_fold_level: usize,
        num_polynomials: usize,
    ) -> Result<Self, AkitaError> {
        let logical_fold_level = owner_fold_level
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidSetup("fold level overflow".into()))?;
        params.validate_commitment_request(logical_fold_level, num_polynomials)?;
        Self::from_profile(
            owner_fold_level,
            &params.own_group().profile,
            params.payload_mode,
            params.ring_relation_mode,
        )
    }

    /// Build a terminal A-only plan for the predecessor fold whose stack
    /// performs this commitment.
    pub(crate) fn for_terminal(
        params: &TerminalFoldParams,
        fold_level: usize,
    ) -> Result<Self, AkitaError> {
        if params.blocks.live_blocks == 0
            || params.blocks.positions_per_block == 0
            || params.inner.matrix.output_rank() == 0
            || params.inner.digits.num_digits == 0
        {
            return Err(AkitaError::InvalidSetup(
                "terminal commitment plan requires nonzero block, row, and digit geometry".into(),
            ));
        }
        let inner = CommitInnerPlan {
            ring_dimension: params.inner.matrix.ring_dimension(),
            num_live_blocks: params.blocks.live_blocks,
            n_a: params.inner.matrix.output_rank(),
            num_positions_per_block: params.blocks.positions_per_block,
            num_digits_inner: params.inner.digits.num_digits,
            log_basis_inner: params.inner.digits.log_basis,
        };
        Ok(Self {
            fold_level,
            mode: CommitmentExecutionMode::InnerOnly,
            uncompressed: None,
            inner_only: inner,
            compression: None,
            relation_mode: None,
            inner_modulus_profile: params.inner.matrix.sis_modulus_profile(),
        })
    }

    /// Build a standalone fold-zero setup-prefix plan from the exact persisted
    /// slot identity.
    pub(crate) fn for_setup_prefix(slot: &SetupPrefixSlotId) -> Result<Self, AkitaError> {
        validate_setup_prefix_domain(slot.natural_len, slot.n_prefix()?)?;
        Self::for_root(&slot.commitment_profile)
    }

    fn from_profile(
        fold_level: usize,
        profile: &GroupCommitPhaseParams,
        payload_mode: CommitmentPayloadMode,
        relation_mode: RingRelationMode,
    ) -> Result<Self, AkitaError> {
        let geometry = profile.derive_slice_geometry()?;
        let inner = CommitInnerPlan::from_profile(profile);
        let outer = OuterCommitPlan::new(
            profile.outer.matrix.output_rank(),
            profile.outer.matrix.ring_dimension(),
            profile.outer.digits.num_digits,
            profile.outer.digits.log_basis,
            geometry,
        )?;
        let compression = if payload_mode.is_compressed() {
            let source_coefficients = profile
                .outer_slice_count
                .complete_source_coefficients(outer.n_b, outer.ring_dimension)?;
            Some(CompressionChainPlan::for_complete_source(
                profile.outer.matrix.sis_modulus_profile(),
                source_coefficients,
            )?)
        } else {
            None
        };
        let mode = if compression.is_some() {
            CommitmentExecutionMode::Full
        } else {
            CommitmentExecutionMode::Uncompressed
        };
        Ok(Self {
            fold_level,
            mode,
            uncompressed: Some(UncompressedCommitPlan { inner, outer }),
            inner_only: inner,
            compression,
            relation_mode: payload_mode.is_compressed().then_some(relation_mode),
            inner_modulus_profile: profile.inner.matrix.sis_modulus_profile(),
        })
    }

    /// Execution mode.
    pub(crate) const fn mode(&self) -> CommitmentExecutionMode {
        self.mode
    }

    /// Fold whose compute stack owns this commitment execution.
    pub(crate) const fn fold_level(&self) -> usize {
        self.fold_level
    }

    /// A-stage plan shared by every mode.
    pub(crate) const fn inner(&self) -> &CommitInnerPlan {
        &self.inner_only
    }

    /// A/B plan when this mode executes B.
    pub(crate) const fn uncompressed(&self) -> Option<&UncompressedCommitPlan> {
        self.uncompressed.as_ref()
    }

    /// Compression chain when this mode compresses B output.
    pub(crate) const fn compression(&self) -> Option<&CompressionChainPlan> {
        self.compression.as_ref()
    }

    /// Explicit schedule-selected relation mode when compression runs.
    pub(crate) const fn relation_mode(&self) -> Option<RingRelationMode> {
        self.relation_mode
    }

    /// SIS modulus profile governing inner matrix arithmetic.
    pub(crate) const fn inner_modulus_profile(&self) -> SisModulusProfileId {
        self.inner_modulus_profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{CommitmentPayloadMode, SisModulusProfileId};

    fn recursive_params(payload_mode: CommitmentPayloadMode) -> CommittedGroupParams {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q64Offset59,
            64,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        params.payload_mode = payload_mode;
        params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
        params
    }

    #[test]
    fn canonical_plan_preserves_payload_and_relation_modes() {
        let compressed = recursive_params(CommitmentPayloadMode::Compressed);
        let plan = CommitmentExecutionPlan::from_profile(
            0,
            &compressed.own_group().profile,
            compressed.payload_mode,
            compressed.ring_relation_mode,
        )
        .unwrap();
        assert_eq!(plan.mode(), CommitmentExecutionMode::Full);
        assert!(plan.compression().is_some());
        assert_eq!(
            plan.relation_mode(),
            Some(RingRelationMode::ReducedEvaluation)
        );
        assert_eq!(plan.inner().ring_dimension, 64);
        assert_eq!(plan.inner().num_live_blocks, 2);

        let raw = recursive_params(CommitmentPayloadMode::Raw);
        let plan = CommitmentExecutionPlan::from_profile(
            0,
            &raw.own_group().profile,
            raw.payload_mode,
            raw.ring_relation_mode,
        )
        .unwrap();
        assert_eq!(plan.mode(), CommitmentExecutionMode::Uncompressed);
        assert!(plan.compression().is_none());
        assert_eq!(plan.relation_mode(), None);
    }

    #[test]
    fn recursive_plan_preserves_resource_owner_fold() {
        let params = recursive_params(CommitmentPayloadMode::Raw);
        let plan = CommitmentExecutionPlan::from_profile(
            7,
            &params.own_group().profile,
            params.payload_mode,
            params.ring_relation_mode,
        )
        .unwrap();
        let inner = plan
            .inner_ntt_requirement(crate::commitment::PolynomialType::Dense(
                crate::commitment::DenseType::Coefficients,
            ))
            .unwrap()
            .unwrap();
        let outer = plan.outer_ntt_requirement().unwrap().unwrap();
        assert_eq!(inner.fold_level(), 7);
        assert_eq!(outer.fold_level(), 7);
    }

    #[test]
    fn outer_plan_checks_output_geometry_at_construction() {
        let profile = recursive_params(CommitmentPayloadMode::Raw)
            .own_group()
            .profile;
        let geometry = profile.derive_slice_geometry().unwrap();
        assert!(OuterCommitPlan::new(0, 64, 1, 1, geometry.clone()).is_err());
        assert!(OuterCommitPlan::new(1, 63, 1, 1, geometry.clone()).is_err());

        let plan = OuterCommitPlan::new(2, 64, 1, 1, geometry).unwrap();
        assert_eq!(
            plan.output_coefficient_len().unwrap(),
            plan.geometry().slice_count().get() * 2 * 64
        );
    }
}
