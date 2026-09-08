use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;

use crate::descriptor_bytes::push_usize;
use crate::schedule::GroupCommitPhaseParams;
use crate::sis::{
    num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    InnerCommitMatrixParams, SisMatrixRole, SisModulusProfileId, SisSecurityPolicyId,
    SisTableDigest,
};
use crate::{CommitmentRingDims, DecompositionParams};

/// Schedule-selected procedure for opening one committed group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OpeningMethod {
    /// Open full A-ring partials with evaluation-trace weights.
    EvaluationTrace,
    /// Pack coefficients over the selected challenge subring.
    SubringCoefficientPacking {
        /// Dimension of the sparse fold-challenge subring.
        challenge_subring_dimension: usize,
    },
}

/// Runtime value carried by one of Akita's two opening methods.
///
/// The schedule chooses an [`OpeningMethod`]; this family preserves that same
/// method distinction while each protocol stage supplies its own payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpeningFamily<Trace, Packing> {
    /// Value belonging to the full-A evaluation-trace method.
    EvaluationTrace(Trace),
    /// Value belonging to subring coefficient packing.
    SubringCoefficientPacking(Packing),
}

impl OpeningMethod {
    /// Whether this method uses extension-opening reduction for the field tower.
    #[must_use]
    pub const fn requires_extension_opening_reduction(self, extension_degree: usize) -> bool {
        matches!(self, Self::EvaluationTrace) && extension_degree > 1
    }

    pub(crate) fn append_descriptor_bytes(self, bytes: &mut Vec<u8>) {
        match self {
            Self::EvaluationTrace => bytes.push(0),
            Self::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => {
                bytes.push(1);
                push_usize(bytes, challenge_subring_dimension);
            }
        }
    }

    /// Physical base-field coefficient width opened by this method.
    pub fn physical_coefficient_width(
        self,
        extension_degree: usize,
        inner_ring_dimension: usize,
    ) -> Result<usize, AkitaError> {
        match self {
            Self::EvaluationTrace => Ok(inner_ring_dimension),
            Self::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => Ok(crate::SubringCoefficientPackingGeometry::try_new(
                extension_degree,
                inner_ring_dimension,
                challenge_subring_dimension,
            )?
            .partial_base_field_width()),
        }
    }
}

/// Ring-column width of one group's decomposed opening segment in the shared
/// D matrix.
///
/// EvaluationTrace decomposes a full A-ring partial. Coefficient packing
/// decomposes its `k * s` physical base-field coordinates instead. This is the
/// canonical sizing authority used by planners, artifact validation, and
/// authenticated schedule replay.
pub fn opening_d_segment_width(
    opening_method: OpeningMethod,
    extension_degree: usize,
    inner_ring_dimension: usize,
    opening_ring_dimension: usize,
    num_digits_open: usize,
    num_live_blocks: usize,
    num_claims: usize,
) -> Result<usize, AkitaError> {
    if opening_ring_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "group D opening dimension must be nonzero".into(),
        ));
    }
    let physical_width =
        opening_method.physical_coefficient_width(extension_degree, inner_ring_dimension)?;
    let role_subcolumns = physical_width
        .checked_div(opening_ring_dimension)
        .filter(|_| physical_width.is_multiple_of(opening_ring_dimension))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "group opening width does not decompose into D-native subcolumns".into(),
            )
        })?;
    num_digits_open
        .checked_mul(num_live_blocks)
        .and_then(|width| width.checked_mul(num_claims))
        .and_then(|width| width.checked_mul(role_subcolumns))
        .ok_or_else(|| AkitaError::InvalidSetup("group D segment width overflow".into()))
}

/// Opening policy selected by the fold that consumes a committed group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GroupOpeningPlan {
    /// Procedure used to reduce and open the group's coefficients.
    pub opening_method: OpeningMethod,
    /// Sparse fold-challenge family certified for this group's A ring or subring.
    pub fold_challenge_config: SparseChallengeConfig,
    /// Opening basis used by the shared D matrix for fresh `e_hat` digits.
    pub log_basis_open: u32,
    /// Gadget decomposition depth for fresh `e_hat` values.
    pub num_digits_open: usize,
    /// Exact folded-witness digit depth selected by this schedule row.
    pub num_digits_fold: usize,
}

impl GroupOpeningPlan {
    /// Build the opening plan used by every schedule before subring packing.
    #[must_use]
    pub const fn evaluation_trace(
        fold_challenge_config: SparseChallengeConfig,
        log_basis_open: u32,
        num_digits_open: usize,
        num_digits_fold: usize,
    ) -> Self {
        Self {
            opening_method: OpeningMethod::EvaluationTrace,
            fold_challenge_config,
            log_basis_open,
            num_digits_open,
            num_digits_fold,
        }
    }

    /// Canonical schedule descriptor for this consuming opening policy.
    #[must_use]
    pub fn canonical_descriptor_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    pub(crate) fn append_descriptor_bytes(self, bytes: &mut Vec<u8>) {
        self.opening_method.append_descriptor_bytes(bytes);
        crate::descriptor_bytes::push_u32(bytes, self.log_basis_open);
        super::append_schedule_sparse_challenge_descriptor_bytes(
            bytes,
            &self.fold_challenge_config,
        );
        push_usize(bytes, self.num_digits_open);
        push_usize(bytes, self.num_digits_fold);
    }
}

/// One commitment group taking part in one fold's opening batch.
///
/// Every group in a fold has this type: the final/new group, each precommitted
/// group, and the setup prefix. The fold owns the shared D matrix; a group owns
/// only its contribution of D digits, through `opening`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GroupOpenPhaseParams {
    /// Frozen commit-phase identity of this group.
    pub profile: GroupCommitPhaseParams,
    /// Opening policy owned by the fold that consumes this commitment.
    pub opening: GroupOpeningPlan,
    /// Active setup-weight support, in flat field coefficients.
    ///
    /// `Some` exactly when this group is the consuming fold's setup prefix.
    /// This is the sole record of that fact and the sole record of the length,
    /// so there is no second field to audit it against.
    pub setup_natural_len: Option<usize>,
}

/// Security and decomposition policy needed to admit a frozen precommit into
/// a grouped opening. Planner and runtime replay both use this exact context.
#[derive(Debug, Clone, Copy)]
pub struct PrecommittedGroupAdmissionPolicy {
    /// Field and signed-digit decomposition policy.
    pub decomposition: DecompositionParams,
    /// Bound policy used by the canonical SIS lookup.
    pub sis_security_policy: SisSecurityPolicyId,
    /// Digest binding the exact generated SIS table.
    pub sis_table_digest: SisTableDigest,
    /// Modulus family required for both frozen matrices.
    pub sis_modulus_profile: SisModulusProfileId,
    /// Number of equal-envelope folded responses retained by the consuming fold.
    /// This value prices the shared A rows and is owned by the consuming level;
    /// it is not copied into the frozen group's descriptor.
    pub num_response_chunks: usize,
}

impl GroupOpenPhaseParams {
    /// Registry identity for a prefix group; `None` for an ordinary group.
    ///
    /// `SetupPrefixSlotId` stays the runtime registry key. It is derived here
    /// rather than stored, which removes the third copy of a prefix's frozen
    /// commit-phase identity.
    #[must_use]
    pub fn slot_id(&self) -> Option<crate::SetupPrefixSlotId> {
        self.setup_natural_len
            .map(|natural_len| crate::SetupPrefixSlotId {
                natural_len,
                commitment_profile: self.profile,
            })
    }

    /// Full power-of-two flat coefficient length committed for this prefix.
    pub fn n_prefix(&self) -> Result<usize, AkitaError> {
        self.slot_id()
            .ok_or_else(|| AkitaError::InvalidSetup("group is not a setup prefix".to_string()))?
            .n_prefix()
    }

    /// Ring dimension used for the frozen setup-prefix commitment.
    #[must_use]
    pub fn d_setup(&self) -> usize {
        self.profile.inner.matrix.ring_dimension()
    }

    /// Canonical bytes for deterministic planner ordering and schedule identity.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    /// Validate and materialize one ordinary frozen precommit at the
    /// batch-shared opening basis.
    pub fn admit(
        layout: GroupCommitPhaseParams,
        num_digits_fold: usize,
        policy: PrecommittedGroupAdmissionPolicy,
        opening_method: OpeningMethod,
        fold_challenge_config: SparseChallengeConfig,
        log_basis_open: u32,
    ) -> Result<Self, AkitaError> {
        Self::admit_with_setup_natural_len(
            layout,
            None,
            num_digits_fold,
            policy,
            opening_method,
            fold_challenge_config,
            log_basis_open,
        )
    }

    /// Validate and materialize a recursive setup prefix at the batch-shared
    /// opening basis.
    pub fn admit_setup_prefix(
        layout: GroupCommitPhaseParams,
        natural_len: usize,
        num_digits_fold: usize,
        policy: PrecommittedGroupAdmissionPolicy,
        opening_method: OpeningMethod,
        fold_challenge_config: SparseChallengeConfig,
        log_basis_open: u32,
    ) -> Result<Self, AkitaError> {
        Self::admit_with_setup_natural_len(
            layout,
            Some(natural_len),
            num_digits_fold,
            policy,
            opening_method,
            fold_challenge_config,
            log_basis_open,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_with_setup_natural_len(
        layout: GroupCommitPhaseParams,
        setup_natural_len: Option<usize>,
        num_digits_fold: usize,
        policy: PrecommittedGroupAdmissionPolicy,
        opening_method: OpeningMethod,
        fold_challenge_config: SparseChallengeConfig,
        log_basis_open: u32,
    ) -> Result<Self, AkitaError> {
        if let Some(natural_len) = setup_natural_len {
            layout.validate(policy.decomposition.field_bits())?;
            layout.validate_setup_prefix_geometry(natural_len)?;
        } else {
            layout.validate_frozen_precommit(policy.decomposition.field_bits())?;
        }
        if layout.inner.matrix.sis_modulus_profile() != policy.sis_modulus_profile
            || layout.outer.matrix.sis_modulus_profile() != policy.sis_modulus_profile
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted group modulus profile does not match admission policy".into(),
            ));
        }

        let outer_decomposition = DecompositionParams {
            log_basis: layout.outer.digits.log_basis,
            ..policy.decomposition
        };
        if num_digits_open(outer_decomposition) != layout.outer.digits.num_digits {
            return Err(AkitaError::InvalidSetup(
                "precommitted outer digit depth does not match its frozen basis".into(),
            ));
        }
        let frozen_b_bound = rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Outer,
            layout.outer.matrix.ring_dimension(),
            layout.outer.digits.log_basis,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no precommitted B-role norm".into()))?;
        if layout.outer.matrix.coeff_linf_bound() < frozen_b_bound {
            return Err(AkitaError::InvalidSetup(
                "precommitted group B bound is below its frozen outer-basis requirement".into(),
            ));
        }
        if log_basis_open < layout.outer.digits.log_basis {
            return Err(AkitaError::InvalidSetup(
                "certified opening basis must dominate the precommitted outer basis".into(),
            ));
        }

        let opening_decomposition = DecompositionParams {
            log_basis: log_basis_open,
            ..policy.decomposition
        };
        let num_digits_open = num_digits_open(opening_decomposition);
        let required_a_bound = rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            layout.inner.matrix.ring_dimension(),
            log_basis_open,
            &fold_challenge_config,
            num_digits_fold,
            policy.num_response_chunks,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no precommitted A-role norm".into()))?;
        let declared_a_bound = layout.inner.matrix.coeff_linf_bound().ok_or_else(|| {
            AkitaError::InvalidSetup("precommitted A cannot use an L2 security route".into())
        })?;
        if required_a_bound > declared_a_bound {
            return Err(AkitaError::InvalidSetup(
                "precommitted A bound does not cover the certified opening basis".into(),
            ));
        }
        let required_b_bound = rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Outer,
            layout.outer.matrix.ring_dimension(),
            log_basis_open,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no precommitted B-role norm".into()))?;
        if required_b_bound > layout.outer.matrix.coeff_linf_bound() {
            return Err(AkitaError::InvalidSetup(
                "precommitted B bound does not cover the certified opening basis".into(),
            ));
        }

        let params = Self {
            profile: layout,
            setup_natural_len,
            opening: GroupOpeningPlan {
                opening_method,
                fold_challenge_config,
                log_basis_open,
                num_digits_open,
                num_digits_fold,
            },
        };
        params.validate()?;
        Ok(params)
    }

    /// Worst-case L1 mass of this group's fold-round challenge.
    #[inline]
    #[must_use]
    pub fn challenge_l1_mass(&self) -> usize {
        self.opening.fold_challenge_config.l1_norm()
    }

    /// This group's A/B dimensions completed with the consuming level's shared
    /// D dimension.
    #[must_use]
    pub fn role_dims(&self, shared_opening_ring_dimension: usize) -> CommitmentRingDims {
        CommitmentRingDims {
            inner: self.profile.inner.matrix.ring_dimension(),
            outer: self.profile.outer.matrix.ring_dimension(),
            opening: shared_opening_ring_dimension,
        }
    }

    /// Validate role ownership and exact A/B widths for serialized group params.
    pub fn validate(&self) -> Result<(), AkitaError> {
        let field_bits = self.profile.inner.matrix.sis_modulus_profile().field_bits();
        self.profile.validate(field_bits)?;
        if let Some(natural_len) = self.setup_natural_len {
            self.profile.validate_setup_prefix_geometry(natural_len)?;
        }
        if self.opening.fold_challenge_config.weight() != 0 {
            let challenge_dimension = match self.opening.opening_method {
                OpeningMethod::EvaluationTrace => self.profile.inner.matrix.ring_dimension(),
                OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } => challenge_subring_dimension,
            };
            self.opening
                .fold_challenge_config
                .validate_for_ring_dim(challenge_dimension)
                .map_err(|msg| AkitaError::InvalidSetup(msg.to_string()))?;
        }
        if self.opening.log_basis_open == 0
            || self.opening.num_digits_open == 0
            || self.opening.num_digits_fold == 0
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted exact fold plan is missing or inconsistent".to_string(),
            ));
        }
        if self.opening.log_basis_open < self.profile.outer.digits.log_basis {
            return Err(AkitaError::InvalidSetup(
                "certified opening basis must dominate the precommitted outer basis".to_string(),
            ));
        }
        let expected_a_width = self
            .profile
            .blocks
            .positions_per_block
            .checked_mul(self.profile.inner.digits.num_digits)
            .ok_or_else(|| AkitaError::InvalidSetup("precommitted A width overflow".to_string()))?;
        let inner_ring_dimension = self.profile.inner.matrix.ring_dimension();
        let outer_ring_dimension = self.profile.outer.matrix.ring_dimension();
        if outer_ring_dimension == 0 || !inner_ring_dimension.is_multiple_of(outer_ring_dimension) {
            return Err(AkitaError::InvalidSetup(
                "precommitted A-native source rings do not decompose into B-native subcolumns"
                    .to_string(),
            ));
        }
        let expected_b_width = crate::CommitmentSliceGeometry::try_new(
            self.profile.outer_slice_count,
            self.profile.blocks.live_blocks,
            self.profile.group.num_polynomials(),
            self.profile.inner.matrix.output_rank(),
            self.profile.outer.digits.num_digits,
            inner_ring_dimension,
            outer_ring_dimension,
        )?
        .physical_input_width();
        if self.profile.inner.matrix.input_width() != expected_a_width
            || self.profile.outer.matrix.input_width() != expected_b_width
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted A/B keys do not match frozen ranks, bounds, or digit depths"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Width of this group's A matrix.
    #[inline]
    pub fn inner_width(&self) -> usize {
        self.profile.inner.matrix.input_width()
    }

    /// Width of this group's B matrix.
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.profile.outer.matrix.input_width()
    }

    /// Width contribution to the consuming batch's shared D matrix
    /// (`w_hat_g` segment).
    ///
    /// Group metadata owns its A/B dimensions. The D role is batch-shared, so
    /// the caller supplies the consuming level's opening dimension.
    pub fn d_segment_width(
        &self,
        extension_degree: usize,
        opening_ring_dimension: usize,
    ) -> Result<usize, AkitaError> {
        opening_d_segment_width(
            self.opening.opening_method,
            extension_degree,
            self.profile.inner.matrix.ring_dimension(),
            opening_ring_dimension,
            self.opening.num_digits_open,
            self.profile.blocks.live_blocks,
            self.profile.group.num_polynomials(),
        )
    }

    /// Width contribution of this group's decomposed folded response.
    pub fn z_segment_width(&self, num_digits_fold: usize) -> Result<usize, AkitaError> {
        self.inner_width()
            .checked_mul(num_digits_fold)
            .ok_or_else(|| AkitaError::InvalidSetup("group z segment width overflow".to_string()))
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.profile.append_descriptor_bytes(bytes);
        self.opening.append_descriptor_bytes(bytes);
    }
}

impl GroupOpenPhaseParams {
    /// Logical B output rows after un-slicing the physical matrix.
    ///
    /// Mirrors the trait's default body.
    pub fn logical_b_rows_len(&self) -> Result<usize, AkitaError> {
        self.outer_slice_count()
            .logical_output_rows(self.b_rows_len())
    }
}

/// Commitment geometry and opening policy carried by one group.
///
/// These were the 22 methods of the deleted `LevelParamsLike` trait, which
/// existed only to let a caller treat a fold's final group and a precommitted
/// group uniformly without knowing which it held. Both are the same
/// type.
impl GroupOpenPhaseParams {
    #[inline]
    #[must_use]
    pub fn source_encoding(&self) -> crate::CommittedSourceEncoding {
        crate::CommittedSourceEncoding::CanonicalCoefficientTable
    }

    #[inline]
    #[must_use]
    pub fn opening_method(&self) -> OpeningMethod {
        self.opening.opening_method
    }

    #[inline]
    #[must_use]
    pub fn inner_commit_matrix_params(&self) -> &InnerCommitMatrixParams {
        &self.profile.inner.matrix
    }

    #[inline]
    #[must_use]
    pub fn a_rows_len(&self) -> usize {
        self.profile.inner.matrix.output_rank()
    }

    #[inline]
    #[must_use]
    pub fn a_col_len(&self) -> usize {
        self.profile.inner.matrix.input_width()
    }

    #[inline]
    #[must_use]
    pub fn b_rows_len(&self) -> usize {
        self.profile.outer.matrix.output_rank()
    }

    #[inline]
    #[must_use]
    pub fn outer_slice_count(&self) -> crate::CommitmentSliceCount {
        self.profile.outer_slice_count
    }

    #[inline]
    #[must_use]
    pub fn b_col_len(&self) -> usize {
        self.profile.outer.matrix.input_width()
    }

    #[inline]
    #[must_use]
    pub fn num_live_ring_elements_per_claim(&self) -> usize {
        self.profile.blocks.live_ring_elements_per_claim
    }

    #[inline]
    #[must_use]
    pub fn num_positions_per_block(&self) -> usize {
        self.profile.blocks.positions_per_block
    }

    #[inline]
    #[must_use]
    pub fn num_live_blocks(&self) -> usize {
        self.profile.blocks.live_blocks
    }

    #[inline]
    #[must_use]
    pub fn fold_challenge_config(&self) -> SparseChallengeConfig {
        self.opening.fold_challenge_config
    }

    #[inline]
    #[must_use]
    pub fn position_index_bits(&self) -> usize {
        self.profile.blocks().position_index_bits()
    }

    #[inline]
    #[must_use]
    pub fn block_index_bits(&self) -> usize {
        self.profile.blocks().block_index_bits()
    }

    #[inline]
    #[must_use]
    pub fn num_digits_inner(&self) -> usize {
        self.profile.inner.digits.num_digits
    }

    #[inline]
    #[must_use]
    pub fn num_digits_outer(&self) -> usize {
        self.profile.outer.digits.num_digits
    }

    #[inline]
    #[must_use]
    pub fn num_digits_open(&self) -> usize {
        self.opening.num_digits_open
    }

    #[inline]
    #[must_use]
    pub fn num_digits_fold(&self) -> usize {
        self.opening.num_digits_fold
    }

    #[inline]
    #[must_use]
    pub fn log_basis_outer(&self) -> u32 {
        self.profile.outer.digits.log_basis
    }

    #[inline]
    #[must_use]
    pub fn log_basis_inner(&self) -> u32 {
        self.profile.inner.digits.log_basis
    }

    #[inline]
    #[must_use]
    pub fn log_basis_open(&self) -> u32 {
        self.opening.log_basis_open
    }
}
