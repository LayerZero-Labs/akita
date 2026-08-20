use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

use crate::descriptor_bytes::push_usize;
use crate::schedule::CommittedGroupProfile;
use crate::sis::{
    num_digits_open, rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm,
    InnerCommitMatrixParams, SisMatrixRole, SisModulusProfileId, SisSecurityPolicyId,
    SisTableDigest,
};
use crate::{CommitmentRingDims, DecompositionParams};

use super::CommittedGroupParams;

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
/// canonical sizing authority used by planners, generated-row expansion, and
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

/// One frozen commitment profile and the policy selected by its consuming fold.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrecommittedLevelParams {
    /// Frozen standalone group layout bound into commitment identity.
    pub layout: CommittedGroupProfile,
    /// Opening policy owned by the fold that consumes this commitment.
    pub opening: GroupOpeningPlan,
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
}

impl PrecommittedLevelParams {
    /// Canonical bytes for deterministic planner ordering and schedule identity.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    /// Validate and materialize one frozen group at the batch-shared opening
    /// basis. This is the canonical admission path for planner generation and
    /// generated-schedule replay.
    pub fn admit(
        layout: CommittedGroupProfile,
        num_digits_fold: usize,
        policy: PrecommittedGroupAdmissionPolicy,
        opening_method: OpeningMethod,
        fold_challenge_config: SparseChallengeConfig,
        log_basis_open: u32,
    ) -> Result<Self, AkitaError> {
        layout.validate_frozen_precommit(policy.decomposition.field_bits())?;
        if layout.inner_commit_matrix.sis_modulus_profile() != policy.sis_modulus_profile
            || layout.outer_commit_matrix.sis_modulus_profile() != policy.sis_modulus_profile
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted group modulus profile does not match admission policy".into(),
            ));
        }

        let outer_decomposition = DecompositionParams {
            log_basis: layout.log_basis_outer,
            ..policy.decomposition
        };
        if num_digits_open(outer_decomposition) != layout.num_digits_outer {
            return Err(AkitaError::InvalidSetup(
                "precommitted outer digit depth does not match its frozen basis".into(),
            ));
        }
        let frozen_b_bound = rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            policy.sis_modulus_profile,
            SisMatrixRole::Outer,
            layout.outer_commit_matrix.ring_dimension(),
            layout.log_basis_outer,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no precommitted B-role norm".into()))?;
        if layout.outer_commit_matrix.coeff_linf_bound() < frozen_b_bound {
            return Err(AkitaError::InvalidSetup(
                "precommitted group B bound is below its frozen outer-basis requirement".into(),
            ));
        }
        if log_basis_open < layout.log_basis_outer {
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
            layout.inner_commit_matrix.ring_dimension(),
            log_basis_open,
            &fold_challenge_config,
            num_digits_fold,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no precommitted A-role norm".into()))?;
        let declared_a_bound = layout
            .inner_commit_matrix
            .coeff_linf_bound()
            .ok_or_else(|| {
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
            layout.outer_commit_matrix.ring_dimension(),
            log_basis_open,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("no precommitted B-role norm".into()))?;
        if required_b_bound > layout.outer_commit_matrix.coeff_linf_bound() {
            return Err(AkitaError::InvalidSetup(
                "precommitted B bound does not cover the certified opening basis".into(),
            ));
        }

        let params = Self {
            layout,
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
            inner: self.layout.inner_commit_matrix.ring_dimension(),
            outer: self.layout.outer_commit_matrix.ring_dimension(),
            opening: shared_opening_ring_dimension,
        }
    }

    /// Validate role ownership and exact A/B widths for serialized group params.
    pub fn validate(&self) -> Result<(), AkitaError> {
        let field_bits = self
            .layout
            .inner_commit_matrix
            .sis_modulus_profile()
            .field_bits();
        self.layout.validate(field_bits)?;
        if self.opening.fold_challenge_config.weight() != 0 {
            let challenge_dimension = match self.opening.opening_method {
                OpeningMethod::EvaluationTrace => self.layout.inner_commit_matrix.ring_dimension(),
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
        if self.opening.log_basis_open < self.layout.log_basis_outer {
            return Err(AkitaError::InvalidSetup(
                "certified opening basis must dominate the precommitted outer basis".to_string(),
            ));
        }
        let expected_a_width = self
            .layout
            .num_positions_per_block
            .checked_mul(self.layout.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("precommitted A width overflow".to_string()))?;
        let inner_ring_dimension = self.layout.inner_commit_matrix.ring_dimension();
        let outer_ring_dimension = self.layout.outer_commit_matrix.ring_dimension();
        if outer_ring_dimension == 0 || !inner_ring_dimension.is_multiple_of(outer_ring_dimension) {
            return Err(AkitaError::InvalidSetup(
                "precommitted A-native source rings do not decompose into B-native subcolumns"
                    .to_string(),
            ));
        }
        let expected_b_width = crate::CommitmentSliceGeometry::try_new(
            self.layout.outer_slice_count,
            self.layout.num_live_blocks,
            self.layout.group.num_polynomials(),
            self.layout.inner_commit_matrix.output_rank(),
            self.layout.num_digits_outer,
            inner_ring_dimension,
            outer_ring_dimension,
        )?
        .physical_input_width();
        if self.layout.inner_commit_matrix.input_width() != expected_a_width
            || self.layout.outer_commit_matrix.input_width() != expected_b_width
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
        self.layout.inner_commit_matrix.input_width()
    }

    /// Width of this group's B matrix.
    #[inline]
    pub fn outer_width(&self) -> usize {
        self.layout.outer_commit_matrix.input_width()
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
            self.layout.inner_commit_matrix.ring_dimension(),
            opening_ring_dimension,
            self.opening.num_digits_open,
            self.layout.num_live_blocks,
            self.layout.group.num_polynomials(),
        )
    }

    /// Width contribution of this group's decomposed folded response.
    pub fn z_segment_width(&self, num_digits_fold: usize) -> Result<usize, AkitaError> {
        self.inner_width()
            .checked_mul(num_digits_fold)
            .ok_or_else(|| AkitaError::InvalidSetup("group z segment width overflow".to_string()))
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.layout.append_descriptor_bytes(bytes);
        self.opening.append_descriptor_bytes(bytes);
    }
}

/// Common view over full and precommitted level parameters.
///
/// Use this trait when code only needs the shared commitment geometry carried
/// by both [`CommittedGroupParams`] and [`PrecommittedLevelParams`].
pub trait LevelParamsLike {
    fn source_encoding(&self) -> crate::CommittedSourceEncoding;
    fn opening_method(&self) -> OpeningMethod;
    fn inner_commit_matrix_params(&self) -> &InnerCommitMatrixParams;
    fn a_rows_len(&self) -> usize;
    fn a_col_len(&self) -> usize;
    fn b_rows_len(&self) -> usize;
    fn outer_slice_count(&self) -> crate::CommitmentSliceCount;
    fn logical_b_rows_len(&self) -> Result<usize, AkitaError> {
        self.outer_slice_count()
            .logical_output_rows(self.b_rows_len())
    }
    fn b_col_len(&self) -> usize;
    fn num_live_ring_elements_per_claim(&self) -> usize;
    fn num_positions_per_block(&self) -> usize;
    fn num_live_blocks(&self) -> usize;
    fn fold_challenge_config(&self) -> SparseChallengeConfig;
    fn position_index_bits(&self) -> usize;
    fn block_index_bits(&self) -> usize;
    fn num_digits_inner(&self) -> usize;
    fn num_digits_outer(&self) -> usize;
    fn num_digits_open(&self) -> usize;
    fn num_digits_fold(&self) -> usize;
    fn log_basis_inner(&self) -> u32;
    fn log_basis_outer(&self) -> u32;
    fn log_basis_open(&self) -> u32;
}

impl LevelParamsLike for CommittedGroupParams {
    fn source_encoding(&self) -> crate::CommittedSourceEncoding {
        self.source_encoding
    }

    fn opening_method(&self) -> OpeningMethod {
        self.opening_method
    }

    fn inner_commit_matrix_params(&self) -> &InnerCommitMatrixParams {
        &self.inner_commit_matrix
    }

    fn a_rows_len(&self) -> usize {
        self.inner_commit_matrix.output_rank()
    }

    fn a_col_len(&self) -> usize {
        self.inner_commit_matrix.input_width()
    }

    fn b_rows_len(&self) -> usize {
        self.outer_commit_matrix.output_rank()
    }

    fn outer_slice_count(&self) -> crate::CommitmentSliceCount {
        self.outer_slice_count
    }

    fn b_col_len(&self) -> usize {
        self.outer_commit_matrix.input_width()
    }

    fn num_live_ring_elements_per_claim(&self) -> usize {
        self.num_live_ring_elements_per_claim
    }

    fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }

    fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    fn fold_challenge_config(&self) -> SparseChallengeConfig {
        self.fold_challenge_config
    }

    fn position_index_bits(&self) -> usize {
        self.position_index_bits()
    }

    fn block_index_bits(&self) -> usize {
        self.block_index_bits()
    }

    fn num_digits_inner(&self) -> usize {
        self.num_digits_inner
    }

    fn num_digits_outer(&self) -> usize {
        self.num_digits_outer
    }

    fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    fn num_digits_fold(&self) -> usize {
        self.num_digits_fold
    }

    fn log_basis_outer(&self) -> u32 {
        self.log_basis_outer
    }

    fn log_basis_inner(&self) -> u32 {
        self.log_basis_inner
    }

    fn log_basis_open(&self) -> u32 {
        self.log_basis_open
    }
}

impl LevelParamsLike for PrecommittedLevelParams {
    fn source_encoding(&self) -> crate::CommittedSourceEncoding {
        crate::CommittedSourceEncoding::CanonicalCoefficientTable
    }

    fn opening_method(&self) -> OpeningMethod {
        self.opening.opening_method
    }

    fn inner_commit_matrix_params(&self) -> &InnerCommitMatrixParams {
        &self.layout.inner_commit_matrix
    }

    fn a_rows_len(&self) -> usize {
        self.layout.inner_commit_matrix.output_rank()
    }

    fn a_col_len(&self) -> usize {
        self.layout.inner_commit_matrix.input_width()
    }

    fn b_rows_len(&self) -> usize {
        self.layout.outer_commit_matrix.output_rank()
    }

    fn outer_slice_count(&self) -> crate::CommitmentSliceCount {
        self.layout.outer_slice_count
    }

    fn b_col_len(&self) -> usize {
        self.layout.outer_commit_matrix.input_width()
    }

    fn num_live_ring_elements_per_claim(&self) -> usize {
        self.layout.num_live_ring_elements_per_claim
    }

    fn num_positions_per_block(&self) -> usize {
        self.layout.num_positions_per_block
    }

    fn num_live_blocks(&self) -> usize {
        self.layout.num_live_blocks
    }

    fn fold_challenge_config(&self) -> SparseChallengeConfig {
        self.opening.fold_challenge_config
    }

    fn position_index_bits(&self) -> usize {
        self.layout.num_positions_per_block.trailing_zeros() as usize
    }

    fn block_index_bits(&self) -> usize {
        self.layout
            .num_live_blocks
            .checked_next_power_of_two()
            .map_or(0, |capacity| capacity.trailing_zeros() as usize)
    }

    fn num_digits_inner(&self) -> usize {
        self.layout.num_digits_inner
    }

    fn num_digits_outer(&self) -> usize {
        self.layout.num_digits_outer
    }

    fn num_digits_open(&self) -> usize {
        self.opening.num_digits_open
    }

    fn num_digits_fold(&self) -> usize {
        self.opening.num_digits_fold
    }

    fn log_basis_outer(&self) -> u32 {
        self.layout.log_basis_outer
    }

    fn log_basis_inner(&self) -> u32 {
        self.layout.log_basis_inner
    }

    fn log_basis_open(&self) -> u32 {
        self.opening.log_basis_open
    }
}
