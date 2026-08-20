//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::{
    CommittedGroupParams, InnerCommitMatrixParams, InnerCommitSecurityRoute, LevelParamsLike,
    OpeningMethod, RelationAddressGeometry, SetupContributionMode, TerminalResponseShape,
};
use akita_error::AkitaError;

mod descriptor;
mod profiles;
mod sizing;

pub use profiles::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, CommittedGroupProfile,
    CommittedSourceEncoding, PrecommittedGroupProfiles,
};
pub use sizing::{detect_field_modulus, r_decomp_levels};

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
    pub incoming_setup_prefix: Option<crate::ScheduledSetupPrefix>,
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
/// The terminal relation binds the source decomposition through the inner
/// commitment matrix. It also retains the terminal fold basis and digit count
/// needed to audit a calibrated L2 route. It has no outer/open commitment
/// matrix and no outer/open response decomposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalCommittedGroupParams {
    pub log_basis_inner: u32,
    /// Response basis used by the planner for this terminal fold.
    pub fold_log_basis: u32,
    /// Number of response digits used by the planner for this terminal fold.
    pub fold_digit_count: usize,
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
    /// Canonical byte encoding used to order semantically distinct terminal candidates.
    #[must_use]
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    pub fn from_expanded_group(params: CommittedGroupParams) -> Self {
        Self {
            log_basis_inner: params.log_basis_inner,
            fold_log_basis: params.log_basis_open,
            fold_digit_count: params.num_digits_fold,
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
        let (unconstrained_target, _) = crate::sis::fold_witness_linf_cap(
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

    /// Largest raw response admitted by a terminal Linf route's selected
    /// inner-matrix SIS bucket and signed coefficient representation.
    ///
    /// The matrix rank can incidentally support a larger collision bucket. The
    /// terminal wire does not consume that slack because doing so would change
    /// its admission and encoding bounds when an unrelated rank frontier moves.
    pub fn certified_response_linf_cap(
        &self,
        sparse: &akita_challenges::SparseChallengeConfig,
    ) -> Result<u128, AkitaError> {
        if matches!(
            self.inner_commit_matrix.security_route(),
            crate::sis::InnerCommitSecurityRoute::L2 { .. }
        ) {
            return Err(AkitaError::InvalidSetup(
                "terminal L2 route has no independent Linf cap".into(),
            ));
        }
        let challenge = crate::sis::FoldChallengeNorms::new(sparse);
        let collision_capacity = self.inner_commit_matrix.coeff_linf_bound().ok_or_else(|| {
            AkitaError::InvalidSetup("terminal A cannot use an L2 security route".into())
        })?;
        let certified_capacity = crate::sis::max_response_linf_for_role_a_collision(
            collision_capacity,
            challenge.l1_norm,
        )
        .filter(|value| *value > 0)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal A cannot certify a response".into()))?;
        // Terminal NTT kernels currently consume signed i16 coefficients.
        // This representation limit is independent of the SIS capacity.
        Ok(certified_capacity.min(i16::MAX as u128))
    }

    /// Validate that the wire carries exactly the norm cap required by the
    /// selected terminal security route.
    pub fn validate_terminal_linf_cap(
        &self,
        sparse: &akita_challenges::SparseChallengeConfig,
        scheduled_cap: Option<u128>,
    ) -> Result<(), AkitaError> {
        match self.inner_commit_matrix.security_route() {
            crate::sis::InnerCommitSecurityRoute::Linf(_) => {
                let cap = scheduled_cap.ok_or_else(|| {
                    AkitaError::InvalidSetup("terminal Linf route is missing its cap".into())
                })?;
                if cap == 0 || cap > self.certified_response_linf_cap(sparse)? {
                    return Err(AkitaError::InvalidSetup(
                        "terminal Linf cap exceeds its matrix-certified capacity".into(),
                    ));
                }
            }
            crate::sis::InnerCommitSecurityRoute::L2 { .. } => {
                if scheduled_cap.is_some() {
                    return Err(AkitaError::InvalidSetup(
                        "terminal L2 route must not carry an independent Linf cap".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Verifier-enforced complete physical L2 cap for a clear terminal route.
    #[must_use]
    pub fn response_l2_sq_cap(&self) -> Option<u128> {
        match self.inner_commit_matrix.security_route() {
            crate::sis::InnerCommitSecurityRoute::Linf(_) => None,
            crate::sis::InnerCommitSecurityRoute::L2 {
                response_l2_sq_cap, ..
            } => Some(response_l2_sq_cap),
        }
    }

    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_u32(bytes, self.log_basis_inner);
        push_u32(bytes, self.fold_log_basis);
        push_usize(bytes, self.fold_digit_count);
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

/// Borrowed nonterminal step used to encode a checked planner candidate
/// without constructing a temporary [`FoldSchedule`].
#[derive(Clone, Copy)]
pub struct FoldScheduleDescriptorStep<'a> {
    pub params: &'a CommittedGroupParams,
    pub payload_mode: crate::CommitmentPayloadMode,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
}

/// Borrowed terminal step used by canonical schedule descriptor encoding.
#[derive(Clone, Copy)]
pub struct TerminalFoldDescriptor<'a> {
    pub witness: &'a TerminalCommittedGroupParams,
    pub sparse_challenge_config: &'a akita_challenges::SparseChallengeConfig,
    pub response_shape: &'a TerminalResponseShape,
    pub input_witness_len: usize,
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
        let root_commitment = &self.root.params.final_group.commitment;
        root_commitment
            .validate_commitment_request(0, root_commitment.commitment_polynomial_count()?)?;
        for group in &self.root.params.precommitted_groups {
            group.commitment.validate()?;
        }
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
            step.params
                .witness
                .validate_commitment_request(index + 1, 1)?;
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
                prefix.commitment_params.validate()?;
                prefix
                    .commitment_params
                    .layout
                    .outer_slice_count
                    .validate_for_commitment(
                        0,
                        crate::CommitmentPayloadMode::Compressed,
                        prefix.commitment_params.layout.num_live_blocks,
                    )?;
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

    /// Validate the opening methods currently admitted by nonterminal proving
    /// and verification.
    ///
    /// Subring coefficient packing is required at absolute levels 0 and 1.
    /// Evaluation trace is required at later nonterminal levels. Every group
    /// consumed by one fold uses the same method family. Packing requires the
    /// audited production challenge family under the L-infinity A route.
    pub fn validate_nonterminal_opening_execution(
        &self,
        extension_degree: usize,
    ) -> Result<(), AkitaError> {
        self.validate_structure()?;
        if !extension_degree.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "opening extension degree must be a nonzero power of two".into(),
            ));
        }
        if !self.root.input_witness_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "root input witness length must be a power of two".into(),
            ));
        }
        let root_final = &self.root.params.final_group.commitment;
        let mut root_groups = vec![OpeningExecutionGroup {
            params: root_final,
            expected_source_encoding: Some(crate::CommittedSourceEncoding::for_producer(
                root_final.opening_method,
                extension_degree,
                root_final.d_a(),
                self.root.input_witness_len.trailing_zeros() as usize,
                true,
            )),
        }];
        root_groups.extend(self.root.params.precommitted_groups.iter().map(|group| {
            let commitment = &group.commitment;
            OpeningExecutionGroup {
                params: commitment as &dyn LevelParamsLike,
                expected_source_encoding: Some(crate::CommittedSourceEncoding::for_producer(
                    commitment.opening.opening_method,
                    extension_degree,
                    commitment.layout.inner_commit_matrix.ring_dimension(),
                    group.descriptor.group.num_vars(),
                    true,
                )),
            }
        }));
        validate_level_opening_execution(0, extension_degree, &root_groups)?;
        for (index, step) in self.recursive_folds.iter().enumerate() {
            let witness = &step.params.witness;
            let mut groups = vec![OpeningExecutionGroup {
                params: witness,
                expected_source_encoding: Some(crate::CommittedSourceEncoding::for_producer(
                    witness.opening_method,
                    extension_degree,
                    witness.d_a(),
                    0,
                    false,
                )),
            }];
            if let Some(prefix) = &step.params.incoming_setup_prefix {
                groups.push(OpeningExecutionGroup {
                    params: &prefix.commitment_params,
                    expected_source_encoding: None,
                });
            }
            validate_level_opening_execution(index + 1, extension_degree, &groups)?;
        }
        Ok(())
    }

    pub fn initial_witness_len(&self) -> usize {
        self.root.input_witness_len
    }
}

struct OpeningExecutionGroup<'a> {
    params: &'a dyn LevelParamsLike,
    expected_source_encoding: Option<crate::CommittedSourceEncoding>,
}

fn validate_level_opening_execution(
    absolute_level: usize,
    extension_degree: usize,
    groups: &[OpeningExecutionGroup<'_>],
) -> Result<(), AkitaError> {
    let first = groups
        .first()
        .ok_or_else(|| AkitaError::InvalidSetup("nonterminal fold has no opening groups".into()))?;
    let packing_family = matches!(
        first.params.opening_method(),
        OpeningMethod::SubringCoefficientPacking { .. }
    );
    let packing_required = absolute_level <= 1;
    if packing_family != packing_required {
        let required = if packing_required {
            "subring coefficient packing"
        } else {
            "evaluation trace"
        };
        return Err(AkitaError::InvalidSetup(format!(
            "nonterminal level {absolute_level} requires {required}"
        )));
    }
    if groups.iter().any(|group| {
        matches!(
            group.params.opening_method(),
            OpeningMethod::SubringCoefficientPacking { .. }
        ) != packing_family
    }) {
        return Err(AkitaError::InvalidSetup(
            "all groups consumed by one fold must use the same opening-method family".into(),
        ));
    }
    for group in groups {
        let opening_method = group.params.opening_method();
        match (opening_method, group.params.source_encoding()) {
            (
                OpeningMethod::EvaluationTrace,
                crate::CommittedSourceEncoding::TensorSubfieldProjection {
                    extension_degree: encoded_degree,
                },
            ) if encoded_degree != extension_degree => {
                return Err(AkitaError::InvalidSetup(
                    "tensor source encoding does not match the protocol extension degree".into(),
                ));
            }
            (
                OpeningMethod::SubringCoefficientPacking { .. },
                crate::CommittedSourceEncoding::TensorSubfieldProjection { .. },
            ) => {
                return Err(AkitaError::InvalidSetup(
                    "coefficient packing requires the canonical coefficient source encoding".into(),
                ));
            }
            _ => {}
        }
        if group
            .expected_source_encoding
            .is_some_and(|expected| expected != group.params.source_encoding())
        {
            return Err(AkitaError::InvalidSetup(
                "committed source encoding does not match its producer geometry and opening method"
                    .into(),
            ));
        }
        let OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } = opening_method
        else {
            continue;
        };
        if absolute_level > 1 {
            return Err(AkitaError::InvalidSetup(
                "subring coefficient packing is restricted to nonterminal levels 0 and 1".into(),
            ));
        }
        let expected = akita_challenges::SparseChallengeConfig::production_for_ring_dim(
            challenge_subring_dimension,
        )
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "coefficient-packing challenge subring is not in the production ladder".into(),
            )
        })?;
        let matrix = group.params.inner_commit_matrix_params();
        if !matches!(matrix.security_route(), InnerCommitSecurityRoute::Linf(_)) {
            return Err(AkitaError::InvalidSetup(
                "coefficient packing requires the L-infinity A security route".into(),
            ));
        }
        if group.params.fold_challenge_config() != expected {
            return Err(AkitaError::InvalidSetup(
                "coefficient packing requires its audited production challenge family".into(),
            ));
        }
        crate::SubringCoefficientPackingGeometry::try_new(
            extension_degree,
            matrix.ring_dimension(),
            challenge_subring_dimension,
        )?;
    }
    Ok(())
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
    let role_dims = predecessor.role_dims();
    let shared_d = role_dims.d_d();
    let mut relation_coefficient_block_len = role_dims.common_relation_coeff_count();
    if let OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension,
    } = predecessor.opening_method
    {
        relation_coefficient_block_len =
            relation_coefficient_block_len.min(challenge_subring_dimension);
    }
    for group in predecessor.precommitted_group_iter() {
        let group_dims = group.role_dims(shared_d);
        relation_coefficient_block_len =
            relation_coefficient_block_len.min(group_dims.common_relation_coeff_count());
        if let OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } = group.opening.opening_method
        {
            relation_coefficient_block_len =
                relation_coefficient_block_len.min(challenge_subring_dimension);
        }
    }
    let geometry = RelationAddressGeometry::new_with_coefficient_block(
        role_dims,
        relation_coefficient_block_len,
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
    /// Natural (unpadded) setup length at the first direct edge for setup-first
    /// schedule selection.
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
