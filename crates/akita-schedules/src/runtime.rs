//! Planner-free runtime schedule expansion support.

use akita_field::AkitaError;
use akita_types::{
    ChunkedWitnessCfg, CommitmentRingDims, CommittedGroupParams, DecompositionParams, FoldSchedule,
    FoldScheduleEstimate, PlannedFoldSchedule, PolynomialGroupLayout, RecursiveFoldParams,
    RecursiveFoldStep, RingRole, RootFinalGroupParams, RootFoldParams, RootFoldStep,
    RootPrecommittedGroupParams, SisModulusProfileId, SisSecurityPolicyId, TerminalFoldParams,
    TerminalFoldStep, TerminalResponseShape, WitnessLayout, WitnessPartition,
    DEFAULT_SIS_SECURITY_POLICY, MAX_I16_LOG_BASIS, MAX_I8_LOG_BASIS,
};
use std::sync::Arc;

/// Quantities materialized and checked by the current bounded planner cost model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerCostModelId {
    /// Exact protocol payload plus setup-envelope accounting.
    ExactPayloadAndSetupEnvelope,
}

/// Offline response-energy model used to admit selective L2 candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectiveL2ResponseModelId {
    /// Do not derive modeled L2 caps.
    Disabled,
    /// Typed Z/E/T/R/compression moment propagation with extension tensor
    /// packing and a Markov-backed grinding cap.
    TypedProtocolMomentsV1,
}

impl SelectiveL2ResponseModelId {
    /// Stable identity tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::Disabled => 0,
            Self::TypedProtocolMomentsV1 => 1,
        }
    }

    /// Stable identity name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Disabled => "Disabled",
            Self::TypedProtocolMomentsV1 => "TypedProtocolMomentsV1",
        }
    }
}

impl PlannerCostModelId {
    /// Stable identity tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::ExactPayloadAndSetupEnvelope => 1,
        }
    }

    /// Stable identity name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::ExactPayloadAndSetupEnvelope => "ExactPayloadAndSetupEnvelope",
        }
    }
}

/// Deterministic schedule-selection policy bound into generated catalogs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionPolicyId {
    /// Pick proof bytes, then physical setup fields, then canonical descriptor.
    MinEstimatedProofPayload,
    /// Pick physical setup fields, then proof bytes, then canonical descriptor.
    MinSetupMatrixFieldElementsThenProofPayload,
    /// Pick first direct setup, proof bytes, total setup, then descriptor.
    MinFirstDirectSetupThenPayload,
}

impl SelectionPolicyId {
    /// Canonical selection objective for one schedule policy shape.
    pub fn for_policy(
        recursive_setup_planning: bool,
        ring_dimension_schedule_mode: RingDimensionScheduleMode,
    ) -> Self {
        if recursive_setup_planning {
            Self::MinFirstDirectSetupThenPayload
        } else if matches!(
            ring_dimension_schedule_mode,
            RingDimensionScheduleMode::AdaptiveDimension { .. }
        ) {
            Self::MinSetupMatrixFieldElementsThenProofPayload
        } else {
            Self::MinEstimatedProofPayload
        }
    }

    /// Stable identity tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::MinEstimatedProofPayload => 1,
            Self::MinFirstDirectSetupThenPayload => 2,
            Self::MinSetupMatrixFieldElementsThenProofPayload => 3,
        }
    }

    /// Stable identity name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::MinEstimatedProofPayload => "MinEstimatedProofPayload",
            Self::MinSetupMatrixFieldElementsThenProofPayload => {
                "MinSetupMatrixFieldElementsThenProofPayload"
            }
            Self::MinFirstDirectSetupThenPayload => "MinFirstDirectSetupThenPayload",
        }
    }
}

/// Catalog-bound recursive split traversal policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecursiveSplitSearchPolicy {
    /// Traverse every feasible recursive witness split.
    Exhaustive,
    /// Search the two extremes and a fixed radius-two balance window for
    /// states above twelve reduced variables.
    BoundedBalancedExtremesV1,
}

impl RecursiveSplitSearchPolicy {
    pub const fn tag(self) -> u32 {
        match self {
            Self::Exhaustive => 1,
            Self::BoundedBalancedExtremesV1 => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Exhaustive => "Exhaustive",
            Self::BoundedBalancedExtremesV1 => "BoundedBalancedExtremesV1",
        }
    }
}

/// Catalog-bound ring-dimension schedule policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingDimensionScheduleMode {
    /// Use one uniform A/B/D dimension from root through terminal.
    UniformDimension { ring_dimension: usize },
    /// Search exact A/B/D tuples over a bounded prefix, then use a monotone
    /// sequence of uniform dimensions from the catalog-bound suffix domain.
    AdaptiveDimension {
        num_search_levels: usize,
        suffix_dimensions: &'static [usize],
        potential_a_dimensions: &'static [usize],
        potential_b_dimensions: &'static [usize],
        potential_d_dimensions: &'static [usize],
    },
}

/// Number of leading fold levels covered by the audited adaptive search.
pub const ADAPTIVE_SEARCH_LEVELS: usize = 2;

impl RingDimensionScheduleMode {
    #[must_use]
    pub const fn uniform_dimensions(self) -> Option<CommitmentRingDims> {
        match self {
            Self::UniformDimension { ring_dimension } => {
                Some(CommitmentRingDims::uniform(ring_dimension))
            }
            Self::AdaptiveDimension { .. } => None,
        }
    }

    #[must_use]
    pub const fn potential_a_dimensions(self) -> &'static [usize] {
        match self {
            Self::UniformDimension { .. } => &[],
            Self::AdaptiveDimension {
                potential_a_dimensions,
                ..
            } => potential_a_dimensions,
        }
    }
}

/// Runtime schedule validation policy.
///
/// The compatibility name stays `PlannerPolicy` during the migration because
/// generated catalog identities already embed these fields. Runtime code must
/// only use this as validation policy; search remains in `akita-planner`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannerPolicy {
    pub cost_model: PlannerCostModelId,
    pub selective_l2_response_model: SelectiveL2ResponseModelId,
    pub selection_policy: SelectionPolicyId,
    pub recursive_split_search_policy: RecursiveSplitSearchPolicy,
    /// Optional host admission budget for materialized setup field elements.
    /// `None` leaves the deterministic public stream uncapped by protocol policy.
    pub setup_field_budget: Option<usize>,
    pub min_offloaded_witness_contraction: usize,
    /// Ring dimension used when the planner is restricted to a uniform domain.
    pub uniform_ring_dimension: usize,
    /// Default/ceiling A-matrix dimension for setup-prefix catalog identity.
    /// Adaptive schedules derive each actual prefix A dimension from its
    /// consuming fold.
    pub setup_prefix_inner_ring_dimension: usize,
    /// Uniform or bounded-adaptive ring-dimension schedule policy.
    pub ring_dimension_schedule_mode: RingDimensionScheduleMode,
    pub decomposition: DecompositionParams,
    pub sis_modulus_profile: SisModulusProfileId,
    pub sis_security_policy: SisSecurityPolicyId,
    pub sis_table_digest: akita_types::SisTableDigest,
    pub sis_l2_table_digest: akita_types::SisL2TableDigest,
    pub claim_ext_degree: usize,
    pub chal_ext_degree: usize,
    /// Inclusive A/source decomposition basis domain at every level.
    pub inner_basis_range: (u32, u32),
    /// Inclusive B/D opening and folded-response basis domain.
    pub opening_basis_range: (u32, u32),
    pub witness_chunk: ChunkedWitnessCfg,
    pub recursive_setup_planning: bool,
}

/// Preferred public name for runtime callers.
pub type RuntimeSchedulePolicy = PlannerPolicy;

impl PlannerPolicy {
    /// Number of physical witness chunks active at one fold level.
    pub const fn chunks_at_level(&self, fold_level: usize) -> usize {
        if self.witness_chunk.uses_multi_chunk()
            && fold_level < self.witness_chunk.num_activated_levels
        {
            self.witness_chunk.num_chunks
        } else {
            1
        }
    }

    /// Whether this family opts into the typed suffix response model.
    pub fn selective_l2_response_model_enabled(&self) -> bool {
        self.selective_l2_response_model == SelectiveL2ResponseModelId::TypedProtocolMomentsV1
    }

    /// Whether a candidate fits the optional host setup budget.
    pub fn admits_setup_field_elements(&self, num_field_elements: usize) -> bool {
        self.setup_field_budget
            .is_none_or(|budget| num_field_elements <= budget)
    }

    /// Validate extension-field geometry and return the challenge-field width.
    ///
    /// The checked conversion and multiplication keep malformed custom policy
    /// values from truncating or overflowing in verifier-reachable pricing.
    pub fn challenge_field_bits(&self) -> Result<u32, AkitaError> {
        for (name, degree) in [
            ("claim extension degree", self.claim_ext_degree),
            ("challenge extension degree", self.chal_ext_degree),
        ] {
            if degree == 0 || !degree.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(format!(
                    "{name} must be a nonzero power of two, got {degree}"
                )));
            }
        }
        let challenge_degree = u32::try_from(self.chal_ext_degree).map_err(|_| {
            AkitaError::InvalidSetup(format!(
                "challenge extension degree {} exceeds u32",
                self.chal_ext_degree
            ))
        })?;
        self.decomposition
            .field_bits()
            .checked_mul(challenge_degree)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("challenge field bit width overflow".to_string())
            })
    }
}

/// Suffix-DP depth cap shared by planner search and runtime policy validation.
pub const MAX_RECURSION_DEPTH: usize = 12;
/// Validate runtime policy values used by schedule expansion and validation.
pub fn validate_policy(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    policy.challenge_field_bits()?;
    if !akita_types::sis::SUPPORTED_SIS_SECURITY_POLICIES.contains(&policy.sis_security_policy) {
        return Err(AkitaError::InvalidSetup(format!(
            "unsupported SIS security policy {:?}",
            policy.sis_security_policy
        )));
    }
    validate_ring_dimension_schedule_mode(policy)?;
    let expected_selection_policy = SelectionPolicyId::for_policy(
        policy.recursive_setup_planning,
        policy.ring_dimension_schedule_mode,
    );
    if policy.selection_policy != expected_selection_policy {
        return Err(AkitaError::InvalidSetup(
            "schedule selection policy disagrees with recursive setup capability".to_string(),
        ));
    }
    if policy.setup_field_budget == Some(0) {
        return Err(AkitaError::InvalidSetup(
            "explicit setup field budget must be positive".to_string(),
        ));
    }
    if !policy.setup_prefix_inner_ring_dimension.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "schedule setup-prefix inner ring dimension must be a nonzero power of two".to_string(),
        ));
    }
    if policy.min_offloaded_witness_contraction == 0 {
        return Err(AkitaError::InvalidSetup(
            "minimum offloaded witness contraction must be positive".to_string(),
        ));
    }
    if policy.selective_l2_response_model_enabled()
        && policy.sis_l2_table_digest != akita_types::SisL2TableDigest::CURRENT
    {
        return Err(AkitaError::InvalidSetup(
            "selective L2 planning requires the current audited Euclidean table".into(),
        ));
    }
    for (label, (min, max), supported_max) in [
        ("opening", policy.opening_basis_range, MAX_I8_LOG_BASIS),
        ("inner", policy.inner_basis_range, MAX_I16_LOG_BASIS),
    ] {
        if min == 0 || min > max || max > supported_max {
            return Err(AkitaError::InvalidSetup(format!(
                "{label} basis range [{min}, {max}] is outside 1..={supported_max}"
            )));
        }
    }
    policy.witness_chunk.validate()?;
    if policy.witness_chunk.num_activated_levels > MAX_RECURSION_DEPTH {
        return Err(AkitaError::InvalidSetup(format!(
            "num_activated_levels={} exceeds the schedule recursion cap {MAX_RECURSION_DEPTH}",
            policy.witness_chunk.num_activated_levels
        )));
    }
    Ok(())
}

fn validate_ring_dimension_schedule_mode(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    match policy.ring_dimension_schedule_mode {
        RingDimensionScheduleMode::UniformDimension { ring_dimension } => {
            if ring_dimension != policy.uniform_ring_dimension {
                return Err(AkitaError::InvalidSetup(format!(
                    "uniform schedule D{ring_dimension} must equal configured D{}",
                    policy.uniform_ring_dimension
                )));
            }
            for role in [RingRole::Inner, RingRole::Outer, RingRole::Opening] {
                validate_scheduled_dimension(policy, role, ring_dimension)?;
            }
        }
        RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            if num_search_levels != ADAPTIVE_SEARCH_LEVELS {
                return Err(AkitaError::InvalidSetup(format!(
                    "adaptive search currently requires exactly {ADAPTIVE_SEARCH_LEVELS} levels, got {num_search_levels}"
                )));
            }
            validate_dimension_list(policy, RingRole::Inner, suffix_dimensions)?;
            for (role, dimensions) in [
                (RingRole::Inner, potential_a_dimensions),
                (RingRole::Outer, potential_b_dimensions),
                (RingRole::Opening, potential_d_dimensions),
            ] {
                validate_dimension_list(policy, role, dimensions)?;
                for &suffix_dimension in suffix_dimensions {
                    if !dimensions.contains(&suffix_dimension) {
                        return Err(AkitaError::InvalidSetup(format!(
                            "adaptive {} domain must contain suffix D{suffix_dimension}",
                            role_name(role)
                        )));
                    }
                }
                let minimum_suffix_dimension = suffix_dimensions[0];
                if dimensions.iter().any(|&d| d < minimum_suffix_dimension) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "adaptive {} dimensions must be at least minimum suffix D{minimum_suffix_dimension}",
                        role_name(role)
                    )));
                }
            }
        }
    }
    Ok(())
}

fn role_name(role: RingRole) -> &'static str {
    match role {
        RingRole::Inner => "A",
        RingRole::Outer => "B",
        RingRole::Opening => "D",
    }
}

fn sis_role(role: RingRole) -> akita_types::SisMatrixRole {
    match role {
        RingRole::Inner => akita_types::SisMatrixRole::Inner,
        RingRole::Outer => akita_types::SisMatrixRole::Outer,
        RingRole::Opening => akita_types::SisMatrixRole::Open,
    }
}

fn validate_scheduled_dimension(
    policy: &PlannerPolicy,
    role: RingRole,
    dimension: usize,
) -> Result<(), AkitaError> {
    let tier = akita_types::protocol_dispatch_tier_for_sis_profile(policy.sis_modulus_profile);
    if !akita_types::dispatch::role_dim_supported_for_tier(tier, role, dimension) {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled {} dimension D{dimension} is unsupported by the {:?} protocol dispatch",
            role_name(role),
            policy.sis_modulus_profile
        )));
    }
    let dimension_u32 = u32::try_from(dimension).map_err(|_| {
        AkitaError::InvalidSetup(format!(
            "scheduled {} dimension D{dimension} exceeds u32",
            role_name(role)
        ))
    })?;
    if !akita_types::sis::sis_role_dimension_supported(
        sis_role(role),
        policy.sis_modulus_profile,
        dimension_u32,
    ) {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled {} dimension D{dimension} has no SIS security-table coverage for {:?}",
            role_name(role),
            policy.sis_modulus_profile
        )));
    }
    if role == RingRole::Inner && !akita_types::SUPPORTED_CHALLENGE_RING_DIMS.contains(&dimension) {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled A dimension D{dimension} has no production fold-challenge configuration"
        )));
    }
    Ok(())
}

fn validate_dimension_list(
    policy: &PlannerPolicy,
    role: RingRole,
    dimensions: &[usize],
) -> Result<(), AkitaError> {
    if dimensions.is_empty() {
        return Err(AkitaError::InvalidSetup(format!(
            "adaptive {} domain must be nonempty",
            role_name(role)
        )));
    }
    for (index, &dimension) in dimensions.iter().enumerate() {
        validate_scheduled_dimension(policy, role, dimension)?;
        if index > 0 && dimensions[index - 1] >= dimension {
            return Err(AkitaError::InvalidSetup(format!(
                "adaptive {} domain must be strictly sorted and duplicate-free",
                role_name(role)
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
/// One fully priced non-terminal fold awaiting schedule materialization.
pub struct CandidateFoldStep {
    pub params: Arc<CommittedGroupParams>,
    pub input_witness_len: usize,
    pub output_witness_len: usize,
    pub estimated_direct_payload_bytes: usize,
    pub estimated_stage3_payload_bytes: usize,
}

#[derive(Clone, Debug)]
/// Fully priced terminal response awaiting schedule materialization.
pub struct CandidateTerminalResponse {
    pub params: akita_types::TerminalCommittedGroupParams,
    pub sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub input_witness_len: usize,
    pub estimated_direct_payload_bytes: usize,
    pub response_shape: TerminalResponseShape,
    pub estimated_payload_bytes: usize,
}

/// Exact Stage-3 payload induced when `successor` consumes a setup prefix.
pub fn stage3_payload_bytes_for_successor(
    policy: &PlannerPolicy,
    successor: Option<&CommittedGroupParams>,
) -> Result<usize, AkitaError> {
    let Some(prefix) = successor.and_then(|params| params.setup_prefix.as_ref()) else {
        return Ok(usize::default());
    };
    let n_prefix = prefix.n_prefix()?;
    if prefix.d_setup() == 0 || !n_prefix.is_multiple_of(prefix.d_setup()) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix field length does not align with its ring dimension".to_string(),
        ));
    }
    let challenge_field_bits = policy.challenge_field_bits()?;
    Ok(akita_types::proof_size::stage3_setup_product_bytes(
        challenge_field_bits,
        prefix.d_setup(),
        n_prefix / prefix.d_setup(),
    ))
}

/// Materialize and validate the schedule shared by offline search and generated replay.
///
/// `cached_num_setup_field_elements` is the exact shared flat setup capacity.
pub fn materialize_candidate_schedule(
    cached_total: usize,
    cached_num_setup_field_elements: usize,
    first_direct_setup_field_len: Option<usize>,
    mut folds: Vec<CandidateFoldStep>,
    terminal_response: CandidateTerminalResponse,
) -> Result<PlannedFoldSchedule, AkitaError> {
    if folds.is_empty() {
        return Err(AkitaError::UnsupportedSchedule(
            "a fold schedule requires root and terminal folds".to_string(),
        ));
    }
    let root = folds.remove(0);
    let mut estimate = FoldScheduleEstimate {
        estimated_root_direct_payload_bytes: root.estimated_direct_payload_bytes,
        estimated_root_stage3_payload_bytes: root.estimated_stage3_payload_bytes,
        estimated_recursive_direct_payload_bytes: folds
            .iter()
            .map(|fold| fold.estimated_direct_payload_bytes)
            .collect(),
        estimated_recursive_stage3_payload_bytes: folds
            .iter()
            .map(|fold| fold.estimated_stage3_payload_bytes)
            .collect(),
        estimated_terminal_direct_payload_bytes: terminal_response
            .estimated_direct_payload_bytes
            .checked_add(terminal_response.estimated_payload_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal estimate overflow".to_string()))?,
        estimated_terminal_response_payload_bytes: terminal_response.estimated_payload_bytes,
        estimated_num_setup_field_elements: cached_num_setup_field_elements,
        first_direct_setup_field_len,
        selected_offload_edges: 0,
    };
    let recomputed = estimate.estimated_proof_payload_bytes()?;
    if recomputed != cached_total {
        return Err(AkitaError::InvalidSetup(format!(
            "cached schedule cost {cached_total} disagrees with materialized estimate {recomputed}"
        )));
    }
    let root_params = Arc::unwrap_or_clone(root.params);
    let schedule = FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    commitment: root_params.clone(),
                },
                precommitted_groups: root_params
                    .precommitted_groups
                    .iter()
                    .cloned()
                    .map(|commitment| RootPrecommittedGroupParams {
                        descriptor: commitment.layout,
                        commitment,
                    })
                    .collect(),
                open_commit_matrix: root_params.open_commit_matrix,
                sparse_challenge_config: root_params.fold_challenge_config,
                witness_partition: witness_partition(root_params.witness_chunk.num_chunks),
            },
            input_witness_len: root.input_witness_len,
            output_witness_len: root.output_witness_len,
        },
        recursive_folds: folds
            .into_iter()
            .map(|fold| {
                let params = Arc::unwrap_or_clone(fold.params);
                RecursiveFoldStep {
                    params: RecursiveFoldParams {
                        open_commit_matrix: params.open_commit_matrix,
                        sparse_challenge_config: params.fold_challenge_config,
                        incoming_setup_prefix: params.setup_prefix.clone(),
                        witness_partition: witness_partition(params.witness_chunk.num_chunks),
                        witness: params,
                    },
                    input_witness_len: fold.input_witness_len,
                    output_witness_len: fold.output_witness_len,
                }
            })
            .collect(),
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                sparse_challenge_config: terminal_response.sparse_challenge_config,
                witness: terminal_response.params,
                response_shape: terminal_response.response_shape,
            },
            input_witness_len: terminal_response.input_witness_len,
        },
    };
    schedule.validate_structure()?;
    let recomputed_num_setup_field_elements =
        akita_types::setup_matrix_capacity_for_schedule(&schedule)?.num_field_elements;
    if recomputed_num_setup_field_elements != cached_num_setup_field_elements {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup capacity {cached_num_setup_field_elements} field elements disagrees with materialized capacity {recomputed_num_setup_field_elements}"
        )));
    }
    estimate.selected_offload_edges = schedule
        .recursive_folds
        .iter()
        .filter(|fold| fold.params.incoming_setup_prefix.is_some())
        .count();
    Ok(PlannedFoldSchedule { schedule, estimate })
}

fn witness_partition(num_chunks: usize) -> WitnessPartition {
    if num_chunks == 1 {
        WitnessPartition::Single
    } else {
        WitnessPartition::Distributed { num_chunks }
    }
}

#[allow(clippy::too_many_arguments)]
/// Count ring elements in one grouped witness segment with checked arithmetic.
pub fn grouped_segment_rings(
    num_polys: usize,
    num_live_blocks: usize,
    num_chunks: usize,
    num_positions_per_block: usize,
    n_a: usize,
    num_digits_inner: usize,
    num_digits_outer: usize,
    num_digits_open: usize,
    num_digits_fold: usize,
) -> Result<usize, AkitaError> {
    let e_hat = num_polys
        .checked_mul(num_live_blocks)
        .and_then(|n| n.checked_mul(num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("group e-hat witness overflow".to_string()))?;
    let t_hat = num_polys
        .checked_mul(num_live_blocks)
        .and_then(|n| n.checked_mul(n_a))
        .and_then(|n| n.checked_mul(num_digits_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("group t-hat witness overflow".to_string()))?;
    let z_hat = num_positions_per_block
        .checked_mul(num_digits_inner)
        .and_then(|n| n.checked_mul(num_digits_fold))
        .and_then(|n| n.checked_mul(num_chunks))
        .ok_or_else(|| AkitaError::InvalidSetup("group z-hat witness overflow".to_string()))?;

    e_hat
        .checked_add(t_hat)
        .and_then(|n| n.checked_add(z_hat))
        .ok_or_else(|| AkitaError::InvalidSetup("group witness overflow".to_string()))
}

/// Derive the canonical next-witness field length for a scalar planner level.
pub fn planned_next_witness_len(
    field_bits: u32,
    extension_degree: usize,
    params: &CommittedGroupParams,
    final_num_polys: usize,
    num_chunks: usize,
) -> Result<Option<usize>, AkitaError> {
    if !params.precommitted_groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root witness sizing must use CommittedGroupParams::output_witness_len"
                .to_string(),
        ));
    }
    let opening_batch =
        params.opening_layout_for_final_group(PolynomialGroupLayout::new(0, final_num_polys))?;
    let quotient_depth =
        akita_types::sis::compute_num_digits_field_width(field_bits, params.log_basis_open);
    if !params.compression_sources_supported()? {
        return Ok(None);
    }
    let relation_geometry =
        akita_types::RelationWitnessGeometry::for_level(params, &opening_batch, extension_degree)?;
    if params.setup_prefix.is_none() {
        return WitnessLayout::try_scalar_live_coeff_len(
            params,
            &opening_batch,
            &relation_geometry,
            num_chunks,
            quotient_depth,
        );
    }
    Ok(Some(
        WitnessLayout::new(
            params,
            &opening_batch,
            &relation_geometry,
            num_chunks,
            quotient_depth,
        )?
        .live_coeff_len(),
    ))
}

/// Convenience policy used by config adapters.
pub fn default_sis_security_policy() -> SisSecurityPolicyId {
    DEFAULT_SIS_SECURITY_POLICY
}

#[cfg(test)]
mod tests {
    use super::*;

    const A_DIMENSIONS_WITHOUT_GLOBAL_CARRIER: &[usize] = &[64, 512];
    const SUFFIX_DIMENSIONS: &[usize] = &[64];

    fn adaptive_policy() -> PlannerPolicy {
        PlannerPolicy {
            cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
            selective_l2_response_model: SelectiveL2ResponseModelId::TypedProtocolMomentsV1,
            selection_policy: SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload,
            recursive_split_search_policy: crate::RecursiveSplitSearchPolicy::Exhaustive,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 3,
            // This remains the uniform preset candidate. It is deliberately
            // smaller than D512 to prove that it is not an adaptive carrier.
            uniform_ring_dimension: 64,
            setup_prefix_inner_ring_dimension: 64,
            ring_dimension_schedule_mode: RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels: 2,
                suffix_dimensions: &[64],
                potential_a_dimensions: A_DIMENSIONS_WITHOUT_GLOBAL_CARRIER,
                potential_b_dimensions: SUFFIX_DIMENSIONS,
                potential_d_dimensions: SUFFIX_DIMENSIONS,
            },
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(128),
            },
            sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
            sis_security_policy: DEFAULT_SIS_SECURITY_POLICY,
            sis_table_digest: akita_types::SisTableDigest::CURRENT,
            sis_l2_table_digest: akita_types::SisL2TableDigest::CURRENT,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            inner_basis_range: (3, 16),
            opening_basis_range: (3, 6),
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: false,
        }
    }

    #[test]
    fn adaptive_dimensions_do_not_require_a_global_carrier() {
        let mut policy = adaptive_policy();
        policy.uniform_ring_dimension = 3;
        validate_policy(&policy)
            .expect("individually supported D512 A must not depend on the uniform-only field");
    }

    #[test]
    fn typed_response_model_requires_current_l2_table_identity() {
        let mut policy = adaptive_policy();
        policy.sis_l2_table_digest = akita_types::SisL2TableDigest([0; 32]);
        let error = validate_policy(&policy).expect_err("stale L2 table identity");
        assert!(error
            .to_string()
            .contains("selective L2 planning requires the current audited Euclidean table"));
    }

    #[test]
    fn adaptive_dimensions_still_require_role_specific_dispatch_support() {
        const UNSUPPORTED_B_DIMENSIONS: &[usize] = &[64, 512];
        let mut policy = adaptive_policy();
        policy.ring_dimension_schedule_mode = RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels: 2,
            suffix_dimensions: &[64],
            potential_a_dimensions: A_DIMENSIONS_WITHOUT_GLOBAL_CARRIER,
            potential_b_dimensions: UNSUPPORTED_B_DIMENSIONS,
            potential_d_dimensions: SUFFIX_DIMENSIONS,
        };

        let error = validate_policy(&policy).expect_err("fp128 B has no D512 dispatch");
        assert!(error.to_string().contains("scheduled B dimension D512"));
    }

    #[test]
    fn adaptive_depth_is_limited_to_the_audited_l0_l1_cutover() {
        for num_search_levels in [1, 3] {
            let mut policy = adaptive_policy();
            policy.ring_dimension_schedule_mode = RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels,
                suffix_dimensions: &[64],
                potential_a_dimensions: A_DIMENSIONS_WITHOUT_GLOBAL_CARRIER,
                potential_b_dimensions: SUFFIX_DIMENSIONS,
                potential_d_dimensions: SUFFIX_DIMENSIONS,
            };

            let error = validate_policy(&policy).expect_err("unsupported adaptive depth");
            assert!(error
                .to_string()
                .contains("adaptive search currently requires exactly 2 levels"));
        }
    }
}
