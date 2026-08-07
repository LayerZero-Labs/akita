//! Planner-free runtime schedule expansion support.

use akita_field::AkitaError;
use akita_types::{
    ChunkedWitnessCfg, CommitmentRingDims, CommittedGroupParams, DecompositionParams, FoldSchedule,
    FoldScheduleEstimate, PlannedFoldSchedule, PolynomialGroupLayout, RecursiveFoldParams,
    RecursiveFoldStep, RingRole, RootFinalGroupParams, RootFoldParams, RootFoldStep,
    RootPrecommittedGroupParams, SisModulusProfileId, SisSecurityPolicyId, TerminalFoldParams,
    TerminalFoldStep, TerminalResponseShape, WitnessLayout, WitnessPartition,
    DEFAULT_SIS_SECURITY_POLICY,
};

/// Quantities materialized and checked by the current bounded planner cost model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlannerCostModelId {
    /// Exact protocol payload plus setup-envelope accounting.
    ExactPayloadAndSetupEnvelope,
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

/// Catalog-bound ring-dimension schedule policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingDimensionScheduleMode {
    /// Use one uniform A/B/D dimension from root through terminal.
    UniformDimension { ring_dimension: usize },
    /// Search A over a bounded prefix, derive B/D by minimum rank, then use a uniform suffix.
    AdaptiveDimension {
        num_search_levels: usize,
        uniform_suffix_dimension: usize,
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
    pub selection_policy: SelectionPolicyId,
    /// Optional host admission budget for materialized setup field elements.
    /// `None` leaves the deterministic public stream uncapped by protocol policy.
    pub setup_field_budget: Option<usize>,
    pub min_offloaded_witness_contraction: usize,
    /// Ring dimension used when the planner is restricted to a uniform domain.
    pub uniform_ring_dimension: usize,
    /// A-matrix ring dimension used to commit offloaded setup prefixes.
    pub setup_prefix_inner_ring_dimension: usize,
    /// Uniform or bounded-adaptive ring-dimension schedule policy.
    pub ring_dimension_schedule_mode: RingDimensionScheduleMode,
    pub decomposition: DecompositionParams,
    pub sis_modulus_profile: SisModulusProfileId,
    pub sis_security_policy: SisSecurityPolicyId,
    pub sis_table_digest: akita_types::SisTableDigest,
    pub ring_subfield_norm_bound: u32,
    pub claim_ext_degree: usize,
    pub chal_ext_degree: usize,
    pub basis_range: (u32, u32),
    pub witness_chunk: ChunkedWitnessCfg,
    pub recursive_setup_planning: bool,
}

/// Preferred public name for runtime callers.
pub type RuntimeSchedulePolicy = PlannerPolicy;

impl PlannerPolicy {
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

    /// Direct-only counterpart used when scalar schedules are cataloged under
    /// the non-recursive family identity. It deliberately restores the
    /// proof-payload objective: callers crossing from a grouped/recursive
    /// adapter into the uniform scalar planner must not reuse a mixed-domain
    /// setup-first policy.
    pub fn direct_only(self) -> Self {
        Self {
            recursive_setup_planning: false,
            selection_policy: SelectionPolicyId::for_policy(
                false,
                self.ring_dimension_schedule_mode,
            ),
            ..self
        }
    }

    /// Number of chunks emitted by fold level `fold_level`.
    pub fn chunks_at_level(&self, fold_level: usize) -> usize {
        let mc = self.witness_chunk;
        if mc.uses_multi_chunk() && fold_level < mc.num_activated_levels {
            mc.num_chunks
        } else {
            1
        }
    }

    /// Per-level witness chunk metadata.
    pub fn witness_chunk_for_level(&self, fold_level: usize) -> ChunkedWitnessCfg {
        let num_chunks = self.chunks_at_level(fold_level);
        if num_chunks > 1 {
            ChunkedWitnessCfg {
                num_chunks,
                num_activated_levels: self.witness_chunk.num_activated_levels,
            }
        } else {
            ChunkedWitnessCfg::default()
        }
    }

    /// Inclusive `(min, max)` `log_basis` values to evaluate at an absolute fold
    /// level.
    ///
    /// The root fold is fixed to the configured minimum basis. Deeper folds can
    /// search the full configured range, while the suffix DP separately enforces
    /// non-decreasing bases across adjacent folds.
    pub fn log_basis_search_range_at_level(&self, level: usize) -> (u32, u32) {
        let (configured_min, max) = self.basis_range;
        if level == 0 {
            return (configured_min, configured_min);
        }
        (configured_min, max)
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
            uniform_suffix_dimension,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            if num_search_levels != ADAPTIVE_SEARCH_LEVELS {
                return Err(AkitaError::InvalidSetup(format!(
                    "adaptive search currently requires exactly {ADAPTIVE_SEARCH_LEVELS} levels, got {num_search_levels}"
                )));
            }
            for (role, dimensions) in [
                (RingRole::Inner, potential_a_dimensions),
                (RingRole::Outer, potential_b_dimensions),
                (RingRole::Opening, potential_d_dimensions),
            ] {
                validate_dimension_list(policy, role, dimensions)?;
                if !dimensions.contains(&uniform_suffix_dimension) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "adaptive {} domain must contain suffix D{uniform_suffix_dimension}",
                        role_name(role)
                    )));
                }
                if dimensions.iter().any(|&d| d < uniform_suffix_dimension) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "adaptive {} dimensions must be at least suffix D{uniform_suffix_dimension}",
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
    pub params: CommittedGroupParams,
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
    let schedule = FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    commitment: root.params.clone(),
                },
                precommitted_groups: root
                    .params
                    .precommitted_groups
                    .iter()
                    .cloned()
                    .map(|commitment| RootPrecommittedGroupParams {
                        descriptor: commitment.layout,
                        commitment,
                    })
                    .collect(),
                open_commit_matrix: root.params.open_commit_matrix,
                sparse_challenge_config: root.params.fold_challenge_config,
                witness_partition: witness_partition(root.params.witness_chunk.num_chunks),
            },
            input_witness_len: root.input_witness_len,
            output_witness_len: root.output_witness_len,
        },
        recursive_folds: folds
            .into_iter()
            .map(|fold| RecursiveFoldStep {
                params: RecursiveFoldParams {
                    open_commit_matrix: fold.params.open_commit_matrix,
                    sparse_challenge_config: fold.params.fold_challenge_config,
                    incoming_setup_prefix: fold.params.setup_prefix.clone(),
                    witness_partition: witness_partition(fold.params.witness_chunk.num_chunks),
                    witness: fold.params,
                },
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
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
    if !params.compression_sources_supported()? {
        return Ok(None);
    }
    let opening_batch =
        params.opening_layout_for_final_group(PolynomialGroupLayout::new(0, final_num_polys))?;
    let layout = WitnessLayout::new(
        params,
        &opening_batch,
        num_chunks,
        akita_types::sis::compute_num_digits_field_width(field_bits, params.log_basis_open),
    )?;
    Ok(Some(layout.live_coeff_len()))
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
            selection_policy: SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 3,
            // This remains the uniform preset candidate. It is deliberately
            // smaller than D512 to prove that it is not an adaptive carrier.
            uniform_ring_dimension: 64,
            setup_prefix_inner_ring_dimension: 64,
            ring_dimension_schedule_mode: RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels: 2,
                uniform_suffix_dimension: 64,
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
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (3, 6),
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
    fn adaptive_dimensions_still_require_role_specific_dispatch_support() {
        const UNSUPPORTED_B_DIMENSIONS: &[usize] = &[64, 512];
        let mut policy = adaptive_policy();
        policy.ring_dimension_schedule_mode = RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels: 2,
            uniform_suffix_dimension: 64,
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
                uniform_suffix_dimension: 64,
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
