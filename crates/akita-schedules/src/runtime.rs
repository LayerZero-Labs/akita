//! Planner-free runtime schedule expansion support.

use akita_error::AkitaError;
use akita_types::{
    ChunkedWitnessCfg, CommitmentRingDims, CommittedGroupParams, DecompositionParams, FoldParams,
    FoldSchedule, FoldScheduleEstimate, FoldSuccessor, OpeningClaimsLayout, PlannedFoldSchedule,
    PolynomialGroupLayout, RingRole, SisModulusProfileId, SisSecurityPolicyId, TerminalFoldParams,
    TerminalResponseShape, WitnessLayout, DEFAULT_SIS_SECURITY_POLICY, MAX_I16_LOG_BASIS,
    MAX_I8_LOG_BASIS,
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

/// Deterministic schedule-selection policy bound into trusted catalog artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionPolicyId {
    /// Pick proof bytes, physical setup fields, root output witness, then descriptor.
    MinEstimatedProofPayloadV2,
    /// Pick first direct setup, proof bytes, total setup, root output witness,
    /// then descriptor.
    MinFirstDirectSetupThenPayloadV2,
    /// Pick power-of-two setup-envelope capacity, first direct setup, proof
    /// bytes, first direct output witness, then descriptor.
    MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3,
}

impl SelectionPolicyId {
    /// Canonical selection objective for one schedule policy shape.
    pub fn for_policy(
        recursive_setup_planning: bool,
        ring_dimension_schedule_mode: RingDimensionScheduleMode,
    ) -> Self {
        if recursive_setup_planning {
            Self::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
        } else if matches!(
            ring_dimension_schedule_mode,
            RingDimensionScheduleMode::AdaptiveDimension { .. }
        ) {
            Self::MinFirstDirectSetupThenPayloadV2
        } else {
            Self::MinEstimatedProofPayloadV2
        }
    }

    /// Stable identity tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::MinEstimatedProofPayloadV2 => 4,
            Self::MinFirstDirectSetupThenPayloadV2 => 5,
            Self::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3 => 6,
            // Tags 1 and 2 belong to the descriptor-only predecessors. Tag 3
            // belonged to the retired setup-envelope-first policy. Never reuse
            // an objective tag: trusted catalog admission depends on it.
        }
    }

    /// Stable identity name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::MinEstimatedProofPayloadV2 => "MinEstimatedProofPayloadV2",
            Self::MinFirstDirectSetupThenPayloadV2 => "MinFirstDirectSetupThenPayloadV2",
            Self::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3 => {
                "MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3"
            }
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

/// Catalog-bound search policy for recursive setup-offload edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecursiveSetupSearchPolicy {
    /// Consider an offloaded child at every admissible producer level.
    Exhaustive,
    /// Consider offloaded children produced by the root or its direct child.
    ///
    /// This bounds production catalog generation without changing direct-edge
    /// traversal. It is an explicit search-domain choice, not a dominance
    /// claim about deeper offload points.
    RootAndFirstChildV1,
}

impl RecursiveSetupSearchPolicy {
    pub const fn tag(self) -> u32 {
        match self {
            Self::Exhaustive => 1,
            Self::RootAndFirstChildV1 => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Exhaustive => "Exhaustive",
            Self::RootAndFirstChildV1 => "RootAndFirstChildV1",
        }
    }

    /// Whether search admits an offloaded edge produced at `level`.
    pub const fn admits_offloaded_edge_at(self, level: usize) -> bool {
        match self {
            Self::Exhaustive => true,
            Self::RootAndFirstChildV1 => level <= 1,
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
/// trusted catalog identities already embed these fields. Runtime code must
/// only use this as validation policy; search remains in `akita-planner`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannerPolicy {
    pub cost_model: PlannerCostModelId,
    pub selective_l2_response_model: SelectiveL2ResponseModelId,
    pub selection_policy: SelectionPolicyId,
    pub recursive_split_search_policy: RecursiveSplitSearchPolicy,
    pub recursive_setup_search_policy: RecursiveSetupSearchPolicy,
    /// Optional host admission budget for materialized setup field elements.
    /// `None` leaves the deterministic public stream uncapped by protocol policy.
    pub setup_field_budget: Option<usize>,
    pub min_offloaded_witness_contraction: usize,
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
    // The same check descriptor deserialization applies, so a policy reaching the
    // planner or the runtime row audit cannot carry a basis or committed-source
    // bound the digit math is unable to represent. Defense in depth: a schedule
    // resolved from an in-process policy never passes through `SetupSection`.
    policy.decomposition.validate()?;
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
            "schedule selection policy disagrees with the schedule mode".to_string(),
        ));
    }
    if policy.setup_field_budget == Some(0) {
        return Err(AkitaError::InvalidSetup(
            "explicit setup field budget must be positive".to_string(),
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
    pub params: akita_types::TerminalFoldParams,
    pub sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub input_witness_len: usize,
    pub estimated_direct_payload_bytes: usize,
    pub response_shape: TerminalResponseShape,
    pub estimated_payload_bytes: usize,
}

fn fold_schedule_from_candidate_parts(
    folds: &[CandidateFoldStep],
    terminal_response: &CandidateTerminalResponse,
) -> Result<FoldSchedule, AkitaError> {
    let (root, recursive_folds) = folds.split_first().ok_or_else(|| {
        AkitaError::UnsupportedSchedule(
            "a fold schedule requires root and terminal folds".to_string(),
        )
    })?;
    Ok(FoldSchedule {
        root: FoldParams {
            params: (*root.params).clone(),
            input_witness_len: root.input_witness_len,
            output_witness_len: root.output_witness_len,
        },
        recursive_folds: recursive_folds
            .iter()
            .map(|fold| FoldParams {
                params: (*fold.params).clone(),
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            })
            .collect(),
        terminal: TerminalFoldParams {
            fold_challenge_config: terminal_response.sparse_challenge_config,
            response_shape: terminal_response.response_shape.clone(),
            input_witness_len: terminal_response.input_witness_len,
            ..terminal_response.params.clone()
        },
    })
}

/// Price the canonical grinding plan for one complete schedule candidate.
#[doc(hidden)]
pub fn candidate_grinding_nonce_bits(
    policy: &PlannerPolicy,
    root_layout: &OpeningClaimsLayout,
    folds: &[CandidateFoldStep],
    terminal_response: &CandidateTerminalResponse,
) -> Result<usize, AkitaError> {
    let schedule = fold_schedule_from_candidate_parts(folds, terminal_response)?;
    akita_types::transcript_grinding_nonce_bits_for_planner_candidate(
        &schedule,
        root_layout,
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
    )
}

/// Exact Stage-3 payload induced when `successor` consumes a setup prefix.
pub fn stage3_payload_bytes_for_successor(
    policy: &PlannerPolicy,
    successor: FoldSuccessor<'_>,
) -> Result<usize, AkitaError> {
    let Some(prefix) = (match successor {
        FoldSuccessor::Recursive(params) => params.setup_prefix(),
        FoldSuccessor::Terminal(_) => None,
    }) else {
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

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NonterminalLevelPayloadBytes {
    pub direct: usize,
    pub stage3: usize,
    pub relation_geometry: akita_types::RelationAddressGeometry,
}

#[doc(hidden)]
pub fn nonterminal_level_payload_bytes(
    policy: &PlannerPolicy,
    params: &CommittedGroupParams,
    opening_layout: &OpeningClaimsLayout,
    successor: FoldSuccessor<'_>,
    output_witness_len: usize,
) -> Result<NonterminalLevelPayloadBytes, AkitaError> {
    let challenge_field_bits = policy.challenge_field_bits()?;
    let next_outer_payload = match successor {
        FoldSuccessor::Recursive(params) => Some(params),
        FoldSuccessor::Terminal(_) => None,
    };
    let relation_geometry = params.relation_address_geometry(
        opening_layout,
        policy.claim_ext_degree,
        successor.ring_dimension(),
        output_witness_len,
    )?;
    let direct = akita_types::level_proof_bytes(
        policy.decomposition.field_bits(),
        challenge_field_bits,
        params,
        relation_geometry,
        next_outer_payload,
    )?;
    let eor = if matches!(
        params.opening_method(),
        akita_types::OpeningMethod::EvaluationTrace
    ) {
        let opening_shape = opening_layout.aggregate_polynomial_group_layout()?;
        akita_types::extension_opening_reduction_level_bytes(
            challenge_field_bits,
            policy.claim_ext_degree,
            opening_shape,
        )?
    } else {
        0
    };
    let direct = direct
        .checked_add(eor)
        .ok_or_else(|| AkitaError::InvalidSetup("level proof payload size overflow".into()))?;
    Ok(NonterminalLevelPayloadBytes {
        direct,
        stage3: stage3_payload_bytes_for_successor(policy, successor)?,
        relation_geometry,
    })
}

/// Recompute the exact serialized proof payload for one expanded schedule.
///
/// This is the non-leaking reporting counterpart to artifact validation. It
/// consumes only the public lookup key, expanded schedule, and catalog policy;
/// no compact intermediate row or planner candidate is constructed.
pub fn expanded_schedule_proof_payload_bytes(
    key: &akita_types::AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
    policy: &PlannerPolicy,
) -> Result<usize, AkitaError> {
    let field_bits = policy.decomposition.field_bits();
    key.validate(field_bits)?;
    schedule.validate_structure()?;
    let nonterminal_levels = schedule
        .recursive_folds
        .len()
        .checked_add(1)
        .ok_or_else(|| AkitaError::InvalidSetup("fold level count overflow".into()))?;
    let mut total = 0usize;
    let mut predecessor_rounds = None;
    for level in 0..nonterminal_levels {
        let (params, output_witness_len) = if level == 0 {
            (&schedule.root.params, schedule.root.output_witness_len)
        } else {
            let fold = schedule.recursive_folds.get(level - 1).ok_or_else(|| {
                AkitaError::InvalidSetup("recursive fold index is out of range".into())
            })?;
            (&fold.params, fold.output_witness_len)
        };
        let opening_layout = match predecessor_rounds {
            None => key.opening_layout()?,
            Some(rounds) => {
                params.opening_layout_for_final_group(PolynomialGroupLayout::singleton(rounds))?
            }
        };
        let successor = schedule.recursive_folds.get(level).map_or_else(
            || FoldSuccessor::Terminal(&schedule.terminal),
            |fold| FoldSuccessor::Recursive(&fold.params),
        );
        let payload = nonterminal_level_payload_bytes(
            policy,
            params,
            &opening_layout,
            successor,
            output_witness_len,
        )?;
        predecessor_rounds = Some(payload.relation_geometry.relation_point_variable_count());
        total = total
            .checked_add(payload.direct)
            .and_then(|value| value.checked_add(payload.stage3))
            .ok_or_else(|| AkitaError::InvalidSetup("proof payload size overflow".into()))?;
    }

    let terminal_predecessor_rounds = predecessor_rounds.ok_or_else(|| {
        AkitaError::InvalidSetup("terminal proof is missing predecessor relation geometry".into())
    })?;
    let terminal_eor = akita_types::extension_opening_reduction_level_bytes(
        policy.challenge_field_bits()?,
        policy.claim_ext_degree,
        PolynomialGroupLayout::singleton(terminal_predecessor_rounds),
    )?;
    let terminal_response = akita_types::terminal_response_planner_bytes(
        field_bits,
        &schedule.terminal.response_shape,
        schedule.terminal.response_l2_sq_cap(),
    );
    let grinding_plan = akita_types::derive_transcript_grinding_plan_from_public_shape(
        schedule,
        &key.opening_layout()?,
        field_bits,
        policy.claim_ext_degree,
    )?;
    let nonce_stream_bytes = akita_error::checked::div_ceil(grinding_plan.total_nonce_bits(), 8)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid nonce stream byte width".into()))?;
    total
        .checked_add(terminal_eor)
        .and_then(|value| value.checked_add(terminal_response))
        .and_then(|value| value.checked_add(nonce_stream_bytes))
        .ok_or_else(|| AkitaError::InvalidSetup("proof payload size overflow".into()))
}

/// Materialize and validate the schedule shared by offline search and generated replay.
///
/// `cached_num_setup_field_elements` is the exact shared flat setup capacity.
pub fn materialize_candidate_schedule(
    cached_total: usize,
    cached_num_setup_field_elements: usize,
    cached_first_direct_setup_field_len: Option<usize>,
    policy: &PlannerPolicy,
    root_layout: &OpeningClaimsLayout,
    folds: Vec<CandidateFoldStep>,
    terminal_response: CandidateTerminalResponse,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let schedule = fold_schedule_from_candidate_parts(&folds, &terminal_response)?;
    let (root, recursive_folds) = folds.split_first().ok_or_else(|| {
        AkitaError::UnsupportedSchedule(
            "a fold schedule requires root and terminal folds".to_string(),
        )
    })?;
    let mut estimate = FoldScheduleEstimate {
        nonce_stream_bytes: 0,
        estimated_root_direct_payload_bytes: root.estimated_direct_payload_bytes,
        estimated_root_stage3_payload_bytes: root.estimated_stage3_payload_bytes,
        estimated_recursive_direct_payload_bytes: recursive_folds
            .iter()
            .map(|fold| fold.estimated_direct_payload_bytes)
            .collect(),
        estimated_recursive_stage3_payload_bytes: recursive_folds
            .iter()
            .map(|fold| fold.estimated_stage3_payload_bytes)
            .collect(),
        estimated_terminal_direct_payload_bytes: terminal_response
            .estimated_direct_payload_bytes
            .checked_add(terminal_response.estimated_payload_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal estimate overflow".to_string()))?,
        estimated_terminal_response_payload_bytes: terminal_response.estimated_payload_bytes,
        estimated_num_setup_field_elements: cached_num_setup_field_elements,
        first_direct_setup_field_len: None,
        selected_offload_edges: 0,
    };
    let grinding_plan = akita_types::derive_transcript_grinding_plan_from_public_shape(
        &schedule,
        root_layout,
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
    )?;
    estimate.nonce_stream_bytes =
        akita_error::checked::div_ceil(grinding_plan.total_nonce_bits(), 8)
            .ok_or_else(|| AkitaError::InvalidSetup("invalid nonce stream byte width".into()))?;
    let recomputed = estimate.estimated_proof_payload_bytes()?;
    if recomputed != cached_total {
        return Err(AkitaError::InvalidSetup(format!(
            "cached schedule cost {cached_total} disagrees with materialized estimate {recomputed}"
        )));
    }
    let first_direct_setup_field_len = match policy.selection_policy {
        SelectionPolicyId::MinEstimatedProofPayloadV2 => None,
        SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
        | SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3 => Some(
            first_direct_setup_field_len_for_schedule(&schedule, root_layout)?,
        ),
    };
    if let Some(cached) = cached_first_direct_setup_field_len {
        if first_direct_setup_field_len != Some(cached) {
            return Err(AkitaError::InvalidSetup(format!(
                "cached first direct setup length {cached} disagrees with materialized length {}",
                first_direct_setup_field_len
                    .map_or_else(|| "none".to_string(), |value| value.to_string())
            )));
        }
    }
    if first_direct_setup_field_len.is_none() {
        schedule.validate_structure()?;
    }
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
        .filter(|fold| fold.params.setup_prefix().is_some())
        .count();
    estimate.first_direct_setup_field_len = first_direct_setup_field_len;
    Ok(PlannedFoldSchedule { schedule, estimate })
}

/// Natural active setup length at the first direct edge in a materialized schedule.
pub fn first_direct_setup_field_len_for_schedule(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    schedule.validate_structure()?;

    for (successor_index, successor) in schedule.recursive_folds.iter().enumerate() {
        if successor.params.setup_prefix().is_some() {
            continue;
        }
        return if successor_index == 0 {
            akita_types::active_setup_field_len(&schedule.root.params, root_layout)
        } else {
            active_setup_field_len_for_recursive_producer(
                &schedule.recursive_folds[successor_index - 1],
            )
        };
    }

    schedule.recursive_folds.last().map_or_else(
        || akita_types::active_setup_field_len(&schedule.root.params, root_layout),
        active_setup_field_len_for_recursive_producer,
    )
}

/// Padded active setup capacity at the first direct edge.
pub fn first_direct_setup_capacity_for_schedule(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    Ok(akita_types::padded_setup_prefix_len(
        first_direct_setup_field_len_for_schedule(schedule, root_layout)?,
    ))
}

fn active_setup_field_len_for_recursive_producer(
    producer: &FoldParams,
) -> Result<usize, AkitaError> {
    let incoming_prefix_len = producer
        .params
        .setup_prefix()
        .map(|prefix| prefix.setup_natural_len.expect("setup prefix group"));
    let layout =
        akita_types::suffix_opening_layout(producer.input_witness_len, incoming_prefix_len)?;
    akita_types::active_setup_field_len(&producer.params, &layout)
}

/// Derive the canonical next-witness field length for a scalar planner level.
pub fn planned_next_witness_len(
    field_bits: u32,
    extension_degree: usize,
    params: &CommittedGroupParams,
    final_num_polys: usize,
    num_chunks: usize,
) -> Result<Option<usize>, AkitaError> {
    if !params.precommitted_groups().is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root witness sizing must use CommittedGroupParams::output_witness_len"
                .to_string(),
        ));
    }
    let opening_batch =
        params.opening_layout_for_final_group(PolynomialGroupLayout::new(0, final_num_polys))?;
    let quotient_plan = akita_types::RelationQuotientPlan::for_field_bits(params, field_bits)?;
    if !params.compression_sources_supported()? {
        return Ok(None);
    }
    let relation_geometry =
        akita_types::RelationWitnessGeometry::for_level(params, &opening_batch, extension_degree)?;
    if params.setup_prefix().is_none() {
        return Ok(Some(WitnessLayout::scalar_live_coeff_len(
            params,
            &opening_batch,
            &relation_geometry,
            num_chunks,
            quotient_plan,
        )?));
    }
    Ok(Some(
        WitnessLayout::new(
            params,
            &opening_batch,
            &relation_geometry,
            num_chunks,
            quotient_plan,
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
            selection_policy: SelectionPolicyId::MinFirstDirectSetupThenPayloadV2,
            recursive_split_search_policy: crate::RecursiveSplitSearchPolicy::Exhaustive,
            recursive_setup_search_policy: crate::RecursiveSetupSearchPolicy::Exhaustive,
            setup_field_budget: None,
            min_offloaded_witness_contraction: 3,
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
    fn recursive_setup_search_policy_has_stable_domain_semantics() {
        assert_eq!(RecursiveSetupSearchPolicy::Exhaustive.tag(), 1);
        assert_eq!(RecursiveSetupSearchPolicy::RootAndFirstChildV1.tag(), 2);
        assert!(RecursiveSetupSearchPolicy::Exhaustive.admits_offloaded_edge_at(12));
        assert!(RecursiveSetupSearchPolicy::RootAndFirstChildV1.admits_offloaded_edge_at(0));
        assert!(RecursiveSetupSearchPolicy::RootAndFirstChildV1.admits_offloaded_edge_at(1));
        assert!(!RecursiveSetupSearchPolicy::RootAndFirstChildV1.admits_offloaded_edge_at(2));
    }

    #[test]
    fn recursive_and_adaptive_direct_policies_use_distinct_setup_objectives() {
        let adaptive = adaptive_policy().ring_dimension_schedule_mode;
        assert_eq!(
            SelectionPolicyId::for_policy(false, adaptive),
            SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
        );
        assert_eq!(
            SelectionPolicyId::for_policy(true, adaptive),
            SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
        );
    }

    #[test]
    fn adaptive_dimensions_are_validated_from_their_role_domains() {
        validate_policy(&adaptive_policy())
            .expect("individually supported D512 A must validate from the adaptive domain");
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
