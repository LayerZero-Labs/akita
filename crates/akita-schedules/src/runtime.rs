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

/// One empirical calibration for selective L2 response planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SelectiveL2FoldCap {
    pub fold_level: usize,
    pub input_witness_len: usize,
    /// Source signed-digit basis used by the measured state.
    pub source_log_basis: u32,
    /// Ring dimension whose challenge distribution produced the measurement.
    pub challenge_ring_dimension: usize,
    /// Exact squared challenge-energy bound used by the measurement.
    pub challenge_l2_sq: u128,
    /// Exact physical response length, or zero for a source-state calibration.
    /// A zero-length row also opts the family into the canonical balanced-digit
    /// response model at other eligible suffix states.
    pub physical_response_len: usize,
    pub fold_basis: usize,
    pub fold_digit_count: usize,
    pub response_l2_sq_cap: u128,
}

impl SelectiveL2FoldCap {
    /// Materialize a verifier-enforced cap from a measured source energy.
    ///
    /// `headroom_ppm` scales the exact conditional mean
    /// `challenge_l2_sq * source_l2_sq`. A model row intentionally leaves the
    /// response length at zero so planner search can price the canonical split
    /// without binding the calibration to one response geometry.
    #[allow(clippy::too_many_arguments)]
    pub const fn from_source_energy_model(
        fold_level: usize,
        input_witness_len: usize,
        source_log_basis: u32,
        challenge_ring_dimension: usize,
        fold_basis: usize,
        fold_digit_count: usize,
        source_l2_sq: u128,
        challenge_l2_sq: u128,
        headroom_ppm: u128,
    ) -> Self {
        const SCALE: u128 = 1_000_000;
        let conditional_mean = match source_l2_sq.checked_mul(challenge_l2_sq) {
            Some(value) => value,
            None => panic!("selective L2 conditional mean overflow"),
        };
        let scaled = match conditional_mean.checked_mul(headroom_ppm) {
            Some(value) => value,
            None => panic!("selective L2 headroom overflow"),
        };
        let rounded = match scaled.checked_add(SCALE - 1) {
            Some(value) => value,
            None => panic!("selective L2 cap rounding overflow"),
        };
        Self {
            fold_level,
            input_witness_len,
            source_log_basis,
            challenge_ring_dimension,
            challenge_l2_sq,
            physical_response_len: 0,
            fold_basis,
            fold_digit_count,
            response_l2_sq_cap: rounded / SCALE,
        }
    }
}

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
    /// Empirical calibration caps. Empty keeps every level on the Linf route.
    pub selective_l2_fold_caps: &'static [SelectiveL2FoldCap],
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

    /// Whether this family opts into the balanced-digit suffix response model.
    pub fn selective_l2_response_model_enabled(&self) -> bool {
        self.selective_l2_fold_caps
            .iter()
            .any(|entry| entry.physical_response_len == 0)
    }

    /// Return the data-backed L2 cap for this candidate geometry.
    ///
    /// Exact measured rows take precedence. Other scalar single-chunk states
    /// in an opted-in family use the balanced-digit second moment
    /// `n * (B^2 + 2) / 12`, multiplied by the exact challenge energy and a
    /// 1.75 empirical headroom. The model is a completeness and planning
    /// input only; the resulting cap is frozen into the schedule and enforced
    /// exactly by the verifier.
    #[allow(clippy::too_many_arguments)]
    pub fn selective_l2_cap_for_candidate(
        &self,
        fold_level: usize,
        input_witness_len: usize,
        physical_response_len: usize,
        source_log_basis: u32,
        challenge_ring_dimension: usize,
        fold_basis: usize,
        fold_digit_count: usize,
        challenge_l2_sq: u128,
    ) -> Option<u128> {
        let exact = self
            .selective_l2_fold_caps
            .iter()
            .find(|entry| {
                (
                    entry.fold_level,
                    entry.input_witness_len,
                    entry.source_log_basis,
                    entry.challenge_ring_dimension,
                    entry.challenge_l2_sq,
                    entry.fold_basis,
                    entry.fold_digit_count,
                ) == (
                    fold_level,
                    input_witness_len,
                    source_log_basis,
                    challenge_ring_dimension,
                    challenge_l2_sq,
                    fold_basis,
                    fold_digit_count,
                ) && (entry.physical_response_len == 0
                    || entry.physical_response_len == physical_response_len)
            })
            .map(|entry| entry.response_l2_sq_cap);
        if exact.is_some() {
            return exact;
        }
        if !self.selective_l2_response_model_enabled()
            || fold_level < 3
            || input_witness_len == 0
            || challenge_l2_sq == 0
        {
            return None;
        }

        const MODEL_HEADROOM_PPM: u128 = 1_750_000;
        const MODEL_SCALE: u128 = 12_000_000;
        let basis = 1u128.checked_shl(source_log_basis)?;
        let second_moment_numerator = basis.checked_mul(basis)?.checked_add(2)?;
        let scaled = (input_witness_len as u128)
            .checked_mul(second_moment_numerator)?
            .checked_mul(challenge_l2_sq)?
            .checked_mul(MODEL_HEADROOM_PPM)?;
        scaled
            .checked_add(MODEL_SCALE - 1)
            .map(|rounded| rounded / MODEL_SCALE)
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
/// First fold level eligible for a calibrated L2 candidate.
const SELECTIVE_L2_CAP_FIRST_LEVEL: usize = 3;

fn validate_selective_l2_caps(caps: &[SelectiveL2FoldCap]) -> Result<(), AkitaError> {
    for (index, cap) in caps.iter().enumerate() {
        let invalid_geometry = cap.fold_level < SELECTIVE_L2_CAP_FIRST_LEVEL
            || cap.fold_level >= MAX_RECURSION_DEPTH
            || cap.input_witness_len == 0
            || cap.source_log_basis == 0
            || cap.source_log_basis > MAX_I16_LOG_BASIS
            || akita_challenges::selective_l2_challenge_config(cap.challenge_ring_dimension)
                .is_none_or(|cfg| cfg.challenge_l2_sq_max() != cap.challenge_l2_sq)
            || cap.fold_basis < 16
            || !cap.fold_basis.is_power_of_two()
            || cap.fold_digit_count == 0
            || cap.response_l2_sq_cap == 0;
        let invalid_order = index > 0 && caps[index - 1] >= *cap;
        let duplicate_key = index > 0 && {
            let previous = caps[index - 1];
            previous.fold_level == cap.fold_level
                && previous.input_witness_len == cap.input_witness_len
                && previous.source_log_basis == cap.source_log_basis
                && previous.challenge_ring_dimension == cap.challenge_ring_dimension
                && previous.challenge_l2_sq == cap.challenge_l2_sq
                && previous.physical_response_len == cap.physical_response_len
                && previous.fold_basis == cap.fold_basis
                && previous.fold_digit_count == cap.fold_digit_count
        };
        if invalid_geometry || invalid_order || duplicate_key {
            return Err(AkitaError::InvalidSetup(
                "selective L2 caps must be valid, later-fold, strictly sorted candidates".into(),
            ));
        }
    }
    Ok(())
}

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
    if !policy.selective_l2_fold_caps.is_empty()
        && policy.sis_l2_table_digest != akita_types::SisL2TableDigest::CURRENT
    {
        return Err(AkitaError::InvalidSetup(
            "selective L2 caps require the current audited Euclidean table".into(),
        ));
    }
    validate_selective_l2_caps(policy.selective_l2_fold_caps)?;
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
    if params.setup_prefix.is_none() {
        return WitnessLayout::try_scalar_live_coeff_len(
            params,
            &opening_batch,
            num_chunks,
            quotient_depth,
        );
    }
    if !params.compression_sources_supported()? {
        return Ok(None);
    }
    Ok(Some(
        WitnessLayout::new(params, &opening_batch, num_chunks, quotient_depth)?.live_coeff_len(),
    ))
}

/// Convenience policy used by config adapters.
pub fn default_sis_security_policy() -> SisSecurityPolicyId {
    DEFAULT_SIS_SECURITY_POLICY
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(fold_level: usize, input_witness_len: usize) -> SelectiveL2FoldCap {
        SelectiveL2FoldCap {
            fold_level,
            input_witness_len,
            source_log_basis: 1,
            challenge_ring_dimension: 64,
            challenge_l2_sq: 75,
            physical_response_len: 64,
            fold_basis: 16,
            fold_digit_count: 3,
            response_l2_sq_cap: 1,
        }
    }

    const A_DIMENSIONS_WITHOUT_GLOBAL_CARRIER: &[usize] = &[64, 512];
    const SUFFIX_DIMENSIONS: &[usize] = &[64];

    fn adaptive_policy() -> PlannerPolicy {
        PlannerPolicy {
            cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
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
            selective_l2_fold_caps: &[],
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            inner_basis_range: (3, 16),
            opening_basis_range: (3, 6),
            witness_chunk: ChunkedWitnessCfg::default(),
            recursive_setup_planning: false,
        }
    }

    #[test]
    fn selective_l2_caps_allow_sparse_later_exact_candidates() {
        assert!(validate_selective_l2_caps(&[cap(4, 10), cap(7, 6)]).is_ok());
        assert!(validate_selective_l2_caps(&[cap(2, 10)]).is_err());
        assert!(validate_selective_l2_caps(&[cap(4, 10), cap(4, 10)]).is_err());
    }

    #[test]
    fn selective_l2_caps_reject_unbound_model_sentinels() {
        let valid = cap(4, 10);
        for invalid in [
            SelectiveL2FoldCap {
                source_log_basis: 0,
                ..valid
            },
            SelectiveL2FoldCap {
                challenge_ring_dimension: 256,
                ..valid
            },
            SelectiveL2FoldCap {
                challenge_l2_sq: 31,
                ..valid
            },
            SelectiveL2FoldCap {
                fold_basis: 8,
                ..valid
            },
        ] {
            assert!(validate_selective_l2_caps(&[invalid]).is_err());
        }

        let mut duplicate_key = valid;
        duplicate_key.response_l2_sq_cap = 2;
        assert!(validate_selective_l2_caps(&[valid, duplicate_key]).is_err());
    }

    #[test]
    fn source_energy_model_materializes_cap_and_matches_every_split() {
        static MODEL: [SelectiveL2FoldCap; 1] = [SelectiveL2FoldCap::from_source_energy_model(
            3, 2_083_904, 3, 64, 16, 3, 9_441_218, 75, 1_060_000,
        )];
        assert_eq!(MODEL[0].physical_response_len, 0);
        assert_eq!(MODEL[0].response_l2_sq_cap, 750_576_831);
        assert!(validate_selective_l2_caps(&MODEL).is_ok());

        let mut policy = adaptive_policy();
        policy.selective_l2_fold_caps = &MODEL;
        for physical_response_len in [32_768, 65_536, 131_072, 262_144] {
            assert_eq!(
                policy.selective_l2_cap_for_candidate(
                    3,
                    2_083_904,
                    physical_response_len,
                    3,
                    64,
                    16,
                    3,
                    75,
                ),
                Some(750_576_831),
            );
        }
        assert_eq!(
            policy.selective_l2_cap_for_candidate(3, 2_083_904, 65_536, 3, 64, 32, 3, 75),
            Some(1_504_318_200),
        );
        for mismatched in [
            policy.selective_l2_cap_for_candidate(3, 2_083_904, 65_536, 4, 64, 16, 3, 75),
            policy.selective_l2_cap_for_candidate(3, 2_083_904, 65_536, 3, 128, 16, 3, 75),
            policy.selective_l2_cap_for_candidate(3, 2_083_904, 65_536, 3, 64, 16, 3, 31),
        ] {
            assert_ne!(mismatched, Some(750_576_831));
        }
    }

    #[test]
    fn balanced_digit_model_covers_every_measured_field_profile_state() {
        static OPT_IN: [SelectiveL2FoldCap; 1] = [SelectiveL2FoldCap::from_source_energy_model(
            3, 1, 3, 64, 16, 3, 1, 75, 1_000_000,
        )];
        let mut policy = adaptive_policy();
        policy.selective_l2_fold_caps = &OPT_IN;
        let measured_caps = [
            (511_872, 5, 75, 2_493_100_682u128),
            (2_083_904, 3, 75, 750_576_831),
            (231_488, 6, 75, 3_898_932_413),
            (144_384, 6, 75, 3_003_850_896),
            (252_544, 6, 31, 2_982_511_152),
            (130_816, 6, 31, 1_754_411_666),
            (594_624, 3, 75, 304_965_657),
            (223_744, 5, 75, 1_694_697_605),
            (124_672, 6, 75, 3_572_946_399),
        ];
        for (input_witness_len, source_log_basis, challenge_l2_sq, measured_cap) in measured_caps {
            let modeled = policy
                .selective_l2_cap_for_candidate(
                    99,
                    input_witness_len,
                    1,
                    source_log_basis,
                    64,
                    16,
                    3,
                    challenge_l2_sq,
                )
                .expect("opted-in model row");
            assert!(modeled >= measured_cap, "{modeled} < {measured_cap}");
        }
    }

    #[test]
    fn balanced_digit_model_keeps_empirical_margin_across_all_calibrated_profiles() {
        static OPT_IN: [SelectiveL2FoldCap; 1] = [SelectiveL2FoldCap::from_source_energy_model(
            3, 1, 3, 64, 16, 3, 1, 75, 1_000_000,
        )];
        let mut policy = adaptive_policy();
        policy.selective_l2_fold_caps = &OPT_IN;
        // Maxima from independent end-to-end profile transcripts after the
        // modeled schedules reached their final fixed point. These cover every
        // supported field profile, dense and one-hot witness families, and
        // every selected source basis in the CI rows.
        let measured_response_maxima = [
            // fp32, nv30: L3, L4, terminal (five transcripts).
            (550_400, 4, 31, 543_303_338u128),
            (253_440, 4, 31, 237_644_475),
            (125_568, 5, 31, 404_233_159),
            // fp64, nv30: L3, L4, L5, terminal (three transcripts).
            (707_264, 3, 75, 381_539_045),
            (224_576, 5, 75, 1_627_247_429),
            (113_408, 6, 75, 3_039_802_323),
            (80_640, 6, 75, 2_428_004_762),
            // fp128, nv36: L3, L4, L5, terminal (three transcripts).
            (1_046_016, 4, 75, 1_466_514_925),
            (396_800, 4, 75, 506_859_809),
            (267_776, 4, 75, 353_853_006),
            (184_320, 4, 75, 244_149_861),
            // fp128 multi-group recursive, nv32/4 polys (one transcript).
            (1_574_400, 4, 75, 1_944_140_528),
            (488_960, 4, 75, 640_793_094),
            (288_256, 4, 75, 384_556_280),
            (226_816, 4, 75, 272_556_161),
            // fp128 recursive W8R2 multi-group, nv32/4 polys (one transcript).
            (800_256, 4, 75, 1_041_804_258),
            (366_080, 4, 75, 451_575_010),
            (257_536, 4, 75, 325_401_197),
            (176_128, 4, 75, 226_971_740),
            // fp128 direct W2R2, nv32 (one transcript).
            (1_009_152, 4, 75, 1_410_227_575),
            (396_800, 4, 75, 505_517_105),
            (267_776, 4, 75, 343_828_764),
            (184_320, 4, 75, 245_434_915),
            // fp128 direct W4R2, nv32 (one transcript).
            (380_800, 5, 75, 1_869_008_722),
            (159_872, 6, 75, 3_180_422_664),
            // fp128 direct W8R2, nv32 (one transcript).
            (280_960, 5, 75, 1_247_240_235),
            (160_320, 6, 75, 2_694_523_385),
            // fp128 multi-group direct, nv32/4 polys (one transcript).
            (1_031_232, 3, 75, 326_301_206),
            (361_984, 4, 75, 409_562_093),
            (257_536, 4, 75, 321_178_079),
            (176_128, 4, 75, 228_249_358),
            // fp32 dense, nv26 (four transcripts).
            (550_400, 4, 31, 529_173_246),
            (253_440, 4, 31, 233_433_991),
            (125_568, 5, 31, 406_864_610),
            // fp64 dense, nv26 (four transcripts).
            (488_960, 4, 75, 990_837_236),
            (255_488, 4, 75, 495_567_628),
            (112_704, 6, 75, 2_976_911_626),
            (80_640, 6, 75, 2_416_267_258),
            // fp128 dense, nv28 (four transcripts).
            (1_097_280, 3, 75, 368_452_280),
            (407_040, 4, 75, 498_875_502),
            (267_776, 4, 75, 352_749_212),
            (184_320, 4, 75, 253_707_961),
            // fp128 dense W8R2, nv16 (four transcripts).
            (157_504, 6, 75, 2_643_733_056),
        ];
        for (input_witness_len, source_log_basis, challenge_l2_sq, measured_max) in
            measured_response_maxima
        {
            let modeled = policy
                .selective_l2_cap_for_candidate(
                    99,
                    input_witness_len,
                    1,
                    source_log_basis,
                    64,
                    16,
                    3,
                    challenge_l2_sq,
                )
                .expect("opted-in model row");
            assert!(
                modeled * 1_000_000 >= measured_max * 1_150_000,
                "modeled cap {modeled} lacks 15% margin over measured max {measured_max}"
            );
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
