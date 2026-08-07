//! Planner-free runtime schedule expansion support.

use akita_field::AkitaError;
use akita_types::{
    ChunkedWitnessCfg, CommitmentRingDims, CommittedGroupParams, DecompositionParams, FoldSchedule,
    FoldScheduleEstimate, PlannedFoldSchedule, PolynomialGroupLayout, RecursiveFoldParams,
    RecursiveFoldStep, RootFinalGroupParams, RootFoldParams, RootFoldStep,
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
    /// Pick per-level A rank/dimension, then setup fields and proof payload.
    MinAdaptiveARankDimensionThenSetupAndProof,
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
            Self::MinAdaptiveARankDimensionThenSetupAndProof
        } else {
            Self::MinEstimatedProofPayload
        }
    }

    /// Stable identity tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::MinEstimatedProofPayload => 1,
            Self::MinFirstDirectSetupThenPayload => 2,
            Self::MinAdaptiveARankDimensionThenSetupAndProof => 4,
        }
    }

    /// Stable identity name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::MinEstimatedProofPayload => "MinEstimatedProofPayload",
            Self::MinAdaptiveARankDimensionThenSetupAndProof => {
                "MinAdaptiveARankDimensionThenSetupAndProof"
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

/// Suffix-DP depth cap carried into runtime validation for chunk policy bounds.
pub(crate) const MAX_RECURSION_DEPTH: usize = 12;

/// Validate runtime policy values used by schedule expansion and validation.
pub fn validate_policy(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    policy.challenge_field_bits()?;
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
    for (label, dimension) in [
        ("uniform", policy.uniform_ring_dimension),
        (
            "setup-prefix inner",
            policy.setup_prefix_inner_ring_dimension,
        ),
    ] {
        if !dimension.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule {label} ring dimension must be a nonzero power of two"
            )));
        }
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
            validate_dimension(policy.uniform_ring_dimension, "uniform", ring_dimension)?;
            if ring_dimension != policy.uniform_ring_dimension {
                return Err(AkitaError::InvalidSetup(format!(
                    "uniform schedule D{ring_dimension} must equal configured D{}",
                    policy.uniform_ring_dimension
                )));
            }
        }
        RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            uniform_suffix_dimension,
            potential_a_dimensions,
            potential_b_dimensions,
            potential_d_dimensions,
        } => {
            if num_search_levels == 0 || num_search_levels > MAX_RECURSION_DEPTH {
                return Err(AkitaError::InvalidSetup(format!(
                    "adaptive search levels must be in 1..={MAX_RECURSION_DEPTH}"
                )));
            }
            validate_dimension(
                policy.uniform_ring_dimension,
                "uniform suffix",
                uniform_suffix_dimension,
            )?;
            for (role, dimensions) in [
                ("A", potential_a_dimensions),
                ("B", potential_b_dimensions),
                ("D", potential_d_dimensions),
            ] {
                validate_dimension_list(policy.uniform_ring_dimension, role, dimensions)?;
                if !dimensions.contains(&uniform_suffix_dimension) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "adaptive {role} domain must contain suffix D{uniform_suffix_dimension}"
                    )));
                }
                if dimensions.iter().any(|&d| d < uniform_suffix_dimension) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "adaptive {role} dimensions must be at least suffix D{uniform_suffix_dimension}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_dimension(
    setup_dimension: usize,
    label: &str,
    dimension: usize,
) -> Result<(), AkitaError> {
    if dimension == 0 || !dimension.is_power_of_two() || !setup_dimension.is_multiple_of(dimension)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "{label} D{dimension} must be a nonzero power of two dividing setup D{setup_dimension}"
        )));
    }
    Ok(())
}

fn validate_dimension_list(
    setup_dimension: usize,
    role: &str,
    dimensions: &[usize],
) -> Result<(), AkitaError> {
    if dimensions.is_empty() {
        return Err(AkitaError::InvalidSetup(format!(
            "adaptive {role} domain must be nonempty"
        )));
    }
    for (index, &dimension) in dimensions.iter().enumerate() {
        validate_dimension(setup_dimension, role, dimension)?;
        if index > 0 && dimensions[index - 1] >= dimension {
            return Err(AkitaError::InvalidSetup(format!(
                "adaptive {role} domain must be strictly sorted and duplicate-free"
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
) -> Result<usize, AkitaError> {
    let opening_batch =
        params.opening_layout_for_final_group(PolynomialGroupLayout::new(0, final_num_polys))?;
    let layout = WitnessLayout::new(
        params,
        &opening_batch,
        num_chunks,
        akita_types::sis::compute_num_digits_field_width(field_bits, params.log_basis_open),
    )?;
    Ok(layout.live_coeff_len())
}

/// Convenience policy used by config adapters.
pub fn default_sis_security_policy() -> SisSecurityPolicyId {
    DEFAULT_SIS_SECURITY_POLICY
}
