//! Planner-free runtime schedule expansion support.

use akita_challenges::TensorChallengeShape;
use akita_field::AkitaError;
use akita_types::{
    intermediate_w_ring_element_count_for_chunks, padded_setup_prefix_len, ChunkedWitnessCfg,
    CommittedGroupParams, DecompositionParams, FoldSchedule, FoldScheduleEstimate,
    OpeningClaimsLayout, PlannedFoldSchedule, PolynomialGroupLayout, RecursiveFoldParams,
    RecursiveFoldStep, RootFinalChallenge, RootFinalGroupParams, RootFoldParams, RootFoldStep,
    RootPrecommittedGroupParams, RootSource, SisModulusProfileId, SisSecurityPolicyId,
    TerminalFoldParams, TerminalFoldStep, TerminalResponseShape, WitnessPartition,
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
    /// Pick the minimum estimated proof payload.
    MinEstimatedProofPayload,
    /// Pick the first direct setup footprint, then payload, within setup support.
    MinFirstDirectSetupThenPayloadWithinSupportedEnvelope,
}

impl SelectionPolicyId {
    /// Stable identity tag.
    pub const fn tag(self) -> u32 {
        match self {
            Self::MinEstimatedProofPayload => 1,
            Self::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope => 2,
        }
    }

    /// Stable identity name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::MinEstimatedProofPayload => "MinEstimatedProofPayload",
            Self::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope => {
                "MinFirstDirectSetupThenPayloadWithinSupportedEnvelope"
            }
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
    pub max_setup_envelope_field_elements: usize,
    pub min_offloaded_witness_contraction: usize,
    pub ring_dimension: usize,
    pub decomposition: DecompositionParams,
    pub sis_modulus_profile: SisModulusProfileId,
    pub sis_security_policy: SisSecurityPolicyId,
    pub sis_table_digest: akita_types::SisTableDigest,
    pub ring_subfield_norm_bound: u32,
    pub claim_ext_degree: usize,
    pub chal_ext_degree: usize,
    pub basis_range: (u32, u32),
    pub onehot_chunk_size: usize,
    pub witness_chunk: ChunkedWitnessCfg,
    pub recursive_setup_planning: bool,
}

/// Preferred public name for runtime callers.
pub type RuntimeSchedulePolicy = PlannerPolicy;

impl PlannerPolicy {
    /// Direct-only counterpart used when scalar schedules are cataloged under
    /// the non-recursive family identity.
    pub fn direct_only(self) -> Self {
        Self {
            recursive_setup_planning: false,
            selection_policy: SelectionPolicyId::MinEstimatedProofPayload,
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
}

/// Suffix-DP depth cap carried into runtime validation for chunk policy bounds.
pub(crate) const MAX_RECURSION_DEPTH: usize = 12;

/// Validate runtime policy values used by schedule expansion and validation.
pub(crate) fn validate_policy(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    let expected_selection_policy = if policy.recursive_setup_planning {
        SelectionPolicyId::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope
    } else {
        SelectionPolicyId::MinEstimatedProofPayload
    };
    if policy.selection_policy != expected_selection_policy {
        return Err(AkitaError::InvalidSetup(
            "schedule selection policy disagrees with recursive setup capability".to_string(),
        ));
    }
    if policy.max_setup_envelope_field_elements == 0 {
        return Err(AkitaError::InvalidSetup(
            "maximum setup envelope must be positive".to_string(),
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

/// Resolve the tensor low length independently from the block split.
pub(crate) fn optimize_fold_challenge_shape(
    requested: TensorChallengeShape,
    num_live_blocks: usize,
) -> Result<TensorChallengeShape, AkitaError> {
    if num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold-shape optimization requires a positive num_live_blocks".to_string(),
        ));
    }
    if matches!(requested, TensorChallengeShape::Flat) {
        return Ok(TensorChallengeShape::Flat);
    }

    let capacity = num_live_blocks.checked_next_power_of_two().ok_or_else(|| {
        AkitaError::InvalidSetup("tensor low-length capacity overflow".to_string())
    })?;
    let mut best = None;
    let mut low_len = 1usize;
    loop {
        let high_len = num_live_blocks.div_ceil(low_len);
        let work = high_len
            .checked_add(low_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor verifier-work overflow".to_string()))?;
        if best.is_none_or(|(best_work, best_low)| (work, low_len) < (best_work, best_low)) {
            best = Some((work, low_len));
        }
        if low_len == capacity {
            break;
        }
        low_len = low_len.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidSetup("tensor low-length enumeration overflow".to_string())
        })?;
    }
    let (_, fold_low_len) = best.ok_or_else(|| {
        AkitaError::InvalidSetup("tensor low-length enumeration was empty".to_string())
    })?;
    Ok(TensorChallengeShape::Tensor { fold_low_len })
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateFoldStep {
    pub(crate) params: CommittedGroupParams,
    pub(crate) input_witness_len: usize,
    pub(crate) output_witness_len: usize,
    pub(crate) estimated_direct_payload_bytes: usize,
    pub(crate) estimated_stage3_payload_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateTerminalResponse {
    pub(crate) params: akita_types::TerminalCommittedGroupParams,
    pub(crate) sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub(crate) input_witness_len: usize,
    pub(crate) estimated_direct_payload_bytes: usize,
    pub(crate) response_shape: TerminalResponseShape,
    pub(crate) estimated_payload_bytes: usize,
}

/// Exact Stage-3 payload induced when `successor` consumes a setup prefix.
pub(crate) fn stage3_payload_bytes_for_successor(
    policy: &PlannerPolicy,
    successor: Option<&CommittedGroupParams>,
    output_witness_len: usize,
) -> Result<usize, AkitaError> {
    let Some(prefix) = successor.and_then(|params| params.setup_prefix.as_ref()) else {
        return Ok(usize::default());
    };
    let n_prefix = prefix.n_prefix()?;
    if prefix.d_setup == 0 || !n_prefix.is_multiple_of(prefix.d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix field length does not align with its ring dimension".to_string(),
        ));
    }
    let challenge_field_bits = policy
        .decomposition
        .field_bits()
        .checked_mul(policy.chal_ext_degree as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("challenge field bit width overflow".to_string())
        })?;
    Ok(akita_types::proof_size::stage3_setup_product_bytes(
        challenge_field_bits,
        prefix.d_setup,
        n_prefix / prefix.d_setup,
        output_witness_len,
    ))
}

pub(crate) fn materialize_candidate_schedule(
    cached_total: usize,
    cached_setup_envelope: usize,
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
        estimated_setup_envelope_ring_elements: cached_setup_envelope,
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
                    source: if root.params.onehot_chunk_size == 0 {
                        RootSource::Dense {
                            coefficient_bits: root.params.field_bits_for_cache(),
                        }
                    } else {
                        RootSource::OneHot {
                            chunk_size: root.params.onehot_chunk_size,
                        }
                    },
                    challenge: match root.params.fold_challenge_shape {
                        TensorChallengeShape::Flat => RootFinalChallenge::Flat,
                        TensorChallengeShape::Tensor { fold_low_len } => {
                            RootFinalChallenge::Tensor { fold_low_len }
                        }
                    },
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
                open_commit_matrix: root.params.open_commit_matrix.clone(),
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
                    open_commit_matrix: fold.params.open_commit_matrix.clone(),
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
    let recomputed_envelope =
        akita_types::setup_matrix_envelope_for_schedule(&schedule)?.max_setup_len;
    if recomputed_envelope != cached_setup_envelope {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup envelope {cached_setup_envelope} disagrees with materialized envelope {recomputed_envelope}"
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

fn checked_power_of_two_vars(field_len: usize, context: &'static str) -> Result<usize, AkitaError> {
    if field_len == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "{context} must be nonzero"
        )));
    }
    let padded = field_len.checked_next_power_of_two().ok_or_else(|| {
        AkitaError::InvalidSetup(format!("{context} power-of-two padding overflow"))
    })?;
    Ok(padded.trailing_zeros() as usize)
}

pub fn suffix_opening_layout(
    current_witness_len: usize,
    incoming_setup_prefix: Option<usize>,
) -> Result<OpeningClaimsLayout, AkitaError> {
    let witness_vars = checked_power_of_two_vars(current_witness_len, "suffix witness length")?;
    let witness_group = PolynomialGroupLayout::singleton(witness_vars);
    match incoming_setup_prefix {
        Some(natural_len) => {
            let n_prefix = padded_setup_prefix_len(natural_len);
            if n_prefix == 0 || !n_prefix.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "incoming setup prefix length must be a nonzero power of two".to_string(),
                ));
            }
            let prefix_vars = checked_power_of_two_vars(n_prefix, "incoming setup prefix length")?;
            OpeningClaimsLayout::from_groups(vec![
                PolynomialGroupLayout::singleton(prefix_vars),
                witness_group,
            ])
        }
        None => OpeningClaimsLayout::from_groups(vec![witness_group]),
    }
}

#[allow(clippy::too_many_arguments)]
fn grouped_segment_rings(
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

pub(crate) fn planned_next_witness_len(
    field_bits: u32,
    params: &CommittedGroupParams,
    final_num_polys: usize,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    if !params.precommitted_groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "multi-group root witness sizing must use CommittedGroupParams::output_witness_len"
                .to_string(),
        ));
    }
    if params.setup_prefix.is_some() {
        return grouped_setup_prefix_next_witness_len(
            field_bits,
            params,
            final_num_polys,
            num_chunks,
        );
    }

    intermediate_w_ring_element_count_for_chunks(field_bits, params, final_num_polys, num_chunks)?
        .checked_mul(params.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("next witness length overflow".into()))
}

fn grouped_setup_prefix_next_witness_len(
    field_bits: u32,
    params: &CommittedGroupParams,
    final_num_polys: usize,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    let mut total = grouped_segment_rings(
        final_num_polys,
        params.num_live_blocks,
        num_chunks,
        params.num_positions_per_block,
        params.inner_commit_matrix.output_rank(),
        params.num_digits_inner,
        params.num_digits_outer,
        params.num_digits_open,
        params.num_digits_fold(final_num_polys, field_bits)?,
    )?;
    for group in params.precommitted_group_iter() {
        let group_rings = grouped_segment_rings(
            group.layout.group.num_polynomials(),
            group.layout.num_live_blocks,
            num_chunks,
            group.layout.num_positions_per_block,
            group.inner_commit_matrix.output_rank(),
            group.num_digits_inner,
            group.num_digits_outer,
            group.num_digits_open,
            group.num_digits_fold_one,
        )?;
        total = total
            .checked_add(group_rings)
            .ok_or_else(|| AkitaError::InvalidSetup("grouped witness overflow".to_string()))?;
    }

    let r_rows = params.relation_matrix_row_count(params.precommitted_group_count() + 1)?;
    let r_count = r_rows
        .checked_mul(akita_types::sis::compute_num_digits_field_width(
            field_bits,
            params.log_basis_open,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("grouped r-tail witness overflow".to_string()))?;
    let rings = total
        .checked_add(r_count)
        .ok_or_else(|| AkitaError::InvalidSetup("grouped witness overflow".to_string()))?;

    rings
        .checked_mul(params.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("grouped next witness length overflow".to_string()))
}

/// Convenience policy used by config adapters.
pub fn default_sis_security_policy() -> SisSecurityPolicyId {
    DEFAULT_SIS_SECURITY_POLICY
}
