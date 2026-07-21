//! Typed mixed-D schedule used only by `mixed_d_per_level_e2e.rs`.
//!
//! The fixture retains the shipped envelope prefix and reprices each suffix
//! commitment at the smaller ring dimension. It constructs the same typed
//! root/recursive/terminal topology consumed by production; there is no
//! homogeneous fold-step adapter.

use akita_challenges::TensorChallengeShape;
use akita_config::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_planner::PlannerPolicy;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    min_secure_rank, num_digits_inner, num_digits_open, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, SisMatrixRole, SisTableKey,
};
use akita_types::{
    intermediate_w_ring_element_count_with_counts_bits, AkitaScheduleInputs,
    AkitaScheduleLookupKey, CommittedGroupParams, DecompositionParams, FoldSchedule,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams,
    PolynomialGroupLayout, RecursiveFoldParams, RecursiveFoldStep, TerminalCommittedGroupParams,
    TerminalFoldParams, TerminalFoldStep, TerminalResponseShape, WitnessPartition,
};

fn secure_rank(
    policy: &PlannerPolicy,
    role: SisMatrixRole,
    ring_dimension: usize,
    coeff_linf_bound: u128,
    width: usize,
) -> Result<usize, AkitaError> {
    min_secure_rank(
        SisTableKey {
            policy: policy.sis_security_policy,
            table_digest: policy.sis_table_digest,
            modulus_profile: policy.sis_modulus_profile,
            role,
            ring_dimension: ring_dimension as u32,
            coeff_linf_bound,
        },
        width as u64,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "mixed-D fixture has no audited {role:?} rank for d={ring_dimension}, width={width}, bound={coeff_linf_bound}"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn retarget_recursive_params<Cfg: CommitmentConfig>(
    template: &CommittedGroupParams,
    input_witness_len: usize,
    ring_dimension: usize,
    level: usize,
    num_vars: usize,
    log_basis_inner: u32,
    num_positions_per_block: usize,
) -> Result<CommittedGroupParams, AkitaError> {
    if ring_dimension == 0 || !input_witness_len.is_multiple_of(ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "mixed-D suffix witness is not ring-aligned".into(),
        ));
    }
    let policy = policy_of::<Cfg>();
    let log_basis_outer = template.log_basis_outer;
    let log_basis_open = template.log_basis_open;
    let inner_decomposition = DecompositionParams {
        log_basis: log_basis_inner,
        ..policy.decomposition
    };
    let outer_decomposition = DecompositionParams {
        log_basis: log_basis_outer,
        ..policy.decomposition
    };
    let open_decomposition = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_live_ring_elements_per_claim = input_witness_len / ring_dimension;
    let num_live_blocks = num_live_ring_elements_per_claim.div_ceil(num_positions_per_block);
    let num_digits_inner = num_digits_inner(inner_decomposition, false);
    let num_digits_outer = num_digits_open(outer_decomposition);
    let num_digits_open = num_digits_open(open_decomposition);
    let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-D A width overflow".into()))?;
    let sparse = Cfg::ring_challenge_config(ring_dimension)?;
    let fold_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars,
        level,
        input_witness_len,
    });
    if fold_shape != TensorChallengeShape::Flat {
        return Err(AkitaError::InvalidSetup(
            "mixed-D recursive fixture requires flat challenges".into(),
        ));
    }
    let a_bound = rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        ring_dimension,
        inner_decomposition,
        log_basis_open,
        &sparse,
        fold_shape,
        false,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        num_live_blocks,
        1,
        inner_width as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("mixed-D A bound is unsupported".into()))?;
    let n_a = secure_rank(
        &policy,
        SisMatrixRole::Inner,
        ring_dimension,
        a_bound,
        inner_width,
    )?;
    let outer_width = decomposed_t_ring_count(n_a, num_digits_outer, num_live_blocks, 1)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-D B width overflow".into()))?;
    let b_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Outer,
        ring_dimension,
        log_basis_outer,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("mixed-D B bound is unsupported".into()))?;
    let n_b = secure_rank(
        &policy,
        SisMatrixRole::Outer,
        ring_dimension,
        b_bound,
        outer_width,
    )?;
    let open_width = decomposed_w_ring_count(num_digits_open, num_live_blocks, 1)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-D D width overflow".into()))?;
    let d_bound = rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        SisMatrixRole::Open,
        ring_dimension,
        log_basis_open,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("mixed-D D bound is unsupported".into()))?;
    let n_d = secure_rank(
        &policy,
        SisMatrixRole::Open,
        ring_dimension,
        d_bound,
        open_width,
    )?;
    let mut params = CommittedGroupParams {
        log_basis_inner,
        log_basis_outer,
        log_basis_open,
        inner_commit_matrix: InnerCommitMatrixParams::try_new(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            n_a,
            inner_width,
            a_bound,
            ring_dimension,
        )?,
        outer_commit_matrix: OuterCommitMatrixParams::try_new(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            n_b,
            outer_width,
            b_bound,
            ring_dimension,
        )?,
        open_commit_matrix: OpenCommitMatrixParams::try_new(
            policy.sis_security_policy,
            policy.sis_table_digest,
            policy.sis_modulus_profile,
            n_d,
            open_width,
            d_bound,
            ring_dimension,
        )?,
        num_live_ring_elements_per_claim,
        num_positions_per_block,
        num_live_blocks,
        fold_challenge_config: sparse,
        fold_challenge_shape: fold_shape,
        num_digits_inner,
        num_digits_outer,
        num_digits_open,
        onehot_chunk_size: template.onehot_chunk_size,
        fold_linf_cap_config: akita_types::sis::FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_block_claims: 0,
        cached_num_digits_fold_value: 1,
        witness_chunk: akita_types::ChunkedWitnessCfg::default_non_chunked(),
        precommitted_groups: Vec::new(),
        setup_prefix: None,
    };
    params = params.with_fold_linf_cap_config(policy.decomposition.field_bits(), 1)?;
    Ok(params)
}

/// Build a mixed-D schedule without recreating the retired homogeneous step
/// model. Fold level 0 is the typed root, levels 1.. are recursive entries,
/// and the last level is the typed direct terminal fold.
pub(super) fn mixed_d_per_level_schedule<EnvelopeCfg, SuffixCfg>(
    num_vars: usize,
    num_polynomials: usize,
    switch_at_fold: usize,
) -> Result<FoldSchedule, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    if num_polynomials != 1 || switch_at_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "mixed-D fixture requires a singleton and a non-root switch".into(),
        ));
    }
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polynomials));
    let envelope = EnvelopeCfg::runtime_schedule(key)?;
    let total_levels = envelope.num_fold_levels();
    if switch_at_fold >= total_levels {
        return Err(AkitaError::InvalidSetup(format!(
            "mixed-D switch level {switch_at_fold} is outside {total_levels} fold levels"
        )));
    }
    let keep_recursive = switch_at_fold - 1;
    if keep_recursive > envelope.recursive_folds.len() {
        return Err(AkitaError::InvalidSetup(
            "mixed-D switch skips beyond the recursive suffix".into(),
        ));
    }
    let ring_dimension = SuffixCfg::D;
    let field_bits = SuffixCfg::decomposition().field_bits();
    let mut recursive_folds = envelope.recursive_folds[..keep_recursive].to_vec();
    let mut input_witness_len = if let Some(last) = recursive_folds.last() {
        last.output_witness_len
    } else {
        envelope.root.output_witness_len
    };
    for (offset, template_step) in envelope.recursive_folds[keep_recursive..]
        .iter()
        .enumerate()
    {
        let level = switch_at_fold + offset;
        let witness = retarget_recursive_params::<SuffixCfg>(
            &template_step.params.witness,
            input_witness_len,
            ring_dimension,
            level,
            num_vars,
            template_step.params.witness.log_basis_inner,
            template_step.params.witness.num_positions_per_block,
        )?;
        let output_witness_len =
            intermediate_w_ring_element_count_with_counts_bits(field_bits, &witness, 1, 1)?
                .checked_mul(ring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("mixed-D witness length overflow".into())
                })?;
        recursive_folds.push(RecursiveFoldStep {
            params: RecursiveFoldParams {
                open_commit_matrix: witness.open_commit_matrix.clone(),
                sparse_challenge_config: SuffixCfg::ring_challenge_config(ring_dimension)?,
                witness,
                incoming_setup_prefix: None,
                witness_partition: WitnessPartition::Single,
            },
            input_witness_len,
            output_witness_len,
        });
        input_witness_len = output_witness_len;
    }

    let template = recursive_folds
        .last()
        .map(|step| &step.params.witness)
        .unwrap_or(&envelope.root.params.final_group.commitment);
    let terminal_expanded = retarget_recursive_params::<SuffixCfg>(
        template,
        input_witness_len,
        ring_dimension,
        total_levels - 1,
        num_vars,
        envelope.terminal.params.witness.log_basis_inner,
        envelope.terminal.params.witness.num_positions_per_block,
    )?;
    let terminal_witness = TerminalCommittedGroupParams::from_expanded_group(terminal_expanded);
    let sparse = SuffixCfg::ring_challenge_config(ring_dimension)?;
    let (honest_cap, _) = terminal_witness.response_linf_bounds(&sparse)?;
    let response_shape = TerminalResponseShape::derive(&terminal_witness, honest_cap)?;
    let schedule = FoldSchedule {
        root: envelope.root,
        recursive_folds,
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: terminal_witness,
                sparse_challenge_config: sparse,
                response_shape,
            },
            input_witness_len,
        },
    };
    schedule.validate_structure()?;
    let opening_batch = OpeningClaimsLayout::new(num_vars, 1)?;
    schedule
        .root
        .params
        .final_group
        .commitment
        .validate_opening_batch(&opening_batch)?;
    Ok(schedule)
}
