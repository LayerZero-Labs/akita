//! Mixed-D per-level schedule fixture for the runtime ring cutover capstone E2E.
//!
//! Only imported by `mixed_d_per_level_e2e.rs`. Production schedules stay
//! uniform-D until the planner grows a DP `ring_d` search
//! (`specs/mixed-row-ring-dimensions.md`).

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_config::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_planner::generated::{table_entry, GeneratedFoldStep};
use akita_planner::PlannerPolicy;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    min_secure_rank, num_digits_open, num_digits_s_commit, rounded_up_collision_inf_norm,
    rounded_up_role_a_inf_norm, SisMatrixRole, SisTableDigest, SisTableKey,
};
use akita_types::{
    direct_witness_bytes, intermediate_w_ring_element_count_with_counts_bits, level_proof_bytes,
    segment_typed_witness_shape_from_groups, AjtaiKeyParams, AkitaScheduleInputs,
    AkitaScheduleLookupKey, CommitmentRingDims, DecompositionParams, DirectStep, FoldStep,
    LevelParams, LevelParamsLike, PolynomialGroupLayout, RelationMatrixRowLayout, Schedule,
};
struct MixedSuffixFoldPlan {
    params: LevelParams,
    current_w_len: usize,
    next_w_len: usize,
    is_terminal: bool,
}

fn generated_fold_step<Cfg: CommitmentConfig>(
    key: PolynomialGroupLayout,
    level: usize,
) -> Result<GeneratedFoldStep, AkitaError> {
    let catalog = Cfg::schedule_catalog().ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "{} missing generated schedule catalog",
            std::any::type_name::<Cfg>()
        ))
    })?;
    let table_key = AkitaScheduleLookupKey::single(key);
    let entry = table_entry(catalog, &table_key).ok_or_else(|| {
        AkitaError::InvalidSetup(format!("missing generated schedule for {key:?}"))
    })?;
    entry
        .folds
        .get(level)
        .map(|fold| *fold.fold_step())
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "fold level {level} missing from {} table entry {key:?}",
                std::any::type_name::<Cfg>()
            ))
        })
}

/// Expand a compact generated fold step's witness geometry at a different
/// active ring dimension.
///
/// Mixed-D hand schedules keep the envelope `current_w_len` / block geometry
/// while halving `ring_d`. Ranks are recomputed for the target geometry
/// instead of reusing the stored compact tuple from either table (the same
/// role-pricing primitives `GeneratedFoldStep::expand_to_level_params` audits
/// stored ranks against). `extra_block_vars` is the additional `block_index_bits` added
/// when the previous fold executed at a larger ring dimension (typically `1`
/// on the first suffix level after a `128 → 64` drop, `0` thereafter).
#[allow(clippy::too_many_arguments)]
fn expand_envelope_witness_at_ring_d(
    step: &GeneratedFoldStep,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_level: usize,
    envelope_current_w_len: usize,
    target_ring_d: usize,
    fold_shape: TensorChallengeShape,
    num_claims: usize,
    extra_block_vars: usize,
    block_m_vars: Option<usize>,
    block_r_vars: Option<usize>,
) -> Result<LevelParams, AkitaError> {
    if target_ring_d == 0 {
        return Err(AkitaError::InvalidSetup(
            "mixed-D target ring dimension must be nonzero".into(),
        ));
    }
    let is_root = fold_level == 0;
    let log_basis = step.log_basis;
    let sis_modulus_profile = policy.sis_modulus_profile;
    let sis_policy = policy.sis_security_policy;
    let position_index_bits = block_m_vars.unwrap_or(step.position_index_bits as usize);
    let block_index_bits = block_r_vars
        .unwrap_or(step.block_index_bits as usize)
        .checked_add(extra_block_vars)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-D block variable count overflow".into()))?;
    let num_positions_per_block =
        1usize
            .checked_shl(position_index_bits as u32)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "generated schedule 2^position_index_bits overflows usize".to_string(),
                )
            })?;
    let num_live_ring_elements_per_claim = if is_root {
        let capacity = 1usize.checked_shl(block_index_bits as u32).ok_or_else(|| {
            AkitaError::InvalidSetup(
                "generated schedule block-index domain size overflows usize".to_string(),
            )
        })?;
        num_positions_per_block
            .checked_mul(capacity)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("mixed-D root source length overflow".to_string())
            })?
    } else {
        envelope_current_w_len / target_ring_d
    };
    let num_live_blocks = num_live_ring_elements_per_claim.div_ceil(num_positions_per_block);
    let no_layout = |role: &str| {
        AkitaError::InvalidSetup(format!(
            "no audited {role}-role layout for mixed-D schedule \
             (family={sis_modulus_profile:?}, d={target_ring_d}, log_basis={log_basis})"
        ))
    };
    let decomp = DecompositionParams {
        log_basis,
        ..policy.decomposition
    };
    let ring_challenge_cfg = ring_challenge_config(target_ring_d)?;
    let num_digits_commit = num_digits_s_commit(decomp, is_root);
    let num_digits_open_val = num_digits_open(decomp);
    let inner_width = decomposed_s_block_ring_count(num_positions_per_block, num_digits_commit)
        .ok_or_else(|| no_layout("A"))?;
    let a_bucket = rounded_up_role_a_inf_norm(
        sis_policy,
        sis_modulus_profile,
        target_ring_d,
        decomp,
        &ring_challenge_cfg,
        fold_shape,
        is_root,
        policy.onehot_chunk_size,
        policy.ring_subfield_norm_bound,
        num_live_blocks,
        num_claims,
        inner_width as u64,
    )
    .ok_or_else(|| no_layout("A"))?;
    let n_a = min_secure_rank(
        SisTableKey {
            policy: sis_policy,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: sis_modulus_profile,
            role: SisMatrixRole::A,
            ring_dimension: target_ring_d as u32,
            coeff_linf_bound: a_bucket,
        },
        inner_width as u64,
    )
    .ok_or_else(|| no_layout("A"))?;
    let b_bucket = rounded_up_collision_inf_norm(
        sis_policy,
        sis_modulus_profile,
        SisMatrixRole::B,
        target_ring_d,
        log_basis,
    )
    .ok_or_else(|| no_layout("B"))?;
    let outer_width =
        decomposed_t_ring_count(n_a, num_digits_open_val, num_live_blocks, num_claims)
            .ok_or_else(|| no_layout("B"))?;
    let d_bucket = rounded_up_collision_inf_norm(
        sis_policy,
        sis_modulus_profile,
        SisMatrixRole::D,
        target_ring_d,
        log_basis,
    )
    .ok_or_else(|| no_layout("D"))?;
    let d_matrix_width = decomposed_w_ring_count(num_digits_open_val, num_live_blocks, num_claims)
        .ok_or_else(|| no_layout("D"))?;
    let n_b = min_secure_rank(
        SisTableKey {
            policy: sis_policy,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: sis_modulus_profile,
            role: SisMatrixRole::B,
            ring_dimension: target_ring_d as u32,
            coeff_linf_bound: b_bucket,
        },
        outer_width as u64,
    )
    .ok_or_else(|| no_layout("B"))?;
    let n_d = min_secure_rank(
        SisTableKey {
            policy: sis_policy,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: sis_modulus_profile,
            role: SisMatrixRole::D,
            ring_dimension: target_ring_d as u32,
            coeff_linf_bound: d_bucket,
        },
        d_matrix_width as u64,
    )
    .ok_or_else(|| no_layout("D"))?;
    let onehot_chunk_size = if is_root && policy.decomposition.log_commit_bound == 1 {
        policy.onehot_chunk_size
    } else {
        0
    };
    let mut params = LevelParams {
        ring_dimension: target_ring_d,
        log_basis,
        a_key: AjtaiKeyParams::try_new(
            sis_policy,
            SisTableDigest::CURRENT,
            sis_modulus_profile,
            SisMatrixRole::A,
            n_a,
            inner_width,
            a_bucket,
            target_ring_d,
        )?,
        b_key: AjtaiKeyParams::try_new(
            sis_policy,
            SisTableDigest::CURRENT,
            sis_modulus_profile,
            SisMatrixRole::B,
            n_b,
            outer_width,
            b_bucket,
            target_ring_d,
        )?,
        d_key: AjtaiKeyParams::try_new(
            sis_policy,
            SisTableDigest::CURRENT,
            sis_modulus_profile,
            SisMatrixRole::D,
            n_d,
            d_matrix_width,
            d_bucket,
            target_ring_d,
        )?,
        num_live_ring_elements_per_claim,
        num_live_blocks,
        num_positions_per_block,
        fold_challenge_config: ring_challenge_cfg,
        fold_challenge_shape: fold_shape,
        num_digits_commit,
        num_digits_open: num_digits_open_val,
        onehot_chunk_size,
        fold_linf_cap_config: akita_types::sis::FoldWitnessLinfCapConfig::worst_case_beta_only(),
        num_digits_fold_one: 1,
        field_bits_hint: 0,
        cached_num_digits_block_claims: 0,
        cached_num_digits_fold_value: 1,
        precommitted_groups: Vec::new(),
        setup_prefix: None,
        witness_chunk: akita_types::witness::ChunkedWitnessCfg::default_non_chunked(),
        role_dims: CommitmentRingDims::uniform(target_ring_d),
        setup_contribution_mode: akita_types::SetupContributionMode::Direct,
    };
    params.stamp_role_dims_from_keys();
    params.with_fold_linf_cap_config(policy.decomposition.field_bits(), num_claims)
}

fn mixed_continue_suffix_level<EnvelopeCfg, SuffixCfg>(
    key: PolynomialGroupLayout,
    level: usize,
    current_w_len: usize,
    suffix_ring_d: usize,
    block_m_vars: usize,
    block_r_vars: usize,
) -> Result<LevelParams, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    let envelope_gen = generated_fold_step::<EnvelopeCfg>(key, level)?;
    let suffix_policy = policy_of::<SuffixCfg>();
    let fold_shape = SuffixCfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level,
        current_w_len,
    });
    expand_envelope_witness_at_ring_d(
        &envelope_gen,
        &suffix_policy,
        SuffixCfg::ring_challenge_config,
        level,
        current_w_len,
        suffix_ring_d,
        fold_shape,
        1,
        0,
        Some(block_m_vars),
        Some(block_r_vars),
    )
}

fn mixed_level_params<EnvelopeCfg, SuffixCfg>(
    key: PolynomialGroupLayout,
    level: usize,
    envelope_current_w_len: usize,
    envelope_params: &LevelParams,
    suffix_ring_d: usize,
    prev_ring_d: usize,
) -> Result<LevelParams, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    let envelope_gen = generated_fold_step::<EnvelopeCfg>(key, level)?;
    let extra_block_vars = extra_block_vars_for_drop(prev_ring_d, suffix_ring_d);
    if envelope_gen.ring_d as usize == suffix_ring_d && extra_block_vars == 0 {
        return Ok(envelope_params.clone());
    }
    let suffix_policy = policy_of::<SuffixCfg>();
    let num_claims = if level == 0 { key.num_polynomials() } else { 1 };
    let fold_shape = EnvelopeCfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level,
        current_w_len: envelope_current_w_len,
    });
    expand_envelope_witness_at_ring_d(
        &envelope_gen,
        &suffix_policy,
        SuffixCfg::ring_challenge_config,
        level,
        envelope_current_w_len,
        suffix_ring_d,
        fold_shape,
        num_claims,
        extra_block_vars,
        None,
        None,
    )
}

/// Extra block-select variables when dropping ring dimension by a
/// power-of-two factor.
fn extra_block_vars_for_drop(prev_ring_d: usize, suffix_ring_d: usize) -> usize {
    if prev_ring_d > suffix_ring_d && prev_ring_d.is_multiple_of(suffix_ring_d) {
        let downscale = prev_ring_d / suffix_ring_d;
        if downscale.is_power_of_two() {
            return downscale.trailing_zeros() as usize;
        }
    }
    0
}

/// Hand-built mixed-D schedule for the runtime ring cutover acceptance
/// fixture.
///
/// Fold levels `[0, switch_at_fold)` use [`EnvelopeCfg`]'s shipped table
/// unchanged. Levels `[switch_at_fold, …)` keep the envelope witness-length
/// chain while dropping the per-level ring dimension to [`SuffixCfg`]'s
/// (typically `128 → 64`): block counts and Ajtai ranks are recomputed for
/// the suffix geometry so `commit_next_w` dispatch at the suffix `D` matches
/// the envelope field-length ladder.
///
/// The fixture supports `num_polynomials == 1` only.
///
/// # Errors
///
/// Returns an error when either preset schedule cannot be resolved, the
/// suffix table has fewer fold levels than the envelope, or `switch_at_fold`
/// is out of range.
pub(super) fn mixed_d_per_level_schedule<EnvelopeCfg, SuffixCfg>(
    num_vars: usize,
    num_polynomials: usize,
    switch_at_fold: usize,
) -> Result<Schedule, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    if num_polynomials != 1 {
        return Err(AkitaError::InvalidSetup(format!(
            "mixed-D fixture supports singleton batches only (got {num_polynomials})"
        )));
    }
    let lookup_key =
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polynomials));
    let envelope = EnvelopeCfg::runtime_schedule(lookup_key.clone())?;
    let suffix = SuffixCfg::runtime_schedule(lookup_key.clone())?;

    let envelope_folds: Vec<FoldStep> = envelope.fold_steps().cloned().collect();
    let suffix_folds: Vec<FoldStep> = suffix.fold_steps().cloned().collect();
    if envelope_folds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "mixed-D fixture requires a folded schedule".into(),
        ));
    }
    if suffix_folds.len() < envelope_folds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "suffix schedule has fewer fold levels than the envelope: {} < {}",
            suffix_folds.len(),
            envelope_folds.len()
        )));
    }
    if switch_at_fold > envelope_folds.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "switch_at_fold={switch_at_fold} exceeds fold count {}",
            envelope_folds.len()
        )));
    }
    if switch_at_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "switch_at_fold=0 is unsupported; use switch_at_fold >= 1 for mixed-D fixtures".into(),
        ));
    }

    let mut mixed_folds: Vec<FoldStep> = envelope_folds
        .iter()
        .take(switch_at_fold)
        .cloned()
        .collect();

    if switch_at_fold < envelope_folds.len() {
        let suffix_policy = policy_of::<SuffixCfg>();
        let field_bits = suffix_policy.decomposition.field_bits();
        let challenge_field_bits = field_bits * suffix_policy.chal_ext_degree as u32;
        let suffix_ring_d = suffix_folds[switch_at_fold].params.ring_dimension;
        let num_fold_levels = envelope_folds.len();
        let mut w_len = envelope_folds[switch_at_fold - 1].next_w_len;
        let mut prev_ring_d = envelope_folds[switch_at_fold - 1].params.ring_dimension;
        let mut suffix_plan: Vec<MixedSuffixFoldPlan> = Vec::new();

        for (level, envelope_step) in envelope_folds.iter().enumerate().skip(switch_at_fold) {
            let params = if level == switch_at_fold {
                mixed_level_params::<EnvelopeCfg, SuffixCfg>(
                    lookup_key.final_group,
                    level,
                    w_len,
                    &envelope_step.params,
                    suffix_ring_d,
                    prev_ring_d,
                )?
            } else {
                let prev = mixed_folds
                    .last()
                    .expect("mixed suffix fold chain must be nonempty");
                mixed_continue_suffix_level::<EnvelopeCfg, SuffixCfg>(
                    lookup_key.final_group,
                    level,
                    w_len,
                    suffix_ring_d,
                    prev.params.position_index_bits(),
                    prev.params.block_index_bits(),
                )?
            };
            let is_terminal_fold = level + 1 == num_fold_levels;
            let next_w_len = if is_terminal_fold {
                segment_typed_witness_shape_from_groups(
                    &params,
                    field_bits,
                    [(&params as &dyn LevelParamsLike, 1, 1, 1)],
                )?
                .logical_num_elems()
            } else {
                intermediate_w_ring_element_count_with_counts_bits(field_bits, &params, 1, 1)?
                    .checked_mul(params.ring_dimension)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("mixed-D witness length overflow".into())
                    })?
            };
            suffix_plan.push(MixedSuffixFoldPlan {
                params,
                current_w_len: w_len,
                next_w_len,
                is_terminal: is_terminal_fold,
            });
            w_len = next_w_len;
            prev_ring_d = suffix_ring_d;
        }

        for (idx, plan) in suffix_plan.iter().enumerate() {
            let layout = if plan.is_terminal {
                RelationMatrixRowLayout::WithoutCommitmentBlocks
            } else {
                RelationMatrixRowLayout::WithDBlock
            };
            let next_lp = if plan.is_terminal {
                None
            } else {
                Some(&suffix_plan[idx + 1].params)
            };
            let level_bytes = level_proof_bytes(
                field_bits,
                challenge_field_bits,
                &plan.params,
                next_lp,
                plan.next_w_len,
                layout,
                if plan.is_terminal {
                    None
                } else if suffix_plan[idx + 1].is_terminal {
                    Some(akita_types::NextWitnessBindingPolicy::TerminalInnerState)
                } else {
                    Some(akita_types::NextWitnessBindingPolicy::OuterCommitment)
                },
            );
            mixed_folds.push(FoldStep {
                params: plan.params.clone(),
                current_w_len: plan.current_w_len,
                next_w_len: plan.next_w_len,
                level_bytes,
            });
        }
    }

    let envelope_terminal = &envelope.terminal;
    let needs_terminal_override = mixed_folds
        .iter()
        .zip(envelope_folds.iter())
        .any(|(mixed, envelope)| mixed.params.ring_dimension != envelope.params.ring_dimension);

    let terminal_current_w_len = if needs_terminal_override {
        mixed_folds
            .last()
            .map(|fold| fold.next_w_len)
            .unwrap_or(envelope_terminal.current_w_len)
    } else {
        envelope_terminal.current_w_len
    };

    let terminal = if needs_terminal_override {
        let terminal_fold = mixed_folds.last().expect("mixed folds");
        let terminal_lp = &terminal_fold.params;
        let terminal_fold_level = mixed_folds.len() - 1;
        let field_bits = SuffixCfg::decomposition().field_bits();
        let terminal_num_polynomials = if terminal_fold_level == 0 {
            lookup_key.final_group.num_polynomials()
        } else {
            1
        };
        let witness_shape = segment_typed_witness_shape_from_groups(
            terminal_lp,
            field_bits,
            [(
                terminal_lp as &dyn akita_types::LevelParamsLike,
                terminal_num_polynomials,
                terminal_num_polynomials,
                1,
            )],
        )?;
        let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
        let current_w_len = witness_shape.logical_num_elems();
        if let Some(terminal_fold) = mixed_folds.last_mut() {
            let challenge_field_bits = field_bits * policy_of::<SuffixCfg>().chal_ext_degree as u32;
            terminal_fold.next_w_len = current_w_len;
            terminal_fold.level_bytes = level_proof_bytes(
                field_bits,
                challenge_field_bits,
                &terminal_fold.params,
                None,
                current_w_len,
                RelationMatrixRowLayout::WithoutCommitmentBlocks,
                None,
            );
        }
        DirectStep {
            current_w_len,
            witness_shape,
            direct_bytes,
        }
    } else {
        DirectStep {
            current_w_len: terminal_current_w_len,
            witness_shape: envelope_terminal.witness_shape.clone(),
            direct_bytes: envelope_terminal.direct_bytes,
        }
    };

    let total_bytes = mixed_folds
        .iter()
        .map(|fold| fold.level_bytes)
        .try_fold(0usize, |acc, bytes| {
            acc.checked_add(bytes)
                .ok_or_else(|| AkitaError::InvalidSetup("mixed-D total_bytes overflow".into()))
        })?
        .checked_add(terminal.direct_bytes)
        .ok_or_else(|| AkitaError::InvalidSetup("mixed-D total_bytes overflow".into()))?;

    Ok(Schedule {
        folds: mixed_folds,
        terminal,
        total_bytes,
    })
}
