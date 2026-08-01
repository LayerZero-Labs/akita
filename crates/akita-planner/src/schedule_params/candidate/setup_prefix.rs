use super::*;

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

#[allow(clippy::too_many_arguments)]
pub(in crate::schedule_params) fn derive_setup_prefix_group(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
    log_basis_outer: u32,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    outer_ring_dimension: usize,
) -> Result<Option<PrecommittedLevelParams>, AkitaError> {
    if outer_ring_dimension == 0
        || !outer_ring_dimension.is_power_of_two()
        || !policy.ring_dimension.is_multiple_of(outer_ring_dimension)
    {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix B dimension must be a power-of-two divisor of its A dimension"
                .to_string(),
        ));
    }
    if n_prefix == 0 || !n_prefix.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a nonzero power of two".to_string(),
        ));
    }
    if !n_prefix.is_multiple_of(policy.ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a multiple of the ring dimension".to_string(),
        ));
    }
    if log_basis_outer != log_basis_open {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix checkpoint requires one consuming inner/outer/open basis".to_string(),
        ));
    }
    let ring_slots = n_prefix / policy.ring_dimension;
    let reduced_vars = checked_power_of_two_vars(ring_slots, "setup prefix ring slots")?;
    let prefix_num_vars = checked_power_of_two_vars(n_prefix, "setup prefix field length")?;
    let family = policy.sis_modulus_profile;
    let d = policy.ring_dimension;
    let outer_decomp = DecompositionParams {
        log_basis: log_basis_outer,
        ..policy.decomposition
    };
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_outer = num_digits_open(outer_decomp);
    let num_digits_open_val = num_digits_open(open_decomp);
    let mut best: Option<(LayoutCandidateScore, PrecommittedLevelParams)> = None;

    // The current protocol has one Stage-1 range polynomial. Until role-specific
    // range proofs exist, setup-prefix source, commitment, and opening digits use
    // the consuming fold's certified basis and only block geometry is searched.
    let log_basis_inner = log_basis_open;
    let inner_decomp = DecompositionParams {
        log_basis: log_basis_inner,
        ..policy.decomposition
    };
    let num_digits_inner = num_digits_setup_prefix_commit(inner_decomp);
    for block_index_bits in (0..=reduced_vars).rev() {
        let Some(num_live_blocks) = 1usize.checked_shl(block_index_bits as u32) else {
            continue;
        };
        let position_index_bits = reduced_vars - block_index_bits;
        let Some(num_positions_per_block) = 1usize.checked_shl(position_index_bits as u32) else {
            continue;
        };
        let fold_shape = optimize_fold_challenge_shape(requested_fold_shape, num_live_blocks)?;
        if num_live_blocks < num_chunks {
            continue;
        }
        let Some(width_s) =
            decomposed_s_block_ring_count(num_positions_per_block, num_digits_inner)
        else {
            continue;
        };
        let Some(num_fold_coeffs) = width_s.checked_mul(d) else {
            continue;
        };
        let fold_policy = BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
            policy.decomposition.field_bits(),
            FoldWitnessNorms::bounded(inner_decomp.log_basis, d),
        );
        let Ok(num_digits_fold) = fold_policy.num_digits_fold(HonestFoldSizingQuery {
            ring_dimension: d,
            num_claims: 1,
            num_live_blocks,
            num_fold_coeffs,
            log_basis: log_basis_open,
            challenge_config: ring_challenge_cfg,
            challenge_shape: fold_shape,
        }) else {
            continue;
        };
        let Some(norm_s) = rounded_up_role_a_inf_norm(
            policy.sis_security_policy,
            policy.sis_table_digest,
            family,
            d,
            log_basis_open,
            ring_challenge_cfg,
            fold_shape,
            num_digits_fold,
            policy.ring_subfield_norm_bound,
        ) else {
            continue;
        };
        let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
            sis_key_at_dimension(
                policy,
                akita_types::SisMatrixRole::Inner,
                policy.ring_dimension,
                norm_s,
            ),
            width_s,
        ) else {
            continue;
        };
        let Some(norm_t) = rounded_up_collision_inf_norm(
            policy.sis_security_policy,
            family,
            akita_types::SisMatrixRole::Outer,
            outer_ring_dimension,
            log_basis_open,
        ) else {
            continue;
        };
        let Some(width_t) = decomposed_t_ring_count(
            inner_commit_matrix.output_rank(),
            num_digits_outer,
            num_live_blocks,
            1,
        )
        .and_then(|width| width.checked_mul(d / outer_ring_dimension)) else {
            continue;
        };
        let Ok(outer_commit_matrix) = OuterCommitMatrixParams::try_new_with_min_rank(
            sis_key_at_dimension(
                policy,
                akita_types::SisMatrixRole::Outer,
                outer_ring_dimension,
                norm_t,
            ),
            width_t,
        ) else {
            continue;
        };
        let layout = CommittedGroupProfile {
            version: CommittedGroupProfile::VERSION,
            group: PolynomialGroupLayout::singleton(prefix_num_vars),
            num_live_ring_elements_per_claim: ring_slots,
            num_positions_per_block,
            num_live_blocks,
            log_basis_inner,
            num_digits_inner,
            inner_commit_matrix,
            log_basis_outer,
            num_digits_outer,
            outer_commit_matrix,
        };
        let params = PrecommittedLevelParams {
            layout,
            log_basis_open,
            fold_challenge_config: *ring_challenge_cfg,
            num_digits_open: num_digits_open_val,
            num_digits_fold,
        };
        let physical_width = grouped_segment_rings(
            1,
            num_live_blocks,
            num_chunks,
            num_positions_per_block,
            params.layout.inner_commit_matrix.output_rank(),
            num_digits_inner,
            num_digits_outer,
            num_digits_open_val,
            num_digits_fold,
        )?;
        let score =
            layout_candidate_score(physical_width, num_live_blocks, num_chunks, fold_shape)?;
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, params));
        }
    }

    Ok(best.map(|(_, params)| params))
}
