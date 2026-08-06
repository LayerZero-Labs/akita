use super::*;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SetupPrefixSearchKey {
    policy_digest: [u8; 32],
    ring_challenge: SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    outer_ring_dimension: usize,
}

static SETUP_PREFIX_SEARCH_CACHE: LazyLock<
    Mutex<HashMap<SetupPrefixSearchKey, Vec<PrecommittedLevelParams>>>,
> = LazyLock::new(|| Mutex::new(HashMap::new()));

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

#[allow(clippy::too_many_arguments)]
pub(in crate::schedule_params) fn derive_setup_prefix_groups(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    outer_ring_dimension: usize,
) -> Result<Vec<PrecommittedLevelParams>, AkitaError> {
    let cache_key = SetupPrefixSearchKey {
        policy_digest: akita_schedules::policy_digest(policy),
        ring_challenge: *ring_challenge_cfg,
        fold_shape: requested_fold_shape,
        log_basis_open,
        n_prefix,
        num_chunks,
        outer_ring_dimension,
    };
    if let Some(cached) = SETUP_PREFIX_SEARCH_CACHE
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("setup-prefix search cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(cached);
    }
    if outer_ring_dimension == 0
        || !outer_ring_dimension.is_power_of_two()
        || !policy
            .setup_prefix_inner_ring_dimension
            .is_multiple_of(outer_ring_dimension)
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
    if !n_prefix.is_multiple_of(policy.setup_prefix_inner_ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a multiple of the ring dimension".to_string(),
        ));
    }
    let ring_slots = n_prefix / policy.setup_prefix_inner_ring_dimension;
    let reduced_vars = checked_power_of_two_vars(ring_slots, "setup prefix ring slots")?;
    let prefix_num_vars = checked_power_of_two_vars(n_prefix, "setup prefix field length")?;
    let family = policy.sis_modulus_profile;
    let d = policy.setup_prefix_inner_ring_dimension;
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_outer = num_digits_open(open_decomp);
    let num_digits_open_val = num_digits_open(open_decomp);
    let mut frontier: Vec<(usize, usize, LayoutCandidateScore, PrecommittedLevelParams)> =
        Vec::new();

    let (inner_basis_min, inner_basis_max) = policy.inner_basis_search_range();
    for log_basis_inner in inner_basis_min..=inner_basis_max {
        let inner_decomp = DecompositionParams {
            log_basis: log_basis_inner,
            ..policy.decomposition
        };
        let num_digits_inner =
            num_digits_inner_for_bound(inner_decomp, policy.decomposition.field_bits());
        for block_index_bits in (0..=reduced_vars).rev() {
            let Some(num_live_blocks) = 1usize.checked_shl(block_index_bits as u32) else {
                continue;
            };
            let position_index_bits = reduced_vars - block_index_bits;
            let Some(num_positions_per_block) = 1usize.checked_shl(position_index_bits as u32)
            else {
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
            let Some(num_fold_coeffs) = width_s
                .checked_mul(d)
                .and_then(|count| count.checked_mul(num_chunks))
            else {
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
                num_chunks,
                num_fold_coeffs,
                witness_norms: FoldWitnessNorms::bounded(log_basis_inner, d),
                log_basis_response: log_basis_open,
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
                    policy.setup_prefix_inner_ring_dimension,
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
                log_basis_outer: log_basis_open,
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
            let setup_fields = akita_types::setup_prefix_slot_field_elements(
                &akita_types::setup_prefix_slot_id(n_prefix, params.clone()),
            )?;
            let padded_setup_fields = padded_setup_prefix_len(setup_fields);
            if frontier.iter().any(|(best_witness, best_setup, _, _)| {
                *best_witness <= physical_width && *best_setup <= padded_setup_fields
            }) {
                continue;
            }
            frontier.retain(|(other_witness, other_setup, _, _)| {
                physical_width > *other_witness || padded_setup_fields > *other_setup
            });
            frontier.push((physical_width, padded_setup_fields, score, params));
        }
    }

    frontier.sort_by_key(|(witness, setup, score, params)| {
        (
            *witness,
            *setup,
            *score,
            params.layout.log_basis_inner,
            params.layout.num_live_blocks,
        )
    });
    let result = frontier
        .into_iter()
        .map(|(_, _, _, params)| params)
        .collect::<Vec<_>>();
    SETUP_PREFIX_SEARCH_CACHE
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("setup-prefix search cache poisoned".into()))?
        .insert(cache_key, result.clone());
    Ok(result)
}
