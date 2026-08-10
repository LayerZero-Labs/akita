use super::*;
use akita_types::CompressionChainPlan;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SetupPrefixSearchKey {
    ring_challenge: SparseChallengeConfig,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
}

#[derive(Default)]
pub(crate) struct SetupPrefixSearchCache {
    entries: HashMap<SetupPrefixSearchKey, Arc<[PrecommittedLevelParams]>>,
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

#[allow(clippy::too_many_arguments)]
pub(in crate::schedule_params) fn derive_setup_prefix_groups(
    cache: &mut SetupPrefixSearchCache,
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
) -> Result<Vec<PrecommittedLevelParams>, AkitaError> {
    let cache_key = SetupPrefixSearchKey {
        ring_challenge: *ring_challenge_cfg,
        log_basis_open,
        n_prefix,
        num_chunks,
        inner_ring_dimension,
        outer_ring_dimension,
    };
    if let Some(cached) = cache.entries.get(&cache_key) {
        return Ok(cached.to_vec());
    }
    if outer_ring_dimension == 0
        || !outer_ring_dimension.is_power_of_two()
        || !inner_ring_dimension.is_multiple_of(outer_ring_dimension)
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
    if !n_prefix.is_multiple_of(inner_ring_dimension) {
        return Err(AkitaError::InvalidSetup(
            "setup prefix length must be a multiple of the ring dimension".to_string(),
        ));
    }
    let ring_slots = n_prefix / inner_ring_dimension;
    let reduced_vars = checked_power_of_two_vars(ring_slots, "setup prefix ring slots")?;
    let prefix_num_vars = checked_power_of_two_vars(n_prefix, "setup prefix field length")?;
    let family = policy.sis_modulus_profile;
    let d = inner_ring_dimension;
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_outer = num_digits_open(open_decomp);
    let num_digits_open_val = num_digits_open(open_decomp);
    let mut frontier: Vec<(
        [usize; 2],
        Vec<u8>,
        LayoutCandidateScore,
        PrecommittedLevelParams,
    )> = Vec::new();

    let (inner_basis_min, inner_basis_max) = crate::InnerBasisSource::RawCoefficients {
        log_bound: policy.decomposition.field_bits(),
    }
    .search_range(policy)?;
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
                num_digits_fold,
                policy.ring_subfield_norm_bound,
            ) else {
                continue;
            };
            let Ok(inner_commit_matrix) = InnerCommitMatrixParams::try_new_with_min_rank(
                sis_key_at_dimension(
                    policy,
                    akita_types::SisMatrixRole::Inner,
                    inner_ring_dimension,
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
            let physical_width = akita_schedules::planner_support::grouped_segment_rings(
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
            let score = layout_candidate_score(physical_width, num_live_blocks, num_chunks)?;
            let compression_source_coefficients = params
                .layout
                .outer_commit_matrix
                .output_rank()
                .checked_mul(params.layout.outer_commit_matrix.ring_dimension())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "setup-prefix outer compression source overflow".into(),
                    )
                })?;
            if CompressionChainPlan::try_for_complete_source(
                params.layout.outer_commit_matrix.sis_modulus_profile(),
                compression_source_coefficients,
            )?
            .is_none()
            {
                continue;
            }
            let setup_fields = akita_types::setup_prefix_slot_field_elements(
                &akita_types::setup_prefix_slot_id(n_prefix, params.clone()),
            )?;
            let padded_setup_fields = padded_setup_prefix_len(setup_fields);
            let coords = [physical_width, padded_setup_fields];
            let descriptor = params.canonical_descriptor_bytes();
            crate::schedule_params::pareto::insert(
                &mut frontier,
                (coords, descriptor, score, params),
                |(best, best_descriptor, best_score, _),
                 (candidate, candidate_descriptor, candidate_score, _)| {
                    let best_tie = (*best_score, best_descriptor.as_slice());
                    let candidate_tie = (*candidate_score, candidate_descriptor.as_slice());
                    crate::schedule_params::pareto::canonical_dominates(
                        best,
                        &best_tie,
                        candidate,
                        &candidate_tie,
                    )
                },
            );
        }
    }

    frontier.sort_by_key(|(coords, _, score, params)| {
        (
            coords[0],
            coords[1],
            *score,
            params.layout.log_basis_inner,
            params.layout.num_live_blocks,
        )
    });
    let result: Arc<[PrecommittedLevelParams]> = frontier
        .into_iter()
        .map(|(_, _, _, params)| params)
        .collect();
    cache.entries.insert(cache_key, Arc::clone(&result));
    Ok(result.to_vec())
}
