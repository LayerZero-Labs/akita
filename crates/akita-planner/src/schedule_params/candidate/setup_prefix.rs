use super::*;
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
    producer_fold_level: usize,
}

#[derive(Default)]
pub(crate) struct SetupPrefixSearchCache {
    entries: HashMap<SetupPrefixSearchKey, Arc<[PrecommittedLevelParams]>>,
}

pub(crate) struct SetupPrefixSearchRequest<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) ring_challenge_cfg: &'a SparseChallengeConfig,
    pub(crate) log_basis_open: u32,
    pub(crate) n_prefix: usize,
    pub(crate) num_chunks: usize,
    pub(crate) inner_ring_dimension: usize,
    pub(crate) outer_ring_dimension: usize,
    pub(crate) producer_fold_level: usize,
}

type SetupPrefixFrontierEntry = (
    [usize; 2],
    Vec<u8>,
    LayoutCandidateScore,
    PrecommittedLevelParams,
);

#[derive(Clone, Copy)]
struct SetupPrefixSplit {
    log_basis_inner: u32,
    num_digits_inner: usize,
    num_live_blocks: usize,
    num_positions_per_block: usize,
    width_s: usize,
}

struct SetupPrefixCandidateContext<'a> {
    policy: &'a PlannerPolicy,
    ring_challenge_cfg: &'a SparseChallengeConfig,
    dimensions: CommitmentRingDims,
    log_basis_open: u32,
    n_prefix: usize,
    prefix_num_vars: usize,
    ring_slots: usize,
    num_chunks: usize,
    num_digits_outer: usize,
    num_digits_open: usize,
}

impl SetupPrefixCandidateContext<'_> {
    fn derive(
        &self,
        split: SetupPrefixSplit,
        outer_slice_count: akita_types::CommitmentSliceCount,
    ) -> Result<Option<SetupPrefixFrontierEntry>, AkitaError> {
        let d_a = self.dimensions.d_a();
        let fold_policy = BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
            self.policy.decomposition.field_bits(),
            FoldWitnessNorms::bounded(split.log_basis_inner, d_a),
        );
        let Some(ab_candidate) = derive_ab_commitment_candidate(AbCommitmentCandidateRequest {
            policy: self.policy,
            fold_policy: &fold_policy,
            ring_challenge_cfg: self.ring_challenge_cfg,
            dimensions: self.dimensions,
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            num_claims: 1,
            num_live_blocks: split.num_live_blocks,
            num_chunks: self.num_chunks,
            outer_slice_count,
            witness_norms: FoldWitnessNorms::bounded(split.log_basis_inner, d_a),
            log_basis_open: self.log_basis_open,
            width_s: split.width_s,
            num_digits_outer: self.num_digits_outer,
        })?
        else {
            return Ok(None);
        };
        let layout = CommittedGroupProfile {
            version: CommittedGroupProfile::VERSION,
            group: PolynomialGroupLayout::singleton(self.prefix_num_vars),
            num_live_ring_elements_per_claim: self.ring_slots,
            num_positions_per_block: split.num_positions_per_block,
            num_live_blocks: split.num_live_blocks,
            outer_slice_count,
            log_basis_inner: split.log_basis_inner,
            num_digits_inner: split.num_digits_inner,
            inner_commit_matrix: ab_candidate.inner_commit_matrix,
            log_basis_outer: self.log_basis_open,
            num_digits_outer: self.num_digits_outer,
            outer_commit_matrix: ab_candidate.outer_commit_matrix,
        };
        let params = PrecommittedLevelParams {
            layout,
            log_basis_open: self.log_basis_open,
            fold_challenge_config: *self.ring_challenge_cfg,
            num_digits_open: self.num_digits_open,
            num_digits_fold: ab_candidate.num_digits_fold,
        };
        let physical_width = akita_schedules::planner_support::grouped_segment_rings(
            1,
            split.num_live_blocks,
            self.num_chunks,
            split.num_positions_per_block,
            params.layout.inner_commit_matrix.output_rank(),
            split.num_digits_inner,
            self.num_digits_outer,
            self.num_digits_open,
            params.num_digits_fold,
        )?;
        let score = layout_candidate_score(physical_width, split.num_live_blocks, self.num_chunks)?;
        let setup_fields = akita_types::setup_prefix_slot_field_elements(
            &akita_types::setup_prefix_slot_id(self.n_prefix, params.clone()),
        )?;
        let coords = [physical_width, padded_setup_prefix_len(setup_fields)];
        let descriptor = params.canonical_descriptor_bytes();
        Ok(Some((coords, descriptor, score, params)))
    }
}

fn setup_prefix_slice_counts(
    producer_fold_level: usize,
    num_live_blocks: usize,
) -> impl Iterator<Item = akita_types::CommitmentSliceCount> {
    akita_types::CommitmentSliceCount::ALL
        .into_iter()
        .filter(move |&count| {
            count
                .validate_for_commitment(
                    producer_fold_level,
                    akita_types::CommitmentPayloadMode::Compressed,
                    num_live_blocks,
                )
                .is_ok()
        })
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

pub(in crate::schedule_params) fn derive_setup_prefix_groups(
    cache: &mut SetupPrefixSearchCache,
    request: SetupPrefixSearchRequest<'_>,
) -> Result<Vec<PrecommittedLevelParams>, AkitaError> {
    let SetupPrefixSearchRequest {
        policy,
        ring_challenge_cfg,
        log_basis_open,
        n_prefix,
        num_chunks,
        inner_ring_dimension,
        outer_ring_dimension,
        producer_fold_level,
    } = request;
    let cache_key = SetupPrefixSearchKey {
        ring_challenge: *ring_challenge_cfg,
        log_basis_open,
        n_prefix,
        num_chunks,
        inner_ring_dimension,
        outer_ring_dimension,
        producer_fold_level,
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
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_outer = num_digits_open(open_decomp);
    let num_digits_open_val = num_digits_open(open_decomp);
    let mut frontier = Vec::<SetupPrefixFrontierEntry>::new();
    let candidate_context = SetupPrefixCandidateContext {
        policy,
        ring_challenge_cfg,
        dimensions: CommitmentRingDims {
            inner: inner_ring_dimension,
            outer: outer_ring_dimension,
            opening: outer_ring_dimension,
        },
        log_basis_open,
        n_prefix,
        prefix_num_vars,
        ring_slots,
        num_chunks,
        num_digits_outer,
        num_digits_open: num_digits_open_val,
    };

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
            let split = SetupPrefixSplit {
                log_basis_inner,
                num_digits_inner,
                num_live_blocks,
                num_positions_per_block,
                width_s,
            };
            for outer_slice_count in setup_prefix_slice_counts(producer_fold_level, num_live_blocks)
            {
                let Some(entry) = candidate_context.derive(split, outer_slice_count)? else {
                    continue;
                };
                crate::schedule_params::pareto::insert(
                    &mut frontier,
                    entry,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_prefix_slicing_uses_the_absolute_producer_level() {
        assert_eq!(
            setup_prefix_slice_counts(1, 8)
                .map(akita_types::CommitmentSliceCount::get)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );
        assert_eq!(
            setup_prefix_slice_counts(2, 8)
                .map(akita_types::CommitmentSliceCount::get)
                .collect::<Vec<_>>(),
            vec![1]
        );
    }
}
