use super::*;
use akita_error::checked;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SetupPrefixSearchKey {
    opening_method: akita_types::OpeningMethod,
    ring_challenge: SparseChallengeConfig,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
    guide: Option<SetupPrefixLayoutGuide>,
}

#[derive(Default)]
pub(crate) struct SetupPrefixSearchCache {
    entries: HashMap<SetupPrefixSearchKey, Arc<[GroupOpenPhaseParams]>>,
    hits: usize,
    misses: usize,
}

impl SetupPrefixSearchCache {
    pub(crate) const fn diagnostics(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }
}

pub(crate) struct SetupPrefixSearchRequest<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) opening: PlannerOpeningCandidate,
    pub(crate) log_basis_open: u32,
    pub(crate) n_prefix: usize,
    pub(crate) num_chunks: usize,
    pub(crate) inner_ring_dimension: usize,
    pub(crate) outer_ring_dimension: usize,
    pub(crate) guide: Option<SetupPrefixLayoutGuide>,
}

type SetupPrefixFrontierEntry = (
    [usize; 2],
    Vec<u8>,
    LayoutCandidateScore,
    GroupOpenPhaseParams,
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
    opening: PlannerOpeningCandidate,
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
    fn derive_inner(
        &self,
        split: SetupPrefixSplit,
    ) -> Result<Option<InnerCommitmentCandidate>, AkitaError> {
        let d_a = self.dimensions.d_a();
        let fold_policy =
            BalancedSignedDigitFoldPolicy::universal(self.policy.decomposition.field_bits());
        let Some(num_fold_coeffs) = split
            .width_s
            .checked_mul(d_a)
            .and_then(|count| count.checked_mul(self.num_chunks))
        else {
            return Ok(None);
        };
        let prefix_moment = crate::response_model::uniform_field_source_moment(
            self.n_prefix,
            self.policy.decomposition.field_bits(),
            split.log_basis_inner,
            split.num_digits_inner,
        )?;
        let modeled_linf_cap = prefix_moment.response_linf_cap(
            self.opening.challenge_config().challenge_l2_sq_max(),
            split.num_live_blocks,
            self.num_chunks,
            num_fold_coeffs,
            d_a,
        );
        derive_inner_commitment_candidate(InnerCommitmentCandidateRequest {
            policy: self.policy,
            fold_policy: &fold_policy,
            ring_challenge_cfg: &self.opening.challenge_config(),
            challenge_dimension: self.opening.challenge_dimension(d_a),
            dimensions: self.dimensions,
            num_claims: 1,
            num_live_ring_elements_per_claim: self.ring_slots,
            num_positions_per_block: split.num_positions_per_block,
            num_live_blocks: split.num_live_blocks,

            num_chunks: self.num_chunks,
            witness_norms: FoldWitnessNorms::bounded(split.log_basis_inner, d_a),
            log_basis_open: self.log_basis_open,
            width_s: split.width_s,
            modeled_linf_cap,
        })
    }

    fn derive_for_slice(
        &self,
        split: SetupPrefixSplit,
        inner_candidate: &InnerCommitmentCandidate,
        outer_slice_count: akita_types::CommitmentSliceCount,
    ) -> Result<Option<SetupPrefixFrontierEntry>, AkitaError> {
        let Some(outer_commit_matrix) =
            derive_outer_commitment_candidate(OuterCommitmentCandidateRequest {
                policy: self.policy,
                dimensions: self.dimensions,
                payload_mode: akita_types::CommitmentPayloadMode::Compressed,
                num_claims: 1,
                num_live_blocks: split.num_live_blocks,
                outer_slice_count,
                log_basis_open: self.log_basis_open,
                num_digits_outer: self.num_digits_outer,
                inner_output_rank: inner_candidate.inner_commit_matrix.output_rank(),
            })?
        else {
            return Ok(None);
        };
        let profile = GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::singleton(self.prefix_num_vars),

            blocks: akita_types::BlockGeometry::new(
                self.ring_slots,
                split.num_positions_per_block,
                split.num_live_blocks,
            ),

            outer_slice_count,
            inner: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(split.log_basis_inner, split.num_digits_inner),
                inner_candidate.inner_commit_matrix,
            ),
            outer: akita_types::RoleParams::new(
                akita_types::GadgetDigits::new(self.log_basis_open, self.num_digits_outer),
                outer_commit_matrix,
            ),
        };
        let params = GroupOpenPhaseParams {
            setup_natural_len: None,
            profile,
            opening: akita_types::GroupOpeningPlan {
                opening_method: self.opening.method(),
                fold_challenge_config: self.opening.challenge_config(),
                log_basis_open: self.log_basis_open,
                num_digits_open: self.num_digits_open,
                num_digits_fold: inner_candidate.num_digits_fold,
            },
        };
        let physical_width = akita_types::grouped_witness_body_coefficients(
            &params,
            // A setup prefix is a frozen standalone commitment, so canonical by
            // admission.
            akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
            self.dimensions,
            self.policy.claim_ext_degree,
            1,
            self.num_chunks,
        )?;
        let score = layout_candidate_score(physical_width, split.num_live_blocks, self.num_chunks)?;
        let setup_fields = akita_types::setup_prefix_slot_field_elements(
            &akita_types::scheduled_setup_prefix(self.n_prefix, params)
                .slot_id()
                .expect("setup prefix group"),
        )?;
        let coords = [physical_width, padded_setup_prefix_len(setup_fields)];
        let descriptor = params.canonical_descriptor_bytes();
        Ok(Some((coords, descriptor, score, params)))
    }
}

fn setup_prefix_slice_counts(
    num_live_blocks: usize,
) -> impl Iterator<Item = akita_types::CommitmentSliceCount> {
    akita_types::CommitmentSliceCount::ALL
        .into_iter()
        .filter(move |&count| {
            count
                .validate_for_commitment(
                    0,
                    akita_types::CommitmentPayloadMode::Compressed,
                    num_live_blocks,
                )
                .is_ok()
        })
}

pub(in crate::schedule_params) fn derive_setup_prefix_groups(
    cache: &mut SetupPrefixSearchCache,
    request: SetupPrefixSearchRequest<'_>,
) -> Result<Vec<GroupOpenPhaseParams>, AkitaError> {
    let SetupPrefixSearchRequest {
        policy,
        opening,
        log_basis_open,
        n_prefix,
        num_chunks,
        inner_ring_dimension,
        outer_ring_dimension,
        guide,
    } = request;
    let cache_key = SetupPrefixSearchKey {
        opening_method: opening.method(),
        ring_challenge: opening.challenge_config(),
        log_basis_open,
        n_prefix,
        num_chunks,
        inner_ring_dimension,
        outer_ring_dimension,
        guide,
    };
    if let Some(cached) = cache.entries.get(&cache_key) {
        cache.hits = cache.hits.saturating_add(1);
        return Ok(cached.to_vec());
    }
    cache.misses = cache.misses.saturating_add(1);
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
    let reduced_vars = checked::ceil_log2(ring_slots).ok_or_else(|| {
        AkitaError::InvalidSetup("setup prefix ring slots are zero or too large".into())
    })?;
    let prefix_num_vars = checked::ceil_log2(n_prefix).ok_or_else(|| {
        AkitaError::InvalidSetup("setup prefix field length is zero or too large".into())
    })?;
    let open_decomp = DecompositionParams {
        log_basis: log_basis_open,
        ..policy.decomposition
    };
    let num_digits_outer = num_digits_open(open_decomp);
    let num_digits_open_val = num_digits_open(open_decomp);
    let mut frontier = Vec::<SetupPrefixFrontierEntry>::new();
    let candidate_context = SetupPrefixCandidateContext {
        policy,
        opening,
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
        if guide.is_some_and(|guide| log_basis_inner != guide.log_basis_inner) {
            continue;
        }
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
            if guide.is_some_and(|guide| position_index_bits != guide.position_index_bits) {
                continue;
            }
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
            let Some(inner_candidate) = candidate_context.derive_inner(split)? else {
                continue;
            };
            for outer_slice_count in setup_prefix_slice_counts(num_live_blocks) {
                if guide.is_some_and(|guide| outer_slice_count != guide.outer_slice_count) {
                    continue;
                }
                let Some(entry) = candidate_context.derive_for_slice(
                    split,
                    &inner_candidate,
                    outer_slice_count,
                )?
                else {
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
            params.profile.inner.digits.log_basis,
            params.profile.blocks.live_blocks,
        )
    });
    let result: Arc<[GroupOpenPhaseParams]> = frontier
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
    fn setup_prefix_slicing_uses_standalone_precommitment_eligibility() {
        assert_eq!(
            setup_prefix_slice_counts(8)
                .map(akita_types::CommitmentSliceCount::get)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );
        assert_eq!(
            setup_prefix_slice_counts(3)
                .map(akita_types::CommitmentSliceCount::get)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn guided_setup_prefix_survives_pareto_pruning() {
        use akita_config::{
            policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
        };

        type Recursive = RecursiveCommitmentConfig<OneHot>;
        let policy = policy_of::<Recursive>();
        let n_prefix = 1usize << 14;
        let opening = PlannerOpeningCandidate::evaluation_trace(
            Recursive::ring_challenge_config(64).expect("challenge config"),
        );
        let mut cache = SetupPrefixSearchCache::default();
        let request = |guide| SetupPrefixSearchRequest {
            policy: &policy,
            opening,
            log_basis_open: 3,
            n_prefix,
            num_chunks: 1,
            inner_ring_dimension: 64,
            outer_ring_dimension: 64,
            guide,
        };
        let unguided = derive_setup_prefix_groups(&mut cache, request(None))
            .expect("unguided setup-prefix frontier");
        let unguided_layouts = unguided
            .iter()
            .map(|candidate| {
                (
                    candidate.profile.inner.digits.log_basis,
                    candidate.profile.blocks.position_index_bits(),
                    candidate.profile.outer_slice_count,
                )
            })
            .collect::<std::collections::HashSet<_>>();
        let reduced_vars = (n_prefix / 64).trailing_zeros() as usize;
        let (min_basis, max_basis) = crate::InnerBasisSource::RawCoefficients {
            log_bound: policy.decomposition.field_bits(),
        }
        .search_range(&policy)
        .expect("setup-prefix basis range");

        for log_basis_inner in min_basis..=max_basis {
            for position_index_bits in 0..=reduced_vars {
                for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
                    let guide = SetupPrefixLayoutGuide {
                        log_basis_inner,
                        position_index_bits,
                        outer_slice_count,
                    };
                    let guided = derive_setup_prefix_groups(&mut cache, request(Some(guide)))
                        .expect("guided setup-prefix candidate");
                    if !guided.is_empty()
                        && !unguided_layouts.contains(&(
                            log_basis_inner,
                            position_index_bits,
                            outer_slice_count,
                        ))
                    {
                        assert!(guided.iter().all(|candidate| {
                            candidate.profile.inner.digits.log_basis == log_basis_inner
                                && candidate.profile.blocks.position_index_bits()
                                    == position_index_bits
                                && candidate.profile.outer_slice_count == outer_slice_count
                        }));
                        return;
                    }
                }
            }
        }
        panic!("fixture must contain a feasible setup-prefix layout outside the Pareto frontier");
    }
}
