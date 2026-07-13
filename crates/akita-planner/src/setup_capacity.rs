//! Structural setup-capacity certification.
//!
//! The certificate ranges over polynomial counts and planner candidate
//! parameters, never over ordered multi-group layouts. Precommitted groups are
//! compressed into role summaries, then an unbounded total-polynomial DP
//! computes a complete role-wise upper bound for every capacity. This covers
//! arbitrary group counts in polynomial time.

use std::collections::HashMap;

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, LevelParams, PolynomialGroupLayout, PrecommittedGroupParams,
    SetupMatrixEnvelope,
};

use crate::group_batch::{
    group_root_params_candidate_from_layout, multi_group_root_main_level_params_candidate,
    MultiGroupRootCandidateCtx,
};
use crate::schedule_params::{
    compute_root_direct_level_params, recursive_fold_level_params_candidate,
    scalar_root_fold_level_params_candidate, validate_policy_witness_chunk, MAX_RECURSION_DEPTH,
};
use crate::PlannerPolicy;

/// One conservative single-group plan supplied by the config bridge.
#[derive(Clone, Debug)]
pub struct ConservativeGroupPlan {
    /// Polynomial shape committed by this plan.
    pub group: PolynomialGroupLayout,
    /// Canonical conservative commitment parameters for `group`.
    pub params: LevelParams,
}

/// Auditable counts for one structural setup-capacity computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupCapacityCertificate {
    /// Certified packed setup envelope.
    pub envelope: SetupMatrixEnvelope,
    /// Scalar root candidates included in the certificate.
    pub scalar_root_candidates: usize,
    /// Conservative group plans included in the total-capacity DP.
    pub conservative_group_plans: usize,
    /// Multi-group main-root candidates included in the certificate.
    pub multi_group_root_candidates: usize,
    /// Recursive suffix candidates included in the certificate.
    pub suffix_candidates: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct GroupRoleBound {
    max_a_len: usize,
    max_b_len: usize,
    summed_d_width: usize,
}

impl GroupRoleBound {
    fn add_group(self, group: GroupRoleBound) -> Result<Self, AkitaError> {
        Ok(Self {
            max_a_len: self.max_a_len.max(group.max_a_len),
            max_b_len: self.max_b_len.max(group.max_b_len),
            summed_d_width: self
                .summed_d_width
                .checked_add(group.summed_d_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "precommitted D setup-capacity DP overflow".to_string(),
                    )
                })?,
        })
    }

    fn rolewise_max(self, other: Self) -> Self {
        // The packed setup is max(A footprint, B footprint, D footprint).
        // A/B depend on one group's coordinate; D depends only on the summed
        // D width. Therefore componentwise maxima are a complete upper bound
        // even when the three extrema come from different partitions. Group
        // count and folded-witness size affect schedule feasibility, not these
        // matrix coordinates; recursive levels are certified separately from
        // the planner's mandatory strict-shrink inequality below.
        Self {
            max_a_len: self.max_a_len.max(other.max_a_len),
            max_b_len: self.max_b_len.max(other.max_b_len),
            summed_d_width: self.summed_d_width.max(other.summed_d_width),
        }
    }
}

fn checked_matrix_len(
    rows: usize,
    columns: usize,
    role: &'static str,
) -> Result<usize, AkitaError> {
    rows.checked_mul(columns).ok_or_else(|| {
        AkitaError::InvalidSetup(format!("precommitted {role} setup footprint overflow"))
    })
}

fn precommitted_role_bound(
    plan: &ConservativeGroupPlan,
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    fold_challenge_shape: TensorChallengeShape,
) -> Result<Option<GroupRoleBound>, AkitaError> {
    let frozen = PrecommittedGroupParams::from_params(plan.group, &plan.params);
    let Some(group) = group_root_params_candidate_from_layout(
        &frozen,
        policy,
        ring_challenge_cfg,
        fold_challenge_shape,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(GroupRoleBound {
        max_a_len: checked_matrix_len(group.a_key.row_len(), group.a_key.col_len(), "A")?,
        max_b_len: checked_matrix_len(group.b_key.row_len(), group.b_key.col_len(), "B")?,
        summed_d_width: group.d_segment_width()?,
    }))
}

fn precommitted_capacity_dp(
    max_num_batched_polys: usize,
    groups: &[(usize, GroupRoleBound)],
) -> Result<Vec<GroupRoleBound>, AkitaError> {
    let mut exact = vec![None; max_num_batched_polys + 1];
    exact[0] = Some(GroupRoleBound::default());
    for total in 1..=max_num_batched_polys {
        let mut bound: Option<GroupRoleBound> = None;
        for &(cost, group) in groups {
            if cost > total {
                continue;
            }
            let Some(previous) = exact[total - cost] else {
                continue;
            };
            let candidate = previous.add_group(group)?;
            bound = Some(bound.map_or(candidate, |current| current.rolewise_max(candidate)));
        }
        exact[total] = bound;
    }

    let mut at_most = vec![GroupRoleBound::default(); max_num_batched_polys + 1];
    for total in 1..=max_num_batched_polys {
        at_most[total] = exact[total]
            .unwrap_or_default()
            .rolewise_max(at_most[total - 1]);
    }
    Ok(at_most)
}

fn precommitted_bounds_for_shape(
    policy: &PlannerPolicy,
    max_num_batched_polys: usize,
    conservative_plans: &[ConservativeGroupPlan],
    ring_challenge_cfg: &SparseChallengeConfig,
    fold_shape: TensorChallengeShape,
) -> Result<Vec<GroupRoleBound>, AkitaError> {
    let mut group_items = Vec::with_capacity(conservative_plans.len());
    for plan in conservative_plans {
        if let Some(role_bound) =
            precommitted_role_bound(plan, policy, ring_challenge_cfg, fold_shape)?
        {
            group_items.push((plan.group.num_polynomials(), role_bound));
        }
    }
    precommitted_capacity_dp(max_num_batched_polys, &group_items)
}

/// Certify a complete setup-matrix envelope for all roots within `(N, K)`.
///
/// Complexity is polynomial in `N` and `K`: scalar and conservative plans are
/// `O(NK)`, total-capacity group aggregation is `O(NK²)`, and root/suffix split
/// scans are `O(N²K)`. No ordered group partition is materialized.
pub fn certify_setup_capacity(
    policy: &PlannerPolicy,
    max_num_vars: usize,
    max_num_batched_polys: usize,
    conservative_plans: &[ConservativeGroupPlan],
    ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<SetupCapacityCertificate, AkitaError> {
    if max_num_vars == 0 || max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup capacity requires positive variable and polynomial bounds".to_string(),
        ));
    }
    validate_policy_witness_chunk(policy)?;
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let mut envelope = SetupMatrixEnvelope::empty();
    let mut scalar_root_candidates = 0usize;
    let mut multi_group_root_candidates = 0usize;
    let mut suffix_candidates = 0usize;
    let alpha = policy.ring_dimension.trailing_zeros() as usize;
    let (min_log_basis, max_log_basis) = policy.basis_range;
    if min_log_basis == 0 || min_log_basis > max_log_basis {
        return Err(AkitaError::InvalidSetup(
            "setup capacity requires a positive, ordered basis range".to_string(),
        ));
    }

    for num_vars in 1..=max_num_vars {
        let root_inputs =
            AkitaScheduleInputs::for_root(PolynomialGroupLayout::singleton(num_vars))?;
        let fold_shape = fold_challenge_shape_at_level(root_inputs);
        for num_claims in 1..=max_num_batched_polys {
            if let Some(params) = compute_root_direct_level_params(
                policy,
                &ring_challenge_cfg,
                num_vars,
                policy.decomposition.log_basis,
                fold_shape,
                num_claims,
            )? {
                envelope.include_level(&params)?;
                scalar_root_candidates += 1;
            }
            let reduced_vars = num_vars.saturating_sub(alpha);
            let min_r_vars = usize::from(reduced_vars >= 3);
            for log_basis in min_log_basis..=max_log_basis {
                for r_vars in min_r_vars..reduced_vars {
                    let Some(params) = scalar_root_fold_level_params_candidate(
                        policy,
                        &ring_challenge_cfg,
                        num_vars,
                        num_claims,
                        log_basis,
                        r_vars,
                        fold_shape,
                    )?
                    else {
                        continue;
                    };
                    envelope.include_level(&params)?;
                    scalar_root_candidates += 1;
                }
            }
        }
    }

    let requires_conservative_plans = policy.decomposition.log_commit_bound == 1;
    let supports_grouped_roots =
        requires_conservative_plans && !policy.witness_chunk.uses_multi_chunk();
    let expected_plan_count = max_num_vars
        .checked_mul(max_num_batched_polys)
        .ok_or_else(|| AkitaError::InvalidSetup("setup plan count overflow".to_string()))?;
    if requires_conservative_plans && conservative_plans.len() != expected_plan_count {
        return Err(AkitaError::InvalidSetup(format!(
            "one-hot setup capacity requires {expected_plan_count} conservative group plans, got {}",
            conservative_plans.len()
        )));
    }
    let mut seen_plans = vec![false; expected_plan_count];
    for plan in conservative_plans {
        plan.group.validate()?;
        if plan.group.num_vars() > max_num_vars
            || plan.group.num_polynomials() > max_num_batched_polys
        {
            return Err(AkitaError::InvalidSetup(
                "conservative setup plan exceeds declared capacity".to_string(),
            ));
        }
        let plan_index = (plan.group.num_vars() - 1)
            .checked_mul(max_num_batched_polys)
            .and_then(|index| index.checked_add(plan.group.num_polynomials() - 1))
            .ok_or_else(|| AkitaError::InvalidSetup("setup plan index overflow".to_string()))?;
        if seen_plans[plan_index] {
            return Err(AkitaError::InvalidSetup(
                "duplicate conservative setup group plan".to_string(),
            ));
        }
        seen_plans[plan_index] = true;
        envelope.include_level(&plan.params)?;
    }

    if supports_grouped_roots {
        let mut bounds_by_shape = HashMap::new();
        for final_num_vars in 1..=max_num_vars {
            let root_inputs =
                AkitaScheduleInputs::for_root(PolynomialGroupLayout::singleton(final_num_vars))?;
            let fold_shape = fold_challenge_shape_at_level(root_inputs);
            if let std::collections::hash_map::Entry::Vacant(entry) =
                bounds_by_shape.entry(fold_shape)
            {
                let bounds = precommitted_bounds_for_shape(
                    policy,
                    max_num_batched_polys,
                    conservative_plans,
                    &ring_challenge_cfg,
                    fold_shape,
                )?;
                entry.insert(bounds);
            }
            let precommitted_bounds = &bounds_by_shape[&fold_shape];
            let reduced_vars = final_num_vars.saturating_sub(alpha);
            for final_num_claims in 1..max_num_batched_polys {
                let pre_bound = precommitted_bounds[max_num_batched_polys - final_num_claims];
                envelope.max_setup_len = envelope
                    .max_setup_len
                    .max(pre_bound.max_a_len)
                    .max(pre_bound.max_b_len);
                let candidate_ctx = MultiGroupRootCandidateCtx {
                    policy,
                    ring_challenge_cfg: &ring_challenge_cfg,
                    fold_challenge_shape: fold_shape,
                    precommitted_d_width: pre_bound.summed_d_width,
                    precommitted_groups: &[],
                };
                for log_basis in min_log_basis..=max_log_basis {
                    if reduced_vars == 0 {
                        if let Some(params) = multi_group_root_main_level_params_candidate(
                            &candidate_ctx,
                            final_num_claims,
                            log_basis,
                            0,
                            0,
                        )? {
                            envelope.include_level(&params)?;
                            multi_group_root_candidates += 1;
                        }
                    } else {
                        let min_r_vars = usize::from(reduced_vars >= 3);
                        for r_vars in min_r_vars..reduced_vars {
                            let Some(params) = multi_group_root_main_level_params_candidate(
                                &candidate_ctx,
                                final_num_claims,
                                log_basis,
                                reduced_vars - r_vars,
                                r_vars,
                            )?
                            else {
                                continue;
                            };
                            envelope.include_level(&params)?;
                            multi_group_root_candidates += 1;
                        }
                    }
                }
            }
        }
    }

    // Every root fold admitted by either scalar or grouped schedule search
    // satisfies
    //
    //   next_w_len * candidate_log_basis < 2^N * field_bits.
    //
    // Since every candidate basis is at least `min_log_basis`, the largest
    // possible recursive input has at most
    // `(2^N * field_bits - 1) / min_log_basis` field elements. Convert that
    // strict bit bound to a ring-element bucket. This covers both intermediate
    // and terminal successor lengths without assuming their bucket is at most
    // `N` (which is false for some policy dimensions).
    let max_root_witness_len = 1usize.checked_shl(max_num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("setup root witness length overflow".to_string())
    })?;
    let max_root_bits = max_root_witness_len
        .checked_mul(policy.decomposition.field_bits() as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("setup root bit length overflow".to_string()))?;
    let max_suffix_field_elems = max_root_bits
        .checked_sub(1)
        .ok_or_else(|| AkitaError::InvalidSetup("empty setup root bit range".to_string()))?
        / min_log_basis as usize;
    let max_suffix_ring_elems = max_suffix_field_elems / policy.ring_dimension;
    let max_suffix_reduced_vars = max_suffix_ring_elems
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("suffix setup bucket length overflow".to_string()))?
        .trailing_zeros() as usize;

    for reduced_vars in 3..=max_suffix_reduced_vars.min(52) {
        let bucket_max_ring_elems = 1usize.checked_shl(reduced_vars as u32).ok_or_else(|| {
            AkitaError::InvalidSetup("suffix setup bucket length overflow".to_string())
        })?;
        let num_ring_elems = bucket_max_ring_elems.min(max_suffix_ring_elems);
        for fold_level in 1..=MAX_RECURSION_DEPTH + 1 {
            for log_basis in min_log_basis..=max_log_basis {
                for r_vars in 1..reduced_vars {
                    let Some(params) = recursive_fold_level_params_candidate(
                        policy,
                        &ring_challenge_cfg,
                        num_ring_elems,
                        reduced_vars,
                        log_basis,
                        fold_level,
                        r_vars,
                    )?
                    else {
                        continue;
                    };
                    envelope.include_level(&params)?;
                    suffix_candidates += 1;
                }
            }
        }
    }

    Ok(SetupCapacityCertificate {
        envelope,
        scalar_root_candidates,
        conservative_group_plans: conservative_plans.len(),
        multi_group_root_candidates,
        suffix_candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::{DecompositionParams, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS};
    use std::cell::Cell;

    fn policy() -> PlannerPolicy {
        PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(128),
            },
            sis_family: SisModulusFamily::Q128,
            min_sis_security_bits: DEFAULT_SIS_SECURITY_BITS,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (3, 4),
            onehot_chunk_size: 1,
            witness_chunk: akita_types::ChunkedWitnessCfg::default(),
        }
    }

    fn ring_challenge(_: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Ok(SparseChallengeConfig::pm1_only(1))
    }

    fn flat_shape(_: AkitaScheduleInputs) -> TensorChallengeShape {
        TensorChallengeShape::Flat
    }

    fn level_setup_len(params: &LevelParams) -> usize {
        (params.a_key.row_len() * params.inner_width())
            .max(params.b_key.row_len() * params.outer_width())
            .max(params.d_key.row_len() * params.d_matrix_width())
    }

    #[test]
    fn certificate_includes_runtime_r_zero_domain_for_small_reduced_root() {
        let policy = policy();
        let ring_cfg = ring_challenge(policy.ring_dimension).expect("ring challenge");
        let candidate = scalar_root_fold_level_params_candidate(
            &policy,
            &ring_cfg,
            7,
            1,
            3,
            0,
            TensorChallengeShape::Flat,
        )
        .expect("candidate construction")
        .expect("runtime r=0 candidate");

        let certificate = certify_setup_capacity(&policy, 7, 1, &[], ring_challenge, flat_shape)
            .expect("setup certificate");
        let candidate_len = level_setup_len(&candidate);
        assert!(certificate.envelope.max_setup_len >= candidate_len);
    }

    #[test]
    fn total_capacity_dp_matches_small_unbounded_partition_oracle() {
        let groups = [
            (
                1,
                GroupRoleBound {
                    max_a_len: 3,
                    max_b_len: 9,
                    summed_d_width: 2,
                },
            ),
            (
                2,
                GroupRoleBound {
                    max_a_len: 11,
                    max_b_len: 4,
                    summed_d_width: 7,
                },
            ),
        ];
        let actual = precommitted_capacity_dp(4, &groups).expect("capacity DP");

        fn visit(
            remaining: usize,
            groups: &[(usize, GroupRoleBound)],
            current: GroupRoleBound,
            best: &mut GroupRoleBound,
        ) {
            *best = best.rolewise_max(current);
            for &(cost, group) in groups {
                if cost <= remaining {
                    visit(
                        remaining - cost,
                        groups,
                        current.add_group(group).expect("oracle sum"),
                        best,
                    );
                }
            }
        }

        for capacity in 0..=4 {
            let mut expected = GroupRoleBound::default();
            visit(capacity, &groups, GroupRoleBound::default(), &mut expected);
            assert_eq!(actual[capacity].max_a_len, expected.max_a_len);
            assert_eq!(actual[capacity].max_b_len, expected.max_b_len);
            assert_eq!(actual[capacity].summed_d_width, expected.summed_d_width);
        }
    }

    #[test]
    fn suffix_certificate_includes_depth_boundary_and_first_unchunked_level() {
        let mut policy = policy();
        policy.witness_chunk = akita_types::ChunkedWitnessCfg {
            num_chunks: 2,
            num_activated_levels: MAX_RECURSION_DEPTH,
        };
        let ring_cfg = ring_challenge(policy.ring_dimension).expect("ring challenge");
        let first_unchunked = recursive_fold_level_params_candidate(
            &policy,
            &ring_cfg,
            16,
            4,
            3,
            MAX_RECURSION_DEPTH,
            1,
        )
        .expect("first unchunked candidate")
        .expect("first unchunked candidate exists");
        let depth_boundary = recursive_fold_level_params_candidate(
            &policy,
            &ring_cfg,
            16,
            4,
            3,
            MAX_RECURSION_DEPTH + 1,
            1,
        )
        .expect("depth-boundary candidate")
        .expect("depth-boundary candidate exists");

        let certificate = certify_setup_capacity(&policy, 7, 1, &[], ring_challenge, flat_shape)
            .expect("setup certificate");
        assert!(certificate.envelope.max_setup_len >= level_setup_len(&first_unchunked));
        assert!(certificate.envelope.max_setup_len >= level_setup_len(&depth_boundary));
        assert_eq!(first_unchunked.witness_chunk.num_chunks, 1);
        assert_eq!(depth_boundary.witness_chunk.num_chunks, 1);
    }

    #[test]
    fn partial_top_suffix_bucket_dominates_every_smaller_ring_count() {
        let policy = policy();
        let ring_cfg = ring_challenge(policy.ring_dimension).expect("ring challenge");
        let top = recursive_fold_level_params_candidate(&policy, &ring_cfg, 25, 5, 3, 1, 2)
            .expect("top candidate")
            .expect("top candidate exists");
        let top_len = level_setup_len(&top);
        for ring_count in 17..=25 {
            let candidate =
                recursive_fold_level_params_candidate(&policy, &ring_cfg, ring_count, 5, 3, 1, 2)
                    .expect("bucket candidate")
                    .expect("bucket candidate exists");
            assert!(top_len >= level_setup_len(&candidate));
        }
    }

    #[test]
    fn planner_and_certificate_resolve_ring_hook_once_per_entry() {
        let policy = policy();
        let planner_calls = Cell::new(0usize);
        crate::find_schedule(
            PolynomialGroupLayout::new(7, 1),
            &policy,
            |_| {
                planner_calls.set(planner_calls.get() + 1);
                ring_challenge(policy.ring_dimension)
            },
            flat_shape,
        )
        .expect("runtime schedule");
        assert_eq!(planner_calls.get(), 1);

        let certificate_calls = Cell::new(0usize);
        certify_setup_capacity(
            &policy,
            7,
            1,
            &[],
            |_| {
                certificate_calls.set(certificate_calls.get() + 1);
                ring_challenge(policy.ring_dimension)
            },
            flat_shape,
        )
        .expect("setup certificate");
        assert_eq!(certificate_calls.get(), 1);
    }
}
