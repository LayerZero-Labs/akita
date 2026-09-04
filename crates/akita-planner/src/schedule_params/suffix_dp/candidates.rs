use akita_error::AkitaError;
use akita_types::{
    try_extension_opening_reduction_level_bytes, AkitaScheduleLookupKey, CommitmentRingDims,
    CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout,
};

use crate::{
    planner::{precommitted_group_equivalence_classes, root_level_candidates_for_basis},
    PlannerPolicy,
};

use super::{
    derive_fold_candidates, derive_recursive_candidate_views, derive_terminal_candidates,
    dimension_candidates, suffix_opening_layout, FoldCandidatePolicy, RecursiveCandidateRequest,
    RecursiveFoldWork, SetupPrefixSearchCache, SplitBoundPolicy, SuffixCtx, SuffixState,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpeningPurpose {
    TerminalOnly,
    FoldOnly,
    TerminalAndFold,
}

impl OpeningPurpose {
    const fn allows_terminal(self) -> bool {
        matches!(self, Self::TerminalOnly | Self::TerminalAndFold)
    }

    const fn allows_fold(self) -> bool {
        matches!(self, Self::FoldOnly | Self::TerminalAndFold)
    }
}

const fn trace_opening_purpose(
    early_packing_level: bool,
    terminal_seed_is_relevant: bool,
) -> Option<OpeningPurpose> {
    match (early_packing_level, terminal_seed_is_relevant) {
        (true, true) => Some(OpeningPurpose::TerminalOnly),
        (true, false) => None,
        (false, true) => Some(OpeningPurpose::TerminalAndFold),
        (false, false) => Some(OpeningPurpose::FoldOnly),
    }
}

#[derive(Clone)]
struct OpeningWork {
    dimensions: CommitmentRingDims,
    opening: crate::schedule_params::PlannerOpeningCandidate,
    precommitted_openings: Vec<crate::schedule_params::PlannerOpeningCandidate>,
    opening_reduction_bytes: usize,
    purpose: OpeningPurpose,
}

pub(super) struct RawTerminalCandidate {
    pub(super) params: CommittedGroupParams,
    pub(super) opening_reduction_bytes: usize,
}

pub(super) struct RawFoldCandidate {
    pub(super) params: CommittedGroupParams,
    pub(super) next_witness_len: usize,
    pub(super) opening_reduction_bytes: usize,
}

pub(super) struct GeneratedCandidates {
    pub(super) terminal: Vec<RawTerminalCandidate>,
    pub(super) folds: Vec<RawFoldCandidate>,
}

pub(super) struct CandidateDomain<'a> {
    pub(super) root_level_key: Option<&'a AkitaScheduleLookupKey>,
    pub(super) opening_layout: OpeningClaimsLayout,
    inner_source: crate::InnerBasisSource,
    inner_basis_range: std::ops::RangeInclusive<u32>,
    pub(super) opening_basis_range: std::ops::RangeInclusive<u32>,
    opening_work: Vec<OpeningWork>,
    fold_policy: FoldCandidatePolicy,
    pub(super) require_child_fold: bool,
}

pub(crate) const fn state_allows_terminal_seed(
    is_root_level: bool,
    has_incoming_setup_prefix: bool,
) -> bool {
    !is_root_level && !has_incoming_setup_prefix
}

pub(crate) fn packing_precommit_opening_products(
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    key: &AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: &[akita_types::sis::HonestFoldPolicySpec],
) -> Result<Vec<Vec<crate::schedule_params::PlannerOpeningCandidate>>, AkitaError> {
    if key.precommitteds.len() != precommitted_honest_fold_policies.len() {
        return Err(AkitaError::InvalidSetup(
            "root precommit opening products require one policy per profile".into(),
        ));
    }
    if !crate::schedule_params::precommitted_groups_support_opening_dimension(
        key.precommitteds.iter(),
        dimensions.d_d(),
    ) {
        return Ok(Vec::new());
    }
    let equivalence_classes = precommitted_group_equivalence_classes(
        &key.precommitteds,
        precommitted_honest_fold_policies,
    )?;

    let mut products = vec![vec![None; key.precommitteds.len()]];
    for indices in equivalence_classes {
        let representative = indices[0];
        let profile = &key.precommitteds[representative];
        let domain = crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
            0,
            policy.claim_ext_degree,
            CommitmentRingDims {
                inner: profile.inner.matrix.ring_dimension(),
                outer: profile.outer.matrix.ring_dimension(),
                opening: dimensions.d_d(),
            },
        )?;
        if domain.is_empty() {
            return Ok(Vec::new());
        }
        let assignments = nondecreasing_opening_assignments(&domain, indices.len());
        let next_len = products
            .len()
            .checked_mul(assignments.len())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("root precommit opening search domain overflow".into())
            })?;
        let mut next = Vec::new();
        next.try_reserve_exact(next_len).map_err(|_| {
            AkitaError::InvalidSetup("root precommit opening search domain is too large".into())
        })?;
        for product in products {
            for assignment in &assignments {
                let mut extended = product.clone();
                for (&index, &opening) in indices.iter().zip(assignment) {
                    extended[index] = Some(opening);
                }
                next.push(extended);
            }
        }
        products = next;
    }
    products
        .into_iter()
        .map(|product| {
            product
                .into_iter()
                .map(|opening| {
                    opening.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "root precommit opening product is incomplete".into(),
                        )
                    })
                })
                .collect()
        })
        .collect()
}

/// Canonical assignments for interchangeable precommitted groups.
///
/// Every multiset of opening candidates is retained, while permutations among
/// groups with the same profile and honest-fold policy are removed. The chosen
/// representative is canonicalized from fully materialized group descriptors
/// before root candidate construction.
fn nondecreasing_opening_assignments(
    domain: &[crate::schedule_params::PlannerOpeningCandidate],
    width: usize,
) -> Vec<Vec<crate::schedule_params::PlannerOpeningCandidate>> {
    fn extend(
        domain: &[crate::schedule_params::PlannerOpeningCandidate],
        width: usize,
        minimum: usize,
        prefix: &mut Vec<crate::schedule_params::PlannerOpeningCandidate>,
        output: &mut Vec<Vec<crate::schedule_params::PlannerOpeningCandidate>>,
    ) {
        if prefix.len() == width {
            output.push(prefix.clone());
            return;
        }
        for index in minimum..domain.len() {
            prefix.push(domain[index]);
            extend(domain, width, index, prefix, output);
            prefix.pop();
        }
    }

    let mut output = Vec::new();
    extend(
        domain,
        width,
        0,
        &mut Vec::with_capacity(width),
        &mut output,
    );
    output
}

/// Enumerate the method/dimension work for one suffix state.
///
/// EvaluationTrace work remains before coefficient-packing work to preserve
/// deterministic tie behavior from the original search order.
fn opening_work_domain(
    ctx: &SuffixCtx<'_>,
    state: SuffixState,
    root_level_key: Option<&AkitaScheduleLookupKey>,
    opening_shape: PolynomialGroupLayout,
) -> Result<Vec<OpeningWork>, AkitaError> {
    let policy = ctx.policy;
    let early_packing_level = state.level <= 1;
    let terminal_seed_is_relevant = state_allows_terminal_seed(
        root_level_key.is_some(),
        state.topology.incoming_setup_prefix().is_some(),
    );
    let mut trace_work = Vec::new();
    let mut packing_work = Vec::new();

    for dimensions in dimension_candidates(policy, state.level, state.dimension_ceiling)? {
        if root_level_key.is_some_and(|root_key| {
            !crate::schedule_params::precommitted_groups_support_opening_dimension(
                root_key.precommitteds.iter(),
                dimensions.d_d(),
            )
        }) {
            continue;
        }
        let packing_domain = early_packing_level
            .then(|| {
                crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
                    state.level,
                    policy.claim_ext_degree,
                    dimensions,
                )
            })
            .transpose()?
            .unwrap_or_default();
        let root_precommit_products = if early_packing_level {
            root_level_key
                .map(|root_key| {
                    packing_precommit_opening_products(
                        policy,
                        dimensions,
                        root_key,
                        ctx.precommitted_honest_fold_policies,
                    )
                })
                .transpose()?
        } else {
            None
        };

        if let Ok(ring_challenge_cfg) = (ctx.ring_challenge_config)(dimensions.d_a()) {
            if let Some(opening_reduction_bytes) = try_extension_opening_reduction_level_bytes(
                policy.challenge_field_bits()?,
                policy.claim_ext_degree,
                opening_shape,
            )? {
                let precommitted_openings = if let Some(root_key) = root_level_key {
                    let mut openings = Vec::with_capacity(root_key.precommitteds.len());
                    let mut valid = true;
                    for profile in &root_key.precommitteds {
                        let Ok(config) =
                            (ctx.ring_challenge_config)(profile.inner.matrix.ring_dimension())
                        else {
                            valid = false;
                            break;
                        };
                        openings.push(
                            crate::schedule_params::PlannerOpeningCandidate::evaluation_trace(
                                config,
                            ),
                        );
                    }
                    valid.then_some(openings)
                } else {
                    Some(Vec::new())
                };
                if let Some(precommitted_openings) = precommitted_openings {
                    if let Some(purpose) =
                        trace_opening_purpose(early_packing_level, terminal_seed_is_relevant)
                    {
                        trace_work.push(OpeningWork {
                            dimensions,
                            opening:
                                crate::schedule_params::PlannerOpeningCandidate::evaluation_trace(
                                    ring_challenge_cfg,
                                ),
                            precommitted_openings,
                            opening_reduction_bytes,
                            purpose,
                        });
                    }
                }
            }
        }

        if let Some(precommit_products) = root_precommit_products.as_ref() {
            for opening in packing_domain {
                for precommitted_openings in precommit_products {
                    packing_work.push(OpeningWork {
                        dimensions,
                        opening,
                        precommitted_openings: precommitted_openings.clone(),
                        opening_reduction_bytes: 0,
                        purpose: OpeningPurpose::FoldOnly,
                    });
                }
            }
        } else {
            packing_work.extend(packing_domain.into_iter().map(|opening| OpeningWork {
                dimensions,
                opening,
                precommitted_openings: Vec::new(),
                opening_reduction_bytes: 0,
                purpose: OpeningPurpose::FoldOnly,
            }));
        }
    }

    trace_work.extend(packing_work);
    Ok(trace_work)
}

impl<'a> CandidateDomain<'a> {
    pub(super) fn prepare(ctx: &SuffixCtx<'a>, state: SuffixState) -> Result<Self, AkitaError> {
        let policy = ctx.policy;
        let root_level_key = ctx.root_lookup_key.filter(|_| state.level == 0);
        let incoming_setup_prefix = state.topology.incoming_setup_prefix();
        if root_level_key.is_some() && incoming_setup_prefix.is_some() {
            return Err(AkitaError::InvalidSetup(
                "root batch cannot consume an incoming setup prefix".into(),
            ));
        }
        if ctx.level_zero_is_root && state.level == 0 && root_level_key.is_none() {
            return Err(AkitaError::InvalidSetup(
                "root-level suffix state is missing its opening lookup key".into(),
            ));
        }
        let opening_layout = if let Some(root_key) = root_level_key {
            root_key.opening_layout()?
        } else {
            suffix_opening_layout(state.current_witness_len, incoming_setup_prefix)?
        };
        let opening_shape = opening_layout.aggregate_polynomial_group_layout()?;
        let inner_source = if ctx.level_zero_is_root && state.level == 0 {
            crate::schedule_params::root_inner_basis_source(
                ctx.root_honest_fold_policy.ok_or_else(|| {
                    AkitaError::InvalidSetup("root batch is missing its honest fold policy".into())
                })?,
                policy.decomposition.log_commit_bound,
            )
        } else {
            crate::InnerBasisSource::BalancedDigits {
                log_basis: state.current_lb,
            }
        };
        let (min_inner_basis, max_inner_basis) = inner_source.search_range(policy)?;
        let (min_open_basis, max_open_basis) =
            crate::policy::log_basis_search_range_at_level(policy, state.level);
        let opening_work = opening_work_domain(ctx, state, root_level_key, opening_shape)?;
        let retain_split_frontier = state.topology.incoming_setup_prefix().is_some()
            || policy.selection_policy == crate::SelectionPolicyId::MinEstimatedProofPayloadV2
            || matches!(
                policy.ring_dimension_schedule_mode,
                crate::RingDimensionScheduleMode::AdaptiveDimension {
                    num_search_levels,
                    ..
                } if state.level < num_search_levels
            );
        let fold_policy = if retain_split_frontier {
            FoldCandidatePolicy::Frontier(SplitBoundPolicy::Enabled)
        } else {
            FoldCandidatePolicy::Best
        };
        let require_child_fold =
            root_level_key.is_some_and(|root_key| !root_key.precommitteds.is_empty());

        Ok(Self {
            root_level_key,
            opening_layout,
            inner_source,
            inner_basis_range: min_inner_basis..=max_inner_basis,
            opening_basis_range: min_open_basis.max(state.current_lb)..=max_open_basis,
            opening_work,
            fold_policy,
            require_child_fold,
        })
    }

    pub(super) fn generate_for_opening_basis(
        &self,
        ctx: &SuffixCtx<'_>,
        state: SuffixState,
        open_lb: u32,
        setup_prefixes: &mut SetupPrefixSearchCache,
    ) -> Result<GeneratedCandidates, AkitaError> {
        let policy = ctx.policy;
        let incoming_setup_prefix = state.topology.incoming_setup_prefix();
        let mut terminal = Vec::new();
        let mut folds = Vec::new();

        for inner_lb in self.inner_basis_range.clone() {
            if let Some(root_key) = self.root_level_key {
                for work in &self.opening_work {
                    let dimension_candidates = root_level_candidates_for_basis(
                        root_key,
                        ctx.root_honest_fold_policy.ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "root batch is missing its honest fold policy".into(),
                            )
                        })?,
                        ctx.precommitted_honest_fold_policies,
                        policy,
                        work.dimensions,
                        work.opening,
                        &work.precommitted_openings,
                        inner_lb,
                        open_lb,
                    )?;
                    let relation_domain = state
                        .topology
                        .relation_domain(state.level, work.opening.method(), ctx.diagnostics)?
                        .filtered(ctx.relation_mode_filter)?;
                    let relation_transition = relation_domain.only_transition()?;
                    for (params, next_witness_len) in dimension_candidates {
                        if params.ring_relation_mode != relation_transition {
                            return Err(AkitaError::InvalidSetup(
                                "materialized mode disagrees with relation domain".into(),
                            ));
                        }
                        if work.purpose.allows_terminal() {
                            terminal.push(RawTerminalCandidate {
                                params: params.clone(),
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            });
                        }
                        if work.purpose.allows_fold() {
                            folds.push(RawFoldCandidate {
                                params,
                                next_witness_len,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            });
                        }
                    }
                }
                continue;
            }

            for work in &self.opening_work {
                for &payload_mode in state
                    .topology
                    .payload_phase()
                    .candidate_modes(state.level, incoming_setup_prefix.is_some())
                {
                    let request = RecursiveCandidateRequest {
                        policy,
                        payload_mode,
                        opening: work.opening,
                        dimensions: work.dimensions,
                        current_witness_len: state.current_witness_len,
                        source: self.inner_source,
                        log_basis_inner: inner_lb,
                        log_basis_open: open_lb,
                        fold_level: state.level,
                        source_moment: state.source_moment,
                        relation_traversal_order: ctx.relation_traversal_order,
                    };
                    let relation_domain = state
                        .topology
                        .relation_domain(state.level, work.opening.method(), ctx.diagnostics)?
                        .filtered(ctx.relation_mode_filter)?;
                    if work.purpose == OpeningPurpose::TerminalAndFold {
                        let views = derive_recursive_candidate_views(
                            request,
                            self.fold_policy,
                            relation_domain,
                        )?;
                        terminal.extend(views.terminal.into_iter().map(|params| {
                            RawTerminalCandidate {
                                params,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            }
                        }));
                        for (candidate, next_witness_len) in views.folds {
                            if !relation_domain.admits(candidate.ring_relation_mode) {
                                return Err(AkitaError::InvalidSetup(
                                    "combined recursive view emitted a fold outside its relation domain"
                                        .into(),
                                ));
                            }
                            folds.push(RawFoldCandidate {
                                params: candidate,
                                next_witness_len,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            });
                        }
                        continue;
                    }
                    if work.purpose.allows_terminal() {
                        terminal.extend(derive_terminal_candidates(request)?.into_iter().map(
                            |params| RawTerminalCandidate {
                                params,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            },
                        ));
                    }
                    if !work.purpose.allows_fold() {
                        continue;
                    }
                    let fold_work = if let Some(natural_len) = incoming_setup_prefix {
                        RecursiveFoldWork::setup_prefixed(setup_prefixes, natural_len)
                    } else {
                        RecursiveFoldWork::direct(relation_domain)
                    };
                    let level_candidates =
                        derive_fold_candidates(request, fold_work, self.fold_policy)?;
                    for (candidate, next_witness_len) in level_candidates {
                        folds.push(RawFoldCandidate {
                            params: candidate,
                            next_witness_len,
                            opening_reduction_bytes: work.opening_reduction_bytes,
                        });
                    }
                }
            }
        }

        Ok(GeneratedCandidates { terminal, folds })
    }

    /// Visit one bounded root work batch at a time.
    ///
    /// Root frontiers are shared across visits, so this changes temporary
    /// ownership only: it preserves every candidate while avoiding one large
    /// vector spanning the full grouped-opening product.
    pub(super) fn visit_root_batches(
        &self,
        ctx: &SuffixCtx<'_>,
        state: SuffixState,
        open_lb: u32,
        mut visit: impl FnMut(GeneratedCandidates) -> Result<(), AkitaError>,
    ) -> Result<(), AkitaError> {
        let root_key = self.root_level_key.ok_or_else(|| {
            AkitaError::InvalidSetup("root batch visitor requires a root lookup key".into())
        })?;
        let final_policy = ctx.root_honest_fold_policy.ok_or_else(|| {
            AkitaError::InvalidSetup("root batch is missing its honest fold policy".into())
        })?;
        for inner_lb in self.inner_basis_range.clone() {
            for work in &self.opening_work {
                let dimension_candidates = root_level_candidates_for_basis(
                    root_key,
                    final_policy,
                    ctx.precommitted_honest_fold_policies,
                    ctx.policy,
                    work.dimensions,
                    work.opening,
                    &work.precommitted_openings,
                    inner_lb,
                    open_lb,
                )?;
                let relation_transition = state
                    .topology
                    .relation_domain(state.level, work.opening.method(), ctx.diagnostics)?
                    .filtered(ctx.relation_mode_filter)?
                    .only_transition()?;
                let mut terminal = Vec::new();
                let mut folds = Vec::new();
                for (params, next_witness_len) in dimension_candidates {
                    if params.ring_relation_mode != relation_transition {
                        return Err(AkitaError::InvalidSetup(
                            "materialized mode disagrees with relation domain".into(),
                        ));
                    }
                    if work.purpose.allows_terminal() {
                        terminal.push(RawTerminalCandidate {
                            params: params.clone(),
                            opening_reduction_bytes: work.opening_reduction_bytes,
                        });
                    }
                    if work.purpose.allows_fold() {
                        folds.push(RawFoldCandidate {
                            params,
                            next_witness_len,
                            opening_reduction_bytes: work.opening_reduction_bytes,
                        });
                    }
                }
                if !terminal.is_empty() || !folds.is_empty() {
                    visit(GeneratedCandidates { terminal, folds })?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_purpose_separates_early_packing_and_terminal_admission() {
        assert_eq!(
            trace_opening_purpose(true, true),
            Some(OpeningPurpose::TerminalOnly)
        );
        assert_eq!(trace_opening_purpose(true, false), None);
        assert_eq!(
            trace_opening_purpose(false, true),
            Some(OpeningPurpose::TerminalAndFold)
        );
        assert_eq!(
            trace_opening_purpose(false, false),
            Some(OpeningPurpose::FoldOnly)
        );
    }
}
