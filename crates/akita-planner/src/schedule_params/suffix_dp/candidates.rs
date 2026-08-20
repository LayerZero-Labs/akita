use akita_error::AkitaError;
use akita_types::{
    try_extension_opening_reduction_level_bytes, AkitaScheduleLookupKey, CommitmentRingDims,
    CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout,
};

use crate::{planner::root_level_candidates_for_basis, PlannerPolicy};

use super::{
    derive_fold_candidates, derive_recursive_candidate_views, derive_terminal_candidates,
    dimension_candidates, suffix_opening_layout, FoldCandidatePolicy, RecursiveCandidateRequest,
    RecursiveSetupPrefix, SetupPrefixSearchCache, SplitBoundPolicy, SuffixCtx, SuffixState,
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

pub(super) struct RawLevelCandidate {
    pub(super) params: CommittedGroupParams,
    pub(super) next_witness_len: usize,
    pub(super) opening_reduction_bytes: usize,
}

pub(super) struct GeneratedCandidates {
    pub(super) terminal: Vec<RawLevelCandidate>,
    pub(super) folds: Vec<RawLevelCandidate>,
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
) -> Result<Vec<Vec<crate::schedule_params::PlannerOpeningCandidate>>, AkitaError> {
    if !crate::schedule_params::precommitted_groups_support_opening_dimension(
        key.precommitteds.iter(),
        dimensions.d_d(),
    ) {
        return Ok(Vec::new());
    }
    let mut products = vec![Vec::new()];
    for profile in &key.precommitteds {
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
        let next_len = products.len().checked_mul(domain.len()).ok_or_else(|| {
            AkitaError::InvalidSetup("root precommit opening search domain overflow".into())
        })?;
        let mut next = Vec::new();
        next.try_reserve_exact(next_len).map_err(|_| {
            AkitaError::InvalidSetup("root precommit opening search domain is too large".into())
        })?;
        for product in products {
            for &opening in &domain {
                let mut extended = product.clone();
                extended.push(opening);
                next.push(extended);
            }
        }
        products = next;
    }
    Ok(products)
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
        state.incoming_setup_prefix.is_some(),
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
                .map(|root_key| packing_precommit_opening_products(policy, dimensions, root_key))
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
        if root_level_key.is_some() && state.incoming_setup_prefix.is_some() {
            return Err(AkitaError::InvalidSetup(
                "root batch cannot consume an incoming setup prefix".into(),
            ));
        }
        if ctx.level_zero_is_root && state.level == 0 && root_level_key.is_none() {
            return Err(AkitaError::InvalidSetup(
                "root-level suffix state is missing its opening lookup key".into(),
            ));
        }
        if state.payload_phase == akita_types::CommitmentPayloadPhase::RawSuffix
            && state.incoming_setup_prefix.is_some()
        {
            return Err(AkitaError::InvalidSetup(
                "raw commitment suffix cannot consume a recursive setup prefix".into(),
            ));
        }

        let opening_layout = if let Some(root_key) = root_level_key {
            root_key.opening_layout()?
        } else {
            suffix_opening_layout(state.current_witness_len, state.incoming_setup_prefix)?
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
        let retain_split_frontier = state.incoming_setup_prefix.is_some()
            || (policy.selection_policy == crate::SelectionPolicyId::MinEstimatedProofPayload
                && state.level < akita_schedules::ADAPTIVE_SEARCH_LEVELS)
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
                        state.current_witness_len,
                        inner_lb,
                        open_lb,
                        true,
                    )?;
                    for (params, next_witness_len) in dimension_candidates {
                        if work.purpose.allows_terminal() {
                            terminal.push(RawLevelCandidate {
                                params: params.clone(),
                                next_witness_len,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            });
                        }
                        if work.purpose.allows_fold() {
                            folds.push(RawLevelCandidate {
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
                    .payload_phase
                    .candidate_modes(state.level, state.incoming_setup_prefix.is_some())
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
                    };
                    if work.purpose == OpeningPurpose::TerminalAndFold
                        && state.incoming_setup_prefix.is_none()
                    {
                        let views = derive_recursive_candidate_views(request, self.fold_policy)?;
                        terminal.extend(views.terminal.into_iter().map(|params| {
                            RawLevelCandidate {
                                params,
                                next_witness_len: 0,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            }
                        }));
                        folds.extend(views.folds.into_iter().map(|(params, next_witness_len)| {
                            RawLevelCandidate {
                                params,
                                next_witness_len,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            }
                        }));
                        continue;
                    }
                    if work.purpose.allows_terminal() {
                        terminal.extend(derive_terminal_candidates(request)?.into_iter().map(
                            |params| RawLevelCandidate {
                                params,
                                next_witness_len: 0,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            },
                        ));
                    }
                    if !work.purpose.allows_fold() {
                        continue;
                    }
                    let setup_prefix = if let Some(natural_len) = state.incoming_setup_prefix {
                        RecursiveSetupPrefix::Search {
                            cache: setup_prefixes,
                            natural_len,
                        }
                    } else {
                        RecursiveSetupPrefix::None
                    };
                    let level_candidates =
                        derive_fold_candidates(request, setup_prefix, self.fold_policy)?;
                    for (params, next_witness_len) in level_candidates {
                        folds.push(RawLevelCandidate {
                            params,
                            next_witness_len,
                            opening_reduction_bytes: work.opening_reduction_bytes,
                        });
                    }
                }
            }
        }

        Ok(GeneratedCandidates { terminal, folds })
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
