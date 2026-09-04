use akita_error::AkitaError;
use akita_types::{
    try_extension_opening_reduction_level_bytes, AkitaScheduleLookupKey, CommitmentRingDims,
    CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout, TerminalFoldParams,
};

use crate::{
    planner::{precommitted_group_equivalence_classes, root_level_candidates_for_basis},
    PlannerPolicy,
};

use super::{
    derive_fold_candidates, derive_recursive_candidate_views, derive_terminal_candidates,
    dimension_candidates, suffix_opening_layout, FoldCandidatePolicy, RecursiveCandidateGuide,
    RecursiveCandidateRequest, RecursiveFoldWork, SetupPrefixSearchCache, SplitBoundPolicy,
    SuffixCtx, SuffixState,
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

const MAX_ADAPTED_PRECOMMIT_OPENING_PRODUCTS: usize = 256;

pub(super) struct RawTerminalCandidate {
    pub(super) params: CommittedGroupParams,
    pub(super) opening_reduction_bytes: usize,
}

pub(super) struct RawFoldCandidate {
    pub(super) params: CommittedGroupParams,
    pub(super) next_witness_len: usize,
    pub(super) opening_reduction_bytes: usize,
    pub(super) relation_transition: super::RelationTransition,
}

pub(super) struct GeneratedCandidates {
    pub(super) terminal: Vec<RawTerminalCandidate>,
    pub(super) folds: Vec<RawFoldCandidate>,
}

pub(super) struct CandidateDomain<'a> {
    pub(super) root_level_key: Option<&'a AkitaScheduleLookupKey>,
    root_main_constraint: Option<&'a CommittedGroupParams>,
    guide_fold: Option<&'a CommittedGroupParams>,
    guide_terminal: Option<&'a TerminalFoldParams>,
    adaptation_guided: bool,
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
    max_products: Option<usize>,
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
        if let Some(max_products) = max_products {
            let remaining_products = max_products / products.len();
            if !multiset_assignment_count_fits(domain.len(), indices.len(), remaining_products) {
                return Err(AkitaError::UnsupportedSchedule(format!(
                    "adapted precommit opening domain exceeds the maximum of {max_products} assignments"
                )));
            }
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

/// Return whether the number of multisets of `width` values drawn from a
/// `domain_len`-element domain fits within `limit`.
fn multiset_assignment_count_fits(domain_len: usize, width: usize, limit: usize) -> bool {
    if domain_len == 0 {
        return width == 0;
    }
    if domain_len == 1 {
        return limit >= 1;
    }
    let mut count = 1_u128;
    let limit = limit as u128;
    for index in 1..=width {
        let Some(numerator) = domain_len
            .checked_add(index)
            .and_then(|sum| sum.checked_sub(1))
        else {
            return false;
        };
        count = count.saturating_mul(numerator as u128) / index as u128;
        if count > limit {
            return false;
        }
    }
    true
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
    root_main_constraint: Option<&CommittedGroupParams>,
    guide_fold: Option<&CommittedGroupParams>,
    guide_terminal: Option<&TerminalFoldParams>,
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

    let fold_constraint = root_main_constraint.or(guide_fold);
    let dimension_domain = if let Some(constraint) = fold_constraint {
        vec![constraint.role_dims()]
    } else if let Some(terminal) = guide_terminal {
        vec![CommitmentRingDims::uniform(terminal.d_a())]
    } else {
        dimension_candidates(policy, state.level, state.dimension_ceiling)?
    };
    for dimensions in dimension_domain {
        if root_level_key.is_some_and(|root_key| {
            !crate::schedule_params::precommitted_groups_support_opening_dimension(
                root_key.precommitteds.iter(),
                dimensions.d_d(),
            )
        }) {
            continue;
        }
        let constrained_opening = fold_constraint
            .map(|constraint| guided_opening(policy, state.level, constraint))
            .transpose()?
            .or_else(|| {
                guide_terminal.and_then(|terminal| {
                    (ctx.ring_challenge_config)(terminal.d_a())
                        .ok()
                        .map(crate::schedule_params::PlannerOpeningCandidate::evaluation_trace)
                })
            });
        let packing_domain = if !early_packing_level {
            Vec::new()
        } else if let Some(opening) = constrained_opening {
            opening
                .is_coefficient_packing()
                .then_some(opening)
                .into_iter()
                .collect()
        } else {
            crate::schedule_params::PlannerOpeningCandidate::coefficient_packing_domain(
                state.level,
                policy.claim_ext_degree,
                dimensions,
            )?
        };
        let root_precommit_products = if early_packing_level {
            root_level_key
                .map(|root_key| {
                    let products = packing_precommit_opening_products(
                        policy,
                        dimensions,
                        root_key,
                        ctx.precommitted_honest_fold_policies,
                        root_main_constraint.map(|_| MAX_ADAPTED_PRECOMMIT_OPENING_PRODUCTS),
                    )?;
                    Ok(products)
                })
                .transpose()?
        } else {
            None
        };

        let trace_opening = match constrained_opening {
            Some(opening) if !opening.is_coefficient_packing() => Some(opening),
            Some(_) => None,
            None => (ctx.ring_challenge_config)(dimensions.d_a())
                .ok()
                .map(crate::schedule_params::PlannerOpeningCandidate::evaluation_trace),
        };
        if let Some(trace_opening) = trace_opening {
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
                            opening: trace_opening,
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

fn guided_opening(
    policy: &PlannerPolicy,
    absolute_level: usize,
    constraint: &CommittedGroupParams,
) -> Result<crate::schedule_params::PlannerOpeningCandidate, AkitaError> {
    let opening = match constraint.opening_method() {
        akita_types::OpeningMethod::EvaluationTrace => {
            crate::schedule_params::PlannerOpeningCandidate::evaluation_trace(
                constraint.fold_challenge_config(),
            )
        }
        akita_types::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => crate::schedule_params::PlannerOpeningCandidate::coefficient_packing(
            absolute_level,
            policy.claim_ext_degree,
            constraint.role_dims(),
            challenge_subring_dimension,
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup("adapted opening is outside the current packing domain".into())
        })?,
    };
    if opening.challenge_config() != constraint.fold_challenge_config() {
        return Err(AkitaError::InvalidSetup(
            "adapted opening challenge does not match its main-row guide".into(),
        ));
    }
    opening.validate_for(
        absolute_level,
        policy.claim_ext_degree,
        constraint.role_dims(),
    )?;
    Ok(opening)
}

fn root_candidate_matches_constraint(
    candidate: &CommittedGroupParams,
    constraint: &CommittedGroupParams,
) -> bool {
    candidate.own_group() == constraint.own_group()
        && candidate.payload_mode == constraint.payload_mode
        && candidate.ring_relation_mode == constraint.ring_relation_mode
        && candidate.source_encoding == constraint.source_encoding
        && candidate.witness_chunk == constraint.witness_chunk
        && candidate.role_dims() == constraint.role_dims()
        && candidate.open_matrix.sis_table_key() == constraint.open_matrix.sis_table_key()
}

fn inner_route_kind_matches(
    candidate: akita_types::InnerCommitSecurityRoute,
    guide: akita_types::InnerCommitSecurityRoute,
) -> bool {
    matches!(
        (candidate, guide),
        (
            akita_types::InnerCommitSecurityRoute::Linf(_),
            akita_types::InnerCommitSecurityRoute::Linf(_)
        ) | (
            akita_types::InnerCommitSecurityRoute::L2 { .. },
            akita_types::InnerCommitSecurityRoute::L2 { .. }
        )
    )
}

fn setup_prefix_structure_matches(
    candidate: Option<&akita_types::GroupOpenPhaseParams>,
    guide: Option<&akita_types::GroupOpenPhaseParams>,
) -> bool {
    match (candidate, guide) {
        (None, None) => true,
        (Some(candidate), Some(guide)) => {
            candidate.profile.inner.matrix.ring_dimension()
                == guide.profile.inner.matrix.ring_dimension()
                && candidate.profile.outer.matrix.ring_dimension()
                    == guide.profile.outer.matrix.ring_dimension()
                && candidate.profile.blocks.positions_per_block
                    == guide.profile.blocks.positions_per_block
                && candidate.profile.outer_slice_count == guide.profile.outer_slice_count
                && candidate.profile.inner.digits.log_basis == guide.profile.inner.digits.log_basis
                && candidate.profile.outer.digits.log_basis == guide.profile.outer.digits.log_basis
                && candidate.opening.opening_method == guide.opening.opening_method
                && candidate.opening.fold_challenge_config == guide.opening.fold_challenge_config
                && candidate.opening.log_basis_open == guide.opening.log_basis_open
        }
        _ => false,
    }
}

fn recursive_candidate_matches_guide(
    candidate: &CommittedGroupParams,
    guide: &CommittedGroupParams,
) -> bool {
    candidate.payload_mode == guide.payload_mode
        && candidate.ring_relation_mode == guide.ring_relation_mode
        && candidate.source_encoding == guide.source_encoding
        && candidate.witness_chunk == guide.witness_chunk
        && candidate.role_dims() == guide.role_dims()
        && candidate.opening_method() == guide.opening_method()
        && candidate.fold_challenge_config() == guide.fold_challenge_config()
        && candidate.inner().digits.log_basis == guide.inner().digits.log_basis
        && candidate.outer().digits.log_basis == guide.outer().digits.log_basis
        && candidate.open().digits.log_basis == guide.open().digits.log_basis
        && candidate.blocks().positions_per_block == guide.blocks().positions_per_block
        && candidate.outer_slice_count() == guide.outer_slice_count()
        && inner_route_kind_matches(
            candidate.inner().matrix.security_route(),
            guide.inner().matrix.security_route(),
        )
        && setup_prefix_structure_matches(candidate.setup_prefix(), guide.setup_prefix())
}

fn terminal_candidate_matches_guide(
    candidate: &CommittedGroupParams,
    guide: &TerminalFoldParams,
) -> bool {
    candidate.d_a() == guide.d_a()
        && candidate.blocks().positions_per_block == guide.blocks.positions_per_block
        && candidate.inner().digits.log_basis == guide.inner.digits.log_basis
        && candidate.open().digits.log_basis == guide.fold.log_basis
        && candidate.opening_method() == akita_types::OpeningMethod::EvaluationTrace
        && matches!(
            candidate.inner().matrix.security_route(),
            akita_types::InnerCommitSecurityRoute::Linf(_)
        )
        && candidate.setup_prefix().is_none()
}

impl<'a> CandidateDomain<'a> {
    pub(super) fn prepare(ctx: &SuffixCtx<'a>, state: SuffixState) -> Result<Self, AkitaError> {
        let policy = ctx.policy;
        let root_level_key = ctx.root_lookup_key.filter(|_| state.level == 0);
        let root_main_constraint = ctx.root_main_constraint.filter(|_| state.level == 0);
        let (guide_fold, guide_terminal) = if let Some(guide) = ctx.adaptation_guide {
            let fold = if state.level == 0 {
                Some(&guide.root.params)
            } else {
                guide
                    .recursive_folds
                    .get(state.level.saturating_sub(1))
                    .map(|step| &step.params)
            };
            let terminal_level = guide.recursive_folds.len() + 1;
            let terminal = (state.level == terminal_level).then_some(&guide.terminal);
            if fold.is_none() && terminal.is_none() {
                return Err(AkitaError::UnsupportedSchedule(format!(
                    "adapted schedule reached level {} outside its frozen depth {terminal_level}",
                    state.level
                )));
            }
            (fold, terminal)
        } else {
            (None, None)
        };
        let incoming_setup_prefix = state.topology.incoming_setup_prefix();
        if let Some(guide) = guide_fold {
            if incoming_setup_prefix.is_some() != guide.setup_prefix().is_some() {
                return Err(AkitaError::UnsupportedSchedule(format!(
                    "adapted schedule cannot reproduce the frozen setup topology at level {}",
                    state.level
                )));
            }
        } else if guide_terminal.is_some() && incoming_setup_prefix.is_some() {
            return Err(AkitaError::UnsupportedSchedule(
                "adapted schedule cannot offload setup directly into its terminal fold".into(),
            ));
        }
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
        let (allowed_min_inner_basis, allowed_max_inner_basis) =
            inner_source.search_range(policy)?;
        let (allowed_min_open_basis, allowed_max_open_basis) =
            crate::policy::log_basis_search_range_at_level(policy, state.level);
        let guided_bases = root_main_constraint
            .or(guide_fold)
            .map(|constraint| {
                (
                    constraint.inner().digits.log_basis,
                    constraint.open().digits.log_basis,
                )
            })
            .or_else(|| {
                guide_terminal
                    .map(|terminal| (terminal.inner.digits.log_basis, terminal.fold.log_basis))
            });
        let (min_inner_basis, max_inner_basis, min_open_basis, max_open_basis) =
            if let Some((inner, open)) = guided_bases {
                if !(allowed_min_inner_basis..=allowed_max_inner_basis).contains(&inner)
                    || !(allowed_min_open_basis.max(state.current_lb)..=allowed_max_open_basis)
                        .contains(&open)
                {
                    return Err(AkitaError::UnsupportedSchedule(
                        "adapted schedule bases are outside the current planner policy".into(),
                    ));
                }
                (inner, inner, open, open)
            } else {
                (
                    allowed_min_inner_basis,
                    allowed_max_inner_basis,
                    allowed_min_open_basis,
                    allowed_max_open_basis,
                )
            };
        let opening_work = opening_work_domain(
            ctx,
            state,
            root_level_key,
            root_main_constraint,
            guide_fold,
            guide_terminal,
            opening_shape,
        )?;
        let retain_split_frontier = state.topology.incoming_setup_prefix().is_some()
            || policy.selection_policy == crate::SelectionPolicyId::MinEstimatedProofPayloadV2
            || matches!(
                policy.ring_dimension_schedule_mode,
                crate::RingDimensionScheduleMode::AdaptiveDimension {
                    num_search_levels,
                    ..
                } if state.level < num_search_levels
            );
        let fold_policy = if ctx.adaptation_guide.is_some() {
            FoldCandidatePolicy::Best
        } else if retain_split_frontier {
            FoldCandidatePolicy::Frontier(SplitBoundPolicy::Enabled)
        } else {
            FoldCandidatePolicy::Best
        };
        let require_child_fold =
            root_level_key.is_some_and(|root_key| !root_key.precommitteds.is_empty());

        Ok(Self {
            root_level_key,
            root_main_constraint,
            guide_fold,
            guide_terminal,
            adaptation_guided: ctx.adaptation_guide.is_some(),
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
                    let mut dimension_candidates = root_level_candidates_for_basis(
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
                    if let Some(constraint) = self.root_main_constraint {
                        dimension_candidates.retain(|(params, _)| {
                            root_candidate_matches_constraint(params, constraint)
                        });
                    }
                    let relation_domain = state
                        .topology
                        .relation_domain(state.level, work.opening.method())?
                        .filtered(ctx.relation_mode_filter)?;
                    let relation_transition = relation_domain.only_transition()?;
                    for (params, next_witness_len) in dimension_candidates {
                        if (!self.adaptation_guided || self.guide_terminal.is_some())
                            && work.purpose.allows_terminal()
                        {
                            terminal.push(RawTerminalCandidate {
                                params: params.clone(),
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            });
                        }
                        if (!self.adaptation_guided || self.guide_fold.is_some())
                            && work.purpose.allows_fold()
                        {
                            folds.push(RawFoldCandidate {
                                relation_transition,
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
                    if self
                        .guide_fold
                        .is_some_and(|guide| payload_mode != guide.payload_mode)
                    {
                        continue;
                    }
                    let guide = self
                        .guide_fold
                        .map(|guide| RecursiveCandidateGuide {
                            position_index_bits: guide.blocks().position_index_bits(),
                            outer_slice_count: guide.outer_slice_count(),
                        })
                        .or_else(|| {
                            self.guide_terminal.map(|guide| RecursiveCandidateGuide {
                                position_index_bits: guide.blocks.position_index_bits(),
                                outer_slice_count: akita_types::CommitmentSliceCount::ONE,
                            })
                        });
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
                        guide,
                    };
                    let relation_domain = state
                        .topology
                        .relation_domain(state.level, work.opening.method())?
                        .filtered(ctx.relation_mode_filter)?;
                    if work.purpose == OpeningPurpose::TerminalAndFold {
                        let views = derive_recursive_candidate_views(
                            request,
                            self.fold_policy,
                            relation_domain,
                        )?;
                        terminal.extend(views.terminal.into_iter().filter_map(|params| {
                            (!self.adaptation_guided
                                || self.guide_terminal.is_some_and(|guide| {
                                    terminal_candidate_matches_guide(&params, guide)
                                }))
                            .then_some(RawTerminalCandidate {
                                params,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            })
                        }));
                        for (candidate, next_witness_len) in views.folds {
                            if !relation_domain.admits(candidate.relation_transition) {
                                return Err(AkitaError::InvalidSetup(
                                    "combined recursive view emitted a fold outside its relation domain"
                                        .into(),
                                ));
                            }
                            if self.adaptation_guided
                                && !self.guide_fold.is_some_and(|guide| {
                                    recursive_candidate_matches_guide(&candidate.params, guide)
                                })
                            {
                                continue;
                            }
                            folds.push(RawFoldCandidate {
                                relation_transition: candidate.relation_transition,
                                params: candidate.params,
                                next_witness_len,
                                opening_reduction_bytes: work.opening_reduction_bytes,
                            });
                        }
                        continue;
                    }
                    if work.purpose.allows_terminal() {
                        terminal.extend(
                            derive_terminal_candidates(request)?
                                .into_iter()
                                .filter_map(|params| {
                                    (!self.adaptation_guided
                                        || self.guide_terminal.is_some_and(|guide| {
                                            terminal_candidate_matches_guide(&params, guide)
                                        }))
                                    .then_some(
                                        RawTerminalCandidate {
                                            params,
                                            opening_reduction_bytes: work.opening_reduction_bytes,
                                        },
                                    )
                                }),
                        );
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
                        if self.adaptation_guided
                            && !self.guide_fold.is_some_and(|guide| {
                                recursive_candidate_matches_guide(&candidate.params, guide)
                            })
                        {
                            continue;
                        }
                        folds.push(RawFoldCandidate {
                            relation_transition: candidate.relation_transition,
                            params: candidate.params,
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
                let mut dimension_candidates = root_level_candidates_for_basis(
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
                if let Some(constraint) = self.root_main_constraint {
                    dimension_candidates.retain(|(params, _)| {
                        root_candidate_matches_constraint(params, constraint)
                    });
                }
                let relation_transition = state
                    .topology
                    .relation_domain(state.level, work.opening.method())?
                    .filtered(ctx.relation_mode_filter)?
                    .only_transition()?;
                let mut terminal = Vec::new();
                let mut folds = Vec::new();
                for (params, next_witness_len) in dimension_candidates {
                    if (!self.adaptation_guided || self.guide_terminal.is_some())
                        && work.purpose.allows_terminal()
                    {
                        terminal.push(RawTerminalCandidate {
                            params: params.clone(),
                            opening_reduction_bytes: work.opening_reduction_bytes,
                        });
                    }
                    if (!self.adaptation_guided || self.guide_fold.is_some())
                        && work.purpose.allows_fold()
                    {
                        folds.push(RawFoldCandidate {
                            relation_transition,
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
