use super::*;

/// Result of the suffix DP at one state. Each selection objective retains the
/// candidates its parent needs because proof-size and setup-envelope pricing
/// depend on the child's first step:
///
/// - setup and payload winners keyed by the parent-visible first fold. Direct
///   states store only payload winners; prefix/root states share each key
///   between both projections. The setup projection is lexicographically best
///   by first direct setup scan and then proof payload. The payload projection
///   is the smallest-payload schedule used after an earlier direct edge has
///   fixed the setup-size objective.
/// - `mixed_frontier` — nondominated setup-envelope/proof candidates for the
///   direct adaptive-dimension objective.
pub(crate) struct SuffixResult {
    pub(super) payload_only: BTreeMap<ParentObservableKey, Vec<ScheduleCandidate>>,
    pub(super) setup_and_payload: BTreeMap<ParentObservableKey, frontier::ObjectiveChoices>,
    /// Nondominated setup-envelope/proof candidates used by adaptive scalar
    /// planning, bucketed by the exact geometry visible to their parent.
    pub(crate) mixed_frontier: MixedFrontier,
}

pub(crate) struct MixedFrontier {
    by_parent: BTreeMap<ParentObservableKey, Vec<ScheduleCandidate>>,
}

impl MixedFrontier {
    pub(super) const fn new() -> Self {
        Self {
            by_parent: BTreeMap::new(),
        }
    }

    pub(crate) fn candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.by_parent.values().flatten()
    }
}

impl SuffixResult {
    pub(crate) fn payload_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.payload_only.values().flatten().chain(
            self.setup_and_payload
                .values()
                .flat_map(frontier::ObjectiveChoices::payload_candidates),
        )
    }

    pub(crate) fn setup_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.setup_and_payload
            .values()
            .flat_map(frontier::ObjectiveChoices::setup_candidates)
    }
}

fn mixed_score(candidate: &ScheduleCandidate) -> MixedScore {
    MixedScore {
        setup_field_elements: candidate.setup_field_elements,
        proof_bytes: candidate.total_bytes,
    }
}

pub(super) fn dominates_mixed_score(left: MixedScore, right: MixedScore) -> bool {
    left.setup_field_elements <= right.setup_field_elements
        // A setup-only improvement cannot prune `right`: a parent can mask
        // both setup footprints, leaving proof bytes and the descriptor to
        // decide the complete schedule.
        && left.proof_bytes < right.proof_bytes
}

pub(super) fn insert_mixed_frontier(
    policy: &PlannerPolicy,
    frontier: &mut MixedFrontier,
    candidate: ScheduleCandidate,
) -> Result<(), AkitaError> {
    if policy.selection_policy
        != crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
        || !policy.admits_setup_field_elements(candidate.setup_field_elements)
    {
        return Ok(());
    }
    let candidate_key = ParentObservableKey::new(policy, candidate.first_fold_params())?;
    let bucket = frontier.by_parent.entry(candidate_key).or_default();
    let mut dominated = false;
    let mut retained = Vec::with_capacity(bucket.len() + 1);
    for existing in bucket.drain(..) {
        if dominates_mixed_score(mixed_score(&existing), mixed_score(&candidate)) {
            dominated = true;
        }
        if !dominates_mixed_score(mixed_score(&candidate), mixed_score(&existing)) {
            retained.push(existing);
        }
    }
    if !dominated {
        retained.push(candidate);
    }
    *bucket = retained;
    Ok(())
}

/// Exact successor geometry visible to a parent fold.
///
/// The parent prices only the child's outgoing commitment payload and optional
/// Stage-3 setup-prefix payload. The child's other matrix and opening choices
/// remain part of the retained full schedule for the canonical tie-break, but
/// cannot affect the parent edge price.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ParentObservableKey {
    outer_payload_bytes: usize,
    setup_prefix_payload_bytes: usize,
}

impl ParentObservableKey {
    pub(super) fn new(
        policy: &PlannerPolicy,
        first: Option<&akita_types::CommittedGroupParams>,
    ) -> Result<Self, AkitaError> {
        let Some(first) = first else {
            return Ok(Self {
                outer_payload_bytes: 0,
                setup_prefix_payload_bytes: 0,
            });
        };
        let payload = first.outer_payload_geometry()?;
        let outer_payload_bytes = payload
            .transmitted_coefficients()
            .checked_mul(akita_types::layout::proof_size::field_bytes(
                policy.decomposition.field_bits(),
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("outer payload byte count overflow".into()))?;
        Ok(Self {
            outer_payload_bytes,
            setup_prefix_payload_bytes:
                akita_schedules::planner_support::stage3_payload_bytes_for_successor(
                    policy,
                    Some(first),
                )?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ScheduleMemoKey {
    pub(super) level: usize,
    pub(super) current_witness_len: usize,
    pub(super) current_lb: u32,
    pub(super) source_moment: Option<crate::response_model::SourceMomentEstimate>,
    pub(super) incoming_setup_prefix: Option<usize>,
    pub(super) d_a: usize,
    pub(super) d_b: usize,
    pub(super) d_d: usize,
    pub(super) payload_phase: akita_types::CommitmentPayloadPhase,
}

pub(crate) struct ScheduleMemo {
    // Every completed state is retained for the lifetime of one row search.
    // Evicting completed exact-DP states turns a wide packing search into
    // repeated subtree evaluation; the compact suffix frontiers and persistent
    // fold chains are the memory bound instead.
    entries: HashMap<ScheduleMemoKey, Arc<SuffixResult>>,
    pub(super) setup_prefixes: SetupPrefixSearchCache,
}

impl ScheduleMemo {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            setup_prefixes: SetupPrefixSearchCache::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(super) fn contains(&self, key: &ScheduleMemoKey) -> bool {
        self.entries.contains_key(key)
    }

    pub(super) fn get(&self, key: &ScheduleMemoKey) -> Option<&Arc<SuffixResult>> {
        self.entries.get(key)
    }

    pub(super) fn insert(&mut self, key: ScheduleMemoKey, result: Arc<SuffixResult>) {
        self.entries.insert(key, result);
    }
}

pub(super) fn empty_suffix_result() -> Arc<SuffixResult> {
    Arc::new(SuffixResult {
        payload_only: BTreeMap::new(),
        setup_and_payload: BTreeMap::new(),
        mixed_frontier: MixedFrontier::new(),
    })
}

/// DP-invariant inputs for the suffix search.
///
/// `policy`, the challenge-family provider, and `num_vars` are constant across the whole
/// recursion, so they are carried in one context value rather than as
/// per-call arguments (keeps the recursive signature small).
#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) ring_challenge_config:
        &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    pub(crate) num_vars: usize,
    pub(crate) key: PolynomialGroupLayout,
    pub(crate) setup_field_budget: Option<usize>,
    pub(crate) root_lookup_key: Option<&'a AkitaScheduleLookupKey>,
    pub(crate) root_honest_fold_policy: Option<akita_types::sis::HonestFoldPolicySpec>,
    pub(crate) precommitted_honest_fold_policies: &'a [akita_types::sis::HonestFoldPolicySpec],
    pub(crate) level_zero_is_root: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct SuffixState {
    pub(crate) level: usize,
    pub(crate) current_witness_len: usize,
    pub(crate) current_lb: u32,
    pub(crate) source_moment: Option<crate::response_model::SourceMomentEstimate>,
    pub(crate) incoming_setup_prefix: Option<usize>,
    pub(crate) dimension_ceiling: CommitmentRingDims,
    pub(crate) payload_phase: akita_types::CommitmentPayloadPhase,
}

impl SuffixState {
    pub(super) fn memo_key(self, policy: &PlannerPolicy) -> ScheduleMemoKey {
        let memo_dimensions = match policy.ring_dimension_schedule_mode {
            crate::RingDimensionScheduleMode::AdaptiveDimension {
                num_search_levels,
                suffix_dimensions,
                ..
            } if self.level >= num_search_levels => {
                crate::schedule_params::suffix_dimension_ceiling(
                    suffix_dimensions,
                    self.dimension_ceiling,
                )
                .map_or(self.dimension_ceiling, CommitmentRingDims::uniform)
            }
            _ => self.dimension_ceiling,
        };
        ScheduleMemoKey {
            level: self.level,
            current_witness_len: self.current_witness_len,
            current_lb: self.current_lb,
            source_moment: self.source_moment,
            incoming_setup_prefix: self.incoming_setup_prefix,
            d_a: memo_dimensions.d_a(),
            d_b: memo_dimensions.d_b(),
            d_d: memo_dimensions.d_d(),
            payload_phase: self.payload_phase,
        }
    }
}
