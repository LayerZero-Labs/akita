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
    pub(super) payload_only: BTreeMap<FirstFoldKey, ScheduleCandidate>,
    pub(super) setup_and_payload: BTreeMap<FirstFoldKey, frontier::ObjectiveChoices>,
    /// Nondominated setup-envelope/proof candidates used by adaptive scalar
    /// planning. Candidates with different first folds remain distinct because
    /// the parent proof price and canonical descriptor can distinguish them.
    pub(crate) mixed_frontier: Vec<ScheduleCandidate>,
}

impl SuffixResult {
    pub(crate) fn payload_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.payload_only.values().chain(
            self.setup_and_payload
                .values()
                .filter_map(|choices| choices.payload.as_ref()),
        )
    }

    pub(crate) fn setup_candidates(&self) -> impl Iterator<Item = &ScheduleCandidate> {
        self.setup_and_payload
            .values()
            .filter_map(|choices| choices.setup.as_ref())
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
    frontier: &mut Vec<ScheduleCandidate>,
    candidate: ScheduleCandidate,
) {
    if policy.selection_policy
        != crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
        || !policy.admits_setup_field_elements(candidate.setup_field_elements)
    {
        return;
    }
    crate::schedule_params::pareto::insert(frontier, candidate, |left, right| {
        left.first_fold_params() == right.first_fold_params()
            && dominates_mixed_score(mixed_score(left), mixed_score(right))
    });
}

/// Parent-visible first-fold class. A parent edge prices the child's outgoing
/// commitment payload, so suffixes with different first payload sizes are not
/// interchangeable even when they use the same digit basis.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FirstFoldKey {
    pub(super) descriptor: Option<Vec<u8>>,
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

impl ScheduleMemoKey {
    const fn is_direct(self) -> bool {
        self.incoming_setup_prefix.is_none()
    }
}

pub(crate) struct ScheduleMemo {
    entries: HashMap<ScheduleMemoKey, MemoEntry>,
    direct_insertion_order: VecDeque<ScheduleMemoKey>,
    prefixed_insertion_order: VecDeque<ScheduleMemoKey>,
    capacity: usize,
    pub(super) setup_prefixes: SetupPrefixSearchCache,
}

pub(super) struct MemoEntry {
    pub(super) result: Arc<SuffixResult>,
    pub(super) referenced: bool,
}

const MAX_SUFFIX_SEARCH_CACHE_ENTRIES: usize = 262_144;
// Direct and prefixed states share the hard total bound. Before the cache is
// full, either class may use free capacity. Once it fills, each insertion
// normally evicts from its own class, so the population split becomes
// phase-stable instead of allowing a wide prefix stream to churn hot direct
// states. The other class is used only when the inserted class is empty.
const MAX_SECOND_CHANCE_PROBES: usize = 16;

pub(super) fn eviction_uses_direct_queue(
    inserting_direct: bool,
    direct_has_entries: bool,
    prefixed_has_entries: bool,
) -> bool {
    (inserting_direct && direct_has_entries) || !prefixed_has_entries
}

pub(super) fn evict_suffix_entry(
    entries: &mut HashMap<ScheduleMemoKey, MemoEntry>,
    insertion_order: &mut VecDeque<ScheduleMemoKey>,
) {
    let mut probes = 0;
    while let Some(evicted) = insertion_order.pop_front() {
        let recently_referenced = probes < MAX_SECOND_CHANCE_PROBES
            && entries.get_mut(&evicted).is_some_and(|entry| {
                let referenced = entry.referenced;
                entry.referenced = false;
                referenced
            });
        if recently_referenced {
            insertion_order.push_back(evicted);
            probes += 1;
        } else {
            entries.remove(&evicted);
            break;
        }
    }
}

impl ScheduleMemo {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            direct_insertion_order: VecDeque::new(),
            prefixed_insertion_order: VecDeque::new(),
            capacity: MAX_SUFFIX_SEARCH_CACHE_ENTRIES,
            setup_prefixes: SetupPrefixSearchCache::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "suffix memo capacity must be nonzero");
        Self {
            capacity,
            ..Self::new()
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

    #[cfg(test)]
    pub(super) fn queue_lengths(&self) -> (usize, usize) {
        (
            self.direct_insertion_order.len(),
            self.prefixed_insertion_order.len(),
        )
    }

    #[cfg(test)]
    pub(super) fn internal_invariants_hold(&self) -> bool {
        let direct = self
            .direct_insertion_order
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let prefixed = self
            .prefixed_insertion_order
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        self.entries.len() <= self.capacity
            && self.direct_insertion_order.len() == direct.len()
            && self.prefixed_insertion_order.len() == prefixed.len()
            && direct.is_disjoint(&prefixed)
            && direct.len() + prefixed.len() == self.entries.len()
            && direct
                .iter()
                .all(|key| key.is_direct() && self.entries.contains_key(key))
            && prefixed
                .iter()
                .all(|key| !key.is_direct() && self.entries.contains_key(key))
            && self.entries.keys().all(|key| {
                if key.is_direct() {
                    direct.contains(key)
                } else {
                    prefixed.contains(key)
                }
            })
    }

    pub(super) fn get(&mut self, key: &ScheduleMemoKey) -> Option<&Arc<SuffixResult>> {
        self.entries.get_mut(key).map(|entry| {
            entry.referenced = true;
            &entry.result
        })
    }

    pub(super) fn insert(&mut self, key: ScheduleMemoKey, result: Arc<SuffixResult>) {
        if let Entry::Occupied(mut existing) = self.entries.entry(key) {
            existing.insert(MemoEntry {
                result,
                referenced: true,
            });
            return;
        }
        if self.entries.len() >= self.capacity {
            let evict_direct = eviction_uses_direct_queue(
                key.is_direct(),
                !self.direct_insertion_order.is_empty(),
                !self.prefixed_insertion_order.is_empty(),
            );
            let insertion_order = if evict_direct {
                &mut self.direct_insertion_order
            } else {
                &mut self.prefixed_insertion_order
            };
            evict_suffix_entry(&mut self.entries, insertion_order);
        }
        let insertion_order = if key.is_direct() {
            &mut self.direct_insertion_order
        } else {
            &mut self.prefixed_insertion_order
        };
        insertion_order.push_back(key);
        self.entries.insert(
            key,
            MemoEntry {
                result,
                referenced: false,
            },
        );
    }
}

pub(super) fn empty_suffix_result() -> Arc<SuffixResult> {
    Arc::new(SuffixResult {
        payload_only: BTreeMap::new(),
        setup_and_payload: BTreeMap::new(),
        mixed_frontier: Vec::new(),
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
