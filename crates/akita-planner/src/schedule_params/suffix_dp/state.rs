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
    pub(super) source_mean_l2_sq: Option<u128>,
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
    pub(super) setup_prefixes: SetupPrefixSearchCache,
}

pub(super) struct MemoEntry {
    pub(super) result: Arc<SuffixResult>,
    pub(super) referenced: bool,
}

const MAX_SUFFIX_SEARCH_CACHE_ENTRIES: usize = 262_144;
// Prefix layouts create a much wider stream of one-off states than ordinary
// suffixes. Separate FIFO quotas prevent that stream from evicting the direct
// states reused across basis, dimension, and split candidates while preserving
// the original hard bound on total cached results.
const MAX_DIRECT_SUFFIX_CACHE_ENTRIES: usize = 196_608;
const MAX_PREFIXED_SUFFIX_CACHE_ENTRIES: usize =
    MAX_SUFFIX_SEARCH_CACHE_ENTRIES - MAX_DIRECT_SUFFIX_CACHE_ENTRIES;
const MAX_SECOND_CHANCE_PROBES: usize = 16;

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
            setup_prefixes: SetupPrefixSearchCache::default(),
        }
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
        let (insertion_order, capacity) = if key.is_direct() {
            (
                &mut self.direct_insertion_order,
                MAX_DIRECT_SUFFIX_CACHE_ENTRIES,
            )
        } else {
            (
                &mut self.prefixed_insertion_order,
                MAX_PREFIXED_SUFFIX_CACHE_ENTRIES,
            )
        };
        if insertion_order.len() >= capacity {
            evict_suffix_entry(&mut self.entries, insertion_order);
        }
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
/// `policy`, `ring_challenge_cfg`, and `num_vars` are constant across the whole
/// recursion, so they are carried in one context value rather than as
/// per-call arguments (keeps the recursive signature small).
#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) default_ring_challenge_cfg: &'a akita_challenges::SparseChallengeConfig,
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
            source_mean_l2_sq: self.source_moment.map(|moment| moment.mean_l2_sq()),
            incoming_setup_prefix: self.incoming_setup_prefix,
            d_a: memo_dimensions.d_a(),
            d_b: memo_dimensions.d_b(),
            d_d: memo_dimensions.d_d(),
            payload_phase: self.payload_phase,
        }
    }
}
