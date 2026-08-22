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
pub(crate) struct SuffixResult {
    pub(super) payload_only: BTreeMap<ParentObservableKey, Vec<ScheduleCandidate>>,
    pub(super) setup_and_payload: BTreeMap<ParentObservableKey, frontier::ObjectiveChoices>,
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

// Completed frontier entries omit construction-only descriptors. The larger
// quota stayed within the former peak for the measured high pressure row.
const MAX_SUFFIX_SEARCH_CACHE_ENTRIES: usize = 524_288;
// Prefix layouts create a much wider stream of one-off states than ordinary
// suffixes. Separate quotas keep that stream from evicting direct states while
// preserving a hard bound on the completed exact-DP cache.
const MAX_DIRECT_SUFFIX_CACHE_ENTRIES: usize = 393_216;
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

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(super) fn contains(&self, key: &ScheduleMemoKey) -> bool {
        self.entries.contains_key(key)
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

    pub(crate) fn setup_prefix_cache_diagnostics(&self) -> (usize, usize) {
        self.setup_prefixes.diagnostics()
    }
}

pub(super) fn empty_suffix_result() -> Arc<SuffixResult> {
    Arc::new(SuffixResult {
        payload_only: BTreeMap::new(),
        setup_and_payload: BTreeMap::new(),
    })
}

/// DP-invariant inputs for the suffix search.
///
/// Values that remain constant across the whole recursion are carried in one
/// context value rather than as per-call arguments.
#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) diagnostics: Option<&'a crate::diagnostics::PlannerDiagnostics>,
    pub(crate) ring_challenge_config:
        &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
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
