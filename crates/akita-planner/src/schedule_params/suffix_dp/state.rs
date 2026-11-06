use super::*;
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "parallel")]
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, MutexGuard, PoisonError};

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
/// The parent prices the child's commitment and optional Stage-3 setup-prefix
/// payloads, and uses its grinding geometry and source-block width to price
/// the parent fold. Other matrix and opening choices remain part of the full
/// schedule for the canonical tie-break, but cannot affect the parent edge price.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ParentObservableKey {
    outer_payload_bytes: usize,
    setup_prefix_payload_bytes: usize,
    grinding_successor: GrindingSuccessorKey,
    /// Successor source-block coefficient width used to align multi-chunk
    /// predecessor bodies. Retained policy-wide whenever multi-chunk folds
    /// are enabled, including states whose predecessor has already contracted.
    source_block_len: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GrindingSuccessorKey {
    Recursive {
        d_a: usize,
        opening_vars: usize,
        stage3_rounds: usize,
    },
    Terminal {
        d_a: usize,
        opening_vars: usize,
        live_blocks: u64,
        /// `log2` size of the terminal challenge set that prices the
        /// terminal fold proof of work.
        challenge_set_bits: u32,
    },
}

impl ParentObservableKey {
    pub(super) fn new(
        policy: &PlannerPolicy,
        recursive: Option<&akita_params::CommittedGroupParams>,
        terminal: Option<&CandidateTerminalResponse>,
    ) -> Result<Self, AkitaError> {
        if recursive.is_some() == terminal.is_some() {
            return Err(AkitaError::InvalidSetup(
                "parent key requires exactly one successor".into(),
            ));
        }
        let Some(first) = recursive else {
            let CandidateTerminalResponse {
                params: terminal,
                fold_cost,
                ..
            } = terminal.ok_or_else(|| {
                AkitaError::InvalidSetup("parent key is missing its terminal successor".into())
            })?;
            return Ok(Self {
                source_block_len: (policy.chunks_at_level(0) > 1)
                    .then(|| akita_params::FoldSuccessor::Terminal(terminal).source_block_len())
                    .transpose()?,
                outer_payload_bytes: 0,
                setup_prefix_payload_bytes: 0,
                grinding_successor: GrindingSuccessorKey::Terminal {
                    d_a: terminal.d_a(),
                    opening_vars: terminal.recursive_opening_num_vars()?,
                    live_blocks: u64::try_from(terminal.blocks.live_blocks).map_err(|_| {
                        AkitaError::InvalidSetup(
                            "terminal live-block count exceeds grinding query width".into(),
                        )
                    })?,
                    challenge_set_bits: fold_cost.challenge_set_bits(),
                },
            });
        };
        let payload = first.outer_payload_geometry()?;
        let outer_payload_bytes = payload
            .transmitted_coefficients()
            .checked_mul(akita_params::layout::proof_size::field_bytes(
                policy.decomposition.field_bits(),
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("outer payload byte count overflow".into()))?;
        Ok(Self {
            source_block_len: (policy.chunks_at_level(0) > 1)
                .then(|| akita_params::FoldSuccessor::Recursive(first).source_block_len())
                .transpose()?,
            outer_payload_bytes,
            setup_prefix_payload_bytes:
                akita_schedules::planner_support::stage3_payload_bytes_for_successor(
                    policy,
                    akita_params::FoldSuccessor::Recursive(first),
                )?,
            grinding_successor: GrindingSuccessorKey::Recursive {
                d_a: first.d_a(),
                opening_vars: first.recursive_opening_num_vars()?,
                stage3_rounds: first
                    .setup_prefix()
                    .map_or(0, |prefix| prefix.profile.group.num_vars()),
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ScheduleMemoKey {
    pub(super) level: usize,
    pub(super) current_witness_len: usize,
    /// Immutable producer geometry, shared with child states and eviction keys.
    pub(super) input_chunks: Option<Arc<akita_params::WitnessChunkShape>>,
    pub(super) current_lb: u32,
    pub(super) source_moment: Option<crate::response_model::SourceMomentEstimate>,
    pub(super) d_a: usize,
    pub(super) d_b: usize,
    pub(super) d_d: usize,
    pub(super) topology: SuffixTopology,
}

impl ScheduleMemoKey {
    const fn is_direct(&self) -> bool {
        self.topology.incoming_setup_prefix().is_none()
    }
}

/// Completed suffix states, shared by the search's worker threads.
///
/// An entry is the state's result as a function of its key alone, so which
/// thread computes it, and when, never changes what a reader sees. The
/// states are split over [`MEMO_SHARDS`] locks by key hash: one lock
/// serialized every worker's lookups (about a tenth of the pool's time was
/// spent waiting on it in `jolt:specs/whiteout-o2b-planner.md`'s profile).
pub(crate) struct ScheduleMemo {
    shards: Box<[Mutex<MemoStates>]>,
    pub(super) setup_prefixes: SetupPrefixSearchCache,
}

const MEMO_SHARDS: usize = 64;

#[derive(Default)]
struct MemoStates {
    entries: HashMap<ScheduleMemoKey, MemoEntry>,
    /// States some thread is completing now.
    #[cfg(feature = "parallel")]
    claimed: HashSet<ScheduleMemoKey>,
    direct_insertion_order: VecDeque<ScheduleMemoKey>,
    prefixed_insertion_order: VecDeque<ScheduleMemoKey>,
}

pub(super) struct MemoEntry {
    pub(super) result: Arc<SuffixResult>,
    pub(super) referenced: bool,
}

// Completed frontier entries omit construction-only descriptors and share
// incoming ownership shapes between the map, eviction queue, and child states.
const MAX_SUFFIX_SEARCH_CACHE_ENTRIES: usize = 524_288;
// Prefix layouts create a much wider stream of one-off states than ordinary
// suffixes. Separate quotas keep that stream from evicting direct states while
// preserving a hard bound on the completed exact-DP cache.
const MAX_DIRECT_SUFFIX_CACHE_ENTRIES: usize = 393_216;
const MAX_PREFIXED_SUFFIX_CACHE_ENTRIES: usize =
    MAX_SUFFIX_SEARCH_CACHE_ENTRIES - MAX_DIRECT_SUFFIX_CACHE_ENTRIES;
// Each shard holds its share of each quota.
const MAX_DIRECT_ENTRIES_PER_SHARD: usize = MAX_DIRECT_SUFFIX_CACHE_ENTRIES / MEMO_SHARDS;
const MAX_PREFIXED_ENTRIES_PER_SHARD: usize = MAX_PREFIXED_SUFFIX_CACHE_ENTRIES / MEMO_SHARDS;
const MAX_SECOND_CHANCE_PROBES: usize = 16;

#[cfg(test)]
#[test]
#[ignore = "manual peak-RSS measurement with both cache quotas saturated"]
fn memo_cache_pressure() {
    let multi = std::env::var_os("AKITA_MEMO_MULTI_CHUNK").is_some();
    let memo = ScheduleMemo::new();
    let result = empty_suffix_result();
    for (topology, capacity) in [
        (
            SuffixTopology::Direct {
                payload_phase: akita_params::CommitmentPayloadPhase::CompressedPrefix,
                relation_phase: RingRelationPhase::QuotientPrefix,
            },
            MAX_DIRECT_SUFFIX_CACHE_ENTRIES,
        ),
        (
            SuffixTopology::SetupPrefixed { natural_len: 64 },
            MAX_PREFIXED_SUFFIX_CACHE_ENTRIES,
        ),
    ] {
        for index in 0..capacity + 1 {
            // Give every multi-chunk state its own allocation: even without
            // sharing between sibling states, map/deque copies share the shape.
            let input_chunks = multi.then(|| {
                let mut body_lengths = [64; akita_params::MAX_WITNESS_CHUNKS];
                body_lengths[0] += index;
                Arc::new(akita_params::WitnessChunkShape {
                    body_lengths,
                    num_chunks: 8,
                    tail_prefix_len: 0,
                    tail_alignment: 1,
                    tail_len: 5,
                    tail_suffix_alignment: 1,
                })
            });
            memo.insert(
                ScheduleMemoKey {
                    level: 1,
                    current_witness_len: 1024 + index,
                    input_chunks,
                    current_lb: 3,
                    source_moment: None,
                    d_a: 64,
                    d_b: 64,
                    d_d: 64,
                    topology,
                },
                Arc::clone(&result),
                None,
            );
        }
    }
    // Each shard evicts at its own share of the quota, so a shard that
    // hashes more keys than its share evicts before the memo as a whole is
    // full: the total is bounded by the quota, not equal to it.
    assert!(memo.len() <= MAX_SUFFIX_SEARCH_CACHE_ENTRIES);
    eprintln!(
        "multi={multi}, memo_entries={}, key_bytes={}",
        memo.len(),
        std::mem::size_of::<ScheduleMemoKey>()
    );
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
            shards: (0..MEMO_SHARDS)
                .map(|_| Mutex::new(MemoStates::default()))
                .collect(),
            setup_prefixes: SetupPrefixSearchCache::default(),
        }
    }

    fn states(&self, key: &ScheduleMemoKey) -> MutexGuard<'_, MemoStates> {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        // The remainder is below `MEMO_SHARDS`, so it fits `usize`.
        let shard = (hasher.finish() % MEMO_SHARDS as u64) as usize;
        self.shards[shard]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn all_states(&self) -> impl Iterator<Item = MutexGuard<'_, MemoStates>> {
        self.shards
            .iter()
            .map(|shard| shard.lock().unwrap_or_else(PoisonError::into_inner))
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.all_states().map(|states| states.entries.len()).sum()
    }

    #[cfg(test)]
    pub(super) fn contains(&self, key: &ScheduleMemoKey) -> bool {
        self.states(key).entries.contains_key(key)
    }

    /// Whether `key` is completed, or some thread is completing it.
    #[cfg(feature = "parallel")]
    pub(super) fn is_completed_or_claimed(&self, key: &ScheduleMemoKey) -> bool {
        let states = self.states(key);
        states.entries.contains_key(key) || states.claimed.contains(key)
    }

    /// Records that the caller is completing `key` until the claim drops.
    #[cfg(feature = "parallel")]
    /// A claim only steers which successors a thread starts; two threads
    /// may still complete one state, to the same result.
    pub(super) fn claim(&self, key: &ScheduleMemoKey) -> MemoClaim<'_> {
        self.states(key).claimed.insert(key.clone());
        MemoClaim {
            memo: self,
            key: key.clone(),
        }
    }

    pub(super) fn get(&self, key: &ScheduleMemoKey) -> Option<Arc<SuffixResult>> {
        self.states(key).entries.get_mut(key).map(|entry| {
            entry.referenced = true;
            Arc::clone(&entry.result)
        })
    }

    pub(super) fn insert(
        &self,
        key: ScheduleMemoKey,
        result: Arc<SuffixResult>,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
    ) {
        let mut states = self.states(&key);
        let MemoStates {
            entries,
            direct_insertion_order,
            prefixed_insertion_order,
            ..
        } = &mut *states;
        if let Entry::Occupied(mut existing) = entries.entry(key.clone()) {
            existing.insert(MemoEntry {
                result,
                referenced: true,
            });
            return;
        }
        let (insertion_order, capacity) = if key.is_direct() {
            (&mut *direct_insertion_order, MAX_DIRECT_ENTRIES_PER_SHARD)
        } else {
            (
                &mut *prefixed_insertion_order,
                MAX_PREFIXED_ENTRIES_PER_SHARD,
            )
        };
        if insertion_order.len() >= capacity {
            evict_suffix_entry(entries, insertion_order);
        }
        insertion_order.push_back(key.clone());
        entries.insert(
            key,
            MemoEntry {
                result,
                referenced: false,
            },
        );
        drop(states);
        // Diagnostics only (`diagnostics` is `None` outside a captured
        // planner run): totals need every shard's lock, so a search pays it
        // per insert only when it asked for diagnostics.
        if let Some(diagnostics) = diagnostics {
            let (mut total, mut direct, mut prefixed) = (0, 0, 0);
            for states in self.all_states() {
                total += states.entries.len();
                direct += states.direct_insertion_order.len();
                prefixed += states.prefixed_insertion_order.len();
            }
            diagnostics.record_memo_occupancy(total, direct, prefixed);
        }
    }

    pub(crate) fn setup_prefix_cache_diagnostics(&self) -> (usize, usize) {
        self.setup_prefixes.diagnostics()
    }
}

#[cfg(feature = "parallel")]
pub(super) struct MemoClaim<'a> {
    memo: &'a ScheduleMemo,
    key: ScheduleMemoKey,
}

#[cfg(feature = "parallel")]
impl Drop for MemoClaim<'_> {
    fn drop(&mut self) {
        self.memo.states(&self.key).claimed.remove(&self.key);
    }
}

pub(super) fn empty_suffix_result() -> Arc<SuffixResult> {
    Arc::new(SuffixResult {
        payload_only: BTreeMap::new(),
        setup_and_payload: BTreeMap::new(),
    })
}

/// Upper bound on a schedule's total setup field elements for one search.
///
/// Every rejection records its setup size, so a caller whose search found
/// nothing can learn the smallest size it excluded.
pub(crate) struct SetupFieldBudget {
    limit: usize,
    // Successor states complete in parallel, so rejections are recorded
    // under a lock.
    min_rejected: Mutex<Option<usize>>,
}

impl SetupFieldBudget {
    pub(crate) const fn new(limit: usize) -> Self {
        Self {
            limit,
            min_rejected: Mutex::new(None),
        }
    }

    /// Whether `setup_field_elements` fits, recording it when it does not.
    pub(crate) fn admits(&self, setup_field_elements: usize) -> bool {
        if setup_field_elements <= self.limit {
            return true;
        }
        let mut min_rejected = self
            .min_rejected
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *min_rejected =
            Some(min_rejected.map_or(setup_field_elements, |min| min.min(setup_field_elements)));
        false
    }

    /// Smallest setup size this budget rejected, if any.
    pub(crate) fn min_rejected(&self) -> Option<usize> {
        *self
            .min_rejected
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// DP-invariant inputs for the suffix search.
///
/// Values that remain constant across the whole recursion are carried in one
/// context value rather than as per-call arguments.
#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) challenge_order: akita_params::ChallengeFieldOrder,
    pub(crate) diagnostics: Option<&'a crate::diagnostics::PlannerDiagnostics>,
    pub(crate) ring_challenge_config: &'a (dyn Fn(usize) -> Result<akita_challenges::FoldChallengeDistribution, AkitaError>
             + Sync),
    pub(crate) key: PolynomialGroupLayout,
    pub(crate) setup_field_budget: Option<&'a SetupFieldBudget>,
    pub(crate) root_lookup_key: Option<&'a ScheduleLookupKey>,
    /// Optional exact main-group root selected by an earlier scalar plan.
    ///
    /// Adapted grouped planning keeps this root's own A/B geometry and opening
    /// plan, while allowing the fold-owned shared D matrix and every successor
    /// level to be rebuilt for the added groups.
    pub(crate) root_main_constraint: Option<&'a CommittedGroupParams>,
    /// Approved scalar schedule whose structural suffix choices guide adapted
    /// planning. All length-, rank-, and security-derived values are rebuilt.
    pub(crate) adaptation_guide: Option<&'a akita_params::FoldSchedule>,
    /// Extension seam: restrict a caller-selected root setup edge without
    /// changing canonical planning; <https://github.com/LayerZero-Labs/akita/issues/285>.
    pub(crate) root_setup_mode: Option<akita_params::SetupContributionMode>,
    pub(crate) root_source_contract: Option<akita_params::sis::CommittedSourceContract>,
    pub(crate) precommitted_source_contracts: &'a [akita_params::sis::CommittedSourceContract],
    pub(crate) level_zero_is_root: bool,
    pub(crate) relation_traversal_order: RelationTraversalOrder,
    pub(crate) relation_mode_filter: RelationModeFilter,
    /// Zero-knowledge cut whose levels are planned masked; `None` plans a
    /// clear schedule.
    pub(crate) zk: Option<ZkSearch>,
}

/// A zero-knowledge search: the cut, and how the edge into the cut level
/// treats a setup prefix.
#[derive(Clone, Copy)]
pub(crate) struct ZkSearch {
    pub(crate) schedule: akita_params::zk::ZkOpeningSchedule,
    pub(crate) cut_offload: CutOffload,
}

impl ZkSearch {
    /// Whether the edge from `level` into the next one must offload the
    /// parent's setup to a setup prefix the cut level consumes.
    pub(crate) fn forces_offload_from(self, level: usize) -> bool {
        matches!(self.cut_offload, CutOffload::Forced) && self.enters_cut_from(level)
    }

    /// Whether the edge from `level` into the next one must stay direct.
    pub(crate) fn forbids_offload_from(self, level: usize) -> bool {
        matches!(self.cut_offload, CutOffload::Direct) && self.enters_cut_from(level)
    }

    fn enters_cut_from(self, level: usize) -> bool {
        level + 1 == self.schedule.cut_level() as usize
    }
}

/// Whether the cut level consumes a setup prefix.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    not(feature = "zk"),
    expect(
        dead_code,
        reason = "zero-knowledge searches are planned only under zk"
    )
)]
pub(crate) enum CutOffload {
    /// The edge into the cut level is direct: the cut level never consumes a
    /// setup prefix. A setup prefix there adds the parent's response and
    /// inner digits to w_k, which the planner's objective does not price:
    /// they grow the Σ-protocol's private vector and with it the response
    /// vector `e`, past `jolt_zk::whiteout::MAX_RESPONSE_NUM_VARS` on the
    /// 36:1 headline over an `R_128` residual
    /// (`specs/whiteout-akita.md` §5.3).
    Direct,
    /// Test support: the edge into the cut level is offloaded only (no
    /// direct candidate), so a test can prove a masked Stage 3 at the cut.
    #[cfg_attr(
        all(feature = "zk", not(feature = "test-support")),
        expect(
            dead_code,
            reason = "constructed only by the test-support planner entry"
        )
    )]
    Forced,
}

#[derive(Clone, Copy)]
pub(crate) struct SuffixState<'a> {
    pub(crate) level: usize,
    pub(crate) current_witness_len: usize,
    pub(crate) input_chunks: Option<&'a Arc<akita_params::WitnessChunkShape>>,
    pub(crate) current_lb: u32,
    pub(crate) source_moment: Option<crate::response_model::SourceMomentEstimate>,
    pub(crate) dimension_ceiling: CommitmentRingDims,
    pub(crate) topology: SuffixTopology,
}

/// Complete suffix topology. Prefix-bearing states cannot also carry raw
/// payload or reduced-relation phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SuffixTopology {
    Direct {
        payload_phase: akita_params::CommitmentPayloadPhase,
        relation_phase: RingRelationPhase,
    },
    SetupPrefixed {
        natural_len: usize,
    },
}

impl SuffixTopology {
    #[must_use]
    pub(crate) const fn incoming_setup_prefix(self) -> Option<usize> {
        match self {
            Self::Direct { .. } => None,
            Self::SetupPrefixed { natural_len } => Some(natural_len),
        }
    }

    #[must_use]
    pub(crate) const fn payload_phase(self) -> akita_params::CommitmentPayloadPhase {
        match self {
            Self::Direct { payload_phase, .. } => payload_phase,
            Self::SetupPrefixed { .. } => akita_params::CommitmentPayloadPhase::CompressedPrefix,
        }
    }

    pub(crate) fn relation_domain(
        self,
        absolute_level: usize,
        witness_chunk: akita_params::ChunkedWitnessCfg,
        opening: akita_params::OpeningMethod,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
    ) -> Result<RelationSearchDomain, AkitaError> {
        let (relation_phase, consumes_setup_prefix) = match self {
            Self::Direct { relation_phase, .. } => (relation_phase, false),
            Self::SetupPrefixed { .. } => (RingRelationPhase::QuotientPrefix, true),
        };
        RelationSearchDomain::for_topology(
            relation_phase,
            absolute_level,
            witness_chunk,
            RelationCandidateTopology::new(consumes_setup_prefix, opening),
            diagnostics,
        )
    }

    #[must_use]
    pub(crate) const fn relation_phase(self) -> RingRelationPhase {
        match self {
            Self::Direct { relation_phase, .. } => relation_phase,
            Self::SetupPrefixed { .. } => RingRelationPhase::QuotientPrefix,
        }
    }

    #[must_use]
    pub(crate) const fn direct_successor(
        self,
        payload_mode: akita_params::CommitmentPayloadMode,
        transition: akita_params::RingRelationMode,
    ) -> Self {
        Self::Direct {
            payload_phase: self.payload_phase().after(payload_mode),
            relation_phase: self.relation_phase().after(transition),
        }
    }

    #[must_use]
    pub(crate) const fn offloaded_successor(
        transition: akita_params::RingRelationMode,
        payload_mode: akita_params::CommitmentPayloadMode,
        natural_len: usize,
    ) -> Option<Self> {
        if !transition.is_reduced_evaluation() && payload_mode.is_compressed() {
            Some(Self::SetupPrefixed { natural_len })
        } else {
            None
        }
    }
}

impl SuffixState<'_> {
    pub(super) fn memo_key(self, policy: &PlannerPolicy) -> ScheduleMemoKey {
        let domain = policy.ring_dimension_schedule;
        let memo_dimensions = if self.level >= domain.num_search_levels {
            crate::schedule_params::suffix_dimension_ceiling(
                domain.suffix_dimensions,
                self.dimension_ceiling,
            )
            .map_or(self.dimension_ceiling, CommitmentRingDims::uniform)
        } else {
            self.dimension_ceiling
        };
        ScheduleMemoKey {
            level: self.level,
            current_witness_len: self.current_witness_len,
            input_chunks: self.input_chunks.cloned(),
            current_lb: self.current_lb,
            source_moment: self.source_moment,
            d_a: memo_dimensions.d_a(),
            d_b: memo_dimensions.d_b(),
            d_d: memo_dimensions.d_d(),
            topology: self.topology,
        }
    }
}
