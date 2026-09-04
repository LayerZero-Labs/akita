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
    grinding_successor: GrindingSuccessorKey,
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
    },
}

impl ParentObservableKey {
    pub(super) fn new(
        policy: &PlannerPolicy,
        recursive: Option<&akita_types::CommittedGroupParams>,
        terminal: Option<&akita_types::TerminalFoldParams>,
    ) -> Result<Self, AkitaError> {
        if recursive.is_some() == terminal.is_some() {
            return Err(AkitaError::InvalidSetup(
                "parent key requires exactly one successor".into(),
            ));
        }
        let Some(first) = recursive else {
            let terminal = terminal.ok_or_else(|| {
                AkitaError::InvalidSetup("parent key is missing its terminal successor".into())
            })?;
            return Ok(Self {
                outer_payload_bytes: 0,
                setup_prefix_payload_bytes: 0,
                grinding_successor: GrindingSuccessorKey::Terminal {
                    d_a: terminal.d_a(),
                    opening_vars: terminal.recursive_opening_num_vars()?,
                },
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
                    akita_types::FoldSuccessor::Recursive(first),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct ScheduleMemoKey {
    pub(super) level: usize,
    pub(super) current_witness_len: usize,
    pub(super) current_lb: u32,
    pub(super) source_moment: Option<crate::response_model::SourceMomentEstimate>,
    pub(super) d_a: usize,
    pub(super) d_b: usize,
    pub(super) d_d: usize,
    pub(super) topology: SuffixTopology,
}

impl ScheduleMemoKey {
    const fn is_direct(self) -> bool {
        self.topology.incoming_setup_prefix().is_none()
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

    pub(super) fn insert(
        &mut self,
        key: ScheduleMemoKey,
        result: Arc<SuffixResult>,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
    ) {
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
        if let Some(diagnostics) = diagnostics {
            diagnostics.record_memo_occupancy(
                self.entries.len(),
                self.direct_insertion_order.len(),
                self.prefixed_insertion_order.len(),
            );
        }
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
    pub(crate) relation_traversal_order: RelationTraversalOrder,
    pub(crate) relation_mode_filter: RelationModeFilter,
}

#[derive(Clone, Copy)]
pub(crate) struct SuffixState {
    pub(crate) level: usize,
    pub(crate) current_witness_len: usize,
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
        payload_phase: akita_types::CommitmentPayloadPhase,
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
    pub(crate) const fn payload_phase(self) -> akita_types::CommitmentPayloadPhase {
        match self {
            Self::Direct { payload_phase, .. } => payload_phase,
            Self::SetupPrefixed { .. } => akita_types::CommitmentPayloadPhase::CompressedPrefix,
        }
    }

    pub(crate) fn relation_domain(
        self,
        absolute_fold_level: usize,
        opening: akita_types::OpeningMethod,
        diagnostics: Option<&crate::diagnostics::PlannerDiagnostics>,
    ) -> Result<RelationSearchDomain, AkitaError> {
        let (relation_phase, consumes_setup_prefix) = match self {
            Self::Direct { relation_phase, .. } => (relation_phase, false),
            Self::SetupPrefixed { .. } => (RingRelationPhase::QuotientPrefix, true),
        };
        RelationSearchDomain::for_topology(
            relation_phase,
            absolute_fold_level,
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
        payload_mode: akita_types::CommitmentPayloadMode,
        transition: akita_types::RingRelationMode,
    ) -> Self {
        Self::Direct {
            payload_phase: self.payload_phase().after(payload_mode),
            relation_phase: self.relation_phase().after(transition),
        }
    }

    #[must_use]
    pub(crate) const fn offloaded_successor(
        transition: akita_types::RingRelationMode,
        payload_mode: akita_types::CommitmentPayloadMode,
        natural_len: usize,
    ) -> Option<Self> {
        if !transition.is_reduced_evaluation() && payload_mode.is_compressed() {
            Some(Self::SetupPrefixed { natural_len })
        } else {
            None
        }
    }
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
            d_a: memo_dimensions.d_a(),
            d_b: memo_dimensions.d_b(),
            d_d: memo_dimensions.d_d(),
            topology: self.topology,
        }
    }
}
