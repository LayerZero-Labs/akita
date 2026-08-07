use std::{
    collections::{hash_map::Entry, BTreeMap, HashMap, VecDeque},
    sync::Arc,
};

use akita_field::AkitaError;
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

use crate::{schedule_params::SetupPrefixSearchCache, PlannerPolicy};

use super::ScheduleCandidate;

#[derive(Clone)]
pub(crate) struct SuffixResult {
    pub(crate) best_by_first_direct_setup_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate>,
    pub(crate) best_by_payload_per_lb: BTreeMap<FirstFoldKey, ScheduleCandidate>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct FirstFoldKey {
    pub(super) log_basis: u32,
    pub(super) parent_cost: ParentVisibleCost,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct ParentVisibleCost {
    pub(super) outer_payload_bytes: usize,
    pub(super) stage3_payload_bytes: usize,
}

pub(crate) struct SuffixSearchCache {
    pub(super) entries: HashMap<SuffixState, Arc<SuffixResult>>,
    insertion_order: VecDeque<SuffixState>,
    pub(super) setup_prefixes: SetupPrefixSearchCache,
}

// Memoization changes recomputation cost only. Keep exact search states, but
// evict old cached results so a wide geometry sweep cannot retain the entire
// recursive search graph at once.
pub(super) const MAX_SUFFIX_SEARCH_CACHE_ENTRIES: usize = 262_144;

impl SuffixSearchCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            insertion_order: VecDeque::new(),
            setup_prefixes: SetupPrefixSearchCache::default(),
        }
    }

    pub(super) fn get(&self, key: &SuffixState) -> Option<&Arc<SuffixResult>> {
        self.entries.get(key)
    }

    pub(super) fn insert(&mut self, key: SuffixState, result: &Arc<SuffixResult>) {
        if let Entry::Occupied(mut existing) = self.entries.entry(key) {
            existing.insert(Arc::clone(result));
            return;
        }
        if self.entries.len() >= MAX_SUFFIX_SEARCH_CACHE_ENTRIES {
            if let Some(evicted) = self.insertion_order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.insertion_order.push_back(key);
        self.entries.insert(key, Arc::clone(result));
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SuffixCtx<'a> {
    pub(crate) policy: &'a PlannerPolicy,
    pub(crate) default_ring_challenge_cfg: &'a akita_challenges::SparseChallengeConfig,
    pub(crate) ring_challenge_config:
        &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    pub(crate) fold_challenge_shape_at_level:
        &'a dyn Fn(akita_types::AkitaScheduleInputs) -> akita_challenges::TensorChallengeShape,
    pub(crate) num_vars: usize,
    pub(crate) key: PolynomialGroupLayout,
    pub(crate) setup_field_budget: Option<usize>,
    pub(crate) root_lookup_key: Option<&'a AkitaScheduleLookupKey>,
    pub(crate) root_honest_fold_policy: Option<akita_types::sis::HonestFoldPolicySpec>,
    pub(crate) precommitted_honest_fold_policies: &'a [akita_types::sis::HonestFoldPolicySpec],
    pub(crate) level_zero_is_root: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SuffixState {
    pub(crate) level: usize,
    pub(crate) current_witness_len: usize,
    pub(crate) current_lb: u32,
    pub(crate) incoming_setup_prefix: Option<usize>,
    pub(crate) payload_phase: akita_types::CommitmentPayloadPhase,
}

pub(super) fn empty_result() -> Arc<SuffixResult> {
    Arc::new(SuffixResult {
        best_by_first_direct_setup_per_lb: BTreeMap::new(),
        best_by_payload_per_lb: BTreeMap::new(),
    })
}
