//! Reusable external schedule-artifact emitter and audit helpers.
//!
//! The `gen_schedule_artifacts` binary adapts preset metadata into [`EmitSpec`]
//! values and calls this module. Jolt can invoke the same API with an explicit
//! [`PlannerPolicy`] and hook function pointers.

use std::path::PathBuf;

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::sis::{CommittedSourceContract, HonestFoldPolicySpec};
use akita_types::{
    AkitaScheduleLookupKey, FoldSchedule, GroupCommitPhaseParams, PolynomialGroupLayout,
};

use crate::PlannerPolicy;

mod materialize;
mod publish;
mod render;

pub(super) use materialize::materialized_entries_for_specs;
pub use materialize::MaterializedEntry;
pub use publish::publish_artifact_outputs;
pub use render::{render_schedule_artifact_outputs_with_validation, ArtifactOutput};

/// Optional observability for the offline row-materialization queue.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterializationDiagnostics {
    pub row_progress: bool,
}

/// One frozen precommit descriptor with its canonical producer facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecommittedProducer {
    descriptor: GroupCommitPhaseParams,
    contract: CommittedSourceContract,
    fold_policy: HonestFoldPolicySpec,
}

impl PrecommittedProducer {
    /// Bind one frozen descriptor to the producer facts used by offline
    /// grouped-root sizing.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the descriptor is invalid for
    /// the producer field or the fold policy disagrees with its declared source
    /// class.
    pub fn try_new(
        descriptor: GroupCommitPhaseParams,
        contract: CommittedSourceContract,
        fold_policy: HonestFoldPolicySpec,
    ) -> Result<Self, AkitaError> {
        let field_bits = contract.decomposition().field_bits();
        descriptor.validate_frozen_precommit(field_bits)?;
        if fold_policy != contract.class().honest_fold_policy(field_bits) {
            return Err(AkitaError::InvalidSetup(
                "precommitted producer fold policy does not match its source contract".into(),
            ));
        }
        Ok(Self {
            descriptor,
            contract,
            fold_policy,
        })
    }

    /// Capture the producer contract and its offline sizing projection together.
    #[cfg(feature = "catalog-gen")]
    pub fn from_config<Cfg: akita_config::CommitmentConfig>(
        descriptor: GroupCommitPhaseParams,
    ) -> Result<Self, AkitaError> {
        Self::try_new(
            descriptor,
            Cfg::committed_source_contract()?,
            akita_config::honest_fold_policy_of::<Cfg>(),
        )
    }

    /// Frozen commit-phase descriptor used in the grouped lookup key.
    #[must_use]
    pub const fn descriptor(self) -> GroupCommitPhaseParams {
        self.descriptor
    }

    /// Producer declaration used to disambiguate otherwise identical grouped
    /// catalog layouts in revision evidence.
    #[must_use]
    pub const fn source_contract(self) -> CommittedSourceContract {
        self.contract
    }

    const fn fold_policy(self) -> HonestFoldPolicySpec {
        self.fold_policy
    }
}

/// One grouped generation request whose lookup key is derived from its producers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedGenerationRequest {
    final_group: PolynomialGroupLayout,
    precommitted_producers: Vec<PrecommittedProducer>,
}

impl GroupedGenerationRequest {
    #[must_use]
    pub fn new(
        final_group: PolynomialGroupLayout,
        precommitted_producers: Vec<PrecommittedProducer>,
    ) -> Self {
        Self {
            final_group,
            precommitted_producers,
        }
    }

    #[must_use]
    pub fn key(&self) -> AkitaScheduleLookupKey {
        AkitaScheduleLookupKey {
            final_group: self.final_group,
            precommitteds: self
                .precommitted_producers
                .iter()
                .copied()
                .map(PrecommittedProducer::descriptor)
                .collect(),
        }
    }

    fn precommitted_producers(&self) -> &[PrecommittedProducer] {
        &self.precommitted_producers
    }

    pub(crate) fn fold_policies(&self) -> Vec<HonestFoldPolicySpec> {
        self.precommitted_producers
            .iter()
            .copied()
            .map(PrecommittedProducer::fold_policy)
            .collect()
    }
}

/// One schedule family emitted as a canonical external artifact.
#[derive(Clone)]
pub struct EmitSpec {
    pub family_name: &'static str,
    pub policy: PlannerPolicy,
    pub source_contract: CommittedSourceContract,
    pub keys: Vec<PolynomialGroupLayout>,
    pub grouped_requests: Vec<GroupedGenerationRequest>,
    /// Exact successful scalar results already needed to construct grouped keys.
    pub preplanned_scalar: Vec<(PolynomialGroupLayout, FoldSchedule)>,
    pub output_dir: PathBuf,
    pub regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    pub regen_group_batch: fn(GroupedGenerationRequest) -> Result<FoldSchedule, AkitaError>,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
}

// Schedule search is memory bound. Keep the default below host-wide
// parallelism while allowing explicit tuning for large generation machines.
const DEFAULT_OFFLINE_PLANNING_WORKERS: usize = 3;

/// Bound memory-heavy offline planner searches for generation and drift checks.
pub fn offline_planning_worker_count(work_items: usize) -> usize {
    let configured = std::env::var("AKITA_SCHEDULE_GEN_JOBS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_OFFLINE_PLANNING_WORKERS);
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(configured)
        .min(work_items.max(1))
}

/// Map independent offline requests with a fixed worker count and input order.
pub fn bounded_parallel_filter_map<T, R>(
    items: &[T],
    workers: usize,
    map: impl Fn(&T) -> Result<Option<R>, String> + Sync,
) -> Result<Vec<R>, String>
where
    T: Sync,
    R: Send,
{
    if workers <= 1 {
        return items
            .iter()
            .filter_map(|item| map(item).transpose())
            .collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut mapped = std::thread::scope(|scope| -> Result<Vec<(usize, R)>, String> {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let map = &map;
                let next = &next;
                scope.spawn(move || {
                    let mut local = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(index) else {
                            break;
                        };
                        if let Some(value) = map(item)? {
                            local.push((index, value));
                        }
                    }
                    Ok::<_, String>(local)
                })
            })
            .collect();
        let mut output = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(Ok(local)) => output.extend(local),
                Ok(Err(error)) => return Err(error),
                Err(_) => return Err("schedule generation worker panicked".into()),
            }
        }
        Ok(output)
    })?;
    mapped.sort_by_key(|(index, _)| *index);
    Ok(mapped.into_iter().map(|(_, value)| value).collect())
}
