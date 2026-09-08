use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Clone)]
enum PlanningRequest {
    Scalar(PolynomialGroupLayout),
    Grouped(GroupedGenerationRequest),
}

struct IndexedPlanningRequest {
    spec_index: usize,
    request_index: usize,
    request: PlanningRequest,
}

/// One planned schedule with the generation request that produced it.
pub struct MaterializedEntry {
    request: PlanningRequest,
    schedule: FoldSchedule,
}

impl MaterializedEntry {
    #[must_use]
    pub fn key(&self) -> AkitaScheduleLookupKey {
        match &self.request {
            PlanningRequest::Scalar(group) => AkitaScheduleLookupKey::single(*group),
            PlanningRequest::Grouped(request) => request.key(),
        }
    }

    #[must_use]
    pub const fn schedule(&self) -> &FoldSchedule {
        &self.schedule
    }

    /// Producer declarations for the frozen groups in opening order.
    #[must_use]
    pub fn precommitted_producers(&self) -> &[PrecommittedProducer] {
        match &self.request {
            PlanningRequest::Scalar(_) => &[],
            PlanningRequest::Grouped(request) => request.precommitted_producers(),
        }
    }
}

enum MaterializedRequestOutcome {
    Planned(MaterializedEntry),
    ReusedPreplan(MaterializedEntry),
    Unsupported,
}

#[derive(Default)]
struct MaterializationCounters {
    reused_preplans: AtomicUsize,
    planned: AtomicUsize,
    unsupported: AtomicUsize,
}

fn compact_request_label(request: &PlanningRequest) -> String {
    let key = match request {
        PlanningRequest::Scalar(layout) => AkitaScheduleLookupKey::single(*layout),
        PlanningRequest::Grouped(request) => request.key(),
    };
    let digest = akita_types::instance_descriptor::digest_descriptor_bytes(
        &key.canonical_descriptor_bytes(),
    );
    let id = digest
        .iter()
        .take(6)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(
        "nv={} polys={} precommits={} key={id}",
        key.final_group.num_vars(),
        key.final_group.num_polynomials(),
        key.precommitteds.len(),
    )
}

pub(crate) fn materialized_entries_for_specs(
    specs: &[EmitSpec],
    diagnostics: MaterializationDiagnostics,
) -> Result<Vec<Vec<MaterializedEntry>>, String> {
    let request_count = specs
        .iter()
        .map(|spec| spec.keys.len() + spec.grouped_requests.len())
        .sum();
    let mut requests = Vec::with_capacity(request_count);
    for (spec_index, spec) in specs.iter().enumerate() {
        requests.extend(spec.keys.iter().copied().map(|key| IndexedPlanningRequest {
            spec_index,
            request_index: 0,
            request: PlanningRequest::Scalar(key),
        }));
        requests.extend(spec.grouped_requests.iter().cloned().map(|request| {
            IndexedPlanningRequest {
                spec_index,
                request_index: 0,
                request: PlanningRequest::Grouped(request),
            }
        }));
    }
    for (request_index, request) in requests.iter_mut().enumerate() {
        request.request_index = request_index;
    }

    let workers = offline_planning_worker_count(requests.len());
    let counters = diagnostics.row_progress.then(|| {
        std::iter::repeat_with(MaterializationCounters::default)
            .take(specs.len())
            .collect::<Vec<_>>()
    });
    let materialized = bounded_parallel_filter_map(&requests, workers, |indexed| {
        let spec = &specs[indexed.spec_index];
        let progress = diagnostics.row_progress.then(|| {
            let label = compact_request_label(&indexed.request);
            eprintln!(
                "planning schedule row {}/{}: {} {label}",
                indexed.request_index + 1,
                requests.len(),
                spec.family_name,
            );
            (Instant::now(), label)
        });
        let (outcome, planner_diagnostics) =
            crate::diagnostics::capture(diagnostics.row_progress, || {
                materialized_entry(spec, &indexed.request)
            });
        if let Some((started, label)) = progress {
            let counters = &counters.as_ref().expect("progress counters")[indexed.spec_index];
            match &outcome {
                Ok(MaterializedRequestOutcome::Planned(entry)) => {
                    counters.planned.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "planned schedule row {}/{}: {} {label} levels={} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.family_name,
                        entry.schedule().num_fold_levels(),
                        started.elapsed(),
                    );
                }
                Ok(MaterializedRequestOutcome::ReusedPreplan(entry)) => {
                    counters.reused_preplans.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "reused schedule row {}/{}: {} {label} levels={} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.family_name,
                        entry.schedule().num_fold_levels(),
                        started.elapsed(),
                    );
                }
                Ok(MaterializedRequestOutcome::Unsupported) => {
                    counters.unsupported.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "unsupported schedule row {}/{}: {} {label} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.family_name,
                        started.elapsed(),
                    );
                }
                Err(_) => {
                    eprintln!(
                        "failed schedule row {}/{}: {} {label} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.family_name,
                        started.elapsed(),
                    );
                }
            }
            if let Some(planner_diagnostics) = planner_diagnostics
                .as_ref()
                .filter(|diagnostics| diagnostics.suffix_calls != 0)
            {
                eprintln!(
                    "planner diagnostics {} {label}: {planner_diagnostics}",
                    spec.family_name,
                );
            }
        }
        outcome.map(|outcome| match outcome {
            MaterializedRequestOutcome::Planned(entry)
            | MaterializedRequestOutcome::ReusedPreplan(entry) => Some((indexed.spec_index, entry)),
            MaterializedRequestOutcome::Unsupported => None,
        })
    })?;
    if let Some(counters) = &counters {
        for (spec, counters) in specs.iter().zip(counters) {
            eprintln!(
                "schedule row summary {}: requested={} reused={} planned={} unsupported={}",
                spec.family_name,
                spec.keys.len() + spec.grouped_requests.len(),
                counters.reused_preplans.load(Ordering::Relaxed),
                counters.planned.load(Ordering::Relaxed),
                counters.unsupported.load(Ordering::Relaxed),
            );
        }
    }
    let mut entries_by_spec = std::iter::repeat_with(Vec::new)
        .take(specs.len())
        .collect::<Vec<_>>();
    for (spec_index, entry) in materialized {
        entries_by_spec[spec_index].push(entry);
    }
    for entries in &mut entries_by_spec {
        entries.sort_by_cached_key(|entry| entry.key().canonical_order_key());
    }
    Ok(entries_by_spec)
}

fn materialized_entry(
    spec: &EmitSpec,
    request: &PlanningRequest,
) -> Result<MaterializedRequestOutcome, String> {
    let (key, result, reused_preplan) = match request {
        PlanningRequest::Scalar(key) => {
            let lookup = AkitaScheduleLookupKey::single(*key);
            let preplanned = spec
                .preplanned_scalar
                .iter()
                .find(|(preplanned_key, _)| preplanned_key == key);
            let result =
                preplanned.map_or_else(|| (spec.regen)(*key), |(_, schedule)| Ok(schedule.clone()));
            (lookup, result, preplanned.is_some())
        }
        PlanningRequest::Grouped(request) => {
            let key = request.key();
            (key, (spec.regen_group_batch)(request.clone()), false)
        }
    };
    let entry = |schedule| MaterializedEntry {
        request: request.clone(),
        schedule,
    };
    match result {
        Ok(schedule) if reused_preplan => {
            Ok(MaterializedRequestOutcome::ReusedPreplan(entry(schedule)))
        }
        Ok(schedule) => Ok(MaterializedRequestOutcome::Planned(entry(schedule))),
        Err(akita_error::AkitaError::UnsupportedSchedule(_)) => {
            Ok(MaterializedRequestOutcome::Unsupported)
        }
        Err(error) => {
            let kind = if key.precommitteds.is_empty() {
                "regen"
            } else {
                "regen multi-group"
            };
            Err(format!("{}: {kind} {key:?}: {error}", spec.family_name))
        }
    }
}
