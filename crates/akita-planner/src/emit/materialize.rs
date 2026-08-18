use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

enum PlanningRequest {
    Scalar(PolynomialGroupLayout),
    Grouped {
        key: AkitaScheduleLookupKey,
        honest_fold_policies: Vec<HonestFoldPolicySpec>,
    },
}

struct IndexedPlanningRequest {
    spec_index: usize,
    request_index: usize,
    request: PlanningRequest,
}

pub type MaterializedEntry = (AkitaScheduleLookupKey, FoldSchedule);

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
        PlanningRequest::Grouped { key, .. } => key.clone(),
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
        .map(|spec| spec.keys.len() + spec.group_batch_keys.len())
        .sum();
    let mut requests = Vec::with_capacity(request_count);
    for (spec_index, spec) in specs.iter().enumerate() {
        requests.extend(spec.keys.iter().copied().map(|key| IndexedPlanningRequest {
            spec_index,
            request_index: 0,
            request: PlanningRequest::Scalar(key),
        }));
        requests.extend(spec.group_batch_keys.iter().cloned().map(
            |(key, honest_fold_policies)| IndexedPlanningRequest {
                spec_index,
                request_index: 0,
                request: PlanningRequest::Grouped {
                    key,
                    honest_fold_policies,
                },
            },
        ));
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
                spec.module_name,
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
                Ok(MaterializedRequestOutcome::Planned((_, schedule))) => {
                    counters.planned.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "planned schedule row {}/{}: {} {label} levels={} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.module_name,
                        schedule.num_fold_levels(),
                        started.elapsed(),
                    );
                }
                Ok(MaterializedRequestOutcome::ReusedPreplan((_, schedule))) => {
                    counters.reused_preplans.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "reused schedule row {}/{}: {} {label} levels={} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.module_name,
                        schedule.num_fold_levels(),
                        started.elapsed(),
                    );
                }
                Ok(MaterializedRequestOutcome::Unsupported) => {
                    counters.unsupported.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "unsupported schedule row {}/{}: {} {label} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.module_name,
                        started.elapsed(),
                    );
                }
                Err(_) => {
                    eprintln!(
                        "failed schedule row {}/{}: {} {label} in {:.2?}",
                        indexed.request_index + 1,
                        requests.len(),
                        spec.module_name,
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
                    spec.module_name,
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
                spec.module_name,
                spec.keys.len() + spec.group_batch_keys.len(),
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
        entries.sort_by(|(left, _), (right, _)| {
            akita_schedules::runtime_schedule_key_cmp(left, right)
        });
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
        PlanningRequest::Grouped {
            key,
            honest_fold_policies,
        } => (
            key.clone(),
            (spec.regen_group_batch)(key.clone(), honest_fold_policies.clone()),
            false,
        ),
    };
    match result {
        Ok(schedule) if reused_preplan => {
            Ok(MaterializedRequestOutcome::ReusedPreplan((key, schedule)))
        }
        Ok(schedule) => Ok(MaterializedRequestOutcome::Planned((key, schedule))),
        Err(akita_field::AkitaError::UnsupportedSchedule(_)) => {
            Ok(MaterializedRequestOutcome::Unsupported)
        }
        Err(error) => {
            let kind = if key.precommitteds.is_empty() {
                "regen"
            } else {
                "regen multi-group"
            };
            Err(format!("{}: {kind} {key:?}: {error}", spec.module_name))
        }
    }
}
