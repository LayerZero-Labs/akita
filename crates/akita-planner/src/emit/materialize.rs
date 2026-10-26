use super::*;
use std::collections::HashMap;
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
    pub fn key(&self) -> ScheduleLookupKey {
        match &self.request {
            PlanningRequest::Scalar(group) => ScheduleLookupKey::single(*group),
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
        PlanningRequest::Scalar(layout) => ScheduleLookupKey::single(*layout),
        PlanningRequest::Grouped(request) => request.key(),
    };
    let digest = akita_params::digest_descriptor_bytes(&key.canonical_descriptor_bytes());
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
    for spec in specs {
        let mut requests_by_key = HashMap::with_capacity(spec.grouped_requests.len());
        for request in &spec.grouped_requests {
            if requests_by_key
                .insert(request.key(), request)
                .is_some_and(|prior| prior != request)
            {
                return Err(format!(
                    "{}: duplicate grouped lookup geometry has conflicting producer contracts",
                    spec.family_name,
                ));
            }
        }
    }
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
    let (key, contracts) = match request {
        PlanningRequest::Scalar(key) => (ScheduleLookupKey::single(*key), Vec::new()),
        PlanningRequest::Grouped(request) => (request.key(), request.source_contracts()),
    };
    let preplanned = spec
        .preplanned
        .iter()
        .find(|(planned_key, _)| *planned_key == key);
    let reused_preplan = preplanned.is_some();
    let result = match preplanned {
        Some((_, schedule)) => Ok(schedule.clone()),
        None => crate::find_schedule(
            &key,
            spec.source_contract,
            &contracts,
            &spec.policy,
            spec.ring_challenge_config,
        )
        .map(|planned| planned.schedule),
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

#[cfg(all(test, feature = "catalog-gen"))]
mod tests {
    use super::*;
    use akita_config::{
        proof_optimized::fp128::{Dense, DenseBounded},
        CommitmentConfig,
    };

    #[test]
    fn conflicting_contracts_for_one_lookup_geometry_reject_before_planning() {
        let group = PolynomialGroupLayout::singleton(14);
        let descriptor = akita_config::test_support::workspace_schedule_catalog::<Dense>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap()
        .resolve_key(&ScheduleLookupKey::single(group))
        .unwrap()
        .profiles()
        .final_group;
        let family = crate::generated_families::ALL_GENERATED_FAMILIES
            .iter()
            .find(|family| family.family_name() == Dense::schedule_family_name())
            .unwrap();
        let mut spec = crate::generated_families::empty_emit_spec(
            family,
            (family.policy)().witness_chunk,
            PathBuf::new(),
        )
        .unwrap();
        spec.grouped_requests = [
            PrecommittedProducer::from_config::<Dense>(descriptor).unwrap(),
            PrecommittedProducer::from_config::<DenseBounded>(descriptor).unwrap(),
        ]
        .into_iter()
        .map(|producer| GroupedGenerationRequest::new(group, vec![producer]))
        .collect();
        let error = materialized_entries_for_specs(&[spec], MaterializationDiagnostics::default())
            .err()
            .expect("conflicting producer contracts must reject");
        assert!(error.contains("conflicting producer contracts"));
    }
    #[test]
    fn emitter_plans_with_the_selected_runtime_geometry() {
        use akita_config::proof_optimized::fp128::OneHot;
        use akita_params::ChunkedWitnessCfg;
        let family = crate::generated_families::ALL_GENERATED_FAMILIES
            .iter()
            .find(|family| family.family_name() == OneHot::schedule_family_name())
            .unwrap();
        let chunking = ChunkedWitnessCfg {
            num_chunks: 2,
            num_activated_levels: 4,
        };
        let mut spec =
            crate::generated_families::empty_emit_spec(family, chunking, PathBuf::new()).unwrap();
        spec.keys = vec![PolynomialGroupLayout::singleton(17)];
        let outputs = render_schedule_artifact_outputs_with_validation(
            &[spec],
            MaterializationDiagnostics::default(),
            |_, rows| {
                assert_eq!(rows.len(), 1);
                Ok(())
            },
        )
        .unwrap();
        let catalog = akita_config::TrustedScheduleCatalog::<OneHot>::from_artifact_bytes(
            outputs[0].body.as_bytes(),
            chunking,
        )
        .unwrap();
        assert_eq!(catalog.policy().witness_chunk, chunking);
        assert!(
            akita_config::TrustedScheduleCatalog::<OneHot>::from_artifact_bytes(
                outputs[0].body.as_bytes(),
                ChunkedWitnessCfg::default_non_chunked(),
            )
            .is_err()
        );
    }

    #[test]
    fn emitter_rejects_preplanned_rows_after_runtime_geometry_changes() {
        use akita_config::proof_optimized::fp128::OneHot;
        use akita_params::{ChunkedWitnessCfg, MultiChunkProfileId};
        let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>(
            ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap();
        // The row is replanned under a single-group key, so it must have no
        // precommitted groups.
        let row = catalog
            .rows()
            .find(|row| row.profiles().precommitteds.is_empty())
            .unwrap();
        let group = row.profiles().final_group.group;
        let family = crate::generated_families::ALL_GENERATED_FAMILIES
            .iter()
            .find(|family| family.family_name() == OneHot::schedule_family_name())
            .unwrap();
        let mut spec = crate::generated_families::empty_emit_spec(
            family,
            (family.policy)().witness_chunk,
            PathBuf::new(),
        )
        .unwrap();
        spec.keys = vec![group];
        spec.preplanned = vec![(ScheduleLookupKey::single(group), row.schedule().clone())];
        render_schedule_artifact_outputs_with_validation(
            std::slice::from_ref(&spec),
            MaterializationDiagnostics::default(),
            |_, _| Ok(()),
        )
        .expect("the preplanned row matches its original policy");
        spec.policy.witness_chunk = MultiChunkProfileId::W2R3.cfg();
        let error = render_schedule_artifact_outputs_with_validation(
            &[spec],
            MaterializationDiagnostics::default(),
            |_, _| Ok(()),
        )
        .expect_err("changed geometry must invalidate the preplanned row");
        assert!(
            error.contains("root final group") && error.contains("witness chunks"),
            "{error}"
        );
    }
}
