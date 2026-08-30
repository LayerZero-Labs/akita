use std::cell::{Cell, RefCell};
use std::fmt;
use std::rc::Rc;
use std::time::Duration;

use akita_types::{CommitmentRingDims, RingRelationMode};

use crate::SelectionPolicyId;

#[derive(Clone, Debug)]
pub(crate) struct SelectedScheduleDiagnostics {
    objective: SelectionPolicyId,
    proof_bytes: usize,
    setup_field_elements: usize,
    first_direct_setup_capacity: usize,
    dimensions: Vec<CommitmentRingDims>,
    relation_modes: Vec<RingRelationMode>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PlannerDiagnosticsSnapshot {
    pub(crate) suffix_dp_time: Duration,
    pub(crate) final_materialization_time: Duration,
    pub(crate) descriptor_time: Duration,
    pub(crate) descriptor_builds: usize,
    pub(crate) suffix_calls: usize,
    pub(crate) memo_hits: usize,
    pub(crate) memo_misses: usize,
    pub(crate) completed_states: usize,
    pub(crate) generated_candidates: usize,
    pub(crate) retained_candidates: usize,
    pub(crate) retained_frontier_candidates: usize,
    pub(crate) peak_state_frontier_candidates: usize,
    pub(crate) setup_prefix_cache_hits: usize,
    pub(crate) setup_prefix_cache_misses: usize,
    pub(crate) guided_direct_edge_prunes: usize,
    selected: Option<SelectedScheduleDiagnostics>,
}

impl fmt::Display for PlannerDiagnosticsSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let memo_total = self.memo_hits.saturating_add(self.memo_misses);
        let memo_hit_percent = if memo_total == 0 {
            0.0
        } else {
            100.0 * self.memo_hits as f64 / memo_total as f64
        };
        write!(
            formatter,
            "dp={:.2?} final_materialize={:.2?} suffix_calls={} completed_states={} memo_hits={}/{} ({memo_hit_percent:.1}%) candidates_generated={} candidates_after_local_prune={} guided_direct_edge_prunes={} frontier_candidates_retained={} peak_state_frontier={} descriptors_built={} descriptor_time={:.2?} setup_prefix_cache_hits={}/{}",
            self.suffix_dp_time,
            self.final_materialization_time,
            self.suffix_calls,
            self.completed_states,
            self.memo_hits,
            memo_total,
            self.generated_candidates,
            self.retained_candidates,
            self.guided_direct_edge_prunes,
            self.retained_frontier_candidates,
            self.peak_state_frontier_candidates,
            self.descriptor_builds,
            self.descriptor_time,
            self.setup_prefix_cache_hits,
            self.setup_prefix_cache_hits
                .saturating_add(self.setup_prefix_cache_misses),
        )?;
        if let Some(selected) = &self.selected {
            write!(
                formatter,
                " selected={{objective={:?} proof={} setup={} first_direct_capacity={} cutover={} dims=[",
                selected.objective,
                selected.proof_bytes,
                selected.setup_field_elements,
                selected.first_direct_setup_capacity,
                selected
                    .relation_modes
                    .iter()
                    .position(|mode| mode.is_reduced_evaluation())
                    .map_or_else(|| "none".to_string(), |level| level.to_string()),
            )?;
            for (index, dimensions) in selected.dimensions.iter().enumerate() {
                if index != 0 {
                    formatter.write_str(",")?;
                }
                write!(
                    formatter,
                    "{}/{}/{}",
                    dimensions.d_a(),
                    dimensions.d_b(),
                    dimensions.d_d(),
                )?;
            }
            formatter.write_str("] rel=[")?;
            for (index, mode) in selected.relation_modes.iter().enumerate() {
                if index != 0 {
                    formatter.write_str(",")?;
                }
                formatter.write_str(match mode {
                    RingRelationMode::QuotientLift => "quotient",
                    RingRelationMode::ReducedEvaluation => "reduced-evaluation",
                })?;
            }
            formatter.write_str("]}")?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct PlannerDiagnostics {
    suffix_dp_time: Cell<Duration>,
    final_materialization_time: Cell<Duration>,
    descriptor_time: Cell<Duration>,
    descriptor_builds: Cell<usize>,
    suffix_calls: Cell<usize>,
    memo_hits: Cell<usize>,
    memo_misses: Cell<usize>,
    completed_states: Cell<usize>,
    generated_candidates: Cell<usize>,
    retained_candidates: Cell<usize>,
    retained_frontier_candidates: Cell<usize>,
    peak_state_frontier_candidates: Cell<usize>,
    setup_prefix_cache_hits: Cell<usize>,
    setup_prefix_cache_misses: Cell<usize>,
    guided_direct_edge_prunes: Cell<usize>,
    selected: RefCell<Option<SelectedScheduleDiagnostics>>,
}

impl PlannerDiagnostics {
    pub(crate) fn add_suffix_dp_time(&self, elapsed: Duration) {
        self.suffix_dp_time
            .set(self.suffix_dp_time.get().saturating_add(elapsed));
    }

    pub(crate) fn add_final_materialization_time(&self, elapsed: Duration) {
        self.final_materialization_time.set(
            self.final_materialization_time
                .get()
                .saturating_add(elapsed),
        );
    }

    pub(crate) fn record_descriptor(&self, elapsed: Duration) {
        self.descriptor_builds
            .set(self.descriptor_builds.get().saturating_add(1));
        self.descriptor_time
            .set(self.descriptor_time.get().saturating_add(elapsed));
    }

    pub(crate) fn record_suffix_call(&self) {
        self.suffix_calls
            .set(self.suffix_calls.get().saturating_add(1));
    }

    pub(crate) fn record_memo_result(&self, hit: bool) {
        let counter = if hit {
            &self.memo_hits
        } else {
            &self.memo_misses
        };
        counter.set(counter.get().saturating_add(1));
    }

    pub(crate) fn record_candidates(&self, generated: usize, retained: usize) {
        self.generated_candidates
            .set(self.generated_candidates.get().saturating_add(generated));
        self.retained_candidates
            .set(self.retained_candidates.get().saturating_add(retained));
    }

    pub(crate) fn record_completed_state(&self, frontier_candidates: usize) {
        self.completed_states
            .set(self.completed_states.get().saturating_add(1));
        self.retained_frontier_candidates.set(
            self.retained_frontier_candidates
                .get()
                .saturating_add(frontier_candidates),
        );
        self.peak_state_frontier_candidates.set(
            self.peak_state_frontier_candidates
                .get()
                .max(frontier_candidates),
        );
    }

    pub(crate) fn record_setup_prefix_cache(&self, hits: usize, misses: usize) {
        self.setup_prefix_cache_hits.set(hits);
        self.setup_prefix_cache_misses.set(misses);
    }

    pub(crate) fn record_guided_direct_edge_prune(&self) {
        self.guided_direct_edge_prunes
            .set(self.guided_direct_edge_prunes.get().saturating_add(1));
    }

    pub(crate) fn record_selected(
        &self,
        objective: SelectionPolicyId,
        proof_bytes: usize,
        setup_field_elements: usize,
        first_direct_setup_capacity: usize,
        dimensions: Vec<CommitmentRingDims>,
        relation_modes: Vec<RingRelationMode>,
    ) {
        self.selected.replace(Some(SelectedScheduleDiagnostics {
            objective,
            proof_bytes,
            setup_field_elements,
            first_direct_setup_capacity,
            dimensions,
            relation_modes,
        }));
    }

    fn snapshot(&self) -> PlannerDiagnosticsSnapshot {
        PlannerDiagnosticsSnapshot {
            suffix_dp_time: self.suffix_dp_time.get(),
            final_materialization_time: self.final_materialization_time.get(),
            descriptor_time: self.descriptor_time.get(),
            descriptor_builds: self.descriptor_builds.get(),
            suffix_calls: self.suffix_calls.get(),
            memo_hits: self.memo_hits.get(),
            memo_misses: self.memo_misses.get(),
            completed_states: self.completed_states.get(),
            generated_candidates: self.generated_candidates.get(),
            retained_candidates: self.retained_candidates.get(),
            retained_frontier_candidates: self.retained_frontier_candidates.get(),
            peak_state_frontier_candidates: self.peak_state_frontier_candidates.get(),
            setup_prefix_cache_hits: self.setup_prefix_cache_hits.get(),
            setup_prefix_cache_misses: self.setup_prefix_cache_misses.get(),
            guided_direct_edge_prunes: self.guided_direct_edge_prunes.get(),
            selected: self.selected.borrow().clone(),
        }
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Rc<PlannerDiagnostics>>> = const { RefCell::new(None) };
}

struct ActiveGuard(Option<Rc<PlannerDiagnostics>>);

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.replace(self.0.take());
        });
    }
}

pub(crate) fn active() -> Option<Rc<PlannerDiagnostics>> {
    ACTIVE.with(|active| active.borrow().clone())
}

pub(crate) fn capture<T>(
    enabled: bool,
    operation: impl FnOnce() -> T,
) -> (T, Option<PlannerDiagnosticsSnapshot>) {
    if !enabled {
        return (operation(), None);
    }
    let diagnostics = Rc::new(PlannerDiagnostics::default());
    let previous = ACTIVE.with(|active| active.replace(Some(Rc::clone(&diagnostics))));
    let _guard = ActiveGuard(previous);
    let result = operation();
    let snapshot = diagnostics.snapshot();
    (result, Some(snapshot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_capture_does_not_install_diagnostics() {
        let (active_during_operation, snapshot) = capture(false, || active().is_some());
        assert!(!active_during_operation);
        assert!(snapshot.is_none());
    }

    #[test]
    fn enabled_capture_returns_recorded_counters() {
        let (_, snapshot) = capture(true, || {
            let diagnostics = active().expect("diagnostics must be active inside capture");
            diagnostics.record_suffix_call();
            diagnostics.record_memo_result(true);
            diagnostics.record_candidates(12, 4);
            diagnostics.record_completed_state(3);
            diagnostics.record_guided_direct_edge_prune();
        });
        let snapshot = snapshot.expect("enabled capture must return diagnostics");
        assert_eq!(snapshot.suffix_calls, 1);
        assert_eq!(snapshot.memo_hits, 1);
        assert_eq!(snapshot.generated_candidates, 12);
        assert_eq!(snapshot.retained_candidates, 4);
        assert_eq!(snapshot.retained_frontier_candidates, 3);
        assert_eq!(snapshot.peak_state_frontier_candidates, 3);
        assert_eq!(snapshot.guided_direct_edge_prunes, 1);
        assert!(active().is_none());
    }
}
