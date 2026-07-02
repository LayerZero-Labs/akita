//! Prover-side fold grind probe metrics (profile / diagnostics only).

use akita_types::sis::FoldWitnessLinfCapPolicy;

/// One fold-level grind outcome recorded during proving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldGrindObservation {
    /// Zero-based fold level index within this prove call (root fold first).
    pub level_index: u32,
    /// Wire nonce committed into the proof.
    pub grind_nonce: u32,
    /// Number of off-sponge probes before acceptance (includes the winner).
    pub grind_probe_count: u32,
    /// Realized centered `‖z‖_inf` on the accepted folded witness.
    pub observed_linf: u32,
    /// Worst-case structural envelope `β_inf`.
    pub beta_inf: u128,
    /// Sub-Gaussian tail cap `t*` when tail-bound-with-grind is active.
    pub t_star: Option<u128>,
    /// Honest grind / digit-sizing cap (`min(β_inf, t*)` or `β_inf` alone).
    pub honest_cap: u128,
    /// Planner `δ_fold` digit depth at this level.
    pub delta_fold: usize,
    /// Verifier digit envelope `fold_witness_verifier_linf_bound(lb, δ_fold)`.
    pub verifier_linf_bound: u128,
    /// Active tail-bound policy for this level.
    pub policy: FoldWitnessLinfCapPolicy,
    /// Gadget base `log_basis` at this level.
    pub log_basis: u32,
    /// Fold arity metadata for cross-run alignment.
    pub r_vars: u32,
    pub num_claims: u32,
}

struct ObserverState {
    active: bool,
    records: Vec<FoldGrindObservation>,
}

thread_local! {
    static FOLD_GRIND_OBSERVER: RefCell<ObserverState> = const {
        RefCell::new(ObserverState {
            active: false,
            records: Vec::new(),
        })
    };
}

use std::cell::RefCell;

/// RAII guard that activates fold-grind observation on the current thread.
pub struct FoldGrindObserverGuard;

impl FoldGrindObserverGuard {
    /// Begin recording fold grind probe counts for subsequent prove calls.
    pub fn install() -> Self {
        FOLD_GRIND_OBSERVER.with(|cell| {
            let mut state = cell.borrow_mut();
            state.active = true;
            state.records.clear();
        });
        Self
    }

    /// Drain recorded observations and deactivate the observer.
    pub fn take() -> Vec<FoldGrindObservation> {
        FOLD_GRIND_OBSERVER.with(|cell| {
            let mut state = cell.borrow_mut();
            state.active = false;
            std::mem::take(&mut state.records)
        })
    }
}

impl Drop for FoldGrindObserverGuard {
    fn drop(&mut self) {
        FOLD_GRIND_OBSERVER.with(|cell| {
            cell.borrow_mut().active = false;
        });
    }
}

pub(crate) fn next_fold_grind_level_index() -> u32 {
    FOLD_GRIND_OBSERVER.with(|cell| {
        u32::try_from(cell.borrow().records.len()).unwrap_or(u32::MAX)
    })
}

pub(crate) fn record_fold_grind_acceptance(observation: FoldGrindObservation) {
    debug_assert!(
        observation.grind_probe_count > 0,
        "grind probe count must be positive"
    );
    FOLD_GRIND_OBSERVER.with(|cell| {
        let mut state = cell.borrow_mut();
        if state.active {
            state.records.push(observation);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_types::sis::FoldWitnessLinfCapPolicy;

    #[test]
    fn install_take_roundtrip_records_probe_metrics() {
        let _guard = FoldGrindObserverGuard::install();
        record_fold_grind_acceptance(FoldGrindObservation {
            level_index: 0,
            grind_nonce: 7,
            grind_probe_count: 3,
            observed_linf: 12,
            beta_inf: 32,
            t_star: Some(173),
            honest_cap: 32,
            delta_fold: 2,
            verifier_linf_bound: 56,
            policy: FoldWitnessLinfCapPolicy::TailBoundWithGrind,
            log_basis: 4,
            r_vars: 4,
            num_claims: 1,
        });
        record_fold_grind_acceptance(FoldGrindObservation {
            level_index: 1,
            grind_nonce: 0,
            grind_probe_count: 1,
            observed_linf: 4,
            beta_inf: 64,
            t_star: None,
            honest_cap: 64,
            delta_fold: 3,
            verifier_linf_bound: 120,
            policy: FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
            log_basis: 4,
            r_vars: 3,
            num_claims: 1,
        });
        let records = FoldGrindObserverGuard::take();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].grind_nonce, 7);
        assert_eq!(records[1].observed_linf, 4);
    }

    #[test]
    fn inactive_observer_drops_records() {
        record_fold_grind_acceptance(FoldGrindObservation {
            level_index: 0,
            grind_nonce: 1,
            grind_probe_count: 1,
            observed_linf: 1,
            beta_inf: 1,
            t_star: None,
            honest_cap: 1,
            delta_fold: 1,
            verifier_linf_bound: 1,
            policy: FoldWitnessLinfCapPolicy::WorstCaseBetaOnly,
            log_basis: 4,
            r_vars: 0,
            num_claims: 1,
        });
        assert!(FoldGrindObserverGuard::take().is_empty());
    }
}
