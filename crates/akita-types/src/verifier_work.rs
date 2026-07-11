//! Optional verifier operation counters used by profiling and regression tests.

/// Coarse verifier work counts that distinguish protocol growth from repeated
/// structured-evaluator work.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerifierWorkCounters {
    /// Number of structured relation groups evaluated.
    pub relation_groups: u64,
    /// Number of chunk bodies visited by structured relation evaluation.
    pub relation_chunks: u64,
    /// Number of direct setup-contribution evaluations.
    pub direct_setup_evals: u64,
    /// Setup ring positions visited across direct-evaluation segments.
    pub direct_setup_ring_visits: u64,
    /// Direct-evaluation setup segments scanned.
    pub direct_setup_segments: u64,
    /// Number of setup-product (Stage 3) verifier instances.
    pub stage3_instances: u64,
    /// Number of live setup ring entries scanned by Stage 3.
    pub setup_rings_scanned: u64,
    /// Equality-table elements materialized for setup indices.
    pub setup_eq_elements: u64,
    /// Equality-table elements materialized for ring coordinates.
    pub ring_eq_elements: u64,
    /// Setup-index weights evaluated through the succinct evaluator.
    pub setup_weight_succinct_evals: u64,
    /// Setup-index weights evaluated through the generic plan.
    pub setup_weight_plan_evals: u64,
    /// Generic-plan groups handled by a factored row/column formula.
    pub setup_weight_factored_groups: u64,
    /// Generic-plan groups handled by packed setup segments.
    pub setup_weight_segment_groups: u64,
    /// Packed setup segments evaluated by the generic plan.
    pub setup_weight_segments: u64,
}

/// One coarse verifier-work event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum VerifierWorkEvent {
    RelationGroup,
    RelationChunks(u64),
    DirectSetupEval,
    DirectSetupRingVisits(u64),
    DirectSetupSegments(u64),
    Stage3Instance,
    SetupRingsScanned(u64),
    SetupEqElements(u64),
    RingEqElements(u64),
    SetupWeightSuccinctEval,
    SetupWeightPlanEval,
    SetupWeightFactoredGroup,
    SetupWeightSegmentGroup,
    SetupWeightSegments(u64),
}

#[cfg(feature = "verifier-work-counters")]
mod enabled {
    use super::{VerifierWorkCounters, VerifierWorkEvent};
    use std::sync::atomic::{AtomicU64, Ordering};

    static STAGE3_INSTANCES: AtomicU64 = AtomicU64::new(0);
    static RELATION_GROUPS: AtomicU64 = AtomicU64::new(0);
    static RELATION_CHUNKS: AtomicU64 = AtomicU64::new(0);
    static DIRECT_SETUP_EVALS: AtomicU64 = AtomicU64::new(0);
    static DIRECT_SETUP_RING_VISITS: AtomicU64 = AtomicU64::new(0);
    static DIRECT_SETUP_SEGMENTS: AtomicU64 = AtomicU64::new(0);
    static SETUP_RINGS_SCANNED: AtomicU64 = AtomicU64::new(0);
    static SETUP_EQ_ELEMENTS: AtomicU64 = AtomicU64::new(0);
    static RING_EQ_ELEMENTS: AtomicU64 = AtomicU64::new(0);
    static SETUP_WEIGHT_SUCCINCT_EVALS: AtomicU64 = AtomicU64::new(0);
    static SETUP_WEIGHT_PLAN_EVALS: AtomicU64 = AtomicU64::new(0);
    static SETUP_WEIGHT_FACTORED_GROUPS: AtomicU64 = AtomicU64::new(0);
    static SETUP_WEIGHT_SEGMENT_GROUPS: AtomicU64 = AtomicU64::new(0);
    static SETUP_WEIGHT_SEGMENTS: AtomicU64 = AtomicU64::new(0);

    const ORDERING: Ordering = Ordering::Relaxed;

    pub(super) fn record(event: VerifierWorkEvent) {
        let (counter, amount) = match event {
            VerifierWorkEvent::RelationGroup => (&RELATION_GROUPS, 1),
            VerifierWorkEvent::RelationChunks(amount) => (&RELATION_CHUNKS, amount),
            VerifierWorkEvent::DirectSetupEval => (&DIRECT_SETUP_EVALS, 1),
            VerifierWorkEvent::DirectSetupRingVisits(amount) => (&DIRECT_SETUP_RING_VISITS, amount),
            VerifierWorkEvent::DirectSetupSegments(amount) => (&DIRECT_SETUP_SEGMENTS, amount),
            VerifierWorkEvent::Stage3Instance => (&STAGE3_INSTANCES, 1),
            VerifierWorkEvent::SetupRingsScanned(amount) => (&SETUP_RINGS_SCANNED, amount),
            VerifierWorkEvent::SetupEqElements(amount) => (&SETUP_EQ_ELEMENTS, amount),
            VerifierWorkEvent::RingEqElements(amount) => (&RING_EQ_ELEMENTS, amount),
            VerifierWorkEvent::SetupWeightSuccinctEval => (&SETUP_WEIGHT_SUCCINCT_EVALS, 1),
            VerifierWorkEvent::SetupWeightPlanEval => (&SETUP_WEIGHT_PLAN_EVALS, 1),
            VerifierWorkEvent::SetupWeightFactoredGroup => (&SETUP_WEIGHT_FACTORED_GROUPS, 1),
            VerifierWorkEvent::SetupWeightSegmentGroup => (&SETUP_WEIGHT_SEGMENT_GROUPS, 1),
            VerifierWorkEvent::SetupWeightSegments(amount) => (&SETUP_WEIGHT_SEGMENTS, amount),
        };
        counter.fetch_add(amount, ORDERING);
    }

    pub(super) fn reset() {
        for counter in [
            &RELATION_GROUPS,
            &RELATION_CHUNKS,
            &DIRECT_SETUP_EVALS,
            &DIRECT_SETUP_RING_VISITS,
            &DIRECT_SETUP_SEGMENTS,
            &STAGE3_INSTANCES,
            &SETUP_RINGS_SCANNED,
            &SETUP_EQ_ELEMENTS,
            &RING_EQ_ELEMENTS,
            &SETUP_WEIGHT_SUCCINCT_EVALS,
            &SETUP_WEIGHT_PLAN_EVALS,
            &SETUP_WEIGHT_FACTORED_GROUPS,
            &SETUP_WEIGHT_SEGMENT_GROUPS,
            &SETUP_WEIGHT_SEGMENTS,
        ] {
            counter.store(0, ORDERING);
        }
    }

    pub(super) fn snapshot() -> VerifierWorkCounters {
        VerifierWorkCounters {
            relation_groups: RELATION_GROUPS.load(ORDERING),
            relation_chunks: RELATION_CHUNKS.load(ORDERING),
            direct_setup_evals: DIRECT_SETUP_EVALS.load(ORDERING),
            direct_setup_ring_visits: DIRECT_SETUP_RING_VISITS.load(ORDERING),
            direct_setup_segments: DIRECT_SETUP_SEGMENTS.load(ORDERING),
            stage3_instances: STAGE3_INSTANCES.load(ORDERING),
            setup_rings_scanned: SETUP_RINGS_SCANNED.load(ORDERING),
            setup_eq_elements: SETUP_EQ_ELEMENTS.load(ORDERING),
            ring_eq_elements: RING_EQ_ELEMENTS.load(ORDERING),
            setup_weight_succinct_evals: SETUP_WEIGHT_SUCCINCT_EVALS.load(ORDERING),
            setup_weight_plan_evals: SETUP_WEIGHT_PLAN_EVALS.load(ORDERING),
            setup_weight_factored_groups: SETUP_WEIGHT_FACTORED_GROUPS.load(ORDERING),
            setup_weight_segment_groups: SETUP_WEIGHT_SEGMENT_GROUPS.load(ORDERING),
            setup_weight_segments: SETUP_WEIGHT_SEGMENTS.load(ORDERING),
        }
    }
}

/// Record one verifier-work event. This is a no-op unless the
/// `verifier-work-counters` feature is enabled.
#[doc(hidden)]
#[inline]
pub fn record_verifier_work(event: VerifierWorkEvent) {
    #[cfg(feature = "verifier-work-counters")]
    enabled::record(event);
    #[cfg(not(feature = "verifier-work-counters"))]
    let _ = event;
}

/// Reset all verifier-work counters.
pub fn reset_verifier_work_counters() {
    #[cfg(feature = "verifier-work-counters")]
    enabled::reset();
}

/// Snapshot all verifier-work counters.
#[must_use]
pub fn verifier_work_counters() -> VerifierWorkCounters {
    #[cfg(feature = "verifier-work-counters")]
    return enabled::snapshot();
    #[cfg(not(feature = "verifier-work-counters"))]
    VerifierWorkCounters::default()
}

#[cfg(all(test, feature = "verifier-work-counters"))]
mod tests {
    use super::*;

    #[test]
    fn counters_reset_and_accumulate() {
        reset_verifier_work_counters();
        record_verifier_work(VerifierWorkEvent::Stage3Instance);
        record_verifier_work(VerifierWorkEvent::SetupRingsScanned(7));
        record_verifier_work(VerifierWorkEvent::SetupWeightSegments(3));
        assert_eq!(
            verifier_work_counters(),
            VerifierWorkCounters {
                stage3_instances: 1,
                setup_rings_scanned: 7,
                setup_weight_segments: 3,
                ..VerifierWorkCounters::default()
            }
        );
        reset_verifier_work_counters();
        assert_eq!(verifier_work_counters(), VerifierWorkCounters::default());
    }
}
