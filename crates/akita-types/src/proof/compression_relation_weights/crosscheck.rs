//! Diagnostic cross-check of the factored compression relation weights.
//!
//! Compiled only with the `compression-weights-crosscheck` feature, which no
//! default feature enables. With it, [`CompressionRelationWeights::evaluate_at_point`]
//! returns the value of the default build and also:
//!
//! - evaluates the same table event by event (every group expanded into its
//!   events) at the requested point and at [`EXTRA_POINTS`] seeded random
//!   points, and compares each value with the grouped evaluation;
//! - counts the table's groups and events per evaluation tier.
//!
//! Each recomposition call and each compression row of the builders records
//! whether its family was grouped or pushed event by event (with the reason),
//! and compares the grouped events with those of the canonical per-event loop
//! on the same arguments.
//!
//! A difference is counted and then returned as [`AkitaError::Internal`], so
//! the verifier rejects instead of accepting on an unchecked value. Counters
//! accumulate process-wide ([`report`]). Each evaluation also emits one
//! `tracing` event at `INFO` with target [`TRACING_TARGET`]; its message is a
//! JSON object for that evaluation, including the builder outcomes recorded
//! while the evaluated table was built.

use super::*;
use rand_core::RngCore;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

/// Seeded random points checked per evaluation, besides its own point.
pub const EXTRA_POINTS: usize = 4;

/// `tracing` target of the per-evaluation events.
pub const TRACING_TARGET: &str = "akita_types::compression_weights_crosscheck";

/// Evaluation `i` (process-wide count) draws its random points from `SEED + i`.
const SEED: u64 = 0x6372_6f73_7363_6865;

/// Builder outcomes kept for attribution to a later evaluation; the oldest
/// half is dropped beyond this (outcomes of tables never evaluated, such as
/// the prover's).
const PENDING_CAP: usize = 1 << 20;

/// Groups, and the table entries they cover, in one evaluation tier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierCount {
    /// Groups.
    pub groups: u64,
    /// Table entries covered by the groups, zero entries included.
    pub entries: u64,
}

impl TierCount {
    fn add(&mut self, other: Self) {
        self.groups += other.groups;
        self.entries += other.entries;
    }
}

/// How the groups and ungrouped events of evaluated tables were evaluated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tiers {
    /// Dyadic blocks of a geometric table (the lifted gadget powers), each
    /// closed as a product of logarithmically many factors.
    pub geometric_dyadic: TierCount,
    /// Dyadic blocks of any other table (evaluated compression columns), each
    /// block's extension cached.
    pub cached_dyadic: TierCount,
    /// High strides that are not a power of two: a loop over the per-event
    /// equality window.
    pub window: TierCount,
    /// Ungrouped events with an aligned span (per-event path).
    pub ungrouped_aligned: u64,
    /// Ungrouped events with an unaligned span (per-event fallback window).
    pub ungrouped_unaligned: u64,
}

impl Tiers {
    fn add(&mut self, other: &Self) {
        self.geometric_dyadic.add(other.geometric_dyadic);
        self.cached_dyadic.add(other.cached_dyadic);
        self.window.add(other.window);
        self.ungrouped_aligned += other.ungrouped_aligned;
        self.ungrouped_unaligned += other.ungrouped_unaligned;
    }

    fn json(&self) -> String {
        format!(
            "{{\"geometric_dyadic\":[{},{}],\"cached_dyadic\":[{},{}],\"window\":[{},{}],\
             \"ungrouped_aligned\":{},\"ungrouped_unaligned\":{}}}",
            self.geometric_dyadic.groups,
            self.geometric_dyadic.entries,
            self.cached_dyadic.groups,
            self.cached_dyadic.entries,
            self.window.groups,
            self.window.entries,
            self.ungrouped_aligned,
            self.ungrouped_unaligned
        )
    }
}

/// Outcomes of the grouped builders.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuilderCounts {
    /// Recomposition calls recorded as groups.
    pub recomposition_grouped: u64,
    /// Groups recorded by those calls.
    pub recomposition_groups: u64,
    /// Recomposition calls pushed event by event.
    pub recomposition_fallback: u64,
    /// Compression rows whose columns were recorded as one group.
    pub column_grouped: u64,
    /// Compression rows with a zero row weight or only zero columns, which
    /// record nothing.
    pub column_zero: u64,
    /// Compression rows pushed event by event.
    pub column_fallback: u64,
    /// Calls whose grouped events were compared with the canonical loop's.
    pub compared: u64,
    /// Compared calls whose events differ.
    pub mismatches: u64,
    /// Event-by-event calls by reason.
    pub fallback_reasons: Vec<(&'static str, u64)>,
}

impl BuilderCounts {
    fn add(&mut self, outcome: BuilderOutcome, compared: Option<bool>) {
        match outcome {
            BuilderOutcome::RecompositionGrouped { groups } => {
                self.recomposition_grouped += 1;
                self.recomposition_groups += groups as u64;
            }
            BuilderOutcome::RecompositionFallback(reason) => {
                self.recomposition_fallback += 1;
                self.add_reason(reason);
            }
            BuilderOutcome::ColumnGrouped => self.column_grouped += 1,
            BuilderOutcome::ColumnZero => self.column_zero += 1,
            BuilderOutcome::ColumnFallback(reason) => {
                self.column_fallback += 1;
                self.add_reason(reason);
            }
        }
        if let Some(equal) = compared {
            self.compared += 1;
            if !equal {
                self.mismatches += 1;
            }
        }
    }

    fn add_reason(&mut self, reason: &'static str) {
        if let Some(entry) = self.fallback_reasons.iter_mut().find(|(r, _)| *r == reason) {
            entry.1 += 1;
        } else {
            self.fallback_reasons.push((reason, 1));
        }
    }

    fn json(&self) -> String {
        let reasons = self
            .fallback_reasons
            .iter()
            .map(|(reason, count)| format!("\"{reason}\":{count}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"recomposition_grouped\":{},\"recomposition_groups\":{},\
             \"recomposition_fallback\":{},\"column_grouped\":{},\"column_zero\":{},\
             \"column_fallback\":{},\"compared\":{},\"mismatches\":{},\
             \"fallback_reasons\":{{{reasons}}}}}",
            self.recomposition_grouped,
            self.recomposition_groups,
            self.recomposition_fallback,
            self.column_grouped,
            self.column_zero,
            self.column_fallback,
            self.compared,
            self.mismatches
        )
    }
}

/// Process-wide cross-check counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrossCheckReport {
    /// `evaluate_at_point` calls.
    pub evaluations: u64,
    /// Points at which the grouped and per-event values were compared.
    pub points: u64,
    /// Compared points whose values differ.
    pub mismatches: u64,
    /// Evaluation tiers, summed over the evaluations.
    pub tiers: Tiers,
    /// Builder outcomes over every table built (prover and verifier).
    pub builders: BuilderCounts,
}

/// A snapshot of the process-wide counters.
pub fn report() -> CrossCheckReport {
    with_state(|state| state.report.clone())
}

/// The outcome of one builder call.
#[derive(Clone, Copy, Debug)]
pub(super) enum BuilderOutcome {
    RecompositionGrouped { groups: usize },
    RecompositionFallback(&'static str),
    ColumnGrouped,
    ColumnZero,
    ColumnFallback(&'static str),
}

struct Observation {
    table: (usize, usize),
    outcome: BuilderOutcome,
    compared: Option<bool>,
}

#[derive(Default)]
struct State {
    report: CrossCheckReport,
    pending: Vec<Observation>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static EVALUATIONS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static OBSERVING: Cell<bool> = const { Cell::new(false) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    f(state.get_or_insert_with(State::default))
}

/// Marks a builder observation in progress on this thread, so the twin builds
/// it runs are not observed themselves.
pub(super) struct ObserverGuard;

impl ObserverGuard {
    pub(super) fn enter() -> Option<Self> {
        OBSERVING.with(|observing| (!observing.replace(true)).then_some(Self))
    }
}

impl Drop for ObserverGuard {
    fn drop(&mut self) {
        OBSERVING.with(|observing| observing.set(false));
    }
}

/// Record one builder call's outcome for the table `table`; a failed
/// comparison is an error after it is counted.
pub(super) fn record_builder(
    table: (usize, usize),
    outcome: BuilderOutcome,
    compared: Option<bool>,
) -> Result<(), AkitaError> {
    with_state(|state| {
        state.report.builders.add(outcome, compared);
        if state.pending.len() >= PENDING_CAP {
            state.pending = state.pending.split_off(PENDING_CAP / 2);
        }
        state.pending.push(Observation {
            table,
            outcome,
            compared,
        });
    });
    if compared == Some(false) {
        return Err(AkitaError::Internal(format!(
            "compression weights cross-check: grouped builder events differ from the canonical \
             loop ({outcome:?})"
        )));
    }
    Ok(())
}

/// Take the builder outcomes of the most recent build of `table`: the last
/// contiguous run of outcomes recorded for it.
fn claim(pending: &mut Vec<Observation>, table: (usize, usize)) -> BuilderCounts {
    let mut counts = BuilderCounts::default();
    let Some(last) = pending.iter().rposition(|entry| entry.table == table) else {
        return counts;
    };
    let first = pending
        .get(..last)
        .and_then(|before| before.iter().rposition(|entry| entry.table != table))
        .map_or(0, |index| index + 1);
    for entry in pending.drain(first..=last) {
        counts.add(entry.outcome, entry.compared);
    }
    counts
}

/// SplitMix64, for the seeded random points.
struct SplitMix64(u64);

impl RngCore for SplitMix64 {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(bytes.get(..chunk.len()).unwrap_or(&bytes[..]));
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl<E: Field> CompressionRelationWeights<E> {
    /// Evaluate the table's multilinear extension at one full witness point.
    ///
    /// The value is the default build's. The evaluation is also compared with
    /// the per-event evaluation of the same table at `point` and at
    /// [`EXTRA_POINTS`] seeded random points; a difference is an error.
    #[tracing::instrument(skip_all, name = "compression_relation_mle")]
    pub fn evaluate_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        let events = self.evaluate_events_at_point(point)?;
        let value = events + self.evaluate_groups_at_point(point)?;
        self.cross_check_evaluation(point, value)?;
        Ok(value)
    }

    fn grouped_value(&self, point: &[E]) -> Option<E> {
        let events = self.evaluate_events_at_point(point).ok()?;
        Some(events + self.evaluate_groups_at_point(point).ok()?)
    }

    fn cross_check_evaluation(&self, point: &[E], value: E) -> Result<(), AkitaError> {
        let index = EVALUATIONS.fetch_add(1, Ordering::Relaxed);
        let per_event = self.crosscheck_expanded().ok();
        let per_event_value = |at: &[E]| {
            per_event
                .as_ref()
                .and_then(|table| table.evaluate_events_at_point(at).ok())
        };
        // Point 0 is the evaluation's own point; points 1.. are random.
        let mut differing = Vec::new();
        if per_event_value(point) != Some(value) {
            differing.push(0);
        }
        let mut rng = SplitMix64(SEED.wrapping_add(index));
        for extra in 1..=EXTRA_POINTS {
            let random = (0..point.len())
                .map(|_| E::random(&mut rng))
                .collect::<Vec<_>>();
            let grouped = self.grouped_value(&random);
            if grouped.is_none() || per_event_value(&random) != grouped {
                differing.push(extra);
            }
        }
        let tiers = self.crosscheck_tiers();
        let points = 1 + EXTRA_POINTS as u64;
        let builder = with_state(|state| {
            state.report.evaluations += 1;
            state.report.points += points;
            state.report.mismatches += differing.len() as u64;
            state.report.tiers.add(&tiers);
            claim(&mut state.pending, self.crosscheck_key())
        });
        tracing::info!(
            target: TRACING_TARGET,
            "{{\"evaluation\":{index},\"num_vars\":{},\"physical_field_len\":{},\
             \"points\":{points},\"mismatches\":{},\"tiers\":{},\"builder\":{}}}",
            point.len(),
            self.physical_field_len,
            differing.len(),
            tiers.json(),
            builder.json()
        );
        if differing.is_empty() {
            Ok(())
        } else {
            Err(AkitaError::Internal(format!(
                "compression weights cross-check: grouped and per-event evaluations differ at \
                 evaluation {index}, points {differing:?} (0 is the evaluation's own point)"
            )))
        }
    }
}
