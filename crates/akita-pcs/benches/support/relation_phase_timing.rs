//! Nested verifier-phase timing for the selected relation-mode benchmarks.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tracing::{field::Visit, span::Attributes, span::Id, Subscriber};
use tracing_subscriber::{layer::Context, prelude::*, registry::LookupSpan, Layer};

const PHASE_COUNT: usize = TimedRelationPhase::ALL.len();
static ELAPSED_NANOS: [[AtomicU64; PHASE_COUNT]; 2] =
    [const { [const { AtomicU64::new(0) }; PHASE_COUNT] }; 2];
static CALLS: [[AtomicU64; PHASE_COUNT]; 2] =
    [const { [const { AtomicU64::new(0) }; PHASE_COUNT] }; 2];
static ACTIVE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimedRelationPhase {
    CoefficientFunctionalPreparation,
    StructuredGroups,
    SetupScan,
    QuotientTail,
    CompleteStage2,
}

impl TimedRelationPhase {
    const ALL: [Self; 5] = [
        Self::CoefficientFunctionalPreparation,
        Self::StructuredGroups,
        Self::SetupScan,
        Self::QuotientTail,
        Self::CompleteStage2,
    ];

    const fn index(self) -> usize {
        match self {
            Self::CoefficientFunctionalPreparation => 0,
            Self::StructuredGroups => 1,
            Self::SetupScan => 2,
            Self::QuotientTail => 3,
            Self::CompleteStage2 => 4,
        }
    }

    const fn span_name(self) -> &'static str {
        match self {
            Self::CoefficientFunctionalPreparation => "relation_coefficient_functional_preparation",
            Self::StructuredGroups => "relation_structured_groups",
            Self::SetupScan => "relation_setup_scan",
            Self::QuotientTail => "relation_quotient_tail",
            Self::CompleteStage2 => "stage2_verifier",
        }
    }

    const fn report_name(self) -> &'static str {
        match self {
            Self::CoefficientFunctionalPreparation => "coefficient_functional_preparation",
            Self::StructuredGroups => "structured_groups",
            Self::SetupScan => "setup_scan",
            Self::QuotientTail => "quotient_tail",
            Self::CompleteStage2 => "complete_stage2",
        }
    }

    fn from_span_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|phase| phase.span_name() == name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimedRelationMode {
    Quotient,
    Reduced,
}

impl TimedRelationMode {
    const ALL: [Self; 2] = [Self::Quotient, Self::Reduced];

    const fn from_reduced(reduced: bool) -> Self {
        if reduced {
            Self::Reduced
        } else {
            Self::Quotient
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Quotient => 0,
            Self::Reduced => 1,
        }
    }

    const fn report_name(self) -> &'static str {
        match self {
            Self::Quotient => "quotient",
            Self::Reduced => "reduced",
        }
    }
}

struct RelationModeVisitor(Option<TimedRelationMode>);

impl Visit for RelationModeVisitor {
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if field.name() == "reduced" {
            self.0 = Some(TimedRelationMode::from_reduced(value));
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

struct PhaseStart {
    started: Instant,
    mode: TimedRelationMode,
    phase: TimedRelationPhase,
}

struct RelationPhaseTimingLayer;

impl<S> Layer<S> for RelationPhaseTimingLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, context: Context<'_, S>) {
        if TimedRelationPhase::from_span_name(attrs.metadata().name())
            != Some(TimedRelationPhase::CompleteStage2)
        {
            return;
        }
        let mut visitor = RelationModeVisitor(None);
        attrs.record(&mut visitor);
        if let (Some(span), Some(mode)) = (context.span(id), visitor.0) {
            span.extensions_mut().insert(mode);
        }
    }

    fn on_enter(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else {
            return;
        };
        let Some(phase) = TimedRelationPhase::from_span_name(span.metadata().name()) else {
            return;
        };
        let mode = span
            .scope()
            .from_root()
            .find_map(|ancestor| ancestor.extensions().get::<TimedRelationMode>().copied());
        if let Some(mode) = mode {
            span.extensions_mut().insert(PhaseStart {
                started: Instant::now(),
                mode,
                phase,
            });
        }
    }

    fn on_exit(&self, id: &Id, context: Context<'_, S>) {
        let Some(span) = context.span(id) else {
            return;
        };
        let start = span.extensions_mut().remove::<PhaseStart>();
        if let Some(PhaseStart {
            started,
            mode,
            phase,
        }) = start
        {
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            ELAPSED_NANOS[mode.index()][phase.index()].fetch_add(elapsed, Ordering::Relaxed);
            CALLS[mode.index()][phase.index()].fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct ActiveCapture;

impl ActiveCapture {
    fn start() -> Self {
        reset();
        ACTIVE.store(true, Ordering::Relaxed);
        Self
    }
}

impl Drop for ActiveCapture {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Relaxed);
    }
}

fn reset() {
    for counters in [&ELAPSED_NANOS, &CALLS] {
        for mode in counters {
            for counter in mode {
                counter.store(0, Ordering::Relaxed);
            }
        }
    }
}

/// Install the phase-only tracing subscriber used by this benchmark process.
pub(crate) fn init() {
    tracing_subscriber::registry()
        .with(
            RelationPhaseTimingLayer.with_filter(tracing_subscriber::filter::dynamic_filter_fn(
                |metadata, _context| {
                    ACTIVE.load(Ordering::Relaxed)
                        && TimedRelationPhase::from_span_name(metadata.name()).is_some()
                },
            )),
        )
        .init();
}

/// Run a few honest replays and print the selected per-mode phase table.
pub(crate) fn report(label: &str, num_vars: usize, iterations: u64, mut operation: impl FnMut()) {
    let _capture = ActiveCapture::start();
    for _ in 0..iterations {
        operation();
    }
    assert!(
        CALLS
            .iter()
            .any(
                |mode| mode[TimedRelationPhase::CompleteStage2.index()].load(Ordering::Relaxed)
                    != 0
            ),
        "selected verifier replay produced no Stage-2 phase samples"
    );
    for mode in TimedRelationMode::ALL {
        let mode_index = mode.index();
        if CALLS[mode_index][TimedRelationPhase::CompleteStage2.index()].load(Ordering::Relaxed)
            == 0
        {
            continue;
        }
        for phase in TimedRelationPhase::ALL {
            let calls = CALLS[mode_index][phase.index()].load(Ordering::Relaxed);
            let elapsed = ELAPSED_NANOS[mode_index][phase.index()].load(Ordering::Relaxed);
            eprintln!(
                "relation_phase\t{label}\tnv{num_vars}\t{}\t{}\t{calls}\t{}",
                mode.report_name(),
                phase.report_name(),
                elapsed.checked_div(calls).unwrap_or(0),
            );
        }
    }
}

/// Measure only complete Stage-2 spans while replaying the public verifier.
pub(crate) fn measure_complete_stage2(iterations: u64, mut operation: impl FnMut()) -> Duration {
    let _capture = ActiveCapture::start();
    for _ in 0..iterations {
        operation();
    }
    assert!(
        CALLS
            .iter()
            .any(
                |mode| mode[TimedRelationPhase::CompleteStage2.index()].load(Ordering::Relaxed)
                    != 0
            ),
        "selected verifier replay produced no Stage-2 phase samples"
    );
    let nanos = ELAPSED_NANOS
        .iter()
        .map(|mode| mode[TimedRelationPhase::CompleteStage2.index()].load(Ordering::Relaxed))
        .sum();
    Duration::from_nanos(nanos)
}
