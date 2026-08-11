//! TEMP BENCH HOOK: per-span busy-time aggregation for the z-first verifier
//! benchmark. Accumulates wall-clock time spent inside each named `tracing`
//! span (keyed by span name) between [`reset`] and [`snapshot`]. Spans nested
//! under `verify_L{n}` parent spans are keyed as `L{n}/name`. Remove with the
//! rest of the z-first benchmark instrumentation.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::span::Id;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// `None` until [`reset`] is called; only spans closing while `Some` are
/// recorded, so the prove phase (which runs before the first `reset`) is
/// ignored.
static AGG: Mutex<Option<HashMap<String, (Duration, u64)>>> = Mutex::new(None);

/// Begin a fresh aggregation window.
pub(crate) fn reset() {
    *AGG.lock().unwrap() = Some(HashMap::new());
}

/// Read the current window, sorted by descending total busy time.
pub(crate) fn snapshot() -> Vec<(String, Duration, u64)> {
    let guard = AGG.lock().unwrap();
    let mut out: Vec<(String, Duration, u64)> = guard
        .as_ref()
        .map(|map| {
            map.iter()
                .map(|(k, (d, c))| (k.clone(), *d, *c))
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    out
}

fn verify_level_from_name(name: &str) -> Option<usize> {
    name.strip_prefix("verify_L")
        .and_then(|suffix| suffix.parse().ok())
}

fn agg_key<'a, S>(span: tracing_subscriber::registry::SpanRef<'a, S>, _ctx: &Context<'_, S>) -> String
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    let name = span.metadata().name();
    let mut current = span.parent();
    while let Some(parent) = current {
        if let Some(level) = verify_level_from_name(parent.metadata().name()) {
            return format!("L{level}/{name}");
        }
        current = parent.parent();
    }
    name.to_string()
}

struct EnterAt(Instant);

pub(crate) struct SpanAggLayer;

impl<S> Layer<S> for SpanAggLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().replace(EnterAt(Instant::now()));
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let elapsed = span.extensions().get::<EnterAt>().map(|e| e.0.elapsed());
        let Some(elapsed) = elapsed else { return };
        let key = agg_key(span, &ctx);
        let mut guard = AGG.lock().unwrap();
        if let Some(map) = guard.as_mut() {
            let entry = map.entry(key).or_insert((Duration::ZERO, 0));
            entry.0 += elapsed;
            entry.1 += 1;
        }
    }
}
