//! Per-process reach counters and timing, flushed periodically to a file.
//!
//! The campaign runner sets `AKITA_FUZZ_STATS_FILE`; each worker overwrites
//! its own file atomically at most every few seconds. Counter names form a
//! small fixed set, so memory stays bounded.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct State {
    counters: BTreeMap<&'static str, u64>,
    nanos: BTreeMap<&'static str, u128>,
    last_flush: Instant,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
const FLUSH_INTERVAL: Duration = Duration::from_secs(5);

fn with_state(f: impl FnOnce(&mut State)) {
    let Ok(mut guard) = STATE.lock() else {
        return;
    };
    let state = guard.get_or_insert_with(|| State {
        counters: BTreeMap::new(),
        nanos: BTreeMap::new(),
        last_flush: Instant::now(),
    });
    f(state);
    if state.last_flush.elapsed() >= FLUSH_INTERVAL {
        state.last_flush = Instant::now();
        flush(state);
    }
}

pub fn count(name: &'static str) {
    with_state(|state| *state.counters.entry(name).or_default() += 1);
}

/// Time `f` under `name`; the phase count is recorded too.
pub fn time<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = f();
    let elapsed = started.elapsed().as_nanos();
    with_state(|state| {
        *state.nanos.entry(name).or_default() += elapsed;
        *state.counters.entry(name).or_default() += 1;
    });
    out
}

fn flush(state: &State) {
    let Some(path) = std::env::var_os("AKITA_FUZZ_STATS_FILE") else {
        return;
    };
    let mut json = String::from("{\"counters\":{");
    for (index, (name, value)) in state.counters.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&format!("\"{name}\":{value}"));
    }
    json.push_str("},\"seconds\":{");
    for (index, (name, nanos)) in state.nanos.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        json.push_str(&format!("\"{name}\":{:.3}", *nanos as f64 / 1e9));
    }
    json.push_str("}}\n");
    let path = std::path::PathBuf::from(path);
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, json).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}
