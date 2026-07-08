#![allow(missing_docs)]

mod modes;
mod report;
mod workload;

use std::env;
use std::fs;
use std::io::BufWriter;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| value != "0")
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    if cfg!(debug_assertions) && env::var("AKITA_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        eprintln!("examples/profile must be run with --release for meaningful timings.");
        eprintln!("Re-run with: cargo run --release --example profile");
        eprintln!("Set AKITA_ALLOW_DEBUG_PROFILE=1 to override this guard.");
        std::process::exit(2);
    }

    let nv: usize = env::var("AKITA_NUM_VARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let num_polys = env_usize("AKITA_NUM_POLYS", 1);

    // Keep the default explicit: old adaptive `full`/`onehot` selectors are
    // intentionally not part of the profile surface. D64 is the default fp128
    // profile preset (`onehot_fp128_d64`); use best_*_schedule to compare D64 vs D128.
    let mode = env::var("AKITA_MODE").unwrap_or_else(|_| "onehot_fp128_d64".to_string());
    let enable_trace = env_flag("AKITA_PROFILE_TRACE", true);
    let enable_ansi = env_flag("AKITA_PROFILE_ANSI", true);
    let span_events = if env_flag("AKITA_PROFILE_SPAN_CLOSES", true) {
        FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };
    let log_filter =
        EnvFilter::try_new(env::var("AKITA_PROFILE_LOG").unwrap_or_else(|_| "trace".to_string()))
            .unwrap_or_else(|_| EnvFilter::new("trace"));

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let trace_file = if num_polys == 1 {
        format!("profile_traces/akita_nv{nv}_{mode}_{timestamp}.json")
    } else {
        format!("profile_traces/akita_nv{nv}_np{num_polys}_{mode}_{timestamp}.json")
    };

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(enable_ansi)
        .with_span_events(span_events)
        .compact()
        .with_target(false);
    let _chrome_guard = if enable_trace {
        fs::create_dir_all("profile_traces").ok();
        let file = fs::File::create(&trace_file).expect("Failed to create trace file");
        let buffered = BufWriter::with_capacity(4 * 1024 * 1024, file);
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .writer(buffered)
            .build();
        tracing_subscriber::registry()
            .with(log_filter)
            .with(fmt_layer)
            .with(chrome_layer)
            .init();
        tracing::info!(trace_file = %trace_file, "Perfetto trace");
        Some(guard)
    } else {
        tracing_subscriber::registry()
            .with(log_filter)
            .with(fmt_layer)
            .init();
        tracing::info!("Perfetto trace disabled");
        None
    };
    tracing::info!(num_vars = nv, num_polys, mode = %mode, "profile config");
    modes::log_active_fp128_prime_probe();

    if mode == "all" {
        modes::run_all_profile_modes(nv);
    } else {
        modes::run_profile_mode(&mode, nv, num_polys);
    }

    if enable_trace {
        tracing::info!(trace_file = %trace_file, "Done. Trace saved");
    } else {
        tracing::info!("Done");
    }
}
