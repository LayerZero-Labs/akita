#![allow(missing_docs)]

#[cfg(all(
    feature = "profile-bench-selected",
    not(any(
        feature = "profile-ci-fp32",
        feature = "profile-ci-fp64",
        feature = "profile-ci-fp128-base",
        feature = "profile-ci-multi-group-direct",
        feature = "profile-ci-multi-group-recursive",
        feature = "profile-ci-multi-group-recursive-w8r2",
        feature = "profile-ci-distributed",
    ))
))]
compile_error!("profile-bench-selected is internal; enable one profile-ci-* group instead");

mod modes;
mod monitor;
mod ntt_prewarm;
mod parallel;
#[path = "../../benches/support/relation_phase_timing.rs"]
mod relation_phase_timing;
mod report;
mod trace_report;
mod verifier;
#[cfg_attr(
    any(feature = "profile-onehot-fp128", feature = "profile-bench-selected"),
    allow(dead_code)
)]
mod workload;

use akita_prover::CpuBackend;
use std::env;
use std::fs;
use std::io::BufWriter;
use std::path::Path;
use std::time::Duration;
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
    parallel::ProfileThreadPools::init();

    if cfg!(debug_assertions) && env::var("AKITA_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        eprintln!("examples/profile must be run with --release for meaningful timings.");
        eprintln!("Re-run with: cargo run --release --example profile");
        eprintln!("Set AKITA_ALLOW_DEBUG_PROFILE=1 to override this guard.");
        std::process::exit(2);
    }

    let nv: usize = env::var("AKITA_NUM_VARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let num_polys = env_usize("AKITA_NUM_POLYS", 1);

    let mode = env::var("AKITA_MODE").unwrap_or_else(|_| "onehot_fp128".to_string());
    let enable_trace = env_flag("AKITA_PROFILE_TRACE", true);
    let enable_ansi = env_flag("AKITA_PROFILE_ANSI", true);
    let span_events = if env_flag("AKITA_PROFILE_SPAN_CLOSES", true) {
        FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };
    let log_filter = env::var("AKITA_PROFILE_LOG").unwrap_or_else(|_| "trace".to_string());

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
        .with_target(false)
        .with_filter(EnvFilter::try_new(&log_filter).unwrap_or_else(|_| EnvFilter::new("trace")));
    let chrome_guard = if enable_trace {
        fs::create_dir_all("profile_traces").ok();
        let file = fs::File::create(&trace_file).expect("Failed to create trace file");
        let buffered = BufWriter::with_capacity(4 * 1024 * 1024, file);
        let (chrome_layer, guard) = ChromeLayerBuilder::new()
            .include_args(true)
            .writer(buffered)
            .build();
        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(relation_phase_timing::RelationPhaseTimingLayer)
            .with(chrome_layer.with_filter(EnvFilter::new("trace")))
            .init();
        tracing::info!(trace_file = %trace_file, "Perfetto trace");
        Some(guard)
    } else {
        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(relation_phase_timing::RelationPhaseTimingLayer)
            .init();
        tracing::info!("Perfetto trace disabled");
        None
    };
    let monitor_enabled = enable_trace && env_flag("AKITA_PROFILE_MONITOR", true);
    tracing::info!(num_vars = nv, num_polys, mode = %mode, "profile config");
    let cpu = CpuBackend::DEFAULT;
    tracing::info!(
        max_cached_ring_switch_elements = cpu.max_cached_ring_switch_elements(),
        commit_scratch_bytes_per_worker = cpu.commit_scratch_bytes_per_worker(),
        "CPU resource policy"
    );
    eprintln!(
        "[profile] cpu_policy: max_cached_ring_switch_elements={}, commit_scratch_bytes_per_worker={}",
        cpu.max_cached_ring_switch_elements(),
        cpu.commit_scratch_bytes_per_worker(),
    );
    modes::log_active_fp128_prime_probe();

    {
        let _run_span = tracing::info_span!(
            trace_report::ROOT_SPAN,
            mode = %mode,
            num_vars = nv,
            num_polys,
            prove_threads = parallel::ProfileThreadPools::get().prove_threads(),
        )
        .entered();
        let resource_monitor = monitor_enabled.then(|| {
            let interval_ms = env_usize("AKITA_PROFILE_MONITOR_INTERVAL_MS", 100).max(10);
            tracing::info!(interval_ms, "starting process resource monitor");
            monitor::ResourceMonitor::start(Duration::from_millis(interval_ms as u64))
        });
        #[cfg(not(feature = "profile-onehot-fp128"))]
        {
            if mode == "all" {
                modes::run_all_profile_modes(nv);
            } else {
                modes::run_profile_mode(&mode, nv, num_polys);
            }
        }
        #[cfg(feature = "profile-onehot-fp128")]
        modes::run_profile_mode(&mode, nv, num_polys);
        drop(resource_monitor);
    }

    let peak_rss_bytes = monitor::peak_rss_bytes();
    if enable_trace {
        tracing::info!(trace_file = %trace_file, "Done. Trace saved");
    } else {
        tracing::info!("Done");
    }
    drop(chrome_guard);
    if enable_trace {
        let context = trace_report::ReportContext {
            mode: &mode,
            num_vars: nv,
            num_polys,
            prove_threads: parallel::ProfileThreadPools::get().prove_threads(),
            logical_cpus: std::thread::available_parallelism().map_or(1, |count| count.get()),
            timestamp_unix_secs: timestamp,
            peak_rss_bytes,
        };
        match trace_report::finalize_trace(Path::new(&trace_file), &context) {
            Ok(summary_file) => eprintln!(
                "[profile] Perfetto trace and profile summary ready: trace={} summary={}",
                trace_file,
                summary_file.display()
            ),
            Err(error) => eprintln!("[profile] trace finalization failed: {error}"),
        }
    }
}
