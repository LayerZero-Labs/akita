#![allow(missing_docs)]

#[path = "fold_linf_stats_report.rs"]
mod fold_linf_stats_report;
#[path = "profile/report.rs"]
mod report;
#[path = "profile/workload.rs"]
mod workload;

use std::env;

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_types::{AkitaScheduleLookupKey, CommitmentGroupScheduleKey, LevelParams, OpeningBatchShape};
use fold_linf_stats_report::{print_fold_linf_aggregate_report, FoldLinfLevelSample};
use workload::{run_dense_fold_linf_sample, run_onehot_fold_linf_sample};

type F = fp128::Field;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn resolve_layout<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> LevelParams {
    let opening_batch = OpeningBatchShape::new(nv, 1).expect("singleton opening");
    Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout")
}

fn collect_samples<Cfg: CommitmentConfig<Field = F>>(
    nv: usize,
    iterations: usize,
    seed_base: u64,
    sample: fn(usize, &LevelParams, &akita_types::Schedule, u64) -> Vec<akita_prover::FoldGrindObservation>,
) -> Vec<FoldLinfLevelSample> {
    let layout = resolve_layout::<Cfg>(nv);
    let plan = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        CommitmentGroupScheduleKey::singleton(nv),
    ))
    .expect("schedule");
    report::print_layout(&layout, 1, Cfg::decomposition().field_bits());

    let mut samples = Vec::new();
    for iter in 0..iterations {
        let seed = seed_base.wrapping_add(iter as u64);
        eprintln!("--- iteration {iter} seed={seed:#x} ---");
        let observations = sample(nv, &layout, &plan, seed);
        for observation in observations {
            samples.push(FoldLinfLevelSample {
                iteration: iter as u32,
                observation,
            });
        }
    }
    samples
}

fn run_mode(
    mode: &str,
    nv: usize,
    iterations: usize,
    seed_base: u64,
    max_tstar_reduction: f64,
) {
    eprintln!(
        "fold_linf_stats: mode={mode} nv={nv} iterations={iterations} seed_base={seed_base:#x} max_tstar_reduction={max_tstar_reduction:.2}"
    );

    let samples = match mode {
        "onehot_fp128_d64" => {
            type Cfg = fp128::D64OneHot;
            collect_samples::<Cfg>(nv, iterations, seed_base, run_onehot_fold_linf_sample::<F, { Cfg::D }, Cfg>)
        }
        "dense_fp128_d64" => {
            type Cfg = fp128::D64Full;
            collect_samples::<Cfg>(nv, iterations, seed_base, run_dense_fold_linf_sample::<F, { Cfg::D }, Cfg>)
        }
        other => {
            eprintln!("unsupported AKITA_MODE for fold_linf_stats: {other}");
            eprintln!("supported: onehot_fp128_d64, dense_fp128_d64");
            std::process::exit(2);
        }
    };

    print_fold_linf_aggregate_report(mode, iterations as u32, &samples, max_tstar_reduction);
}

fn main() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    if cfg!(debug_assertions) && env::var("AKITA_ALLOW_DEBUG_PROFILE").as_deref() != Ok("1") {
        eprintln!("examples/fold_linf_stats must be run with --release.");
        eprintln!("Set AKITA_ALLOW_DEBUG_PROFILE=1 to override.");
        std::process::exit(2);
    }

    let nv = env_usize("AKITA_NUM_VARS", 23);
    let iterations = env_usize("AKITA_FOLD_LINF_ITERATIONS", 10);
    let seed_base = env::var("AKITA_FOLD_LINF_SEED_BASE")
        .ok()
        .and_then(|s| {
            if let Some(hex) = s.strip_prefix("0x") {
                u64::from_str_radix(hex, 16).ok()
            } else {
                s.parse().ok()
            }
        })
        .unwrap_or(0xbeef_cafe);
    let max_tstar_reduction = env_f64("AKITA_FOLD_LINF_SNAP_MAX_TSTAR_REDUCTION", 0.50);
    let mode = env::var("AKITA_MODE").unwrap_or_else(|_| "dense_fp128_d64".to_string());

    run_mode(&mode, nv, iterations, seed_base, max_tstar_reduction);
}
