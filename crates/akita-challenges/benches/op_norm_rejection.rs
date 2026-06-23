//! Microbenchmarks for D=64 exact-shell operator-norm rejection sampling.
//!
//! Run all groups:
//!
//! ```text
//! cargo bench -p akita-challenges --bench op_norm_rejection
//! ```
//!
//! Predicate only (fast sanity check):
//!
//! ```text
//! cargo bench -p akita-challenges --bench op_norm_rejection -- decide
//! ```
//!
//! Production XOF replay at `n = 2^16`:
//!
//! ```text
//! cargo bench -p akita-challenges --bench op_norm_rejection -- xof_rejection/n=65536
//! ```

#![allow(missing_docs)]

use akita_challenges::op_norm_bench::Table;
use akita_challenges::{
    sample_sparse_challenges, sparse_challenges_from_seed, SparseChallenge, SparseChallengeConfig,
};
use akita_field::Prime128OffsetA7F7;
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use criterion::measurement::WallTime;
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use std::time::Duration;

type F = Prime128OffsetA7F7;

const D: usize = 64;

const POOL: usize = 4096;

/// Batch sizes: overhead probes, one mid-scale point, then `2^14`..=`2^16` stage-1 fan-out.
const BATCH_SIZES: &[usize] = &[1, 16, 256, 1 << 14, 1 << 15, 1 << 16];

struct ExactShellCase {
    name: &'static str,
    count_mag1: usize,
    count_mag2: usize,
    operator_norm_threshold: u32,
    /// Pass `true` to `sparse_challenges_from_seed` / `sample_sparse_challenges`.
    op_norm_rejection: bool,
}

const CASES: &[ExactShellCase] = &[
    ExactShellCase {
        name: "legacy_30_12_no_rejection",
        count_mag1: 30,
        count_mag2: 12,
        operator_norm_threshold: 54,
        op_norm_rejection: false,
    },
    ExactShellCase {
        name: "prod_31_11_t18_rejection",
        count_mag1: 31,
        count_mag2: 11,
        operator_norm_threshold: 18,
        op_norm_rejection: true,
    },
];

fn cfg(case: &ExactShellCase) -> SparseChallengeConfig {
    SparseChallengeConfig::ExactShell {
        count_mag1: case.count_mag1,
        count_mag2: case.count_mag2,
        operator_norm_threshold: case.operator_norm_threshold,
    }
}

fn fresh_transcript() -> AkitaTranscript<F> {
    let mut t = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"bench-seed", &F::from_u64(0xC0FFEE));
    t
}

fn op_norm_table() -> Table {
    Table::d64_q48()
}

fn shell_pool(case: &ExactShellCase) -> Vec<SparseChallenge> {
    let seed = [0x42u8; 32];
    sparse_challenges_from_seed::<D>(&seed, POOL, &cfg(case), case.op_norm_rejection)
        .expect("shell pool")
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, WallTime>, n: usize) {
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(n as u64));
    match n {
        n if n >= 1 << 16 => {
            group.sample_size(15);
            group.measurement_time(Duration::from_secs(12));
        }
        n if n >= 1 << 15 => {
            group.sample_size(20);
            group.measurement_time(Duration::from_secs(10));
        }
        n if n >= 1 << 12 => {
            group.sample_size(30);
            group.measurement_time(Duration::from_secs(7));
        }
        _ => {
            group.sample_size(50);
            group.measurement_time(Duration::from_secs(5));
        }
    }
}

/// Predicate-only: production transposed `i64` decide on the `(31,11)` pool at `T=18`.
fn bench_decide(c: &mut Criterion) {
    let case = &CASES[1];
    let table = op_norm_table();
    let pool = shell_pool(case);
    let t = u64::from(case.operator_norm_threshold);

    let mut group = c.benchmark_group("decide");
    group.throughput(Throughput::Elements(POOL as u64));

    group.bench_function("transposed_i64_production", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let ch = &pool[i & (pool.len() - 1)];
            i += 1;
            black_box(
                table
                    .decide_production(&ch.positions, &ch.coeffs, t)
                    .expect("production decide"),
            );
        });
    });

    group.finish();
}

/// One-time certified table construction (amortized cost reference).
fn bench_table_build(c: &mut Criterion) {
    c.bench_function("table_build_d64_q48", |b| {
        b.iter(|| black_box(Table::d64_q48()));
    });
}

/// Full production path: transcript absorb + SHAKE + rejection loop.
fn bench_transcript_batch(c: &mut Criterion) {
    for &n in BATCH_SIZES {
        let mut group = c.benchmark_group(format!("transcript_rejection/n={n}"));
        configure_group(&mut group, n);
        for case in CASES {
            let shell_cfg = cfg(case);
            let id = BenchmarkId::from_parameter(case.name);
            group.bench_with_input(id, &shell_cfg, |b, shell_cfg| {
                b.iter(|| {
                    let mut tr = fresh_transcript();
                    let challenges = sample_sparse_challenges::<F, _, D>(
                        &mut tr,
                        b"bench/opnorm",
                        n,
                        black_box(shell_cfg),
                        0,
                        case.op_norm_rejection,
                    )
                    .expect("sparse challenges");
                    black_box(challenges)
                });
            });
        }
        group.finish();
    }
}

/// XOF replay only: fixed 32-byte seed, no transcript/SHAKE overhead.
fn bench_xof_batch(c: &mut Criterion) {
    let seed = [0x42u8; 32];
    for &n in BATCH_SIZES {
        let mut group = c.benchmark_group(format!("xof_rejection/n={n}"));
        configure_group(&mut group, n);
        for case in CASES {
            let shell_cfg = cfg(case);
            let id = BenchmarkId::from_parameter(case.name);
            group.bench_with_input(id, &shell_cfg, |b, shell_cfg| {
                b.iter(|| {
                    let challenges = sparse_challenges_from_seed::<D>(
                        black_box(&seed),
                        n,
                        black_box(shell_cfg),
                        case.op_norm_rejection,
                    )
                    .expect("sparse challenges from seed");
                    black_box(challenges)
                });
            });
        }
        group.finish();
    }
}

criterion_group! {
    name = op_norm_rejection;
    config = Criterion::default().with_measurement::<WallTime>(WallTime);
    targets = bench_decide, bench_table_build, bench_transcript_batch, bench_xof_batch
}
criterion_main!(op_norm_rejection);
