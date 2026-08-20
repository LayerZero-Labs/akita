//! Microbenchmarks for ring fold challenge sampling.
//!
//! Compares production signed-sparse `(31, 10)` against pm1-only `{23, 0}` at
//! the same ring degree to bracket position-shuffle vs sign-decode cost.
//!
//! Each `batch_<N>` case measures one `sample_sparse_challenges(N)` call:
//! one transcript absorb, one group-root squeeze, and `N` indexed coordinate
//! streams with one sparse decode each.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p akita-challenges --bench sparse_challenge
//! ```

#![allow(missing_docs)]

use akita_challenges::{
    sample_sparse_challenges, Challenges, SparseChallengeConfig, D64_PRODUCTION_PM1_COUNT,
    D64_PRODUCTION_PM2_COUNT,
};
use akita_field::Prime128OffsetA7F7;
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

type F = Prime128OffsetA7F7;

const D: usize = 64;

const BATCH_SIZES: &[usize] = &[1, 1 << 6, 1 << 12, 1 << 15];

fn fresh_transcript() -> AkitaTranscript<F> {
    let mut t = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"bench-seed", &F::from_u64(0xC0FFEE));
    t
}

fn cfg_signed_sparse_production() -> SparseChallengeConfig {
    SparseChallengeConfig {
        count_pm1: D64_PRODUCTION_PM1_COUNT,
        count_pm2: D64_PRODUCTION_PM2_COUNT,
    }
}

fn cfg_pm1_only_d64() -> SparseChallengeConfig {
    SparseChallengeConfig::pm1_only(23)
}

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_challenge_d64_batch");
    let cases: &[(&str, SparseChallengeConfig)] = &[
        ("signed_sparse_production", cfg_signed_sparse_production()),
        ("pm1_only_w23", cfg_pm1_only_d64()),
    ];
    for &n in BATCH_SIZES {
        group.throughput(Throughput::Elements(n as u64));
        for (name, cfg) in cases {
            let id = BenchmarkId::new(*name, n);
            group.bench_with_input(id, &n, |b, &n| {
                b.iter(|| {
                    let mut tr = fresh_transcript();
                    let challenges = sample_sparse_challenges::<F, _>(
                        &mut tr,
                        b"bench/batch",
                        D,
                        n,
                        black_box(cfg),
                        0,
                    )
                    .expect("batch sparse challenges");
                    black_box(challenges)
                });
            });
        }
    }
    group.finish();
}

fn bench_sparse_ladder(c: &mut Criterion) {
    const BATCH: usize = 1 << 12;
    let mut group = c.benchmark_group("sparse_challenge_ladder_batch_4096");
    group.throughput(Throughput::Elements(BATCH as u64));
    for ring_d in [256usize, 512, 1024, 2048] {
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d)
            .expect("production challenge configuration");
        group.bench_with_input(
            BenchmarkId::from_parameter(ring_d),
            &ring_d,
            |b, &ring_d| {
                b.iter(|| {
                    let mut tr = fresh_transcript();
                    let challenges = sample_sparse_challenges::<F, _>(
                        &mut tr,
                        b"bench/sparse-ladder",
                        ring_d,
                        BATCH,
                        black_box(&cfg),
                        0,
                    )
                    .expect("batch sparse challenges");
                    black_box(challenges)
                });
            },
        );
    }
    group.finish();
}

fn bench_sparse_evaluation(c: &mut Criterion) {
    let alpha = F::from_u64(17);
    let mut power = F::one();
    let alpha_powers = (0..D)
        .map(|_| {
            let current = power;
            power *= alpha;
            current
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("sparse_challenge_evaluation");
    for batch in [64usize, 256, 512, 1024, 2048, 4096, 1 << 16] {
        let mut transcript = fresh_transcript();
        let sampled = sample_sparse_challenges::<F, _>(
            &mut transcript,
            b"bench/evaluation",
            D,
            batch,
            &cfg_signed_sparse_production(),
            0,
        )
        .expect("batch sparse challenges");
        let challenges = Challenges::from_sparse(sampled, batch, 1).expect("valid batch layout");
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::new("hybrid", batch),
            &challenges,
            |b, challenges| {
                b.iter(|| {
                    black_box(
                        challenges
                            .evals_at_pows::<F, F>(black_box(&alpha_powers))
                            .expect("valid challenge evaluations"),
                    );
                });
            },
        );
        if batch >= 512 {
            group.bench_with_input(
                BenchmarkId::new("sequential", batch),
                &challenges,
                |b, challenges| {
                    b.iter(|| {
                        black_box(
                            challenges
                                .as_slice()
                                .iter()
                                .map(|challenge| challenge.eval_at_pows::<F, F>(&alpha_powers))
                                .collect::<Result<Vec<_>, _>>()
                                .expect("valid challenge evaluations"),
                        );
                    });
                },
            );
        }
    }
    for batch in [1024, 2048, 4096] {
        for ring_d in [128usize, 256, 512, 1024, 2048] {
            let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d)
                .expect("production challenge configuration");
            let mut transcript = fresh_transcript();
            let sampled = sample_sparse_challenges::<F, _>(
                &mut transcript,
                b"bench/evaluation-ladder",
                ring_d,
                batch,
                &cfg,
                0,
            )
            .expect("batch sparse challenges");
            let challenges =
                Challenges::from_sparse(sampled, batch, 1).expect("valid batch layout");
            let mut power = F::one();
            let alpha_powers = (0..ring_d)
                .map(|_| {
                    let current = power;
                    power *= alpha;
                    current
                })
                .collect::<Vec<_>>();
            for mode in ["hybrid", "sequential"] {
                group.bench_with_input(
                    BenchmarkId::new(format!("d{ring_d}_{mode}"), batch),
                    &challenges,
                    |b, challenges| {
                        b.iter(|| {
                            let evaluations = if mode == "hybrid" {
                                challenges.evals_at_pows::<F, F>(&alpha_powers)
                            } else {
                                challenges
                                    .as_slice()
                                    .iter()
                                    .map(|challenge| challenge.eval_at_pows::<F, F>(&alpha_powers))
                                    .collect()
                            };
                            black_box(evaluations.expect("valid challenge evaluations"));
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(
    sparse_challenge,
    bench_batch,
    bench_sparse_ladder,
    bench_sparse_evaluation
);
criterion_main!(sparse_challenge);
