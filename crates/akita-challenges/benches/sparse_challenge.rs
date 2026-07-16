//! Microbenchmarks for ring fold challenge sampling at `D=64`.
//!
//! Compares production signed-sparse `(31, 10)` against pm1-only `{23, 0}` at
//! the same ring degree to bracket position-shuffle vs sign-decode cost.
//!
//! Each `batch_<N>` case measures one `sample_sparse_challenges(N)` call:
//! one transcript absorb, one XOF seeding, and `N` per-challenge decodes.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p akita-challenges --bench sparse_challenge
//! ```

#![allow(missing_docs)]

use akita_challenges::{
    sample_sparse_challenges, SparseChallengeConfig, D64_PRODUCTION_PM1_COUNT,
    D64_PRODUCTION_PM2_COUNT,
};
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::Prime128OffsetA7F7;

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

criterion_group!(sparse_challenge, bench_batch);
criterion_main!(sparse_challenge);
