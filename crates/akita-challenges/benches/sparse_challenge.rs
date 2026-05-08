//! Microbenchmarks for sparse stage-1 challenge sampling at `D=32`.
//!
//! These benchmarks compare the fp128 `D=32` family
//! `Uniform { weight: 32, nonzero_coeffs: ±[1..8] }` against the current
//! preset `BoundedL1Norm` (`M=8, B=121`) from
//! `specs/bounded-l1-sparse-challenge.md`.
//!
//! Each `batch_<N>` case measures one `sample_sparse_challenges(N)` call:
//! one transcript absorb, one XOF seeding, and `N` per-challenge decodes.
//! Reads the steady-state per-challenge cost; the `BoundedL1Norm`
//! suffix-count table is precomputed at compile time so the gap between
//! `Uniform` and `BoundedL1Norm` here reflects the streaming
//! rank-unranking decode cost.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p akita-challenges --bench sparse_challenge
//! ```

#![allow(missing_docs)]

use akita_challenges::{sample_sparse_challenges, SparseChallengeConfig};
use akita_field::Prime128OffsetA7F7;
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{Blake2bTranscript, Transcript};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

// 128-bit base field used by the production stage-1 path; matches the field
// used by the broader e2e benches in `akita-pcs`.
type F = Prime128OffsetA7F7;

const D: usize = 32;

/// Batch counts that bracket the realistic stage-1 fan-out for `D=32`
/// presets. Smaller counts emphasize per-call (transcript absorb) overhead;
/// larger counts emphasize the amortized per-challenge decode cost.
const BATCH_SIZES: &[usize] = &[1, 1 << 6, 1 << 12, 1 << 15];

fn fresh_transcript() -> Blake2bTranscript<F> {
    let mut t = Blake2bTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    // Seed once with a non-empty transcript so we are not measuring the
    // empty-transcript fast path.
    t.append_field(b"bench-seed", &F::from_u64(0xC0FFEE));
    t
}

fn cfg_uniform_d32_legacy() -> SparseChallengeConfig {
    SparseChallengeConfig::Uniform {
        weight: 32,
        nonzero_coeffs: (-8..=8).filter(|&c| c != 0).collect(),
    }
}

fn cfg_bounded_l1_d32() -> SparseChallengeConfig {
    SparseChallengeConfig::BoundedL1Norm
}

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_challenge_d32_batch");
    let cases: &[(&str, SparseChallengeConfig)] = &[
        ("uniform_w32_alpha8", cfg_uniform_d32_legacy()),
        ("bounded_l1_m8_b121", cfg_bounded_l1_d32()),
    ];
    for &n in BATCH_SIZES {
        group.throughput(Throughput::Elements(n as u64));
        for (name, cfg) in cases {
            let id = BenchmarkId::new(*name, n);
            group.bench_with_input(id, &n, |b, &n| {
                b.iter(|| {
                    let mut tr = fresh_transcript();
                    let challenges = sample_sparse_challenges::<F, _, D>(
                        &mut tr,
                        b"bench/batch",
                        n,
                        black_box(cfg),
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
