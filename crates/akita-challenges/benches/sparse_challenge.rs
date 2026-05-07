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

use akita_challenges::{
    sample_sparse_challenges, IntegerChallenge, SparseChallenge, SparseChallengeConfig,
    TensorStage1Challenges,
};
use akita_field::{FieldCore, Prime128OffsetA7F7};
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{Blake2bTranscript, Transcript};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

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

fn scalar_powers<F: FieldCore>(alpha: F, count: usize) -> Vec<F> {
    (0..count)
        .scan(F::one(), |power, _| {
            let out = *power;
            *power *= alpha;
            Some(out)
        })
        .collect()
}

fn bench_sparse(pos0: usize, pos1: usize, c0: i8, c1: i8) -> SparseChallenge {
    SparseChallenge {
        positions: vec![pos0 as u32, pos1 as u32],
        coeffs: vec![c0, c1],
    }
}

fn tensor_bench_fixture<const TD: usize>(
    left_len: usize,
    right_len: usize,
) -> (TensorStage1Challenges, Vec<F>, Vec<F>, Vec<F>, F) {
    let left = (0..left_len)
        .map(|idx| {
            bench_sparse(
                idx % TD,
                (idx * 7 + 3) % TD,
                if idx % 2 == 0 { 1 } else { -1 },
                2,
            )
        })
        .collect();
    let right = (0..right_len)
        .map(|idx| {
            bench_sparse(
                (idx * 5 + 1) % TD,
                (idx * 11 + 9) % TD,
                1,
                if idx % 3 == 0 { -2 } else { 2 },
            )
        })
        .collect();
    let u_weights: Vec<F> = (0..left_len)
        .map(|idx| F::from_u64((idx as u64 % 17) + 1))
        .collect();
    let v_weights: Vec<F> = (0..right_len)
        .map(|idx| F::from_u64((idx as u64 % 19) + 1))
        .collect();
    let alpha = F::from_u64(13);
    let alpha_pows = scalar_powers(alpha, TD);
    let alpha_pow_d_plus_one = alpha_pows[TD - 1] * alpha + F::one();
    (
        TensorStage1Challenges {
            left,
            right,
            left_len,
            right_len,
            num_claims: 1,
        },
        u_weights,
        v_weights,
        alpha_pows,
        alpha_pow_d_plus_one,
    )
}

fn expanded_tensor_weighted_sum<const TD: usize>(
    tensor: &TensorStage1Challenges,
    u_weights: &[F],
    v_weights: &[F],
    alpha_pows: &[F],
) -> F {
    let mut acc = F::zero();
    for (p, u) in u_weights.iter().copied().enumerate() {
        for (q, v) in v_weights.iter().copied().enumerate() {
            let product = IntegerChallenge::tensor_product::<TD>(&tensor.left[p], &tensor.right[q])
                .expect("tensor product");
            acc += u * v * product.eval_at_pows::<F, TD>(alpha_pows).expect("eval");
        }
    }
    acc
}

fn product_only_weighted_sum<const TD: usize>(
    tensor: &TensorStage1Challenges,
    u_weights: &[F],
    v_weights: &[F],
    alpha_pows: &[F],
) -> F {
    let left =
        tensor
            .left
            .iter()
            .zip(u_weights.iter())
            .fold(F::zero(), |acc, (challenge, &weight)| {
                acc + weight
                    * challenge
                        .eval_at_pows::<F, TD>(alpha_pows)
                        .expect("left eval")
            });
    let right =
        tensor
            .right
            .iter()
            .zip(v_weights.iter())
            .fold(F::zero(), |acc, (challenge, &weight)| {
                acc + weight
                    * challenge
                        .eval_at_pows::<F, TD>(alpha_pows)
                        .expect("right eval")
            });
    left * right
}

fn bench_tensor_aggregate_case<const TD: usize>(
    c: &mut Criterion,
    left_len: usize,
    right_len: usize,
) {
    let (tensor, u_weights, v_weights, alpha_pows, alpha_pow_d_plus_one) =
        tensor_bench_fixture::<TD>(left_len, right_len);
    let mut group = c.benchmark_group(format!(
        "tensor_challenge_aggregate_d{TD}_{left_len}x{right_len}"
    ));
    group.throughput(Throughput::Elements(
        (tensor.left_len * tensor.right_len) as u64,
    ));

    group.bench_function("expanded_exact", |b| {
        b.iter(|| {
            black_box(expanded_tensor_weighted_sum::<TD>(
                black_box(&tensor),
                black_box(&u_weights),
                black_box(&v_weights),
                black_box(&alpha_pows),
            ))
        });
    });
    group.bench_function("aggregate_exact", |b| {
        b.iter(|| {
            black_box(
                tensor
                    .eval_factored_aggregate_at_pows::<F, TD>(
                        0,
                        black_box(&u_weights),
                        black_box(&v_weights),
                        black_box(&alpha_pows),
                        black_box(alpha_pow_d_plus_one),
                    )
                    .expect("aggregate eval"),
            )
        });
    });
    group.bench_function("product_only_diagnostic", |b| {
        b.iter(|| {
            black_box(product_only_weighted_sum::<TD>(
                black_box(&tensor),
                black_box(&u_weights),
                black_box(&v_weights),
                black_box(&alpha_pows),
            ))
        });
    });
    group.finish();
}

fn bench_tensor_aggregate(c: &mut Criterion) {
    bench_tensor_aggregate_case::<64>(c, 64, 64);
    bench_tensor_aggregate_case::<128>(c, 64, 64);
}

criterion_group!(sparse_challenge, bench_batch, bench_tensor_aggregate);
criterion_main!(sparse_challenge);
