#![allow(missing_docs)]

use akita_algebra::CyclotomicRing;
use akita_cpu_backend::benchmark_support::fused_split_eq_quotients_prover_bounds;
use akita_types::layout::FlatMatrix;
use akita_types::{prepare_ntt_cache, NttCacheMode};
use criterion::{criterion_group, criterion_main, Criterion};
use jolt_field::{CanonicalEncoding, Prime128Offset275};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;

type F = Prime128Offset275;
const D: usize = 64;
const LOG_BASIS_OUTER: u32 = 4;

/// Root B quotient rows of the nv36 one-hot fp128 profile: each outer slice
/// multiplies two cyclic B rows by one `t_hat` digit vector.
fn bench_b_cyclic_rows(c: &mut Criterion) {
    let (n_b, width) = (2, 44_032);
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let entries: Vec<CyclotomicRing<F, D>> = (0..n_b * width)
        .map(|_| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|_| {
                F::from_u128_reduced(rng.gen::<u128>())
            }))
        })
        .collect();
    let flat = FlatMatrix::from_ring_slice(&entries);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(1, n_b * width).expect("matrix view"),
        NttCacheMode::BothTransforms,
    )
    .expect("Q128 D=64 cache");
    let half = 1i8 << (LOG_BASIS_OUTER - 1);
    let t_hat: Vec<[i8; D]> = (0..width)
        .map(|_| std::array::from_fn(|_| rng.gen_range(-half..half)))
        .collect();

    let mut group = c.benchmark_group("quotient_kernels");
    group.sample_size(20);
    group.bench_function("b_cyclic_rows_q128_d64_nb2_w44032", |b| {
        b.iter(|| {
            black_box(
                fused_split_eq_quotients_prover_bounds::<F, D>(
                    &cache,
                    &cache,
                    n_b,
                    0,
                    black_box(&t_hat),
                    &[],
                    0,
                    LOG_BASIS_OUTER,
                )
                .expect("B rows"),
            )
        })
    });
    group.finish();
}

criterion_group!(quotient_kernels, bench_b_cyclic_rows);
criterion_main!(quotient_kernels);
