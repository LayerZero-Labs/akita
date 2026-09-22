use std::hint::black_box;

use akita_algebra::binary::BinaryField162 as F;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[path = "binary162/packed.rs"]
mod packed;

fn inputs(len: usize) -> (Vec<F>, Vec<F>) {
    let mut state = 0x1234_5678_9abc_def0u64;
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let lhs = F::from_words([state, state.rotate_left(23), state & 0x3_ffff_ffff]).unwrap();
            let rhs = F::from_words([
                state.rotate_right(9) ^ index as u64,
                state.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                state.rotate_left(7) & 0x3_ffff_ffff,
            ])
            .unwrap();
            (lhs, rhs)
        })
        .unzip()
}

fn binary162(c: &mut Criterion) {
    let lhs = F::from_words([0x1234_5678_abcd_ef01, u64::MAX, 0x1234_5678]).unwrap();
    let rhs = F::from_words([u64::MAX, 0x9876_5432_10fe_dcba, 0x3_ffff_ffff]).unwrap();

    let mut scalar = c.benchmark_group("binary162_scalar");
    scalar.bench_function("mul", |b| {
        b.iter(|| black_box(black_box(lhs) * black_box(rhs)))
    });
    scalar.bench_function("square", |b| b.iter(|| black_box(black_box(lhs).square())));
    scalar.bench_function("inverse", |b| {
        b.iter(|| black_box(black_box(lhs).inverse()))
    });
    scalar.finish();

    let mut dot = c.benchmark_group("binary162_dot_product");
    for len in [8, 64, 1024] {
        let (lhs, rhs) = inputs(len);
        dot.throughput(Throughput::Elements(len as u64));
        dot.bench_with_input(BenchmarkId::new("deferred", len), &len, |b, _| {
            b.iter(|| black_box(F::dot_product(black_box(&lhs), black_box(&rhs))))
        });
        dot.bench_with_input(BenchmarkId::new("multiply_then_sum", len), &len, |b, _| {
            b.iter(|| {
                let lhs = black_box(&lhs);
                let rhs = black_box(&rhs);
                black_box(
                    lhs.iter()
                        .zip(rhs)
                        .fold(F::ZERO, |sum, (&a, &b)| sum + a * b),
                )
            })
        });
    }
    dot.finish();
}

criterion_group!(benches, binary162, packed::bench);
criterion_main!(benches);
