use std::hint::black_box;

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt};
use akita_algebra::ntt::NttTwiddles;
use akita_algebra::tables::{q128_primes, Q128_NUM_PRIMES};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, MontCoeff, NttKernelPlan};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn input<const D: usize>(seed: i32) -> CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D> {
    let primes = q128_primes();
    CyclotomicCrtNtt {
        limbs: std::array::from_fn(|k| {
            std::array::from_fn(|i| {
                let p = primes[k].p;
                let x = ((i as i64 * 0x1f123bb5 + i64::from(seed)) % i64::from(p)) as i32;
                MontCoeff::from_raw(if i & 1 == 0 { x } else { -x })
            })
        }),
    }
}

fn bench_d<const D: usize>(c: &mut Criterion) {
    let params = CrtNttParamSet::new(q128_primes());
    let lhs = input::<D>(17);
    let rhs = input::<D>(93);
    let mut group = c.benchmark_group("q128_crt_ntt_ops");

    group.bench_with_input(BenchmarkId::new("add", D), &D, |b, _| {
        b.iter(|| black_box(lhs.add_reduced(black_box(&rhs), black_box(&params))))
    });
    group.bench_with_input(BenchmarkId::new("sub", D), &D, |b, _| {
        b.iter(|| black_box(lhs.sub_reduced(black_box(&rhs), black_box(&params))))
    });
    group.bench_with_input(BenchmarkId::new("neg", D), &D, |b, _| {
        b.iter(|| black_box(lhs.neg_reduced(black_box(&params))))
    });
    group.bench_with_input(BenchmarkId::new("mul", D), &D, |b, _| {
        b.iter(|| black_box(lhs.pointwise_mul(black_box(&rhs), black_box(&params))))
    });
    group.finish();

    let prime = q128_primes()[0];
    let twiddles = NttTwiddles::compute(prime);
    let coeffs = lhs.limbs[0];
    let mut evals = coeffs;
    let plan = NttKernelPlan::detect::<i32>();
    forward_ntt(&mut evals, prime, &twiddles, plan);
    let mut ntt_group = c.benchmark_group("q128_ntt_i32");

    ntt_group.bench_with_input(BenchmarkId::new("forward", D), &D, |b, _| {
        b.iter_batched(
            || coeffs,
            |mut values| {
                forward_ntt(&mut values, black_box(prime), black_box(&twiddles), plan);
                black_box(values)
            },
            BatchSize::SmallInput,
        )
    });
    ntt_group.bench_with_input(BenchmarkId::new("inverse", D), &D, |b, _| {
        b.iter_batched(
            || evals,
            |mut values| {
                inverse_ntt(&mut values, black_box(prime), black_box(&twiddles), plan);
                black_box(values)
            },
            BatchSize::SmallInput,
        )
    });
    ntt_group.finish();
}

fn benches(c: &mut Criterion) {
    bench_d::<64>(c);
    bench_d::<128>(c);
    bench_d::<256>(c);
    bench_d::<512>(c);
}

criterion_group!(crt_ntt_ops, benches);
criterion_main!(crt_ntt_ops);
