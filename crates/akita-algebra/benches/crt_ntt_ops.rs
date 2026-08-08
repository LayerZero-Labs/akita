use std::hint::black_box;

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt};
use akita_algebra::ntt::NttTwiddles;
use akita_algebra::tables::{
    q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES, Q64_NUM_PRIMES,
    Q64_PRIMES,
};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, MontCoeff, NttKernelPlan, NttPrime};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn input<const K: usize, const D: usize>(
    primes: &[NttPrime<i32>; K],
    seed: i32,
) -> CyclotomicCrtNtt<i32, K, D> {
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

fn bench_profile<const K: usize, const D: usize>(
    c: &mut Criterion,
    profile: &str,
    primes: [NttPrime<i32>; K],
) {
    let params = CrtNttParamSet::new(primes);
    let lhs = input::<K, D>(&primes, 17);
    let rhs = input::<K, D>(&primes, 93);
    let mut group = c.benchmark_group(format!("{profile}_crt_ntt_ops"));

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
    group.bench_with_input(BenchmarkId::new("mac", D), &D, |b, _| {
        b.iter_batched(
            || lhs.clone(),
            |mut accumulator| {
                accumulator.add_assign_pointwise_mul(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&params),
                );
                black_box(accumulator)
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();

    let prime = primes[0];
    let twiddles = NttTwiddles::compute(prime);
    let coeffs = lhs.limbs[0];
    let mut evals = coeffs;
    let plan = NttKernelPlan::detect::<i32>();
    forward_ntt(&mut evals, prime, &twiddles, plan);
    let mut ntt_group = c.benchmark_group(format!("{profile}_ntt_i32"));

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

fn bench_i16_tail<const D: usize>(c: &mut Criterion) {
    let prime = I16_TAIL_PRIME;
    let twiddles = NttTwiddles::compute(prime);
    let coeffs: [MontCoeff<i16>; D] = std::array::from_fn(|i| {
        prime.from_canonical(((i as i64 * 251 + 17) % i64::from(prime.p)) as i16)
    });
    let mut evals = coeffs;
    let plan = NttKernelPlan::detect::<i16>();
    forward_ntt(&mut evals, prime, &twiddles, plan);
    let mut group = c.benchmark_group("i16_tail_ntt");

    group.bench_with_input(BenchmarkId::new("forward", D), &D, |b, _| {
        b.iter_batched(
            || coeffs,
            |mut values| {
                forward_ntt(&mut values, black_box(prime), black_box(&twiddles), plan);
                black_box(values)
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_with_input(BenchmarkId::new("inverse", D), &D, |b, _| {
        b.iter_batched(
            || evals,
            |mut values| {
                inverse_ntt(&mut values, black_box(prime), black_box(&twiddles), plan);
                black_box(values)
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_profile::<Q32_NUM_PRIMES, 64>(c, "q32", Q32_PRIMES);
    bench_profile::<Q32_NUM_PRIMES, 128>(c, "q32", Q32_PRIMES);
    bench_profile::<Q32_NUM_PRIMES, 256>(c, "q32", Q32_PRIMES);
    bench_profile::<Q32_NUM_PRIMES, 512>(c, "q32", Q32_PRIMES);
    bench_profile::<Q32_NUM_PRIMES, 1024>(c, "q32", Q32_PRIMES);

    bench_profile::<Q64_NUM_PRIMES, 64>(c, "q64", Q64_PRIMES);
    bench_profile::<Q64_NUM_PRIMES, 128>(c, "q64", Q64_PRIMES);
    bench_profile::<Q64_NUM_PRIMES, 256>(c, "q64", Q64_PRIMES);
    bench_profile::<Q64_NUM_PRIMES, 512>(c, "q64", Q64_PRIMES);
    bench_profile::<Q64_NUM_PRIMES, 1024>(c, "q64", Q64_PRIMES);

    let q128 = q128_primes();
    bench_profile::<Q128_NUM_PRIMES, 64>(c, "q128", q128);
    bench_profile::<Q128_NUM_PRIMES, 128>(c, "q128", q128);
    bench_profile::<Q128_NUM_PRIMES, 256>(c, "q128", q128);
    bench_profile::<Q128_NUM_PRIMES, 512>(c, "q128", q128);

    bench_i16_tail::<64>(c);
    bench_i16_tail::<128>(c);
    bench_i16_tail::<256>(c);
    bench_i16_tail::<512>(c);
    bench_i16_tail::<1024>(c);
}

criterion_group!(crt_ntt_ops, benches);
criterion_main!(crt_ntt_ops);
