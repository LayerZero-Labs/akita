use std::hint::black_box;

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt};
use akita_algebra::ntt::NttTwiddles;
use akita_algebra::tables::{
    q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q32_PRIMES, Q64_NUM_PRIMES,
    Q64_PRIMES,
};
use akita_algebra::{
    CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut, MontCoeff, NttKernelPlan, NttPrime,
};
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

/// Number of prepared inputs cycled through by the three-prime forward
/// scenarios, so each iteration transforms a different input.
const FORWARD_INPUT_POOL: usize = 128;

/// Three-prime (q64) forward negacyclic NTT from signed i8 digits and from
/// general Montgomery limbs in `(-p, p)`.
fn bench_q64_forward_inputs<const D: usize>(c: &mut Criterion) {
    let params = CrtNttParamSet::<_, Q64_NUM_PRIMES, D>::new(Q64_PRIMES);
    let lut = DigitMontLut::new_with_digit_bound(&params, 128);
    let mut state = 0x39128fa024ce7u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let signed: Vec<[i8; D]> = (0..FORWARD_INPUT_POOL)
        .map(|_| std::array::from_fn(|_| next() as i8))
        .collect();
    let general: Vec<[[MontCoeff<i32>; D]; Q64_NUM_PRIMES]> = (0..FORWARD_INPUT_POOL)
        .map(|_| {
            std::array::from_fn(|k| {
                let p = i64::from(params.primes[k].p);
                std::array::from_fn(|_| {
                    MontCoeff::from_raw(((next() % (2 * p - 1) as u64) as i64 - p + 1) as i32)
                })
            })
        })
        .collect();
    let mut group = c.benchmark_group("q64_forward_ntt_three_primes");

    group.bench_with_input(BenchmarkId::new("signed_i8", D), &D, |b, _| {
        let mut index = 0usize;
        b.iter(|| {
            index = (index + 1) % FORWARD_INPUT_POOL;
            black_box(CyclotomicCrtNtt::from_i8_with_lut(
                black_box(&signed[index]),
                &params,
                &lut,
            ))
        })
    });
    group.bench_with_input(BenchmarkId::new("general_montgomery", D), &D, |b, _| {
        let mut index = 0usize;
        b.iter_batched(
            || {
                index = (index + 1) % FORWARD_INPUT_POOL;
                general[index]
            },
            |mut limbs| {
                for (k, limb) in limbs.iter_mut().enumerate() {
                    forward_ntt(
                        limb,
                        params.primes[k],
                        &params.twiddles[k],
                        params.kernel_plan(),
                    );
                }
                black_box(limbs)
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

    bench_q64_forward_inputs::<128>(c);
    bench_q64_forward_inputs::<256>(c);
    bench_q64_forward_inputs::<512>(c);
    bench_q64_forward_inputs::<1024>(c);

    bench_i16_tail::<64>(c);
    bench_i16_tail::<128>(c);
    bench_i16_tail::<256>(c);
    bench_i16_tail::<512>(c);
    bench_i16_tail::<1024>(c);
}

criterion_group!(crt_ntt_ops, benches);
criterion_main!(crt_ntt_ops);
