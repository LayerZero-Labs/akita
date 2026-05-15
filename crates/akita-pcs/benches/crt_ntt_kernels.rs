#![allow(missing_docs)]

//! Microbenchmarks for CRT+NTT-domain kernels.
//!
//! The SIMD backend is chosen once per process. Run scalar and vector modes as
//! separate invocations, for example:
//!
//! ```text
//! AKITA_SCALAR_NTT=1 cargo bench -p akita-pcs --bench crt_ntt_kernels
//! AKITA_X86_NTT=avx2 cargo bench -p akita-pcs --bench crt_ntt_kernels
//! RUSTFLAGS="-C target-cpu=native" AKITA_X86_NTT=avx512 \
//!   cargo bench -p akita-pcs --bench crt_ntt_kernels
//! ```
//!
//! Use `AKITA_BENCH_CRT_NTT_REPS=<n>` to change how many tiny kernel calls are
//! grouped into each Criterion iteration.

use std::array::from_fn;
use std::env;

use akita_algebra::ntt::tables::{q64_primes, Q32_NUM_PRIMES, Q32_PRIMES, Q64_NUM_PRIMES};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut, MontCoeff, PrimeWidth};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

const DEFAULT_REPS: usize = 4096;

fn bench_reps() -> usize {
    env::var("AKITA_BENCH_CRT_NTT_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_REPS)
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn sample_crt_ntt<W: PrimeWidth, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
    seed: u64,
) -> CyclotomicCrtNtt<W, K, D> {
    CyclotomicCrtNtt {
        limbs: from_fn(|k| {
            let prime = params.primes[k];
            let modulus = prime.p.to_i64() as u64;
            from_fn(|i| {
                let x = mix(seed ^ ((k as u64) << 32) ^ i as u64);
                prime.from_canonical(W::from_i64((x % modulus) as i64))
            })
        }),
    }
}

fn sample_digits<const D: usize>(seed: u64) -> [i8; D] {
    from_fn(|i| ((mix(seed ^ i as u64) & 63) as i16 - 32) as i8)
}

fn bench_family<W: PrimeWidth, const K: usize, const D: usize>(
    c: &mut Criterion,
    label: &'static str,
    params: CrtNttParamSet<W, K, D>,
) {
    let reps = bench_reps();
    let lhs: Vec<_> = (0..reps)
        .map(|i| sample_crt_ntt(&params, 0x1000 + i as u64))
        .collect();
    let rhs: Vec<_> = (0..reps)
        .map(|i| sample_crt_ntt(&params, 0x2000 + i as u64))
        .collect();
    let addends: Vec<_> = (0..reps)
        .map(|i| sample_crt_ntt(&params, 0x3000 + i as u64))
        .collect();
    let digits: Vec<_> = (0..reps)
        .map(|i| sample_digits::<D>(0x4000 + i as u64))
        .collect();
    let lut = DigitMontLut::new(&params);
    let scratch_zero = [[MontCoeff::from_raw(W::default()); D]; K];

    let mut group = c.benchmark_group(format!("crt_ntt_kernels/{label}"));
    group.throughput(Throughput::Elements((reps * K * D) as u64));

    group.bench_function("pointwise_mul_acc", |b| {
        b.iter_batched(
            CyclotomicCrtNtt::<W, K, D>::zero,
            |mut acc| {
                for i in 0..reps {
                    acc.add_assign_pointwise_mul_with_params(
                        black_box(&lhs[i]),
                        black_box(&rhs[i]),
                        &params,
                    );
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("add_reduce", |b| {
        b.iter_batched(
            CyclotomicCrtNtt::<W, K, D>::zero,
            |mut acc| {
                for addend in &addends {
                    acc.add_assign_reduced_with_params(black_box(addend), &params);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("digit_lut_ntt_mul_acc", |b| {
        b.iter_batched(
            || (CyclotomicCrtNtt::<W, K, D>::zero(), scratch_zero),
            |(mut acc, mut scratch)| {
                for i in 0..reps {
                    acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                        black_box(&lhs[i]),
                        black_box(&digits[i]),
                        &params,
                        &lut,
                        &mut scratch,
                    );
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_crt_ntt_kernels(c: &mut Criterion) {
    bench_family::<i16, Q32_NUM_PRIMES, 64>(c, "q32_d64_k4", CrtNttParamSet::new(Q32_PRIMES));
    bench_family::<i32, Q64_NUM_PRIMES, 32>(c, "q64_d32_k3", CrtNttParamSet::new(q64_primes()));
}

criterion_group!(crt_ntt_kernels, bench_crt_ntt_kernels);
criterion_main!(crt_ntt_kernels);
