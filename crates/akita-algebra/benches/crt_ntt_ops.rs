use std::hint::black_box;

use akita_algebra::tables::{q128_primes, Q128_NUM_PRIMES};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, MontCoeff};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn input<const D: usize>(salt: i32) -> CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, D> {
    let primes = q128_primes();
    CyclotomicCrtNtt {
        limbs: std::array::from_fn(|k| {
            std::array::from_fn(|i| {
                let p = primes[k].p;
                let x = ((i as i64 * 0x1f123bb5 + i64::from(salt)) % i64::from(p)) as i32;
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
        b.iter(|| black_box(lhs.add_reduced_with_params(black_box(&rhs), black_box(&params))))
    });
    group.bench_with_input(BenchmarkId::new("sub", D), &D, |b, _| {
        b.iter(|| black_box(lhs.sub_reduced_with_params(black_box(&rhs), black_box(&params))))
    });
    group.bench_with_input(BenchmarkId::new("neg", D), &D, |b, _| {
        b.iter(|| black_box(lhs.neg_reduced_with_params(black_box(&params))))
    });
    group.bench_with_input(BenchmarkId::new("mul", D), &D, |b, _| {
        b.iter(|| black_box(lhs.pointwise_mul_with_params(black_box(&rhs), black_box(&params))))
    });
    group.finish();
}

fn benches(c: &mut Criterion) {
    bench_d::<32>(c);
    bench_d::<64>(c);
    bench_d::<128>(c);
    bench_d::<256>(c);
    bench_d::<512>(c);
}

criterion_group!(crt_ntt_ops, benches);
criterion_main!(crt_ntt_ops);
