#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hachi_pcs::algebra::{Fp128, Prime128M13M4P0, Prime128M8M4M1M0};
use hachi_pcs::{CanonicalField, FieldCore, Invertible};
use rand::{rngs::StdRng, RngCore, SeedableRng};

fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    lo | (hi << 64)
}

fn bench_mul(c: &mut Criterion) {
    const P13: u128 = 0xffffffffffffffffffffffffffffdff1u128;
    const P275: u128 = 0xfffffffffffffffffffffffffffffeedu128;

    type S13 = Prime128M13M4P0;
    type F13 = Fp128<P13>;
    type S275 = Prime128M8M4M1M0;
    type F275 = Fp128<P275>;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let inputs_u128: Vec<u128> = (0..2048).map(|_| rand_u128(&mut rng)).collect();

    let inputs_s13: Vec<S13> = inputs_u128
        .iter()
        .copied()
        .map(S13::from_canonical_u128_reduced)
        .collect();
    let inputs_f13: Vec<F13> = inputs_u128
        .iter()
        .copied()
        .map(F13::from_canonical_u128_reduced)
        .collect();

    let inputs_s275: Vec<S275> = inputs_u128
        .iter()
        .copied()
        .map(S275::from_canonical_u128_reduced)
        .collect();
    let inputs_f275: Vec<F275> = inputs_u128
        .iter()
        .copied()
        .map(F275::from_canonical_u128_reduced)
        .collect();

    let mut group = c.benchmark_group("field_mul");

    group.bench_function("solinas_prime128m13m4p0", |b| {
        b.iter(|| {
            let mut acc = S13::one();
            for x in inputs_s13.iter() {
                acc = acc * *x + acc;
            }
            black_box(acc)
        })
    });

    group.bench_function("fp128_p13", |b| {
        b.iter(|| {
            let mut acc = F13::one();
            for x in inputs_f13.iter() {
                acc = acc * *x + acc;
            }
            black_box(acc)
        })
    });

    group.bench_function("solinas_prime128m8m4m1m0", |b| {
        b.iter(|| {
            let mut acc = S275::one();
            for x in inputs_s275.iter() {
                acc = acc * *x + acc;
            }
            black_box(acc)
        })
    });

    group.bench_function("fp128_p275", |b| {
        b.iter(|| {
            let mut acc = F275::one();
            for x in inputs_f275.iter() {
                acc = acc * *x + acc;
            }
            black_box(acc)
        })
    });

    group.finish();
}

fn bench_inv(c: &mut Criterion) {
    type S13 = Prime128M13M4P0;

    let mut rng = StdRng::seed_from_u64(0x1a2b_3c4d_5e6f_7788);
    let inputs: Vec<S13> = (0..256)
        .map(|_| S13::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    c.bench_function("solinas_inv_or_zero_prime128m13m4p0", |b| {
        b.iter(|| {
            let mut acc = S13::one();
            for x in inputs.iter() {
                acc = acc * x.inv_or_zero();
            }
            black_box(acc)
        })
    });
}

criterion_group!(field_arith, bench_mul, bench_inv);
criterion_main!(field_arith);
