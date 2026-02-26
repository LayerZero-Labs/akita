#![allow(missing_docs)]

use ark_bn254::Fr as BN254Fr;
use ark_ff::{AdditiveGroup, Field};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hachi_pcs::algebra::{Prime128M13M4P0, Prime128M8M4M1M0};
use hachi_pcs::{CanonicalField, FieldCore, Invertible};
use rand::{rngs::StdRng, RngCore, SeedableRng};

fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    lo | (hi << 64)
}

fn bench_mul(c: &mut Criterion) {
    type F13 = Prime128M13M4P0;
    type F275 = Prime128M8M4M1M0;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let inputs_u128: Vec<u128> = (0..2048).map(|_| rand_u128(&mut rng)).collect();

    let inputs_f13: Vec<F13> = inputs_u128
        .iter()
        .copied()
        .map(F13::from_canonical_u128_reduced)
        .collect();

    let inputs_f275: Vec<F275> = inputs_u128
        .iter()
        .copied()
        .map(F275::from_canonical_u128_reduced)
        .collect();

    let mut group = c.benchmark_group("field_mul");

    group.bench_function("fp128_prime128m13m4p0", |b| {
        b.iter(|| {
            let mut acc = F13::one();
            for x in inputs_f13.iter() {
                acc = acc * *x + acc;
            }
            black_box(acc)
        })
    });

    group.bench_function("fp128_prime128m8m4m1m0", |b| {
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

fn bench_mul_only(c: &mut Criterion) {
    type F13 = Prime128M13M4P0;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let inputs: Vec<F13> = (0..2048)
        .map(|_| F13::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let mut group = c.benchmark_group("field_mul_only");

    group.bench_function("mul_chain_2048", |b| {
        b.iter(|| {
            let mut acc = F13::one();
            for x in inputs.iter() {
                acc = acc * *x;
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_parallel_1024", |b| {
        b.iter(|| {
            let mut sum = F13::zero();
            for pair in inputs.chunks_exact(2) {
                sum = sum + pair[0] * pair[1];
            }
            black_box(sum)
        })
    });

    group.finish();
}

fn bench_sqr(c: &mut Criterion) {
    type F13 = Prime128M13M4P0;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let start = F13::from_canonical_u128_reduced(rand_u128(&mut rng));

    let mut group = c.benchmark_group("field_sqr");

    group.bench_function("sqr_chain_2048", |b| {
        b.iter(|| {
            let mut acc = start;
            for _ in 0..2048 {
                acc = acc.square();
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_self_chain_2048", |b| {
        b.iter(|| {
            let mut acc = start;
            for _ in 0..2048 {
                acc = acc * acc;
            }
            black_box(acc)
        })
    });

    group.finish();
}

fn bench_inv(c: &mut Criterion) {
    type F13 = Prime128M13M4P0;

    let mut rng = StdRng::seed_from_u64(0x1a2b_3c4d_5e6f_7788);
    let inputs: Vec<F13> = (0..256)
        .map(|_| F13::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    c.bench_function("fp128_inv_or_zero_prime128m13m4p0", |b| {
        b.iter(|| {
            let mut acc = F13::one();
            for x in inputs.iter() {
                acc = acc * x.inv_or_zero();
            }
            black_box(acc)
        })
    });
}

fn bench_bn254(c: &mut Criterion) {
    use ark_ff::UniformRand;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let inputs: Vec<BN254Fr> = (0..2048).map(|_| BN254Fr::rand(&mut rng)).collect();

    let mut group = c.benchmark_group("bn254_fr");

    group.bench_function("mul_add_chain_2048", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ONE;
            for x in inputs.iter() {
                acc = acc * x + acc;
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_chain_2048", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ONE;
            for x in inputs.iter() {
                acc *= x;
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_parallel_1024", |b| {
        b.iter(|| {
            let mut sum = BN254Fr::ZERO;
            for pair in inputs.chunks_exact(2) {
                sum += pair[0] * pair[1];
            }
            black_box(sum)
        })
    });

    group.bench_function("sqr_chain_2048", |b| {
        b.iter(|| {
            let mut acc = inputs[0];
            for _ in 0..2048 {
                acc.square_in_place();
            }
            black_box(acc)
        })
    });

    group.bench_function("inv_256", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ONE;
            for x in inputs[..256].iter() {
                acc *= x.inverse().unwrap_or(BN254Fr::ZERO);
            }
            black_box(acc)
        })
    });

    group.finish();
}

criterion_group!(
    field_arith,
    bench_mul,
    bench_mul_only,
    bench_sqr,
    bench_inv,
    bench_bn254
);
criterion_main!(field_arith);
