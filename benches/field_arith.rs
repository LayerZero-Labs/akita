#![allow(missing_docs)]

use ark_bn254::Fr as BN254Fr;
use ark_ff::{AdditiveGroup, Field};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hachi_pcs::algebra::fields::fp128::{Prime128M18M0, Prime128M54P0};
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
    type F2p18p1 = Prime128M18M0;
    type F2p54m1 = Prime128M54P0;

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
    let inputs_f2p18p1: Vec<F2p18p1> = inputs_u128
        .iter()
        .copied()
        .map(F2p18p1::from_canonical_u128_reduced)
        .collect();
    let inputs_f2p54m1: Vec<F2p54m1> = inputs_u128
        .iter()
        .copied()
        .map(F2p54m1::from_canonical_u128_reduced)
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

    group.bench_function("fp128_prime128m18m0_shift_special", |b| {
        b.iter(|| {
            let mut acc = F2p18p1::one();
            for x in inputs_f2p18p1.iter() {
                acc = acc * *x + acc;
            }
            black_box(acc)
        })
    });

    group.bench_function("fp128_prime128m54p0_shift_special", |b| {
        b.iter(|| {
            let mut acc = F2p54m1::one();
            for x in inputs_f2p54m1.iter() {
                acc = acc * *x + acc;
            }
            black_box(acc)
        })
    });

    group.finish();
}

fn bench_mul_only(c: &mut Criterion) {
    type F13 = Prime128M13M4P0;
    type F2p18p1 = Prime128M18M0;
    type F2p54m1 = Prime128M54P0;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let inputs_f13: Vec<F13> = (0..2048)
        .map(|_| F13::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let inputs_f2p18p1: Vec<F2p18p1> = (0..2048)
        .map(|_| F2p18p1::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let inputs_f2p54m1: Vec<F2p54m1> = (0..2048)
        .map(|_| F2p54m1::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let mut group = c.benchmark_group("field_mul_only");

    group.bench_function("mul_chain_2048", |b| {
        b.iter(|| {
            let mut acc = F13::one();
            for x in inputs_f13.iter() {
                acc = acc * *x;
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_chain_16384", |b| {
        b.iter(|| {
            let mut acc = F13::one();
            for _ in 0..8 {
                for x in inputs_f13.iter() {
                    acc = acc * *x;
                }
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_parallel_1024", |b| {
        b.iter(|| {
            let mut sum = F13::zero();
            for pair in inputs_f13.chunks_exact(2) {
                sum = sum + pair[0] * pair[1];
            }
            black_box(sum)
        })
    });

    group.bench_function("mul_chain_2048_special_m18m0", |b| {
        b.iter(|| {
            let mut acc = F2p18p1::one();
            for x in inputs_f2p18p1.iter() {
                acc = acc * *x;
            }
            black_box(acc)
        })
    });

    group.bench_function("mul_chain_2048_special_m54p0", |b| {
        b.iter(|| {
            let mut acc = F2p54m1::one();
            for x in inputs_f2p54m1.iter() {
                acc = acc * *x;
            }
            black_box(acc)
        })
    });

    group.finish();
}

fn bench_mul_isolated(c: &mut Criterion) {
    use ark_ff::UniformRand;

    type F13 = Prime128M13M4P0;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let a_fp128 = F13::from_canonical_u128_reduced(rand_u128(&mut rng));
    let b_fp128 = F13::from_canonical_u128_reduced(rand_u128(&mut rng));
    let a_bn254 = BN254Fr::rand(&mut rng);
    let b_bn254 = BN254Fr::rand(&mut rng);

    let mut group = c.benchmark_group("field_mul_isolated");

    group.bench_function("fp128_black_box_only", |b| b.iter(|| black_box(a_fp128)));

    group.bench_function("bn254_black_box_only", |b| b.iter(|| black_box(a_bn254)));

    group.bench_function("fp128_pair_passthrough", |b| {
        b.iter(|| {
            let x = black_box(a_fp128);
            let y = black_box(b_fp128);
            black_box((x, y))
        })
    });

    group.bench_function("bn254_pair_passthrough", |b| {
        b.iter(|| {
            let x = black_box(a_bn254);
            let y = black_box(b_bn254);
            black_box((x, y))
        })
    });

    group.bench_function("fp128_mul_single", |b| {
        b.iter(|| {
            let x = black_box(a_fp128);
            let y = black_box(b_fp128);
            black_box(x * y)
        })
    });

    group.bench_function("bn254_mul_single", |b| {
        b.iter(|| {
            let x = black_box(a_bn254);
            let y = black_box(b_bn254);
            black_box(x * y)
        })
    });

    let lanes_fp128: [(F13, F13); 8] = std::array::from_fn(|_| {
        (
            F13::from_canonical_u128_reduced(rand_u128(&mut rng)),
            F13::from_canonical_u128_reduced(rand_u128(&mut rng)),
        )
    });
    let lanes_bn254: [(BN254Fr, BN254Fr); 8] =
        std::array::from_fn(|_| (BN254Fr::rand(&mut rng), BN254Fr::rand(&mut rng)));

    group.bench_function("fp128_mul_8way_independent", |b| {
        b.iter(|| {
            let lanes = black_box(&lanes_fp128);
            let p0 = lanes[0].0 * lanes[0].1;
            let p1 = lanes[1].0 * lanes[1].1;
            let p2 = lanes[2].0 * lanes[2].1;
            let p3 = lanes[3].0 * lanes[3].1;
            let p4 = lanes[4].0 * lanes[4].1;
            let p5 = lanes[5].0 * lanes[5].1;
            let p6 = lanes[6].0 * lanes[6].1;
            let p7 = lanes[7].0 * lanes[7].1;
            black_box([p0, p1, p2, p3, p4, p5, p6, p7])
        })
    });

    group.bench_function("fp128_8way_passthrough", |b| {
        b.iter(|| {
            let lanes = black_box(&lanes_fp128);
            let p0 = lanes[0].0;
            let p1 = lanes[1].0;
            let p2 = lanes[2].0;
            let p3 = lanes[3].0;
            let p4 = lanes[4].0;
            let p5 = lanes[5].0;
            let p6 = lanes[6].0;
            let p7 = lanes[7].0;
            black_box([p0, p1, p2, p3, p4, p5, p6, p7])
        })
    });

    group.bench_function("bn254_mul_8way_independent", |b| {
        b.iter(|| {
            let lanes = black_box(&lanes_bn254);
            let p0 = lanes[0].0 * lanes[0].1;
            let p1 = lanes[1].0 * lanes[1].1;
            let p2 = lanes[2].0 * lanes[2].1;
            let p3 = lanes[3].0 * lanes[3].1;
            let p4 = lanes[4].0 * lanes[4].1;
            let p5 = lanes[5].0 * lanes[5].1;
            let p6 = lanes[6].0 * lanes[6].1;
            let p7 = lanes[7].0 * lanes[7].1;
            black_box([p0, p1, p2, p3, p4, p5, p6, p7])
        })
    });

    group.bench_function("bn254_8way_passthrough", |b| {
        b.iter(|| {
            let lanes = black_box(&lanes_bn254);
            let p0 = lanes[0].0;
            let p1 = lanes[1].0;
            let p2 = lanes[2].0;
            let p3 = lanes[3].0;
            let p4 = lanes[4].0;
            let p5 = lanes[5].0;
            let p6 = lanes[6].0;
            let p7 = lanes[7].0;
            black_box([p0, p1, p2, p3, p4, p5, p6, p7])
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

    group.bench_function("mul_chain_16384", |b| {
        b.iter(|| {
            let mut acc = BN254Fr::ONE;
            for _ in 0..8 {
                for x in inputs.iter() {
                    acc *= x;
                }
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
    bench_mul_isolated,
    bench_sqr,
    bench_inv,
    bench_bn254
);
criterion_main!(field_arith);
