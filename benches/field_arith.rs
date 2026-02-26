#![allow(missing_docs)]

use ark_bn254::Fr as BN254Fr;
use ark_ff::{AdditiveGroup, Field};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use hachi_pcs::algebra::fields::fp128::{Prime128M18M0, Prime128M54P0};
use hachi_pcs::algebra::{HasPacking, PackedField, PackedValue, Prime128M13M4P0, Prime128M8M4M1M0};
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

fn bench_packed_fp128_backend(c: &mut Criterion) {
    type F = Prime128M13M4P0;
    type PF = <F as HasPacking>::Packing;
    const PACKED_STREAMS: usize = 8;
    const LATENCY_ITERS: usize = 4096;
    const THROUGHPUT_ITERS: usize = 256;
    const STREAM_ITERS: usize = 2048;
    const MULS_PER_STREAM: usize = THROUGHPUT_ITERS + 1;

    let backend = if cfg!(all(target_arch = "aarch64", target_feature = "neon")) {
        "aarch64_neon"
    } else {
        "scalar_fallback"
    };
    let mut group = c.benchmark_group(format!("field_packed_backend/{backend}/w{}", PF::WIDTH));

    let mut rng = StdRng::seed_from_u64(0xd00d_f00d_1122_3344);
    let scalar_stream_len = PF::WIDTH * STREAM_ITERS;
    let lhs: Vec<F> = (0..scalar_stream_len)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let rhs: Vec<F> = (0..scalar_stream_len)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let packed_lhs: Vec<PF> = PF::pack_slice(&lhs);
    let packed_rhs: Vec<PF> = PF::pack_slice(&rhs);
    let scalar_latency_inputs: [F; LATENCY_ITERS] =
        std::array::from_fn(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)));
    let packed_latency_inputs: [PF; LATENCY_ITERS] = std::array::from_fn(|_| {
        PF::from_fn(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
    });

    const fn scalar_stream_count(width: usize) -> usize {
        PACKED_STREAMS * width
    }
    let scalar_streams = scalar_stream_count(PF::WIDTH);
    let scalar_lanes: Vec<(F, F)> = (0..scalar_streams)
        .map(|_| {
            (
                F::from_canonical_u128_reduced(rand_u128(&mut rng)),
                F::from_canonical_u128_reduced(rand_u128(&mut rng)),
            )
        })
        .collect();
    let packed_lanes: [(PF, PF); PACKED_STREAMS] = std::array::from_fn(|_| {
        (
            PF::from_fn(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng))),
            PF::from_fn(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng))),
        )
    });

    group.throughput(Throughput::Elements(scalar_stream_len as u64));
    group.bench_function("scalar_add_stream", |b| {
        let mut out = lhs.clone();
        b.iter(|| {
            for (dst, src) in out.iter_mut().zip(rhs.iter()) {
                *dst = *dst + *src;
            }
            black_box(out[0])
        })
    });

    group.throughput(Throughput::Elements(scalar_stream_len as u64));
    group.bench_function("packed_add_stream", |b| {
        let mut out = packed_lhs.clone();
        b.iter(|| {
            for (dst, src) in out.iter_mut().zip(packed_rhs.iter()) {
                *dst = *dst + *src;
            }
            black_box(out[0].extract(0))
        })
    });

    group.throughput(Throughput::Elements(LATENCY_ITERS as u64));
    group.bench_function("scalar_mul_latency_chain", |b| {
        b.iter(|| {
            let mut acc = F::one();
            for x in scalar_latency_inputs.iter() {
                acc = acc * *x;
            }
            black_box(acc)
        })
    });

    group.throughput(Throughput::Elements((LATENCY_ITERS * PF::WIDTH) as u64));
    group.bench_function("packed_mul_latency_chain", |b| {
        b.iter(|| {
            let mut acc = PF::broadcast(F::one());
            for x in packed_latency_inputs.iter() {
                acc = acc * *x;
            }
            black_box(acc.extract(0))
        })
    });

    group.throughput(Throughput::Elements(
        (scalar_streams * MULS_PER_STREAM) as u64,
    ));
    group.bench_function("scalar_mul_throughput_8way", |b| {
        b.iter(|| {
            let lanes = black_box(&scalar_lanes);
            let mut acc: Vec<F> = lanes.iter().map(|(a, b)| *a * *b).collect();
            for _ in 0..THROUGHPUT_ITERS {
                for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                    *acc_i = *acc_i * lane.0;
                }
            }
            black_box(acc[0])
        })
    });

    group.throughput(Throughput::Elements(
        (PACKED_STREAMS * MULS_PER_STREAM * PF::WIDTH) as u64,
    ));
    group.bench_function("packed_mul_throughput_8way", |b| {
        b.iter(|| {
            let lanes = black_box(&packed_lanes);
            let mut acc: [PF; PACKED_STREAMS] = std::array::from_fn(|i| lanes[i].0 * lanes[i].1);
            for _ in 0..THROUGHPUT_ITERS {
                for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                    *acc_i = *acc_i * lane.0;
                }
            }
            black_box(acc[0].extract(0))
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
    bench_packed_fp128_backend,
    bench_bn254
);
criterion_main!(field_arith);
