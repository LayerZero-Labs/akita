#![allow(missing_docs)]

use ark_bn254::Fr as BN254Fr;
use ark_ff::{AdditiveGroup, Field};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use hachi_pcs::algebra::fields::fp128::{Prime128M18M0, Prime128M54P0};
use hachi_pcs::algebra::{HasPacking, PackedField, PackedValue, Prime128M13M4P0, Prime128M8M4M1M0};
use hachi_pcs::algebra::{
    Pow2Offset24Field, Pow2Offset30Field, Pow2Offset31Field, Pow2Offset32Field, Pow2Offset40Field,
    Pow2Offset64Field,
};
use hachi_pcs::{CanonicalField, FieldCore, FieldSampling, Invertible};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::env;

fn rand_u128<R: RngCore>(rng: &mut R) -> u128 {
    let lo = rng.next_u64() as u128;
    let hi = rng.next_u64() as u128;
    lo | (hi << 64)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
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
    let packed_streams = env_usize("HACHI_BENCH_PACKED_STREAMS", 8);
    let latency_iters = env_usize("HACHI_BENCH_LATENCY_ITERS", 4096);
    let throughput_iters = env_usize("HACHI_BENCH_THROUGHPUT_ITERS", 256);
    let stream_iters = env_usize("HACHI_BENCH_STREAM_ITERS", 2048);
    let mix_iters = env_usize("HACHI_BENCH_MIX_ITERS", 256);
    let mix_muls = env_usize("HACHI_BENCH_MIX_MULS", 3);
    let mix_adds = env_usize("HACHI_BENCH_MIX_ADDS", 1);
    let mix_subs = env_usize("HACHI_BENCH_MIX_SUBS", 1);

    assert!(packed_streams > 0, "HACHI_BENCH_PACKED_STREAMS must be > 0");
    assert!(latency_iters > 0, "HACHI_BENCH_LATENCY_ITERS must be > 0");
    assert!(
        throughput_iters > 0,
        "HACHI_BENCH_THROUGHPUT_ITERS must be > 0"
    );
    assert!(stream_iters > 0, "HACHI_BENCH_STREAM_ITERS must be > 0");
    assert!(mix_iters > 0, "HACHI_BENCH_MIX_ITERS must be > 0");

    let muls_per_stream = throughput_iters + 1;
    let mix_ops = mix_muls + mix_adds + mix_subs;
    assert!(mix_ops > 0, "at least one mix operation must be enabled");

    let backend = if cfg!(all(target_arch = "aarch64", target_feature = "neon")) {
        "aarch64_neon"
    } else {
        "scalar_fallback"
    };
    let mut group = c.benchmark_group(format!("field_packed_backend/{backend}/w{}", PF::WIDTH));

    let mut rng = StdRng::seed_from_u64(0xd00d_f00d_1122_3344);
    let scalar_stream_len = PF::WIDTH * stream_iters;
    let lhs: Vec<F> = (0..scalar_stream_len)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let rhs: Vec<F> = (0..scalar_stream_len)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let packed_lhs: Vec<PF> = PF::pack_slice(&lhs);
    let packed_rhs: Vec<PF> = PF::pack_slice(&rhs);
    let scalar_latency_inputs: Vec<F> = (0..latency_iters)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let packed_latency_inputs: Vec<PF> = (0..latency_iters)
        .map(|_| PF::from_fn(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng))))
        .collect();

    let scalar_streams = packed_streams * PF::WIDTH;
    let scalar_lanes: Vec<(F, F)> = (0..scalar_streams)
        .map(|_| {
            (
                F::from_canonical_u128_reduced(rand_u128(&mut rng)),
                F::from_canonical_u128_reduced(rand_u128(&mut rng)),
            )
        })
        .collect();
    let packed_lanes: Vec<(PF, PF)> = (0..packed_streams)
        .map(|_| {
            (
                PF::from_fn(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng))),
                PF::from_fn(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng))),
            )
        })
        .collect();

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

    group.throughput(Throughput::Elements(latency_iters as u64));
    group.bench_function("scalar_mul_latency_chain", |b| {
        b.iter(|| {
            let mut acc = F::one();
            for x in scalar_latency_inputs.iter() {
                acc = acc * *x;
            }
            black_box(acc)
        })
    });

    group.throughput(Throughput::Elements((latency_iters * PF::WIDTH) as u64));
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
        (scalar_streams * muls_per_stream) as u64,
    ));
    group.bench_function("scalar_mul_throughput_8way", |b| {
        b.iter(|| {
            let lanes = black_box(&scalar_lanes);
            let mut acc: Vec<F> = lanes.iter().map(|(a, b)| *a * *b).collect();
            for _ in 0..throughput_iters {
                for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                    *acc_i = *acc_i * lane.0;
                }
            }
            black_box(acc[0])
        })
    });

    group.throughput(Throughput::Elements(
        (packed_streams * muls_per_stream * PF::WIDTH) as u64,
    ));
    group.bench_function("packed_mul_throughput_8way", |b| {
        b.iter(|| {
            let lanes = black_box(&packed_lanes);
            let mut acc: Vec<PF> = lanes.iter().map(|(a, b)| *a * *b).collect();
            for _ in 0..throughput_iters {
                for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                    *acc_i = *acc_i * lane.0;
                }
            }
            black_box(acc[0].extract(0))
        })
    });

    group.throughput(Throughput::Elements(
        (scalar_streams * mix_iters * mix_ops) as u64,
    ));
    group.bench_function("scalar_mix_sumcheck_like", |b| {
        b.iter(|| {
            let lanes = black_box(&scalar_lanes);
            let mut acc: Vec<F> = lanes.iter().map(|(a, b)| *a + *b).collect();
            for _ in 0..mix_iters {
                for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                    let (x, y) = *lane;
                    for _ in 0..mix_muls {
                        *acc_i = *acc_i * x;
                    }
                    for _ in 0..mix_adds {
                        *acc_i = *acc_i + y;
                    }
                    for _ in 0..mix_subs {
                        *acc_i = *acc_i - x;
                    }
                }
            }
            black_box(acc[0])
        })
    });

    group.throughput(Throughput::Elements(
        (packed_streams * PF::WIDTH * mix_iters * mix_ops) as u64,
    ));
    group.bench_function("packed_mix_sumcheck_like", |b| {
        b.iter(|| {
            let lanes = black_box(&packed_lanes);
            let mut acc: Vec<PF> = lanes.iter().map(|(a, b)| *a + *b).collect();
            for _ in 0..mix_iters {
                for (acc_i, lane) in acc.iter_mut().zip(lanes.iter()) {
                    let (x, y) = *lane;
                    for _ in 0..mix_muls {
                        *acc_i = *acc_i * x;
                    }
                    for _ in 0..mix_adds {
                        *acc_i = *acc_i + y;
                    }
                    for _ in 0..mix_subs {
                        *acc_i = *acc_i - x;
                    }
                }
            }
            black_box(acc[0].extract(0))
        })
    });

    group.finish();
}

fn bench_fp32_fp64_mul(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0x3264_3264);
    let n = 2048;

    let inputs_24: Vec<Pow2Offset24Field> =
        (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let inputs_30: Vec<Pow2Offset30Field> =
        (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let inputs_31: Vec<Pow2Offset31Field> =
        (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let inputs_32: Vec<Pow2Offset32Field> =
        (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let inputs_40: Vec<Pow2Offset40Field> =
        (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let inputs_64: Vec<Pow2Offset64Field> =
        (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();

    let mut group = c.benchmark_group("fp32_fp64_mul");

    macro_rules! chain_bench {
        ($name:expr, $ty:ty, $inputs:expr) => {
            group.bench_function(concat!($name, "_mul_chain_2048"), |b| {
                b.iter(|| {
                    let mut acc = <$ty>::one();
                    for x in $inputs.iter() {
                        acc = acc * *x;
                    }
                    black_box(acc)
                })
            });
            group.bench_function(concat!($name, "_mul_add_chain_2048"), |b| {
                b.iter(|| {
                    let mut acc = <$ty>::one();
                    for x in $inputs.iter() {
                        acc = acc * *x + acc;
                    }
                    black_box(acc)
                })
            });
        };
    }

    chain_bench!("fp32_2pow24m3", Pow2Offset24Field, inputs_24);
    chain_bench!("fp32_2pow30m35", Pow2Offset30Field, inputs_30);
    chain_bench!("fp32_2pow31m19", Pow2Offset31Field, inputs_31);
    chain_bench!("fp32_2pow32m99", Pow2Offset32Field, inputs_32);
    chain_bench!("fp64_2pow40m195", Pow2Offset40Field, inputs_40);
    chain_bench!("fp64_2pow64m59", Pow2Offset64Field, inputs_64);

    group.finish();
}

fn bench_widening_ops(c: &mut Criterion) {
    type F = Prime128M8M4M1M0;

    let mut rng = StdRng::seed_from_u64(0x01de_be0c_0001);
    let a = F::from_canonical_u128_reduced(rand_u128(&mut rng));
    let b = F::from_canonical_u128_reduced(rand_u128(&mut rng));
    let b_u64 = rng.next_u64();

    let mut group = c.benchmark_group("widening_ops");

    // -- Single-op isolation: widening only (no reduce) ----------------------
    group.bench_function("mul_wide_u64_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide_u64(black_box(b_u64))))
    });

    group.bench_function("mul_wide_only", |bench| {
        bench.iter(|| black_box(black_box(a).mul_wide(black_box(b))))
    });

    // -- Baseline: full mul+reduce (the current path) ------------------------
    group.bench_function("full_mul_u64_reduce", |bench| {
        bench.iter(|| black_box(black_box(a) * F::from_u64(black_box(b_u64))))
    });

    group.bench_function("full_mul_reduce", |bench| {
        bench.iter(|| black_box(black_box(a) * black_box(b)))
    });

    // -- solinas_reduce isolation per limb count -----------------------------
    let wide3 = a.mul_wide_u64(b_u64);
    let wide4 = a.mul_wide(b);
    let wide5 = {
        let mut l = [0u64; 5];
        l[..3].copy_from_slice(&wide3);
        l[4] = rng.next_u64() & 0xFF;
        l
    };

    group.bench_function("solinas_reduce_3_limbs", |bench| {
        bench.iter(|| black_box(F::solinas_reduce(black_box(&wide3))))
    });

    group.bench_function("solinas_reduce_4_limbs", |bench| {
        bench.iter(|| black_box(F::solinas_reduce(black_box(&wide4))))
    });

    group.bench_function("solinas_reduce_5_limbs", |bench| {
        bench.iter(|| black_box(F::solinas_reduce(black_box(&wide5))))
    });

    // -- Round-trip: widen + reduce ------------------------------------------
    group.bench_function("mul_wide_u64_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let y = black_box(b_u64);
            black_box(F::solinas_reduce(&x.mul_wide_u64(y)))
        })
    });

    group.bench_function("mul_wide_roundtrip", |bench| {
        bench.iter(|| {
            let x = black_box(a);
            let y = black_box(b);
            black_box(F::solinas_reduce(&x.mul_wide(y)))
        })
    });

    group.finish();
}

fn bench_accumulator_pattern(c: &mut Criterion) {
    type F = Prime128M8M4M1M0;

    let mut rng = StdRng::seed_from_u64(0xacc0_1a70_0002);
    let inputs_a: Vec<F> = (0..256)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let inputs_b_u64: Vec<u64> = (0..256).map(|_| rng.next_u64()).collect();
    let inputs_b_f: Vec<F> = (0..256)
        .map(|_| F::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let mut group = c.benchmark_group("accumulator_pattern");

    for &n in &[16, 64, 256] {
        group.bench_function(format!("eager_mul_u64_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_u64[..n]);
                let mut acc = F::zero();
                for i in 0..n {
                    acc = acc + a_s[i] * F::from_u64(b_s[i]);
                }
                black_box(acc)
            })
        });

        group.bench_function(format!("widening_accum_u64_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_u64[..n]);
                let mut acc = [0u64; 5];
                for i in 0..n {
                    let wide = a_s[i].mul_wide_u64(b_s[i]);
                    let mut carry: u64 = 0;
                    for j in 0..3 {
                        let sum = acc[j] as u128 + wide[j] as u128 + carry as u128;
                        acc[j] = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                    for j in 3..5 {
                        let sum = acc[j] as u128 + carry as u128;
                        acc[j] = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                }
                black_box(F::solinas_reduce(&acc))
            })
        });

        group.bench_function(format!("eager_mul_full_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_f[..n]);
                let mut acc = F::zero();
                for i in 0..n {
                    acc = acc + a_s[i] * b_s[i];
                }
                black_box(acc)
            })
        });

        group.bench_function(format!("widening_accum_full_{n}"), |bench| {
            bench.iter(|| {
                let a_s = black_box(&inputs_a[..n]);
                let b_s = black_box(&inputs_b_f[..n]);
                let mut acc = [0u64; 6];
                for i in 0..n {
                    let wide = a_s[i].mul_wide(b_s[i]);
                    let mut carry: u64 = 0;
                    for j in 0..4 {
                        let sum = acc[j] as u128 + wide[j] as u128 + carry as u128;
                        acc[j] = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                    for j in 4..6 {
                        let sum = acc[j] as u128 + carry as u128;
                        acc[j] = sum as u64;
                        carry = (sum >> 64) as u64;
                    }
                }
                black_box(F::solinas_reduce(&acc))
            })
        });
    }

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let n = 4096u64;
    let mut rng = StdRng::seed_from_u64(0xdead_cafe);

    let a24: Vec<Pow2Offset24Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let b24: Vec<Pow2Offset24Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let a30: Vec<Pow2Offset30Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let b30: Vec<Pow2Offset30Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let a31: Vec<Pow2Offset31Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let b31: Vec<Pow2Offset31Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let a32: Vec<Pow2Offset32Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let b32: Vec<Pow2Offset32Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let a40: Vec<Pow2Offset40Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let b40: Vec<Pow2Offset40Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let a64: Vec<Pow2Offset64Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let b64: Vec<Pow2Offset64Field> = (0..n).map(|_| FieldSampling::sample(&mut rng)).collect();
    let a128: Vec<Prime128M8M4M1M0> = (0..n)
        .map(|_| Prime128M8M4M1M0::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();
    let b128: Vec<Prime128M8M4M1M0> = (0..n)
        .map(|_| Prime128M8M4M1M0::from_canonical_u128_reduced(rand_u128(&mut rng)))
        .collect();

    let mut out24 = vec![Pow2Offset24Field::zero(); n as usize];
    let mut out30 = vec![Pow2Offset30Field::zero(); n as usize];
    let mut out31 = vec![Pow2Offset31Field::zero(); n as usize];
    let mut out32 = vec![Pow2Offset32Field::zero(); n as usize];
    let mut out40 = vec![Pow2Offset40Field::zero(); n as usize];
    let mut out64 = vec![Pow2Offset64Field::zero(); n as usize];
    let mut out128 = vec![Prime128M8M4M1M0::zero(); n as usize];

    let mut group = c.benchmark_group("throughput");
    group.throughput(Throughput::Elements(n));

    macro_rules! bench_op {
        ($name:expr, $a:expr, $b:expr, $out:expr, $op:tt) => {
            group.bench_function($name, |bench| {
                bench.iter(|| {
                    let a = black_box(&$a);
                    let b = black_box(&$b);
                    let out = &mut $out;
                    for i in 0..n as usize {
                        out[i] = a[i] $op b[i];
                    }
                })
            });
        };
    }

    bench_op!("fp32_24b_mul", a24, b24, out24, *);
    bench_op!("fp32_24b_add", a24, b24, out24, +);
    bench_op!("fp32_30b_mul", a30, b30, out30, *);
    bench_op!("fp32_30b_add", a30, b30, out30, +);
    bench_op!("fp32_31b_mul", a31, b31, out31, *);
    bench_op!("fp32_31b_add", a31, b31, out31, +);
    bench_op!("fp32_32b_mul", a32, b32, out32, *);
    bench_op!("fp32_32b_add", a32, b32, out32, +);
    bench_op!("fp64_40b_mul", a40, b40, out40, *);
    bench_op!("fp64_40b_add", a40, b40, out40, +);
    bench_op!("fp64_64b_mul", a64, b64, out64, *);
    bench_op!("fp64_64b_add", a64, b64, out64, +);
    bench_op!("fp128_mul", a128, b128, out128, *);
    bench_op!("fp128_add", a128, b128, out128, +);

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
    bench_bn254,
    bench_fp32_fp64_mul,
    bench_widening_ops,
    bench_accumulator_pattern,
    bench_throughput
);
criterion_main!(field_arith);
