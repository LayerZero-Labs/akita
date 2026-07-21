#![allow(missing_docs)]

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use akita_algebra::tables::{
    q128_primes, q32_garner, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q32_MODULUS, Q32_NUM_PRIMES,
    Q32_PRIMES,
};
use akita_algebra::{
    CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut, MixedCrtNtt,
    MixedCrtNttParamSet, MontCoeff, PackedPartialSplitEval16, PartialSplitEval16,
    PartialSplitNtt16,
};
use akita_field::packed::HasPacking;
use akita_field::{Fp64, HalvingField, Prime128Offset159, Prime128OffsetA7F7};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

type F = Fp64<{ Q32_MODULUS }>;
type R = CyclotomicRing<F, 64>;
type N = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;
type F128 = Prime128Offset159;
type R128 = CyclotomicRing<F128, 32>;
type N128 = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 32>;
type MixedN128 = MixedCrtNtt<Q128_NUM_PRIMES, 32>;
type PF128 = <F128 as HasPacking>::Packing;
type ProductionF128 = Prime128OffsetA7F7;
type ProductionR128D64 = CyclotomicRing<ProductionF128, 64>;
type ProductionN128D64 = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 64>;
type ProductionMixedN128D64 = MixedCrtNtt<Q128_NUM_PRIMES, 64>;
const CACHE_MAT_ROWS: usize = 8;
const CACHE_MAT_COLS: usize = 16;
const PRODUCTION_CACHE_MAT_COLS: usize = 128;
const MUL_BATCH_FACTORS: [usize; 3] = [1, 4, 16];

fn sample_ring(seed: u64) -> R {
    let coeffs = std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(31)
            .wrapping_add((i as u64).wrapping_mul(17));
        F::from_u64(x % Q32_MODULUS)
    });
    R::from_coefficients(coeffs)
}

fn sample_ring_q128m159(seed: u64) -> R128 {
    let coeffs = std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(29)
            .wrapping_add((i as u64).wrapping_mul(13));
        let centered = (x % 257) as i64 - 128;
        F128::from_i64(centered)
    });
    R128::from_coefficients(coeffs)
}

fn sample_centered_i8(seed: u64) -> [i8; 32] {
    std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(43)
            .wrapping_add((i as u64).wrapping_mul(17));
        ((x % 256) as i16 - 128) as i8
    })
}

fn sample_centered_i16(seed: u64) -> [i16; 32] {
    std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(43)
            .wrapping_add((i as u64).wrapping_mul(17));
        (x % 2048) as i16 - 1024
    })
}

fn sample_ring_q128m159_tag(seed: u64, tag: u64) -> R128 {
    sample_ring_q128m159(seed.wrapping_mul(131).wrapping_add(tag))
}

fn sample_production_ring_q128_d64(seed: u64) -> ProductionR128D64 {
    let coeffs = std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(29)
            .wrapping_add((i as u64).wrapping_mul(13));
        ProductionF128::from_i64((x % 257) as i64 - 128)
    });
    ProductionR128D64::from_coefficients(coeffs)
}

fn sample_production_i8_d64(seed: u64) -> [i8; 64] {
    std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(43)
            .wrapping_add((i as u64).wrapping_mul(17));
        ((x % 256) as i16 - 128) as i8
    })
}

fn sample_production_i16_d64(seed: u64) -> [i16; 64] {
    std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(43)
            .wrapping_add((i as u64).wrapping_mul(17));
        (x % 2048) as i16 - 1024
    })
}

fn pack_split_batch(batch: &[PartialSplitEval16<F128>]) -> Vec<PackedPartialSplitEval16<PF128>> {
    let width = PackedPartialSplitEval16::<PF128>::WIDTH;
    debug_assert_eq!(batch.len() % width, 0);
    batch
        .chunks_exact(width)
        .map(|chunk| PackedPartialSplitEval16::<PF128>::from_fn(|lane| chunk[lane]))
        .collect()
}

fn bench_ring_schoolbook_mul(c: &mut Criterion) {
    let lhs = sample_ring(3);
    let rhs = sample_ring(11);
    c.bench_function("ring_schoolbook_mul_d64", |b| {
        b.iter(|| black_box(lhs) * black_box(rhs))
    });
}

fn bench_ntt_single_prime_round_trip(c: &mut Criterion) {
    let primes = Q32_PRIMES;
    let prime = primes[0];
    let tw = NttTwiddles::<i32, 64>::compute(prime);
    let base: [MontCoeff<i32>; 64] =
        std::array::from_fn(|i| prime.from_canonical(((i * 5 + 7) as i32) % prime.p));

    c.bench_function("ntt_single_prime_forward_inverse_d64", |b| {
        b.iter(|| {
            let mut a = base;
            forward_ntt(&mut a, prime, &tw);
            inverse_ntt(&mut a, prime, &tw);
            black_box(a)
        })
    });
}

fn bench_ntt_i16_tail_round_trip(c: &mut Criterion) {
    let prime = I16_TAIL_PRIME;
    let tw = NttTwiddles::<i16, 64>::compute(prime);
    let base: [MontCoeff<i16>; 64] =
        std::array::from_fn(|i| prime.from_canonical(((i * 5 + 7) as i16) % prime.p));
    c.bench_function("ntt_i16_tail_forward_inverse_d64", |b| {
        b.iter(|| {
            let mut values = base;
            forward_ntt(&mut values, prime, &tw);
            inverse_ntt(&mut values, prime, &tw);
            black_box(values)
        })
    });
}

fn bench_crt_round_trip(c: &mut Criterion) {
    let ring = sample_ring(19);
    let primes = Q32_PRIMES;
    let twiddles: [NttTwiddles<i32, 64>; Q32_NUM_PRIMES] =
        std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
    let garner = q32_garner();

    c.bench_function("ring_ntt_crt_round_trip_d64_q32_2xi32", |b| {
        b.iter(|| {
            let ntt = N::from_ring(black_box(&ring), &primes, &twiddles);
            let back: R = ntt.to_ring(&primes, &twiddles, &garner);
            black_box(back)
        })
    });
}

fn bench_ring_schoolbook_mul_q128m159(c: &mut Criterion) {
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_ring_q128m159(41);
    c.bench_function("ring_schoolbook_mul_d32_q128m159", |b| {
        b.iter(|| black_box(lhs) * black_box(rhs))
    });
}

fn bench_partial_split_mul_q128m159(c: &mut Criterion) {
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_ring_q128m159(41);
    let split = PartialSplitNtt16::<F128>::compute();
    c.bench_function("ring_partial_split_mul_d32_q128m159", |b| {
        b.iter(|| split.multiply_d32(black_box(&lhs), black_box(&rhs)))
    });
}

fn bench_crt_mul_q128m159(c: &mut Criterion) {
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_ring_q128m159(41);
    let params = CrtNttParamSet::new(q128_primes());

    c.bench_function("ring_crt_ntt_mul_d32_q128m159_k5", |b| {
        b.iter(|| {
            let lhs_ntt = N128::from_ring_with_params(black_box(&lhs), &params);
            let rhs_ntt = N128::from_ring_with_params(black_box(&rhs), &params);
            let prod = lhs_ntt.pointwise_mul_with_params(&rhs_ntt, &params);
            let out: R128 = prod.to_ring_with_params(&params);
            black_box(out)
        })
    });
}

fn bench_partial_split_mul_i8_rhs_q128m159(c: &mut Criterion) {
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_centered_i8(41);
    let split = PartialSplitNtt16::<F128>::compute();
    c.bench_function("ring_partial_split_mul_i8_rhs_d32_q128m159", |b| {
        b.iter(|| split.multiply_d32_rhs_i8(black_box(&lhs), black_box(&rhs)))
    });
}

fn bench_crt_mul_i8_rhs_q128m159(c: &mut Criterion) {
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_centered_i8(41);
    let params = CrtNttParamSet::new(q128_primes());

    c.bench_function("ring_crt_ntt_mul_i8_rhs_d32_q128m159_k5", |b| {
        b.iter(|| {
            let lhs_ntt = N128::from_ring_with_params(black_box(&lhs), &params);
            let rhs_ntt = N128::from_i8_with_params(black_box(&rhs), &params);
            let prod = lhs_ntt.pointwise_mul_with_params(&rhs_ntt, &params);
            let out: R128 = prod.to_ring_with_params(&params);
            black_box(out)
        })
    });
}

fn bench_cached_mul_batch_scaling_q128m159(c: &mut Criterion) {
    let width = PackedPartialSplitEval16::<PF128>::WIDTH;
    let split = PartialSplitNtt16::<F128>::compute();
    let packed = split.packed::<PF128>();
    let params = CrtNttParamSet::new(q128_primes());
    let mut group = c.benchmark_group("ring_cached_mul_batch_scaling_d32_q128m159");

    for factor in MUL_BATCH_FACTORS {
        let count = factor * width;
        let lhs_split: Vec<PartialSplitEval16<F128>> = (0..count)
            .map(|idx| {
                PartialSplitEval16::from_ring(&split, &sample_ring_q128m159_tag(23, idx as u64))
            })
            .collect();
        let rhs_split: Vec<PartialSplitEval16<F128>> = (0..count)
            .map(|idx| {
                PartialSplitEval16::from_ring(&split, &sample_ring_q128m159_tag(41, idx as u64))
            })
            .collect();
        let lhs_packed = pack_split_batch(&lhs_split);
        let rhs_packed = pack_split_batch(&rhs_split);
        let lhs_crt: Vec<N128> = (0..count)
            .map(|idx| {
                N128::from_ring_with_params(&sample_ring_q128m159_tag(23, idx as u64), &params)
            })
            .collect();
        let rhs_crt: Vec<N128> = (0..count)
            .map(|idx| {
                N128::from_ring_with_params(&sample_ring_q128m159_tag(41, idx as u64), &params)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("split_scalar", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let out: Vec<R128> = (0..count)
                        .map(|idx| {
                            lhs_split[idx]
                                .pointwise_mul(black_box(&rhs_split[idx]), &split)
                                .to_ring(&split)
                        })
                        .collect();
                    black_box(out)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("split_packed", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut out = Vec::with_capacity(count);
                    for idx in 0..(count / width) {
                        let acc =
                            packed.pointwise_mul(&lhs_packed[idx], black_box(&rhs_packed[idx]));
                        packed.append_rings(&acc, &mut out);
                    }
                    black_box(out)
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("crt_simd", count), &count, |b, &count| {
            b.iter(|| {
                let out: Vec<R128> = (0..count)
                    .map(|idx| {
                        let mut acc = N128::zero();
                        acc.add_assign_pointwise_mul_with_params(
                            &lhs_crt[idx],
                            black_box(&rhs_crt[idx]),
                            &params,
                        );
                        acc.to_ring_with_params(&params)
                    })
                    .collect();
                black_box(out)
            })
        });
    }

    group.finish();
}

fn bench_cached_mul_batch_scaling_i8_rhs_q128m159(c: &mut Criterion) {
    let width = PackedPartialSplitEval16::<PF128>::WIDTH;
    let split = PartialSplitNtt16::<F128>::compute();
    let packed = split.packed::<PF128>();
    let params = CrtNttParamSet::new(q128_primes());
    let mut group = c.benchmark_group("ring_cached_mul_batch_scaling_i8_rhs_d32_q128m159");

    for factor in MUL_BATCH_FACTORS {
        let count = factor * width;
        let lhs_split: Vec<PartialSplitEval16<F128>> = (0..count)
            .map(|idx| {
                PartialSplitEval16::from_ring(&split, &sample_ring_q128m159_tag(23, idx as u64))
            })
            .collect();
        let rhs_split: Vec<PartialSplitEval16<F128>> = (0..count)
            .map(|idx| PartialSplitEval16::from_i8(&split, &sample_centered_i8(41 + idx as u64)))
            .collect();
        let lhs_packed = pack_split_batch(&lhs_split);
        let rhs_packed = pack_split_batch(&rhs_split);
        let lhs_crt: Vec<N128> = (0..count)
            .map(|idx| {
                N128::from_ring_with_params(&sample_ring_q128m159_tag(23, idx as u64), &params)
            })
            .collect();
        let rhs_crt: Vec<N128> = (0..count)
            .map(|idx| N128::from_i8_with_params(&sample_centered_i8(41 + idx as u64), &params))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("split_scalar", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let out: Vec<R128> = (0..count)
                        .map(|idx| {
                            lhs_split[idx]
                                .pointwise_mul(black_box(&rhs_split[idx]), &split)
                                .to_ring(&split)
                        })
                        .collect();
                    black_box(out)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("split_packed", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut out = Vec::with_capacity(count);
                    for idx in 0..(count / width) {
                        let acc =
                            packed.pointwise_mul(&lhs_packed[idx], black_box(&rhs_packed[idx]));
                        packed.append_rings(&acc, &mut out);
                    }
                    black_box(out)
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("crt_simd", count), &count, |b, &count| {
            b.iter(|| {
                let out: Vec<R128> = (0..count)
                    .map(|idx| {
                        let mut acc = N128::zero();
                        acc.add_assign_pointwise_mul_with_params(
                            &lhs_crt[idx],
                            black_box(&rhs_crt[idx]),
                            &params,
                        );
                        acc.to_ring_with_params(&params)
                    })
                    .collect();
                black_box(out)
            })
        });
    }

    group.finish();
}

fn bench_partial_split_cyclic_mul_q128m159(c: &mut Criterion) {
    let split = PartialSplitNtt16::<F128>::compute();
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_ring_q128m159(41);

    c.bench_function("ring_partial_split_cyclic_mul_d32_q128m159", |b| {
        b.iter(|| {
            let out = split.multiply_cyclic_d32(black_box(&lhs), black_box(&rhs));
            black_box(out)
        })
    });
}

fn bench_crt_cyclic_mul_q128m159(c: &mut Criterion) {
    let params = CrtNttParamSet::new(q128_primes());
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_ring_q128m159(41);

    c.bench_function("ring_crt_ntt_cyclic_mul_d32_q128m159_k5", |b| {
        b.iter(|| {
            let lhs_ntt = N128::from_ring_cyclic(black_box(&lhs), &params);
            let rhs_ntt = N128::from_ring_cyclic(black_box(&rhs), &params);
            let prod = lhs_ntt.pointwise_mul_with_params(&rhs_ntt, &params);
            let out: R128 = prod.to_ring_cyclic(&params);
            black_box(out)
        })
    });
}

fn bench_partial_split_quotient_q128m159(c: &mut Criterion) {
    let split = PartialSplitNtt16::<F128>::compute();
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_ring_q128m159(41);

    c.bench_function("ring_partial_split_quotient_d32_q128m159", |b| {
        b.iter(|| {
            let out = split.unreduced_quotient_d32(black_box(&lhs), black_box(&rhs));
            black_box(out)
        })
    });
}

fn bench_crt_quotient_q128m159(c: &mut Criterion) {
    let params = CrtNttParamSet::new(q128_primes());
    let lhs = sample_ring_q128m159(23);
    let rhs = sample_ring_q128m159(41);

    c.bench_function("ring_crt_ntt_quotient_d32_q128m159_k5", |b| {
        b.iter(|| {
            let lhs_neg = N128::from_ring_with_params(black_box(&lhs), &params);
            let rhs_neg = N128::from_ring_with_params(black_box(&rhs), &params);
            let neg: R128 = lhs_neg
                .pointwise_mul_with_params(&rhs_neg, &params)
                .to_ring_with_params(&params);

            let lhs_cyc = N128::from_ring_cyclic(black_box(&lhs), &params);
            let rhs_cyc = N128::from_ring_cyclic(black_box(&rhs), &params);
            let cyc: R128 = lhs_cyc
                .pointwise_mul_with_params(&rhs_cyc, &params)
                .to_ring_cyclic(&params);

            let out = R128::from_coefficients(std::array::from_fn(|i| {
                (cyc.coefficients()[i] - neg.coefficients()[i]).half()
            }));
            black_box(out)
        })
    });
}

fn bench_partial_split_cached_matvec_q128m159(c: &mut Criterion) {
    let split = PartialSplitNtt16::<F128>::compute();
    let matrix: Vec<Vec<PartialSplitEval16<F128>>> = (0..CACHE_MAT_ROWS)
        .map(|r| {
            (0..CACHE_MAT_COLS)
                .map(|col| {
                    PartialSplitEval16::from_ring(
                        &split,
                        &sample_ring_q128m159_tag(23, (r * CACHE_MAT_COLS + col) as u64),
                    )
                })
                .collect()
        })
        .collect();
    let vector: Vec<PartialSplitEval16<F128>> = (0..CACHE_MAT_COLS)
        .map(|col| PartialSplitEval16::from_ring(&split, &sample_ring_q128m159_tag(41, col as u64)))
        .collect();

    c.bench_function("ring_partial_split_cached_matvec_d32_q128m159", |b| {
        b.iter(|| {
            let out: Vec<R128> = matrix
                .iter()
                .map(|row| {
                    let mut acc = PartialSplitEval16::zero();
                    for (mat_entry, vec_entry) in row.iter().zip(vector.iter()) {
                        acc.add_mul_assign(mat_entry, black_box(vec_entry), &split);
                    }
                    acc.to_ring(&split)
                })
                .collect();
            black_box(out)
        })
    });
}

fn bench_partial_split_cached_matvec_i8_rhs_q128m159(c: &mut Criterion) {
    let split = PartialSplitNtt16::<F128>::compute();
    let matrix: Vec<Vec<PartialSplitEval16<F128>>> = (0..CACHE_MAT_ROWS)
        .map(|r| {
            (0..CACHE_MAT_COLS)
                .map(|col| {
                    PartialSplitEval16::from_ring(
                        &split,
                        &sample_ring_q128m159_tag(23, (r * CACHE_MAT_COLS + col) as u64),
                    )
                })
                .collect()
        })
        .collect();
    let vector: Vec<PartialSplitEval16<F128>> = (0..CACHE_MAT_COLS)
        .map(|col| PartialSplitEval16::from_i8(&split, &sample_centered_i8(41 + col as u64)))
        .collect();

    c.bench_function(
        "ring_partial_split_cached_matvec_i8_rhs_d32_q128m159",
        |b| {
            b.iter(|| {
                let out: Vec<R128> = matrix
                    .iter()
                    .map(|row| {
                        let mut acc = PartialSplitEval16::zero();
                        for (mat_entry, vec_entry) in row.iter().zip(vector.iter()) {
                            acc.add_mul_assign(mat_entry, black_box(vec_entry), &split);
                        }
                        acc.to_ring(&split)
                    })
                    .collect();
                black_box(out)
            })
        },
    );
}

fn bench_partial_split_packed_cached_matvec_q128m159(c: &mut Criterion) {
    let split = PartialSplitNtt16::<F128>::compute();
    let packed = split.packed::<PF128>();
    let matrix_scalar: Vec<Vec<PartialSplitEval16<F128>>> = (0..CACHE_MAT_ROWS)
        .map(|r| {
            (0..CACHE_MAT_COLS)
                .map(|col| {
                    PartialSplitEval16::from_ring(
                        &split,
                        &sample_ring_q128m159_tag(23, (r * CACHE_MAT_COLS + col) as u64),
                    )
                })
                .collect()
        })
        .collect();
    let vector_scalar: Vec<PartialSplitEval16<F128>> = (0..CACHE_MAT_COLS)
        .map(|col| PartialSplitEval16::from_ring(&split, &sample_ring_q128m159_tag(41, col as u64)))
        .collect();
    let mut matrix_chunks = matrix_scalar.chunks_exact(PackedPartialSplitEval16::<PF128>::WIDTH);
    let matrix_packed: Vec<Vec<PackedPartialSplitEval16<PF128>>> = matrix_chunks
        .by_ref()
        .map(|row_chunk| {
            (0..CACHE_MAT_COLS)
                .map(|col| PackedPartialSplitEval16::<PF128>::from_fn(|lane| row_chunk[lane][col]))
                .collect()
        })
        .collect();
    let matrix_scalar_tail = matrix_chunks.remainder();
    let vector_packed: Vec<PackedPartialSplitEval16<PF128>> = vector_scalar
        .iter()
        .map(PackedPartialSplitEval16::<PF128>::broadcast)
        .collect();

    c.bench_function(
        "ring_partial_split_packed_cached_matvec_d32_q128m159",
        |b| {
            b.iter(|| {
                let mut out = Vec::with_capacity(CACHE_MAT_ROWS);
                for packed_row in &matrix_packed {
                    let mut acc = PackedPartialSplitEval16::<PF128>::zero();
                    for (mat_entry, vec_entry) in packed_row.iter().zip(vector_packed.iter()) {
                        packed.add_mul_assign(&mut acc, mat_entry, black_box(vec_entry));
                    }
                    packed.append_rings(&acc, &mut out);
                }
                for row in matrix_scalar_tail {
                    let mut acc = PartialSplitEval16::zero();
                    for (mat_entry, vec_entry) in row.iter().zip(vector_scalar.iter()) {
                        acc.add_mul_assign(mat_entry, black_box(vec_entry), &split);
                    }
                    out.push(acc.to_ring(&split));
                }
                black_box(out)
            })
        },
    );
}

fn bench_crt_simd_cached_matvec_q128m159(c: &mut Criterion) {
    let params = CrtNttParamSet::new(q128_primes());
    let matrix: Vec<Vec<N128>> = (0..CACHE_MAT_ROWS)
        .map(|r| {
            (0..CACHE_MAT_COLS)
                .map(|col| {
                    N128::from_ring_with_params(
                        &sample_ring_q128m159_tag(23, (r * CACHE_MAT_COLS + col) as u64),
                        &params,
                    )
                })
                .collect()
        })
        .collect();
    let vector: Vec<N128> = (0..CACHE_MAT_COLS)
        .map(|col| N128::from_ring_with_params(&sample_ring_q128m159_tag(41, col as u64), &params))
        .collect();

    c.bench_function("ring_crt_ntt_simd_cached_matvec_d32_q128m159_k5", |b| {
        b.iter(|| {
            let out: Vec<R128> = matrix
                .iter()
                .map(|row| {
                    let mut acc = N128::zero();
                    for (mat_entry, vec_entry) in row.iter().zip(vector.iter()) {
                        acc.add_assign_pointwise_mul_with_params(
                            mat_entry,
                            black_box(vec_entry),
                            &params,
                        );
                    }
                    acc.to_ring_with_params(&params)
                })
                .collect();
            black_box(out)
        })
    });
}

fn bench_partial_split_packed_cached_matvec_i8_rhs_q128m159(c: &mut Criterion) {
    let split = PartialSplitNtt16::<F128>::compute();
    let packed = split.packed::<PF128>();
    let matrix_scalar: Vec<Vec<PartialSplitEval16<F128>>> = (0..CACHE_MAT_ROWS)
        .map(|r| {
            (0..CACHE_MAT_COLS)
                .map(|col| {
                    PartialSplitEval16::from_ring(
                        &split,
                        &sample_ring_q128m159_tag(23, (r * CACHE_MAT_COLS + col) as u64),
                    )
                })
                .collect()
        })
        .collect();
    let vector_scalar: Vec<PartialSplitEval16<F128>> = (0..CACHE_MAT_COLS)
        .map(|col| PartialSplitEval16::from_i8(&split, &sample_centered_i8(41 + col as u64)))
        .collect();
    let mut matrix_chunks = matrix_scalar.chunks_exact(PackedPartialSplitEval16::<PF128>::WIDTH);
    let matrix_packed: Vec<Vec<PackedPartialSplitEval16<PF128>>> = matrix_chunks
        .by_ref()
        .map(|row_chunk| {
            (0..CACHE_MAT_COLS)
                .map(|col| PackedPartialSplitEval16::<PF128>::from_fn(|lane| row_chunk[lane][col]))
                .collect()
        })
        .collect();
    let matrix_scalar_tail = matrix_chunks.remainder();
    let vector_packed: Vec<PackedPartialSplitEval16<PF128>> = vector_scalar
        .iter()
        .map(PackedPartialSplitEval16::<PF128>::broadcast)
        .collect();

    c.bench_function(
        "ring_partial_split_packed_cached_matvec_i8_rhs_d32_q128m159",
        |b| {
            b.iter(|| {
                let mut out = Vec::with_capacity(CACHE_MAT_ROWS);
                for packed_row in &matrix_packed {
                    let mut acc = PackedPartialSplitEval16::<PF128>::zero();
                    for (mat_entry, vec_entry) in packed_row.iter().zip(vector_packed.iter()) {
                        packed.add_mul_assign(&mut acc, mat_entry, black_box(vec_entry));
                    }
                    packed.append_rings(&acc, &mut out);
                }
                for row in matrix_scalar_tail {
                    let mut acc = PartialSplitEval16::zero();
                    for (mat_entry, vec_entry) in row.iter().zip(vector_scalar.iter()) {
                        acc.add_mul_assign(mat_entry, black_box(vec_entry), &split);
                    }
                    out.push(acc.to_ring(&split));
                }
                black_box(out)
            })
        },
    );
}

fn bench_crt_simd_cached_matvec_i8_rhs_q128m159(c: &mut Criterion) {
    let params = CrtNttParamSet::new(q128_primes());
    let matrix: Vec<Vec<N128>> = (0..CACHE_MAT_ROWS)
        .map(|r| {
            (0..CACHE_MAT_COLS)
                .map(|col| {
                    N128::from_ring_with_params(
                        &sample_ring_q128m159_tag(23, (r * CACHE_MAT_COLS + col) as u64),
                        &params,
                    )
                })
                .collect()
        })
        .collect();
    let vector: Vec<N128> = (0..CACHE_MAT_COLS)
        .map(|col| N128::from_i8_with_params(&sample_centered_i8(41 + col as u64), &params))
        .collect();

    c.bench_function(
        "ring_crt_ntt_simd_cached_matvec_i8_rhs_d32_q128m159_k5",
        |b| {
            b.iter(|| {
                let out: Vec<R128> = matrix
                    .iter()
                    .map(|row| {
                        let mut acc = N128::zero();
                        for (mat_entry, vec_entry) in row.iter().zip(vector.iter()) {
                            acc.add_assign_pointwise_mul_with_params(
                                mat_entry,
                                black_box(vec_entry),
                                &params,
                            );
                        }
                        acc.to_ring_with_params(&params)
                    })
                    .collect();
                black_box(out)
            })
        },
    );
}

fn bench_mixed_crt_cached_matvec_i16_rhs_q128m159(c: &mut Criterion) {
    let params = MixedCrtNttParamSet::new(
        CrtNttParamSet::new(q128_primes()),
        CrtNttParamSet::new([I16_TAIL_PRIME]),
    );
    let matrix: Vec<Vec<MixedN128>> = (0..CACHE_MAT_ROWS)
        .map(|row| {
            (0..CACHE_MAT_COLS)
                .map(|column| {
                    MixedN128::from_ring(
                        &sample_ring_q128m159_tag(23, (row * CACHE_MAT_COLS + column) as u64),
                        &params,
                    )
                })
                .collect()
        })
        .collect();
    let vector: Vec<MixedN128> = (0..CACHE_MAT_COLS)
        .map(|column| MixedN128::from_i16(&sample_centered_i16(41 + column as u64), &params))
        .collect();

    c.bench_function(
        "ring_mixed_crt_ntt_cached_matvec_i16_rhs_d32_q128m159_k5_plus_i16",
        |b| {
            b.iter(|| {
                let out: Vec<R128> = matrix
                    .iter()
                    .map(|row| {
                        let mut accumulator = MixedN128::zero();
                        for (matrix_entry, vector_entry) in row.iter().zip(vector.iter()) {
                            accumulator.add_assign_pointwise_mul(
                                matrix_entry,
                                black_box(vector_entry),
                                &params,
                            );
                        }
                        accumulator.to_ring(&params)
                    })
                    .collect();
                black_box(out)
            })
        },
    );
}

fn bench_crt_cached_dot_components_q128m159(c: &mut Criterion) {
    let wide_params = CrtNttParamSet::new(q128_primes());
    let mixed_params =
        MixedCrtNttParamSet::new(wide_params.clone(), CrtNttParamSet::new([I16_TAIL_PRIME]));
    let wide_matrix: Vec<N128> = (0..CACHE_MAT_COLS)
        .map(|column| {
            N128::from_ring_with_params(&sample_ring_q128m159_tag(23, column as u64), &wide_params)
        })
        .collect();
    let wide_vector: Vec<N128> = (0..CACHE_MAT_COLS)
        .map(|column| {
            N128::from_i8_with_params(&sample_centered_i8(41 + column as u64), &wide_params)
        })
        .collect();
    let mixed_matrix: Vec<MixedN128> = (0..CACHE_MAT_COLS)
        .map(|column| {
            MixedN128::from_ring(&sample_ring_q128m159_tag(23, column as u64), &mixed_params)
        })
        .collect();
    let mixed_vector: Vec<MixedN128> = (0..CACHE_MAT_COLS)
        .map(|column| MixedN128::from_i16(&sample_centered_i16(41 + column as u64), &mixed_params))
        .collect();
    let mut wide_product = N128::zero();
    let mut mixed_product = MixedN128::zero();
    for (matrix_entry, vector_entry) in wide_matrix.iter().zip(&wide_vector) {
        wide_product.add_assign_pointwise_mul_with_params(matrix_entry, vector_entry, &wide_params);
    }
    for (matrix_entry, vector_entry) in mixed_matrix.iter().zip(&mixed_vector) {
        mixed_product.add_assign_pointwise_mul(matrix_entry, vector_entry, &mixed_params);
    }

    c.bench_function("ring_crt_ntt_cached_dot_pointwise_d32_q128m159_k5", |b| {
        b.iter(|| {
            let mut accumulator = N128::zero();
            for (matrix_entry, vector_entry) in wide_matrix.iter().zip(&wide_vector) {
                accumulator.add_assign_pointwise_mul_with_params(
                    matrix_entry,
                    black_box(vector_entry),
                    &wide_params,
                );
            }
            black_box(accumulator)
        })
    });
    c.bench_function(
        "ring_mixed_crt_ntt_cached_dot_pointwise_d32_q128m159_k5_plus_i16",
        |b| {
            b.iter(|| {
                let mut accumulator = MixedN128::zero();
                for (matrix_entry, vector_entry) in mixed_matrix.iter().zip(&mixed_vector) {
                    accumulator.add_assign_pointwise_mul(
                        matrix_entry,
                        black_box(vector_entry),
                        &mixed_params,
                    );
                }
                black_box(accumulator)
            })
        },
    );
    c.bench_function("ring_crt_ntt_reconstruct_d32_q128m159_k5", |b| {
        b.iter(|| black_box(&wide_product).to_ring_with_params::<F128>(&wide_params))
    });
    c.bench_function(
        "ring_mixed_crt_ntt_reconstruct_d32_q128m159_k5_plus_i16",
        |b| b.iter(|| black_box(&mixed_product).to_ring::<F128>(&mixed_params)),
    );
}

fn bench_digit_lut_i8_range_q128m159(c: &mut Criterion) {
    let params: CrtNttParamSet<i32, Q128_NUM_PRIMES, 32> = CrtNttParamSet::new(q128_primes());
    let mut group = c.benchmark_group("digit_mont_lut_q128m159_k5");
    group.bench_function("construct_l6", |b| {
        b.iter(|| DigitMontLut::new_with_digit_bound(black_box(&params), 32))
    });
    group.bench_function("construct_l8", |b| {
        b.iter(|| DigitMontLut::new_with_digit_bound(black_box(&params), 128))
    });
    group.finish();
}

fn bench_production_crt_cached_matvec_d64_q128a7f7(c: &mut Criterion) {
    let wide_params: CrtNttParamSet<i32, Q128_NUM_PRIMES, 64> = CrtNttParamSet::new(q128_primes());
    let mixed_params =
        MixedCrtNttParamSet::new(wide_params.clone(), CrtNttParamSet::new([I16_TAIL_PRIME]));
    let wide_matrix: Vec<Vec<ProductionN128D64>> = (0..CACHE_MAT_ROWS)
        .map(|row| {
            (0..PRODUCTION_CACHE_MAT_COLS)
                .map(|column| {
                    ProductionN128D64::from_ring_with_params(
                        &sample_production_ring_q128_d64(
                            23 + (row * PRODUCTION_CACHE_MAT_COLS + column) as u64,
                        ),
                        &wide_params,
                    )
                })
                .collect()
        })
        .collect();
    let wide_vector: Vec<ProductionN128D64> = (0..PRODUCTION_CACHE_MAT_COLS)
        .map(|column| {
            ProductionN128D64::from_i8_with_params(
                &sample_production_i8_d64(41 + column as u64),
                &wide_params,
            )
        })
        .collect();
    let mixed_matrix: Vec<Vec<ProductionMixedN128D64>> = (0..CACHE_MAT_ROWS)
        .map(|row| {
            (0..PRODUCTION_CACHE_MAT_COLS)
                .map(|column| {
                    ProductionMixedN128D64::from_ring(
                        &sample_production_ring_q128_d64(
                            23 + (row * PRODUCTION_CACHE_MAT_COLS + column) as u64,
                        ),
                        &mixed_params,
                    )
                })
                .collect()
        })
        .collect();
    let mixed_vector: Vec<ProductionMixedN128D64> = (0..PRODUCTION_CACHE_MAT_COLS)
        .map(|column| {
            ProductionMixedN128D64::from_i16(
                &sample_production_i16_d64(41 + column as u64),
                &mixed_params,
            )
        })
        .collect();

    c.bench_function(
        "ring_crt_ntt_cached_matvec_i8_rhs_d64_q128a7f7_8x128_k5",
        |b| {
            b.iter(|| {
                let out: Vec<ProductionR128D64> = wide_matrix
                    .iter()
                    .map(|row| {
                        let mut accumulator = ProductionN128D64::zero();
                        for (matrix_entry, vector_entry) in row.iter().zip(&wide_vector) {
                            accumulator.add_assign_pointwise_mul_with_params(
                                matrix_entry,
                                black_box(vector_entry),
                                &wide_params,
                            );
                        }
                        accumulator.to_ring_with_params(&wide_params)
                    })
                    .collect();
                black_box(out)
            })
        },
    );
    c.bench_function(
        "ring_mixed_crt_ntt_cached_matvec_i16_rhs_d64_q128a7f7_8x128_k5_plus_i16",
        |b| {
            b.iter(|| {
                let out: Vec<ProductionR128D64> = mixed_matrix
                    .iter()
                    .map(|row| {
                        let mut accumulator = ProductionMixedN128D64::zero();
                        for (matrix_entry, vector_entry) in row.iter().zip(&mixed_vector) {
                            accumulator.add_assign_pointwise_mul(
                                matrix_entry,
                                black_box(vector_entry),
                                &mixed_params,
                            );
                        }
                        accumulator.to_ring(&mixed_params)
                    })
                    .collect();
                black_box(out)
            })
        },
    );
}

criterion_group!(
    ring_ntt,
    bench_ring_schoolbook_mul,
    bench_ntt_single_prime_round_trip,
    bench_ntt_i16_tail_round_trip,
    bench_crt_round_trip,
    bench_ring_schoolbook_mul_q128m159,
    bench_partial_split_mul_q128m159,
    bench_crt_mul_q128m159,
    bench_partial_split_mul_i8_rhs_q128m159,
    bench_crt_mul_i8_rhs_q128m159,
    bench_cached_mul_batch_scaling_q128m159,
    bench_cached_mul_batch_scaling_i8_rhs_q128m159,
    bench_partial_split_cyclic_mul_q128m159,
    bench_crt_cyclic_mul_q128m159,
    bench_partial_split_quotient_q128m159,
    bench_crt_quotient_q128m159,
    bench_partial_split_cached_matvec_q128m159,
    bench_partial_split_packed_cached_matvec_q128m159,
    bench_crt_simd_cached_matvec_q128m159,
    bench_partial_split_cached_matvec_i8_rhs_q128m159,
    bench_partial_split_packed_cached_matvec_i8_rhs_q128m159,
    bench_crt_simd_cached_matvec_i8_rhs_q128m159,
    bench_mixed_crt_cached_matvec_i16_rhs_q128m159,
    bench_crt_cached_dot_components_q128m159,
    bench_digit_lut_i8_range_q128m159,
    bench_production_crt_cached_matvec_d64_q128a7f7
);
criterion_main!(ring_ntt);
