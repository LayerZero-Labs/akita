#![allow(missing_docs)]

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use akita_algebra::tables::{
    q128_primes, q32_garner, Q128_NUM_PRIMES, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES,
};
use akita_algebra::{
    CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, MontCoeff, PackedPartialSplitEval16,
    PartialSplitEval16, PartialSplitNtt16,
};
use akita_field::{Fp64, HalvingField, HasPacking, Prime128Offset159};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

type F = Fp64<{ Q32_MODULUS }>;
type R = CyclotomicRing<F, 64>;
type N = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;
type F128 = Prime128Offset159;
type R128 = CyclotomicRing<F128, 32>;
type N128 = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 32>;
type PF128 = <F128 as HasPacking>::Packing;
const CACHE_MAT_ROWS: usize = 8;
const CACHE_MAT_COLS: usize = 16;
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

fn sample_ring_q128m159_tag(seed: u64, tag: u64) -> R128 {
    sample_ring_q128m159(seed.wrapping_mul(131).wrapping_add(tag))
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

fn bench_batched_forward_ntt_d32(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    {
        use akita_algebra::ntt::avx::batch::{batched_forward_ntt_16rows, BATCH_LANES};
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        let prime = q128_primes()[0];
        let tw = NttTwiddles::<i32, 32>::compute(prime);
        // Identical Vec-clone harness for all cases; the ONLY variable across the
        // batched cases is `row_stride`. Tight stride = 32 (rows packed), realistic
        // setup/commit stride = K*32 (limb[k] of 16 consecutive CrtNtt elements,
        // K=5 here). Per-poly uses the same buffer at stride 32.
        const K: usize = 5;
        let make_buf = |stride: usize| {
            let mut buf = vec![0i32; BATCH_LANES * stride];
            for r in 0..BATCH_LANES {
                for i in 0..32 {
                    buf[r * stride + i] = prime
                        .from_canonical(((r * 32 + i * 5 + 7) as i32) % prime.p)
                        .raw();
                }
            }
            buf
        };
        let buf32 = make_buf(32);
        let buf_kd = make_buf(K * 32);

        let mut group = c.benchmark_group("batched_forward_ntt_d32_16rows_q128");
        group.bench_function("per_poly_stride32", |b| {
            b.iter(|| {
                let mut buf = black_box(buf32.clone());
                for r in 0..BATCH_LANES {
                    // SAFETY: MontCoeff<i32> is a transparent i32; each row is 32
                    // contiguous i32 at offset r*32.
                    let row = unsafe {
                        &mut *(buf.as_mut_ptr().add(r * 32) as *mut [MontCoeff<i32>; 32])
                    };
                    forward_ntt::<i32, 32>(row, prime, &tw);
                }
                black_box(buf)
            })
        });
        group.bench_function("batched_tight_stride32", |b| {
            b.iter(|| {
                let mut buf = black_box(buf32.clone());
                // SAFETY: guarded by the AVX-512 feature detection above.
                unsafe { batched_forward_ntt_16rows::<32>(buf.as_mut_ptr(), 32, prime, &tw) };
                black_box(buf)
            })
        });
        group.bench_function("batched_interleaved_strideKD", |b| {
            b.iter(|| {
                let mut buf = black_box(buf_kd.clone());
                // SAFETY: guarded by AVX-512 detection; buf holds 16 rows at K*32.
                unsafe { batched_forward_ntt_16rows::<32>(buf.as_mut_ptr(), K * 32, prime, &tw) };
                black_box(buf)
            })
        });
        group.finish();
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = c;
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

criterion_group!(
    ring_ntt,
    bench_ring_schoolbook_mul,
    bench_ntt_single_prime_round_trip,
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
    bench_batched_forward_ntt_d32
);
criterion_main!(ring_ntt);
