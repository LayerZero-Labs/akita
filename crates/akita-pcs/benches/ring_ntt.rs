#![allow(missing_docs)]

use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::tables::{
    q128_primes, I16_TAIL_PRIME, Q128_NUM_PRIMES, Q32_MODULUS, Q32_NUM_PRIMES, Q32_PRIMES,
};
use akita_algebra::{
    balanced_decompose_coefficients_pow2_i8_into, mat_vec_i16_with_tail, CrtNttParamSet,
    CyclotomicCrtNtt, CyclotomicRing, DigitMontLut, I16TailParams, MontCoeff, NttKernelPlan,
};
use akita_field::{CanonicalField, Fp64, Prime128OffsetA7F7, Prime32Offset99};
use akita_types::{prepare_ntt_cache, FlatMatrix, NttCacheMode};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

type F = Fp64<{ Q32_MODULUS }>;
type R = CyclotomicRing<F, 64>;
type N = CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, 64>;
type ProductionF128 = Prime128OffsetA7F7;
type ProductionR128D64 = CyclotomicRing<ProductionF128, 64>;
type ProductionN128D64 = CyclotomicCrtNtt<i32, Q128_NUM_PRIMES, 64>;
type ProductionTailN128D64 = CyclotomicCrtNtt<i16, 1, 64>;
const CACHE_MAT_ROWS: usize = 8;
const PRODUCTION_CACHE_MAT_COLS: usize = 128;

fn sample_ring(seed: u64) -> R {
    let coeffs = std::array::from_fn(|i| {
        let x = seed
            .wrapping_mul(31)
            .wrapping_add((i as u64).wrapping_mul(17));
        F::from_u64(x % Q32_MODULUS)
    });
    R::from_coefficients(coeffs)
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

fn legacy_radix64_i8_reference_matvec(
    matrix: &[Vec<ProductionN128D64>],
    rhs: &[[i16; 64]],
    params: &CrtNttParamSet<i32, Q128_NUM_PRIMES, 64>,
) -> Vec<ProductionR128D64> {
    let mut remaining = rhs
        .iter()
        .map(|ring| ring.map(i32::from))
        .collect::<Vec<_>>();
    let mut out = vec![ProductionR128D64::zero(); matrix.len()];
    let mut scale = ProductionF128::one();
    while remaining
        .iter()
        .flatten()
        .any(|&coefficient| coefficient != 0)
    {
        let mut plane = vec![[0i8; 64]; rhs.len()];
        for (source, digits) in remaining.iter_mut().zip(&mut plane) {
            for (coefficient, digit) in source.iter_mut().zip(digits) {
                let residue = *coefficient & 63;
                let balanced = if residue >= 32 { residue - 64 } else { residue };
                *coefficient = (*coefficient - balanced) >> 6;
                *digit = balanced as i8;
            }
        }
        let lut = DigitMontLut::new_with_digit_bound(params, 32);
        let transformed = plane
            .iter()
            .map(|digits| ProductionN128D64::from_i8_with_lut(digits, params, &lut))
            .collect::<Vec<_>>();
        for (dst, row) in out.iter_mut().zip(matrix) {
            let mut accumulator = ProductionN128D64::zero();
            for (matrix_entry, vector_entry) in row.iter().zip(&transformed) {
                accumulator.add_assign_pointwise_mul(matrix_entry, vector_entry, params);
            }
            *dst += accumulator.to_ring(params).scale(&scale);
        }
        scale *= ProductionF128::from_i64(64);
    }
    out
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
    let plan = NttKernelPlan::detect::<i32>();

    c.bench_function("ntt_single_prime_forward_inverse_d64", |b| {
        b.iter(|| {
            let mut a = base;
            forward_ntt(&mut a, prime, &tw, plan);
            inverse_ntt(&mut a, prime, &tw, plan);
            black_box(a)
        })
    });
}

fn bench_ntt_i16_tail_round_trip_dimension<const D: usize>(c: &mut Criterion) {
    let prime = I16_TAIL_PRIME;
    let tw = NttTwiddles::<i16, D>::compute(prime);
    let base: [MontCoeff<i16>; D] =
        std::array::from_fn(|i| prime.from_canonical(((i * 5 + 7) as i16) % prime.p));
    let plan = NttKernelPlan::detect::<i16>();
    let name = format!("ntt_i16_tail_forward_inverse_d{D}");
    c.bench_function(&name, |b| {
        b.iter(|| {
            let mut values = base;
            forward_ntt(&mut values, prime, &tw, plan);
            inverse_ntt(&mut values, prime, &tw, plan);
            black_box(values)
        })
    });
}

fn bench_ntt_i16_tail_round_trip(c: &mut Criterion) {
    bench_ntt_i16_tail_round_trip_dimension::<64>(c);
    bench_ntt_i16_tail_round_trip_dimension::<128>(c);
    bench_ntt_i16_tail_round_trip_dimension::<256>(c);
    bench_ntt_i16_tail_round_trip_dimension::<512>(c);
}

fn bench_fp32_decomposition_dimension<const D: usize, const LOG_BASIS: u32>(c: &mut Criterion) {
    let coefficients: [Prime32Offset99; D] = std::array::from_fn(|i| {
        Prime32Offset99::from_canonical_u128_reduced(
            (i as u128 * 0x9e37_79b9 + 0x7f4a_7c15) % ((1u128 << 32) - 99),
        )
    });
    let levels = 32usize.div_ceil(LOG_BASIS as usize);
    let params = BalancedDecomposePow2Params::new(levels, LOG_BASIS, (1u128 << 32) - 99);
    let mut digits = vec![0i8; D * levels];
    let name = format!("fp32_balanced_decompose_l{LOG_BASIS}_d{D}");
    c.bench_function(&name, |b| {
        b.iter(|| {
            balanced_decompose_coefficients_pow2_i8_into(
                black_box(&coefficients),
                black_box(&mut digits),
                &params,
            );
        })
    });
}

fn bench_fp32_decomposition(c: &mut Criterion) {
    bench_fp32_decomposition_dimension::<64, 8>(c);
    bench_fp32_decomposition_dimension::<128, 8>(c);
    bench_fp32_decomposition_dimension::<256, 8>(c);
    bench_fp32_decomposition_dimension::<512, 8>(c);
    bench_fp32_decomposition_dimension::<128, 3>(c);
    bench_fp32_decomposition_dimension::<128, 4>(c);
    bench_fp32_decomposition_dimension::<128, 6>(c);
}

fn bench_crt_round_trip(c: &mut Criterion) {
    let ring = sample_ring(19);
    let params = CrtNttParamSet::new(Q32_PRIMES);

    c.bench_function("ring_ntt_crt_round_trip_d64_q32_2xi32", |b| {
        b.iter(|| {
            let ntt = N::from_ring(black_box(&ring), &params);
            let back: R = ntt.to_ring(&params);
            black_box(back)
        })
    });
}

fn bench_digit_lut_i8_range_q128(c: &mut Criterion) {
    let params: CrtNttParamSet<i32, Q128_NUM_PRIMES, 64> = CrtNttParamSet::new(q128_primes());
    let mut group = c.benchmark_group("digit_mont_lut_q128_k5");
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
        I16TailParams::new(wide_params.clone(), CrtNttParamSet::new([I16_TAIL_PRIME]));
    let wide_matrix: Vec<Vec<ProductionN128D64>> = (0..CACHE_MAT_ROWS)
        .map(|row| {
            (0..PRODUCTION_CACHE_MAT_COLS)
                .map(|column| {
                    ProductionN128D64::from_ring(
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
    let source_rings = (0..CACHE_MAT_ROWS * PRODUCTION_CACHE_MAT_COLS)
        .map(|index| sample_production_ring_q128_d64(23 + index as u64))
        .collect::<Vec<_>>();
    let mixed_wide_matrix = source_rings
        .iter()
        .map(|ring| ProductionN128D64::from_ring(ring, &mixed_params.wide))
        .collect::<Vec<_>>();
    let mixed_tail_matrix = source_rings
        .iter()
        .map(|ring| ProductionTailN128D64::from_ring(ring, &mixed_params.tail))
        .collect::<Vec<_>>();
    let terminal_rhs = (0..PRODUCTION_CACHE_MAT_COLS)
        .map(|column| sample_production_i16_d64(41 + column as u64))
        .collect::<Vec<_>>();

    c.bench_function(
        "ring_crt_ntt_cached_matvec_i8_rhs_d64_q128a7f7_8x128_k5",
        |b| {
            b.iter(|| {
                let out: Vec<ProductionR128D64> = wide_matrix
                    .iter()
                    .map(|row| {
                        let mut accumulator = ProductionN128D64::zero();
                        for (matrix_entry, vector_entry) in row.iter().zip(&wide_vector) {
                            accumulator.add_assign_pointwise_mul(
                                matrix_entry,
                                black_box(vector_entry),
                                &wide_params,
                            );
                        }
                        accumulator.to_ring(&wide_params)
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
                black_box(
                    mat_vec_i16_with_tail::<ProductionF128, Q128_NUM_PRIMES, 64>(
                        &mixed_wide_matrix,
                        &mixed_tail_matrix,
                        CACHE_MAT_ROWS,
                        PRODUCTION_CACHE_MAT_COLS,
                        black_box(&terminal_rhs),
                        &mixed_params,
                    )
                    .expect("mixed i16 matvec"),
                )
            })
        },
    );
    c.bench_function(
        "terminal_relation_legacy_radix64_i8_reference_d64_q128a7f7_8x128_k5",
        |b| {
            b.iter(|| {
                black_box(legacy_radix64_i8_reference_matvec(
                    &wide_matrix,
                    black_box(&terminal_rhs),
                    &wide_params,
                ))
            })
        },
    );
    c.bench_function(
        "terminal_relation_mixed_i16_d64_q128a7f7_8x128_k5_plus_i16",
        |b| {
            b.iter(|| {
                black_box(
                    mat_vec_i16_with_tail::<ProductionF128, Q128_NUM_PRIMES, 64>(
                        &mixed_wide_matrix,
                        &mixed_tail_matrix,
                        CACHE_MAT_ROWS,
                        PRODUCTION_CACHE_MAT_COLS,
                        black_box(&terminal_rhs),
                        &mixed_params,
                    )
                    .expect("terminal mixed i16 matvec"),
                )
            })
        },
    );
}

fn bench_ntt_cache_construction_d64_q128a7f7(c: &mut Criterion) {
    let rings = (0..CACHE_MAT_ROWS * PRODUCTION_CACHE_MAT_COLS)
        .map(|index| sample_production_ring_q128_d64(23 + index as u64))
        .collect::<Vec<_>>();
    let flat = FlatMatrix::from_ring_slice(&rings);
    let mut group = c.benchmark_group("ntt_cache_construction_d64_q128a7f7_1024_rings");
    group.bench_function("both_transforms", |b| {
        b.iter(|| {
            let view = flat
                .ring_view::<64>(1, rings.len())
                .expect("production matrix view");
            black_box(prepare_ntt_cache(view, NttCacheMode::BothTransforms).expect("both cache"))
        })
    });
    group.bench_function("exact_negacyclic_base_l10_w64", |b| {
        b.iter(|| {
            let view = flat
                .ring_view::<64>(1, rings.len())
                .expect("production matrix view");
            black_box(
                prepare_ntt_cache(
                    view,
                    NttCacheMode::ExactNegacyclic {
                        width: 64,
                        rhs_abs_bound: 1 << 9,
                    },
                )
                .expect("base cache"),
            )
        })
    });
    group.bench_function("exact_negacyclic_tail_l10_w128", |b| {
        b.iter(|| {
            let view = flat
                .ring_view::<64>(1, rings.len())
                .expect("production matrix view");
            black_box(
                prepare_ntt_cache(
                    view,
                    NttCacheMode::ExactNegacyclic {
                        width: 128,
                        rhs_abs_bound: 1 << 9,
                    },
                )
                .expect("tail cache"),
            )
        })
    });
    group.finish();
}

criterion_group!(
    ring_ntt,
    bench_ring_schoolbook_mul,
    bench_ntt_single_prime_round_trip,
    bench_ntt_i16_tail_round_trip,
    bench_fp32_decomposition,
    bench_crt_round_trip,
    bench_digit_lut_i8_range_q128,
    bench_production_crt_cached_matvec_d64_q128a7f7,
    bench_ntt_cache_construction_d64_q128a7f7
);
criterion_main!(ring_ntt);
