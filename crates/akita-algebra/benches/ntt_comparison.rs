use std::fmt::Debug;
use std::hint::black_box;
use std::time::Duration;

use akita_algebra::tables::{q128_primes, Q128_NUM_PRIMES, Q64_NUM_PRIMES, Q64_PRIMES};
use akita_algebra::{
    CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, CyclotomicRing, DigitMontLut, Field,
    MinusTrinomial, MontCoeff, PlusTrinomial, Prime64Offset23703, SmoothFftField, TrinomialI8Lut,
    TrinomialModulus, TrinomialNtt, TrinomialNttDomain, TrinomialNttWorkspace, TrinomialRing,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use jolt_field::{Prime128OffsetA7F7, WithPacking};

#[path = "ntt_comparison/ifma.rs"]
mod ifma;

fn small_coefficients<F: Field, const D: usize>(seed: u64) -> [F; D] {
    std::array::from_fn(|index| {
        let centered = ((seed + index as u64 * 29) % 127) as i64 - 63;
        F::from_i64(centered)
    })
}

fn full_field_coefficients<F: Field, const D: usize>(seed: u64) -> [F; D] {
    std::array::from_fn(|index| {
        let value = (u128::from(seed) << 64)
            .wrapping_add(index as u128 * 0x9e37_79b9_7f4a_7c15_6a09_e667_f3bc_c909)
            .rotate_left((index % 127) as u32);
        F::from_u128(value)
    })
}

fn configure(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    d: usize,
) {
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(d as u64));
}

const MATVEC_ROWS: usize = 4;
const MATVEC_COLS: usize = 6;
const MATVEC_LOG_BASIS: u32 = 4;
const CRT_DOT_BATCH: usize = 6;

fn matvec_digits<const D: usize>() -> [[i8; D]; MATVEC_COLS] {
    std::array::from_fn(|column| {
        std::array::from_fn(|index| ((column * 5 + index * 7 + 3) % 16) as i8 - 8)
    })
}

fn run_crt_matvec_into<const K: usize, const D: usize>(
    accumulators: &mut [CyclotomicCrtNtt<i32, K, D>],
    matrix_rows: &[&[CyclotomicCrtNtt<i32, K, D>]],
    rhs: &[[i8; D]; MATVEC_COLS],
    params: &CrtNttParamSet<i32, K, D>,
    lut: &DigitMontLut<i32, K>,
    scratch: &mut [[MontCoeff<i32>; D]; CRT_DOT_BATCH],
) {
    for accumulator in accumulators.iter_mut() {
        *accumulator = CyclotomicCrtNtt::zero();
    }
    if params.pointwise_dot_batch_size() >= MATVEC_COLS {
        CyclotomicCrtNtt::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
            accumulators,
            matrix_rows,
            0,
            rhs,
            params,
            lut,
            scratch,
        );
        return;
    }

    for (column, digits) in rhs.iter().enumerate() {
        let transformed = CyclotomicCrtNtt::from_i8_with_lut(digits, params, lut);
        for (accumulator, matrix_row) in accumulators.iter_mut().zip(matrix_rows) {
            accumulator.add_assign_pointwise_mul(&matrix_row[column], &transformed, params);
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark kernel keeps each reusable buffer visible"
)]
fn run_trinomial_matvec_into<F, const D: usize, M, const PACKED: bool>(
    domain: &TrinomialNttDomain<F, D, M>,
    accumulators: &mut [TrinomialNtt<F, D, M>],
    zero: &TrinomialNtt<F, D, M>,
    matrix: &[TrinomialNtt<F, D, M>],
    rhs: &[[i8; D]; MATVEC_COLS],
    lut: &TrinomialI8Lut<F, D, M>,
    transformed_rhs: &mut TrinomialNtt<F, D, M>,
    workspace: &mut TrinomialNttWorkspace<F, D, M>,
) where
    F: SmoothFftField + WithPacking + Debug,
    M: TrinomialModulus,
{
    for accumulator in accumulators.iter_mut() {
        *accumulator = zero.clone();
    }
    for (column, digits) in rhs.iter().enumerate() {
        domain
            .forward_i8_with_lut_into_workspace(digits, lut, transformed_rhs, workspace)
            .unwrap();
        for (row, accumulator) in accumulators.iter_mut().enumerate() {
            let matrix_entry = &matrix[row * MATVEC_COLS + column];
            if PACKED {
                accumulator.add_assign_pointwise_mul_packed(matrix_entry, transformed_rhs);
            } else {
                accumulator.add_assign_pointwise_mul(matrix_entry, transformed_rhs);
            }
        }
    }
}

fn bench_crt<F, const K: usize, const D: usize>(
    criterion: &mut Criterion,
    field: &str,
    primes: [akita_algebra::NttPrime<i32>; K],
) where
    F: CrtNttConvertibleField + Debug,
{
    let params = CrtNttParamSet::new(primes);
    assert!(params.crt_capacity().supports::<F, D>(1, 64));
    eprintln!("{field} D={D} CRT kernel plan: {:?}", params.kernel_plan());
    let lhs = CyclotomicRing::<F, D>::from_coefficients(full_field_coefficients(17));
    let rhs_digits: [i8; D] = std::array::from_fn(|index| ((93 + index * 29) % 127) as i8 - 63);
    let rhs = CyclotomicRing::<F, D>::from_coefficients(
        rhs_digits.map(|digit| F::from_i64(i64::from(digit))),
    );
    let digit_lut = DigitMontLut::new_with_digit_bound(&params, 64);
    let lhs_ntt = CyclotomicCrtNtt::from_ring(&lhs, &params);
    let rhs_ntt = CyclotomicCrtNtt::from_i8_with_lut(&rhs_digits, &params, &digit_lut);
    let product_ntt = lhs_ntt.pointwise_mul(&rhs_ntt, &params);
    assert_eq!(product_ntt.to_ring::<F>(&params), lhs * rhs);
    let label = format!("crt_{field}_negacyclic/D={D}");
    let mut group = criterion.benchmark_group("ntt_comparison");
    configure(&mut group, D);

    group.bench_with_input(
        BenchmarkId::new("conversion_inclusive_multiply", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                let lhs_ntt = CyclotomicCrtNtt::from_ring(black_box(&lhs), black_box(&params));
                let rhs_ntt = CyclotomicCrtNtt::from_i8_with_lut(
                    black_box(&rhs_digits),
                    black_box(&params),
                    black_box(&digit_lut),
                );
                let product = lhs_ntt.pointwise_mul(black_box(&rhs_ntt), black_box(&params));
                black_box(product.to_ring::<F>(black_box(&params)))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_lhs_multiply", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                let rhs_ntt = CyclotomicCrtNtt::from_i8_with_lut(
                    black_box(&rhs_digits),
                    black_box(&params),
                    black_box(&digit_lut),
                );
                let product = lhs_ntt.pointwise_mul(black_box(&rhs_ntt), black_box(&params));
                black_box(product.to_ring::<F>(black_box(&params)))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("forward_i8_lut", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                black_box(CyclotomicCrtNtt::from_i8_with_lut(
                    black_box(&rhs_digits),
                    black_box(&params),
                    black_box(&digit_lut),
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("inverse_reconstruction", &label),
        &label,
        |bench, _| bench.iter(|| black_box(product_ntt.to_ring::<F>(black_box(&params)))),
    );
    group.bench_with_input(
        BenchmarkId::new("pretransformed_pointwise_mul", &label),
        &label,
        |bench, _| bench.iter(|| black_box(lhs_ntt.pointwise_mul(black_box(&rhs_ntt), &params))),
    );
    group.bench_with_input(
        BenchmarkId::new("pretransformed_mac", &label),
        &label,
        |bench, _| {
            bench.iter_batched(
                || lhs_ntt.clone(),
                |mut accumulator| {
                    accumulator.add_assign_pointwise_mul(
                        black_box(&lhs_ntt),
                        black_box(&rhs_ntt),
                        black_box(&params),
                    );
                    black_box(accumulator)
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();
}

fn bench_trinomial<F, const D: usize, M>(criterion: &mut Criterion, field: &str, modulus: &str)
where
    F: SmoothFftField + WithPacking + Debug,
    M: TrinomialModulus,
{
    let domain = TrinomialNttDomain::<F, D, M>::new().expect("comparison shape must split");
    let lhs = TrinomialRing::from_coefficients(full_field_coefficients(17)).unwrap();
    let rhs = TrinomialRing::from_coefficients(small_coefficients(93)).unwrap();
    let rhs_digits: [i8; D] = std::array::from_fn(|index| ((93 + index * 29) % 127) as i8 - 63);
    let digit_lut = domain.prepare_i8_lut(7).unwrap();
    let lhs_ntt = domain.forward(&lhs);
    let rhs_ntt = domain
        .forward_i8_with_lut_workspace(&rhs_digits, &digit_lut, &mut domain.workspace())
        .unwrap();
    let product_ntt = lhs_ntt.pointwise_mul(&rhs_ntt);
    assert_eq!(
        domain.inverse(&product_ntt),
        lhs.schoolbook_mul(&rhs).unwrap()
    );
    let label = format!("trinomial_{field}_{modulus}/D={D}");
    let mut group = criterion.benchmark_group("ntt_comparison");
    configure(&mut group, D);

    let mut multiply_workspace = domain.workspace();
    let mut multiply_lhs = domain.zero_ntt();
    let mut multiply_rhs = domain.zero_ntt();
    group.bench_with_input(
        BenchmarkId::new("conversion_inclusive_multiply", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                domain.forward_into_with_workspace(
                    black_box(&lhs),
                    black_box(&mut multiply_lhs),
                    black_box(&mut multiply_workspace),
                );
                domain
                    .forward_i8_with_lut_into_workspace(
                        black_box(&rhs_digits),
                        black_box(&digit_lut),
                        black_box(&mut multiply_rhs),
                        black_box(&mut multiply_workspace),
                    )
                    .unwrap();
                let product = multiply_lhs.pointwise_mul(&multiply_rhs);
                black_box(
                    domain.inverse_with_workspace(&product, black_box(&mut multiply_workspace)),
                )
            })
        },
    );
    let mut prepared_workspace = domain.workspace();
    let mut prepared_rhs = domain.zero_ntt();
    group.bench_with_input(
        BenchmarkId::new("prepared_lhs_multiply", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                domain
                    .forward_i8_with_lut_into_workspace(
                        black_box(&rhs_digits),
                        black_box(&digit_lut),
                        black_box(&mut prepared_rhs),
                        black_box(&mut prepared_workspace),
                    )
                    .unwrap();
                let product = lhs_ntt.pointwise_mul(black_box(&prepared_rhs));
                black_box(
                    domain.inverse_with_workspace(&product, black_box(&mut prepared_workspace)),
                )
            })
        },
    );
    let mut forward_workspace = domain.workspace();
    group.bench_with_input(
        BenchmarkId::new("forward_field_coefficients", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    domain
                        .forward_with_workspace(black_box(&rhs), black_box(&mut forward_workspace)),
                )
            })
        },
    );
    let mut digit_workspace = domain.workspace();
    let mut digit_output = domain.zero_ntt();
    group.bench_with_input(
        BenchmarkId::new("forward_i8_position_lut_into", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                domain
                    .forward_i8_with_lut_into_workspace(
                        black_box(&rhs_digits),
                        black_box(&digit_lut),
                        black_box(&mut digit_output),
                        black_box(&mut digit_workspace),
                    )
                    .unwrap();
                black_box(&mut digit_output);
            })
        },
    );
    let mut inverse_workspace = domain.workspace();
    group.bench_with_input(
        BenchmarkId::new("inverse_reconstruction", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                black_box(domain.inverse_with_workspace(
                    black_box(&product_ntt),
                    black_box(&mut inverse_workspace),
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("pretransformed_pointwise_mul", &label),
        &label,
        |bench, _| bench.iter(|| black_box(lhs_ntt.pointwise_mul(black_box(&rhs_ntt)))),
    );
    group.bench_with_input(
        BenchmarkId::new("pretransformed_mac", &label),
        &label,
        |bench, _| {
            bench.iter_batched(
                || lhs_ntt.clone(),
                |mut accumulator| {
                    accumulator.add_assign_pointwise_mul(black_box(&lhs_ntt), black_box(&rhs_ntt));
                    black_box(accumulator)
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.bench_with_input(
        BenchmarkId::new("pretransformed_mac_packed", &label),
        &label,
        |bench, _| {
            bench.iter_batched(
                || lhs_ntt.clone(),
                |mut accumulator| {
                    accumulator
                        .add_assign_pointwise_mul_packed(black_box(&lhs_ntt), black_box(&rhs_ntt));
                    black_box(accumulator)
                },
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();
}

fn bench_crt_matvec<F, const K: usize, const D: usize>(
    criterion: &mut Criterion,
    field: &str,
    primes: [akita_algebra::NttPrime<i32>; K],
) where
    F: CrtNttConvertibleField + Debug,
{
    let params = CrtNttParamSet::new(primes);
    assert!(
        params
            .crt_capacity()
            .supports::<F, D>(MATVEC_COLS, 1 << (MATVEC_LOG_BASIS - 1)),
        "native CRT profile must cover the full-field by bounded-digit workload"
    );
    let matrix: Vec<CyclotomicRing<F, D>> = (0..MATVEC_ROWS * MATVEC_COLS)
        .map(|entry| CyclotomicRing::from_coefficients(full_field_coefficients(101 + entry as u64)))
        .collect();
    let rhs_digits = matvec_digits::<D>();
    let rhs: [CyclotomicRing<F, D>; MATVEC_COLS] = rhs_digits.map(|digits| {
        CyclotomicRing::from_coefficients(digits.map(|digit| F::from_i64(i64::from(digit))))
    });
    let prepared: Vec<_> = matrix
        .iter()
        .map(|entry| CyclotomicCrtNtt::from_ring(entry, &params))
        .collect();
    let matrix_rows: Vec<_> = prepared.chunks_exact(MATVEC_COLS).collect();
    let lut = DigitMontLut::new_with_digit_bound(&params, 1 << (MATVEC_LOG_BASIS - 1));
    let mut accumulators = vec![CyclotomicCrtNtt::zero(); MATVEC_ROWS];
    let mut scratch = [[MontCoeff::from_raw(0i32); D]; CRT_DOT_BATCH];
    run_crt_matvec_into(
        &mut accumulators,
        &matrix_rows,
        &rhs_digits,
        &params,
        &lut,
        &mut scratch,
    );
    for row in 0..MATVEC_ROWS {
        let mut expected = CyclotomicRing::zero();
        for column in 0..MATVEC_COLS {
            matrix[row * MATVEC_COLS + column].mul_accumulate_into(&rhs[column], &mut expected);
        }
        assert_eq!(accumulators[row].to_ring::<F>(&params), expected);
    }

    eprintln!(
        "CRT {field} D={D}: plan={:?}, prepared={} B/coeff, LUT={} B",
        params.kernel_plan(),
        K * core::mem::size_of::<i32>(),
        core::mem::size_of_val(&lut),
    );
    let label = format!("crt_{field}_negacyclic/D={D}");
    let mut group = criterion.benchmark_group("ntt_matvec_comparison");
    configure(&mut group, D * MATVEC_ROWS * MATVEC_COLS);
    group.bench_with_input(
        BenchmarkId::new("prepared_matrix_setup", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    matrix
                        .iter()
                        .map(|entry| CyclotomicCrtNtt::from_ring(black_box(entry), &params))
                        .collect::<Vec<_>>(),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("digit_lut_setup", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                black_box(DigitMontLut::new_with_digit_bound(
                    black_box(&params),
                    1 << (MATVEC_LOG_BASIS - 1),
                ))
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_matvec_ntt", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                run_crt_matvec_into(
                    black_box(&mut accumulators),
                    black_box(&matrix_rows),
                    black_box(&rhs_digits),
                    black_box(&params),
                    black_box(&lut),
                    black_box(&mut scratch),
                );
                black_box(&mut accumulators);
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_matvec_with_output", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                run_crt_matvec_into(
                    black_box(&mut accumulators),
                    black_box(&matrix_rows),
                    black_box(&rhs_digits),
                    black_box(&params),
                    black_box(&lut),
                    black_box(&mut scratch),
                );
                for accumulator in &accumulators {
                    black_box(accumulator.to_ring::<F>(black_box(&params)));
                }
            })
        },
    );
    group.finish();
}

fn bench_trinomial_matvec<F, const D: usize, M>(
    criterion: &mut Criterion,
    field: &str,
    modulus: &str,
) where
    F: SmoothFftField + WithPacking + Debug,
    M: TrinomialModulus,
{
    let domain = TrinomialNttDomain::<F, D, M>::new().expect("comparison shape must split");
    let matrix: Vec<TrinomialRing<F, D, M>> = (0..MATVEC_ROWS * MATVEC_COLS)
        .map(|entry| {
            TrinomialRing::from_coefficients(full_field_coefficients(101 + entry as u64)).unwrap()
        })
        .collect();
    let rhs_digits = matvec_digits::<D>();
    let rhs: [TrinomialRing<F, D, M>; MATVEC_COLS] = rhs_digits.map(|digits| {
        TrinomialRing::from_coefficients(digits.map(|digit| F::from_i64(i64::from(digit)))).unwrap()
    });
    let prepared: Vec<_> = matrix.iter().map(|entry| domain.forward(entry)).collect();
    let lut = domain.prepare_i8_lut(MATVEC_LOG_BASIS).unwrap();
    let zero = domain.zero_ntt();
    let mut accumulators = vec![zero.clone(); MATVEC_ROWS];
    let mut transformed_rhs = zero.clone();
    let mut workspace = domain.workspace();
    run_trinomial_matvec_into::<F, D, M, false>(
        &domain,
        &mut accumulators,
        &zero,
        &prepared,
        &rhs_digits,
        &lut,
        &mut transformed_rhs,
        &mut workspace,
    );
    for row in 0..MATVEC_ROWS {
        let mut expected = TrinomialRing::zero().unwrap();
        for column in 0..MATVEC_COLS {
            expected += matrix[row * MATVEC_COLS + column]
                .schoolbook_mul(&rhs[column])
                .unwrap();
        }
        assert_eq!(
            domain.inverse_with_workspace(&accumulators[row], &mut workspace),
            expected
        );
    }

    eprintln!(
        "trinomial {field} {modulus} D={D}: prepared={} B/coeff, LUT={} B",
        core::mem::size_of::<F>(),
        lut.table_bytes(),
    );
    let label = format!("trinomial_{field}_{modulus}/D={D}");
    let mut group = criterion.benchmark_group("ntt_matvec_comparison");
    configure(&mut group, D * MATVEC_ROWS * MATVEC_COLS);
    group.bench_with_input(
        BenchmarkId::new("prepared_matrix_setup", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    matrix
                        .iter()
                        .map(|entry| domain.forward(black_box(entry)))
                        .collect::<Vec<_>>(),
                )
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("digit_lut_setup", &label),
        &label,
        |bench, _| bench.iter(|| black_box(domain.prepare_i8_lut(MATVEC_LOG_BASIS).unwrap())),
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_matvec_ntt", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                run_trinomial_matvec_into::<F, D, M, false>(
                    black_box(&domain),
                    black_box(&mut accumulators),
                    black_box(&zero),
                    black_box(&prepared),
                    black_box(&rhs_digits),
                    black_box(&lut),
                    black_box(&mut transformed_rhs),
                    black_box(&mut workspace),
                );
                black_box(&mut accumulators);
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_matvec_ntt_packed", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                run_trinomial_matvec_into::<F, D, M, true>(
                    black_box(&domain),
                    black_box(&mut accumulators),
                    black_box(&zero),
                    black_box(&prepared),
                    black_box(&rhs_digits),
                    black_box(&lut),
                    black_box(&mut transformed_rhs),
                    black_box(&mut workspace),
                );
                black_box(&mut accumulators);
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_matvec_with_output", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                run_trinomial_matvec_into::<F, D, M, false>(
                    black_box(&domain),
                    black_box(&mut accumulators),
                    black_box(&zero),
                    black_box(&prepared),
                    black_box(&rhs_digits),
                    black_box(&lut),
                    black_box(&mut transformed_rhs),
                    black_box(&mut workspace),
                );
                for accumulator in &accumulators {
                    black_box(
                        domain.inverse_with_workspace(accumulator, black_box(&mut workspace)),
                    );
                }
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_matvec_with_output_packed", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                run_trinomial_matvec_into::<F, D, M, true>(
                    black_box(&domain),
                    black_box(&mut accumulators),
                    black_box(&zero),
                    black_box(&prepared),
                    black_box(&rhs_digits),
                    black_box(&lut),
                    black_box(&mut transformed_rhs),
                    black_box(&mut workspace),
                );
                for accumulator in &accumulators {
                    black_box(
                        domain.inverse_with_workspace(accumulator, black_box(&mut workspace)),
                    );
                }
            })
        },
    );
    group.finish();
}

fn benches(criterion: &mut Criterion) {
    bench_crt::<Prime64Offset23703, Q64_NUM_PRIMES, 128>(criterion, "p64_23703", Q64_PRIMES);
    bench_crt::<Prime64Offset23703, Q64_NUM_PRIMES, 256>(criterion, "p64_23703", Q64_PRIMES);
    bench_trinomial::<Prime64Offset23703, 162, PlusTrinomial>(criterion, "p64_23703", "plus");
    bench_trinomial::<Prime64Offset23703, 324, MinusTrinomial>(criterion, "p64_23703", "minus");
    bench_trinomial::<Prime64Offset23703, 648, MinusTrinomial>(criterion, "p64_23703", "minus");

    let q128 = q128_primes();
    bench_crt::<Prime128OffsetA7F7, Q128_NUM_PRIMES, 128>(criterion, "p128_a7f7", q128);
    bench_crt::<Prime128OffsetA7F7, Q128_NUM_PRIMES, 256>(criterion, "p128_a7f7", q128);
    bench_trinomial::<Prime128OffsetA7F7, 162, PlusTrinomial>(criterion, "p128_a7f7", "plus");
    bench_trinomial::<Prime128OffsetA7F7, 324, MinusTrinomial>(criterion, "p128_a7f7", "minus");
    bench_trinomial::<Prime128OffsetA7F7, 648, MinusTrinomial>(criterion, "p128_a7f7", "minus");

    bench_crt_matvec::<Prime64Offset23703, Q64_NUM_PRIMES, 256>(criterion, "p64_23703", Q64_PRIMES);
    bench_trinomial_matvec::<Prime64Offset23703, 162, PlusTrinomial>(
        criterion,
        "p64_23703",
        "plus",
    );
    bench_trinomial_matvec::<Prime64Offset23703, 324, MinusTrinomial>(
        criterion,
        "p64_23703",
        "minus",
    );
    bench_trinomial_matvec::<Prime64Offset23703, 648, MinusTrinomial>(
        criterion,
        "p64_23703",
        "minus",
    );

    bench_crt_matvec::<Prime128OffsetA7F7, Q128_NUM_PRIMES, 256>(criterion, "p128_a7f7", q128);
    bench_trinomial_matvec::<Prime128OffsetA7F7, 162, PlusTrinomial>(
        criterion,
        "p128_a7f7",
        "plus",
    );
    bench_trinomial_matvec::<Prime128OffsetA7F7, 324, MinusTrinomial>(
        criterion,
        "p128_a7f7",
        "minus",
    );
    bench_trinomial_matvec::<Prime128OffsetA7F7, 648, MinusTrinomial>(
        criterion,
        "p128_a7f7",
        "minus",
    );
    ifma::bench(criterion);
}

criterion_group!(ntt_comparison, benches);
criterion_main!(ntt_comparison);
