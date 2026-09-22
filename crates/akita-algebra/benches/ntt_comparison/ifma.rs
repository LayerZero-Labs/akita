//! Native IFMA52 comparison using the existing prepared-matrix implementation.

pub(super) fn bench(criterion: &mut criterion::Criterion) {
    #[cfg(target_arch = "x86_64")]
    if akita_algebra::ntt::ifma52::ifma52_enabled() {
        use akita_algebra::ntt::ifma52::IFMA52_PRIMES;
        bench_profile::<super::Prime64Offset23703, 2>(
            criterion,
            "p64_23703",
            [IFMA52_PRIMES[0], IFMA52_PRIMES[1]],
        );
        bench_profile::<super::Prime128OffsetA7F7, 3>(criterion, "p128_a7f7", IFMA52_PRIMES);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = criterion;
}

#[cfg(target_arch = "x86_64")]
fn bench_profile<F: super::CrtNttConvertibleField + std::fmt::Debug, const K: usize>(
    criterion: &mut criterion::Criterion,
    field: &str,
    primes: [u64; K],
) {
    use std::hint::black_box;

    use akita_algebra::{CyclotomicRing, Ifma52NttMatrix, Ifma52Params};
    use criterion::BenchmarkId;

    use super::{configure, full_field_coefficients, matvec_digits, MATVEC_COLS, MATVEC_ROWS};

    const D: usize = 256;
    let params = Ifma52Params::<K, D>::new(primes).expect("supported IFMA52 parameters");
    assert!(params.crt_capacity().supports::<F, D>(MATVEC_COLS, 8));
    let matrix: Vec<CyclotomicRing<F, D>> = (0..MATVEC_ROWS * MATVEC_COLS)
        .map(|entry| CyclotomicRing::from_coefficients(full_field_coefficients(101 + entry as u64)))
        .collect();
    let rhs = matvec_digits::<D>().map(|digits| digits.map(i16::from));
    let prepared = Ifma52NttMatrix::prepare(&matrix, &params);
    let expected: Vec<_> = (0..MATVEC_ROWS)
        .map(|row| {
            let mut sum = CyclotomicRing::zero();
            for (column, digits) in rhs.iter().enumerate() {
                let ring = CyclotomicRing::from_coefficients(
                    digits.map(|digit| F::from_i64(i64::from(digit))),
                );
                matrix[row * MATVEC_COLS + column].mul_accumulate_into(&ring, &mut sum);
            }
            sum
        })
        .collect();
    assert_eq!(
        prepared.mat_vec_i16::<F>(MATVEC_ROWS, &rhs).unwrap(),
        expected
    );
    eprintln!(
        "IFMA52 {field} D={D}: prepared={} B/coeff, matrix={} B; native API includes temporary/output allocations",
        K * core::mem::size_of::<u64>(),
        prepared.cache_bytes(),
    );

    let label = format!("ifma52_{field}_negacyclic/D={D}");
    let mut group = criterion.benchmark_group("ntt_matvec_comparison");
    configure(&mut group, D * MATVEC_ROWS * MATVEC_COLS);
    group.bench_with_input(
        BenchmarkId::new("prepared_matrix_setup", &label),
        &label,
        |bench, _| bench.iter(|| black_box(Ifma52NttMatrix::prepare(black_box(&matrix), &params))),
    );
    group.bench_with_input(
        BenchmarkId::new("prepared_matvec_with_output", &label),
        &label,
        |bench, _| {
            bench.iter(|| {
                black_box(
                    black_box(&prepared)
                        .mat_vec_i16::<F>(MATVEC_ROWS, black_box(&rhs))
                        .unwrap(),
                )
            })
        },
    );
    group.finish();
}
