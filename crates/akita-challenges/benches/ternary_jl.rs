use akita_algebra::{
    jl::{
        build_ternary_column_weights, eval_ternary_matrix_mle,
        eval_ternary_matrix_mle_from_eq_tables, TernaryProjectionMatrix, TernaryProjectionShape,
    },
    EqPolynomial,
};
use akita_challenges::expand_balanced_ternary_matrix;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::{Ext2, Field, Fp64, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use std::hint::black_box;

type BaselineF64 = Fp64<4294967197>;
type ProductionF32Ext = FpExt4<Prime32Offset99>;
type ProductionF64Ext = Ext2<Prime64Offset59>;
type ProductionF128 = Prime128OffsetA7F7;

const DEFAULT_LOG_WIDTHS: &str = "12,13,14,15,16";
const MAX_LOG_WIDTH: u32 = 16;

fn benchmark_log_widths() -> Vec<u32> {
    let configured = std::env::var("AKITA_JL_BENCH_LOG_WIDTHS")
        .unwrap_or_else(|_| DEFAULT_LOG_WIDTHS.to_owned());
    let mut widths: Vec<u32> = configured
        .split(',')
        .map(str::trim)
        .filter(|width| !width.is_empty())
        .map(|width| {
            width
                .parse::<u32>()
                .unwrap_or_else(|error| panic!("invalid JL benchmark log-width `{width}`: {error}"))
        })
        .collect();
    assert!(
        !widths.is_empty(),
        "at least one JL benchmark log-width is required"
    );
    assert!(
        widths.iter().all(|&width| width <= MAX_LOG_WIDTH),
        "JL benchmark log-widths must not exceed {MAX_LOG_WIDTH}"
    );
    widths.sort_unstable();
    widths.dedup();
    widths
}

fn benchmark_ternary_jl(c: &mut Criterion) {
    let mut group = c.benchmark_group("balanced_ternary_jl");
    for log_width in benchmark_log_widths() {
        let cols = 1usize << log_width;
        let shape = TernaryProjectionShape::new(256, cols).unwrap();
        let matrix = expand_balanced_ternary_matrix(&[0x42u8; 32], shape).unwrap();
        let input_i8: Vec<i8> = (0..shape.cols())
            .map(|index| (index % 31) as i8 - 15)
            .collect();
        let input_i16: Vec<i16> = input_i8.iter().copied().map(i16::from).collect();
        let input_i32: Vec<i32> = input_i8.iter().copied().map(i32::from).collect();
        let input_i64: Vec<i64> = input_i8.iter().copied().map(i64::from).collect();
        let input_i128: Vec<i128> = input_i8.iter().copied().map(i128::from).collect();
        let input_i8_blocks = input_i8.repeat(8);
        group.throughput(Throughput::Elements((shape.rows() * shape.cols()) as u64));

        group.bench_with_input(
            BenchmarkId::new("expand", cols),
            &shape,
            |bencher, &shape| {
                bencher.iter(|| {
                    expand_balanced_ternary_matrix(black_box(&[0x42u8; 32]), shape).unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("expand_project_i32_cold", cols),
            &input_i32,
            |bencher, input| {
                bencher.iter(|| {
                    expand_balanced_ternary_matrix(black_box(&[0x42u8; 32]), shape)
                        .unwrap()
                        .project(black_box(input))
                        .unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("project_i8", cols),
            &input_i8,
            |bencher, input| bencher.iter(|| matrix.project(black_box(input)).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("project_i8_8blocks", cols),
            &input_i8_blocks,
            |bencher, input| bencher.iter(|| matrix.project_blocks(black_box(input)).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("project_i16", cols),
            &input_i16,
            |bencher, input| bencher.iter(|| matrix.project(black_box(input)).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("project_i32", cols),
            &input_i32,
            |bencher, input| bencher.iter(|| matrix.project(black_box(input)).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("project_i64", cols),
            &input_i64,
            |bencher, input| bencher.iter(|| matrix.project(black_box(input)).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("project_i128", cols),
            &input_i128,
            |bencher, input| bencher.iter(|| matrix.project_i128(black_box(input)).unwrap()),
        );
    }
    group.finish();
}

fn challenge_point<F: Field>(num_vars: usize, seed: u64) -> Vec<F> {
    (0..num_vars)
        .map(|index| F::from_u64(seed.wrapping_add(index as u64 * 0x9e37_79b9)))
        .collect()
}

fn mle_sample_size(cols: usize) -> usize {
    if cols >= 1 << 16 {
        10
    } else if cols >= 1 << 14 {
        20
    } else {
        30
    }
}

struct MleCase<F> {
    cols: usize,
    matrix: TernaryProjectionMatrix,
    row_point: Vec<F>,
    col_point: Vec<F>,
    row_eq: Vec<F>,
    col_eq: Vec<F>,
}

fn benchmark_mle_field<F: Field + 'static>(c: &mut Criterion, field: &str) {
    let cases: Vec<MleCase<F>> = benchmark_log_widths()
        .into_iter()
        .map(|log_width| {
            let cols = 1usize << log_width;
            let shape = TernaryProjectionShape::new(256, cols).unwrap();
            let matrix = expand_balanced_ternary_matrix(&[0x42u8; 32], shape).unwrap();
            let row_point = challenge_point::<F>(shape.row_num_vars().unwrap(), 0xd00d_0000);
            let col_point = challenge_point::<F>(shape.col_num_vars().unwrap(), 0xbeef_0000);
            let row_eq = EqPolynomial::evals(&row_point).unwrap();
            let col_eq = EqPolynomial::evals(&col_point).unwrap();
            MleCase {
                cols,
                matrix,
                row_point,
                col_point,
                row_eq,
                col_eq,
            }
        })
        .collect();

    let mut kernel_group = c.benchmark_group(format!("balanced_ternary_jl_mle_kernel/{field}"));
    for case in &cases {
        let cols = case.cols;
        let elements = (256 * cols) as u64;
        let sample_size = mle_sample_size(cols);
        kernel_group.sample_size(sample_size);
        kernel_group.throughput(Throughput::Elements(elements));
        kernel_group.bench_with_input(BenchmarkId::new("cached_eq", cols), &cols, |bencher, _| {
            bencher.iter(|| {
                black_box(
                    eval_ternary_matrix_mle_from_eq_tables(
                        black_box(&case.matrix),
                        black_box(&case.row_eq),
                        black_box(&case.col_eq),
                    )
                    .unwrap(),
                )
            })
        });
    }
    kernel_group.finish();

    let mut e2e_group = c.benchmark_group(format!("balanced_ternary_jl_mle_e2e/{field}"));
    for case in &cases {
        let cols = case.cols;
        let elements = (256 * cols) as u64;
        let sample_size = mle_sample_size(cols);
        e2e_group.sample_size(sample_size);
        e2e_group.throughput(Throughput::Elements(elements));
        e2e_group.bench_with_input(
            BenchmarkId::new("with_eq_tables", cols),
            &cols,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(
                        eval_ternary_matrix_mle(
                            black_box(&case.matrix),
                            black_box(&case.row_point),
                            black_box(&case.col_point),
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }
    e2e_group.finish();

    let mut column_group =
        c.benchmark_group(format!("balanced_ternary_jl_column_weights_e2e/{field}"));
    for case in &cases {
        let cols = case.cols;
        let elements = (256 * cols) as u64;
        let sample_size = mle_sample_size(cols);
        column_group.sample_size(sample_size);
        column_group.throughput(Throughput::Elements(elements));
        column_group.bench_with_input(BenchmarkId::new("from_point", cols), &cols, |bencher, _| {
            bencher.iter(|| {
                black_box(
                    build_ternary_column_weights(
                        black_box(&case.matrix),
                        black_box(&case.row_point),
                    )
                    .unwrap(),
                )
            })
        });
    }
    column_group.finish();
}

fn benchmark_mle(c: &mut Criterion) {
    benchmark_mle_field::<BaselineF64>(c, "fp64_baseline");
    benchmark_mle_field::<ProductionF32Ext>(c, "fp32_ext4");
    benchmark_mle_field::<ProductionF64Ext>(c, "fp64_ext2");
    benchmark_mle_field::<ProductionF128>(c, "fp128");
}

criterion_group!(benches, benchmark_ternary_jl, benchmark_mle);
criterion_main!(benches);
