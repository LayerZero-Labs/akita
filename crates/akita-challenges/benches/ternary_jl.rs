use akita_algebra::jl::TernaryProjectionShape;
use akita_challenges::expand_balanced_ternary_matrix;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

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

criterion_group!(benches, benchmark_ternary_jl);
criterion_main!(benches);
