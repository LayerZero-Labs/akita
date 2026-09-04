use akita_algebra::jl::TernaryProjectionShape;
use akita_challenges::expand_balanced_ternary_matrix;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_ternary_jl(c: &mut Criterion) {
    let shape = TernaryProjectionShape::new(256, 4096).unwrap();
    let matrix = expand_balanced_ternary_matrix(&[0x42u8; 32], shape).unwrap();
    let input_i8: Vec<i8> = (0..shape.cols())
        .map(|index| (index % 31) as i8 - 15)
        .collect();
    let input_i16: Vec<i16> = input_i8.iter().copied().map(i16::from).collect();
    let input_i32: Vec<i32> = input_i8.iter().copied().map(i32::from).collect();
    let input_i64: Vec<i64> = input_i8.iter().copied().map(i64::from).collect();
    let input_i128: Vec<i128> = input_i8.iter().copied().map(i128::from).collect();
    let input_i8_blocks = input_i8.repeat(8);

    c.bench_function("balanced_ternary_expand_256x4096", |bencher| {
        bencher.iter(|| expand_balanced_ternary_matrix(black_box(&[0x42u8; 32]), shape).unwrap())
    });
    c.bench_function("balanced_ternary_project_i8_256x4096", |bencher| {
        bencher.iter(|| matrix.project(black_box(&input_i8)).unwrap())
    });
    c.bench_function("balanced_ternary_project_i8_8blocks_256x4096", |bencher| {
        bencher.iter(|| matrix.project_blocks(black_box(&input_i8_blocks)).unwrap())
    });
    c.bench_function("balanced_ternary_project_i16_256x4096", |bencher| {
        bencher.iter(|| matrix.project(black_box(&input_i16)).unwrap())
    });
    c.bench_function("balanced_ternary_project_i32_256x4096", |bencher| {
        bencher.iter(|| matrix.project(black_box(&input_i32)).unwrap())
    });
    c.bench_function("balanced_ternary_project_i64_256x4096", |bencher| {
        bencher.iter(|| matrix.project(black_box(&input_i64)).unwrap())
    });
    c.bench_function("balanced_ternary_project_i128_256x4096", |bencher| {
        bencher.iter(|| matrix.project_i128(black_box(&input_i128)).unwrap())
    });
}

criterion_group!(benches, benchmark_ternary_jl);
criterion_main!(benches);
