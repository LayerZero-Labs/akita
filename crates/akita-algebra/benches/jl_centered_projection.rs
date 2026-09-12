#![allow(missing_docs)]

use std::{hint::black_box, time::Duration};

use akita_algebra::jl::{base_field_modulus, TernaryProjectionMatrix, TernaryProjectionShape};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use jolt_field::{CanonicalEncoding, Field, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const ROWS: usize = 256;
const COLS: usize = 1 << 14;

fn fixture() -> TernaryProjectionMatrix {
    let shape = TernaryProjectionShape::new(ROWS, COLS).unwrap();
    let mut rng = StdRng::seed_from_u64(0xacca_71a5_5eed);
    let mut first = vec![0u8; shape.plane_len()];
    let mut second = vec![0u8; shape.plane_len()];
    rng.fill_bytes(&mut first);
    rng.fill_bytes(&mut second);
    TernaryProjectionMatrix::from_rademacher_bitplanes(shape, first, second).unwrap()
}

fn center<F: Field + CanonicalEncoding>(value: F, modulus: u128) -> i128 {
    let residue = value.to_u128_checked().unwrap();
    if residue <= modulus / 2 {
        i128::try_from(residue).unwrap()
    } else {
        -i128::try_from(modulus - residue).unwrap()
    }
}

fn field_reference<F: Field + CanonicalEncoding>(
    matrix: &TernaryProjectionMatrix,
    input: &[i128],
) -> Vec<i128> {
    let embedded = input.iter().copied().map(F::from_i128).collect::<Vec<_>>();
    let modulus = base_field_modulus::<F>().unwrap();
    matrix
        .project_field_blocks(&embedded)
        .unwrap()
        .into_iter()
        .map(|value| center(value, modulus))
        .collect()
}

fn centered_input(kind: &str, len: usize) -> Vec<i128> {
    let scale = match kind {
        "i8" => 1,
        "i16" => 257,
        "i32" => 100_003,
        _ => unreachable!("benchmark input kind is fixed"),
    };
    (0..len)
        .map(|index| ((index % 127) as i128 - 63) * scale)
        .collect()
}

fn bench_field<F: Field + CanonicalEncoding + std::fmt::Debug + 'static>(
    c: &mut Criterion,
    field: &str,
) {
    let matrix = fixture();
    matrix.project_i128(&centered_input("i8", COLS)).unwrap();
    let mut group = c.benchmark_group(format!("jl_centered_projection/{field}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for kind in ["i8", "i16", "i32"] {
        for blocks in [1usize, 8] {
            let input = centered_input(kind, blocks * COLS);
            let expected = field_reference::<F>(&matrix, &input);
            assert_eq!(
                matrix.project_centered_i128_blocks::<F>(&input).unwrap(),
                expected
            );
            group.throughput(Throughput::Elements((blocks * ROWS * COLS) as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("dynamic_narrow_hot_{kind}"), blocks),
                &input,
                |b, input| {
                    b.iter(|| {
                        black_box(
                            matrix
                                .project_centered_i128_blocks::<F>(black_box(input))
                                .unwrap(),
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("field_reference_hot_{kind}"), blocks),
                &input,
                |b, input| {
                    b.iter(|| black_box(field_reference::<F>(black_box(&matrix), black_box(input))))
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("dynamic_narrow_cold_{kind}"), blocks),
                &input,
                |b, input| {
                    b.iter_batched_ref(
                        || matrix.clone(),
                        |cold| {
                            black_box(
                                cold.project_centered_i128_blocks::<F>(black_box(input))
                                    .unwrap(),
                            )
                        },
                        BatchSize::PerIteration,
                    )
                },
            );
        }
    }
    group.finish();
}

fn benchmarks(c: &mut Criterion) {
    bench_field::<Prime32Offset99>(c, "fp32");
    bench_field::<Prime64Offset59>(c, "fp64");
    bench_field::<Prime128OffsetA7F7>(c, "fp128");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
