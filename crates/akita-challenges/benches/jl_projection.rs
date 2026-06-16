//! Microbenchmarks for the standalone JL projection prototype (`akita-challenges::jl`).
//!
//! Measures matrix sampling (`JlProjectionMatrix::sample`), integer projection
//! (`project` on small balanced digits), fast vs reference kernels (scalar,
//! NEON, AVX2, AVX-512), and the squared-norm check (`JlImage::l2_norm_sq_checked`).
//!
//! Column counts are powers of two anchored to CI profile-bench tail geometry
//! (`profile-bench-data`, `scripts/profile_bench_report.py`). See
//! `planned_levels[].next_w_ring * D` and `tail_*_field_elems` in each case's
//! `summary.json` (e.g. `fp128-onehot-nv32-np1-d64`):
//!
//! | bench `cols` | anchor (fp128 onehot nv=32, D=64)        |
//! |-------------|-------------------------------------------|
//! | 2^12 = 4096 | e+t+r field coeffs (2944)                 |
//! | 2^14 = 16384| z Golomb coords (17600)                   |
//! | 2^16 = 65536| bracket between z and full witness        |
//! | 2^17 = 131072 | terminal `next_w_ring * D` (129344)   |
//! | 2^18 = 262144 | penultimate fold / level-4 witness    |
//!
//! Run with:
//!
//! ```text
//! cargo bench -p akita-challenges --bench jl_projection --features parallel,jl-simd -- --noplot
//! ```

#![allow(missing_docs)]

use akita_challenges::{center_coefficients, JlProjectionMatrix, DEFAULT_JL_ROWS};
use akita_field::Prime128OffsetA7F7;
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

type F = Prime128OffsetA7F7;

const N_ROWS: usize = DEFAULT_JL_ROWS;

const COLS_4K: usize = 1 << 12;
const COLS_16K: usize = 1 << 14;
const COLS_64K: usize = 1 << 16;
const COLS_128K: usize = 1 << 17;
const COLS_256K: usize = 1 << 18;

const COLS_SIZES: &[usize] = &[COLS_4K, COLS_16K, COLS_64K, COLS_128K, COLS_256K];

fn fresh_transcript() -> AkitaTranscript<F> {
    let mut t = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"bench-seed", &F::from_u64(0xC0FFEE));
    t
}

fn sample_size_for_cols(cols: usize) -> usize {
    if cols >= COLS_256K {
        10
    } else if cols >= COLS_128K {
        20
    } else if cols >= COLS_64K {
        50
    } else {
        100
    }
}

fn digit_coeffs(cols: usize) -> Vec<F> {
    (0..cols)
        .map(|i| F::from_i64(((i % 33) as i64) - 16))
        .collect()
}

fn bench_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_sample");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, &cols| {
            b.iter(|| {
                let mut tr = fresh_transcript();
                let matrix =
                    JlProjectionMatrix::sample::<F, _>(&mut tr, black_box(N_ROWS), black_box(cols))
                        .expect("JL matrix sample");
                black_box(matrix)
            });
        });
    }
    group.finish();
}

fn bench_project(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_project");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let coeffs = digit_coeffs(cols);

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let image = matrix.project(black_box(&coeffs)).expect("JL project");
                black_box(image)
            });
        });
    }
    group.finish();
}

fn bench_project_digits(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_project_digits");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let coeffs = digit_coeffs(cols);
        let digits = center_coefficients(&coeffs).expect("center coeffs");

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let image = matrix
                    .project_digits(black_box(&digits))
                    .expect("JL project digits");
                black_box(image)
            });
        });
    }
    group.finish();
}

fn bench_project_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_project_reference");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let coeffs = digit_coeffs(cols);
        let digits = center_coefficients(&coeffs).expect("center coeffs");

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let image = matrix
                    .project_digits_reference(black_box(&digits))
                    .expect("JL project reference");
                black_box(image)
            });
        });
    }
    group.finish();
}

fn bench_l2_norm(c: &mut Criterion) {
    let cols = COLS_128K;
    let mut tr = fresh_transcript();
    let matrix = JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
    let coeffs = digit_coeffs(cols);
    let image = matrix.project(&coeffs).expect("setup image");

    let mut group = c.benchmark_group("jl_l2_norm_sq");
    group.throughput(Throughput::Elements(N_ROWS as u64));
    group.bench_function(BenchmarkId::new("n256", cols), |b| {
        b.iter(|| black_box(image.l2_norm_sq_checked().expect("l2 norm")));
    });
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_end_to_end");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let coeffs = digit_coeffs(cols);
        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, &cols| {
            b.iter(|| {
                let mut tr = fresh_transcript();
                let matrix =
                    JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("sample");
                let image = matrix.project(black_box(&coeffs)).expect("project");
                let norm_sq = image.l2_norm_sq_checked().expect("norm");
                black_box(norm_sq)
            });
        });
    }
    group.finish();
}

criterion_group!(
    jl_projection,
    bench_sample,
    bench_project,
    bench_project_digits,
    bench_project_reference,
    bench_l2_norm,
    bench_end_to_end
);
criterion_main!(jl_projection);
