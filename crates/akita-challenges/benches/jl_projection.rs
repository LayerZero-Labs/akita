//! Microbenchmarks for the standalone JL projection prototype (`akita-challenges::jl`).
//!
//! Measures matrix sampling (`JlProjectionMatrix::sample`), integer projection
//! (`project` on small balanced digits), and the squared-norm check
//! (`JlImage::l2_norm_sq_checked`). The hot path is `project` at
//! `O(n_rows * cols)`; matrix sampling is one-time per JL level.
//!
//! Column counts bracket `N_coeff = (# ring elements) * D` at representative
//! tail-adjacent witness sizes for `D = 64`.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p akita-challenges --bench jl_projection --features parallel -- --noplot
//! ```

#![allow(missing_docs)]

use akita_challenges::{center_coefficients, JlProjectionMatrix, DEFAULT_JL_ROWS};
use akita_field::Prime128OffsetA7F7;
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

type F = Prime128OffsetA7F7;

const D: usize = 64;
const N_ROWS: usize = DEFAULT_JL_ROWS;

/// `cols = rings * D` for `rings` in `{1, 8, 32, 128, 512}`.
const COLS_SIZES: &[usize] = &[D, 8 * D, 32 * D, 128 * D, 512 * D];

fn fresh_transcript() -> AkitaTranscript<F> {
    let mut t = AkitaTranscript::<F>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"bench-seed", &F::from_u64(0xC0FFEE));
    t
}

/// Small balanced digits in `[-16, 16]`, the realistic JL input shape.
fn digit_coeffs(cols: usize) -> Vec<F> {
    (0..cols)
        .map(|i| F::from_i64(((i % 33) as i64) - 16))
        .collect()
}

fn bench_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_sample");
    for &cols in COLS_SIZES {
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

fn bench_project_centered(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_project_centered");
    for &cols in COLS_SIZES {
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let coeffs = digit_coeffs(cols);
        let centered = center_coefficients(&coeffs).expect("center coeffs");

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let image = matrix
                    .project_centered(black_box(&centered))
                    .expect("JL project centered");
                black_box(image)
            });
        });
    }
    group.finish();
}

fn bench_l2_norm(c: &mut Criterion) {
    let cols = 32 * D;
    let mut tr = fresh_transcript();
    let matrix = JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
    let coeffs = digit_coeffs(cols);
    let image = matrix.project(&coeffs).expect("setup image");

    let mut group = c.benchmark_group("jl_l2_norm_sq");
    group.throughput(Throughput::Elements(N_ROWS as u64));
    group.bench_function("n256_cols2048", |b| {
        b.iter(|| black_box(image.l2_norm_sq_checked().expect("l2 norm")));
    });
    group.finish();
}

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_end_to_end");
    for &cols in COLS_SIZES {
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
    bench_project_centered,
    bench_l2_norm,
    bench_end_to_end
);
criterion_main!(jl_projection);
