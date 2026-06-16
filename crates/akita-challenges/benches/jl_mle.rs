//! Microbenchmarks for JL joint-matrix MLE evaluation (`akita-challenges::jl::mle`).
//!
//! Column counts match `benches/jl_projection.rs` (tail profile-bench geometry).
//!
//! ```text
//! cargo bench -p akita-challenges --bench jl_mle --features parallel -- --noplot
//! ```

#![allow(missing_docs, non_snake_case)]

use akita_challenges::{
    build_jl_row_weights, build_jl_row_weights_reference, eval_jl_mle_at, eval_jl_mle_at_reference,
    JlProjectionMatrix, DEFAULT_JL_ROWS,
};
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

fn challenge_points(cols: usize) -> (Vec<F>, Vec<F>) {
    let row_bits = N_ROWS.next_power_of_two().trailing_zeros() as usize;
    let col_bits = cols.next_power_of_two().trailing_zeros() as usize;
    let r_J: Vec<F> = (0..row_bits)
        .map(|i| F::from_u64(0xD00D_0000 + i as u64))
        .collect();
    let r_w: Vec<F> = (0..col_bits)
        .map(|i| F::from_u64(0xBEEF_0000 + i as u64))
        .collect();
    (r_J, r_w)
}

fn bench_eval_jl_mle_at(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_mle_eval_at");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, r_w) = challenge_points(cols);

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let val = eval_jl_mle_at(black_box(&matrix), black_box(&r_J), black_box(&r_w))
                    .expect("eval_jl_mle_at");
                black_box(val)
            });
        });
    }
    group.finish();
}

fn bench_build_jl_row_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_mle_row_weights");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, _) = challenge_points(cols);

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let g = build_jl_row_weights(black_box(&matrix), black_box(&r_J))
                    .expect("build_jl_row_weights");
                black_box(g)
            });
        });
    }
    group.finish();
}

fn bench_eval_jl_mle_at_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_mle_eval_reference");
    // Reference is Θ(n_rows·cols) with high constant; keep smaller cols only.
    for &cols in &[COLS_4K, COLS_16K] {
        group.sample_size(50);
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, r_w) = challenge_points(cols);

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let val =
                    eval_jl_mle_at_reference(black_box(&matrix), black_box(&r_J), black_box(&r_w))
                        .expect("eval reference");
                black_box(val)
            });
        });
    }
    group.finish();
}

fn bench_build_jl_row_weights_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_mle_row_weights_reference");
    for &cols in &[COLS_4K, COLS_16K] {
        group.sample_size(50);
        let mut tr = fresh_transcript();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, _) = challenge_points(cols);

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("n256", cols), &cols, |b, _| {
            b.iter(|| {
                let g = build_jl_row_weights_reference(black_box(&matrix), black_box(&r_J))
                    .expect("build reference");
                black_box(g)
            });
        });
    }
    group.finish();
}

criterion_group!(
    jl_mle,
    bench_eval_jl_mle_at,
    bench_build_jl_row_weights,
    bench_eval_jl_mle_at_reference,
    bench_build_jl_row_weights_reference
);
criterion_main!(jl_mle);
