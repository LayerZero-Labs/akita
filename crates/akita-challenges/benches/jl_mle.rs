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
    eval_jl_mle_at_scalar, JlProjectionMatrix, DEFAULT_JL_ROWS,
};
use akita_field::{
    CanonicalBytes, CanonicalField, FieldCore, Fp64, FromPrimitiveInt, Prime128OffsetA7F7,
    Prime32Offset99, TranscriptChallenge,
};
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

type F = Prime128OffsetA7F7;
type F32 = Prime32Offset99;
type F64 = Fp64<4294967197>;

trait BenchField:
    FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + FromPrimitiveInt
{
}
impl<T> BenchField for T where
    T: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + FromPrimitiveInt
{
}

const N_ROWS: usize = DEFAULT_JL_ROWS;

const COLS_4K: usize = 1 << 12;
const COLS_16K: usize = 1 << 14;
const COLS_64K: usize = 1 << 16;
const COLS_128K: usize = 1 << 17;
const COLS_256K: usize = 1 << 18;

const COLS_SIZES: &[usize] = &[COLS_4K, COLS_16K, COLS_64K, COLS_128K, COLS_256K];

fn fresh_transcript<G: BenchField + 'static>() -> AkitaTranscript<G> {
    let mut t = AkitaTranscript::<G>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"bench-seed", &G::from_u64(0xC0FFEE));
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

fn challenge_points<G: BenchField>(cols: usize) -> (Vec<G>, Vec<G>) {
    let row_bits = N_ROWS.next_power_of_two().trailing_zeros() as usize;
    let col_bits = cols.next_power_of_two().trailing_zeros() as usize;
    let r_J: Vec<G> = (0..row_bits)
        .map(|i| G::from_u64(0xD00D_0000 + i as u64))
        .collect();
    let r_w: Vec<G> = (0..col_bits)
        .map(|i| G::from_u64(0xBEEF_0000 + i as u64))
        .collect();
    (r_J, r_w)
}

/// Row-major scalar vs production LUT eval for one field.
fn bench_eval_scalar_vs_lut<G: BenchField + 'static>(c: &mut Criterion, field_tag: &str) {
    let mut group = c.benchmark_group(format!("jl_mle_eval_at/{field_tag}"));
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript::<G>();
        let matrix =
            JlProjectionMatrix::sample::<G, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, r_w) = challenge_points::<G>(cols);

        group.throughput(Throughput::Elements((N_ROWS * cols) as u64));
        group.bench_with_input(BenchmarkId::new("scalar", cols), &cols, |b, _| {
            b.iter(|| {
                let val =
                    eval_jl_mle_at_scalar(black_box(&matrix), black_box(&r_J), black_box(&r_w))
                        .expect("eval_jl_mle_at_scalar");
                black_box(val)
            });
        });
        group.bench_with_input(BenchmarkId::new("lut", cols), &cols, |b, _| {
            b.iter(|| {
                let val = eval_jl_mle_at(black_box(&matrix), black_box(&r_J), black_box(&r_w))
                    .expect("eval_jl_mle_at");
                black_box(val)
            });
        });
    }
    group.finish();
}

fn bench_eval_fp32(c: &mut Criterion) {
    bench_eval_scalar_vs_lut::<F32>(c, "fp32");
}

fn bench_eval_fp64(c: &mut Criterion) {
    bench_eval_scalar_vs_lut::<F64>(c, "fp64");
}

fn bench_eval_fp128(c: &mut Criterion) {
    bench_eval_scalar_vs_lut::<F>(c, "fp128");
}

fn bench_build_jl_row_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_mle_row_weights");
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript::<F>();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, _) = challenge_points::<F>(cols);

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
        let mut tr = fresh_transcript::<F>();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, r_w) = challenge_points::<F>(cols);

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
        let mut tr = fresh_transcript::<F>();
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, _) = challenge_points::<F>(cols);

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
    bench_eval_fp32,
    bench_eval_fp64,
    bench_eval_fp128,
    bench_build_jl_row_weights,
    bench_eval_jl_mle_at_reference,
    bench_build_jl_row_weights_reference
);
criterion_main!(jl_mle);
