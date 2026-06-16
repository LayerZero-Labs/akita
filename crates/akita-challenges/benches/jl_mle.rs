//! Microbenchmarks for JL joint-matrix MLE evaluation (`akita-challenges::jl::mle`).
//!
//! Column counts match `benches/jl_projection.rs` (tail profile-bench geometry).
//! Eval and row-weight benches sweep the shipped claim/challenge fields:
//! `FpExt4<Prime32Offset99>` (fp32), `Ext2<Prime64Offset59>` (fp64), and
//! `Prime128OffsetA7F7` (fp128 base, no extension).
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
    CanonicalBytes, CanonicalField, Ext2, FieldCore, FpExt4, FromPrimitiveInt, Prime128OffsetA7F7,
    Prime32Offset99, Prime64Offset59, TranscriptChallenge,
};
use akita_transcript::labels::DOMAIN_AKITA_PROTOCOL;
use akita_transcript::{AkitaTranscript, Transcript};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

type F32Base = Prime32Offset99;
type F32Ext = FpExt4<F32Base>;
type F64Base = Prime64Offset59;
type F64Ext = Ext2<F64Base>;
type F128 = Prime128OffsetA7F7;

trait SampleField:
    FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + FromPrimitiveInt
{
}
impl<T> SampleField for T where
    T: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + FromPrimitiveInt
{
}

trait EvalField: FieldCore + FromPrimitiveInt {}
impl<T> EvalField for T where T: FieldCore + FromPrimitiveInt {}

const N_ROWS: usize = DEFAULT_JL_ROWS;

const COLS_4K: usize = 1 << 12;
const COLS_16K: usize = 1 << 14;
const COLS_64K: usize = 1 << 16;
const COLS_128K: usize = 1 << 17;
const COLS_256K: usize = 1 << 18;

const COLS_SIZES: &[usize] = &[COLS_4K, COLS_16K, COLS_64K, COLS_128K, COLS_256K];

fn fresh_transcript<B: SampleField + 'static>() -> AkitaTranscript<B> {
    let mut t = AkitaTranscript::<B>::new(DOMAIN_AKITA_PROTOCOL);
    t.append_field(b"bench-seed", &B::from_u64(0xC0FFEE));
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

fn challenge_points<G: EvalField>(cols: usize) -> (Vec<G>, Vec<G>) {
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

/// Row-major scalar vs production LUT eval for one eval field.
fn bench_eval_scalar_vs_lut<G: EvalField + 'static, B: SampleField + 'static>(
    c: &mut Criterion,
    field_tag: &str,
) {
    let mut group = c.benchmark_group(format!("jl_mle_eval_at/{field_tag}"));
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript::<B>();
        let matrix =
            JlProjectionMatrix::sample::<B, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
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

fn bench_build_jl_row_weights_for<G: EvalField + 'static, B: SampleField + 'static>(
    c: &mut Criterion,
    field_tag: &str,
) {
    let mut group = c.benchmark_group(format!("jl_mle_row_weights/{field_tag}"));
    for &cols in COLS_SIZES {
        group.sample_size(sample_size_for_cols(cols));
        let mut tr = fresh_transcript::<B>();
        let matrix =
            JlProjectionMatrix::sample::<B, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, _) = challenge_points::<G>(cols);

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

fn bench_eval_fp32(c: &mut Criterion) {
    bench_eval_scalar_vs_lut::<F32Ext, F32Base>(c, "fp32_ext4");
}

fn bench_eval_fp64(c: &mut Criterion) {
    bench_eval_scalar_vs_lut::<F64Ext, F64Base>(c, "fp64_ext2");
}

fn bench_eval_fp128(c: &mut Criterion) {
    bench_eval_scalar_vs_lut::<F128, F128>(c, "fp128");
}

fn bench_row_weights_fp32(c: &mut Criterion) {
    bench_build_jl_row_weights_for::<F32Ext, F32Base>(c, "fp32_ext4");
}

fn bench_row_weights_fp64(c: &mut Criterion) {
    bench_build_jl_row_weights_for::<F64Ext, F64Base>(c, "fp64_ext2");
}

fn bench_row_weights_fp128(c: &mut Criterion) {
    bench_build_jl_row_weights_for::<F128, F128>(c, "fp128");
}

fn bench_eval_jl_mle_at_reference(c: &mut Criterion) {
    let mut group = c.benchmark_group("jl_mle_eval_reference/fp128");
    // Reference is Θ(n_rows·cols) with high constant; keep smaller cols only.
    for &cols in &[COLS_4K, COLS_16K] {
        group.sample_size(50);
        let mut tr = fresh_transcript::<F128>();
        let matrix =
            JlProjectionMatrix::sample::<F128, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, r_w) = challenge_points::<F128>(cols);

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
    let mut group = c.benchmark_group("jl_mle_row_weights_reference/fp128");
    for &cols in &[COLS_4K, COLS_16K] {
        group.sample_size(50);
        let mut tr = fresh_transcript::<F128>();
        let matrix =
            JlProjectionMatrix::sample::<F128, _>(&mut tr, N_ROWS, cols).expect("setup matrix");
        let (r_J, _) = challenge_points::<F128>(cols);

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
    bench_row_weights_fp32,
    bench_row_weights_fp64,
    bench_row_weights_fp128,
    bench_eval_jl_mle_at_reference,
    bench_build_jl_row_weights_reference
);
criterion_main!(jl_mle);
