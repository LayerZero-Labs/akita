#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hachi_pcs::algebra::fields::Prime128Offset5823;
use hachi_pcs::algebra::{Pow2Offset32Field, Pow2Offset64Field};
use hachi_pcs::protocol::labrador::aggregation::aggregate_jl_contraints_one_lift;
use hachi_pcs::protocol::labrador::LabradorJlMatrix;
use hachi_pcs::protocol::transcript::{labels, Blake2bTranscript};
use hachi_pcs::{CanonicalField, FieldCore, Transcript};

const D: usize = 64;
// Observed in realistic profile runs for NV=25 full mode.
const BENCH_COLS: usize = 4_128_768;

fn sample_omega_from_transcript<F, T>(transcript: &mut T) -> [F; 256]
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    std::array::from_fn(|_| transcript.challenge_scalar(labels::CHALLENGE_LABRADOR_JL_COLLAPSE))
}

fn bench_aggregate_jl_contraints_one_lift_for_field<F: FieldCore + CanonicalField + 'static>(
    c: &mut Criterion,
    field_name: &str,
) {
    let cols = BENCH_COLS;
    let mut transcript = Blake2bTranscript::<F>::new(b"bench/labrador-jl-aggregation");
    let matrix = LabradorJlMatrix::generate::<F, _>(&mut transcript, cols).unwrap();
    let omega = sample_omega_from_transcript::<F, _>(&mut transcript);
    c.bench_function(
        &format!("labrador/aggregate_jl_contraints_one_lift/{field_name}"),
        |b| {
            b.iter(|| {
                let got =
                    aggregate_jl_contraints_one_lift::<F, D>(black_box(&matrix), black_box(&omega))
                        .unwrap();
                black_box(got);
            })
        },
    );
}

fn bench_aggregate_jl_contraints_one_lift_fp32(c: &mut Criterion) {
    bench_aggregate_jl_contraints_one_lift_for_field::<Pow2Offset32Field>(c, "fp32");
}

fn bench_aggregate_jl_contraints_one_lift_fp64(c: &mut Criterion) {
    type F64 = Pow2Offset64Field;
    bench_aggregate_jl_contraints_one_lift_for_field::<F64>(c, "fp64");
}

fn bench_aggregate_jl_contraints_one_lift_fp128(c: &mut Criterion) {
    type F128 = Prime128Offset5823;
    bench_aggregate_jl_contraints_one_lift_for_field::<F128>(c, "fp128");
}

criterion_group!(
    labrador_jl_aggregation,
    bench_aggregate_jl_contraints_one_lift_fp32,
    bench_aggregate_jl_contraints_one_lift_fp64,
    bench_aggregate_jl_contraints_one_lift_fp128
);
criterion_main!(labrador_jl_aggregation);
