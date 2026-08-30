#![allow(missing_docs)]

use akita_types::CommitmentRingDims;
use akita_verifier::{
    relation_evaluator_benchmark_case, relation_evaluator_benchmark_case_with_chunks,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_field::Prime128OffsetA7F7;
use std::time::Duration;

type F = Prime128OffsetA7F7;
const D: usize = 128;

fn bench_relation_evaluator(c: &mut Criterion) {
    let mut group = c.benchmark_group("relation_evaluator");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for (cell, role_dims, outgoing_ring_dimension) in [
        ("U", CommitmentRingDims::uniform(D), 128),
        ("L", CommitmentRingDims::uniform(D), 32),
        (
            "M",
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            64,
        ),
    ] {
        let case = relation_evaluator_benchmark_case(role_dims, outgoing_ring_dimension)
            .expect("valid relation benchmark case");
        group.bench_with_input(
            BenchmarkId::new(cell, "direct"),
            &case,
            |b, benchmark_case| {
                b.iter(|| {
                    black_box(
                        benchmark_case
                            .evaluator
                            .eval_flat_at_point::<F>(
                                black_box(&benchmark_case.point),
                                black_box(&benchmark_case.setup),
                                black_box(benchmark_case.alpha),
                            )
                            .expect("relation evaluation"),
                    )
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new(cell, "deferred"),
            &case,
            |b, benchmark_case| {
                b.iter(|| {
                    black_box(
                        benchmark_case
                            .evaluator
                            .eval_flat_at_point_with_deferred_setup::<F>(
                                black_box(&benchmark_case.point),
                                black_box(&benchmark_case.setup),
                                black_box(benchmark_case.alpha),
                                black_box(F::one()),
                            )
                            .expect("deferred relation evaluation"),
                    )
                });
            },
        );
    }

    let multi_chunk =
        relation_evaluator_benchmark_case_with_chunks(CommitmentRingDims::uniform(D), D, 8)
            .expect("valid multi-chunk relation benchmark case");
    for deferred in [false, true] {
        let mode = if deferred { "deferred" } else { "direct" };
        group.bench_with_input(
            BenchmarkId::new("U-8chunks", mode),
            &multi_chunk,
            |b, benchmark_case| {
                b.iter(|| {
                    let result = if deferred {
                        benchmark_case
                            .evaluator
                            .eval_flat_at_point_with_deferred_setup::<F>(
                                black_box(&benchmark_case.point),
                                black_box(&benchmark_case.setup),
                                black_box(benchmark_case.alpha),
                                black_box(F::one()),
                            )
                    } else {
                        benchmark_case.evaluator.eval_flat_at_point::<F>(
                            black_box(&benchmark_case.point),
                            black_box(&benchmark_case.setup),
                            black_box(benchmark_case.alpha),
                        )
                    };
                    black_box(result.expect("multi-chunk relation evaluation"))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(relation_evaluator, bench_relation_evaluator);
criterion_main!(relation_evaluator);
