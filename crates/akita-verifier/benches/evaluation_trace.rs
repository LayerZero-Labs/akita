#![allow(missing_docs)]

use akita_types::BasisMode;
use akita_verifier::evaluation_trace_benchmark_case;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

fn bench_evaluation_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluation_trace");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for num_live_blocks in [64usize, 256, 1024, 4096, 16384] {
        for (layout, witness_chunks) in [
            ("single_chunk", 1),
            ("up_to_64_chunks", num_live_blocks.min(64)),
        ] {
            for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
                let case = evaluation_trace_benchmark_case(num_live_blocks, witness_chunks, basis)
                    .expect("valid evaluation-trace benchmark case");
                group.bench_with_input(
                    BenchmarkId::new(format!("{basis:?}/{layout}"), num_live_blocks),
                    &case,
                    |b, case| b.iter(|| black_box(case.evaluate().expect("trace evaluation"))),
                );
            }
        }
        let irregular_blocks = num_live_blocks - 3;
        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let case = evaluation_trace_benchmark_case(irregular_blocks, 64, basis)
                .expect("valid irregular evaluation-trace benchmark case");
            group.bench_with_input(
                BenchmarkId::new(format!("{basis:?}/dyadic_64_chunks"), irregular_blocks),
                &case,
                |b, case| b.iter(|| black_box(case.evaluate().expect("trace evaluation"))),
            );
        }
    }
    group.finish();
}

criterion_group!(evaluation_trace, bench_evaluation_trace);
criterion_main!(evaluation_trace);
