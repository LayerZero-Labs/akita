use akita_transcript::{labels, AkitaTranscript};
use akita_verifier::AkitaStage1Verifier;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkId, Criterion, Throughput};

#[path = "digit_range/cases.rs"]
mod cases;
use cases::{BenchmarkCase, BenchmarkField as F, DigitDistribution};

fn bench_digit_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("digit-range");
    group.sample_size(20);

    for basis in [4, 8, 16, 32, 64] {
        for (live_numerator, live_name) in [(4, "full"), (3, "three-quarters")] {
            for distribution in [
                DigitDistribution::Uniform,
                DigitDistribution::ZeroHeavy,
                DigitDistribution::AlternatingEndpoints,
            ] {
                let case = BenchmarkCase::new(basis, live_numerator, distribution);
                let case_name = format!("b{basis}/{live_name}/{}", distribution.name());
                group.throughput(Throughput::Elements(
                    u64::try_from(case.domain.live_len()).expect("benchmark length fits u64"),
                ));
                group.bench_with_input(
                    BenchmarkId::new("construct", &case_name),
                    &case,
                    |bencher, case| {
                        bencher.iter_batched(
                            || case.prover_input(),
                            |input| black_box(input.build()),
                            BatchSize::LargeInput,
                        );
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("prove", &case_name),
                    &case,
                    |bencher, case| {
                        bencher.iter_batched(
                            || {
                                (
                                    case.prover_input().build(),
                                    AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL),
                                )
                            },
                            |(prover, mut transcript)| {
                                black_box(
                                    prover
                                        .prove(&mut transcript)
                                        .expect("benchmark proof succeeds"),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("prove-total", &case_name),
                    &case,
                    |bencher, case| {
                        bencher.iter_batched(
                            || {
                                (
                                    case.prover_input(),
                                    AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL),
                                )
                            },
                            |(input, mut transcript)| {
                                black_box(
                                    input
                                        .build()
                                        .prove(&mut transcript)
                                        .expect("benchmark proof succeeds"),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );

                let mut prover_transcript =
                    AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
                let (proof, _) = case
                    .prover_input()
                    .build()
                    .prove(&mut prover_transcript)
                    .expect("benchmark reference proof");
                group.bench_with_input(
                    BenchmarkId::new("verify", &case_name),
                    &case,
                    |bencher, case| {
                        bencher.iter_batched(
                            || {
                                (
                                    AkitaStage1Verifier::new(
                                        case.equality_point.clone(),
                                        case.plan,
                                    ),
                                    AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL),
                                )
                            },
                            |(verifier, mut transcript)| {
                                black_box(
                                    verifier
                                        .verify(&proof, &mut transcript)
                                        .expect("benchmark verification succeeds"),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_digit_range);

fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
}
