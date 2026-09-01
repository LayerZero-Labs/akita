use akita_transcript::{labels, AkitaTranscript};
use akita_types::{
    GrindingPlan, GrindingRun, GrindingSite, ProverGrindingTranscript, SumcheckProtocol,
    VerifierGrindingTranscript,
};
use akita_verifier::AkitaStage1Verifier;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkId, Criterion, Throughput};

#[path = "digit_range/cases.rs"]
mod cases;
use cases::{BenchmarkCase, BenchmarkField as F, DigitDistribution};

fn digit_range_grinding_plan(case: &BenchmarkCase) -> GrindingPlan {
    let rounds = case.equality_point.coordinates().len();
    let product_stages = case.plan.product_stage_arities().len();
    let mut runs = Vec::new();
    for stage in 0..=product_stages {
        for round in 0..rounds {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::SumcheckRound {
                        protocol: SumcheckProtocol::Stage1,
                        level: 0,
                        stage: u32::try_from(stage).expect("benchmark stage fits u32"),
                        round: u32::try_from(round).expect("benchmark round fits u32"),
                    },
                    1,
                    128,
                )
                .expect("benchmark grinding run"),
            );
        }
        if stage < product_stages {
            runs.push(
                GrindingRun::proof_of_work(
                    GrindingSite::Stage1InterstageBatch {
                        level: 0,
                        stage: u32::try_from(stage).expect("benchmark stage fits u32"),
                    },
                    1,
                    128,
                )
                .expect("benchmark grinding run"),
            );
        }
    }
    GrindingPlan::new(runs, 128).expect("benchmark grinding plan")
}

fn bench_digit_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("digit-range");
    group.sample_size(20);

    for basis in [4, 8, 16, 32, 64] {
        for (live_numerator, live_name) in [(4, "full"), (3, "three-quarters")] {
            for distribution in [
                DigitDistribution::Uniform,
                DigitDistribution::ZeroHeavy,
                DigitDistribution::AlternatingEndpoints,
                DigitDistribution::SeededHighEntropy,
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
                                    digit_range_grinding_plan(case),
                                )
                            },
                            |(prover, mut transcript, plan)| {
                                let mut grinding =
                                    ProverGrindingTranscript::new(&mut transcript, &plan)
                                        .expect("benchmark grinding transcript");
                                let proof = prover
                                    .prove(&mut grinding, None, 0)
                                    .expect("benchmark proof succeeds");
                                grinding.finish().expect("benchmark stream");
                                black_box(proof);
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
                                    digit_range_grinding_plan(case),
                                )
                            },
                            |(input, mut transcript, plan)| {
                                let mut grinding =
                                    ProverGrindingTranscript::new(&mut transcript, &plan)
                                        .expect("benchmark grinding transcript");
                                let proof = input
                                    .build()
                                    .prove(&mut grinding, None, 0)
                                    .expect("benchmark proof succeeds");
                                grinding.finish().expect("benchmark stream");
                                black_box(proof);
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );

                let mut prover_transcript =
                    AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
                let grinding_plan = digit_range_grinding_plan(&case);
                let mut grinding =
                    ProverGrindingTranscript::new(&mut prover_transcript, &grinding_plan)
                        .expect("benchmark grinding transcript");
                let (proof, _) = case
                    .prover_input()
                    .build()
                    .prove(&mut grinding, None, 0)
                    .expect("benchmark reference proof");
                let nonce_stream = grinding.finish().expect("benchmark nonce stream");
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
                                    grinding_plan.clone(),
                                    nonce_stream.clone(),
                                )
                            },
                            |(verifier, mut transcript, plan, stream)| {
                                let mut grinding = VerifierGrindingTranscript::new(
                                    &mut transcript,
                                    &stream,
                                    &plan,
                                )
                                .expect("benchmark grinding transcript");
                                let output = verifier
                                    .verify(&proof, &mut grinding, 0)
                                    .expect("benchmark verification succeeds");
                                grinding.finish().expect("benchmark cursor");
                                black_box(output);
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
