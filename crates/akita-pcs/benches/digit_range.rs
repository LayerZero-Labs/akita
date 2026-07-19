use akita_field::Prime128Offset275;
use akita_prover::DigitRangeProver;
use akita_transcript::{labels, AkitaTranscript};
use akita_types::{DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
use akita_verifier::AkitaStage1Verifier;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;

#[path = "digit_range/measurement.rs"]
mod measurement;

type F = Prime128Offset275;

const NUM_VARIABLES: usize = 18;
const LOW_VARIABLE_COUNT: usize = 6;

#[derive(Clone, Copy)]
enum DigitDistribution {
    Uniform,
    ZeroHeavy,
    AlternatingEndpoints,
}

impl DigitDistribution {
    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::ZeroHeavy => "zero-heavy",
            Self::AlternatingEndpoints => "alternating-endpoints",
        }
    }

    fn witness(self, basis: usize, live_len: usize) -> Arc<[i8]> {
        let half = i16::try_from(basis / 2).expect("supported basis fits i16");
        (0..live_len)
            .map(|index| {
                let digit = match self {
                    Self::Uniform => i16::try_from(index % basis).expect("basis fits i16") - half,
                    Self::ZeroHeavy => {
                        if index % 16 == 0 {
                            half - 1
                        } else {
                            0
                        }
                    }
                    Self::AlternatingEndpoints => {
                        if index & 1 == 0 {
                            -half
                        } else {
                            half - 1
                        }
                    }
                };
                i8::try_from(digit).expect("supported balanced digit fits i8")
            })
            .collect::<Vec<_>>()
            .into()
    }
}

#[derive(Clone)]
struct BenchmarkCase {
    plan: DigitRangePlan,
    domain: FlatBooleanDomain,
    equality_point: DigitRangeEqualityPoint<F>,
    digit_witness: Arc<[i8]>,
}

struct ProverInput {
    plan: DigitRangePlan,
    domain: FlatBooleanDomain,
    equality_point: DigitRangeEqualityPoint<F>,
    digit_witness: Arc<[i8]>,
}

impl ProverInput {
    fn build(self) -> DigitRangeProver<F> {
        DigitRangeProver::new(
            self.digit_witness,
            self.plan,
            self.domain,
            self.equality_point,
        )
        .expect("benchmark prover")
    }
}

impl BenchmarkCase {
    fn new(basis: usize, live_numerator: usize, distribution: DigitDistribution) -> Self {
        let domain_len = 1usize << NUM_VARIABLES;
        let live_len = domain_len * live_numerator / 4;
        let raw_challenges = (0..NUM_VARIABLES)
            .map(|index| F::from_u64(u64::try_from(index + 2).expect("small index")))
            .collect::<Vec<_>>();
        let high_variable_count = NUM_VARIABLES - LOW_VARIABLE_COUNT;
        let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
            &raw_challenges,
            high_variable_count,
            LOW_VARIABLE_COUNT,
        )
        .expect("benchmark point");
        Self {
            plan: DigitRangePlan::new(basis).expect("supported benchmark basis"),
            domain: FlatBooleanDomain::new(live_len, NUM_VARIABLES)
                .expect("aligned benchmark domain"),
            equality_point,
            digit_witness: distribution.witness(basis, live_len),
        }
    }

    fn prover_input(&self) -> ProverInput {
        ProverInput {
            plan: self.plan,
            domain: self.domain,
            equality_point: self.equality_point.clone(),
            digit_witness: Arc::clone(&self.digit_witness),
        }
    }
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
    if measurement::run_requested() {
        return;
    }

    benches();
    Criterion::default().configure_from_args().final_summary();
}
