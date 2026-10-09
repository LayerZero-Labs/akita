//! Complete combined root sumcheck at nu=20, with weights cloned outside timing.

#![cfg(feature = "labinius")]

#[path = "../tests/combined_support.rs"]
mod combined_support;

use std::{hint::black_box, time::Duration};

use akita_labinius_prover::combined_kernel::CombinedRootKernel;
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::SumcheckInstanceProver;
use combined_support::CombinedRootSumcheck;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use jolt_field::{Prime128OffsetA7F7, Ring, Zero};
use rand::{rngs::StdRng, RngCore, SeedableRng};

type F = Prime128OffsetA7F7;
const NU: usize = 20;
const N: usize = 1 << NU;

fn random(rng: &mut StdRng) -> F {
    F::from_u128(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
}

fn rounds<P: SumcheckInstanceProver<F>>(mut prover: P, challenges: &[F]) -> F {
    let mut claim = prover.input_claim();
    for (round, &challenge) in challenges.iter().enumerate() {
        let polynomial = prover.compute_round_univariate(round, claim);
        claim = polynomial.evaluate(challenge);
        prover.ingest_challenge(round, challenge);
    }
    black_box(prover);
    claim
}

fn benchmarks(criterion: &mut Criterion) {
    #[cfg(feature = "parallel")]
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-worker benchmark pool is available");
    #[cfg(feature = "parallel")]
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .build()
        .expect("parallel benchmark pool is available");
    for base in [
        LabiniusDigitBase::Bits1,
        LabiniusDigitBase::Bits2,
        LabiniusDigitBase::Bits4,
    ] {
        let mut rng = StdRng::seed_from_u64(0xc0_6b_1e + u64::from(base.bits()));
        let digits: Vec<_> = (0..N)
            .map(|_| (rng.next_u32() % (1 << base.bits())) as u8)
            .collect();
        let kw: Vec<_> = (0..N).map(|_| random(&mut rng)).collect();
        let tau: Vec<_> = (0..NU).map(|_| random(&mut rng)).collect();
        let challenges: Vec<_> = (0..NU).map(|_| random(&mut rng)).collect();
        let beta = random(&mut rng);
        let s = digits.iter().zip(&kw).fold(F::zero(), |sum, (&w, &k)| {
            sum + F::from_u64(u64::from(w)) * k
        });
        let mut group = criterion.benchmark_group(format!(
            "combined_kernel/p128_a7f7/b={}/nu={NU}",
            base.bits()
        ));
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(N as u64));
        group.bench_function("reference", |b| {
            b.iter_batched(
                || kw.clone(),
                |weights| {
                    let execute = || {
                        let prover = CombinedRootSumcheck::new(
                            base,
                            black_box(&digits),
                            weights,
                            &tau,
                            beta,
                            s,
                        )
                        .expect("benchmark inputs have valid geometry and digits");
                        black_box(rounds(prover, black_box(&challenges)))
                    };
                    #[cfg(feature = "parallel")]
                    {
                        serial_pool.install(execute)
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        execute()
                    }
                },
                BatchSize::LargeInput,
            );
        });
        group.bench_function("kernel", |b| {
            b.iter_batched(
                || kw.clone(),
                |weights| {
                    let execute = || {
                        let prover = CombinedRootKernel::new(
                            base,
                            black_box(&digits),
                            weights,
                            &tau,
                            beta,
                            s,
                        )
                        .expect("benchmark inputs have valid geometry and digits");
                        black_box(rounds(prover, black_box(&challenges)))
                    };
                    #[cfg(feature = "parallel")]
                    {
                        serial_pool.install(execute)
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        execute()
                    }
                },
                BatchSize::LargeInput,
            );
        });
        #[cfg(feature = "parallel")]
        group.bench_function(
            format!(
                "kernel_parallel/threads={}",
                parallel_pool.current_num_threads()
            ),
            |b| {
                b.iter_batched(
                    || kw.clone(),
                    |weights| {
                        parallel_pool.install(|| {
                            let prover = CombinedRootKernel::new(
                                base,
                                black_box(&digits),
                                weights,
                                &tau,
                                beta,
                                s,
                            )
                            .expect("benchmark inputs have valid geometry and digits");
                            black_box(rounds(prover, black_box(&challenges)))
                        })
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        group.finish();
    }
}

criterion_group!(combined_kernel, benchmarks);
criterion_main!(combined_kernel);
