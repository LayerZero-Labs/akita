//! Prepared limb commitments with real source packing. Every row measures one
//! column containing 4096 degree-648 ring elements on one local worker thread.

#![cfg(feature = "labinius")]

use std::{hint::black_box, time::Duration};

use akita_algebra::{
    binary::BinaryField128, MinusTrinomial, TrinomialLimbAccumulator, TrinomialRing,
};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_labinius_prover::{
    commit_binary_clear_limb_prepared, commit_binary_clear_prepared,
    commit_kernel::pack_binary_element_i8, limb_commit_kernel::pack_binary_element_bits,
    PreparedCommitMatrix, PreparedLimbCommitMatrix,
};
use akita_labinius_verifier::BinaryClearSetup;
use akita_params::sis::labinius::{LabiniusCoefficientPrime, LabiniusRingDegree};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::{Prime128OffsetA7F7, Ring};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const K: usize = 4;
const WIDTH: usize = 4096;
const Q: u32 = 268_433_353;
const BITS: u64 = (WIDTH * D) as u64;

fn benchmark_geometry(criterion: &mut Criterion) {
    #[cfg(feature = "parallel")]
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread benchmark pool can be created");
    let mut rng = StdRng::seed_from_u64(0x0256_4096_0648);
    let matrix: Vec<_> = (0..4 * WIDTH * D)
        .map(|_| (rng.next_u64() % u64::from(Q)) as u32)
        .collect();
    let source: Vec<_> = (0..WIDTH * K)
        .map(|_| u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
        .collect();
    let mut group = criterion.benchmark_group("limb_commit_kernel");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(BITS));
    for rank in 1..=4 {
        let start = std::time::Instant::now();
        let prepared =
            PreparedLimbCommitMatrix::prepare(Q, D, rank, WIDTH, &matrix[..rank * WIDTH * D])
                .expect("benchmark matrix has reduced coefficients and admitted geometry");
        eprintln!(
            "limb rank={rank}: build={:.6}s, matrix_with_tags={} bytes, prepared={} bytes, workspace={} bytes, committed_bits={BITS}, live_source_bits={}",
            start.elapsed().as_secs_f64(),
            prepared.matrix_bytes(),
            prepared.prepared_bytes(),
            prepared.workspace_bytes(),
            source.len() * 128,
        );
        group.bench_with_input(
            BenchmarkId::new("limb_rank", rank),
            &prepared,
            |b, prepared| {
                b.iter(|| {
                    let run = || {
                        commit_binary_clear_limb_prepared::<BinaryField128>(
                            black_box(prepared),
                            1,
                            black_box(&source),
                        )
                        .expect("benchmark source has the prepared matrix geometry")
                    };
                    #[cfg(feature = "parallel")]
                    black_box(pool.install(run));
                    #[cfg(not(feature = "parallel"))]
                    black_box(run());
                });
            },
        );
    }
    let rings = matrix[..WIDTH * D]
        .chunks_exact(D)
        .map(|coefficients| {
            TrinomialRing::<Prime128OffsetA7F7, D, MinusTrinomial>::from_coefficients(
                std::array::from_fn(|i| Prime128OffsetA7F7::from_u64(u64::from(coefficients[i]))),
            )
            .expect("benchmark trinomial degree is valid")
        })
        .collect();
    let setup = BinaryClearSetup::new(
        rings,
        1,
        WIDTH,
        1,
        -1024,
        1024,
        128,
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 47)
            .expect("benchmark challenge profile is valid"),
        LabiniusCoefficientPrime::P128OffsetA7F7,
        LabiniusRingDegree::D648,
    )
    .expect("benchmark p128 setup is valid");
    let reference = PreparedCommitMatrix::prepare(&setup).expect("benchmark matrix is valid");
    group.bench_function("p128_rank/1", |b| {
        b.iter(|| {
            let run = || {
                commit_binary_clear_prepared::<BinaryField128, Prime128OffsetA7F7, D, MinusTrinomial>(
                    black_box(&reference),
                    black_box(&setup),
                    black_box(&source),
                )
                .expect("benchmark source has valid geometry")
            };
            #[cfg(feature = "parallel")]
            black_box(pool.install(run));
            #[cfg(not(feature = "parallel"))]
            black_box(run());
        });
    });
    group.finish();

    source_components(criterion, &matrix, &source);
}

fn source_components(criterion: &mut Criterion, matrix: &[u32], source: &[u128]) {
    let prepared = PreparedLimbCommitMatrix::prepare(Q, D, 1, WIDTH, &matrix[..WIDTH * D])
        .expect("benchmark limb matrix is valid");
    let domain = prepared.domain();
    let mut slots = domain.zero_slots();
    let mut group = criterion.benchmark_group("limb_commit_source");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(BITS));
    group.bench_function("pack_bits", |b| {
        b.iter(|| {
            for words in black_box(source).chunks_exact(K) {
                black_box(
                    pack_binary_element_bits::<BinaryField128>(words)
                        .expect("benchmark source element has four words"),
                );
            }
        });
    });
    group.bench_function("pack_i8", |b| {
        b.iter(|| {
            for words in black_box(source).chunks_exact(K) {
                black_box(
                    pack_binary_element_i8::<BinaryField128, D>(words)
                        .expect("benchmark source element has four words"),
                );
            }
        });
    });
    group.bench_function("pack_bits_and_forward", |b| {
        b.iter(|| {
            for words in black_box(source).chunks_exact(K) {
                let bits = pack_binary_element_bits::<BinaryField128>(words)
                    .expect("benchmark source element has four words");
                domain
                    .forward_interleaved_bits(&bits, &mut slots)
                    .expect("benchmark packed bits have zero padding");
                black_box(&slots);
            }
        });
    });
    group.bench_function("pack_i8_and_forward_centered", |b| {
        b.iter(|| {
            for words in black_box(source).chunks_exact(K) {
                let coefficients = pack_binary_element_i8::<BinaryField128, D>(words)
                    .expect("benchmark source element has four words")
                    .map(i32::from);
                domain
                    .forward_centered(&coefficients, &mut slots)
                    .expect("signed bits are centered below the limb prime");
                black_box(&slots);
            }
        });
    });
    let rank_three = PreparedLimbCommitMatrix::prepare(Q, D, 3, WIDTH, &matrix[..3 * WIDTH * D])
        .expect("benchmark rank-three matrix is valid");
    let transformed: Vec<_> = source
        .chunks_exact(K)
        .map(|words| {
            let bits = pack_binary_element_bits::<BinaryField128>(words)
                .expect("benchmark source element has four words");
            let mut output = domain.zero_slots();
            domain
                .forward_interleaved_bits(&bits, &mut output)
                .expect("benchmark packed bits have zero padding");
            output
        })
        .collect();
    let mut accumulators: Vec<_> = (0..3)
        .map(|_| TrinomialLimbAccumulator::new(domain))
        .collect();
    group.bench_function("multiply_accumulate_rank3", |b| {
        b.iter(|| {
            for (element, source) in black_box(&transformed).iter().enumerate() {
                for (accumulator, row) in accumulators
                    .iter_mut()
                    .zip(black_box(rank_three.matrix_ntt()).chunks_exact(WIDTH))
                {
                    accumulator
                        .add_product(&row[element], source)
                        .expect("benchmark transforms have the same prime");
                }
            }
            for accumulator in &mut accumulators {
                accumulator
                    .finish(&mut slots)
                    .expect("benchmark output slots have the same prime");
                black_box(&slots);
            }
        });
    });
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    benchmark_geometry(criterion);
}

criterion_group!(limb_commit_kernel, benchmarks);
criterion_main!(limb_commit_kernel);
