//! Prepared limb commitments with real source packing and degree-648 ring elements.
//! Base rows use width 4096, one column, and one local worker thread; thread rows
//! report eight-column totals at width 4096 for each thread count. The width-1024,
//! 256-column row uses one thread for the end-to-end W=20, log C=8 geometry.

#![cfg(feature = "labinius")]

use std::{hint::black_box, time::Duration};

use akita_algebra::{binary::BinaryField128, MinusTrinomial, TrinomialLimbAccumulator};
use akita_labinius_prover::{
    commit_binary_clear_limb_prepared, commit_binary_clear_prepared,
    limb_commit_kernel::pack_binary_element_bits, PreparedLimbCommitMatrix,
};
use akita_labinius_verifier::AdmittedRootSetup;
use akita_params::sis::labinius::LabiniusRootProfile;
use akita_types::proof::AkitaSetupSeed;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jolt_field::Prime128Offset275;
use rand::{rngs::StdRng, RngCore, SeedableRng};

const D: usize = 648;
const K: usize = 4;
const WIDTH: usize = 4096;
const Q: u32 = 33_568_993;
/// Committed source bits per column: `WIDTH` ring elements of `K` 128-bit words.
/// Each packed polynomial has `D = 648` coefficients, of which 136 are fixed zero.
const BITS: u64 = (WIDTH * K * 128) as u64;

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
            "limb rank={rank}: build={:.6}s, matrix_with_tags={} bytes, prepared={} bytes, workspace={} bytes, source_bits={BITS}, ring_coefficients={}",
            start.elapsed().as_secs_f64(),
            prepared.matrix_bytes(),
            prepared.prepared_bytes(),
            prepared.workspace_bytes(),
            WIDTH * D,
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
        // The profile's certified rank at this width.
        if rank == 2 {
            multiple_columns(&mut group, &prepared);
            let admitted = AdmittedRootSetup::<D, MinusTrinomial>::derive::<Prime128Offset275>(
                LabiniusRootProfile::D648Q25BoundedW46,
                14,
                0,
                128,
                AkitaSetupSeed::shake256_paged_v1([0x28; 32]),
            )
            .expect("benchmark setup is admitted");
            let setup = admitted.setup();
            assert_eq!((setup.n_a(), setup.m(), setup.columns()), (2, WIDTH, 1));
            let bound = PreparedLimbCommitMatrix::prepare_for_setup(setup)
                .expect("benchmark setup has canonical coefficients");
            group.bench_function("limb_setup_rank/2", |b| {
                b.iter(|| {
                    let run = || {
                        commit_binary_clear_prepared::<BinaryField128>(
                            black_box(&bound),
                            black_box(setup),
                            black_box(&source),
                        )
                        .expect("benchmark source has the admitted setup geometry")
                    };
                    #[cfg(feature = "parallel")]
                    black_box(pool.install(run));
                    #[cfg(not(feature = "parallel"))]
                    black_box(run());
                });
            });
        }
    }
    {
        // W=20 with log C=8: 1024 source elements in each of 256 columns.
        const GOAL_WIDTH: usize = 1024;
        const GOAL_COLUMNS: usize = 256;
        const GOAL_RANK: usize = 2;
        let mut rng = StdRng::seed_from_u64(0x0020_0008_0648);
        let matrix: Vec<_> = (0..GOAL_RANK * GOAL_WIDTH * D)
            .map(|_| (rng.next_u64() % u64::from(Q)) as u32)
            .collect();
        let source: Vec<_> = (0..GOAL_COLUMNS * GOAL_WIDTH * K)
            .map(|_| u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
            .collect();
        let prepared = PreparedLimbCommitMatrix::prepare(Q, D, GOAL_RANK, GOAL_WIDTH, &matrix)
            .expect("exact-goal benchmark matrix has admitted geometry and reduced coefficients");
        group.throughput(Throughput::Elements(
            (GOAL_COLUMNS * GOAL_WIDTH * K * 128) as u64,
        ));
        group.bench_function("limb_rank/2_width1024_columns256_threads1", |b| {
            b.iter(|| {
                let run = || {
                    commit_binary_clear_limb_prepared::<BinaryField128>(
                        black_box(&prepared),
                        GOAL_COLUMNS,
                        black_box(&source),
                    )
                    .expect("exact-goal benchmark source has the prepared matrix geometry")
                };
                #[cfg(feature = "parallel")]
                black_box(pool.install(run));
                #[cfg(not(feature = "parallel"))]
                black_box(run());
            });
        });
        group.throughput(Throughput::Elements(BITS));
    }
    group.finish();

    source_components(criterion, &matrix, &source);
}

fn multiple_columns(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    prepared: &PreparedLimbCommitMatrix,
) {
    const COLUMNS: usize = 8;
    let mut rng = StdRng::seed_from_u64(0x08c0_4096_0648);
    let source: Vec<_> = (0..COLUMNS * WIDTH * K)
        .map(|_| u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
        .collect();
    group.throughput(Throughput::Elements(BITS * COLUMNS as u64));
    #[cfg(feature = "parallel")]
    let thread_counts = [1, 2, 4].as_slice();
    #[cfg(not(feature = "parallel"))]
    let thread_counts = [1].as_slice();
    for &threads in thread_counts {
        #[cfg(feature = "parallel")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("multi-column benchmark pool can be created");
        group.bench_function(format!("limb_rank/2_columns8_threads{threads}"), |b| {
            b.iter(|| {
                let run = || {
                    commit_binary_clear_limb_prepared::<BinaryField128>(
                        black_box(prepared),
                        COLUMNS,
                        black_box(&source),
                    )
                    .expect("multi-column benchmark source has the prepared matrix geometry")
                };
                #[cfg(feature = "parallel")]
                black_box(pool.install(run));
                #[cfg(not(feature = "parallel"))]
                black_box(run());
            });
        });
    }
    group.throughput(Throughput::Elements(BITS));
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
                for (row, accumulator) in accumulators.iter_mut().enumerate() {
                    accumulator
                        .add_product(
                            black_box(rank_three.matrix_ntt_element(row, element))
                                .expect("matrix element exists"),
                            source,
                        )
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
