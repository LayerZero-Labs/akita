//! Response-fold arithmetic at the first root profile's per-row geometry.
//!
//! Uses 64 rows rather than 16,384, keeping the reference well below two
//! seconds per iteration. Setup, source generation and challenge draws are
//! untimed. Criterion reports rows per second; the median smoke measurements
//! below additionally print the extrapolated 16,384-row time and speed ratio.

#![cfg(feature = "labinius")]

use std::hint::black_box;
use std::time::{Duration, Instant};

use akita_algebra::binary::BinaryField128;
use akita_challenges::{BinaryChallengeSampler, FoldDraw};
use akita_error::AkitaError;
use akita_labinius_prover::fold_kernel::fold_integer as kernel;
use akita_labinius_verifier::endpoint::fold_integer as reference;
use akita_params::sis::labinius::LabiniusRootProfile;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{rngs::StdRng, RngCore, SeedableRng};

const ROWS: usize = 64;
const COLUMNS: usize = 256;
const ROOT_ROWS: usize = 16_384;

struct SeedDraw;
impl FoldDraw for SeedDraw {
    fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([0x53; 32])
    }
}

fn median_seconds(mut run: impl FnMut()) -> f64 {
    let mut samples = std::array::from_fn::<_, 5, _>(|_| {
        let start = Instant::now();
        run();
        start.elapsed().as_secs_f64()
    });
    samples.sort_by(f64::total_cmp);
    *samples.get(2).expect("five timing samples")
}

fn benchmarks(criterion: &mut Criterion) {
    let profile = LabiniusRootProfile::D648P128BoundedW46Delta16
        .challenge_profile()
        .expect("first root challenge profile is valid");
    let challenges = BinaryChallengeSampler::new(profile.clone())
        .sample_challenges(&mut SeedDraw, b"fold-kernel-benchmark", COLUMNS)
        .expect("benchmark challenge draw is valid");
    let mut rng = StdRng::seed_from_u64(0x215_197);
    let source = (0..ROWS * COLUMNS)
        .map(|_| u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
        .collect::<Vec<_>>();
    let reference_run = || {
        black_box(
            reference::<BinaryField128>(
                black_box(&source),
                ROWS,
                COLUMNS,
                black_box(&challenges),
                black_box(&profile),
            )
            .expect("benchmark reference geometry is valid"),
        )
    };
    let kernel_run = || {
        black_box(
            kernel::<BinaryField128>(
                black_box(&source),
                ROWS,
                COLUMNS,
                black_box(&challenges),
                black_box(&profile),
            )
            .expect("benchmark kernel geometry is valid"),
        )
    };
    #[cfg(feature = "parallel")]
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("single-thread benchmark pool can be created");
    let serial_kernel = || {
        #[cfg(feature = "parallel")]
        {
            serial_pool.install(kernel_run)
        }
        #[cfg(not(feature = "parallel"))]
        {
            kernel_run()
        }
    };

    assert_eq!(reference_run(), serial_kernel());
    let reference_seconds = median_seconds(|| {
        black_box(reference_run());
    });
    let kernel_seconds = median_seconds(|| {
        black_box(serial_kernel());
    });
    eprintln!(
        "fold_kernel/F128: rows={ROWS}, columns={COLUMNS}, cap={}, total_terms={}, one thread; \
         reference={reference_seconds:.6}s ({:.0} rows/s), kernel={kernel_seconds:.6}s ({:.0} rows/s), \
         speedup={:.2}x; extrapolated {ROOT_ROWS} rows: reference={:.3}s, kernel={:.3}s",
        profile.weight_cap(), challenges.iter().map(|c| c.weight()).sum::<usize>(),
        ROWS as f64 / reference_seconds, ROWS as f64 / kernel_seconds,
        reference_seconds / kernel_seconds,
        reference_seconds * ROOT_ROWS as f64 / ROWS as f64,
        kernel_seconds * ROOT_ROWS as f64 / ROWS as f64,
    );

    let mut group = criterion.benchmark_group("fold_kernel/F128/rows=64/columns=256/W46");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(ROWS as u64));
    group.bench_function("reference", |b| b.iter(reference_run));
    group.bench_function("kernel", |b| b.iter(serial_kernel));
    #[cfg(feature = "parallel")]
    {
        let threads = std::thread::available_parallelism()
            .map_or(2, usize::from)
            .min(8);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("parallel benchmark pool can be created");
        let seconds = median_seconds(|| {
            black_box(pool.install(kernel_run));
        });
        eprintln!(
            "fold_kernel/F128: parallel_threads={threads}, time={seconds:.6}s, {:.0} rows/s, \
             extrapolated {ROOT_ROWS} rows={:.3}s",
            ROWS as f64 / seconds,
            seconds * ROOT_ROWS as f64 / ROWS as f64,
        );
        group.bench_function(format!("kernel_parallel/{threads}_threads"), |b| {
            b.iter(|| pool.install(kernel_run));
        });
    }
    group.finish();
}

criterion_group!(fold_kernel, benchmarks);
criterion_main!(fold_kernel);
