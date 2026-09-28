//! Batched negacyclic NTT throughput: device kernels against the CPU's
//! transforms on every core, over the same Q128 batches.
//!
//! `metal` samples are GPU time; `metal_wall` includes submission and waiting.
//! CPU samples are wall time over typed, in-place Montgomery buffers. Each transform
//! reads and writes every word once, so the device bound is memory copy.
//!
//! `cargo bench -p akita-metal --bench ntt`

#[cfg(target_os = "macos")]
mod bench {
    use std::time::{Duration, Instant};

    use akita_algebra::ntt::butterfly::forward_ntt;
    use akita_algebra::tables::q128_primes;
    use akita_algebra::{CrtNttParamSet, MontCoeff};
    use akita_metal::ntt::DeviceCrtNtt;
    use akita_metal::AkitaMetal;
    use criterion::{BenchmarkId, Criterion, Throughput};
    use jolt_metal::runtime::DeviceBuffer;
    use rayon::prelude::*;

    const K: usize = 6;
    /// Words per batch: about 2^24 (64 MiB), past the GPU's caches, rounded
    /// down to whole ring elements.
    const WORDS: usize = 1 << 24;

    fn words(len: usize) -> Vec<i32> {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Forward accepts any i32; keep inputs in (-2^29, 2^29).
                ((state >> 35) as i32) - (1 << 28)
            })
            .collect()
    }

    fn cpu_forward<const D: usize>(
        cpu: &mut [[MontCoeff<i32>; D]],
        params: &CrtNttParamSet<i32, K, D>,
    ) {
        cpu.par_iter_mut().enumerate().for_each(|(row, limb)| {
            let prime = row % K;
            forward_ntt(
                limb,
                params.primes[prime],
                &params.twiddles[prime],
                params.kernel_plan(),
            );
        });
    }

    fn degree<const D: usize>(c: &mut Criterion, metal: &AkitaMetal) {
        let params = CrtNttParamSet::<i32, K, D>::new(q128_primes());
        let device = DeviceCrtNtt::new(metal, &params).expect("tables");
        let len = WORDS / (K * D) * (K * D);
        let input = words(len);
        let mut buffer = DeviceBuffer::from_slice(metal.device(), &input).expect("upload");
        let mut cpu = input
            .chunks_exact(D)
            .map(|row| std::array::from_fn(|index| MontCoeff::from_raw(row[index])))
            .collect::<Vec<[MontCoeff<i32>; D]>>();

        // Check the entire timed shape before sampling. Native CPU kernels may
        // choose different lazy representatives, so compare canonical residues.
        device
            .forward(metal, &mut buffer)
            .expect("forward equality check");
        cpu_forward(&mut cpu, &params);
        for (row, (actual, expected)) in buffer
            .read()
            .expect("read")
            .chunks_exact(D)
            .zip(&cpu)
            .enumerate()
        {
            let prime = params.primes[row % K];
            for (&actual, &expected) in actual.iter().zip(expected) {
                assert_eq!(
                    prime.to_canonical(MontCoeff::from_raw(actual)),
                    prime.to_canonical(expected),
                    "D={D} row={row}"
                );
            }
        }
        // Each timed path repeatedly transforms its resident buffer; iteration
        // counts can differ, but every forward input is a supported i32 word.

        let mut group = c.benchmark_group("ntt_forward_q128");
        group.throughput(Throughput::Bytes((len * 4) as u64));
        group.bench_function(BenchmarkId::new("metal", D), |b| {
            b.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| device.forward(metal, &mut buffer).expect("forward"))
                    .sum::<Duration>()
            });
        });
        group.bench_function(BenchmarkId::new("metal_wall", D), |b| {
            b.iter(|| device.forward(metal, &mut buffer).expect("forward"));
        });
        group.bench_function(BenchmarkId::new("cpu", D), |b| {
            b.iter_custom(|iterations| {
                let start = Instant::now();
                for _ in 0..iterations {
                    cpu_forward(std::hint::black_box(&mut cpu), &params);
                }
                start.elapsed()
            });
        });
        group.finish();
        std::hint::black_box(&cpu);
    }

    pub(crate) fn main() {
        let metal = AkitaMetal::new().expect("an Apple GPU");
        let mut c = Criterion::default().configure_from_args();
        degree::<64>(&mut c, &metal);
        degree::<256>(&mut c, &metal);
        degree::<1024>(&mut c, &metal);
        c.final_summary();
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {}
