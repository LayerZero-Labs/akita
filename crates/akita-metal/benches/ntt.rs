//! Batched negacyclic NTT throughput: device kernels against the CPU's
//! transforms on every core, over the same Q128 batches.
//!
//! Device samples are GPU time; CPU samples are wall time. Each transform
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

    fn degree<const D: usize>(c: &mut Criterion, metal: &AkitaMetal) {
        let params = CrtNttParamSet::<i32, K, D>::new(q128_primes());
        let device = DeviceCrtNtt::new(metal, &params).expect("tables");
        let len = WORDS / (K * D) * (K * D);
        let input = words(len);
        let mut buffer = DeviceBuffer::from_slice(metal.device(), &input).expect("upload");
        let mut cpu = input.clone();

        let mut group = c.benchmark_group("ntt_forward_q128");
        group.throughput(Throughput::Bytes((len * 4) as u64));
        group.bench_function(BenchmarkId::new("metal", D), |b| {
            b.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| device.forward(metal, &mut buffer).expect("forward"))
                    .sum::<Duration>()
            });
        });
        group.bench_function(BenchmarkId::new("cpu", D), |b| {
            b.iter_custom(|iterations| {
                let start = Instant::now();
                for _ in 0..iterations {
                    cpu.par_chunks_exact_mut(D)
                        .enumerate()
                        .for_each(|(row, words)| {
                            let prime = row % K;
                            let mut limb: [MontCoeff<i32>; D] =
                                std::array::from_fn(|index| MontCoeff::from_raw(words[index]));
                            forward_ntt(
                                &mut limb,
                                params.primes[prime],
                                &params.twiddles[prime],
                                params.kernel_plan(),
                            );
                            for (word, coefficient) in words.iter_mut().zip(limb) {
                                *word = coefficient.raw();
                            }
                        });
                }
                start.elapsed()
            });
        });
        group.finish();
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
