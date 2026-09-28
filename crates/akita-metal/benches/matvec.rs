//! NTT matvec on production commitment shapes: the device against the CPU's
//! prepared-cache matvecs on every core. Outputs are compared before timing.
//!
//! Device samples are GPU time for the whole chain (partials, finish, CRT);
//! CPU samples are wall time.
//!
//! `cargo bench -p akita-metal --bench matvec`

#[cfg(target_os = "macos")]
mod bench {
    use std::time::{Duration, Instant};

    use akita_algebra::{CanonicalEncoding, CyclotomicRing};
    use akita_cpu_backend::benchmark_support::mat_vec_mul_ntt_digits_i8;
    use akita_metal::matvec::DeviceNttMatrix;
    use akita_metal::AkitaMetal;
    use akita_types::{prepare_ntt_cache, FlatMatrix, NttCacheMode, PreparedNttCache};
    use criterion::{BenchmarkId, Criterion};
    use jolt_field::Prime128OffsetA7F7;
    use jolt_metal::runtime::DeviceBuffer;
    use rayon::prelude::*;

    type F = Prime128OffsetA7F7;

    fn matrix<const D: usize>(cols: usize) -> Vec<CyclotomicRing<F, D>> {
        (0..cols)
            .map(|entry| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    let low = (entry as u64)
                        .wrapping_mul(0x9E37_79B1_85EB_CA87)
                        .wrapping_add((coefficient as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                    F::from_u128_reduced(u128::from(low) | u128::from(low.rotate_left(29)) << 64)
                }))
            })
            .collect()
    }

    fn prepare<const D: usize>(
        matrix: &[CyclotomicRing<F, D>],
        cols: usize,
        mode: NttCacheMode,
    ) -> PreparedNttCache<D> {
        let flat = FlatMatrix::from_ring_slice(matrix);
        prepare_ntt_cache(flat.ring_view::<D>(1, cols).expect("view"), mode).expect("prepare")
    }

    fn digits<const D: usize>(planes: usize, log_basis: u32) -> Vec<[i16; D]> {
        let half = 1i64 << (log_basis - 1);
        (0..planes)
            .map(|plane| {
                std::array::from_fn(|coefficient| {
                    let mixed = (plane as i64)
                        .wrapping_mul(0x2545_F491)
                        .wrapping_add(coefficient as i64 * 0x9E37);
                    (mixed.rem_euclid(2 * half) - half) as i16
                })
            })
            .collect()
    }

    fn coefficients<const D: usize>(rings: Vec<CyclotomicRing<F, D>>) -> Vec<F> {
        rings
            .into_iter()
            .flat_map(|ring| *ring.coefficients())
            .collect()
    }

    /// Dense inner commitment: `blocks` blocks of `cols` i16 planes.
    fn dense_inner<const D: usize>(
        c: &mut Criterion,
        metal: &AkitaMetal,
        name: &str,
        blocks: usize,
        cols: usize,
    ) {
        let a = matrix::<D>(cols);
        let cache = prepare(
            &a,
            cols,
            NttCacheMode::ExactNegacyclic {
                width: cols,
                rhs_abs_bound: 1 << 15,
            },
        );
        let device =
            DeviceNttMatrix::new(metal, cache.q128_base().expect("q128"), 1, cols).expect("matrix");
        let x = digits::<D>(blocks * cols, 16);
        let flat = x.iter().flatten().copied().collect::<Vec<_>>();
        let planes = DeviceBuffer::from_slice(metal.device(), &flat).expect("planes");
        let mut out = DeviceBuffer::<F>::zeroed(metal.device(), blocks * D).expect("out");
        let cpu = || {
            x.par_chunks_exact(cols)
                .flat_map_iter(|block| {
                    coefficients(cache.mat_vec_i16::<F>(16, 1, block).expect("cpu"))
                })
                .collect::<Vec<_>>()
        };
        device
            .mat_vec(metal, &planes, 16, &mut out)
            .expect("device");
        assert_eq!(out.read().expect("read"), cpu().as_slice(), "{name}");

        let mut group = c.benchmark_group("matvec");
        group.sample_size(10);
        group.bench_function(BenchmarkId::new("metal", name), |b| {
            b.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| {
                        device
                            .mat_vec(metal, &planes, 16, &mut out)
                            .expect("device")
                    })
                    .sum::<Duration>()
            });
        });
        group.bench_function(BenchmarkId::new("cpu", name), |b| {
            b.iter_custom(|iterations| {
                let start = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(cpu());
                }
                start.elapsed()
            });
        });
        group.finish();
    }

    /// Outer commitment: `blocks` slices of `cols` base-8 digit planes.
    fn outer<const D: usize>(
        c: &mut Criterion,
        metal: &AkitaMetal,
        name: &str,
        blocks: usize,
        cols: usize,
    ) {
        let a = matrix::<D>(cols);
        let cache = prepare(&a, cols, NttCacheMode::Negacyclic);
        let device =
            DeviceNttMatrix::new(metal, cache.q128_base().expect("q128"), 1, cols).expect("matrix");
        let x = digits::<D>(blocks * cols, 3)
            .into_iter()
            .map(|plane| plane.map(|digit| digit as i8))
            .collect::<Vec<_>>();
        let flat = x.iter().flatten().copied().collect::<Vec<_>>();
        let planes = DeviceBuffer::from_slice(metal.device(), &flat).expect("planes");
        let mut out = DeviceBuffer::<F>::zeroed(metal.device(), blocks * D).expect("out");
        let cpu = || {
            let slices = x.chunks_exact(cols).collect::<Vec<_>>();
            mat_vec_mul_ntt_digits_i8::<F, D>(&cache, 1, cols, &slices, 3)
                .expect("cpu")
                .into_iter()
                .flat_map(coefficients)
                .collect::<Vec<_>>()
        };
        device.mat_vec(metal, &planes, 3, &mut out).expect("device");
        assert_eq!(out.read().expect("read"), cpu().as_slice(), "{name}");

        let mut group = c.benchmark_group("matvec");
        group.sample_size(10);
        group.bench_function(BenchmarkId::new("metal", name), |b| {
            b.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| device.mat_vec(metal, &planes, 3, &mut out).expect("device"))
                    .sum::<Duration>()
            });
        });
        group.bench_function(BenchmarkId::new("cpu", name), |b| {
            b.iter_custom(|iterations| {
                let start = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(cpu());
                }
                start.elapsed()
            });
        });
        group.finish();
    }

    pub(crate) fn main() {
        let metal = AkitaMetal::new().expect("an Apple GPU");
        let mut c = Criterion::default().configure_from_args();
        // fp128 dense nv26 inner (A): D = 1024, width 2048, i16 digits; a
        // quarter of its 256 blocks.
        dense_inner::<1024>(&mut c, &metal, "dense_nv26_inner_64_blocks", 64, 2048);
        // fp128 one-hot nv32 outer (B): D = 64, width 44032, base-8 digits,
        // four slices.
        outer::<64>(&mut c, &metal, "onehot_nv32_outer", 4, 44032);
        // fp128 dense nv26 outer (B): D = 64, width 22016, eight slices.
        outer::<64>(&mut c, &metal, "dense_nv26_outer", 8, 22016);
        c.final_summary();
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {}
