//! NTT matvec on production commitment shapes: the device against the CPU's
//! prepared-cache matvecs on every core. Outputs are compared before timing.
//!
//! Device samples report both GPU time for the whole chain (partials, finish,
//! CRT) and call wall time; CPU samples are wall time. Limb-matrix preparation
//! is a separate group.
//!
//! `cargo bench -p akita-metal --bench matvec`

#[cfg(target_os = "macos")]
mod bench {
    use std::time::{Duration, Instant};

    use akita_algebra::tables::q128_primes;
    use akita_algebra::{CanonicalEncoding, CrtNttParamSet, CyclotomicRing, One};
    use akita_cpu_backend::benchmark_support::mat_vec_mul_ntt_digits_i8;
    use akita_metal::matvec::{plan_matvec, DeviceNttMatrix, MatvecPlan};
    use akita_metal::{AkitaMetal, DeviceDigitPlanes, DigitPlane};
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

    fn benchmark_planned<T: DigitPlane, const K: usize, const D: usize>(
        c: &mut Criterion,
        metal: &AkitaMetal,
        name: &str,
        matrix: &[CyclotomicRing<F, D>],
        planes: &DeviceDigitPlanes<T>,
        expected: &[F],
        plan: MatvecPlan,
    ) {
        let primes = std::array::from_fn(|index| q128_primes()[index]);
        let params = CrtNttParamSet::<i32, K, D>::new(primes);
        let device =
            DeviceNttMatrix::from_rings(metal, &params, matrix, 1, matrix.len(), plan.limbs)
                .expect("planned matrix");
        let mut out = DeviceBuffer::<F>::zeroed(metal.device(), expected.len()).expect("out");
        device
            .mat_vec(metal, planes, &mut out)
            .expect("planned device");
        assert_eq!(out.read().expect("read"), expected, "{name} planned");

        let variant = format!("{name}_k{K}_l{}", plan.limbs);
        let mut group = c.benchmark_group("matvec");
        group.sample_size(10);
        group.bench_function(BenchmarkId::new("metal_planned_gpu", &variant), |b| {
            b.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| device.mat_vec(metal, planes, &mut out).expect("device"))
                    .sum::<Duration>()
            });
        });
        group.bench_function(BenchmarkId::new("metal_planned_wall", &variant), |b| {
            b.iter_custom(|iterations| {
                let start = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(device.mat_vec(metal, planes, &mut out).expect("device"));
                }
                start.elapsed()
            });
        });
        group.finish();

        let mut prepare = c.benchmark_group("matvec_prepare");
        prepare.sample_size(10);
        prepare.bench_function(BenchmarkId::new("metal_planned_wall", variant), |b| {
            b.iter(|| {
                std::hint::black_box(
                    DeviceNttMatrix::from_rings(
                        metal,
                        &params,
                        matrix,
                        1,
                        matrix.len(),
                        plan.limbs,
                    )
                    .expect("planned matrix"),
                );
            });
        });
        prepare.finish();
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
        let planes = DeviceDigitPlanes::from_slice(metal, &flat, 16).expect("planes");
        let mut out = DeviceBuffer::<F>::zeroed(metal.device(), blocks * D).expect("out");
        let cpu = || {
            x.par_chunks_exact(cols)
                .flat_map_iter(|block| {
                    coefficients(cache.mat_vec_i16::<F>(16, 1, block).expect("cpu"))
                })
                .collect::<Vec<_>>()
        };
        let expected = cpu();
        device.mat_vec(metal, &planes, &mut out).expect("device");
        assert_eq!(out.read().expect("read"), expected.as_slice(), "{name}");

        let mut group = c.benchmark_group("matvec");
        group.sample_size(10);
        group.bench_function(BenchmarkId::new("metal_full_gpu", name), |b| {
            b.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| device.mat_vec(metal, &planes, &mut out).expect("device"))
                    .sum::<Duration>()
            });
        });
        group.bench_function(BenchmarkId::new("metal_full_wall", name), |b| {
            b.iter_custom(|iterations| {
                let start = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(device.mat_vec(metal, &planes, &mut out).expect("device"));
                }
                start.elapsed()
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

        let modulus = (-F::one()).to_u128_checked().expect("fp128 modulus") + 1;
        let plan = plan_matvec(modulus, &q128_primes(), cols, D, 16);
        assert_eq!(
            plan,
            MatvecPlan {
                primes: 3,
                limbs: 3
            }
        );
        benchmark_planned::<i16, 3, D>(c, metal, name, &a, &planes, &expected, plan);
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
        let planes = DeviceDigitPlanes::from_slice(metal, &flat, 3).expect("planes");
        let mut out = DeviceBuffer::<F>::zeroed(metal.device(), blocks * D).expect("out");
        let cpu = || {
            let slices = x.chunks_exact(cols).collect::<Vec<_>>();
            mat_vec_mul_ntt_digits_i8::<F, D>(&cache, 1, cols, &slices, 3)
                .expect("cpu")
                .into_iter()
                .flat_map(coefficients)
                .collect::<Vec<_>>()
        };
        let expected = cpu();
        device.mat_vec(metal, &planes, &mut out).expect("device");
        assert_eq!(out.read().expect("read"), expected.as_slice(), "{name}");

        let mut group = c.benchmark_group("matvec");
        group.sample_size(10);
        group.bench_function(BenchmarkId::new("metal_full_gpu", name), |b| {
            b.iter_custom(|iterations| {
                (0..iterations)
                    .map(|_| device.mat_vec(metal, &planes, &mut out).expect("device"))
                    .sum::<Duration>()
            });
        });
        group.bench_function(BenchmarkId::new("metal_full_wall", name), |b| {
            b.iter_custom(|iterations| {
                let start = Instant::now();
                for _ in 0..iterations {
                    std::hint::black_box(device.mat_vec(metal, &planes, &mut out).expect("device"));
                }
                start.elapsed()
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

        let modulus = (-F::one()).to_u128_checked().expect("fp128 modulus") + 1;
        let plan = plan_matvec(modulus, &q128_primes(), cols, D, 3);
        assert_eq!(
            plan,
            MatvecPlan {
                primes: 2,
                limbs: 4
            }
        );
        benchmark_planned::<i8, 2, D>(c, metal, name, &a, &planes, &expected, plan);
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
