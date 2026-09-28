//! One-hot inner commitment: the device kernel against the CPU column sweep
//! on production-shaped one-hot sources.
//!
//! Every chunk of `K = 256` entries is hot, as in a Jolt one-hot witness,
//! with `n_A = 1` and one digit column per position (the fp128 and fp64
//! one-hot schedules' root rows). Each shape checks the device rows against
//! the CPU's before timing. GPU samples are command-buffer execution time;
//! CPU samples are wall time of `column_sweep_ajtai_onehot_multi` on all
//! cores. Before timing, each shape prints its work: one hot entry adds
//! `n_A D` coefficients, each read once from threadgroup memory.
//!
//! ```text
//! cargo bench -p akita-metal --bench onehot
//! ```

#[cfg(target_os = "macos")]
mod bench {
    use std::time::{Duration, Instant};

    use akita_cpu_backend::benchmark_support::column_sweep_ajtai_onehot_multi;
    use akita_cpu_backend::OneHotSource;
    use akita_metal::onehot::{
        commit_onehot, DeviceFlatMatrix, DeviceOneHotSources, OneHotCommitShape, OneHotSchedule,
    };
    use akita_metal::AkitaMetal;
    use akita_types::FlatMatrix;
    use criterion::{BenchmarkId, Criterion, Throughput};
    use jolt_field::{
        AdditiveGroup, CanonicalEncoding, Prime128OffsetA7F7, Prime64Offset59,
        WithCommitAccumulator,
    };
    use jolt_metal::MetalField;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Entries per one-hot chunk.
    const CHUNK: usize = 256;

    /// One benchmark shape: `2^num_vars` entries at ring degree `D`, in blocks
    /// of `positions` ring elements.
    struct Shape {
        name: &'static str,
        num_vars: usize,
        positions: usize,
    }

    fn bench_shape<F, const D: usize>(c: &mut Criterion, metal: &AkitaMetal, shape: &Shape)
    where
        F: MetalField + CanonicalEncoding + WithCommitAccumulator,
        F::Wide: AdditiveGroup + From<F>,
    {
        let mut rng = StdRng::seed_from_u64(shape.num_vars as u64);
        let n_a = 1;
        let matrix: Vec<F> = (0..n_a * shape.positions * D)
            .map(|_| F::random(&mut rng))
            .collect();
        let indices: Vec<Option<u8>> = (0..(1usize << shape.num_vars) / CHUNK)
            .map(|_| Some(rng.gen()))
            .collect();
        let sources = [OneHotSource {
            indices: &indices,
            chunk_size: CHUNK,
            num_vars: shape.num_vars,
        }];
        let commit_shape = OneHotCommitShape {
            n_a,
            active_a_cols: shape.positions,
            num_digits_inner: 1,
        };

        let flat = FlatMatrix::from_flat_data(matrix.clone());
        let view = flat.ring_view::<D>(n_a, shape.positions).expect("A view");
        let cpu = || {
            column_sweep_ajtai_onehot_multi::<F, D, u8>(&view, &sources, n_a, shape.positions, 1)
                .expect("CPU sweep")
        };
        let expected: Vec<F> = cpu()
            .into_iter()
            .flatten()
            .flatten()
            .flat_map(|ring| *ring.coefficients())
            .collect();

        let device_matrix = DeviceFlatMatrix::new(metal, &matrix).expect("upload A");
        let upload = Instant::now();
        let device_sources = DeviceOneHotSources::new(metal, &sources).expect("upload sources");
        let upload = upload.elapsed();
        let schedule =
            OneHotSchedule::new::<F, D>(metal, &device_sources, commit_shape).expect("schedule");
        let gpu = || {
            commit_onehot::<F, D>(
                metal,
                &device_matrix,
                &device_sources,
                commit_shape,
                schedule,
            )
            .expect("commit")
        };
        let (mut rows, _) = gpu();
        assert!(
            rows.read().expect("canonical rows") == expected.as_slice(),
            "{}: device rows differ from the CPU",
            shape.name
        );

        let hot = indices.len();
        let adds = hot * n_a * D;
        eprintln!(
            "{}: {hot} hot entries, {adds} coefficient adds ({} B read from threadgroup \
             memory each); A {} MiB; source upload {upload:?}; {schedule:?}",
            shape.name,
            size_of::<F>(),
            (n_a * shape.positions * D * size_of::<F>()) >> 20,
        );

        let mut group = c.benchmark_group("onehot_commit");
        group.sample_size(10);
        group.throughput(Throughput::Elements(hot as u64));
        group.bench_function(BenchmarkId::new("gpu", shape.name), |b| {
            b.iter_custom(|iters| (0..iters).map(|_| gpu().1).sum::<Duration>())
        });
        group.bench_function(BenchmarkId::new("cpu", shape.name), |b| b.iter(cpu));
        group.finish();
    }

    pub(crate) fn benches(c: &mut Criterion) {
        let metal = AkitaMetal::new().expect("an Apple GPU with the Akita kernels");
        for shape in [
            Shape {
                name: "fp128_nv28_d256",
                num_vars: 28,
                positions: 4096,
            },
            Shape {
                name: "fp128_nv30_d256",
                num_vars: 30,
                positions: 8192,
            },
            Shape {
                name: "fp128_nv32_d256",
                num_vars: 32,
                positions: 16384,
            },
        ] {
            bench_shape::<Prime128OffsetA7F7, 256>(c, &metal, &shape);
        }
        bench_shape::<Prime128OffsetA7F7, 512>(
            c,
            &metal,
            &Shape {
                name: "fp128_nv32_d512",
                num_vars: 32,
                positions: 16384,
            },
        );
        for shape in [
            Shape {
                name: "fp64_nv28_d512",
                num_vars: 28,
                positions: 2048,
            },
            Shape {
                name: "fp64_nv30_d512",
                num_vars: 30,
                positions: 4096,
            },
        ] {
            bench_shape::<Prime64Offset59, 512>(c, &metal, &shape);
        }
    }
}

#[cfg(target_os = "macos")]
criterion::criterion_group!(onehot, bench::benches);
#[cfg(target_os = "macos")]
criterion::criterion_main!(onehot);

#[cfg(not(target_os = "macos"))]
fn main() {}
