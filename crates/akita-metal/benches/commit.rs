//! End-to-end root commitment: `CpuBackend::commit` with and without the
//! Metal commitment stage provider, on fp128 dense and one-hot sources.
//!
//! Each shape first checks that both routes produce the same committed
//! group, then reports the median wall time of `SAMPLES` commitments per
//! route. The provider's first commitment prepares its device setup matrices
//! and is reported separately (`cold`). CPU times use every core.
//!
//! ```text
//! cargo bench -p akita-metal --bench commit
//! AKITA_COMMIT_BENCH=dense24,onehot30 AKITA_COMMIT_SAMPLES=7 cargo bench -p akita-metal --bench commit
//! ```

#[cfg(target_os = "macos")]
mod bench {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use akita_config::proof_optimized::fp128;
    use akita_config::CommitmentConfig;
    use akita_cpu_backend::commitment_backend::CommitmentStageProvider;
    use akita_cpu_backend::{
        AkitaProverSetup, CpuBackend, CpuSource, DensePoly, GroupContext, OneHotPoly,
    };
    use akita_metal::provider::MetalCommitmentProvider;
    use jolt_field::Ring;

    type F = fp128::Field;
    type E = <fp128::Dense as CommitmentConfig>::ExtField;

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort();
        samples[samples.len() / 2]
    }

    fn run<Cfg, P>(name: &str, num_vars: usize, poly: P, samples: usize)
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
        P: CpuSource<F, E> + Clone,
    {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>().unwrap();
        let capacity = akita_config::SetupRequirements::from_catalog::<Cfg>(&catalog, num_vars, 1)
            .unwrap()
            .matrix_capacity();
        let setup = AkitaProverSetup::<F>::generate_with_capacity(num_vars, 1, capacity).unwrap();
        let cpu = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
        let provider =
            Arc::new(MetalCommitmentProvider::new(setup.expanded.clone()).expect("an Apple GPU"));
        let installed: Arc<dyn CommitmentStageProvider<F>> = provider.clone();
        let metal = CpuBackend::<F, E>::new(setup.expanded.clone())
            .unwrap()
            .with_commitment_stage_provider(installed);

        let commit = |backend: &CpuBackend<F, E>| {
            let source = backend.import_source(vec![poly.clone()]).unwrap();
            let start = Instant::now();
            let output = backend
                .commit(
                    &catalog,
                    &source,
                    GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap();
            (start.elapsed(), output.committed_group)
        };

        let (cpu_first, expected) = commit(&cpu);
        let (cold, actual) = commit(&metal);
        assert_eq!(
            actual, expected,
            "{name}: the Metal route changed the commitment"
        );
        assert!(provider.stage_calls().inner == 1 && provider.stage_calls().outer == 1);

        let cpu_time = median((0..samples).map(|_| commit(&cpu).0).collect());
        let before = provider.stage_calls();
        let metal_time = median((0..samples).map(|_| commit(&metal).0).collect());
        let after = provider.stage_calls();
        let per_commit = |total: Duration| total / u32::try_from(samples).unwrap();
        eprintln!(
            "{name}: cpu {cpu_time:.2?} (first {cpu_first:.2?}) | metal {metal_time:.2?} \
             (cold {cold:.2?}; mean inner stage {:.2?}, outer stage {:.2?}; {} device \
             matrices) | speedup {:.2}x | median of {samples}",
            per_commit(after.inner_time - before.inner_time),
            per_commit(after.outer_time - before.outer_time),
            provider.prepared_matrices(),
            cpu_time.as_secs_f64() / metal_time.as_secs_f64(),
        );
    }

    fn dense(num_vars: usize) -> DensePoly<F> {
        let evals = (0..1u64 << num_vars)
            .map(|index| {
                F::from_u64(index.wrapping_mul(0x9e37_79b9_7f4a_7c15)) * F::from_u64(index)
            })
            .collect::<Vec<_>>();
        DensePoly::from_field_evals(num_vars, &evals).unwrap()
    }

    /// Every chunk hot, as in a Jolt one-hot witness.
    fn onehot(num_vars: usize) -> OneHotPoly<F, u8> {
        let chunk = akita_config::unit_onehot_source_chunk_size::<fp128::OneHot>().unwrap();
        let indices = (0..(1usize << num_vars) / chunk)
            .map(|index| Some(((index * 131 + 17) % chunk) as u8))
            .collect();
        OneHotPoly::new(chunk, indices).unwrap()
    }

    pub(crate) fn main() {
        let samples = std::env::var("AKITA_COMMIT_SAMPLES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);
        let shapes = std::env::var("AKITA_COMMIT_BENCH")
            .unwrap_or_else(|_| "dense24,dense26,onehot28,onehot30".into());
        for shape in shapes.split(',') {
            let (kind, num_vars) = shape.split_at(shape.len() - 2);
            let num_vars: usize = num_vars.parse().expect("a two-digit arity");
            match kind {
                "dense" => run::<fp128::Dense, _>(shape, num_vars, dense(num_vars), samples),
                "onehot" => run::<fp128::OneHot, _>(shape, num_vars, onehot(num_vars), samples),
                other => panic!("unknown shape kind {other}"),
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(bench::main)
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(not(target_os = "macos"))]
fn main() {}
