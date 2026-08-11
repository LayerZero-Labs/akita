#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

use akita_config::proof_optimized::{fp32, fp64};
use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig};
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling,
};
use akita_prover::compute::{RootTensorSource, TensorProjectionKernel};
use akita_prover::{AkitaProverSetup, GroupContext, OneHotPoly, RootTensorProjectionPoly};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    accumulate_matrix_field_elements_for_level, AkitaScheduleLookupKey, FpExtEncoding,
    PolynomialGroupLayout, SetupMatrixCapacity,
};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BenchmarkGroup, Criterion, SamplingMode};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DEFAULT_NUM_VARS: usize = 28;
const DEFAULT_NUM_POLYS: usize = 1;
const MAX_ONEHOT_K: usize = 256;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.nresamples(1001);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(10));
}

fn onehot_k_for_num_vars(num_vars: usize) -> usize {
    let max_supported_log_k = MAX_ONEHOT_K.trailing_zeros() as usize;
    if num_vars >= max_supported_log_k {
        MAX_ONEHOT_K
    } else {
        1usize << num_vars
    }
}

fn make_onehot_indices(num_vars: usize, num_polys: usize) -> Vec<Vec<Option<u8>>> {
    let onehot_k = onehot_k_for_num_vars(num_vars);
    assert!(onehot_k <= usize::from(u8::MAX) + 1);
    let total_evals = 1usize
        .checked_shl(num_vars as u32)
        .expect("benchmark arity should fit usize");
    assert_eq!(total_evals % onehot_k, 0);
    let total_chunks = total_evals / onehot_k;

    (0..num_polys)
        .map(|poly_idx| {
            let mut rng = StdRng::seed_from_u64(0x7072_6f6a_636f_6d6d ^ ((poly_idx as u64) << 32));
            (0..total_chunks)
                .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
                .collect()
        })
        .collect()
}

fn build_onehot_polys<F, const D: usize>(
    num_vars: usize,
    indices: &[Vec<Option<u8>>],
) -> Vec<OneHotPoly<F, u8>>
where
    F: FieldCore,
{
    let onehot_k = onehot_k_for_num_vars(num_vars);
    indices
        .iter()
        .map(|poly_indices| {
            OneHotPoly::<F, u8>::new(onehot_k, D, poly_indices.clone())
                .expect("benchmark onehot poly")
        })
        .collect()
}

fn bench_case<F, Cfg, const D: usize>(c: &mut Criterion, label: &str)
where
    F: FieldCore
        + CanonicalField
        + FromPrimitiveInt
        + RandomSampling
        + HasWide
        + HalvingField
        + PseudoMersenneField
        + akita_field::unreduced::HasCommitAccum
        + AkitaSerialize
        + Valid
        + 'static,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ExtField: FrobeniusExtField<F> + FpExtEncoding<F> + AkitaSerialize,
    Cfg::ExtField: FrobeniusExtField<F>
        + FpExtEncoding<F>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize,
{
    assert_eq!(D, Cfg::D);

    let num_vars = env_usize("AKITA_ROOT_COMMIT_NUM_VARS", DEFAULT_NUM_VARS);
    let num_polys = env_usize("AKITA_ROOT_COMMIT_NUM_POLYS", DEFAULT_NUM_POLYS);
    let indices = make_onehot_indices(num_vars, num_polys);
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(num_vars, num_polys));
    let params = akita_planner::find_schedule(
        &key,
        honest_fold_policy_of::<Cfg>(),
        &[],
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("benchmark commitment schedule")
    .schedule
    .root
    .params
    .final_group
    .commitment;
    let mut num_setup_field_elements = 0;
    accumulate_matrix_field_elements_for_level(&params, &mut num_setup_field_elements)
        .expect("benchmark setup capacity");
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        num_vars,
        num_polys,
        SetupMatrixCapacity {
            num_field_elements: num_setup_field_elements,
        },
    )
    .expect("benchmark setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");

    let mut group = c.benchmark_group(format!(
        "onehot_root_projection_commit/{label}/nv{num_vars}_np{num_polys}"
    ));
    configure_group(&mut group);

    group.bench_function("project_roots_uncached", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let polys = build_onehot_polys::<F, D>(num_vars, &indices);
                let start = Instant::now();
                let projected = polys
                    .iter()
                    .map(|poly| {
                        let view = poly.tensor_view()?;
                        TensorProjectionKernel::<_, F, Cfg::ExtField, D>::root_projection(
                            &CpuBackend::DEFAULT,
                            None,
                            view,
                        )
                    })
                    .collect::<Result<Vec<RootTensorProjectionPoly<F>>, _>>()
                    .expect("benchmark root projection");
                total += start.elapsed();
                black_box(projected);
            }
            total
        })
    });

    group.bench_function("commit_onehot_with_planned_params", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let polys = build_onehot_polys::<F, D>(num_vars, &indices);
                let start = Instant::now();
                // This is an intentionally nonsecurable fixed D64 kernel
                // benchmark. Runtime PCS entrypoints reject its missing
                // generated proof schedule, so pass the offline planned root
                // parameters directly to the canonical commitment operation.
                let committed = akita_prover::commit::<Cfg, OneHotPoly<F, u8>, CpuBackend>(
                    &polys,
                    setup.expanded.as_ref(),
                    &stack,
                    GroupContext::explicit_without_prior_groups(&params),
                )
                .expect("benchmark one hot commitment");
                total += start.elapsed();
                black_box(committed);
            }
            total
        })
    });

    group.finish();
}

fn bench_onehot_root_projection_commit(c: &mut Criterion) {
    bench_case::<fp32::Field, fp32::OneHot, 256>(c, "fp32_adaptive");
    bench_case::<fp64::Field, fp64::OneHot, 256>(c, "fp64_adaptive");
}

criterion_group! {
    name = onehot_root_projection_commit;
    config = Criterion::default()
        .without_plots()
        .nresamples(1001);
    targets = bench_onehot_root_projection_commit
}

fn main() {
    onehot_root_projection_commit();
    Criterion::default()
        .without_plots()
        .configure_from_args()
        .final_summary();
}
