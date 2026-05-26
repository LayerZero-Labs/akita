#![allow(missing_docs)]

use akita_config::proof_optimized::{fp32, fp64};
use akita_config::CommitmentConfig;
use akita_field::fields::wide::{HasWide, ReduceTo};
use akita_field::{
    AdditiveGroup, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, HasUnreducedOps, PseudoMersenneField, RandomSampling,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{commit_with_params, AkitaPolyOps, CommitmentProver, OneHotPoly};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{ClaimIncidenceSummary, RingSubfieldEncoding};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BenchmarkGroup, Criterion, SamplingMode};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DEFAULT_NUM_VARS: usize = 26;
const DEFAULT_NUM_POLYS: usize = 4;
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
) -> Vec<OneHotPoly<F, D, u8>>
where
    F: FieldCore,
{
    let onehot_k = onehot_k_for_num_vars(num_vars);
    indices
        .iter()
        .map(|poly_indices| {
            OneHotPoly::<F, D, u8>::new(onehot_k, poly_indices.clone())
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
        + HasUnreducedOps
        + HalvingField
        + PseudoMersenneField
        + AkitaSerialize
        + Valid
        + 'static,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ClaimField: FrobeniusExtField<F> + RingSubfieldEncoding<F> + AkitaSerialize,
    Cfg::ChallengeField: FrobeniusExtField<F>
        + RingSubfieldEncoding<F>
        + ExtField<Cfg::ClaimField>
        + HasUnreducedOps
        + AkitaSerialize,
{
    assert_eq!(D, Cfg::D);

    type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

    let num_vars = env_usize("AKITA_ROOT_COMMIT_NUM_VARS", DEFAULT_NUM_VARS);
    let num_polys = env_usize("AKITA_ROOT_COMMIT_NUM_POLYS", DEFAULT_NUM_POLYS);
    let indices = make_onehot_indices(num_vars, num_polys);
    let onehot_polys = build_onehot_polys::<F, D>(num_vars, &indices);
    let transformed_polys = onehot_polys
        .iter()
        .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ChallengeField>())
        .collect::<Result<Vec<_>, _>>()
        .expect("benchmark root projection");
    let setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(num_vars, num_polys, 1);
    let incidence =
        ClaimIncidenceSummary::same_point(num_vars, num_polys).expect("benchmark incidence");
    let params =
        Cfg::get_params_for_batched_commitment(&incidence).expect("benchmark commitment params");

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
                    .map(|poly| poly.tensor_packed_extension_root_poly::<Cfg::ChallengeField>())
                    .collect::<Result<Vec<_>, _>>()
                    .expect("benchmark root projection");
                total += start.elapsed();
                black_box(projected);
            }
            total
        })
    });

    group.bench_function("commit_transformed_roots", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let start = Instant::now();
                let committed = commit_with_params::<F, D, _>(&transformed_polys, &setup, &params)
                    .expect("benchmark transformed commitment");
                total += start.elapsed();
                black_box(committed);
            }
            total
        })
    });

    group.bench_function("scheme_commit_uncached_projection", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let polys = build_onehot_polys::<F, D>(num_vars, &indices);
                let start = Instant::now();
                let committed = <Scheme<D, Cfg> as CommitmentProver<F, D>>::commit(&polys, &setup)
                    .expect("benchmark scheme commitment");
                total += start.elapsed();
                black_box(committed);
            }
            total
        })
    });

    group.finish();
}

fn bench_onehot_root_projection_commit(c: &mut Criterion) {
    bench_case::<fp32::Field, fp32::D64OneHot, 64>(c, "fp32_d64");
    bench_case::<fp64::Field, fp64::D32OneHot, 32>(c, "fp64_d32");
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
