#![allow(missing_docs)]

use akita_config::akita_batched_root_layout;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{AkitaPolyOps, CommitmentProver, CommittedPolynomials, OneHotPoly};
use akita_transcript::{Blake2bTranscript, Transcript};
use akita_types::LevelParams;
use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BasisMode, BlockOrder,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BenchmarkGroup, Criterion, SamplingMode, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

type F = fp128::Field;
type Cfg = fp128::D64OneHot;
const D: usize = Cfg::D;

const SINGLE_NUM_VARS: usize = 34;
const BATCH_NUM_VARS: usize = 29;
const BATCH_SIZE: usize = 1 << 5;
const ONEHOT_K: usize = D;
const TOTAL_FIELD_ELEMS: u64 = 1u64 << SINGLE_NUM_VARS;

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(20));
    group.throughput(Throughput::Elements(TOTAL_FIELD_ELEMS));
}

fn make_onehot_poly(layout: &LevelParams, seed: u64) -> OneHotPoly<F, D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let num_vars = layout.m_vars + layout.r_vars + D.trailing_zeros() as usize;
    assert_eq!(total_ring * ONEHOT_K, 1usize << num_vars);

    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();

    OneHotPoly::<F, D, u8>::new(ONEHOT_K, indices).expect("benchmark onehot poly")
}

fn random_point(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn opening_from_poly<const D: usize, P: AkitaPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &LevelParams,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

fn bench_single_case(c: &mut Criterion) {
    let layout = Cfg::commitment_layout(SINGLE_NUM_VARS).expect("single layout");
    let poly = make_onehot_poly(&layout, 0x0bee_fcaf_e000_0034);
    let point = random_point(SINGLE_NUM_VARS);
    let opening = opening_from_poly(&poly, &point, &layout);

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        SINGLE_NUM_VARS,
        1,
        1,
    );
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .expect("single commit");

    let poly_refs: [&OneHotPoly<F, D, u8>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut group = c.benchmark_group("akita/onehot_opening/single_1xnv34");
    configure_group(&mut group);

    group.bench_function("prove", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let prove_hints = vec![hint.clone()];
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/single");
                let start = Instant::now();
                let proof =
                    <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                        &setup,
                        vec![(
                            &point[..],
                            vec![CommittedPolynomials {
                                polynomials: &poly_refs[..],
                                commitment: &commitments[0],
                                hint: prove_hints.into_iter().next().unwrap(),
                            }],
                        )],
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .expect("single prove");
                total += start.elapsed();
                black_box(proof);
            }
            total
        })
    });

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/single");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("single benchmark proof");

    group.bench_function("verify", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/single");
                let start = Instant::now();
                <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                    &proof,
                    &verifier_setup,
                    &mut transcript,
                    vec![(
                        &point[..],
                        vec![CommittedOpenings {
                            openings: opening_groups[0],
                            commitment: &commitments[0],
                        }],
                    )],
                    BasisMode::Lagrange,
                )
                .expect("single verify");
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn bench_batched_case(c: &mut Criterion) {
    let layout =
        akita_batched_root_layout::<Cfg>(BATCH_NUM_VARS, BATCH_SIZE).expect("batch layout");
    let polys: Vec<OneHotPoly<F, D, u8>> = (0..BATCH_SIZE)
        .map(|idx| make_onehot_poly(&layout, 0x0bee_fcaf_e000_2900 + idx as u64))
        .collect();
    let point = random_point(BATCH_NUM_VARS);
    let openings: Vec<F> = polys
        .iter()
        .map(|poly| opening_from_poly(poly, &point, &layout))
        .collect();

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        BATCH_NUM_VARS,
        BATCH_SIZE,
        1,
    );
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let opening_groups = [&openings[..]];
    let (commitment, hint) =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(&polys, &setup)
            .expect("grouped commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut group = c.benchmark_group("akita/onehot_opening/batched_32xnv29");
    configure_group(&mut group);

    group.bench_function("prove", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let prove_hint = hints.clone();
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/batched");
                let start = Instant::now();
                let proof =
                    <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                        &setup,
                        vec![(
                            &point[..],
                            vec![CommittedPolynomials {
                                polynomials: &polys[..],
                                commitment: &commitments[0],
                                hint: prove_hint.into_iter().next().unwrap(),
                            }],
                        )],
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .expect("batched prove");
                total += start.elapsed();
                black_box(proof);
            }
            total
        })
    });

    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/batched");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &point[..],
            vec![CommittedPolynomials {
                polynomials: &polys[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched benchmark proof");

    group.bench_function("verify", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/batched");
                let start = Instant::now();
                <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                    &proof,
                    &verifier_setup,
                    &mut transcript,
                    vec![(
                        &point[..],
                        vec![CommittedOpenings {
                            openings: opening_groups[0],
                            commitment: &commitments[0],
                        }],
                    )],
                    BasisMode::Lagrange,
                )
                .expect("batched verify");
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn bench_onehot_batched_opening(c: &mut Criterion) {
    bench_single_case(c);
    bench_batched_case(c);
}

criterion_group!(onehot_batched_opening_benches, bench_onehot_batched_opening);

fn main() {
    #[cfg(feature = "parallel")]
    {
        let num_threads = match std::env::var("AKITA_PARALLEL").ok().as_deref() {
            None | Some("") | Some("0") => {
                tracing::info!(
                    "onehot_batched_opening: defaulting to single-threaded \
                     (set AKITA_PARALLEL=N to use N threads)"
                );
                1
            }
            Some(v) => v.parse::<usize>().unwrap_or(0),
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .stack_size(64 * 1024 * 1024)
            .build_global()
            .ok();
    }

    onehot_batched_opening_benches();
    Criterion::default().configure_from_args().final_summary();
}
