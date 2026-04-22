#![allow(missing_docs)]

use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BenchmarkGroup, Criterion, SamplingMode, Throughput};
use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{hachi_batched_root_layout, presets::fp128};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BlockOrder,
};
use hachi_pcs::protocol::params::LevelParams;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{BasisMode, CanonicalField, CommitmentScheme, Transcript};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

type F = Fp128<0xfffffffffffffffffffffffffffff6cd>;
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

    OneHotPoly::<F, D, u8>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars)
        .expect("benchmark onehot poly")
}

fn random_point(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn opening_from_poly<const D: usize, P: HachiPolyOps<F, D>>(
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

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
        SINGLE_NUM_VARS,
        1,
        1,
    );
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .expect("single commit");

    let mut group = c.benchmark_group("hachi/onehot_opening/single_1xnv34");
    configure_group(&mut group);

    group.bench_function("prove", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let prove_hint = hint.clone();
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/single");
                let start = Instant::now();
                let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
                    &setup,
                    &poly,
                    &point,
                    prove_hint,
                    &mut transcript,
                    &commitment,
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
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
        &setup,
        &poly,
        &point,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
    )
    .expect("single benchmark proof");

    group.bench_function("verify", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/single");
                let start = Instant::now();
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
                    &proof,
                    &verifier_setup,
                    &mut transcript,
                    &point,
                    &opening,
                    &commitment,
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
        hachi_batched_root_layout::<Cfg, D>(BATCH_NUM_VARS, BATCH_SIZE).expect("batch layout");
    let polys: Vec<OneHotPoly<F, D, u8>> = (0..BATCH_SIZE)
        .map(|idx| make_onehot_poly(&layout, 0x0bee_fcaf_e000_2900 + idx as u64))
        .collect();
    let point = random_point(BATCH_NUM_VARS);
    let openings: Vec<F> = polys
        .iter()
        .map(|poly| opening_from_poly(poly, &point, &layout))
        .collect();

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(
        BATCH_NUM_VARS,
        BATCH_SIZE,
        1,
    );
    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let poly_groups = [&polys[..]];
    let opening_groups = [&openings[..]];
    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(&polys, &setup)
            .expect("grouped commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut group = c.benchmark_group("hachi/onehot_opening/batched_32xnv29");
    configure_group(&mut group);

    group.bench_function("prove", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let prove_hint = hints.clone();
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/batched");
                let start = Instant::now();
                let proof =
                    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_prove(
                        &setup,
                        &[&poly_groups[..]],
                        &[&point[..]],
                        vec![prove_hint],
                        &mut transcript,
                        &[&commitments[..]],
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
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_prove(
        &setup,
        &[&poly_groups[..]],
        &[&point[..]],
        vec![hints],
        &mut prover_transcript,
        &[&commitments[..]],
        BasisMode::Lagrange,
    )
    .expect("batched benchmark proof");

    group.bench_function("verify", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench/onehot-opening/batched");
                let start = Instant::now();
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_verify(
                    &proof,
                    &verifier_setup,
                    &mut transcript,
                    &[&point[..]],
                    &[&opening_groups[..]],
                    &[&commitments[..]],
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
        let num_threads = match std::env::var("HACHI_PARALLEL").ok().as_deref() {
            None | Some("") | Some("0") => {
                tracing::info!(
                    "onehot_batched_opening: defaulting to single-threaded \
                     (set HACHI_PARALLEL=N to use N threads)"
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
