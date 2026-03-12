#![allow(missing_docs)]

use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkGroup, Criterion};
use hachi_pcs::algebra::poly::multilinear_eval;
use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{
    Fp128FullCommitmentConfig, Fp128LogBasisCommitmentConfig, Fp128OneHotCommitmentConfig,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{BasisMode, CanonicalField, CommitmentScheme, FromSmallInt, Transcript};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

fn make_dense_evals<Cfg: CommitmentConfig>(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
        (0..len)
            .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    }
}

fn random_point(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>, nv: usize) {
    if nv >= 20 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }
}

fn bench_dense_phases<const D: usize, Cfg: CommitmentConfig>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) {
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");
    let evals = make_dense_evals::<Cfg>(nv);
    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point(nv);
    let opening = multilinear_eval(&evals, &pt).unwrap();

    let mut group = c.benchmark_group(format!("hachi/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("setup", |b| {
        b.iter(|| {
            black_box(
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(black_box(
                    nv,
                )),
            )
        })
    });

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                    black_box(&poly),
                    black_box(&setup),
                    black_box(&layout),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout)
            .unwrap();

    group.bench_function("prove", |b| {
        b.iter_batched(
            || hint.clone(),
            |h| {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench");
                black_box(
                    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
                        &setup,
                        &poly,
                        &pt,
                        h,
                        &mut transcript,
                        &commitment,
                        BasisMode::Lagrange,
                        &layout,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
        &setup,
        &poly,
        &pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(&pt),
                black_box(&opening),
                black_box(&commitment),
                BasisMode::Lagrange,
                black_box(&layout),
            )
            .unwrap();
        })
    });

    group.bench_function("e2e", |b| {
        b.iter(|| {
            let (cm, h) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                &poly, &setup, &layout,
            )
            .unwrap();
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
                &setup,
                &poly,
                &pt,
                h,
                &mut pt_tr,
                &cm,
                BasisMode::Lagrange,
                &layout,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                &pt,
                &opening,
                &cm,
                BasisMode::Lagrange,
                &layout,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_onehot_phases<const D: usize, Cfg: CommitmentConfig>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) {
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();

    let onehot_poly =
        OneHotPoly::<F, D>::new(onehot_k, indices.clone(), layout.r_vars, layout.m_vars).unwrap();

    let dense_evals: Vec<F> = {
        let mut evals = vec![F::from_u64(0); total_ring * onehot_k];
        for (ci, opt_idx) in indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                evals[ci * onehot_k + idx] = F::from_u64(1);
            }
        }
        evals
    };
    let pt = random_point(nv);
    let opening = multilinear_eval(&dense_evals, &pt).unwrap();

    let setup = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);

    let mut group = c.benchmark_group(format!("hachi/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("commit_onehot", |b| {
        b.iter(|| {
            black_box(
                <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                    black_box(&onehot_poly),
                    black_box(&setup),
                    black_box(&layout),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
        &onehot_poly,
        &setup,
        &layout,
    )
    .unwrap();

    group.bench_function("prove", |b| {
        b.iter_batched(
            || hint.clone(),
            |h| {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench");
                black_box(
                    <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
                        &setup,
                        &onehot_poly,
                        &pt,
                        h,
                        &mut transcript,
                        &commitment,
                        BasisMode::Lagrange,
                        &layout,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let verifier_setup =
        <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
        &setup,
        &onehot_poly,
        &pt,
        hint,
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(&pt),
                black_box(&opening),
                black_box(&commitment),
                BasisMode::Lagrange,
                black_box(&layout),
            )
            .unwrap();
        })
    });

    group.bench_function("e2e", |b| {
        b.iter(|| {
            let (cm, h) = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::commit(
                &onehot_poly,
                &setup,
                &layout,
            )
            .unwrap();
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::prove(
                &setup,
                &onehot_poly,
                &pt,
                h,
                &mut pt_tr,
                &cm,
                BasisMode::Lagrange,
                &layout,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                &pt,
                &opening,
                &cm,
                BasisMode::Lagrange,
                &layout,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_full_nv15(c: &mut Criterion) {
    bench_dense_phases::<{ Fp128FullCommitmentConfig::D }, Fp128FullCommitmentConfig>(
        c, "full", 15,
    );
}
fn bench_full_nv20(c: &mut Criterion) {
    bench_dense_phases::<{ Fp128FullCommitmentConfig::D }, Fp128FullCommitmentConfig>(
        c, "full", 20,
    );
}
fn bench_full_nv25(c: &mut Criterion) {
    bench_dense_phases::<{ Fp128FullCommitmentConfig::D }, Fp128FullCommitmentConfig>(
        c, "full", 25,
    );
}

fn bench_onehot_nv15(c: &mut Criterion) {
    bench_onehot_phases::<{ Fp128OneHotCommitmentConfig::D }, Fp128OneHotCommitmentConfig>(
        c, "onehot", 15,
    );
}
fn bench_onehot_nv20(c: &mut Criterion) {
    bench_onehot_phases::<{ Fp128OneHotCommitmentConfig::D }, Fp128OneHotCommitmentConfig>(
        c, "onehot", 20,
    );
}
fn bench_onehot_nv25(c: &mut Criterion) {
    bench_onehot_phases::<{ Fp128OneHotCommitmentConfig::D }, Fp128OneHotCommitmentConfig>(
        c, "onehot", 25,
    );
}

fn bench_logbasis_nv15(c: &mut Criterion) {
    bench_dense_phases::<{ Fp128LogBasisCommitmentConfig::D }, Fp128LogBasisCommitmentConfig>(
        c, "logbasis", 15,
    );
}
fn bench_logbasis_nv20(c: &mut Criterion) {
    bench_dense_phases::<{ Fp128LogBasisCommitmentConfig::D }, Fp128LogBasisCommitmentConfig>(
        c, "logbasis", 20,
    );
}
fn bench_logbasis_nv25(c: &mut Criterion) {
    bench_dense_phases::<{ Fp128LogBasisCommitmentConfig::D }, Fp128LogBasisCommitmentConfig>(
        c, "logbasis", 25,
    );
}

criterion_group!(
    hachi_benches,
    bench_full_nv15,
    bench_full_nv20,
    bench_full_nv25,
    bench_onehot_nv15,
    bench_onehot_nv20,
    bench_onehot_nv25,
    bench_logbasis_nv15,
    bench_logbasis_nv20,
    bench_logbasis_nv25,
);

/// Set `HACHI_PARALLEL=0` to run benchmarks single-threaded.
fn main() {
    #[cfg(feature = "parallel")]
    {
        let num_threads = if std::env::var("HACHI_PARALLEL")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            eprintln!("HACHI_PARALLEL=0: running single-threaded");
            1
        } else {
            0
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .stack_size(64 * 1024 * 1024)
            .build_global()
            .ok();
    }

    hachi_benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
