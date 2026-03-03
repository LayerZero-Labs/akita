#![allow(missing_docs)]

use criterion::{black_box, criterion_group, BenchmarkId, Criterion};
use hachi_pcs::algebra::Fp128;
use hachi_pcs::error::HachiError;
use hachi_pcs::protocol::commitment::{
    DecompositionParams, Fp128CommitmentConfig, HachiCommitmentLayout,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, OneHotPoly};
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{BasisMode, CommitmentScheme, FromSmallInt, Transcript};
use std::time::Duration;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

const D: usize = Fp128CommitmentConfig::D;

macro_rules! bench_config {
    ($name:ident, M = $m:expr, R = $r:expr) => {
        #[derive(Clone, Copy, Debug)]
        struct $name;
        impl CommitmentConfig for $name {
            const D: usize = D;
            const N_A: usize = Fp128CommitmentConfig::N_A;
            const N_B: usize = Fp128CommitmentConfig::N_B;
            const N_D: usize = Fp128CommitmentConfig::N_D;
            const CHALLENGE_WEIGHT: usize = Fp128CommitmentConfig::CHALLENGE_WEIGHT;

            fn decomposition() -> DecompositionParams {
                Fp128CommitmentConfig::decomposition()
            }

            fn commitment_layout(
                _max_num_vars: usize,
            ) -> Result<HachiCommitmentLayout, HachiError> {
                HachiCommitmentLayout::new::<Self>($m, $r, &Self::decomposition())
            }
        }
    };
}

bench_config!(CfgNv10, M = 4, R = 2);
bench_config!(CfgNv14, M = 6, R = 4);
bench_config!(CfgNv18, M = 8, R = 6);
bench_config!(CfgNv20, M = 8, R = 8);

type Scheme<Cfg> = HachiCommitmentScheme<D, Cfg>;

fn num_vars<Cfg: CommitmentConfig>() -> usize {
    let alpha = Cfg::D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(0).expect("benchmark layout");
    layout.m_vars + layout.r_vars + alpha
}

fn make_dense_poly(nv: usize) -> DensePoly<F, D> {
    let len = 1usize << nv;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    DensePoly::from_field_evals(nv, &evals).unwrap()
}

fn opening_point(nv: usize) -> Vec<F> {
    (0..nv).map(|i| F::from_u64((i + 2) as u64)).collect()
}

fn lagrange_eval(evals: &[F], point: &[F]) -> F {
    let n = point.len();
    let len = 1usize << n;
    let mut weights = vec![F::from_u64(0); len];
    weights[0] = F::from_u64(1);
    for (k, &x) in point.iter().enumerate() {
        let half = 1usize << k;
        for i in (0..half).rev() {
            weights[i + half] = weights[i] * x;
            weights[i] = weights[i] - weights[i + half];
        }
    }
    evals
        .iter()
        .zip(weights.iter())
        .fold(F::from_u64(0), |a, (&e, &w)| a + e * w)
}

fn bench_phases<Cfg: CommitmentConfig>(c: &mut Criterion, label: &str) {
    let nv = num_vars::<Cfg>();
    let poly = make_dense_poly(nv);
    let pt = opening_point(nv);
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");

    let mut group = c.benchmark_group(format!("hachi/{label}/nv{nv}"));
    if nv >= 18 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }

    group.bench_function("setup", |b| {
        b.iter(|| {
            black_box(<Scheme<Cfg> as CommitmentScheme<F, D>>::setup_prover(
                black_box(nv),
            ))
        })
    });

    let setup = <Scheme<Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                <Scheme<Cfg> as CommitmentScheme<F, D>>::commit(
                    black_box(&poly),
                    black_box(&setup),
                    black_box(&layout),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) =
        <Scheme<Cfg> as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout).unwrap();

    group.bench_function("prove", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            black_box(
                <Scheme<Cfg> as CommitmentScheme<F, D>>::prove(
                    black_box(&setup),
                    black_box(&poly),
                    black_box(&pt),
                    hint.clone(),
                    &mut transcript,
                    black_box(&commitment),
                    BasisMode::Lagrange,
                    black_box(&layout),
                )
                .unwrap(),
            )
        })
    });

    let verifier_setup = <Scheme<Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let evals: Vec<F> = (0..(1usize << nv)).map(|i| F::from_u64(i as u64)).collect();
    let opening = lagrange_eval(&evals, &pt);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <Scheme<Cfg> as CommitmentScheme<F, D>>::prove(
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
            <Scheme<Cfg> as CommitmentScheme<F, D>>::verify(
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

    group.bench_function(BenchmarkId::new("e2e", nv), |b| {
        b.iter(|| {
            let (cm, h) =
                <Scheme<Cfg> as CommitmentScheme<F, D>>::commit(&poly, &setup, &layout).unwrap();
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <Scheme<Cfg> as CommitmentScheme<F, D>>::prove(
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
            <Scheme<Cfg> as CommitmentScheme<F, D>>::verify(
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

fn bench_onehot_phases<Cfg: CommitmentConfig>(c: &mut Criterion, label: &str) {
    let nv = num_vars::<Cfg>();
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;
    let num_chunks = total_ring;

    let indices: Vec<Option<usize>> = (0..num_chunks).map(|i| Some(i % onehot_k)).collect();

    let onehot_poly =
        OneHotPoly::<F, D>::new(onehot_k, indices.clone(), layout.r_vars, layout.m_vars).unwrap();

    let dense_evals: Vec<F> = {
        let mut evals = vec![F::from_u64(0); total_ring * D];
        for (ci, opt_idx) in indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                evals[ci * onehot_k + idx] = F::from_u64(1);
            }
        }
        evals
    };
    let dense_poly = DensePoly::<F, D>::from_field_evals(nv, &dense_evals).unwrap();
    let pt = opening_point(nv);

    let setup = <Scheme<Cfg> as CommitmentScheme<F, D>>::setup_prover(nv);

    let mut group = c.benchmark_group(format!("hachi_onehot/{label}/nv{nv}"));
    if nv >= 18 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }

    group.bench_function("commit_onehot", |b| {
        b.iter(|| {
            black_box(
                <Scheme<Cfg> as CommitmentScheme<F, D>>::commit(
                    black_box(&onehot_poly),
                    black_box(&setup),
                    black_box(&layout),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) =
        <Scheme<Cfg> as CommitmentScheme<F, D>>::commit(&onehot_poly, &setup, &layout).unwrap();

    group.bench_function("prove", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            black_box(
                <Scheme<Cfg> as CommitmentScheme<F, D>>::prove(
                    black_box(&setup),
                    black_box(&dense_poly),
                    black_box(&pt),
                    hint.clone(),
                    &mut transcript,
                    black_box(&commitment),
                    BasisMode::Lagrange,
                    black_box(&layout),
                )
                .unwrap(),
            )
        })
    });

    let verifier_setup = <Scheme<Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);
    let opening = lagrange_eval(&dense_evals, &pt);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <Scheme<Cfg> as CommitmentScheme<F, D>>::prove(
        &setup,
        &dense_poly,
        &pt,
        hint.clone(),
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
        &layout,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <Scheme<Cfg> as CommitmentScheme<F, D>>::verify(
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

    group.bench_function(BenchmarkId::new("e2e", nv), |b| {
        b.iter(|| {
            let (cm, h) =
                <Scheme<Cfg> as CommitmentScheme<F, D>>::commit(&onehot_poly, &setup, &layout)
                    .unwrap();
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <Scheme<Cfg> as CommitmentScheme<F, D>>::prove(
                &setup,
                &dense_poly,
                &pt,
                h,
                &mut pt_tr,
                &cm,
                BasisMode::Lagrange,
                &layout,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <Scheme<Cfg> as CommitmentScheme<F, D>>::verify(
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

fn bench_nv10(c: &mut Criterion) {
    bench_phases::<CfgNv10>(c, "fp128_p275");
}
fn bench_nv14(c: &mut Criterion) {
    bench_phases::<CfgNv14>(c, "fp128_p275");
}
fn bench_nv18(c: &mut Criterion) {
    bench_phases::<CfgNv18>(c, "fp128_p275");
}
fn bench_nv20(c: &mut Criterion) {
    bench_phases::<CfgNv20>(c, "fp128_p275");
}
fn bench_onehot_nv14(c: &mut Criterion) {
    bench_onehot_phases::<CfgNv14>(c, "fp128_p275");
}

criterion_group!(
    hachi_benches,
    bench_nv10,
    bench_nv14,
    bench_nv18,
    bench_nv20,
    bench_onehot_nv14,
);

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(64 * 1024 * 1024)
        .build_global()
        .ok();

    hachi_benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
