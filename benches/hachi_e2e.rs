#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hachi_pcs::algebra::Fp64;
use hachi_pcs::primitives::multilinear_evals::DenseMultilinearEvals;
use hachi_pcs::protocol::commitment::DefaultCommitmentConfig;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::CommitmentConfig;
use hachi_pcs::{CommitmentScheme, FromSmallInt, Polynomial, Transcript};
use std::time::Duration;

type F = Fp64<4294967197>;

const D: usize = DefaultCommitmentConfig::D;

macro_rules! bench_config {
    ($name:ident, M = $m:expr, R = $r:expr) => {
        #[derive(Clone, Copy, Debug)]
        struct $name;
        impl CommitmentConfig for $name {
            const D: usize = D;
            const M: usize = $m;
            const R: usize = $r;
            const N_A: usize = DefaultCommitmentConfig::N_A;
            const N_B: usize = DefaultCommitmentConfig::N_B;
            const N_D: usize = DefaultCommitmentConfig::N_D;
            const LOG_BASIS: u32 = DefaultCommitmentConfig::LOG_BASIS;
            const DELTA: usize = DefaultCommitmentConfig::DELTA;
            const TAU: usize = DefaultCommitmentConfig::TAU;
            const BETA: u128 = DefaultCommitmentConfig::BETA;
            const CHALLENGE_WEIGHT: usize = DefaultCommitmentConfig::CHALLENGE_WEIGHT;
        }
    };
}

bench_config!(CfgNv10, M = 4, R = 2);
bench_config!(CfgNv14, M = 6, R = 4);
bench_config!(CfgNv18, M = 8, R = 6);
bench_config!(CfgNv20, M = 8, R = 8);

fn num_vars<Cfg: CommitmentConfig>() -> usize {
    let alpha = Cfg::D.trailing_zeros() as usize;
    Cfg::R + Cfg::M + alpha
}

fn make_poly(nv: usize) -> DenseMultilinearEvals<F> {
    let len = 1usize << nv;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    DenseMultilinearEvals::new_padded(evals)
}

fn opening_point(nv: usize) -> Vec<F> {
    (0..nv).map(|i| F::from_u64((i + 2) as u64)).collect()
}

fn bench_phases<Cfg: CommitmentConfig>(c: &mut Criterion, label: &str)
where
    HachiCommitmentScheme<D, Cfg>: CommitmentScheme<F>,
{
    type S<C> = HachiCommitmentScheme<D, C>;
    let nv = num_vars::<Cfg>();
    let poly = make_poly(nv);
    let pt = opening_point(nv);

    let mut group = c.benchmark_group(format!("hachi/{label}/nv{nv}"));
    if nv >= 18 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }

    group.bench_function("setup", |b| {
        b.iter(|| black_box(<S<Cfg> as CommitmentScheme<F>>::setup_prover(black_box(nv))))
    });

    let setup = <S<Cfg> as CommitmentScheme<F>>::setup_prover(nv);

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                <S<Cfg> as CommitmentScheme<F>>::commit(black_box(&poly), black_box(&setup))
                    .unwrap(),
            )
        })
    });

    let (commitment, hint) = <S<Cfg> as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();

    group.bench_function("prove", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            black_box(
                <S<Cfg> as CommitmentScheme<F>>::prove(
                    black_box(&setup),
                    black_box(&poly),
                    black_box(&pt),
                    Some(hint.clone()),
                    &mut transcript,
                    black_box(&commitment),
                )
                .unwrap(),
            )
        })
    });

    let verifier_setup = <S<Cfg> as CommitmentScheme<F>>::setup_verifier(&setup);
    let opening = poly.evaluate(&pt);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <S<Cfg> as CommitmentScheme<F>>::prove(
        &setup,
        &poly,
        &pt,
        Some(hint),
        &mut prover_transcript,
        &commitment,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <S<Cfg> as CommitmentScheme<F>>::verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(&pt),
                black_box(&opening),
                black_box(&commitment),
            )
            .unwrap();
        })
    });

    group.bench_function(BenchmarkId::new("e2e", nv), |b| {
        b.iter(|| {
            let (cm, h) = <S<Cfg> as CommitmentScheme<F>>::commit(&poly, &setup).unwrap();
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <S<Cfg> as CommitmentScheme<F>>::prove(
                &setup,
                &poly,
                &pt,
                Some(h),
                &mut pt_tr,
                &cm,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <S<Cfg> as CommitmentScheme<F>>::verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                &pt,
                &opening,
                &cm,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_nv10(c: &mut Criterion) {
    bench_phases::<CfgNv10>(c, "fp64");
}
fn bench_nv14(c: &mut Criterion) {
    bench_phases::<CfgNv14>(c, "fp64");
}
fn bench_nv18(c: &mut Criterion) {
    bench_phases::<CfgNv18>(c, "fp64");
}
fn bench_nv20(c: &mut Criterion) {
    bench_phases::<CfgNv20>(c, "fp64");
}

criterion_group!(
    hachi_benches,
    bench_nv10,
    bench_nv14,
    bench_nv18,
    bench_nv20
);
criterion_main!(hachi_benches);
