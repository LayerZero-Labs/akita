#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hachi_pcs::algebra::{CyclotomicRing, Fp128};
use hachi_pcs::error::HachiError;
use hachi_pcs::primitives::multilinear_evals::DenseMultilinearEvals;
use hachi_pcs::protocol::commitment::{
    DecompositionParams, HachiCommitmentCore, HachiCommitmentLayout, HachiProverSetup,
    HachiVerifierSetup, MegaPolyBlock, ProductionFp128CommitmentConfig, RingCommitment,
    SparseBlockEntry,
};
use hachi_pcs::protocol::commitment_scheme::{commit_onehot, HachiCommitmentScheme};
use hachi_pcs::protocol::proof::HachiCommitmentHint;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::{CommitmentConfig, HachiProof};
use hachi_pcs::{BasisMode, CommitmentScheme, FromSmallInt, Polynomial, Transcript};
use std::time::Duration;

type F = Fp128<0xfffffffffffffffffffffffffffffeed>;

const D: usize = ProductionFp128CommitmentConfig::D;

macro_rules! bench_config {
    ($name:ident, M = $m:expr, R = $r:expr) => {
        #[derive(Clone, Copy, Debug)]
        struct $name;
        impl CommitmentConfig for $name {
            const D: usize = D;
            const N_A: usize = ProductionFp128CommitmentConfig::N_A;
            const N_B: usize = ProductionFp128CommitmentConfig::N_B;
            const N_D: usize = ProductionFp128CommitmentConfig::N_D;
            const CHALLENGE_WEIGHT: usize = ProductionFp128CommitmentConfig::CHALLENGE_WEIGHT;

            fn decomposition() -> DecompositionParams {
                ProductionFp128CommitmentConfig::decomposition()
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

fn num_vars<Cfg: CommitmentConfig>() -> usize {
    let alpha = Cfg::D.trailing_zeros() as usize;
    let layout = Cfg::commitment_layout(0).expect("benchmark layout");
    layout.m_vars + layout.r_vars + alpha
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
                    BasisMode::Lagrange,
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
        BasisMode::Lagrange,
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
                BasisMode::Lagrange,
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
                BasisMode::Lagrange,
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
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_onehot_phases<Cfg: CommitmentConfig>(c: &mut Criterion, label: &str)
where
    HachiCommitmentScheme<D, Cfg>: CommitmentScheme<
        F,
        ProverSetup = HachiProverSetup<F, D>,
        VerifierSetup = HachiVerifierSetup<F, D>,
        Commitment = RingCommitment<F, D>,
        Proof = HachiProof<F, D>,
        OpeningProofHint = HachiCommitmentHint<F, D>,
    >,
{
    type S<C> = HachiCommitmentScheme<D, C>;
    let nv = num_vars::<Cfg>();
    let total_elems = 1usize << nv;
    let onehot_k = D;
    let num_chunks = total_elems / onehot_k;

    let indices: Vec<Option<usize>> = (0..num_chunks).map(|i| Some(i % onehot_k)).collect();

    let mut evals = vec![F::from_u64(0); total_elems];
    for (ci, opt_idx) in indices.iter().enumerate() {
        if let Some(idx) = opt_idx {
            evals[ci * onehot_k + idx] = F::from_u64(1);
        }
    }
    let poly = DenseMultilinearEvals::new_padded(evals);
    let pt = opening_point(nv);

    let setup = <S<Cfg> as CommitmentScheme<F>>::setup_prover(nv);

    let mut group = c.benchmark_group(format!("hachi_onehot/{label}/nv{nv}"));
    if nv >= 18 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                commit_onehot::<F, { D }, Cfg>(
                    black_box(onehot_k),
                    black_box(&indices),
                    black_box(&setup),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = commit_onehot::<F, { D }, Cfg>(onehot_k, &indices, &setup).unwrap();

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
                    BasisMode::Lagrange,
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
        Some(hint.clone()),
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
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
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    group.bench_function(BenchmarkId::new("e2e", nv), |b| {
        b.iter(|| {
            let (cm, h) = commit_onehot::<F, { D }, Cfg>(onehot_k, &indices, &setup).unwrap();
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <S<Cfg> as CommitmentScheme<F>>::prove(
                &setup,
                &poly,
                &pt,
                Some(h),
                &mut pt_tr,
                &cm,
                BasisMode::Lagrange,
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
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_mixed_phases<Cfg: CommitmentConfig>(c: &mut Criterion, label: &str)
where
    HachiCommitmentScheme<D, Cfg>: CommitmentScheme<
        F,
        ProverSetup = HachiProverSetup<F, D>,
        VerifierSetup = HachiVerifierSetup<F, D>,
        Commitment = RingCommitment<F, D>,
        Proof = HachiProof<F, D>,
        OpeningProofHint = HachiCommitmentHint<F, D>,
    >,
{
    type S<C> = HachiCommitmentScheme<D, C>;
    let nv = num_vars::<Cfg>();
    let layout = Cfg::commitment_layout(0).expect("benchmark layout");
    let block_len = layout.block_len;
    let num_blocks = layout.num_blocks;
    let dense_blocks = num_blocks / 2;

    let mut ring_coeffs: Vec<CyclotomicRing<F, D>> = Vec::with_capacity(num_blocks * block_len);

    for i in 0..(dense_blocks * block_len) {
        ring_coeffs.push(CyclotomicRing::from_coefficients(std::array::from_fn(
            |j| F::from_u64((i * D + j + 1) as u64),
        )));
    }

    let mut sparse_per_block: Vec<Vec<SparseBlockEntry>> = Vec::new();
    for bi in 0..(num_blocks - dense_blocks) {
        let mut entries = Vec::new();
        for ri in 0..block_len {
            let idx = (bi * block_len + ri) % D;
            let mut coeffs = [F::from_u64(0); D];
            coeffs[idx] = F::from_u64(1);
            ring_coeffs.push(CyclotomicRing::from_coefficients(coeffs));
            entries.push(SparseBlockEntry {
                pos_in_block: ri,
                nonzero_coeffs: vec![idx],
            });
        }
        sparse_per_block.push(entries);
    }

    let evals: Vec<F> = ring_coeffs
        .iter()
        .flat_map(|r| r.coefficients().iter().copied())
        .collect();
    let poly = DenseMultilinearEvals::new_padded(evals);
    let pt = opening_point(nv);

    let setup = <S<Cfg> as CommitmentScheme<F>>::setup_prover(nv);

    let mut group = c.benchmark_group(format!("hachi_mixed/{label}/nv{nv}"));
    if nv >= 18 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }

    group.bench_function("commit", |b| {
        b.iter(|| {
            let blocks: Vec<MegaPolyBlock<'_, F, D>> = (0..num_blocks)
                .map(|i| {
                    if i < dense_blocks {
                        let start = i * block_len;
                        let end = start + block_len;
                        MegaPolyBlock::Dense(&ring_coeffs[start..end])
                    } else {
                        MegaPolyBlock::OneHot(&sparse_per_block[i - dense_blocks])
                    }
                })
                .collect();
            black_box(
                HachiCommitmentCore::commit_mixed::<F, { D }, Cfg>(
                    black_box(&blocks),
                    black_box(&setup),
                )
                .unwrap(),
            )
        })
    });

    let blocks: Vec<MegaPolyBlock<'_, F, D>> = (0..num_blocks)
        .map(|i| {
            if i < dense_blocks {
                let start = i * block_len;
                let end = start + block_len;
                MegaPolyBlock::Dense(&ring_coeffs[start..end])
            } else {
                MegaPolyBlock::OneHot(&sparse_per_block[i - dense_blocks])
            }
        })
        .collect();
    let w = HachiCommitmentCore::commit_mixed::<F, { D }, Cfg>(&blocks, &setup).unwrap();
    let commitment = w.commitment;
    let hint = HachiCommitmentHint {
        t_hat: w.t_hat,
        ring_coeffs: ring_coeffs.clone(),
    };

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
                    BasisMode::Lagrange,
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
        Some(hint.clone()),
        &mut prover_transcript,
        &commitment,
        BasisMode::Lagrange,
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
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    group.bench_function(BenchmarkId::new("e2e", nv), |b| {
        b.iter(|| {
            let blocks: Vec<MegaPolyBlock<'_, F, D>> = (0..num_blocks)
                .map(|i| {
                    if i < dense_blocks {
                        let start = i * block_len;
                        let end = start + block_len;
                        MegaPolyBlock::Dense(&ring_coeffs[start..end])
                    } else {
                        MegaPolyBlock::OneHot(&sparse_per_block[i - dense_blocks])
                    }
                })
                .collect();
            let w = HachiCommitmentCore::commit_mixed::<F, { D }, Cfg>(&blocks, &setup).unwrap();
            let cm = w.commitment;
            let h = HachiCommitmentHint {
                t_hat: w.t_hat,
                ring_coeffs: ring_coeffs.clone(),
            };
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <S<Cfg> as CommitmentScheme<F>>::prove(
                &setup,
                &poly,
                &pt,
                Some(h),
                &mut pt_tr,
                &cm,
                BasisMode::Lagrange,
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
                BasisMode::Lagrange,
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
fn bench_mixed_nv14(c: &mut Criterion) {
    bench_mixed_phases::<CfgNv14>(c, "fp128_p275");
}

criterion_group!(
    hachi_benches,
    bench_nv10,
    bench_nv14,
    bench_nv18,
    bench_nv20,
    bench_onehot_nv14,
    bench_mixed_nv14,
);
criterion_main!(hachi_benches);
