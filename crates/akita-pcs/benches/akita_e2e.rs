#![allow(missing_docs)]

use akita_algebra::poly::multilinear_eval;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalField, FromSmallInt};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly};
use akita_transcript::{Blake2bTranscript, Transcript};
use akita_types::{
    BasisMode, HachiBatchedProof, HachiCommitmentHint, HachiVerifierSetup, RingCommitment,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkGroup, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

type F = fp128::Field;

fn make_dense_evals<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> Vec<F> {
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

fn bench_dense_phases<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        VerifierSetup = HachiVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = HachiCommitmentHint<F, D>,
        BatchedProof = HachiBatchedProof<F>,
    >,
{
    let evals = make_dense_evals::<Cfg>(nv);
    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point(nv);
    let opening = multilinear_eval(&evals, &pt).unwrap();

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("setup", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
                    black_box(nv),
                    black_box(1),
                    black_box(1),
                ),
            )
        })
    });

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    black_box(std::slice::from_ref(&poly)),
                    black_box(&setup),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .unwrap();

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    group.bench_function("prove", |b| {
        b.iter_batched(
            || vec![hint.clone()],
            |h| {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench");
                black_box(
                    <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                        &setup,
                        vec![(
                            &pt[..],
                            vec![CommittedPolynomials {
                                polynomials: &poly_refs[..],
                                commitment: &commitments[0],
                                hint: h.into_iter().next().unwrap(),
                            }],
                        )],
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &commitments[0],
                    }],
                )]),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    group.bench_function("e2e", |b| {
        b.iter(|| {
            let (cm, h) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                std::slice::from_ref(&poly),
                &setup,
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                &setup,
                vec![(
                    &pt[..],
                    vec![CommittedPolynomials {
                        polynomials: &poly_refs[..],
                        commitment: &cms[0],
                        hint: h,
                    }],
                )],
                &mut pt_tr,
                BasisMode::Lagrange,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &cms[0],
                    }],
                )],
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_onehot_phases<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        VerifierSetup = HachiVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = HachiCommitmentHint<F, D>,
        BatchedProof = HachiBatchedProof<F>,
    >,
{
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();

    let onehot_poly = OneHotPoly::<F, D>::new(onehot_k, indices.clone()).unwrap();

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

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("commit_onehot", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    black_box(std::slice::from_ref(&onehot_poly)),
                    black_box(&setup),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&onehot_poly),
        &setup,
    )
    .unwrap();

    let poly_refs: [&OneHotPoly<F, D>; 1] = [&onehot_poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    group.bench_function("prove", |b| {
        b.iter_batched(
            || vec![hint.clone()],
            |h| {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench");
                black_box(
                    <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                        &setup,
                        vec![(
                            &pt[..],
                            vec![CommittedPolynomials {
                                polynomials: &poly_refs[..],
                                commitment: &commitments[0],
                                hint: h.into_iter().next().unwrap(),
                            }],
                        )],
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &commitments[0],
                    }],
                )]),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    group.bench_function("e2e", |b| {
        b.iter(|| {
            let (cm, h) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                std::slice::from_ref(&onehot_poly),
                &setup,
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                &setup,
                vec![(
                    &pt[..],
                    vec![CommittedPolynomials {
                        polynomials: &poly_refs[..],
                        commitment: &cms[0],
                        hint: h,
                    }],
                )],
                &mut pt_tr,
                BasisMode::Lagrange,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &cms[0],
                    }],
                )],
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_full_nv15(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D128Full::D }, fp128::D128Full>(c, "full-d128", 15);
}
fn bench_full_nv20(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D128Full::D }, fp128::D128Full>(c, "full-d128", 20);
}
fn bench_full_nv25(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D128Full::D }, fp128::D128Full>(c, "full-d128", 25);
}

fn bench_onehot_nv15(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(c, "onehot-d64", 15);
}
fn bench_onehot_nv20(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(c, "onehot-d64", 20);
}
fn bench_onehot_nv25(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(c, "onehot-d64", 25);
}

criterion_group!(
    akita_benches,
    bench_full_nv15,
    bench_full_nv20,
    bench_full_nv25,
    bench_onehot_nv15,
    bench_onehot_nv20,
    bench_onehot_nv25,
);

/// Set `AKITA_PARALLEL=0` to run benchmarks single-threaded.
fn main() {
    #[cfg(feature = "parallel")]
    {
        let num_threads = if std::env::var("AKITA_PARALLEL")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            tracing::info!("AKITA_PARALLEL=0: running single-threaded");
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

    akita_benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
