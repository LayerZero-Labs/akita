#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

use akita_algebra::poly::multilinear_eval;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    AkitaProverSetup, CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly,
};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaVerifierSetup, BasisMode, RingCommitment,
    SetupContributionMode,
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

/// Setup-contribution modes benchmarked per phase. Direct scans the expanded
/// setup matrix inline; Recursive delegates each non-terminal fold to the
/// stage-3 setup-product sumcheck. Benching both keeps `prove/{mode}`,
/// `verify/{mode}`, and `e2e/{mode}` regressions independently visible.
fn setup_contribution_modes() -> [(SetupContributionMode, &'static str); 2] {
    [
        (SetupContributionMode::Direct, "direct"),
        (SetupContributionMode::Recursive, "recursive"),
    ]
}

fn bench_dense_phases<const D: usize, Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            F,
            D,
            ProverSetup = AkitaProverSetup<F, D>,
            ExtField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            CommitHint = AkitaCommitmentHint<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
        > + CommitmentVerifier<
            F,
            D,
            ExtField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
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
                )
                .unwrap(),
            )
        })
    });

    let setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    black_box(std::slice::from_ref(&poly)),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

    for (mode, mode_label) in setup_contribution_modes() {
        group.bench_function(format!("prove/{mode_label}"), |b| {
            b.iter_batched(
                || vec![hint.clone()],
                |h| {
                    let mut transcript = AkitaTranscript::<F>::new(b"bench");
                    black_box(
                        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                            &setup,
                            &CpuBackend,
                            &prepared,
                            vec![(
                                &pt[..],
                                CommittedPolynomials {
                                    polynomials: &poly_refs[..],
                                    commitment: &commitments[0],
                                    hint: h.into_iter().next().unwrap(),
                                },
                            )],
                            &mut transcript,
                            BasisMode::Lagrange,
                            mode,
                        )
                        .unwrap(),
                    )
                },
                BatchSize::LargeInput,
            )
        });

        let mut prover_transcript = AkitaTranscript::<F>::new(b"bench");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            vec![(
                &pt[..],
                CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint: hint.clone(),
                },
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
            mode,
        )
        .unwrap();

        group.bench_function(format!("verify/{mode_label}"), |b| {
            b.iter(|| {
                let mut transcript = AkitaTranscript::<F>::new(b"bench");
                <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                    black_box(&proof),
                    black_box(&verifier_setup),
                    &mut transcript,
                    black_box(vec![(
                        &pt[..],
                        CommittedOpenings {
                            openings: opening_groups[0],
                            commitment: &commitments[0],
                        },
                    )]),
                    BasisMode::Lagrange,
                    mode,
                )
                .unwrap();
            })
        });

        group.bench_function(format!("e2e/{mode_label}"), |b| {
            b.iter(|| {
                let (cm, h) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    std::slice::from_ref(&poly),
                )
                .unwrap();
                let cms = [cm];
                let mut pt_tr = AkitaTranscript::<F>::new(b"bench");
                let pf = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    vec![(
                        &pt[..],
                        CommittedPolynomials {
                            polynomials: &poly_refs[..],
                            commitment: &cms[0],
                            hint: h,
                        },
                    )],
                    &mut pt_tr,
                    BasisMode::Lagrange,
                    mode,
                )
                .unwrap();
                let mut vt_tr = AkitaTranscript::<F>::new(b"bench");
                <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                    &pf,
                    &verifier_setup,
                    &mut vt_tr,
                    vec![(
                        &pt[..],
                        CommittedOpenings {
                            openings: opening_groups[0],
                            commitment: &cms[0],
                        },
                    )],
                    BasisMode::Lagrange,
                    mode,
                )
                .unwrap();
                black_box(())
            })
        });
    }

    group.finish();
}

fn bench_onehot_phases<const D: usize, Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
            F,
            D,
            ProverSetup = AkitaProverSetup<F, D>,
            ExtField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            CommitHint = AkitaCommitmentHint<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
        > + CommitmentVerifier<
            F,
            D,
            ExtField = F,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            BatchedProof = AkitaBatchedProof<F, F>,
        >,
{
    let layout = Cfg::get_params_for_batched_commitment(
        &akita_types::ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence"),
    )
    .expect("benchmark layout");
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

    let setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("commit_onehot", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    black_box(std::slice::from_ref(&onehot_poly)),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&onehot_poly),
    )
    .unwrap();

    let poly_refs: [&OneHotPoly<F, D>; 1] = [&onehot_poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

    for (mode, mode_label) in setup_contribution_modes() {
        group.bench_function(format!("prove/{mode_label}"), |b| {
            b.iter_batched(
                || vec![hint.clone()],
                |h| {
                    let mut transcript = AkitaTranscript::<F>::new(b"bench");
                    black_box(
                        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                            &setup,
                            &CpuBackend,
                            &prepared,
                            vec![(
                                &pt[..],
                                CommittedPolynomials {
                                    polynomials: &poly_refs[..],
                                    commitment: &commitments[0],
                                    hint: h.into_iter().next().unwrap(),
                                },
                            )],
                            &mut transcript,
                            BasisMode::Lagrange,
                            mode,
                        )
                        .unwrap(),
                    )
                },
                BatchSize::LargeInput,
            )
        });

        let mut prover_transcript = AkitaTranscript::<F>::new(b"bench");
        let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            vec![(
                &pt[..],
                CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint: hint.clone(),
                },
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
            mode,
        )
        .unwrap();

        group.bench_function(format!("verify/{mode_label}"), |b| {
            b.iter(|| {
                let mut transcript = AkitaTranscript::<F>::new(b"bench");
                <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                    black_box(&proof),
                    black_box(&verifier_setup),
                    &mut transcript,
                    black_box(vec![(
                        &pt[..],
                        CommittedOpenings {
                            openings: opening_groups[0],
                            commitment: &commitments[0],
                        },
                    )]),
                    BasisMode::Lagrange,
                    mode,
                )
                .unwrap();
            })
        });

        group.bench_function(format!("e2e/{mode_label}"), |b| {
            b.iter(|| {
                let (cm, h) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    std::slice::from_ref(&onehot_poly),
                )
                .unwrap();
                let cms = [cm];
                let mut pt_tr = AkitaTranscript::<F>::new(b"bench");
                let pf = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                    &setup,
                    &CpuBackend,
                    &prepared,
                    vec![(
                        &pt[..],
                        CommittedPolynomials {
                            polynomials: &poly_refs[..],
                            commitment: &cms[0],
                            hint: h,
                        },
                    )],
                    &mut pt_tr,
                    BasisMode::Lagrange,
                    mode,
                )
                .unwrap();
                let mut vt_tr = AkitaTranscript::<F>::new(b"bench");
                <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                    &pf,
                    &verifier_setup,
                    &mut vt_tr,
                    vec![(
                        &pt[..],
                        CommittedOpenings {
                            openings: opening_groups[0],
                            commitment: &cms[0],
                        },
                    )],
                    BasisMode::Lagrange,
                    mode,
                )
                .unwrap();
                black_box(())
            })
        });
    }

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
