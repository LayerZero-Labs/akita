#![allow(missing_docs)]

use akita_algebra::poly::multilinear_eval;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    ComputeBackendSetup, CpuBackend, DensePoly, OneHotPoly, SelectedProverOpeningData,
};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaCommitmentHint, BasisMode, CommittedGroup, CommittedGroupBatchProfile,
    GroupBatchStatement, OpeningClaims, OpeningScheduleSelection, PolynomialGroupClaims,
};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkGroup, Criterion};
use jolt_field::{CanonicalEncoding, Ring, Zero};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

#[path = "../tests/support/cross_mode.rs"]
mod cross_mode;
use cross_mode::{CrossModeConfig, QuotientMode, ReducedMode};
#[path = "support/relation_phase_timing.rs"]
mod relation_phase_timing;

type F = fp128::Field;

fn make_dense_evals<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
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
        .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn prover_claims<'a, Cfg, P>(
    point: &'a [F],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<Cfg::Field>,
    hint: AkitaCommitmentHint<Cfg::Field>,
) -> SelectedProverOpeningData<'a, F, akita_prover::PreparedProverGroup<'a, P>, Cfg::Field>
where
    Cfg: CommitmentConfig<ExtField = F>,
    P: akita_prover::RootPolyMeta<Cfg::Field>,
{
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![F::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    SelectedProverOpeningData::from_committed_claims::<Cfg>(
        opening_claims,
        vec![hint],
        vec![polynomials],
    )
    .expect("valid prover opening data")
}

fn verifier_claims<'a>(
    selection: OpeningScheduleSelection,
    point: &[F],
    openings: &[F],
    commitment: &'a CommittedGroup<F>,
) -> GroupBatchStatement<'a, F, F> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        openings.to_vec(),
        commitment,
    )
    .expect("valid verifier claims group")])
    .expect("valid verifier claims");
    GroupBatchStatement::new(selection, claims).expect("valid verifier statement")
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>, nv: usize) {
    if nv >= 20 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }
}

/// Setup-contribution modes benchmarked per phase. Direct scans the expanded
/// These scalar benches instantiate direct-schedule configs. Recursive stage-3
/// setup contribution is benchmarked by the config-typed multi-group profile.
fn bench_dense_phases<const D: usize, Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
    measure_stage2: bool,
) {
    if std::env::var_os("AKITA_RELATION_MODE_BENCH_ONLY").is_some() && !measure_stage2 {
        return;
    }
    let evals = make_dense_evals::<Cfg>(nv);
    let poly = DensePoly::<F>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point(nv);
    let opening = multilinear_eval(&evals, &pt).unwrap();

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("setup", |b| {
        b.iter(|| {
            black_box(
                AkitaCommitmentScheme::<Cfg>::setup_prover(black_box(nv), black_box(1)).unwrap(),
            )
        })
    });

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
                    &setup,
                    black_box(std::slice::from_ref(&poly)),
                    &stack,
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap(),
            )
        })
    });

    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
        &setup,
        std::slice::from_ref(&poly),
        &stack,
        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
    )
    .unwrap();

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let selection = Cfg::resolve_catalog_row_for_profiles(&CommittedGroupBatchProfile {
        final_group: *commitments[0].profile(),
        precommitteds: Vec::new(),
    })
    .expect("select generated schedule row")
    .selection();

    let verifier_setup =
        AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");

    let mode_label = "direct";
    group.bench_function(format!("prove/{mode_label}"), |b| {
        b.iter_batched(
            || vec![hint.clone()],
            |h| {
                let mut transcript = AkitaTranscript::<F>::new(b"bench");
                black_box(
                    AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_claims::<Cfg, _>(
                            &pt[..],
                            &poly_refs[..],
                            &commitments[0],
                            h.into_iter().next().unwrap(),
                        ),
                        &stack,
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let mut prover_transcript = AkitaTranscript::<F>::new(b"bench");
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prover_claims::<Cfg, _>(&pt[..], &poly_refs[..], &commitments[0], hint.clone()),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    group.bench_function(format!("verify/{mode_label}"), |b| {
        b.iter(|| {
            let mut transcript = AkitaTranscript::<F>::new(b"bench");
            AkitaCommitmentScheme::<Cfg>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(verifier_claims(
                    selection,
                    &pt[..],
                    &openings[..],
                    &commitments[0],
                )),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    // Replay the complete honest verifier while Criterion accounts only for
    // the per-fold Stage-2 spans nested inside the public verification call.
    if measure_stage2 {
        relation_phase_timing::report(label, nv, 3, || {
            let mut transcript = AkitaTranscript::<F>::new(b"bench");
            AkitaCommitmentScheme::<Cfg>::batched_verify(
                &proof,
                &verifier_setup,
                &mut transcript,
                verifier_claims(selection, &pt[..], &openings[..], &commitments[0]),
                BasisMode::Lagrange,
            )
            .unwrap();
        });

        group.bench_function(format!("verify_all_stage2/{mode_label}"), |b| {
            b.iter_custom(|iterations| {
                relation_phase_timing::measure_complete_stage2(iterations, || {
                    let mut transcript = AkitaTranscript::<F>::new(b"bench");
                    AkitaCommitmentScheme::<Cfg>::batched_verify(
                        black_box(&proof),
                        black_box(&verifier_setup),
                        &mut transcript,
                        black_box(verifier_claims(
                            selection,
                            &pt[..],
                            &openings[..],
                            &commitments[0],
                        )),
                        BasisMode::Lagrange,
                    )
                    .unwrap();
                })
            })
        });
    }

    group.bench_function(format!("e2e/{mode_label}"), |b| {
        b.iter(|| {
            let akita_prover::CommitOutput {
                committed_group: cm,
                hint: h,
            } = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = AkitaTranscript::<F>::new(b"bench");
            let pf = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
                &setup,
                prover_claims::<Cfg, _>(&pt[..], &poly_refs[..], &cms[0], h),
                &stack,
                &mut pt_tr,
                BasisMode::Lagrange,
            )
            .unwrap();
            let mut vt_tr = AkitaTranscript::<F>::new(b"bench");
            AkitaCommitmentScheme::<Cfg>::batched_verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                verifier_claims(selection, &pt[..], &openings[..], &cms[0]),
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });
    group.finish();
}

fn bench_onehot_phases<Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) {
    if std::env::var_os("AKITA_RELATION_MODE_BENCH_ONLY").is_some() {
        return;
    }
    let layout = Cfg::resolve_catalog_row_for_opening(
        &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
    )
    .expect("benchmark layout")
    .schedule()
    .root
    .params
    .clone();
    let total_ring = layout.blocks().live_blocks * layout.blocks().positions_per_block;
    let root_ring_dimension = layout.inner().matrix.ring_dimension();
    let onehot_k = 256;
    let total_field = total_ring * root_ring_dimension;
    assert_eq!(total_field, 1usize << nv);
    assert_eq!(total_field % onehot_k, 0);
    let total_chunks = total_field / onehot_k;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();

    let onehot_poly = OneHotPoly::<F>::new(onehot_k, indices.clone()).unwrap();

    let dense_evals: Vec<F> = {
        let mut evals = vec![F::from_u64(0); total_field];
        for (ci, opt_idx) in indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                evals[ci * onehot_k + idx] = F::from_u64(1);
            }
        }
        evals
    };
    let pt = random_point(nv);
    let opening = multilinear_eval(&dense_evals, &pt).unwrap();

    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("commit_onehot", |b| {
        b.iter(|| {
            black_box(
                AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
                    &setup,
                    black_box(std::slice::from_ref(&onehot_poly)),
                    &stack,
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
                )
                .unwrap(),
            )
        })
    });

    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
        &setup,
        std::slice::from_ref(&onehot_poly),
        &stack,
        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
    )
    .unwrap();

    let poly_refs: [&OneHotPoly<F>; 1] = [&onehot_poly];
    let commitments = [commitment];
    let openings = [opening];
    let selection = Cfg::resolve_catalog_row_for_profiles(&CommittedGroupBatchProfile {
        final_group: *commitments[0].profile(),
        precommitteds: Vec::new(),
    })
    .expect("select generated schedule row")
    .selection();

    let verifier_setup =
        AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup");

    let mode_label = "direct";
    group.bench_function(format!("prove/{mode_label}"), |b| {
        b.iter_batched(
            || vec![hint.clone()],
            |h| {
                let mut transcript = AkitaTranscript::<F>::new(b"bench");
                black_box(
                    AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
                        &setup,
                        prover_claims::<Cfg, _>(
                            &pt[..],
                            &poly_refs[..],
                            &commitments[0],
                            h.into_iter().next().unwrap(),
                        ),
                        &stack,
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let mut prover_transcript = AkitaTranscript::<F>::new(b"bench");
    let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
        &setup,
        prover_claims::<Cfg, _>(&pt[..], &poly_refs[..], &commitments[0], hint.clone()),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    group.bench_function(format!("verify/{mode_label}"), |b| {
        b.iter(|| {
            let mut transcript = AkitaTranscript::<F>::new(b"bench");
            AkitaCommitmentScheme::<Cfg>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(verifier_claims(
                    selection,
                    &pt[..],
                    &openings[..],
                    &commitments[0],
                )),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    group.bench_function(format!("e2e/{mode_label}"), |b| {
        b.iter(|| {
            let akita_prover::CommitOutput {
                committed_group: cm,
                hint: h,
            } = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
                &setup,
                std::slice::from_ref(&onehot_poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = AkitaTranscript::<F>::new(b"bench");
            let pf = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
                &setup,
                prover_claims::<Cfg, _>(&pt[..], &poly_refs[..], &cms[0], h),
                &stack,
                &mut pt_tr,
                BasisMode::Lagrange,
            )
            .unwrap();
            let mut vt_tr = AkitaTranscript::<F>::new(b"bench");
            AkitaCommitmentScheme::<Cfg>::batched_verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                verifier_claims(selection, &pt[..], &openings[..], &cms[0]),
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });
    group.finish();
}

fn bench_dense_nv14(c: &mut Criterion) {
    bench_dense_phases::<256, fp128::Dense>(c, "dense-adaptive", 14, false);
}
fn bench_dense_nv14_quotient(c: &mut Criterion) {
    bench_dense_phases::<256, CrossModeConfig<fp128::Dense, QuotientMode, 14, 1>>(
        c,
        "dense-quotient",
        14,
        true,
    );
}
fn bench_dense_nv14_reduced(c: &mut Criterion) {
    bench_dense_phases::<256, CrossModeConfig<fp128::Dense, ReducedMode, 14, 1>>(
        c,
        "dense-reduced",
        14,
        true,
    );
}
fn bench_dense_nv16(c: &mut Criterion) {
    bench_dense_phases::<256, fp128::Dense>(c, "dense-adaptive", 16, false);
}
fn bench_dense_nv24(c: &mut Criterion) {
    bench_dense_phases::<256, fp128::Dense>(c, "dense-adaptive", 24, false);
}

fn bench_onehot_nv15(c: &mut Criterion) {
    bench_onehot_phases::<fp128::OneHot>(c, "onehot-adaptive", 15);
}
fn bench_onehot_nv20(c: &mut Criterion) {
    bench_onehot_phases::<fp128::OneHot>(c, "onehot-adaptive", 20);
}
fn bench_onehot_nv25(c: &mut Criterion) {
    bench_onehot_phases::<fp128::OneHot>(c, "onehot-adaptive", 25);
}

criterion_group!(
    akita_benches,
    bench_dense_nv14,
    bench_dense_nv14_quotient,
    bench_dense_nv14_reduced,
    bench_dense_nv16,
    bench_dense_nv24,
    bench_onehot_nv15,
    bench_onehot_nv20,
    bench_onehot_nv25,
);

/// Set `AKITA_PARALLEL=0` to run benchmarks single-threaded. Set
/// `AKITA_RELATION_MODE_BENCH_ONLY=1` to construct only the selected quotient
/// and reduced relation-mode verifier cases.
fn main() {
    relation_phase_timing::init();

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
