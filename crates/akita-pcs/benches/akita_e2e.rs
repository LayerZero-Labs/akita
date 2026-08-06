#![allow(missing_docs)]

use akita_algebra::poly::multilinear_eval;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::{CanonicalField, FieldCore};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    ComputeBackendSetup, CpuBackend, DensePoly, OneHotPoly, ProverOpeningData,
    SelectedProverOpeningData,
};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaCommitmentHint, BasisMode, CommittedGroup, CommittedGroupBatchProfile,
    GroupBatchStatement, OpeningClaims, OpeningScheduleSelection, PolynomialGroupClaims,
};
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

fn prover_claims<'a, P, CommitF: FieldCore>(
    selection: OpeningScheduleSelection,
    point: &'a [F],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> SelectedProverOpeningData<'a, F, akita_prover::PreparedProverGroup<'a, P>, CommitF>
where
    P: akita_prover::RootPolyMeta<CommitF>,
{
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![F::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    (
        selection,
        ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
            .expect("valid prover opening data"),
    )
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
) {
    let evals = make_dense_evals::<Cfg>(nv);
    let poly = DensePoly::<F>::from_field_evals(nv, D, &evals).unwrap();
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
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
                    &setup,
                    black_box(std::slice::from_ref(&poly)),
                    &stack,
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) =
        AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack)
            .unwrap();

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let selection = Cfg::select_schedule_for_profiles(&CommittedGroupBatchProfile {
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
                        prover_claims(
                            selection,
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
        prover_claims(
            selection,
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hint.clone(),
        ),
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
            let (cm, h) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = AkitaTranscript::<F>::new(b"bench");
            let pf = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
                &setup,
                prover_claims(selection, &pt[..], &poly_refs[..], &cms[0], h),
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

fn bench_onehot_phases<const D: usize, Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) {
    let layout = Cfg::get_params_for_batched_commitment(
        &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
    )
    .expect("benchmark layout");
    let total_ring = layout.num_live_blocks * layout.num_positions_per_block;
    let onehot_k = 256;
    let total_field = total_ring * D;
    assert_eq!(total_field % onehot_k, 0);
    let total_chunks = total_field / onehot_k;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();

    let onehot_poly = OneHotPoly::<F>::new(onehot_k, D, indices.clone()).unwrap();

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
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
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
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
        &setup,
        std::slice::from_ref(&onehot_poly),
        &stack,
    )
    .unwrap();

    let poly_refs: [&OneHotPoly<F>; 1] = [&onehot_poly];
    let commitments = [commitment];
    let openings = [opening];
    let selection = Cfg::select_schedule_for_profiles(&CommittedGroupBatchProfile {
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
                        prover_claims(
                            selection,
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
        prover_claims(
            selection,
            &pt[..],
            &poly_refs[..],
            &commitments[0],
            hint.clone(),
        ),
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
            let (cm, h) = AkitaCommitmentScheme::<Cfg>::commit::<_, _>(
                &setup,
                std::slice::from_ref(&onehot_poly),
                &stack,
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = AkitaTranscript::<F>::new(b"bench");
            let pf = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
                &setup,
                prover_claims(selection, &pt[..], &poly_refs[..], &cms[0], h),
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
    bench_dense_phases::<{ fp128::D64Dense::D }, fp128::D64Dense>(c, "dense-d64", 14);
}
fn bench_dense_nv16(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D64Dense::D }, fp128::D64Dense>(c, "dense-d64", 16);
}
fn bench_dense_nv24(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D64Dense::D }, fp128::D64Dense>(c, "dense-d64", 24);
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
    bench_dense_nv14,
    bench_dense_nv16,
    bench_dense_nv24,
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
