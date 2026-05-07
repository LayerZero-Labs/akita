#![allow(missing_docs)]

use akita_algebra::poly::multilinear_eval;
use akita_challenges::SparseChallengeConfig;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly};
use akita_transcript::Blake2bTranscript;
use akita_types::{
    direct_witness_bytes, AjtaiRole, AkitaBatchedProof, AkitaCommitmentHint, AkitaRootBatchSummary,
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, AkitaVerifierSetup, BasisMode,
    CommitmentEnvelope, DecompositionParams, DirectStep, DirectWitnessShape, RingCommitment,
    Schedule, ScheduleProvider, Step,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkGroup, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::marker::PhantomData;
use std::time::Duration;

type F = fp128::Field;

#[derive(Clone, Copy, Debug)]
struct TensorCfg<Base>(PhantomData<Base>);

impl<Base: ScheduleProvider> ScheduleProvider for TensorCfg<Base> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        Base::schedule_table()
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        Base::schedule_key(key)
    }

    fn schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Base::schedule_plan(key)
    }
}

impl<Base> akita_planner::PlannerConfig for TensorCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    const PLANNER_D: usize = Base::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Base::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Base::planner_schedule_plan(key)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_current_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::planner_root_level_params_for_layout_with_log_basis(inputs, lp)?;
        if matches!(
            lp.stage1_challenge_shape,
            akita_challenges::Stage1ChallengeShape::Tensor
        ) {
            Ok(params.with_tensor_stage1_challenges())
        } else {
            Ok(params)
        }
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }
}

impl<Base> CommitmentConfig for TensorCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    type Field = Base::Field;
    type ClaimField = Base::ClaimField;
    type ChallengeField = Base::ChallengeField;
    const D: usize = Base::D;

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::stage1_challenge_config(d)
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Base::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Base::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), akita_field::AkitaError> {
        Base::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn level_params_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> akita_types::LevelParams {
        Base::level_params_with_log_basis(inputs, log_basis)
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::root_level_params_for_layout_with_log_basis(inputs, lp)?;
        if matches!(
            lp.stage1_challenge_shape,
            akita_challenges::Stage1ChallengeShape::Tensor
        ) {
            Ok(params.with_tensor_stage1_challenges())
        } else {
            Ok(params)
        }
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        Base::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::log_basis_search_range(inputs)
    }

    fn commitment_layout(
        max_num_vars: usize,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::commitment_layout(max_num_vars)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, akita_field::AkitaError> {
        let mut schedule =
            Base::get_params_for_prove(max_num_vars, num_vars, layout_num_claims, batch)?;
        tensorize_root_schedule::<Base>(&mut schedule, batch)?;
        Ok(schedule)
    }
}

fn tensorize_root_schedule<Base: CommitmentConfig>(
    schedule: &mut Schedule,
    batch: AkitaRootBatchSummary,
) -> Result<(), akita_field::AkitaError> {
    let (next_w_len, log_basis) = {
        let Some(Step::Fold(root_step)) = schedule.steps.first_mut() else {
            return Ok(());
        };
        root_step.params = root_step.params.clone().with_tensor_stage1_challenges();
        root_step.delta_fold_per_poly = root_step.params.num_digits_fold;
        root_step.w_ring =
            akita_types::w_ring_element_count_with_batch_summary::<F>(&root_step.params, batch);
        root_step.next_w_len = root_step
            .w_ring
            .checked_mul(root_step.params.ring_dimension)
            .ok_or_else(|| {
                akita_field::AkitaError::InvalidSetup("tensor next-w length overflow".to_string())
            })?;
        (root_step.next_w_len, root_step.params.log_basis)
    };
    if matches!(schedule.steps.get(1), Some(Step::Fold(_))) {
        let direct = DirectStep {
            current_w_len: next_w_len,
            bits_per_elem: log_basis,
            direct_bytes: direct_witness_bytes(
                Base::decomposition().field_bits(),
                &DirectWitnessShape::PackedDigits((next_w_len, log_basis)),
            ),
        };
        schedule.steps.truncate(1);
        schedule.steps.push(Step::Direct(direct));
    } else if let Some(Step::Direct(direct)) = schedule.steps.get_mut(1) {
        direct.current_w_len = next_w_len;
    }
    Ok(())
}

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
        VerifierSetup = AkitaVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = AkitaCommitmentHint<F, D>,
        BatchedProof = AkitaBatchedProof<F>,
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
        VerifierSetup = AkitaVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = AkitaCommitmentHint<F, D>,
        BatchedProof = AkitaBatchedProof<F>,
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

fn bench_onehot_verify_only<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        VerifierSetup = AkitaVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = AkitaCommitmentHint<F, D>,
        BatchedProof = AkitaBatchedProof<F>,
    >,
{
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;

    let mut rng = StdRng::seed_from_u64(0x715e_0001);
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
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&onehot_poly),
        &setup,
    )
    .unwrap();
    let poly_refs: [&OneHotPoly<F, D>; 1] = [&onehot_poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"tensor-verify-bench");
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

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"tensor-verify-bench");
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

fn bench_onehot_stage1_verify_flat_nv12(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(
        c,
        "onehot-d64-flat-stage1",
        12,
    );
}

fn bench_onehot_stage1_verify_tensor_nv12(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, TensorCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-tensor-stage1",
        12,
    );
}

criterion_group!(
    akita_benches,
    bench_full_nv15,
    bench_full_nv20,
    bench_full_nv25,
    bench_onehot_nv15,
    bench_onehot_nv20,
    bench_onehot_nv25,
    bench_onehot_stage1_verify_flat_nv12,
    bench_onehot_stage1_verify_tensor_nv12,
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
