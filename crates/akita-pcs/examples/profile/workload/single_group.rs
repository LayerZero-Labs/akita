use super::{
    assert_observed_proof_size, assert_profile_ntt_cache_did_not_grow,
    degree_one_claim_point_to_base, dense_lagrange_opening_from_evals, make_profile_onehot_poly,
    onehot_lagrange_opening, opening_from_poly, planned_payload_bytes,
    profile_setup_contribution_mode, prover_claims, random_claim_point,
    report_proof_size_against_planner, run_verifier_timings, verifier_claims,
};
use crate::ntt_prewarm::prewarm_uniform_profile_execution;
use crate::parallel::ProfileThreadPools;
use crate::report::{
    emit_proof_tail_report, emit_runtime_schedule_summary, print_batched_proof_summary,
    report_crt_profile, report_setup_sizes, report_timing, report_verifier_ntt_cache_size,
};
use akita_config::CommitmentConfig;
use akita_field::unreduced::{
    HasCommitAccum, HasOptimizedFold, HasUnreducedOps, HasWide, ReduceTo,
};
use akita_field::{
    AdditiveGroup, CanonicalBytes, CanonicalField, FrobeniusExtField, FromPrimitiveInt,
    HalvingField, LiftBase, PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::compute::{
    RecursiveProveBackend, RootPolyShape, RuntimeRootCommitBackend, RuntimeRootCommitPoly,
    RuntimeRootProvePoly,
};
use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
use akita_prover::{DensePoly, OneHotPoly};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{
    BasisMode, CommittedGroupBatchProfile, CommittedGroupParams, FoldSchedule, FpExtEncoding,
    OpeningClaimsLayout, PolynomialGroupLayout,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn run_prove<
    FF,
    const D: usize,
    Cfg: CommitmentConfig<Field = FF>,
    P: RuntimeRootProvePoly<FF> + RuntimeRootCommitPoly<FF>,
>(
    label: &str,
    setup: &AkitaProverSetup<Cfg::Field>,
    stack: &akita_prover::UniformProverStack<'_, FF, CpuBackend>,
    poly: &P,
    pt: &[Cfg::ExtField],
    opening: Cfg::ExtField,
    plan: Option<&FoldSchedule>,
    // When `false`, skip the planner proof-size upper-bound assertion. That
    // guard validates shipped-catalog schedules against the offline planner
    // estimate; it is meaningless for a synthetic schedule (e.g. the mixed
    // ring-dimension-per-level experiment) that the planner cannot reproduce
    // from its lookup key. The measured proof size and per-level breakdown are
    // still reported in full.
    validate_against_planner: bool,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HalvingField
        + HasWide
        + HasCommitAccum
        + Valid
        + AkitaSerialize
        + 'static,
    <FF as HasWide>::Wide: From<FF> + ReduceTo<FF> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<FF>
        + FrobeniusExtField<FF>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize
        + Valid,
    CpuBackend: RuntimeRootCommitBackend<FF, P, Cfg::ExtField>
        + RecursiveProveBackend<FF, P, Cfg::ExtField>,
{
    let pools = ProfileThreadPools::get();
    let poly_refs: [&P; 1] = [poly];
    let openings = [opening];
    let setup_contribution_mode = profile_setup_contribution_mode();
    tracing::info!(
        label,
        ?setup_contribution_mode,
        "profile setup-contribution mode"
    );
    eprintln!("[{label}] setup_contribution_mode: {setup_contribution_mode:?}");

    let (commitments, proof) = {
        let t0 = Instant::now();
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit(setup, std::slice::from_ref(poly), stack).unwrap();
        report_timing(label, "commit", t0.elapsed().as_secs_f64());

        let commitments = [commitment];
        let selection = Cfg::select_schedule_for_profiles(&CommittedGroupBatchProfile {
            final_group: *commitments[0].profile(),
            precommitteds: Vec::new(),
        })
        .expect("select generated schedule row")
        .selection();
        let t0 = Instant::now();
        let mut prover_transcript = AkitaTranscript::<FF>::new(b"profile");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove(
            setup,
            prover_claims(selection, pt, &poly_refs[..], &commitments[0], hint),
            stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();
        report_timing(label, "prove", t0.elapsed().as_secs_f64());
        (commitments, proof)
    };

    assert_observed_proof_size::<FF, Cfg::ExtField>(label, &proof);
    print_batched_proof_summary::<FF, Cfg::ExtField, D>(label, &proof, plan);
    tracing::info!(
        label,
        ext_degree = Cfg::EXT_DEGREE,
        "profile extension field"
    );
    eprintln!("[{label}] ext_field: ext_degree={}", Cfg::EXT_DEGREE);
    if let Some(plan) = plan {
        if validate_against_planner {
            report_proof_size_against_planner(
                label,
                &proof,
                planned_payload_bytes::<Cfg>(plan, PolynomialGroupLayout::singleton(pt.len())),
                "planned",
                setup_contribution_mode,
                plan,
            );
        }
        emit_runtime_schedule_summary(
            label,
            plan,
            PolynomialGroupLayout::singleton(pt.len()),
            Cfg::decomposition().field_bits(),
        )
        .expect("runtime schedule report geometry");
        emit_proof_tail_report::<FF, Cfg::ExtField>(
            label,
            &proof,
            plan,
            Cfg::decomposition().field_bits(),
        );
    } else {
        let opening_batch =
            OpeningClaimsLayout::new(pt.len(), 1).expect("same-point opening batch");
        let schedule = Cfg::get_params_for_prove(&opening_batch).expect("runtime schedule");
        if validate_against_planner {
            report_proof_size_against_planner(
                label,
                &proof,
                planned_payload_bytes::<Cfg>(&schedule, PolynomialGroupLayout::singleton(pt.len())),
                "runtime schedule",
                setup_contribution_mode,
                &schedule,
            );
        }
        emit_runtime_schedule_summary(
            label,
            &schedule,
            PolynomialGroupLayout::singleton(pt.len()),
            Cfg::decomposition().field_bits(),
        )
        .expect("runtime schedule report geometry");
        emit_proof_tail_report::<FF, Cfg::ExtField>(
            label,
            &proof,
            &schedule,
            Cfg::decomposition().field_bits(),
        );
    }

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools.in_verify_multi(|| {
        if let Some(schedule) = plan {
            let opening_layout =
                OpeningClaimsLayout::new(pt.len(), 1).expect("singleton opening layout");
            AkitaCommitmentScheme::<Cfg>::setup_verifier_for_schedule(
                setup,
                schedule,
                &opening_layout,
            )
            .expect("schedule verifier setup")
        } else {
            AkitaCommitmentScheme::<Cfg>::setup_verifier(setup).expect("verifier setup")
        }
    });
    report_timing(
        label,
        "verifier_setup",
        t_verifier_setup.elapsed().as_secs_f64(),
    );
    let prepare = || {
        verifier_claims(
            Cfg::select_schedule_for_profiles(&CommittedGroupBatchProfile {
                final_group: *commitments[0].profile(),
                precommitteds: Vec::new(),
            })
            .expect("select verifier schedule row")
            .selection(),
            pt,
            &openings[..],
            &commitments[0],
        )
    };
    let verify = |claims| {
        let mut verifier_transcript = AkitaTranscript::<FF>::new(b"profile");
        AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            claims,
            BasisMode::Lagrange,
        )
    };
    run_verifier_timings(label, pools, "profile", prepare, verify);
    report_verifier_ntt_cache_size(
        label,
        verifier_setup
            .verifier_ntt_cache_bytes()
            .expect("verifier NTT cache metrics"),
    );
}

pub(crate) fn run_dense_for<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
    validate_against_planner: bool,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HalvingField
        + HasWide
        + HasCommitAccum
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: FrobeniusExtField<FF>
        + FpExtEncoding<FF>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize
        + Valid,
{
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let original_pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
    let evals: Vec<FF> = if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| FF::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        (0..len)
            .map(|_| FF::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    };
    let poly = DensePoly::<FF>::from_field_evals(nv, D, &evals).unwrap();
    let opening =
        if let Some(base_pt) = degree_one_claim_point_to_base::<FF, Cfg::ExtField>(&original_pt) {
            Cfg::ExtField::lift_base(opening_from_poly::<_, D, _>(
                &poly,
                &base_pt,
                layout,
                BasisMode::Lagrange,
            ))
        } else {
            dense_lagrange_opening_from_evals::<FF, Cfg::ExtField>(&evals, &original_pt)
        };
    let t0 = Instant::now();
    let setup =
        AkitaCommitmentScheme::<Cfg>::setup_prover(RootPolyShape::<FF, D>::num_vars(&poly), 1)
            .unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    if let Some(schedule) = plan {
        prewarm_uniform_profile_execution(&stack, schedule).expect("prewarm profile execution");
    }
    let prepared_ntt_metrics = prepared
        .shared_ntt_cache_metrics()
        .expect("prepared setup NTT cache metrics");
    report_timing(label, "setup_expand", setup_expand_secs);
    report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
    report_timing(label, "setup", t0.elapsed().as_secs_f64());
    let num_setup_field_elements = setup.expanded.shared_matrix().num_field_elements();
    report_setup_sizes(
        label,
        num_setup_field_elements,
        num_setup_field_elements * std::mem::size_of::<FF>(),
        &prepared_ntt_metrics,
    );
    report_crt_profile(
        label,
        prepared
            .shared_ntt_profile(layout.d_a())
            .expect("prepared setup CRT profile"),
    );
    run_prove::<FF, D, Cfg, DensePoly<FF>>(
        label,
        &setup,
        &stack,
        &poly,
        &original_pt,
        opening,
        plan,
        validate_against_planner,
    );
    let post_execution_ntt_metrics = prepared
        .shared_ntt_cache_metrics()
        .expect("post-execution setup NTT cache metrics");
    assert_profile_ntt_cache_did_not_grow(&prepared_ntt_metrics, &post_execution_ntt_metrics);
}

pub(crate) fn run_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
    validate_against_planner: bool,
) where
    FF: CanonicalField
        + CanonicalBytes
        + TranscriptChallenge
        + RandomSampling
        + FromPrimitiveInt
        + PseudoMersenneField
        + HalvingField
        + HasWide
        + HasCommitAccum
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: FrobeniusExtField<FF>
        + FpExtEncoding<FF>
        + HasUnreducedOps
        + HasOptimizedFold
        + AkitaSerialize
        + Valid,
{
    let onehot_poly = make_profile_onehot_poly::<FF>(nv, layout.d_a(), 0xbeef_cafe);
    let mut rng = StdRng::seed_from_u64(0xfeed_face);
    let pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
    let opening = onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(&onehot_poly, &pt);
    let t0 = Instant::now();
    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    if let Some(schedule) = plan {
        prewarm_uniform_profile_execution(&stack, schedule).expect("prewarm profile execution");
    }
    let prepared_ntt_metrics = prepared
        .shared_ntt_cache_metrics()
        .expect("prepared setup NTT cache metrics");
    report_timing(label, "setup_expand", setup_expand_secs);
    report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
    report_timing(label, "setup", t0.elapsed().as_secs_f64());
    let num_setup_field_elements = setup.expanded.shared_matrix().num_field_elements();
    report_setup_sizes(
        label,
        num_setup_field_elements,
        num_setup_field_elements * std::mem::size_of::<FF>(),
        &prepared_ntt_metrics,
    );
    report_crt_profile(
        label,
        prepared
            .shared_ntt_profile(layout.d_a())
            .expect("prepared setup CRT profile"),
    );
    run_prove::<FF, D, Cfg, OneHotPoly<FF, u8>>(
        label,
        &setup,
        &stack,
        &onehot_poly,
        &pt,
        opening,
        plan,
        validate_against_planner,
    );
    let post_execution_ntt_metrics = prepared
        .shared_ntt_cache_metrics()
        .expect("post-execution setup NTT cache metrics");
    assert_profile_ntt_cache_did_not_grow(&prepared_ntt_metrics, &post_execution_ntt_metrics);
}
