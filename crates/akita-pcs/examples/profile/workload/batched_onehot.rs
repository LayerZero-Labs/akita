use super::{
    assert_observed_proof_size, assert_profile_ntt_cache_did_not_grow, make_profile_onehot_poly,
    onehot_lagrange_opening, planned_payload_bytes, prover_claims, random_claim_point,
    report_proof_size_against_planner, run_verifier_timings, verifier_claims,
};
use crate::ntt_prewarm::prewarm_uniform_profile_execution;
use crate::parallel::ProfileThreadPools;
use crate::report::{
    emit_proof_tail_report, emit_runtime_schedule_summary, print_batched_proof_summary,
    report_crt_profile, report_setup_sizes, report_timing, report_verifier_ntt_cache_size,
};
use akita_config::CommitmentConfig;
use akita_field::unreduced::{HasCommitAccum, HasOptimizedFold, HasUnreducedOps, HasWide};
use akita_field::{
    CanonicalBytes, CanonicalField, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::OneHotPoly;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{
    BasisMode, CommittedGroupBatchProfile, CommittedGroupParams, FoldSchedule, FpExtEncoding,
    OpeningClaimsLayout, PolynomialGroupLayout, SetupContributionMode,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

pub(crate) fn run_batched_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    num_polys: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
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
    let polys: Vec<OneHotPoly<FF, u8>> = (0..num_polys)
        .map(|poly_idx| {
            make_profile_onehot_poly::<FF>(
                nv,
                layout.d_a(),
                0xbeef_cafe ^ ((poly_idx as u64 + 1) << 32),
            )
        })
        .collect();
    let mut point_rng = StdRng::seed_from_u64(0xfeed_face);
    let pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut point_rng);
    let openings: Vec<Cfg::ExtField> = polys
        .iter()
        .map(|poly| onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(poly, &pt))
        .collect();
    let poly_refs: Vec<&OneHotPoly<FF, u8>> = polys.iter().collect();

    let pools = ProfileThreadPools::get();
    let setup_contribution_mode = SetupContributionMode::Direct;
    let (commitments, proof, setup) = {
        let t0 = Instant::now();
        let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, num_polys).unwrap();
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
        let t0 = Instant::now();
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack).unwrap();
        let commitments = [commitment];
        let selection = Cfg::select_schedule_for_profiles(&CommittedGroupBatchProfile {
            final_group: *commitments[0].profile(),
            precommitteds: Vec::new(),
        })
        .expect("select generated schedule row")
        .selection();
        let hints = vec![hint];
        report_timing(label, "commit", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        let mut prover_transcript = AkitaTranscript::<FF>::new(b"profile");
        tracing::info!(
            label,
            ?setup_contribution_mode,
            "profile setup-contribution mode"
        );
        eprintln!("[{label}] setup_contribution_mode: {setup_contribution_mode:?}");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove::<_, _, _>(
            &setup,
            prover_claims(
                selection,
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();
        report_timing(label, "prove", t0.elapsed().as_secs_f64());
        let post_execution_ntt_metrics = prepared
            .shared_ntt_cache_metrics()
            .expect("post-execution setup NTT cache metrics");
        assert_profile_ntt_cache_did_not_grow(&prepared_ntt_metrics, &post_execution_ntt_metrics);
        (commitments, proof, setup)
    };
    assert_observed_proof_size::<FF, Cfg::ExtField>(label, &proof);
    print_batched_proof_summary::<FF, Cfg::ExtField, D>(label, &proof, plan);
    let opening_batch = OpeningClaimsLayout::new(nv, num_polys).expect("same-point opening batch");
    let schedule = Cfg::get_params_for_prove(&opening_batch).expect("batched schedule");
    if let Some(plan) = plan {
        report_proof_size_against_planner(
            label,
            &proof,
            planned_payload_bytes::<Cfg>(plan, PolynomialGroupLayout::new(nv, num_polys)),
            "planned",
            setup_contribution_mode,
            plan,
        );
        emit_runtime_schedule_summary(
            label,
            plan,
            PolynomialGroupLayout::new(nv, num_polys),
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
        report_proof_size_against_planner(
            label,
            &proof,
            planned_payload_bytes::<Cfg>(&schedule, PolynomialGroupLayout::new(nv, num_polys)),
            "runtime schedule",
            setup_contribution_mode,
            &schedule,
        );
        emit_runtime_schedule_summary(
            label,
            &schedule,
            PolynomialGroupLayout::new(nv, num_polys),
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
    tracing::info!(
        label,
        ext_degree = Cfg::EXT_DEGREE,
        "profile extension field"
    );
    eprintln!("[{label}] ext_field: ext_degree={}", Cfg::EXT_DEGREE);
    let root_step = &schedule.root;
    tracing::info!(
        label,
        root_output_witness_len = root_step.output_witness_len,
        observed_total_bytes = proof.size(),
        "batched planner root-fold summary"
    );

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools.in_verify_multi(|| {
        AkitaCommitmentScheme::<Cfg>::setup_verifier_for_schedule(&setup, &schedule, &opening_batch)
            .expect("verifier setup")
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
            &pt[..],
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
    run_verifier_timings(label, pools, "batched profile", prepare, verify);
    report_verifier_ntt_cache_size(
        label,
        verifier_setup
            .verifier_ntt_cache_bytes()
            .expect("verifier NTT cache metrics"),
    );
}
