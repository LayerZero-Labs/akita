use super::{
    assert_observed_proof_size, assert_profile_ntt_cache_did_not_grow, make_profile_onehot_poly,
    onehot_lagrange_opening, proof_size_budgets, prover_claims, random_claim_point,
    report_proof_size_against_planner, run_verifier_timings, verifier_claims,
};
use crate::parallel::ProfileThreadPools;
use crate::report::{
    emit_native_proof_tail_report, emit_runtime_schedule_summary, print_native_proof_summary,
    report_crt_profile, report_setup_sizes, report_timing, report_verifier_ntt_cache_size,
};
use akita_config::{derive_transcript_grinding_plan, CommitmentConfig};
use akita_cpu_backend::CpuBackend;
use akita_cpu_backend::OneHotPoly;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    BasisMode, CommittedGroupBatchProfile, CommittedGroupParams, FoldSchedule, FpExtEncoding,
    OpeningClaimsLayout, PolynomialGroupLayout, SetupContributionMode,
};
use jolt_field::{CanonicalBytes, CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring};
use jolt_field::{Fold, Unreduced, WithCommitAccumulator};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

pub(crate) fn run_batched_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    scheme: &akita_pcs::AkitaCommitmentScheme<Cfg>,
    label: &str,
    nv: usize,
    num_polys: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
) where
    FF: CanonicalEncoding
        + CanonicalBytes
        + CanonicalEncoding
        + Field
        + Ring
        + PseudoMersenne
        + Field
        + Unreduced
        + WithCommitAccumulator
        + Valid
        + AkitaDeserialize<Context = ()>
        + AkitaSerialize
        + 'static,
    <FF as Unreduced>::Wide: From<FF> + jolt_field::AdditiveGroup,
    Cfg::ExtField: jolt_field::MulBaseUnreduced<FF>
        + ExtField<FF>
        + FpExtEncoding<FF>
        + Unreduced
        + Fold
        + AkitaSerialize
        + Valid,
{
    let group_layout = PolynomialGroupLayout::new(nv, num_polys);
    let polys: Vec<OneHotPoly<FF, u8>> = (0..num_polys)
        .map(|poly_idx| {
            make_profile_onehot_poly::<Cfg>(nv, 0xbeef_cafe ^ ((poly_idx as u64 + 1) << 32))
        })
        .collect();
    let mut point_rng = StdRng::seed_from_u64(0xfeed_face);
    let pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut point_rng);
    let openings: Vec<Cfg::ExtField> = polys
        .iter()
        .map(|poly| onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(poly, &pt))
        .collect();

    let pools = ProfileThreadPools::get();
    let setup_contribution_mode = SetupContributionMode::Direct;
    let (commitments, proof, setup) = {
        let t0 = Instant::now();
        let setup = scheme.setup_prover(nv, num_polys).unwrap();
        let setup_expand_secs = t0.elapsed().as_secs_f64();
        let t_prepare = Instant::now();
        let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
        if let Some(schedule) = plan {
            backend
                .prewarm(schedule)
                .expect("prewarm profile execution");
        }
        let prepared_ntt_metrics = backend
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
            backend
                .shared_ntt_profile(layout.d_a())
                .expect("prepared setup CRT profile"),
        );
        let t0 = Instant::now();
        let source = backend.import_source(polys).unwrap();
        let akita_cpu_backend::CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = backend
            .commit(
                scheme.schedules(),
                &source,
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let commitments = [commitment];
        let selection = scheme
            .schedules()
            .resolve_profiles(&CommittedGroupBatchProfile {
                final_group: *commitments[0].profile(),
                precommitteds: Vec::new(),
            })
            .expect("select generated schedule row")
            .selection();
        let hints = vec![hint];
        report_timing(label, "commit", t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        tracing::info!(
            label,
            ?setup_contribution_mode,
            "profile setup-contribution mode"
        );
        eprintln!("[{label}] setup_contribution_mode: {setup_contribution_mode:?}");
        let proof = scheme
            .batched_prove(
                &setup,
                prover_claims::<Cfg>(
                    scheme.schedules(),
                    selection,
                    &pt[..],
                    &openings,
                    &commitments[0],
                    hints.into_iter().next().unwrap(),
                ),
                &backend,
                b"profile",
                BasisMode::Lagrange,
            )
            .unwrap();
        report_timing(label, "prove", t0.elapsed().as_secs_f64());
        let post_execution_ntt_metrics = backend
            .shared_ntt_cache_metrics()
            .expect("post-execution setup NTT cache metrics");
        assert_profile_ntt_cache_did_not_grow(&prepared_ntt_metrics, &post_execution_ntt_metrics);
        (commitments, proof, setup)
    };
    assert_observed_proof_size(label, &proof);
    let opening_batch =
        OpeningClaimsLayout::from_root_groups(&[], group_layout).expect("same-point opening batch");
    let schedule = scheme
        .schedules()
        .resolve_key(&akita_types::AkitaScheduleLookupKey::single(group_layout))
        .expect("batched schedule")
        .schedule()
        .clone();
    let effective_schedule = plan.unwrap_or(&schedule);
    let grinding_plan = derive_transcript_grinding_plan::<Cfg>(effective_schedule, &opening_batch)
        .expect("profile grinding plan");
    print_native_proof_summary(label, &proof, effective_schedule, &grinding_plan);
    if let Some(plan) = plan {
        report_proof_size_against_planner(
            label,
            &proof,
            proof_size_budgets::<Cfg>(plan, group_layout),
            "planned",
            setup_contribution_mode,
        );
        emit_runtime_schedule_summary(
            label,
            plan,
            group_layout,
            Cfg::decomposition().field_bits(),
            Cfg::EXT_DEGREE,
        )
        .expect("runtime schedule report geometry");
        emit_native_proof_tail_report(label, plan, Cfg::decomposition().field_bits());
    } else {
        report_proof_size_against_planner(
            label,
            &proof,
            proof_size_budgets::<Cfg>(&schedule, group_layout),
            "runtime schedule",
            setup_contribution_mode,
        );
        emit_runtime_schedule_summary(
            label,
            &schedule,
            group_layout,
            Cfg::decomposition().field_bits(),
            Cfg::EXT_DEGREE,
        )
        .expect("runtime schedule report geometry");
        emit_native_proof_tail_report(label, &schedule, Cfg::decomposition().field_bits());
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
        observed_total_bytes = proof.len(),
        "batched planner root-fold summary"
    );

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools.in_verify_multi(|| {
        scheme
            .setup_verifier_for_schedule(&setup, &schedule, &opening_batch)
            .expect("verifier setup")
    });
    report_timing(
        label,
        "verifier_setup",
        t_verifier_setup.elapsed().as_secs_f64(),
    );
    let prepare = || {
        verifier_claims(
            scheme
                .schedules()
                .resolve_profiles(&CommittedGroupBatchProfile {
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
        scheme.batched_verify(
            &proof,
            &verifier_setup,
            b"profile",
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
