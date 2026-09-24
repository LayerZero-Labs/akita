use super::{
    assert_observed_proof_size, assert_profile_ntt_cache_did_not_grow,
    degree_one_claim_point_to_base, make_profile_onehot_poly, onehot_lagrange_opening,
    opening_from_poly, profile_setup_contribution_mode, proof_size_budgets, prover_claims,
    random_claim_point, report_proof_size_against_planner, run_verifier_timings, verifier_claims,
};
use crate::parallel::ProfileThreadPools;
use crate::report::{
    emit_native_proof_tail_report, emit_runtime_schedule_summary, print_native_proof_summary,
    report_crt_profile, report_setup_sizes, report_timing, report_verifier_ntt_cache_size,
};
use akita_config::{derive_transcript_grinding_plan, CommitmentConfig};
use akita_cpu_backend::DensePoly;
use akita_cpu_backend::RootPolyShape;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, SourceHandle};
use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    BasisMode, CommittedGroupBatchProfile, CommittedGroupParams, FoldSchedule, FpExtEncoding,
    OpeningClaimsLayout, PolynomialGroupLayout,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{
    AdditiveGroup, CanonicalBytes, CanonicalEncoding, ExtField, Field, MulBaseUnreduced,
    PseudoMersenne, Ring,
};
use jolt_field::{Fold, Unreduced, WithCommitAccumulator};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
fn run_prove<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    scheme: &AkitaCommitmentScheme<Cfg>,
    setup: &AkitaProverSetup<Cfg::Field>,
    backend: &CpuBackend<Cfg>,
    source: &SourceHandle<FF, Cfg::ExtField, Cfg>,
    pt: &[Cfg::ExtField],
    opening: Cfg::ExtField,
    group_layout: PolynomialGroupLayout,
    plan: Option<&FoldSchedule>,
    // When `false`, skip the planner proof-size upper-bound assertion. That
    // guard validates shipped-catalog schedules against the offline planner
    // estimate; it is meaningless for a synthetic schedule (e.g. the mixed
    // ring-dimension-per-level experiment) that the planner cannot reproduce
    // from its lookup key. The measured proof size and per-level breakdown are
    // still reported in full.
    validate_against_planner: bool,
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
    <FF as Unreduced>::Wide: From<FF> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<FF>
        + ExtField<FF>
        + jolt_field::MulBaseUnreduced<FF>
        + Unreduced
        + Fold
        + AkitaSerialize
        + Valid
        + 'static,
{
    let pools = ProfileThreadPools::get();
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
        let akita_cpu_backend::CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = backend
            .commit(
                source,
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        report_timing(label, "commit", t0.elapsed().as_secs_f64());

        let commitments = [commitment];
        let selection = scheme
            .schedules()
            .resolve_profiles(&CommittedGroupBatchProfile {
                final_group: *commitments[0].profile(),
                precommitteds: Vec::new(),
            })
            .expect("select generated schedule row")
            .selection();
        let t0 = Instant::now();
        let proof = scheme
            .batched_prove(
                setup,
                prover_claims::<Cfg>(
                    scheme.schedules(),
                    selection,
                    pt,
                    &openings,
                    &commitments[0],
                    hint,
                ),
                backend,
                b"profile",
                BasisMode::Lagrange,
            )
            .unwrap();
        report_timing(label, "prove", t0.elapsed().as_secs_f64());
        (commitments, proof)
    };

    assert_observed_proof_size(label, &proof);
    let opening_batch =
        OpeningClaimsLayout::from_root_groups(&[], group_layout).expect("same-point opening batch");
    let runtime_schedule = if plan.is_none() {
        Some(
            scheme
                .schedules()
                .resolve_key(&akita_types::AkitaScheduleLookupKey::single(group_layout))
                .expect("runtime schedule")
                .schedule()
                .clone(),
        )
    } else {
        None
    };
    let effective_schedule = plan.unwrap_or_else(|| {
        runtime_schedule
            .as_ref()
            .expect("runtime schedule was resolved")
    });
    let grinding_plan = derive_transcript_grinding_plan::<Cfg>(effective_schedule, &opening_batch)
        .expect("profile grinding plan");
    print_native_proof_summary(label, &proof, effective_schedule, &grinding_plan);
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
                proof_size_budgets::<Cfg>(plan, group_layout),
                "planned",
                setup_contribution_mode,
            );
        }
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
        let schedule = effective_schedule;
        if validate_against_planner {
            report_proof_size_against_planner(
                label,
                &proof,
                proof_size_budgets::<Cfg>(schedule, group_layout),
                "runtime schedule",
                setup_contribution_mode,
            );
        }
        emit_runtime_schedule_summary(
            label,
            schedule,
            group_layout,
            Cfg::decomposition().field_bits(),
            Cfg::EXT_DEGREE,
        )
        .expect("runtime schedule report geometry");
        emit_native_proof_tail_report(label, schedule, Cfg::decomposition().field_bits());
    }

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools.in_verify_multi(|| {
        if let Some(schedule) = plan {
            let opening_layout = OpeningClaimsLayout::from_root_groups(&[], group_layout)
                .expect("singleton opening layout");
            scheme
                .setup_verifier_for_schedule(setup, schedule, &opening_layout)
                .expect("schedule verifier setup")
        } else {
            scheme.setup_verifier(setup).expect("verifier setup")
        }
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
            pt,
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
    run_verifier_timings(label, pools, "profile", prepare, verify);
    report_verifier_ntt_cache_size(
        label,
        verifier_setup
            .verifier_ntt_cache_bytes()
            .expect("verifier NTT cache metrics"),
    );
}

pub(crate) fn run_dense_for<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    scheme: &akita_pcs::AkitaCommitmentScheme<Cfg>,
    label: &str,
    nv: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
    validate_against_planner: bool,
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
    <FF as Unreduced>::Wide: From<FF> + AdditiveGroup,
    Cfg::ExtField: ExtField<FF>
        + FpExtEncoding<FF>
        + Unreduced
        + MulBaseUnreduced<FF>
        + Fold
        + AkitaSerialize
        + Valid,
{
    let statement_prepare_start = Instant::now();
    let statement_prepare_span =
        tracing::info_span!("profile_dense_prepare_statement", num_vars = nv).entered();
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let original_pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
    let evals: Vec<FF> = {
        let _span = tracing::info_span!("profile_dense_generate_evals", len).entered();
        if decomp.log_commit_bound >= 128 {
            cfg_into_iter!(0..len)
                .map(|index| {
                    let lo = splitmix64(0xbeef_cafe_u64.wrapping_add(2 * index as u64));
                    let hi = splitmix64(0xbeef_cafe_u64.wrapping_add(2 * index as u64 + 1));
                    FF::from_u128_reduced(u128::from(lo) | (u128::from(hi) << 64))
                })
                .collect()
        } else {
            let mask = (2 * half_bound - 1) as u64;
            cfg_into_iter!(0..len)
                .map(|index| {
                    let sampled = (splitmix64(0xbeef_cafe_u64.wrapping_add(index as u64)) & mask)
                        as i64
                        - half_bound;
                    FF::from_i64(sampled)
                })
                .collect()
        }
    };
    let poly = {
        let _span = tracing::info_span!("profile_dense_construct_poly").entered();
        DensePoly::<FF>::from_field_evals(nv, evals).unwrap()
    };
    let opening = {
        let _span = tracing::info_span!("profile_dense_compute_expected_opening").entered();
        if let Some(base_pt) = degree_one_claim_point_to_base::<FF, Cfg::ExtField>(&original_pt) {
            Cfg::ExtField::lift_base(opening_from_poly::<_, D, _>(
                &poly,
                &base_pt,
                layout,
                BasisMode::Lagrange,
            ))
        } else {
            akita_types::derive_tensor_extension_opening_claim::<FF, Cfg::ExtField>(
                nv,
                &poly.field_coeffs()[..len],
                &original_pt,
            )
            .expect("valid dense extension opening")
            .0
        }
    };
    drop(statement_prepare_span);
    report_timing(
        label,
        "statement_prepare",
        statement_prepare_start.elapsed().as_secs_f64(),
    );
    let t0 = Instant::now();
    let setup = scheme
        .setup_prover(RootPolyShape::<FF, D>::num_vars(&poly), 1)
        .unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let backend = CpuBackend::<Cfg>::new(setup.expanded.clone(), scheme.schedules()).unwrap();
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
    let source = backend.import_source(vec![poly]).unwrap();
    run_prove::<FF, D, Cfg>(
        label,
        scheme,
        &setup,
        &backend,
        &source,
        &original_pt,
        opening,
        PolynomialGroupLayout::singleton(nv),
        plan,
        validate_against_planner,
    );
    let post_execution_ntt_metrics = backend
        .shared_ntt_cache_metrics()
        .expect("post-execution setup NTT cache metrics");
    assert_profile_ntt_cache_did_not_grow(&prepared_ntt_metrics, &post_execution_ntt_metrics);
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(crate) fn run_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    scheme: &akita_pcs::AkitaCommitmentScheme<Cfg>,
    label: &str,
    nv: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
    validate_against_planner: bool,
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
    let onehot_poly = make_profile_onehot_poly::<Cfg>(nv, 0xbeef_cafe);
    let mut rng = StdRng::seed_from_u64(0xfeed_face);
    let pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
    let opening = onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(&onehot_poly, &pt);
    let t0 = Instant::now();
    let setup = scheme.setup_prover(nv, 1).unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let backend = CpuBackend::<Cfg>::new(setup.expanded.clone(), scheme.schedules()).unwrap();
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
    let source = backend.import_source(vec![onehot_poly]).unwrap();
    run_prove::<FF, D, Cfg>(
        label,
        scheme,
        &setup,
        &backend,
        &source,
        &pt,
        opening,
        PolynomialGroupLayout::new(nv, 1),
        plan,
        validate_against_planner,
    );
    let post_execution_ntt_metrics = backend
        .shared_ntt_cache_metrics()
        .expect("post-execution setup NTT cache metrics");
    assert_profile_ntt_cache_did_not_grow(&prepared_ntt_metrics, &post_execution_ntt_metrics);
}
