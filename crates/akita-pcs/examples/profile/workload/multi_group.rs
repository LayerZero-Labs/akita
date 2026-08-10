use super::{
    assert_observed_proof_size, assert_profile_ntt_cache_did_not_grow, make_profile_onehot_poly,
    onehot_lagrange_opening, planned_payload_bytes, random_claim_point,
    report_proof_size_against_planner, run_verifier_timings,
};
use crate::ntt_prewarm::prewarm_uniform_profile_execution;
use crate::parallel::ProfileThreadPools;
use crate::report::{
    emit_proof_tail_report, emit_runtime_schedule_summary, print_batched_proof_summary,
    report_crt_profile, report_setup_sizes, report_timing, report_verifier_ntt_cache_size,
};
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_field::unreduced::{HasCommitAccum, HasOptimizedFold, HasUnreducedOps, HasWide};
use akita_field::{
    CanonicalBytes, CanonicalField, FieldCore, FrobeniusExtField, FromPrimitiveInt, HalvingField,
    PseudoMersenneField, RandomSampling, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::ProverOpeningData;
use akita_prover::{
    commit_setup_prefix, AkitaProverSetup, ComputeBackendSetup, CpuBackend, DensePoly,
    RuntimeCommitBackendFor,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{
    dispatch_for_field, BasisMode, CommittedGroupBatchProfile, FoldSchedule, FpExtEncoding,
    GroupBatchStatement, OpeningClaims, PolynomialGroupClaims, PolynomialGroupLayout,
    SetupContributionMode,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn materialize_schedule_setup_prefix_slots<F, B>(
    setup: &mut AkitaProverSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup,
    schedule: &FoldSchedule,
) -> Result<(), akita_field::AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HalvingField + 'static,
    B: RuntimeCommitBackendFor<F, DensePoly<F>>,
{
    for slot_id in schedule
        .recursive_folds
        .iter()
        .filter_map(|fold| fold.params.incoming_setup_prefix.as_ref())
    {
        if setup.prefix_slots.get(slot_id).is_some() {
            continue;
        }
        let n_prefix = slot_id.n_prefix()?;
        let slot = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            slot_id.d_setup(),
            |D_SETUP| {
                commit_setup_prefix::<F, D_SETUP, B>(
                    &setup.expanded,
                    backend,
                    prepared,
                    &slot_id.commitment_params,
                    n_prefix,
                    slot_id.natural_len,
                )
            }
        )?;
        setup.prefix_slots.insert(slot)?;
    }
    Ok(())
}

/// Setup-contribution mode selected by the benchmark case.
pub(crate) fn profile_setup_contribution_mode() -> SetupContributionMode {
    match std::env::var("AKITA_SETUP_MODE").ok().as_deref() {
        Some("recursive") => SetupContributionMode::Recursive,
        Some("direct") | None => SetupContributionMode::Direct,
        Some(other) => {
            tracing::warn!(
                value = other,
                "unknown AKITA_SETUP_MODE; defaulting to direct"
            );
            eprintln!("[profile] unknown AKITA_SETUP_MODE={other:?}; defaulting to direct");
            SetupContributionMode::Direct
        }
    }
}

pub(crate) fn run_recursive_multi_group_onehot<FF, const D: usize, Cfg>(
    label: &str,
    pre_num_vars: usize,
    final_num_vars: usize,
    final_num_polys: usize,
) where
    Cfg: CommitmentConfig<Field = FF>,
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
    let setup_contribution_mode = profile_setup_contribution_mode();
    match setup_contribution_mode {
        SetupContributionMode::Direct => {
            run_recursive_multi_group_onehot_with_proof_cfg::<FF, D, Cfg, Cfg>(
                label,
                pre_num_vars,
                final_num_vars,
                final_num_polys,
                setup_contribution_mode,
                true,
            )
        }
        SetupContributionMode::Recursive => run_recursive_multi_group_onehot_with_proof_cfg::<
            FF,
            D,
            Cfg,
            RecursiveCommitmentConfig<Cfg>,
        >(
            label,
            pre_num_vars,
            final_num_vars,
            final_num_polys,
            setup_contribution_mode,
            true,
        ),
    }
}

fn run_recursive_multi_group_onehot_with_proof_cfg<FF, const D: usize, Cfg, ProofCfg>(
    label: &str,
    pre_num_vars: usize,
    final_num_vars: usize,
    final_num_polys: usize,
    setup_contribution_mode: SetupContributionMode,
    validate_against_planner: bool,
) where
    Cfg: CommitmentConfig<Field = FF>,
    ProofCfg: CommitmentConfig<Field = FF, ExtField = Cfg::ExtField>,
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
    const PRE_GROUPS: usize = 2;
    const PRE_POLYS_PER_GROUP: usize = 1;

    let total_polys = PRE_GROUPS * PRE_POLYS_PER_GROUP + final_num_polys;
    let pools = ProfileThreadPools::get();

    let mut point_rng = StdRng::seed_from_u64(0xfeed_face);
    let pre_key = PolynomialGroupLayout::new(pre_num_vars, PRE_POLYS_PER_GROUP);
    let pre_descriptor =
        akita_config::committed_group_profile::<ProofCfg>(&pre_key).expect("precommit profile");
    let final_group = PolynomialGroupLayout::new(final_num_vars, final_num_polys);
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group,
        precommitteds: vec![pre_descriptor; PRE_GROUPS],
    };
    let opening_layout = multi_group_key
        .opening_layout()
        .expect("multi-group layout");
    let schedule =
        ProofCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let pre_points = (0..PRE_GROUPS)
        .map(|_| random_claim_point::<FF, Cfg::ExtField>(pre_num_vars, &mut point_rng))
        .collect::<Vec<_>>();
    let final_point = random_claim_point::<FF, Cfg::ExtField>(final_num_vars, &mut point_rng);

    let (
        proof,
        schedule,
        selection,
        pre_openings,
        pre_commitments,
        final_openings,
        final_commitment,
        setup,
    ) = {
        let t0 = Instant::now();
        let mut setup =
            AkitaCommitmentScheme::<ProofCfg>::setup_prover(final_num_vars, total_polys).unwrap();
        let setup_expand_secs = t0.elapsed().as_secs_f64();
        let t_prepare = Instant::now();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
        materialize_schedule_setup_prefix_slots(
            &mut setup,
            &CpuBackend::DEFAULT,
            &prepared,
            &schedule,
        )
        .expect("materialize schedule setup-prefix slots");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        prewarm_uniform_profile_execution(&stack, &schedule).expect("prewarm profile execution");
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
                .shared_ntt_profile(schedule.root.params.final_group.commitment.d_a())
                .expect("prepared setup CRT profile"),
        );
        let mut pre_keys = Vec::with_capacity(PRE_GROUPS);
        let mut pre_commitments = Vec::with_capacity(PRE_GROUPS);
        let mut pre_hints = Vec::with_capacity(PRE_GROUPS);
        let mut pre_polys_by_group = Vec::with_capacity(PRE_GROUPS);
        let mut pre_openings = Vec::with_capacity(PRE_GROUPS);

        let t_commit = Instant::now();
        for (group_idx, pre_point) in pre_points.iter().enumerate() {
            let polys = vec![make_profile_onehot_poly::<FF>(
                pre_num_vars,
                pre_descriptor.inner_commit_matrix.ring_dimension(),
                0x0bee_fcaf_2100_0000 + group_idx as u64,
            )];
            let openings = polys
                .iter()
                .map(|poly| onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(poly, pre_point))
                .collect::<Vec<_>>();
            let (commitment, hint) =
                AkitaCommitmentScheme::<ProofCfg>::commit_group(&setup, &polys, &stack)
                    .expect("precommit");
            pre_keys.push(pre_key);
            pre_commitments.push(commitment);
            pre_hints.push(hint);
            pre_polys_by_group.push(polys);
            pre_openings.push(openings);
        }

        let main_params = schedule.root.params.final_group.commitment.clone();
        let final_polys = (0..final_num_polys)
            .map(|poly_idx| {
                make_profile_onehot_poly::<FF>(
                    final_num_vars,
                    main_params.d_a(),
                    0x0bee_fcaf_2800_0000 + poly_idx as u64,
                )
            })
            .collect::<Vec<_>>();
        let final_openings = final_polys
            .iter()
            .map(|poly| onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(poly, &final_point))
            .collect::<Vec<_>>();
        let (final_commitment, final_hint, _selection) =
            AkitaCommitmentScheme::<ProofCfg>::commit_final_group(
                &setup,
                &final_polys,
                &stack,
                pre_commitments.iter().map(|group| group.profile).collect(),
            )
            .expect("final multi-group commitment");
        report_timing(label, "commit", t_commit.elapsed().as_secs_f64());

        let pre_refs_by_group = pre_polys_by_group
            .iter()
            .map(|polys| polys.iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let final_refs = final_polys.iter().collect::<Vec<_>>();

        let mut prover_groups = Vec::with_capacity(PRE_GROUPS + 1);
        for (group_idx, openings) in pre_openings.iter().enumerate() {
            prover_groups.push(
                PolynomialGroupClaims::new(
                    pre_points[group_idx].clone(),
                    openings.clone(),
                    pre_commitments[group_idx].clone(),
                )
                .expect("pre prover group"),
            );
        }
        prover_groups.push(
            PolynomialGroupClaims::new(
                final_point.clone(),
                final_openings.clone(),
                final_commitment.clone(),
            )
            .expect("final prover group"),
        );
        let mut prover_polys = pre_refs_by_group
            .iter()
            .map(|refs| refs.as_slice())
            .collect::<Vec<_>>();
        prover_polys.push(final_refs.as_slice());
        let mut prover_hints = pre_hints;
        prover_hints.push(final_hint);
        let selection = ProofCfg::select_schedule_for_profiles(&CommittedGroupBatchProfile {
            final_group: *final_commitment.profile(),
            precommitteds: pre_commitments
                .iter()
                .map(|commitment| *commitment.profile())
                .collect(),
        })
        .expect("select multi-group generated schedule row")
        .selection();

        let t_prove = Instant::now();
        let mut prover_transcript = AkitaTranscript::<FF>::new(b"profile");
        tracing::info!(
            label,
            ?setup_contribution_mode,
            "profile setup-contribution mode"
        );
        eprintln!("[{label}] setup_contribution_mode: {setup_contribution_mode:?}");
        let proof = AkitaCommitmentScheme::<ProofCfg>::batched_prove::<_, _, _>(
            &setup,
            (
                selection,
                ProverOpeningData::new(
                    OpeningClaims::from_groups(prover_groups).expect("prover claims"),
                    prover_hints,
                    prover_polys,
                )
                .expect("multi-group prover data"),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("multi-group prove");
        report_timing(label, "prove", t_prove.elapsed().as_secs_f64());
        let post_execution_ntt_metrics = prepared
            .shared_ntt_cache_metrics()
            .expect("post-execution setup NTT cache metrics");
        assert_profile_ntt_cache_did_not_grow(&prepared_ntt_metrics, &post_execution_ntt_metrics);
        (
            proof,
            schedule,
            selection,
            pre_openings,
            pre_commitments,
            final_openings,
            final_commitment,
            setup,
        )
    };

    assert_observed_proof_size::<FF, Cfg::ExtField>(label, &proof);
    print_batched_proof_summary::<FF, Cfg::ExtField, D>(label, &proof, Some(&schedule));
    if validate_against_planner {
        report_proof_size_against_planner(
            label,
            &proof,
            planned_payload_bytes::<ProofCfg>(&schedule, final_group),
            "planned",
            setup_contribution_mode,
            &schedule,
        );
    } else {
        tracing::info!(
            label,
            "skipping shipped-planner proof-size comparison for synthetic mixed-D schedule"
        );
    }
    emit_runtime_schedule_summary(
        label,
        &schedule,
        final_group,
        Cfg::decomposition().field_bits(),
    );
    emit_proof_tail_report::<FF, Cfg::ExtField>(
        label,
        &proof,
        &schedule,
        Cfg::decomposition().field_bits(),
    );
    tracing::info!(
        label,
        ext_degree = Cfg::EXT_DEGREE,
        "profile extension field"
    );
    eprintln!("[{label}] ext_field: ext_degree={}", Cfg::EXT_DEGREE);

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools.in_verify_multi(|| {
        AkitaCommitmentScheme::<ProofCfg>::setup_verifier_for_schedule(
            &setup,
            &schedule,
            &opening_layout,
        )
        .expect("verifier setup")
    });
    report_timing(
        label,
        "verifier_setup",
        t_verifier_setup.elapsed().as_secs_f64(),
    );
    let prepare = || {
        let mut verifier_groups = Vec::with_capacity(PRE_GROUPS + 1);
        for (group_idx, openings) in pre_openings.iter().enumerate() {
            verifier_groups.push(
                PolynomialGroupClaims::new(
                    pre_points[group_idx].clone(),
                    openings.clone(),
                    &pre_commitments[group_idx],
                )
                .expect("pre verifier group"),
            );
        }
        verifier_groups.push(
            PolynomialGroupClaims::new(
                final_point.clone(),
                final_openings.clone(),
                &final_commitment,
            )
            .expect("final verifier group"),
        );
        GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(verifier_groups).expect("verifier claims"),
        )
        .expect("verifier statement")
    };
    let verify = |statement| {
        let mut verifier_transcript = AkitaTranscript::<FF>::new(b"profile");
        AkitaCommitmentScheme::<ProofCfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            statement,
            BasisMode::Lagrange,
        )
    };
    run_verifier_timings(label, pools, "multi-group profile", prepare, verify);
    report_verifier_ntt_cache_size(
        label,
        verifier_setup
            .verifier_ntt_cache_bytes()
            .expect("verifier NTT cache metrics"),
    );
}
