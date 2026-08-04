use crate::parallel::ProfileThreadPools;
use crate::report::{
    emit_proof_tail_report, emit_runtime_schedule_summary, observed_stage3_setup_product_bytes,
    print_batched_proof_summary, report_crt_profile, report_setup_sizes, report_timing,
    report_verifier_ntt_cache_size,
};
use akita_config::{CommitmentConfig, PrecommittedCommitmentConfig, RecursiveCommitmentConfig};
use jolt_field::{
    CanonicalBytes, CanonicalEncoding, ExtField, Field, Fold, PseudoMersenne, Ring, Unreduced,
};

use akita_pcs::test_support::materialize_schedule_setup_prefix_slots;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::compute::{
    OpeningFoldKernel, OpeningFoldPlan, RecursiveProveBackend, RootPolyShape, RootProvePoly,
    RuntimeRootCommitBackend, RuntimeRootCommitPoly, RuntimeRootProvePoly,
};
use akita_prover::{AkitaProverSetup, ComputeBackendSetup, CpuBackend};
use akita_prover::{DensePoly, OneHotIndex, OneHotPoly, ProverOpeningData};
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
    AkitaBatchedProof, AkitaCommitmentHint, BasisMode, Commitment, CommittedGroupParams,
    FoldSchedule, FpExtEncoding, NttCacheKey, OpeningClaims, OpeningClaimsLayout,
    PolynomialGroupClaims, PolynomialGroupLayout, PrecommittedGroupDescriptor,
    SetupContributionMode,
};

use akita_error::AkitaError;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;
use std::time::Instant;

pub(crate) const ONEHOT_K: usize = 256;

fn planned_payload_bytes<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    final_group: PolynomialGroupLayout,
) -> usize {
    let key = akita_types::AkitaScheduleLookupKey {
        final_group,
        precommitteds: schedule
            .root
            .params
            .precommitted_groups
            .iter()
            .map(|group| group.descriptor)
            .collect(),
    };
    if let Some(catalog) = Cfg::schedule_catalog() {
        if let Some(entry) = akita_schedules::generated::table_entry(catalog, &key) {
            return akita_schedules::estimate_proof_bytes(
                entry,
                &key,
                &akita_config::policy_of::<Cfg>(),
                Cfg::ring_challenge_config,
                Cfg::fold_challenge_shape_at_level,
            )
            .expect("generated schedule estimate");
        }
    }
    akita_planner::find_schedule(
        &key,
        &akita_config::policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
    .and_then(|planned| planned.estimate.estimated_proof_payload_bytes())
    .expect("runtime schedule estimate")
}

fn prover_claims<'a, E: Field, P, CommitF: Field>(
    point: &'a [E],
    polynomials: &'a [&'a P],
    commitment: &'a Commitment<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> ProverOpeningData<'a, E, P, CommitF> {
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![E::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

fn verifier_claims<'a, E: Field, C>(
    point: &[E],
    openings: &[E],
    commitment: &'a C,
) -> OpeningClaims<'static, E, &'a C> {
    OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        openings.to_vec(),
        commitment,
    )
    .expect("valid verifier claims group")])
    .expect("valid verifier claims")
}

/// Register every full-envelope NTT dimension selected by a benchmark schedule.
///
/// `ComputeBackendSetup::prepare_setup` promises only the setup-generation
/// dimension. A mixed schedule legitimately consumes divisor dimensions later,
/// so the profile harness makes those cache slots part of preparation rather
/// than letting the first prove operation build them lazily.
fn register_schedule_ntt_contract<FF>(
    setup: &AkitaProverSetup<FF>,
    prepared: &akita_prover::CpuPreparedSetup<FF>,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError>
where
    FF: Field + CanonicalEncoding,
{
    let mut ring_dimensions = BTreeSet::new();
    let mut add_level_dimensions = |params: &CommittedGroupParams| {
        let dims = params.role_dims();
        ring_dimensions.extend([dims.d_a(), dims.d_b(), dims.d_d()]);
        ring_dimensions.extend(params.precommitted_group_iter().flat_map(|group| {
            let group_dims = group.role_dims(dims.d_d());
            [group_dims.d_a(), group_dims.d_b(), group_dims.d_d()]
        }));
    };
    add_level_dimensions(&schedule.root.params.final_group.commitment);
    for fold in &schedule.recursive_folds {
        add_level_dimensions(&fold.params.witness);
    }
    ring_dimensions.insert(schedule.terminal.params.witness.d_a());

    for ring_d in ring_dimensions {
        let key = NttCacheKey::from_envelope(setup.expanded.as_ref(), ring_d)?;
        CpuBackend.register_setup_contract_ntt_slot(prepared, key)?;
    }
    Ok(())
}

fn make_profile_onehot_poly<FF>(layout: &CommittedGroupParams, seed: u64) -> OneHotPoly<FF, u8>
where
    FF: Field + CanonicalEncoding + Ring,
{
    let d = layout.d_a();
    let total_field = layout
        .num_live_blocks
        .checked_mul(layout.num_positions_per_block)
        .and_then(|n| n.checked_mul(d))
        .expect("onehot total field size overflow");
    let num_vars = layout
        .position_index_bits()
        .checked_add(layout.block_index_bits())
        .and_then(|n| n.checked_add(d.trailing_zeros() as usize))
        .expect("onehot variable count overflow");
    assert_eq!(total_field, 1usize << num_vars);
    let onehot_k = onehot_k_for_num_vars(num_vars);
    let total_chunks = total_field / onehot_k;
    assert_eq!(total_chunks * onehot_k, total_field);

    let mut rng = StdRng::seed_from_u64(seed);
    let indices = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    OneHotPoly::<FF, u8>::new(onehot_k, d, indices).expect("profile onehot poly")
}

pub(crate) fn onehot_k_for_num_vars(nv: usize) -> usize {
    let max_supported_log_k = ONEHOT_K.trailing_zeros() as usize;
    if nv >= max_supported_log_k {
        ONEHOT_K
    } else {
        1usize << nv
    }
}

fn assert_observed_proof_size<FF, E>(label: &str, proof: &AkitaBatchedProof<FF, E>)
where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let mut encoded = Vec::with_capacity(proof.size());
    proof
        .serialize_uncompressed(&mut encoded)
        .expect("profile proof serialization should succeed");
    assert_eq!(
        encoded.len(),
        proof.size(),
        "[{label}] proof.size() must match actual uncompressed serialization length"
    );
}

/// Maximum number of bytes by which the planner's header-stripped proof-size
/// estimate is allowed to *exceed* the real serialized proof.
///
/// The offline formula (`akita_types::level_proof_bytes`) assumes every stage-2
/// sumcheck round ships a degree-3 compressed univariate (three challenge-field
/// coefficients). The prover, however, emits a handful of stage-2 rounds at
/// degree 2 — a y-/x-prefix micro-optimization that trims one leading
/// coefficient and that the header-stripped formula deliberately does not
/// model. The real proof is therefore a few challenge elements *smaller* than
/// the estimate, so the estimate stays a conservative upper bound. We accept
/// that small overcount here rather than couple the offline planner to the
/// prover's exact per-round degree schedule. This is a pre-existing inaccuracy
/// (it reproduces on `main` for schedules whose terminal sumcheck folds an
/// odd-shaped witness) and is tracked for a proper fix in
/// `specs/planner-refactor.md`.
///
/// The overcount scales with the number of stage-2 rounds, so it is largest
/// for small-field / many-level schedules: across the profile-bench matrix the
/// current worst case is `dense_fp32_d64` nv26 (planned vs runtime tail sizing).
/// The
/// bound covers those with margin. The `actual <= planned` upper-bound check
/// above is the primary guard against a runtime proof that *grew*; a dropped
/// level (which would inflate the overcount) is independently caught by the
/// planned/proof level-count guard in `scripts/profile_bench_report.py`, and
/// absolute proof growth is bounded by the CI proof-size regression threshold.
const ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES: usize = 3072;

fn terminal_response_z_planner_slack<FF, E>(
    proof: &AkitaBatchedProof<FF, E>,
    schedule: &FoldSchedule,
) -> usize
where
    FF: Field,
    E: Field,
{
    schedule
        .terminal
        .params
        .response_shape
        .layout
        .z_payload_bytes()
        .saturating_sub(
            proof
                .terminal_response()
                .z_payloads
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
        )
}

/// Check the runtime proof size against a planner estimate, tolerating the
/// small, conservative overcount documented on
/// [`ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES`].
fn assert_runtime_matches_planned_proof_size(
    label: &str,
    actual_bytes: usize,
    planned_bytes: usize,
    source: &str,
    extra_slack: usize,
) {
    assert!(
        actual_bytes <= planned_bytes,
        "[{label}] runtime proof bytes {actual_bytes} exceed the {source} proof size \
         {planned_bytes}; the planner estimate must remain an upper bound"
    );
    let overcount = planned_bytes - actual_bytes;
    let accepted = ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES.saturating_add(extra_slack);
    assert!(
        overcount <= accepted,
        "[{label}] {source} proof size {planned_bytes} overcounts the runtime proof bytes \
         {actual_bytes} by {overcount} bytes, exceeding the accepted \
         {accepted}-byte tolerance (stage-2 degree-2 rounds plus segment-typed z slack)"
    );
    if overcount != 0 {
        tracing::warn!(
            label,
            actual_bytes,
            planned_bytes,
            overcount,
            "planner proof-size estimate overcounts the runtime proof (stage-2 degree-2 rounds; \
             see specs/planner-refactor.md)"
        );
        eprintln!(
            "[{label}] NOTE: {source} estimate {planned_bytes} overcounts runtime proof \
             {actual_bytes} by {overcount} bytes (stage-2 degree-2 round micro-optimization; \
             accepted, see specs/planner-refactor.md)"
        );
    }
}

/// Required setup-contribution mode for the config-typed recursive multi-group
/// profile. Scalar profiles are direct by construction.
fn profile_setup_contribution_mode() -> SetupContributionMode {
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

/// Compare the runtime proof against the planner estimate.
///
/// The planner prices the **direct-mode** payload only. In direct mode the
/// whole proof is checked against it. In recursive mode the stage-3
/// setup-product bytes are pure overhead layered on top, so they are stripped
/// before the comparison and reported as an explicit delta instead of being
/// asserted against `schedule.total_bytes`.
fn report_proof_size_against_planner<FF, E>(
    label: &str,
    proof: &AkitaBatchedProof<FF, E>,
    planned_bytes: usize,
    source: &str,
    mode: SetupContributionMode,
    schedule: &FoldSchedule,
) where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let z_slack = terminal_response_z_planner_slack(proof, schedule);
    match mode {
        SetupContributionMode::Direct => {
            assert_runtime_matches_planned_proof_size(
                label,
                proof.size(),
                planned_bytes,
                source,
                z_slack,
            );
        }
        SetupContributionMode::Recursive => {
            let stage3_bytes = observed_stage3_setup_product_bytes(proof);
            let direct_equivalent = proof
                .size()
                .checked_sub(stage3_bytes)
                .expect("stage-3 setup-product bytes are a subset of the serialized proof size");
            let recursive_source = format!("{source} (recursive; stage-3 setup-product excluded)");
            assert_runtime_matches_planned_proof_size(
                label,
                direct_equivalent,
                planned_bytes,
                &recursive_source,
                z_slack,
            );
            tracing::info!(
                label,
                observed_total_bytes = proof.size(),
                stage3_setup_product_bytes = stage3_bytes,
                direct_mode_planner_bytes = planned_bytes,
                "recursive setup-product proof size"
            );
            eprintln!(
                "[{label}] recursive setup: observed={} bytes = direct-mode payload {} \
                 (+/- planner overcount vs {source} {}) + stage-3 setup-product {} bytes",
                proof.size(),
                direct_equivalent,
                planned_bytes,
                stage3_bytes,
            );
        }
    }
}

fn random_claim_point<FF, E>(nv: usize, rng: &mut StdRng) -> Vec<E>
where
    FF: Field + CanonicalEncoding,
    E: ExtField<FF>,
{
    (0..nv)
        .map(|_| {
            let limbs = (0..E::DEGREE)
                .map(|_| FF::from_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            E::from_base_slice(&limbs)
        })
        .collect()
}

fn degree_one_claim_point_to_base<FF, E>(point: &[E]) -> Option<Vec<FF>>
where
    FF: Field,
    E: ExtField<FF>,
{
    (E::DEGREE == 1).then(|| {
        point
            .iter()
            .map(|coord| coord.to_base_vec()[0])
            .collect::<Vec<_>>()
    })
}

fn dense_lagrange_opening_from_evals<FF, E>(evals: &[FF], point: &[E]) -> E
where
    FF: Field,
    E: ExtField<FF>,
{
    assert_eq!(evals.len(), 1usize << point.len());
    let mut layer = evals.iter().copied().map(E::lift_base).collect::<Vec<_>>();
    for &r in point {
        let one_minus_r = E::one() - r;
        let next_len = layer.len() / 2;
        for i in 0..next_len {
            layer[i] = layer[2 * i] * one_minus_r + layer[2 * i + 1] * r;
        }
        layer.truncate(next_len);
    }
    layer[0]
}

fn onehot_lagrange_opening<FF, E, I>(poly: &OneHotPoly<FF, I>, point: &[E]) -> E
where
    FF: Field,
    E: ExtField<FF>,
    I: OneHotIndex,
{
    let onehot_k = poly.onehot_k();
    assert!(onehot_k.is_power_of_two());
    assert_eq!(poly.indices().len() * onehot_k, 1usize << point.len());

    let low_vars = onehot_k.trailing_zeros() as usize;
    let low_weights = lagrange_weights(&point[..low_vars]).expect("valid low opening point");
    let high_weights = lagrange_weights(&point[low_vars..]).expect("valid high opening point");
    poly.indices()
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, hot_idx)| {
            hot_idx.map(|hot_idx| high_weights[chunk_idx] * low_weights[hot_idx.as_usize()])
        })
        .fold(E::zero(), |acc, weight| acc + weight)
}

fn opening_from_poly<'a, FF, const D: usize, P>(
    poly: &'a P,
    point: &[FF],
    layout: &CommittedGroupParams,
    basis: BasisMode,
) -> FF
where
    FF: Field + CanonicalEncoding,
    P: RootProvePoly<FF, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, FF, D>,
{
    let alpha_bits = D.trailing_zeros() as usize;
    let target_num_vars = alpha_bits + layout.position_index_bits() + layout.block_index_bits();
    assert!(
        point.len() <= target_num_vars,
        "opening point length {} exceeds target root arity {}",
        point.len(),
        target_num_vars
    );
    let mut padded_point = point.to_vec();
    padded_point.resize(target_num_vars, FF::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.num_positions_per_block,
        layout.num_live_blocks,
        basis,
    )
    .expect("opening point shape should match layout");

    let opening = OpeningFoldKernel::<P::OpeningView<'a>, FF, D>::evaluate_and_fold(
        &CpuBackend,
        None,
        poly.opening_view().expect("opening view"),
        OpeningFoldPlan::Base {
            live_block_weights: &ring_opening_point.live_block_weights,
            position_weights: &ring_opening_point.position_weights,
            num_positions_per_block: layout.num_positions_per_block,
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner = reduce_inner_opening_to_ring_element::<FF, D>(inner_point, basis)
        .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}

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
    FF: Field
        + CanonicalEncoding
        + CanonicalBytes
        + Ring
        + PseudoMersenne
        + Unreduced
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: FpExtEncoding<FF> + ExtField<FF> + Unreduced + Fold + AkitaSerialize + Valid,
    CpuBackend: RuntimeRootCommitBackend<FF, P, Cfg::ExtField>
        + RecursiveProveBackend<FF, P, Cfg::ExtField>,
{
    let pools = ProfileThreadPools::get();
    let poly_refs: [&P; 1] = [poly];
    let openings = [opening];
    let setup_contribution_mode = SetupContributionMode::Direct;
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
        let t0 = Instant::now();
        let mut prover_transcript = AkitaTranscript::<FF>::new(b"profile");
        let proof = AkitaCommitmentScheme::<Cfg>::batched_prove(
            setup,
            prover_claims(pt, &poly_refs[..], &commitments[0], hint),
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
    tracing::info!(label, ext_degree = Cfg::DEGREE, "profile extension field");
    eprintln!("[{label}] ext_field: ext_degree={}", Cfg::DEGREE);
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
        emit_runtime_schedule_summary(label, plan, 1, Cfg::D, Cfg::decomposition().field_bits());
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
            1,
            Cfg::D,
            Cfg::decomposition().field_bits(),
        );
        emit_proof_tail_report::<FF, Cfg::ExtField>(
            label,
            &proof,
            &schedule,
            Cfg::decomposition().field_bits(),
        );
    }

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools
        .in_verify(|| AkitaCommitmentScheme::<Cfg>::setup_verifier(setup).expect("verifier setup"));
    report_timing(
        label,
        "verifier_setup",
        t_verifier_setup.elapsed().as_secs_f64(),
    );
    let t0 = Instant::now();
    pools.in_verify(|| {
        let mut verifier_transcript = AkitaTranscript::<FF>::new(b"profile");
        match AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims(pt, &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        ) {
            Ok(()) => {}
            Err(e) => {
                let elapsed_s = t0.elapsed().as_secs_f64();
                tracing::error!(label, elapsed_s, error = %e, "verify FAILED");
                eprintln!("[{label}] verify FAILED: {elapsed_s:.6}s ({e})");
                panic!("[{label}] profile verification failed: {e}");
            }
        }
    });
    report_timing(label, "verify OK", t0.elapsed().as_secs_f64());
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
    FF: Field
        + CanonicalEncoding
        + CanonicalBytes
        + Ring
        + PseudoMersenne
        + Unreduced
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: ExtField<FF> + FpExtEncoding<FF> + Unreduced + Fold + AkitaSerialize + Valid,
{
    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let original_pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
    let evals: Vec<FF> = if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| FF::from_u128_reduced(rng.gen::<u128>()))
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
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    if let Some(schedule) = plan {
        register_schedule_ntt_contract(&setup, &prepared, schedule)
            .expect("register schedule NTT contract");
    }
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    report_timing(label, "setup_expand", setup_expand_secs);
    report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
    report_timing(label, "setup", t0.elapsed().as_secs_f64());
    let setup_ring_elements = setup.expanded.shared_matrix().total_ring_elements();
    report_setup_sizes(
        label,
        setup_ring_elements,
        setup_ring_elements * D * std::mem::size_of::<FF>(),
        prepared.shared_ntt_cache_bytes(),
    );
    report_crt_profile(
        label,
        prepared
            .shared_ntt_profile::<D>()
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
}

pub(crate) fn run_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
    validate_against_planner: bool,
) where
    FF: Field
        + CanonicalEncoding
        + CanonicalBytes
        + Ring
        + PseudoMersenne
        + Unreduced
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: ExtField<FF> + FpExtEncoding<FF> + Unreduced + Fold + AkitaSerialize + Valid,
{
    let onehot_poly = make_profile_onehot_poly::<FF>(layout, 0xbeef_cafe);
    let mut rng = StdRng::seed_from_u64(0xfeed_face);
    let pt = random_claim_point::<FF, Cfg::ExtField>(nv, &mut rng);
    let opening = onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(&onehot_poly, &pt);
    let t0 = Instant::now();
    let setup = AkitaCommitmentScheme::<Cfg>::setup_prover(nv, 1).unwrap();
    let setup_expand_secs = t0.elapsed().as_secs_f64();
    let t_prepare = Instant::now();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    if let Some(schedule) = plan {
        register_schedule_ntt_contract(&setup, &prepared, schedule)
            .expect("register schedule NTT contract");
    }
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    report_timing(label, "setup_expand", setup_expand_secs);
    report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
    report_timing(label, "setup", t0.elapsed().as_secs_f64());
    let setup_ring_elements = setup.expanded.shared_matrix().total_ring_elements();
    report_setup_sizes(
        label,
        setup_ring_elements,
        setup_ring_elements * D * std::mem::size_of::<FF>(),
        prepared.shared_ntt_cache_bytes(),
    );
    report_crt_profile(
        label,
        prepared
            .shared_ntt_profile::<D>()
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
}

pub(crate) fn run_batched_onehot<FF, const D: usize, Cfg: CommitmentConfig<Field = FF>>(
    label: &str,
    nv: usize,
    num_polys: usize,
    layout: &CommittedGroupParams,
    plan: Option<&FoldSchedule>,
) where
    FF: Field
        + CanonicalEncoding
        + CanonicalBytes
        + Ring
        + PseudoMersenne
        + Unreduced
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: ExtField<FF> + FpExtEncoding<FF> + Unreduced + Fold + AkitaSerialize + Valid,
{
    let polys: Vec<OneHotPoly<FF, u8>> = (0..num_polys)
        .map(|poly_idx| {
            make_profile_onehot_poly::<FF>(layout, 0xbeef_cafe ^ ((poly_idx as u64 + 1) << 32))
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
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        report_timing(label, "setup_expand", setup_expand_secs);
        report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
        report_timing(label, "setup", t0.elapsed().as_secs_f64());
        let setup_ring_elements = setup.expanded.shared_matrix().total_ring_elements();
        report_setup_sizes(
            label,
            setup_ring_elements,
            setup_ring_elements * D * std::mem::size_of::<FF>(),
            prepared.shared_ntt_cache_bytes(),
        );
        report_crt_profile(
            label,
            prepared
                .shared_ntt_profile::<D>()
                .expect("prepared setup CRT profile"),
        );

        let t0 = Instant::now();
        let (commitment, hint) =
            AkitaCommitmentScheme::<Cfg>::commit::<_, _>(&setup, &polys, &stack).unwrap();
        let commitments = [commitment];
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
            num_polys,
            Cfg::D,
            Cfg::decomposition().field_bits(),
        );
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
            num_polys,
            Cfg::D,
            Cfg::decomposition().field_bits(),
        );
        emit_proof_tail_report::<FF, Cfg::ExtField>(
            label,
            &proof,
            &schedule,
            Cfg::decomposition().field_bits(),
        );
    }
    tracing::info!(label, ext_degree = Cfg::DEGREE, "profile extension field");
    eprintln!("[{label}] ext_field: ext_degree={}", Cfg::DEGREE);
    let root_step = &schedule.root;
    tracing::info!(
        label,
        root_output_witness_len = root_step.output_witness_len,
        observed_total_bytes = proof.size(),
        "batched planner root-fold summary"
    );

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools.in_verify(|| {
        AkitaCommitmentScheme::<Cfg>::setup_verifier(&setup).expect("verifier setup")
    });
    report_timing(
        label,
        "verifier_setup",
        t_verifier_setup.elapsed().as_secs_f64(),
    );
    let t0 = Instant::now();
    pools.in_verify(|| {
        let mut verifier_transcript = AkitaTranscript::<FF>::new(b"profile");
        match AkitaCommitmentScheme::<Cfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims(&pt[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        ) {
            Ok(()) => {}
            Err(e) => {
                let elapsed_s = t0.elapsed().as_secs_f64();
                tracing::error!(label, elapsed_s, error = %e, "verify FAILED");
                eprintln!("[{label}] verify FAILED: {elapsed_s:.6}s ({e})");
                panic!("[{label}] batched profile verification failed: {e}");
            }
        }
    });
    report_timing(label, "verify OK", t0.elapsed().as_secs_f64());
    report_verifier_ntt_cache_size(
        label,
        verifier_setup
            .verifier_ntt_cache_bytes()
            .expect("verifier NTT cache metrics"),
    );
}

pub(crate) fn run_recursive_multi_group_onehot<FF, const D: usize, Cfg>(
    label: &str,
    pre_num_vars: usize,
    final_num_vars: usize,
    final_num_polys: usize,
) where
    Cfg: CommitmentConfig<Field = FF>,
    FF: Field
        + CanonicalEncoding
        + CanonicalBytes
        + Ring
        + PseudoMersenne
        + Unreduced
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: ExtField<FF> + FpExtEncoding<FF> + Unreduced + Fold + AkitaSerialize + Valid,
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

#[cfg(all(not(feature = "profile-onehot-fp128-d64"), not(feature = "profile-ci")))]
pub(crate) fn run_recursive_multi_group_onehot_mixed<FF, const D: usize, Cfg>(
    label: &str,
    pre_num_vars: usize,
    final_num_vars: usize,
    final_num_polys: usize,
) where
    Cfg: CommitmentConfig<Field = FF>,
    FF: CanonicalEncoding
        + CanonicalBytes
        + CanonicalEncoding
        + Field
        + Ring
        + PseudoMersenne
        + Field
        + Unreduced
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: ExtField<FF> + FpExtEncoding<FF> + Unreduced + Fold + AkitaSerialize + Valid,
{
    assert_eq!(
        profile_setup_contribution_mode(),
        SetupContributionMode::Recursive,
        "mixed recursive profile requires AKITA_SETUP_MODE=recursive"
    );
    run_recursive_multi_group_onehot_with_proof_cfg::<FF, D, Cfg, Cfg>(
        label,
        pre_num_vars,
        final_num_vars,
        final_num_polys,
        SetupContributionMode::Recursive,
        false,
    );
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
    FF: CanonicalEncoding
        + CanonicalBytes
        + CanonicalEncoding
        + Field
        + Ring
        + PseudoMersenne
        + Field
        + Unreduced
        + Valid
        + AkitaSerialize
        + 'static,
    Cfg::ExtField: ExtField<FF> + FpExtEncoding<FF> + Unreduced + Fold + AkitaSerialize + Valid,
{
    const PRE_GROUPS: usize = 2;
    const PRE_POLYS_PER_GROUP: usize = 1;

    let total_polys = PRE_GROUPS * PRE_POLYS_PER_GROUP + final_num_polys;
    let pools = ProfileThreadPools::get();

    let mut point_rng = StdRng::seed_from_u64(0xfeed_face);
    let pre_key = PolynomialGroupLayout::new(pre_num_vars, PRE_POLYS_PER_GROUP);
    let pre_opening_batch =
        OpeningClaimsLayout::new(pre_num_vars, PRE_POLYS_PER_GROUP).expect("precommit batch");
    let pre_params = PrecommittedCommitmentConfig::<ProofCfg>::get_params_for_batched_commitment(
        &pre_opening_batch,
    )
    .expect("precommit layout");
    let pre_descriptor = PrecommittedGroupDescriptor::from_params(pre_key, &pre_params);
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(final_num_vars, final_num_polys),
        precommitteds: vec![pre_descriptor; PRE_GROUPS],
    };
    let schedule =
        ProofCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let pre_points = (0..PRE_GROUPS)
        .map(|_| random_claim_point::<FF, Cfg::ExtField>(pre_num_vars, &mut point_rng))
        .collect::<Vec<_>>();
    let final_point = random_claim_point::<FF, Cfg::ExtField>(final_num_vars, &mut point_rng);

    let (proof, schedule, pre_openings, pre_commitments, final_openings, final_commitment, setup) = {
        let t0 = Instant::now();
        let mut setup =
            AkitaCommitmentScheme::<ProofCfg>::setup_prover(final_num_vars, total_polys).unwrap();
        let setup_expand_secs = t0.elapsed().as_secs_f64();
        let t_prepare = Instant::now();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        materialize_schedule_setup_prefix_slots(&mut setup, &CpuBackend, &prepared, &schedule)
            .expect("materialize schedule setup-prefix slots");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        report_timing(label, "setup_expand", setup_expand_secs);
        report_timing(label, "backend_prepare", t_prepare.elapsed().as_secs_f64());
        report_timing(label, "setup", t0.elapsed().as_secs_f64());
        let setup_ring_elements = setup.expanded.shared_matrix().total_ring_elements();
        report_setup_sizes(
            label,
            setup_ring_elements,
            setup_ring_elements * D * std::mem::size_of::<FF>(),
            prepared.shared_ntt_cache_bytes(),
        );
        report_crt_profile(
            label,
            prepared
                .shared_ntt_profile::<D>()
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
                &pre_params,
                0x0bee_fcaf_2100_0000 + group_idx as u64,
            )];
            let openings = polys
                .iter()
                .map(|poly| onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(poly, pre_point))
                .collect::<Vec<_>>();
            let (commitment, hint) =
                AkitaCommitmentScheme::<PrecommittedCommitmentConfig<ProofCfg>>::batched_commit(
                    &setup, &polys, &stack,
                )
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
                    &main_params,
                    0x0bee_fcaf_2800_0000 + poly_idx as u64,
                )
            })
            .collect::<Vec<_>>();
        let final_openings = final_polys
            .iter()
            .map(|poly| onehot_lagrange_opening::<FF, Cfg::ExtField, u8>(poly, &final_point))
            .collect::<Vec<_>>();
        let (final_commitment, final_hint) = AkitaCommitmentScheme::<ProofCfg>::commit_final_group(
            &setup,
            &final_polys,
            &stack,
            pre_keys,
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
            ProverOpeningData::new(
                OpeningClaims::from_groups(prover_groups).expect("prover claims"),
                prover_hints,
                prover_polys,
            )
            .expect("multi-group prover data"),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("multi-group prove");
        report_timing(label, "prove", t_prove.elapsed().as_secs_f64());
        (
            proof,
            schedule,
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
            planned_payload_bytes::<ProofCfg>(
                &schedule,
                PolynomialGroupLayout::new(final_num_vars, final_num_polys),
            ),
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
        total_polys,
        Cfg::D,
        Cfg::decomposition().field_bits(),
    );
    emit_proof_tail_report::<FF, Cfg::ExtField>(
        label,
        &proof,
        &schedule,
        Cfg::decomposition().field_bits(),
    );
    tracing::info!(label, ext_degree = Cfg::DEGREE, "profile extension field");
    eprintln!("[{label}] ext_field: ext_degree={}", Cfg::DEGREE);

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
        PolynomialGroupClaims::new(final_point, final_openings, &final_commitment)
            .expect("final verifier group"),
    );

    let t_verifier_setup = Instant::now();
    let verifier_setup = pools.in_verify(|| {
        AkitaCommitmentScheme::<ProofCfg>::setup_verifier(&setup).expect("verifier setup")
    });
    report_timing(
        label,
        "verifier_setup",
        t_verifier_setup.elapsed().as_secs_f64(),
    );
    let t_verify = Instant::now();
    pools.in_verify(|| {
        let mut verifier_transcript = AkitaTranscript::<FF>::new(b"profile");
        match AkitaCommitmentScheme::<ProofCfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            OpeningClaims::from_groups(verifier_groups).expect("verifier claims"),
            BasisMode::Lagrange,
        ) {
            Ok(()) => {}
            Err(e) => {
                let elapsed_s = t_verify.elapsed().as_secs_f64();
                tracing::error!(label, elapsed_s, error = %e, "verify FAILED");
                eprintln!("[{label}] verify FAILED: {elapsed_s:.6}s ({e})");
                panic!("[{label}] multi-group profile verification failed: {e}");
            }
        }
    });
    report_timing(label, "verify OK", t_verify.elapsed().as_secs_f64());
    report_verifier_ntt_cache_size(
        label,
        verifier_setup
            .verifier_ntt_cache_bytes()
            .expect("verifier NTT cache metrics"),
    );
}
