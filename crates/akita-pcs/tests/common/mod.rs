#![allow(dead_code)]

pub(super) use akita_config::proof_optimized::fp128;
pub(super) use akita_config::CommitmentConfig;
use akita_config::{PrecommittedCommitmentConfig, RecursiveCommitmentConfig};
use akita_field::Zero;
pub(super) use akita_field::{
    AkitaError, CanonicalBytes, CanonicalField, FieldCore, TranscriptChallenge,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource, RootPolyShape};
pub(super) use akita_prover::DensePoly;
pub(super) use akita_prover::OneHotPoly;
use akita_prover::{ComputeBackendSetup, CpuBackend};
pub(super) use akita_prover::{ProverOpeningData, SelectedProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress};
pub(super) use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaCommitmentHint,
    BasisMode, CommittedGroup, OpeningClaims, PolynomialGroupClaims,
};
use akita_types::{
    AkitaBatchedProof, AkitaScheduleLookupKey, CommittedGroupBatchProfile, CommittedGroupProfile,
    GroupBatchStatement, OpeningClaimsLayout, PolynomialGroupLayout, SetupSumcheckProof,
};
pub(super) use akita_types::{CommittedGroupParams, FoldSchedule};
pub(super) use rand::rngs::StdRng;
pub(super) use rand::{Rng, SeedableRng};
use std::sync::Once;

#[cfg(feature = "logging-transcript")]
use akita_transcript::TranscriptEvent;
use akita_transcript::{labels, AkitaTranscript, Transcript};

pub(super) type F = fp128::Field;
pub(super) const STACK_SIZE: usize = 256 * 1024 * 1024;

// Bare presets: test-only non-singleton batched opening shapes
// fall through to the offline DP planner on table miss via the default
// `runtime_schedule` fallback.
pub(super) type OneHotCfg = fp128::OneHot;
pub(super) const ONEHOT_D: usize = OneHotCfg::D;
// `fp128::OneHot` requires K=256 one-hot schedules
// ring elements), so the committed poly has `2^nv / K` chunks, not one chunk
// per ring element.
pub(super) const ONEHOT_K: usize = 256;

pub(super) type DenseCfg = fp128::Dense;
pub(super) const DENSE_D: usize = DenseCfg::D;

static INIT_RAYON: Once = Once::new();

pub(super) fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

pub(super) fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

pub(super) fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

/// Canonical byte encoding of an ordered logging-transcript event stream.
#[cfg(feature = "logging-transcript")]
pub(super) fn serialize_transcript_events(events: &[TranscriptEvent]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for event in events {
        match event {
            TranscriptEvent::Preamble {
                bytes_digest,
                bytes_len,
            } => {
                bytes.push(0);
                bytes.extend_from_slice(bytes_digest);
                bytes.extend_from_slice(&u64::try_from(*bytes_len).unwrap().to_le_bytes());
            }
            TranscriptEvent::Absorb {
                label,
                bytes_digest,
                bytes_len,
            } => {
                bytes.push(1);
                bytes.extend_from_slice(&u64::try_from(label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(label);
                bytes.extend_from_slice(bytes_digest);
                bytes.extend_from_slice(&u64::try_from(*bytes_len).unwrap().to_le_bytes());
            }
            TranscriptEvent::Squeeze { label, len } => {
                bytes.push(2);
                bytes.extend_from_slice(&u64::try_from(label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(label);
                bytes.extend_from_slice(&u64::try_from(*len).unwrap().to_le_bytes());
            }
            TranscriptEvent::Wire {
                label,
                bytes_digest,
                bytes_len,
            } => {
                bytes.push(3);
                bytes.extend_from_slice(&u64::try_from(label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(label);
                bytes.extend_from_slice(bytes_digest);
                bytes.extend_from_slice(&u64::try_from(*bytes_len).unwrap().to_le_bytes());
            }
        }
    }
    bytes
}

/// Canonical Stage 1 payload bytes in fold-wire order.
pub(super) fn serialize_stage1_payload<FF>(proof: &akita_types::AkitaStage1Proof<FF>) -> Vec<u8>
where
    FF: FieldCore + AkitaSerialize,
{
    let mut bytes = Vec::new();
    for stage in &proof.stages {
        stage
            .sumcheck_proof
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .expect("serialize Stage 1 sumcheck");
        for claim in &stage.child_claims {
            claim
                .serialize_with_mode(&mut bytes, Compress::Yes)
                .expect("serialize Stage 1 child claim");
        }
    }
    proof
        .range_image_evaluation
        .serialize_with_mode(&mut bytes, Compress::Yes)
        .expect("serialize Stage 1 range-image claim");
    bytes
}

/// Stable digest used by versioned protocol epochs.
pub(super) fn protocol_epoch_digest<FF>(payload: &[u8]) -> String
where
    FF: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
{
    let mut transcript = AkitaTranscript::<FF>::new(b"akita/protocol-epoch/digest");
    transcript.append_bytes(labels::ABSORB_OPENING_PAYLOAD, payload);
    transcript
        .challenge_scalar(labels::CHALLENGE_SUMCHECK_BATCH)
        .to_bytes_le_vec()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn prove_input<'a, Cfg, P>(
    point: &'a [Cfg::ExtField],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<Cfg::Field>,
    hint: AkitaCommitmentHint<Cfg::Field>,
) -> SelectedProverOpeningData<
    'a,
    Cfg::ExtField,
    akita_prover::PreparedProverGroup<'a, P>,
    Cfg::Field,
>
where
    Cfg: CommitmentConfig,
    P: akita_prover::RootPolyMeta<Cfg::Field>,
{
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![Cfg::ExtField::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    let profiles = CommittedGroupBatchProfile {
        final_group: *commitment.profile(),
        precommitteds: Vec::new(),
    };
    let selection = Cfg::select_schedule_for_profiles(&profiles)
        .expect("select prover schedule")
        .selection();
    (
        selection,
        ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
            .expect("valid prover opening data"),
    )
}

pub(super) fn selected_prover_data<'a, Cfg, P>(
    claims: OpeningClaims<'a, Cfg::ExtField, CommittedGroup<Cfg::Field>>,
    hints: Vec<AkitaCommitmentHint<Cfg::Field>>,
    polynomials: Vec<&'a [&'a P]>,
) -> SelectedProverOpeningData<
    'a,
    Cfg::ExtField,
    akita_prover::PreparedProverGroup<'a, P>,
    Cfg::Field,
>
where
    Cfg: CommitmentConfig,
    P: akita_prover::RootPolyMeta<Cfg::Field>,
{
    let (final_group, precommitteds) = claims
        .groups()
        .split_last()
        .expect("prover data requires a group");
    let profiles = CommittedGroupBatchProfile {
        final_group: *final_group.commitment().profile(),
        precommitteds: precommitteds
            .iter()
            .map(|group| *group.commitment().profile())
            .collect(),
    };
    let selection = Cfg::select_schedule_for_profiles(&profiles)
        .expect("select prover schedule")
        .selection();
    (
        selection,
        ProverOpeningData::new(claims, hints, polynomials).expect("valid selected prover data"),
    )
}

pub(super) fn selected_statement<'a, Cfg>(
    claims: OpeningClaims<'a, Cfg::ExtField, &'a CommittedGroup<Cfg::Field>>,
) -> GroupBatchStatement<'a, Cfg::ExtField, Cfg::Field>
where
    Cfg: CommitmentConfig,
{
    let (final_group, precommitteds) = claims
        .groups()
        .split_last()
        .expect("verifier statement requires a group");
    let profiles = CommittedGroupBatchProfile {
        final_group: *final_group.commitment().profile(),
        precommitteds: precommitteds
            .iter()
            .map(|group| *group.commitment().profile())
            .collect(),
    };
    let selection = Cfg::select_schedule_for_profiles(&profiles)
        .expect("select verifier statement schedule")
        .selection();
    GroupBatchStatement::new(selection, claims).expect("valid selected verifier statement")
}

pub(super) fn verify_input<'a, Cfg>(
    point: &'a [Cfg::ExtField],
    openings: &'a [Cfg::ExtField],
    commitment: &'a CommittedGroup<Cfg::Field>,
) -> GroupBatchStatement<'a, Cfg::ExtField, Cfg::Field>
where
    Cfg: CommitmentConfig,
{
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        openings.to_vec(),
        commitment,
    )
    .expect("valid verifier claims group")])
    .expect("valid verifier input");
    let profiles = CommittedGroupBatchProfile {
        final_group: *commitment.profile(),
        precommitteds: Vec::new(),
    };
    let selection = Cfg::select_schedule_for_profiles(&profiles)
        .expect("select verifier statement schedule")
        .selection();
    GroupBatchStatement::new(selection, claims).expect("valid verifier statement")
}

pub(super) fn opening_from_poly<'a, const D: usize, P>(
    poly: &'a P,
    point: &[F],
    layout: &CommittedGroupParams,
) -> F
where
    P: RootOpeningSource<F, D> + RootPolyShape<F, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, F, D>,
{
    opening_from_poly_with_basis::<D, P>(poly, point, layout, BasisMode::Lagrange)
}

pub(super) fn opening_from_poly_for_layout<'a, P>(
    poly: &'a P,
    point: &[F],
    layout: &CommittedGroupParams,
) -> F
where
    P: RootOpeningSource<F, 64>
        + RootPolyShape<F, 64>
        + RootOpeningSource<F, 128>
        + RootPolyShape<F, 128>
        + RootOpeningSource<F, 256>
        + RootPolyShape<F, 256>,
    CpuBackend: OpeningFoldKernel<<P as RootOpeningSource<F, 64>>::OpeningView<'a>, F, 64>
        + OpeningFoldKernel<<P as RootOpeningSource<F, 128>>::OpeningView<'a>, F, 128>
        + OpeningFoldKernel<<P as RootOpeningSource<F, 256>>::OpeningView<'a>, F, 256>,
{
    match layout.d_a() {
        64 => opening_from_poly::<64, _>(poly, point, layout),
        128 => opening_from_poly::<128, _>(poly, point, layout),
        256 => opening_from_poly::<256, _>(poly, point, layout),
        dimension => panic!("unsupported test opening ring dimension D={dimension}"),
    }
}

pub(super) fn opening_from_poly_with_basis<'a, const D: usize, P>(
    poly: &'a P,
    point: &[F],
    layout: &CommittedGroupParams,
    basis_mode: BasisMode,
) -> F
where
    P: RootOpeningSource<F, D> + RootPolyShape<F, D>,
    CpuBackend: OpeningFoldKernel<P::OpeningView<'a>, F, D>,
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
    padded_point.resize(target_num_vars, F::zero());

    let inner_point = &padded_point[..alpha_bits];
    let reduced_point = &padded_point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.num_positions_per_block,
        layout.num_live_blocks,
        basis_mode,
    )
    .expect("opening point shape should match layout");

    let opening = OpeningFoldKernel::<P::OpeningView<'a>, F, D>::evaluate_and_fold(
        &CpuBackend::DEFAULT,
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
    let packed_inner = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis_mode)
        .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}

pub(super) fn make_onehot_poly(layout: &CommittedGroupParams, seed: u64) -> OneHotPoly<F, u8> {
    // `2^nv = (num_live_blocks · num_positions_per_block) · D` field elements, grouped into
    // `2^nv / K` one-hot chunks of size `K`.
    let root_d = layout.d_a();
    let total_field = layout.num_live_blocks * layout.num_positions_per_block * root_d;
    let total_chunks = total_field / ONEHOT_K;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, u8>::new(ONEHOT_K, root_d, indices).expect("onehot poly")
}

pub(super) fn make_dense_poly(nv: usize, seed: u64) -> DensePoly<F> {
    let evals = dense_field_evals(nv, seed);
    DensePoly::<F>::from_field_evals(nv, DENSE_D, &evals).expect("dense poly")
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub(super) fn dense_field_evals(nv: usize, seed: u64) -> Vec<F> {
    let n = 1usize << nv;
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        let v = splitmix64_next(&mut state);
        out.push(F::from_canonical_u128_reduced(v as u128));
    }
    out
}

fn multi_group_root_params(schedule: &FoldSchedule) -> &CommittedGroupParams {
    &schedule.root.params.final_group.commitment
}

fn schedule_uses_setup_prefix(schedule: &FoldSchedule) -> bool {
    schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.incoming_setup_prefix.is_some())
}

fn proof_has_recursive_setup_sumcheck(proof: &AkitaBatchedProof<F, F>) -> bool {
    proof.root.stage3_sumcheck_proof.is_some()
        || proof
            .recursive_folds
            .iter()
            .any(|step| step.stage3_sumcheck_proof.is_some())
}

fn first_stage3_proof_mut(
    proof: &mut AkitaBatchedProof<F, F>,
) -> Option<&mut SetupSumcheckProof<F>> {
    if let Some(stage3) = proof.root.stage3_sumcheck_proof.as_mut() {
        return Some(stage3);
    }
    proof
        .recursive_folds
        .iter_mut()
        .find_map(|fold| fold.stage3_sumcheck_proof.as_mut())
}

/// Drives the shared recursive setup-offload profile end to end: two precommitted
/// singleton groups at `nv=16` frozen with exact fixed-root ranks, a two-polynomial
/// main group at `nv=32`, a recursive proof that offloads the setup contribution,
/// a serialization round-trip, an honest verify, and a tampered-opening rejection.
///
/// `BaseCfg` selects the physical witness layout (single-chunk vs chunked); the
/// recursion adapter and exact-precommit adapter are derived from it.
/// `on_schedule` runs profile-specific assertions against the resolved schedule.
pub(super) fn recursive_multi_group_round_trip<BaseCfg>(
    transcript_domain: &'static [u8],
    on_schedule: fn(&FoldSchedule),
) where
    BaseCfg: CommitmentConfig<Field = F, ExtField = F>,
{
    type Recursive<BaseCfg> = AkitaCommitmentScheme<RecursiveCommitmentConfig<BaseCfg>>;
    type Precommitted<BaseCfg> = AkitaCommitmentScheme<PrecommittedCommitmentConfig<BaseCfg>>;

    const PRE_NV: usize = 16;
    const FINAL_NV: usize = 32;
    const PRE_GROUPS: usize = 2;
    const PRE_GROUP_SIZE: usize = 1;
    const FINAL_GROUP_SIZE: usize = 2;
    const TOTAL_GROUP_SIZE: usize = PRE_GROUPS * PRE_GROUP_SIZE + FINAL_GROUP_SIZE;

    init_rayon_pool();
    run_on_large_stack(move || {
        let pre_key = PolynomialGroupLayout::new(PRE_NV, PRE_GROUP_SIZE);
        let pre_layout =
            PrecommittedCommitmentConfig::<BaseCfg>::get_params_for_batched_commitment(
                &OpeningClaimsLayout::new(PRE_NV, PRE_GROUP_SIZE).expect("precommit batch"),
            )
            .expect("precommit params");
        let pre_frozen = CommittedGroupProfile::from_params(pre_key, &pre_layout);
        let schedule_key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE),
            precommitteds: vec![pre_frozen, pre_frozen],
        };
        let opening_layout = schedule_key.opening_layout().expect("opening layout");
        let schedule = RecursiveCommitmentConfig::<BaseCfg>::runtime_schedule(schedule_key)
            .expect("recursive profile schedule resolves");
        assert!(
            schedule_uses_setup_prefix(&schedule),
            "recursive profile must carry setup-prefix metadata"
        );
        on_schedule(&schedule);
        let root_params = multi_group_root_params(&schedule);

        let setup = Recursive::<BaseCfg>::setup_prover(FINAL_NV, TOTAL_GROUP_SIZE)
            .expect("recursive setup");
        assert!(
            !setup.prefix_slots.is_empty(),
            "recursive setup must precompute setup-prefix slots for the generated profile"
        );
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");

        let mut pre_polys_by_group = Vec::new();
        let mut pre_commitments = Vec::new();
        let mut pre_hints = Vec::new();
        for group_idx in 0..PRE_GROUPS {
            let poly = make_onehot_poly(&pre_layout, 0x0bee_fcaf_2026_0000 + group_idx as u64);
            let (commitment, hint) = Precommitted::<BaseCfg>::batched_commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
            )
            .expect("precommit group");
            pre_polys_by_group.push(vec![poly]);
            pre_commitments.push(commitment);
            pre_hints.push(hint);
        }

        let final_polys: Vec<OneHotPoly<F, u8>> = (0..FINAL_GROUP_SIZE)
            .map(|poly_idx| make_onehot_poly(root_params, 0x0bee_fcaf_2026_1000 + poly_idx as u64))
            .collect();
        let (final_commitment, final_hint, _selection) = Recursive::<BaseCfg>::commit_final_group(
            &setup,
            &final_polys,
            &stack,
            pre_commitments.iter().map(|group| group.profile).collect(),
        )
        .expect("final generated-profile commitment");

        let point = random_point(FINAL_NV, 0xcafe_2026_0001);
        let pre_openings: Vec<Vec<F>> = pre_polys_by_group
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly_for_layout(poly, &point[..PRE_NV], &pre_layout))
                    .collect()
            })
            .collect();
        let final_openings: Vec<F> = final_polys
            .iter()
            .map(|poly| opening_from_poly_for_layout(poly, &point, root_params))
            .collect();

        let pre_refs_by_group: Vec<Vec<&OneHotPoly<F, u8>>> = pre_polys_by_group
            .iter()
            .map(|polys| polys.iter().collect())
            .collect();
        let final_refs: Vec<&OneHotPoly<F, u8>> = final_polys.iter().collect();

        let mut prover_groups = Vec::new();
        for (group_idx, openings) in pre_openings.iter().enumerate() {
            prover_groups.push(
                PolynomialGroupClaims::new(
                    point[..PRE_NV].to_vec(),
                    openings.clone(),
                    pre_commitments[group_idx].clone(),
                )
                .expect("pre prover group"),
            );
        }
        prover_groups.push(
            PolynomialGroupClaims::new(
                point.clone(),
                final_openings.clone(),
                final_commitment.clone(),
            )
            .expect("final prover group"),
        );

        let mut prover_polys: Vec<&[&OneHotPoly<F, u8>]> = Vec::new();
        for refs in &pre_refs_by_group {
            prover_polys.push(&refs[..]);
        }
        prover_polys.push(&final_refs[..]);
        let mut prover_hints = pre_hints;
        prover_hints.push(final_hint);

        let prover_claims = selected_prover_data::<RecursiveCommitmentConfig<BaseCfg>, _>(
            OpeningClaims::from_groups(prover_groups).expect("prover claims"),
            prover_hints,
            prover_polys,
        );
        let selection = prover_claims.0;

        let mut prover_transcript = AkitaTranscript::<F>::new(transcript_domain);
        let proof = Recursive::<BaseCfg>::batched_prove(
            &setup,
            prover_claims,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("generated-profile recursive proof");
        assert!(
            proof_has_recursive_setup_sumcheck(&proof),
            "recursive proof must carry stage-3 setup sumcheck evidence"
        );

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof
            .serialize_compressed(&mut bytes)
            .expect("serialize generated-profile proof");
        let proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize generated-profile proof");

        let verifier_setup =
            Recursive::<BaseCfg>::setup_verifier_for_schedule(&setup, &schedule, &opening_layout)
                .expect("verifier setup");
        let verify_claims = |final_openings: Vec<F>| {
            let mut verifier_groups = Vec::new();
            for (group_idx, openings) in pre_openings.iter().enumerate() {
                verifier_groups.push(
                    PolynomialGroupClaims::new(
                        point[..PRE_NV].to_vec(),
                        openings.clone(),
                        &pre_commitments[group_idx],
                    )
                    .expect("pre verifier group"),
                );
            }
            verifier_groups.push(
                PolynomialGroupClaims::new(point.clone(), final_openings, &final_commitment)
                    .expect("final verifier group"),
            );
            let claims = OpeningClaims::from_groups(verifier_groups).expect("verifier claims");
            GroupBatchStatement::new(selection, claims).expect("verifier statement")
        };

        let mut verifier_transcript = AkitaTranscript::<F>::new(transcript_domain);
        Recursive::<BaseCfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_claims(final_openings.clone()),
            BasisMode::Lagrange,
        )
        .expect("generated-profile recursive verify");

        let reject_stage3_tamper = |tampered_proof: AkitaBatchedProof<F, F>, label: &str| {
            let mut transcript = AkitaTranscript::<F>::new(transcript_domain);
            let result = Recursive::<BaseCfg>::batched_verify(
                &tampered_proof,
                &verifier_setup,
                &mut transcript,
                verify_claims(final_openings.clone()),
                BasisMode::Lagrange,
            );
            assert!(
                matches!(
                    result,
                    Err(AkitaError::InvalidProof | AkitaError::InvalidInput(_))
                ),
                "{label} must return a proof/input rejection without panicking, got {result:?}"
            );
        };

        let mut tampered_claim = proof.clone();
        first_stage3_proof_mut(&mut tampered_claim)
            .expect("recursive profile Stage 3 proof")
            .claim += F::one();
        reject_stage3_tamper(tampered_claim, "tampered Stage 3 claim");

        let mut tampered_prefix_eval = proof.clone();
        first_stage3_proof_mut(&mut tampered_prefix_eval)
            .expect("recursive profile Stage 3 proof")
            .setup_prefix_eval += F::one();
        reject_stage3_tamper(
            tampered_prefix_eval,
            "tampered Stage 3 setup-prefix evaluation",
        );

        let mut tampered_round = proof.clone();
        let coefficient = first_stage3_proof_mut(&mut tampered_round)
            .and_then(|stage3| stage3.sumcheck.round_polys.first_mut())
            .and_then(|round| round.coeffs_except_linear_term.first_mut())
            .expect("recursive profile Stage 3 round coefficient");
        *coefficient += F::one();
        reject_stage3_tamper(
            tampered_round,
            "tampered Stage 3 round polynomial and derived point",
        );

        let mut tampered = final_openings;
        tampered[0] += F::from_canonical_u128_reduced(1);
        let mut tampered_transcript = AkitaTranscript::<F>::new(transcript_domain);
        let tampered_result = Recursive::<BaseCfg>::batched_verify(
            &proof,
            &verifier_setup,
            &mut tampered_transcript,
            verify_claims(tampered),
            BasisMode::Lagrange,
        );
        assert!(
            tampered_result.is_err(),
            "recursive verify must reject a tampered final opening"
        );
    });
}

#[cfg(feature = "logging-transcript")]
pub(super) fn public_transcript_events(
    events: &[akita_transcript::TranscriptEvent],
) -> Vec<akita_transcript::TranscriptEvent> {
    events
        .iter()
        .filter(|event| !matches!(event, akita_transcript::TranscriptEvent::Wire { .. }))
        .cloned()
        .collect()
}

#[cfg(feature = "logging-transcript")]
pub(super) fn event_label(event: &akita_transcript::TranscriptEvent) -> Option<&[u8]> {
    match event {
        akita_transcript::TranscriptEvent::Absorb { label, .. }
        | akita_transcript::TranscriptEvent::Squeeze { label, .. }
        | akita_transcript::TranscriptEvent::Wire { label, .. } => Some(label),
        akita_transcript::TranscriptEvent::Preamble { .. } => None,
    }
}

#[cfg(feature = "logging-transcript")]
pub(super) fn first_label_index(
    events: &[akita_transcript::TranscriptEvent],
    label: &[u8],
) -> Option<usize> {
    events
        .iter()
        .position(|event| event_label(event).is_some_and(|candidate| candidate == label))
}

#[cfg(feature = "logging-transcript")]
pub(super) fn first_label_index_after(
    events: &[akita_transcript::TranscriptEvent],
    start: usize,
    label: &[u8],
) -> Option<usize> {
    events[start..]
        .iter()
        .position(|event| event_label(event).is_some_and(|candidate| candidate == label))
        .map(|offset| start + offset)
}

#[cfg(feature = "logging-transcript")]
fn is_label_or_extension_limb(candidate: &[u8], base: &[u8]) -> bool {
    candidate == base || akita_transcript::is_ext_limb_label(candidate, base)
}

#[cfg(feature = "logging-transcript")]
pub(super) fn first_label_or_extension_limb_index_after(
    events: &[akita_transcript::TranscriptEvent],
    start: usize,
    label: &[u8],
) -> Option<usize> {
    events[start..]
        .iter()
        .position(|event| {
            event_label(event).is_some_and(|candidate| is_label_or_extension_limb(candidate, label))
        })
        .map(|offset| start + offset)
}

#[cfg(feature = "logging-transcript")]
fn first_logical_label_span_after(
    events: &[akita_transcript::TranscriptEvent],
    start: usize,
    label: &[u8],
) -> Option<(usize, usize)> {
    let span_start = first_label_or_extension_limb_index_after(events, start, label)?;
    let mut span_end = span_start + 1;
    while span_end < events.len()
        && event_label(&events[span_end])
            .is_some_and(|candidate| is_label_or_extension_limb(candidate, label))
    {
        span_end += 1;
    }
    Some((span_start, span_end))
}

#[cfg(feature = "logging-transcript")]
fn assert_no_logical_label(
    events: &[akita_transcript::TranscriptEvent],
    range: std::ops::Range<usize>,
    label: &[u8],
    message: &str,
) {
    assert!(
        events[range].iter().all(|event| {
            event_label(event).is_none_or(|candidate| !is_label_or_extension_limb(candidate, label))
        }),
        "{message}"
    );
}

#[cfg(feature = "logging-transcript")]
pub(super) fn assert_terminal_event_order_if_present(
    events: &[akita_transcript::TranscriptEvent],
) -> Option<usize> {
    use akita_transcript::labels;

    let e_hat = first_label_index(events, labels::ABSORB_TERMINAL_E_HAT)?;
    let (sparse_seed, sparse_seed_end) =
        first_logical_label_span_after(events, e_hat, labels::CHALLENGE_SPARSE_CHALLENGE)
            .expect("terminal transcript must squeeze sparse seed");
    let remainder =
        first_label_index_after(events, sparse_seed_end, labels::ABSORB_TERMINAL_W_REMAINDER)
            .expect("terminal transcript must absorb final-witness remainder");
    for (label, message) in [
        (
            labels::CHALLENGE_RING_SWITCH,
            "terminal must not squeeze alpha",
        ),
        (labels::CHALLENGE_TAU1, "terminal must not squeeze tau1"),
        (
            labels::CHALLENGE_SUMCHECK_ROUND,
            "terminal must not squeeze stage-2 rounds",
        ),
        (
            labels::CHALLENGE_SUMCHECK_BATCH,
            "terminal must not squeeze stage-2 batching",
        ),
        (labels::CHALLENGE_TAU0, "terminal must not squeeze tau0"),
    ] {
        assert_no_logical_label(events, e_hat + 1..events.len(), label, message);
    }

    assert!(e_hat < sparse_seed, "e_hat must precede sparse seed");
    assert!(
        sparse_seed < remainder,
        "sparse seed must precede witness remainder"
    );
    Some(e_hat)
}
