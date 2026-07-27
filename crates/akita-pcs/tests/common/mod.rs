#![allow(dead_code)]

pub(super) use akita_config::proof_optimized::fp128;
pub(super) use akita_config::CommitmentConfig;
use akita_config::{PrecommittedCommitmentConfig, RecursiveCommitmentConfig};
pub(super) use akita_field::{CanonicalBytes, CanonicalField, FieldCore, TranscriptChallenge};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource, RootPolyShape};
pub(super) use akita_prover::DensePoly;
pub(super) use akita_prover::OneHotPoly;
pub(super) use akita_prover::ProverOpeningData;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress};
pub(super) use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaCommitmentHint,
    BasisMode, Commitment, OpeningClaims, PointVariableSelection, PolynomialGroupClaims,
};
use akita_types::{
    AkitaBatchedProof, AkitaScheduleLookupKey, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupDescriptor,
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
pub(super) type OneHotCfg = fp128::D64OneHot;
pub(super) const ONEHOT_D: usize = OneHotCfg::D;
// `fp128::D64OneHot` requires K=256 one-hot schedules (chunks span `K/D = 4`
// ring elements), so the committed poly has `2^nv / K` chunks, not one chunk
// per ring element. Must match `OneHotCfg::onehot_chunk_size()`.
pub(super) const ONEHOT_K: usize = 256;

pub(super) type DenseCfg = fp128::D64Dense;
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
    transcript.append_bytes(labels::ABSORB_PROVER_V, payload);
    transcript
        .challenge_scalar(labels::CHALLENGE_SUMCHECK_BATCH)
        .to_bytes_le_vec()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn prove_input<'a, FF: FieldCore + Clone, P, CommitF: FieldCore>(
    point: &'a [FF],
    polynomials: &'a [&'a P],
    commitment: &'a Commitment<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> ProverOpeningData<'a, FF, P, CommitF> {
    let group = PolynomialGroupClaims::new(
        PointVariableSelection::prefix(point.len(), point.len()).expect("full-point prover group"),
        vec![FF::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims =
        OpeningClaims::from_groups(point.to_vec(), vec![group]).expect("valid prover claims");
    ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
        .expect("valid prover opening data")
}

pub(super) fn verify_input<'a, FF: FieldCore, C>(
    point: &'a [FF],
    openings: &'a [FF],
    commitment: &'a C,
) -> OpeningClaims<'static, FF, &'a C> {
    OpeningClaims::from_groups(
        point.to_vec(),
        vec![PolynomialGroupClaims::new(
            PointVariableSelection::prefix(point.len(), point.len()).expect("full-point group"),
            openings.to_vec(),
            commitment,
        )
        .expect("valid verifier claims group")],
    )
    .expect("valid verifier input")
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
    let packed_inner = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis_mode)
        .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}

pub(super) fn make_onehot_poly(layout: &CommittedGroupParams, seed: u64) -> OneHotPoly<F, u8> {
    // `2^nv = (num_live_blocks · num_positions_per_block) · D` field elements, grouped into
    // `2^nv / K` one-hot chunks of size `K`.
    let total_field = layout.num_live_blocks * layout.num_positions_per_block * ONEHOT_D;
    let total_chunks = total_field / ONEHOT_K;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, u8>::new(ONEHOT_K, ONEHOT_D, indices).expect("onehot poly")
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
        let pre_frozen = PrecommittedGroupDescriptor::from_params(pre_key, &pre_layout);
        let schedule_key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE),
            precommitteds: vec![pre_frozen, pre_frozen],
        };
        let pre_keys = vec![pre_key; PRE_GROUPS];

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
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
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
        let (final_commitment, final_hint) =
            Recursive::<BaseCfg>::commit_final_group(&setup, &final_polys, &stack, pre_keys)
                .expect("final generated-profile commitment");

        let point = random_point(FINAL_NV, 0xcafe_2026_0001);
        let pre_openings: Vec<Vec<F>> = pre_polys_by_group
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| {
                        opening_from_poly::<ONEHOT_D, _>(poly, &point[..PRE_NV], &pre_layout)
                    })
                    .collect()
            })
            .collect();
        let final_openings: Vec<F> = final_polys
            .iter()
            .map(|poly| opening_from_poly::<ONEHOT_D, _>(poly, &point, root_params))
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
                    PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                    openings.clone(),
                    pre_commitments[group_idx].clone(),
                )
                .expect("pre prover group"),
            );
        }
        prover_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
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

        let prover_claims = ProverOpeningData::new(
            OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
            prover_hints,
            prover_polys,
        )
        .expect("generated-profile prover data");

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

        let verifier_setup = setup.verifier_setup().expect("verifier setup");
        let verify_claims = |final_openings: Vec<F>| {
            let mut verifier_groups = Vec::new();
            for (group_idx, openings) in pre_openings.iter().enumerate() {
                verifier_groups.push(
                    PolynomialGroupClaims::new(
                        PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                        openings.clone(),
                        &pre_commitments[group_idx],
                    )
                    .expect("pre verifier group"),
                );
            }
            verifier_groups.push(
                PolynomialGroupClaims::new(
                    PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                    final_openings,
                    &final_commitment,
                )
                .expect("final verifier group"),
            );
            OpeningClaims::from_groups(point.clone(), verifier_groups).expect("verifier claims")
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
