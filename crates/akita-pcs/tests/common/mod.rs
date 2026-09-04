#![allow(dead_code)]

mod opening_oracles;
#[path = "../../examples/support/workspace_schedules.rs"]
mod workspace_schedules;

pub(super) use opening_oracles::*;
pub(super) use workspace_schedules::load_workspace_scheme;

pub(super) use akita_config::proof_optimized::fp128;
pub(super) use akita_config::CommitmentConfig;
use akita_config::{
    derive_transcript_grinding_plan, RecursiveCommitmentConfig, TrustedScheduleCatalog,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::compute::{OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource, RootPolyShape};
pub(super) use akita_prover::DensePoly;
pub(super) use akita_prover::OneHotPoly;
pub(super) use akita_prover::SelectedProverOpeningData;
use akita_prover::{commit_setup_prefix, AkitaProverSetup};
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress};
use akita_types::{
    canonical_proof_shape, dispatch_for_field, AkitaBatchedProof, AkitaExpandedSetup,
    AkitaScheduleLookupKey, AkitaVerifierSetup, CommittedGroupBatchProfile, FlatMatrix,
    GroupBatchStatement, PolynomialGroupLayout, SetupPrefixProverRegistry, SetupPrefixSlotId,
    SetupPrefixVerifierRegistry, SetupSumcheckProof,
};
pub(super) use akita_types::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, AkitaCommitmentHint,
    BasisMode, CommittedGroup, OpeningClaims, PolynomialGroupClaims, PrecommittedGroupProfiles,
};
pub(super) use akita_types::{CommittedGroupParams, FoldSchedule};
pub(super) use jolt_field::{CanonicalBytes, CanonicalEncoding, Field};
use jolt_field::{One, Zero};
pub(super) use rand::rngs::StdRng;
pub(super) use rand::{Rng, SeedableRng};
use std::sync::{Arc, Once};

#[cfg(feature = "logging-transcript")]
use akita_transcript::TranscriptEvent;
use akita_transcript::{labels, AkitaTranscript, Transcript};

pub(super) type F = fp128::Field;
pub(super) const STACK_SIZE: usize = 256 * 1024 * 1024;

pub(super) type OneHotCfg = fp128::OneHot;
pub(super) const ONEHOT_D: usize = 256;

pub(super) type DenseCfg = fp128::Dense;
pub(super) const DENSE_D: usize = 256;

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
        .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
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

/// Require a logging transcript to consume exactly the public grinding plan
/// and to expose the corresponding live challenge boundaries.
#[cfg(feature = "logging-transcript")]
pub(super) fn assert_production_grinding_audit(
    events: &[TranscriptEvent],
    plan: &akita_types::GrindingPlan,
) -> Vec<(akita_types::GrindingSite, usize)> {
    use akita_types::{GrindingQueryKind, GrindingSite};

    let expected_plan = plan
        .runs()
        .iter()
        .map(|run| (run.site().canonical_bytes(), run.multiplicity()))
        .collect::<Vec<_>>();
    let consumed_plan = events
        .iter()
        .filter_map(|event| match event {
            TranscriptEvent::GrindingPlanQuery { site, multiplicity } => {
                Some((site.clone(), *multiplicity))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        consumed_plan, expected_plan,
        "adapter consumption must equal the public plan"
    );

    let mut run_index = 0usize;
    let mut active_pow = None;
    let mut actual_draw_counts = Vec::new();
    for event in events {
        match event {
            TranscriptEvent::GrindingPlanQuery { .. } => {
                let run = plan
                    .runs()
                    .get(run_index)
                    .expect("validated plan event count");
                run_index += 1;
                active_pow = (run.kind() == GrindingQueryKind::ProofOfWork).then(|| {
                    actual_draw_counts.push((run.site(), 0));
                    actual_draw_counts.len() - 1
                });
            }
            TranscriptEvent::GrindingActualQuery { site, label } => {
                let index = active_pow.expect("actual challenge must follow a proof-of-work run");
                let (expected_site, count) = &mut actual_draw_counts[index];
                assert_eq!(site, &expected_site.canonical_bytes());
                let normalized_label =
                    akita_transcript::ext_limb_base_label(label).unwrap_or(label);
                assert_eq!(
                    normalized_label,
                    expected_site.proof_of_work_label().unwrap()
                );
                *count += 1;
            }
            _ => {}
        }
    }
    assert_eq!(run_index, plan.runs().len());
    assert!(
        actual_draw_counts.iter().all(|(_, count)| *count > 0),
        "every proof-of-work run must protect at least one live draw"
    );

    let expected_ranges = plan
        .runs()
        .iter()
        .filter_map(|run| match run.site() {
            GrindingSite::FoldChallengeGroup { group, .. } => Some((
                group as usize,
                run.fold_coordinate_count().unwrap() as usize,
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let actual_ranges = events
        .iter()
        .filter_map(|event| match event {
            TranscriptEvent::FoldChallengeRange {
                group_index,
                coordinate_count,
            } => Some((*group_index, *coordinate_count)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual_ranges, expected_ranges,
        "live indexed draws must equal coordinate runs"
    );

    let expected_roots = plan
        .runs()
        .iter()
        .filter(|run| run.kind() == GrindingQueryKind::FoldChallengeGroup)
        .count();
    let actual_roots = events
        .iter()
        .filter(|event| {
            matches!(event, TranscriptEvent::Squeeze { label, .. } if label == akita_transcript::labels::CHALLENGE_SPARSE_CHALLENGE)
        })
        .count();
    assert_eq!(
        actual_roots, expected_roots,
        "live fold roots must equal group runs"
    );
    actual_draw_counts
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
            TranscriptEvent::Grinding {
                site_label,
                grind_bits,
                nonce_bits,
                nonce,
                predicate_len,
                predicate,
            } => {
                bytes.push(4);
                bytes.extend_from_slice(&u64::try_from(site_label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(site_label);
                bytes.push(*grind_bits);
                bytes.push(*nonce_bits);
                bytes.extend_from_slice(&nonce.to_le_bytes());
                bytes.extend_from_slice(&u64::try_from(*predicate_len).unwrap().to_le_bytes());
                bytes.extend_from_slice(predicate);
            }
            TranscriptEvent::GrindingPlanQuery { site, multiplicity } => {
                bytes.push(5);
                bytes.extend_from_slice(&u64::try_from(site.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(site);
                bytes.extend_from_slice(&multiplicity.to_le_bytes());
            }
            TranscriptEvent::GrindingActualQuery { site, label } => {
                bytes.push(6);
                bytes.extend_from_slice(&u64::try_from(site.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(site);
                bytes.extend_from_slice(&u64::try_from(label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(label);
            }
            TranscriptEvent::FoldChallengeRange {
                group_index,
                coordinate_count,
            } => {
                bytes.push(7);
                bytes.extend_from_slice(&u64::try_from(*group_index).unwrap().to_le_bytes());
                bytes.extend_from_slice(&u64::try_from(*coordinate_count).unwrap().to_le_bytes());
            }
        }
    }
    bytes
}

/// Canonical Stage 1 payload bytes in fold-wire order.
pub(super) fn serialize_stage1_payload<FF>(proof: &akita_types::AkitaStage1Proof<FF>) -> Vec<u8>
where
    FF: Field + AkitaSerialize,
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
    FF: Field + CanonicalEncoding + CanonicalBytes + 'static,
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
    schedules: &TrustedScheduleCatalog,
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
    SelectedProverOpeningData::from_committed_claims::<Cfg>(
        opening_claims,
        vec![hint],
        vec![polynomials],
        schedules,
    )
    .expect("valid prover opening data")
}

pub(super) fn selected_prover_data<'a, Cfg, P>(
    claims: OpeningClaims<'a, Cfg::ExtField, CommittedGroup<Cfg::Field>>,
    hints: Vec<AkitaCommitmentHint<Cfg::Field>>,
    polynomials: Vec<&'a [&'a P]>,
    schedules: &TrustedScheduleCatalog,
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
    SelectedProverOpeningData::from_committed_claims::<Cfg>(claims, hints, polynomials, schedules)
        .expect("valid selected prover data")
}

pub(super) fn selected_statement<'a, Cfg>(
    claims: OpeningClaims<'a, Cfg::ExtField, &'a CommittedGroup<Cfg::Field>>,
    schedules: &TrustedScheduleCatalog,
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
    let selection = schedules
        .resolve_profiles(&profiles)
        .expect("select verifier statement schedule")
        .selection();
    GroupBatchStatement::new(selection, claims).expect("valid selected verifier statement")
}

pub(super) fn verify_input<'a, Cfg>(
    point: &'a [Cfg::ExtField],
    openings: &'a [Cfg::ExtField],
    commitment: &'a CommittedGroup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog,
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
    let selection = schedules
        .resolve_profiles(&profiles)
        .expect("select verifier statement schedule")
        .selection();
    GroupBatchStatement::new(selection, claims).expect("valid verifier statement")
}

pub(super) fn opening_from_poly_for_layout<'a, P>(
    poly: &'a P,
    point: &[F],
    layout: &akita_types::GroupOpenPhaseParams,
    basis_mode: BasisMode,
) -> F
where
    P: RootOpeningSource<F, 64>
        + RootPolyShape<F, 64>
        + RootOpeningSource<F, 128>
        + RootPolyShape<F, 128>
        + RootOpeningSource<F, 256>
        + RootPolyShape<F, 256>
        + RootOpeningSource<F, 512>
        + RootPolyShape<F, 512>,
    CpuBackend: OpeningFoldKernel<<P as RootOpeningSource<F, 64>>::OpeningView<'a>, F, 64>
        + OpeningFoldKernel<<P as RootOpeningSource<F, 128>>::OpeningView<'a>, F, 128>
        + OpeningFoldKernel<<P as RootOpeningSource<F, 256>>::OpeningView<'a>, F, 256>
        + OpeningFoldKernel<<P as RootOpeningSource<F, 512>>::OpeningView<'a>, F, 512>,
{
    match layout.inner_commit_matrix_params().ring_dimension() {
        64 => opening_from_poly_with_basis::<64, _>(poly, point, layout, basis_mode),
        128 => opening_from_poly_with_basis::<128, _>(poly, point, layout, basis_mode),
        256 => opening_from_poly_with_basis::<256, _>(poly, point, layout, basis_mode),
        512 => opening_from_poly_with_basis::<512, _>(poly, point, layout, basis_mode),
        dimension => panic!("unsupported test opening ring dimension D={dimension}"),
    }
}

pub(super) fn opening_from_poly_with_basis<'a, const D: usize, P>(
    poly: &'a P,
    point: &[F],
    layout: &akita_types::GroupOpenPhaseParams,
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
        layout.num_positions_per_block(),
        layout.num_live_blocks(),
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
            num_positions_per_block: layout.num_positions_per_block(),
        },
    )
    .expect("evaluate_and_fold");
    let folded_ring = opening.eval;
    let packed_inner = reduce_inner_opening_to_ring_element::<F, D>(inner_point, basis_mode)
        .expect("inner opening point should match ring dimension");
    (folded_ring * packed_inner.sigma_m1()).coefficients()[0]
}

pub(super) fn make_onehot_poly<Cfg>(num_vars: usize, seed: u64) -> OneHotPoly<F, u8>
where
    Cfg: CommitmentConfig<Field = F>,
{
    // `2^nv = (num_live_blocks · num_positions_per_block) · D` field elements, grouped into
    // `2^nv / K` one-hot chunks of size `K`.
    let onehot_k = akita_config::unit_onehot_source_chunk_size::<Cfg>()
        .expect("one-hot fixture requires a unit-one-hot commitment config");
    assert!(
        onehot_k <= usize::from(u8::MAX) + 1,
        "test u8 one-hot fixture cannot represent chunk size {onehot_k}"
    );
    let total_field = 1usize << num_vars;
    let total_chunks = total_field / onehot_k;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k) as u8))
        .collect();
    OneHotPoly::<F, u8>::new(onehot_k, indices).expect("onehot poly")
}

pub(super) fn make_dense_poly(nv: usize, seed: u64) -> DensePoly<F> {
    let evals = dense_field_evals(nv, seed);
    DensePoly::<F>::from_field_evals(nv, &evals).expect("dense poly")
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
        out.push(F::from_u128_reduced(v as u128));
    }
    out
}

/// Signed `u64` evaluations: every centered magnitude is a full `u64`, on both
/// signs.
///
/// This is the workload `fp128::DenseBounded` exists for, and it is deliberately
/// *wider* than [`dense_field_evals`]: that generator draws `u64` magnitudes but
/// only on the positive side, whereas a bounded source has to survive
/// `-u64::MAX` too. Both endpoints are forced into every draw sequence by
/// [`u64_magnitude_endpoints`], so a fixture can never drift into staying
/// comfortably small.
///
/// A `u64` magnitude needs `log_commit_bound = 65`, not `64`: the bound is a
/// *signed* bit width, so `k` means `[-2^(k-1), 2^(k-1) - 1]` and covering
/// `u64::MAX = 2^64 - 1` takes one sign bit plus 64 magnitude bits.
pub(super) fn u64_dense_field_evals(nv: usize, seed: u64) -> Vec<F> {
    let n = 1usize << nv;
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    for index in 0..n {
        // Seed the first few slots with the exact endpoints so the fixture always
        // exercises them, then fill the rest with full-width draws.
        if let Some(value) = u64_magnitude_endpoints().get(index) {
            out.push(*value);
            continue;
        }
        let draw = splitmix64_next(&mut state);
        let magnitude = F::from_u128_reduced(u128::from(draw));
        out.push(if draw & 1 == 0 { magnitude } else { -magnitude });
    }
    out
}

/// The centered endpoints a `u64` workload reaches: `±u64::MAX` and `±1`.
///
/// A bounded schedule must accept all of these. `-u64::MAX` is the interesting
/// one — it is the widest *negative* centered magnitude, and balanced digits
/// reach further negative than positive, so a guard stated only on the positive
/// side would miss it.
pub(super) fn u64_magnitude_endpoints() -> [F; 4] {
    let max = F::from_u128_reduced(u128::from(u64::MAX));
    [max, -max, F::one(), -F::one()]
}

pub(super) fn multi_group_root_params(schedule: &FoldSchedule) -> &CommittedGroupParams {
    &schedule.root.params
}

pub(super) fn schedule_uses_setup_prefix(schedule: &FoldSchedule) -> bool {
    schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.setup_prefix().is_some())
}

pub(super) fn proof_has_recursive_setup_sumcheck(proof: &AkitaBatchedProof<F, F>) -> bool {
    proof.root.stage3_sumcheck_proof.is_some()
        || proof
            .recursive_folds
            .iter()
            .any(|step| step.stage3_sumcheck_proof.is_some())
}

pub(super) fn first_stage3_proof_mut(
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

fn first_setup_prefix_slot(schedule: &FoldSchedule) -> SetupPrefixSlotId {
    schedule
        .recursive_folds
        .iter()
        .find_map(|fold| fold.params.setup_prefix())
        .expect("recursive profile must carry a setup prefix")
        .slot_id()
        .expect("setup prefix group")
}

fn verifier_setup_with_alternate_full_prefix(
    setup: &AkitaProverSetup<F>,
    verifier_setup: &AkitaVerifierSetup<F>,
    slot_id: &SetupPrefixSlotId,
) -> Option<AkitaVerifierSetup<F>> {
    let natural_len = slot_id.natural_len;
    let n_prefix = slot_id.n_prefix().expect("prefix length");
    if natural_len == n_prefix {
        return None;
    }

    let original = setup.expanded.shared_matrix().as_field_slice();
    let mut altered = original.to_vec();
    altered[natural_len] += F::one();
    assert_eq!(&altered[..natural_len], &original[..natural_len]);
    assert_ne!(
        &altered[natural_len..n_prefix],
        &original[natural_len..n_prefix]
    );

    let descriptor = setup.expanded.descriptor().clone();
    let setup_seed = descriptor.setup_seed.clone();
    let altered_expanded = Arc::new(
        AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            descriptor,
            FlatMatrix::from_flat_data(altered),
        ),
    );
    let altered_setup = AkitaProverSetup {
        expanded: altered_expanded,
        prefix_slots: SetupPrefixProverRegistry::new(setup_seed.clone()),
    };
    let backend = CpuBackend::DEFAULT;
    let prepared = backend
        .prepare_setup(&altered_setup)
        .expect("prepare altered setup");
    let altered_slot = dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        slot_id.d_setup(),
        |D| {
            commit_setup_prefix::<F, D, _>(
                &altered_setup.expanded,
                &backend,
                &prepared,
                &slot_id.commitment_profile,
                n_prefix,
                natural_len,
            )
        }
    )
    .expect("commit altered full setup prefix");

    let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed);
    for (id, slot) in verifier_setup.prefix_slots.iter() {
        let replacement = if id == slot_id {
            altered_slot.verifier_slot()
        } else {
            slot.clone()
        };
        prefix_slots
            .insert(replacement)
            .expect("insert verifier slot");
    }
    Some(
        AkitaVerifierSetup::from_parts(verifier_setup.expanded.clone(), prefix_slots)
            .expect("alternate verifier setup"),
    )
}

/// Multi-group recursive roundtrip: two user precommitted groups plus one final group.
/// `BaseCfg` selects the physical witness layout (single-chunk vs chunked); the
/// recursion adapter and standalone profiles are derived from it.
/// `on_schedule` runs profile-specific assertions against the resolved schedule.
mod recursive;
#[allow(unused_imports)]
pub(super) use recursive::recursive_multi_group_round_trip;

pub(super) fn make_onehot_poly_with_k(nv: usize, k: usize, seed: u64) -> OneHotPoly<F, u8> {
    let total_chunks = (1usize << nv) / k;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_chunks)
        .map(|_| Some(rng.gen_range(0..k) as u8))
        .collect();
    OneHotPoly::<F, u8>::new(k, indices).expect("onehot poly")
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
        akita_transcript::TranscriptEvent::Grinding { site_label, .. } => Some(site_label),
        akita_transcript::TranscriptEvent::Preamble { .. }
        | akita_transcript::TranscriptEvent::GrindingPlanQuery { .. }
        | akita_transcript::TranscriptEvent::GrindingActualQuery { .. }
        | akita_transcript::TranscriptEvent::FoldChallengeRange { .. } => None,
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

/// Assert that every public claim-batching squeeze belongs to a fold whose
/// complete opening payload was already absorbed.
#[cfg(feature = "logging-transcript")]
pub(super) fn assert_claim_batching_follows_opening_payload(
    events: &[akita_transcript::TranscriptEvent],
) -> usize {
    let mut payload_bound = false;
    let mut batching_squeezes = 0usize;
    for event in events {
        let Some(label) = event_label(event) else {
            continue;
        };
        if label == akita_transcript::labels::ABSORB_OPENING_PAYLOAD {
            payload_bound = true;
        } else if is_label_or_extension_limb(label, akita_transcript::labels::CHALLENGE_EVAL_BATCH)
        {
            assert!(
                payload_bound,
                "public claim-batching challenge preceded its fold opening payload"
            );
            batching_squeezes += 1;
        } else if is_label_or_extension_limb(
            label,
            akita_transcript::labels::CHALLENGE_SPARSE_CHALLENGE,
        ) {
            payload_bound = false;
        }
    }
    batching_squeezes
}

#[cfg(feature = "logging-transcript")]
pub(super) fn is_label_or_extension_limb(candidate: &[u8], base: &[u8]) -> bool {
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
