#![allow(dead_code)]

#[cfg(feature = "logging-transcript")]
pub(crate) mod native_mutations;
mod opening_oracles;
#[path = "../../examples/support/workspace_schedules.rs"]
mod workspace_schedules;

pub(super) use opening_oracles::*;
pub(super) use workspace_schedules::load_workspace_scheme;

pub(super) use akita_config::proof_optimized::fp128;
pub(super) use akita_config::CommitmentConfig;
use akita_config::{RecursiveCommitmentConfig, TrustedScheduleCatalog};
pub(super) use akita_cpu_backend::CommitmentHandle;
pub(super) use akita_cpu_backend::DensePoly;
pub(super) use akita_cpu_backend::OneHotPoly;
use akita_cpu_backend::SetupPrefixProverRegistry;
use akita_cpu_backend::{evaluate_root_polynomial, RootPolyShape};
use akita_cpu_backend::{AkitaProverSetup, CpuBackend};
use akita_pcs::AkitaCommitmentScheme;
pub(super) use akita_prover::SelectedProverOpeningData;
use akita_types::{
    AkitaExpandedSetup, AkitaScheduleLookupKey, AkitaVerifierSetup, CommittedGroupBatchProfile,
    FlatMatrix, GroupBatchStatement, PolynomialGroupLayout, SetupPrefixSlotId,
    SetupPrefixVerifierRegistry,
};
pub(super) use akita_types::{
    BasisMode, CommittedGroup, OpeningClaims, PolynomialGroupClaims, PrecommittedGroupProfiles,
};
pub(super) use akita_types::{CommittedGroupParams, FoldSchedule};
use jolt_field::One;
pub(super) use jolt_field::{CanonicalBytes, CanonicalEncoding, Field};
pub(super) use rand::rngs::StdRng;
pub(super) use rand::{Rng, SeedableRng};
use std::sync::{Arc, Once};

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

/// Stable digest used by versioned protocol epochs.
pub(super) fn protocol_epoch_digest<FF>(payload: &[u8]) -> String
where
    FF: Field + CanonicalEncoding + CanonicalBytes + 'static,
{
    let mut transcript =
        akita_transcript::new_native_prover(b"akita/protocol-epoch/digest", payload).unwrap();
    akita_transcript::prover_context(
        &mut transcript,
        akita_transcript::ProtocolContextRecord::new(
            akita_transcript::ProtocolSiteId {
                family: akita_transcript::SITE_FAMILY_ROOT_STATEMENT,
                detail: 0x4550_4f43,
                ..akita_transcript::ProtocolSiteId::default()
            }
            .to_bytes(),
            akita_transcript::ProtocolMessageKind::Challenge as u32,
            0,
            0,
            akita_transcript::native_field_challenge_bytes::<FF>(),
        ),
    );
    akita_transcript::native_prover_field_challenge::<FF>(&mut transcript)
        .expect("supported protocol field")
        .to_bytes_le_vec()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_input<'a, Cfg>(
    point: &'a [Cfg::ExtField],
    evaluations: &[Cfg::ExtField],
    commitment: &'a CommittedGroup<Cfg::Field>,
    hint: CommitmentHandle<Cfg::Field, Cfg::ExtField>,
    schedules: &TrustedScheduleCatalog<Cfg>,
) -> SelectedProverOpeningData<
    'a,
    Cfg::ExtField,
    CommitmentHandle<Cfg::Field, Cfg::ExtField>,
    Cfg::Field,
>
where
    Cfg: CommitmentConfig,
{
    let group =
        PolynomialGroupClaims::new(point.to_vec(), evaluations.to_vec(), commitment.clone())
            .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    SelectedProverOpeningData::from_committed_claims::<Cfg>(opening_claims, vec![hint], schedules)
        .expect("valid prover opening data")
}

#[allow(clippy::type_complexity)]
pub(super) fn selected_prover_data<'a, Cfg>(
    claims: OpeningClaims<'a, Cfg::ExtField, CommittedGroup<Cfg::Field>>,
    hints: Vec<CommitmentHandle<Cfg::Field, Cfg::ExtField>>,
    schedules: &TrustedScheduleCatalog<Cfg>,
) -> SelectedProverOpeningData<
    'a,
    Cfg::ExtField,
    CommitmentHandle<Cfg::Field, Cfg::ExtField>,
    Cfg::Field,
>
where
    Cfg: CommitmentConfig,
{
    SelectedProverOpeningData::from_committed_claims::<Cfg>(claims, hints, schedules)
        .expect("valid selected prover data")
}

pub(super) fn selected_statement<'a, Cfg>(
    claims: OpeningClaims<'a, Cfg::ExtField, &'a CommittedGroup<Cfg::Field>>,
    schedules: &TrustedScheduleCatalog<Cfg>,
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
    schedules: &TrustedScheduleCatalog<Cfg>,
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

pub(super) fn opening_from_poly_for_layout<P>(
    poly: &P,
    point: &[F],
    layout: &akita_types::GroupOpenPhaseParams,
    basis_mode: BasisMode,
) -> F
where
    P: RootPolyShape<F, 64> + RootPolyShape<F, 128> + RootPolyShape<F, 256> + RootPolyShape<F, 512>,
    P: akita_cpu_backend::RootPolynomialEvaluator<F, 64>
        + akita_cpu_backend::RootPolynomialEvaluator<F, 128>
        + akita_cpu_backend::RootPolynomialEvaluator<F, 256>
        + akita_cpu_backend::RootPolynomialEvaluator<F, 512>,
{
    match layout.inner_commit_matrix_params().ring_dimension() {
        64 => opening_from_poly_with_basis::<64, _>(poly, point, layout, basis_mode),
        128 => opening_from_poly_with_basis::<128, _>(poly, point, layout, basis_mode),
        256 => opening_from_poly_with_basis::<256, _>(poly, point, layout, basis_mode),
        512 => opening_from_poly_with_basis::<512, _>(poly, point, layout, basis_mode),
        dimension => panic!("unsupported test opening ring dimension D={dimension}"),
    }
}

pub(super) fn opening_from_poly_with_basis<const D: usize, P>(
    poly: &P,
    point: &[F],
    layout: &akita_types::GroupOpenPhaseParams,
    basis_mode: BasisMode,
) -> F
where
    P: RootPolyShape<F, D>,
    P: akita_cpu_backend::RootPolynomialEvaluator<F, D>,
{
    evaluate_root_polynomial::<F, P, D>(
        poly,
        point,
        layout.num_positions_per_block(),
        layout.num_live_blocks(),
        basis_mode,
    )
    .expect("root polynomial opening")
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
    let backend =
        CpuBackend::<F, F>::new(altered_setup.expanded.clone()).expect("altered setup backend");
    let artifacts = backend
        .export_setup_prefixes(std::slice::from_ref(slot_id))
        .expect("altered prefix artifact");
    let altered_slot = artifacts.get(slot_id).expect("altered prefix slot");

    let mut prefix_slots = SetupPrefixVerifierRegistry::new(setup_seed);
    for (id, slot) in verifier_setup.prefix_slots().iter() {
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
        AkitaVerifierSetup::from_parts(verifier_setup.expanded().clone(), prefix_slots)
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
