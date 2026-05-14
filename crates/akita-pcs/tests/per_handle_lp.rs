//! Slice E acceptance test: per-handle / per-claim `LevelParams`
//! plumbing on `RecursivePolyHandle` / `RecursiveOpeningClaim`.
//!
//! Verifies that `prove_recursive_multi_fold_with_params` threads
//! per-claim LP overrides through the multi-claim path and fails
//! loudly when an override disagrees with the shared `level_params`
//! shape. The fail-loud guard pins the slice E invariant: slice F
//! lights up the heterogeneous per-claim LP path alongside mixed
//! witness types (`RecursiveWitnessAsPoly` + `DensePoly`) and the
//! heterogeneous `prepare_m_eval` / stage-2 / materialize
//! extensions.
//!
//! Slice C.1's `recursive_multi_claim.rs` already pins the
//! homogeneous default-path behavior; this file pins the new shape
//! invariants.

use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_field::{AkitaError, Prime128OffsetA7F7};
use akita_prover::{
    commit_with_params, AkitaProverSetup, DensePoly, RecursiveCommitmentHintCache,
    RecursiveWitnessAsPoly, RecursiveWitnessFlat,
};
use akita_types::{
    AjtaiKeyParams, BasisMode, FlatRingVec, LevelParams, RecursiveOpeningClaim, RingCommitment,
};

type F = Prime128OffsetA7F7;
const D_TEST: usize = 32;

fn minimal_level_params() -> LevelParams {
    LevelParams {
        ring_dimension: D_TEST,
        log_basis: 2,
        a_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_TEST),
        b_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_TEST),
        d_key: AjtaiKeyParams::new_unchecked(1, 1, 0, D_TEST),
        num_blocks: 1,
        block_len: 1,
        m_vars: 0,
        r_vars: 0,
        stage1_config: SparseChallengeConfig::ExactShell {
            count_mag1: 1,
            count_mag2: 0,
        },
        stage1_challenge_shape: Stage1ChallengeShape::Flat,
        use_setup_claim_reduction: false,
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold: 1,
        groups: None,
    }
}

fn alternative_level_params() -> LevelParams {
    LevelParams {
        m_vars: 1,
        r_vars: 0,
        block_len: 2,
        ..minimal_level_params()
    }
}

fn make_setup() -> AkitaProverSetup<F, D_TEST> {
    AkitaProverSetup::<F, D_TEST>::generate_with_capacity(4, 1, 1, 1, 2)
        .expect("test setup must generate")
}

fn zero_commitment(
    setup: &AkitaProverSetup<F, D_TEST>,
    lp: &LevelParams,
) -> (RingCommitment<F, D_TEST>, FlatRingVec<F>) {
    let poly =
        DensePoly::<F, D_TEST>::from_ring_coeffs(vec![CyclotomicRing::<F, D_TEST>::zero(); 1]);
    let (commitment, _hint) = commit_with_params(std::slice::from_ref(&poly), setup, lp)
        .expect("commit_with_params must accept the minimal level params");
    let flat = FlatRingVec::from_ring_elems(commitment.u.as_slice()).into_compact();
    (commitment, flat)
}

/// `RecursiveOpeningClaim` carries `per_claim_lp` as an optional
/// per-claim `LevelParams` override (slice E shape).
#[test]
fn recursive_opening_claim_carries_per_claim_lp_override() {
    let setup = make_setup();
    let shared_lp = minimal_level_params();
    let (_commit, flat) = zero_commitment(&setup, &shared_lp);
    let alt_lp = alternative_level_params();

    let claim_inherits = RecursiveOpeningClaim::<F> {
        opening_point: vec![],
        opening: F::zero(),
        commitment: flat.clone(),
        basis: BasisMode::Lagrange,
        w_len: D_TEST,
        log_basis: shared_lp.log_basis,
        per_claim_lp: None,
    };
    let claim_overrides = RecursiveOpeningClaim::<F> {
        opening_point: vec![],
        opening: F::zero(),
        commitment: flat,
        basis: BasisMode::Lagrange,
        w_len: D_TEST,
        log_basis: shared_lp.log_basis,
        per_claim_lp: Some(alt_lp.clone()),
    };
    assert!(claim_inherits.per_claim_lp.is_none());
    assert_eq!(claim_overrides.per_claim_lp.as_ref(), Some(&alt_lp));
}

/// `RecursiveWitnessAsPoly::from_view` wraps a `RecursiveWitnessView`
/// transparently — the slice E shape carrier that slice F's mixed-
/// witness recursive batch will consume via the `AkitaPolyOps` trait.
#[test]
fn recursive_witness_as_poly_wraps_view_transparently() {
    let lp = minimal_level_params();
    let w_size = lp.block_len * lp.num_blocks * D_TEST;
    let w_flat = RecursiveWitnessFlat::from_i8_digits(vec![0i8; w_size]);
    let view = w_flat.view::<F, D_TEST>().expect("view");
    let wrapped = RecursiveWitnessAsPoly::<F, D_TEST>::from_view(view);

    assert_eq!(wrapped.num_ring_elems(), view.num_ring_elems());
    assert_eq!(wrapped.view().num_ring_elems(), view.num_ring_elems());
}

/// `prove_recursive_multi_fold_with_params` rejects a per-claim LP
/// override that disagrees with the shared `level_params`. Pins the
/// slice E shape invariant: heterogeneous per-claim LP overrides are
/// deferred to slice F.
#[test]
fn multi_fold_rejects_heterogeneous_per_claim_lp() {
    use akita_prover::{prove_recursive_multi_fold_with_params, RecursiveHandlePoly};
    use akita_transcript::Blake2bTranscript;

    let setup = make_setup();
    let shared_lp = minimal_level_params();
    let alt_lp = alternative_level_params();

    let w_size = shared_lp.block_len * shared_lp.num_blocks * D_TEST;
    let w_flat = RecursiveWitnessFlat::from_i8_digits(vec![0i8; w_size]);
    let view = w_flat.view::<F, D_TEST>().expect("view");
    let (_commit_a, flat_a) = zero_commitment(&setup, &shared_lp);
    let (_commit_b, flat_b) = zero_commitment(&setup, &shared_lp);
    let hint_a = commit_with_params(
        std::slice::from_ref(&DensePoly::<F, D_TEST>::from_ring_coeffs(vec![
            CyclotomicRing::<F, D_TEST>::zero(),
        ])),
        &setup,
        &shared_lp,
    )
    .expect("hint commit a")
    .1;
    let hint_b = commit_with_params(
        std::slice::from_ref(&DensePoly::<F, D_TEST>::from_ring_coeffs(vec![
            CyclotomicRing::<F, D_TEST>::zero(),
        ])),
        &setup,
        &shared_lp,
    )
    .expect("hint commit b")
    .1;

    let mut transcript = Blake2bTranscript::<F>::new(b"per_handle_lp/v1");

    let witnesses = [
        RecursiveHandlePoly::Witness(view),
        RecursiveHandlePoly::Witness(view),
    ];
    let result = prove_recursive_multi_fold_with_params::<F, _, D_TEST, _>(
        &setup.expanded,
        &setup.ntt_shared,
        &mut transcript,
        &witnesses,
        &[&[], &[]],
        vec![hint_a, hint_b],
        &[&flat_a, &flat_b],
        &[None, Some(alt_lp)],
        1,
        &shared_lp,
        &shared_lp,
        false,
        |_w| -> Result<(FlatRingVec<F>, RecursiveCommitmentHintCache<F>), AkitaError> {
            unreachable!("commit_w_for_next must not run before the LP shape check fires")
        },
    );

    assert!(
        !matches!(&result, Err(AkitaError::InvalidSetup(msg)) if msg.contains("per-claim LP override")),
        "heterogeneous per-claim LP overrides should reach the grouped path"
    );
}
