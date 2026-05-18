//! Per-handle / per-claim `LevelParams` plumbing structural test.
//!
//! Pins the `RecursiveOpeningClaim::per_claim_lp` field shape used by
//! the multi-group batched Hachi commit at level `L+1` (book §5.3
//! lines 643–660).

use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_field::Prime128OffsetA7F7;
use akita_prover::{commit_with_params, AkitaProverSetup, DensePoly};
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
        tier_marker: None,
    };
    let claim_overrides = RecursiveOpeningClaim::<F> {
        opening_point: vec![],
        opening: F::zero(),
        commitment: flat,
        basis: BasisMode::Lagrange,
        w_len: D_TEST,
        log_basis: shared_lp.log_basis,
        per_claim_lp: Some(alt_lp.clone()),
        tier_marker: None,
    };
    assert!(claim_inherits.per_claim_lp.is_none());
    assert_eq!(claim_overrides.per_claim_lp.as_ref(), Some(&alt_lp));
}
