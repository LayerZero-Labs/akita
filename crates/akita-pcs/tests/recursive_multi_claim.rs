//! Slice C.1 plumbing test: exercise `QuadraticEquation::new_recursive_prover`
//! with `Vec.len() == 2` opening claims.
//!
//! This is a structural smoke test: it constructs two synthetic recursive
//! witnesses sharing one minimal `LevelParams`, drives
//! `new_recursive_prover` through the multi-claim aggregation, and
//! checks that the resulting `QuadraticEquation` carries the multi-claim
//! shape downstream callers will rely on. It does not exercise a full
//! prover → verifier acceptance roundtrip — that requires a
//! cryptographically consistent prior-level handoff (the recursive
//! witness must be the digit decomposition of the previous level's
//! stage-2 output) and is naturally covered once slice C.2 wires up the
//! first production multi-claim recursive caller.
//!
//! The test does still validate the input-shape error paths to guard the
//! validation that protects the multi-claim aggregation from malformed
//! input.

use akita_algebra::CyclotomicRing;
use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_field::{AkitaError, Prime128OffsetA7F7};
use akita_prover::{
    commit_with_params, AkitaProverSetup, DensePoly, QuadraticEquation, RecursiveHandlePoly,
    RecursiveWitnessFlat,
};
use akita_transcript::Blake2bTranscript;
use akita_types::{
    ring_opening_point_from_field, AjtaiKeyParams, BasisMode, BlockOrder, LevelParams,
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

fn make_setup() -> AkitaProverSetup<F, D_TEST> {
    AkitaProverSetup::<F, D_TEST>::generate_with_capacity(4, 1, 1, 1, 2)
        .expect("test setup must generate")
}

/// Build a zero-content recursive witness + commitment/hint pair that
/// passes shape validation. The cryptographic content is not exercised
/// by this structural test.
fn build_zero_witness_and_commitment(
    setup: &AkitaProverSetup<F, D_TEST>,
    lp: &LevelParams,
) -> (
    RecursiveWitnessFlat,
    akita_types::RingCommitment<F, D_TEST>,
    akita_types::AkitaCommitmentHint<F, D_TEST>,
) {
    let w_size = lp.block_len * lp.num_blocks * D_TEST;
    let w_flat = RecursiveWitnessFlat::from_i8_digits(vec![0i8; w_size]);

    let zero_poly =
        DensePoly::<F, D_TEST>::from_ring_coeffs(vec![CyclotomicRing::<F, D_TEST>::zero(); 1]);
    let (commitment, hint) = commit_with_params(std::slice::from_ref(&zero_poly), setup, lp)
        .expect("commit_with_params must accept the minimal level params");
    (w_flat, commitment, hint)
}

/// Structural smoke test: with `Vec.len() == 2` claims the resulting
/// `QuadraticEquation` carries the expected multi-claim shape that
/// downstream callers (ring switch, stage-1, stage-2) rely on. Zero
/// witnesses keep the aggregation algebra trivial — what we are checking
/// is that the multi-claim code paths fire and produce a consistent
/// shape, not that the resulting equation is cryptographically tied to
/// any particular witness.
#[test]
fn new_recursive_prover_accepts_two_claims_and_aggregates_shape() {
    let setup = make_setup();
    let lp = minimal_level_params();

    let (w1_flat, commit1, hint1) = build_zero_witness_and_commitment(&setup, &lp);
    let (w2_flat, commit2, hint2) = build_zero_witness_and_commitment(&setup, &lp);
    let w1_view = w1_flat.view::<F, D_TEST>().expect("view 1");
    let w2_view = w2_flat.view::<F, D_TEST>().expect("view 2");

    let rop =
        ring_opening_point_from_field::<F>(&[], 0, 0, BasisMode::Lagrange, BlockOrder::ColumnMajor)
            .expect("empty opening point at m_vars=0,r_vars=0");

    let (y_ring1, folded1) = w1_view.evaluate_and_fold(&rop.b, &rop.a, lp.block_len, lp.num_blocks);
    let (y_ring2, folded2) = w2_view.evaluate_and_fold(&rop.b, &rop.a, lp.block_len, lp.num_blocks);

    let mut transcript = Blake2bTranscript::<F>::new(b"recursive_multi_claim/v1");

    let witnesses = [
        RecursiveHandlePoly::Witness(w1_view),
        RecursiveHandlePoly::Witness(w2_view),
    ];
    let quad_eq = QuadraticEquation::<F, D_TEST>::new_recursive_prover(
        &setup.ntt_shared,
        vec![rop.clone(), rop.clone()],
        vec![0usize, 1usize],
        &witnesses,
        vec![folded1, folded2],
        &[1usize, 1usize],
        lp.clone(),
        vec![hint1, hint2],
        &mut transcript,
        &[commit1.u.as_slice(), commit2.u.as_slice()],
        &[y_ring1, y_ring2],
        vec![F::one(); 2],
        2,
        setup.expanded.seed.max_stride,
    )
    .expect("multi-claim new_recursive_prover must accept well-shaped inputs");

    assert_eq!(quad_eq.opening_points().len(), 2);
    assert_eq!(quad_eq.claim_to_point(), &[0usize, 1usize]);
    assert_eq!(quad_eq.claim_group_sizes(), &[1usize, 1usize]);
    assert_eq!(quad_eq.gamma().len(), 2);
    assert_eq!(quad_eq.num_eval_rows(), 2);
    assert_eq!(quad_eq.v().len(), lp.d_key.row_len());

    let expected_y_len = 1 + 2 + lp.d_key.row_len() + 2 * lp.b_key.row_len() + lp.a_key.row_len();
    assert_eq!(
        quad_eq.y().len(),
        expected_y_len,
        "y must have 1 (consistency) + num_eval_rows + n_d + num_commitment_groups*n_b + n_a rows"
    );
}

/// Tampering test: mismatched claim_group_sizes vs commitments must be
/// rejected by the multi-claim input validation. Guards the validation
/// path that prevents malformed N>1 inputs from reaching the aggregation
/// loops.
#[test]
fn new_recursive_prover_rejects_mismatched_commitment_and_claim_group_lengths() {
    let setup = make_setup();
    let lp = minimal_level_params();

    let (w1_flat, commit1, hint1) = build_zero_witness_and_commitment(&setup, &lp);
    let (w2_flat, _commit2, hint2) = build_zero_witness_and_commitment(&setup, &lp);
    let w1_view = w1_flat.view::<F, D_TEST>().expect("view 1");
    let w2_view = w2_flat.view::<F, D_TEST>().expect("view 2");

    let rop =
        ring_opening_point_from_field::<F>(&[], 0, 0, BasisMode::Lagrange, BlockOrder::ColumnMajor)
            .expect("rop");

    let (y_ring1, folded1) = w1_view.evaluate_and_fold(&rop.b, &rop.a, lp.block_len, lp.num_blocks);
    let (y_ring2, folded2) = w2_view.evaluate_and_fold(&rop.b, &rop.a, lp.block_len, lp.num_blocks);

    let mut transcript = Blake2bTranscript::<F>::new(b"recursive_multi_claim/v1");

    // 2 claims, 2 hints, 2 y_rings, but only 1 commitment row supplied —
    // mismatched `commitments.len() != claim_group_sizes.len()`.
    let witnesses = [
        RecursiveHandlePoly::Witness(w1_view),
        RecursiveHandlePoly::Witness(w2_view),
    ];
    let result = QuadraticEquation::<F, D_TEST>::new_recursive_prover(
        &setup.ntt_shared,
        vec![rop.clone(), rop.clone()],
        vec![0usize, 1usize],
        &witnesses,
        vec![folded1, folded2],
        &[1usize, 1usize],
        lp.clone(),
        vec![hint1, hint2],
        &mut transcript,
        &[commit1.u.as_slice()],
        &[y_ring1, y_ring2],
        vec![F::one(); 2],
        2,
        setup.expanded.seed.max_stride,
    );

    assert!(
        matches!(result, Err(AkitaError::InvalidInput(_))),
        "expected InvalidInput on mismatched commitments vs claim_group_sizes, got {:?}",
        result.err()
    );
}
