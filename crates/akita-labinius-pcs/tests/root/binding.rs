use akita_algebra::binary::{BinaryField128, BinaryField192};
use akita_error::AkitaError;
use akita_labinius_pcs::{config::Digits2, F};
use akita_labinius_prover::lowered::flatten_image;
use akita_labinius_verifier::root::{RootEvaluationClaims, RootProverOracle, RootVerifierOracle};
use jolt_field::{One, Ring, Zero};

use super::support::Case;
use crate::common::Recording;

const DOMAIN: &[u8] = b"root-oracle-binding-test/v1";

fn claims(case: &Case<Digits2>) -> RootEvaluationClaims<F> {
    let image = flatten_image(&case.layout, &case.output.image).unwrap();
    let image_point = vec![F::from_u64(2); case.layout.image_log_len()];
    RootEvaluationClaims {
        image_value: akita_algebra::poly::multilinear_eval(&image, &image_point).unwrap(),
        image_point,
        response_point: vec![F::zero(); case.layout.witness_log_len()],
        response_value: F::zero(),
    }
}

fn prove_oracle(
    case: &Case<Digits2>,
    context: &[u8],
    digits: &[u8],
    claims: &RootEvaluationClaims<F>,
) -> Vec<u8> {
    let mut state = akita_transcript::new_prover_channel(DOMAIN, context).unwrap();
    let mut oracle = case.prover.oracle(&case.output).unwrap();
    oracle.bind_image(&case.layout, &mut state).unwrap();
    oracle
        .commit_response(&case.layout, digits, &mut state)
        .unwrap();
    oracle.discharge(claims, &mut state).unwrap();
    state.narg_string().to_vec()
}

fn replay_oracle(
    case: &Case<Digits2>,
    context: &[u8],
    claims: &RootEvaluationClaims<F>,
    proof: &[u8],
) -> (Result<(), AkitaError>, usize, usize) {
    let state = akita_transcript::new_verifier_channel(DOMAIN, context, proof).unwrap();
    let mut record = Recording::new(state);
    let mut oracle = case.verifier.oracle(&case.output.committed_group).unwrap();
    oracle.bind_image(&case.layout, &mut record).unwrap();
    // Assert that the canonical W frame is valid before entering discharge.
    oracle.bind_response(&case.layout, &mut record).unwrap();
    assert_eq!(record.message_calls, 2);
    assert!(record.draws.is_empty());
    let result = oracle.discharge(claims, &mut record);
    (result, record.message_calls, record.draws.len())
}

fn native_discharge_error(result: (Result<(), AkitaError>, usize, usize)) {
    let (result, messages, draws) = result;
    assert_eq!(messages, 4, "discharge must read the complete inner frame");
    assert_eq!(draws, 1, "discharge must derive its nested session");
    let error = result.unwrap_err();
    assert!(
        !matches!(error, AkitaError::Internal(_)),
        "native replay returned Internal: {error}"
    );
}

#[test]
fn discharge_point_only_and_value_only_substitution_reach_native_replay_and_reject() {
    let case = Case::<Digits2>::new(0);
    let claims = claims(&case);
    let proof = prove_oracle(
        &case,
        b"run-a",
        &vec![0; case.layout.witness_len()],
        &claims,
    );
    replay_oracle(&case, b"run-a", &claims, &proof).0.unwrap();
    for image in [false, true] {
        let mut changed_point = claims.clone();
        if image {
            changed_point.image_point[0] += F::one();
        } else {
            changed_point.response_point[0] += F::one();
        }
        // Values and the other point remain exactly the honest claims.
        assert_eq!(changed_point.image_value, claims.image_value);
        assert_eq!(changed_point.response_value, claims.response_value);
        native_discharge_error(replay_oracle(&case, b"run-a", &changed_point, &proof));
        let mut changed_value = claims.clone();
        if image {
            changed_value.image_value += F::one();
        } else {
            changed_value.response_value += F::one();
        }
        assert_eq!(changed_value.image_point, claims.image_point);
        assert_eq!(changed_value.response_point, claims.response_point);
        native_discharge_error(replay_oracle(&case, b"run-a", &changed_value, &proof));
    }
}

#[test]
fn valid_other_run_w_commitment_and_opening_reject_after_response_frame_admission() {
    let case = Case::<Digits2>::new(0);
    let claims = claims(&case);
    let digits_a = vec![0; case.layout.witness_len()];
    let mut digits_b = vec![1; case.layout.witness_len()];
    // Both valid tables open to zero at the fixed Boolean response point.
    digits_b[0] = 0;
    let proof_a = prove_oracle(&case, b"run-a", &digits_a, &claims);
    let proof_b = prove_oracle(&case, b"run-b", &digits_b, &claims);
    replay_oracle(&case, b"run-a", &claims, &proof_a).0.unwrap();
    replay_oracle(&case, b"run-b", &claims, &proof_b).0.unwrap();
    let w_size = u64::from_le_bytes(proof_a[..8].try_into().unwrap()) as usize;
    assert_eq!(proof_a[..8], proof_b[..8]);
    assert_ne!(proof_a[8..8 + w_size], proof_b[8..8 + w_size]);
    // Transplant the valid W frame and its complete grouped opening together.
    // Same Y, both points, both values and layout; only the parent run differs.
    native_discharge_error(replay_oracle(&case, b"run-a", &claims, &proof_b));
}

#[test]
fn host_field_change_with_identical_zero_point_and_value_rejects() {
    let case = Case::<Digits2>::new(0);
    let point128 = vec![BinaryField128::ZERO; case.point.len()];
    let point192 = vec![BinaryField192::ZERO; case.point.len()];
    let source128 = vec![0u128; case.source.len()];
    let source192 = vec![0u64; case.source.len()];
    let image128 = case.prover.commit::<BinaryField128>(&source128).unwrap();
    let image192 = case.prover.commit::<BinaryField192>(&source192).unwrap();
    assert_eq!(image128.committed_group, image192.committed_group);
    let proof128 = case
        .prover
        .open::<BinaryField128>(&source128, &image128, &point128, BinaryField128::ZERO)
        .unwrap();
    let proof192 = case
        .prover
        .open::<BinaryField192>(&source192, &image192, &point192, BinaryField192::ZERO)
        .unwrap();
    case.verifier
        .verify::<BinaryField128>(
            &image128.committed_group,
            &point128,
            BinaryField128::ZERO,
            &proof128,
        )
        .unwrap();
    case.verifier
        .verify::<BinaryField192>(
            &image128.committed_group,
            &point192,
            BinaryField192::ZERO,
            &proof192,
        )
        .unwrap();
    assert!(matches!(
        case.verifier.verify::<BinaryField192>(
            &image128.committed_group,
            &point192,
            BinaryField192::ZERO,
            &proof128,
        ),
        Err(AkitaError::InvalidProof)
    ));
}
