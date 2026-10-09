use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_pcs::{
    config::{DigitConfig, Digits2},
    PreparedRoot, F,
};
use akita_labinius_prover::{lowered::flatten_image, prove_root_reduction_bytes};
use akita_labinius_verifier::{
    channel::{self, ClearChannel},
    lowered::LoweredRootLayout,
    root::{
        check_transparent_evaluations, verify_root_reduction_bytes, RootEvaluationClaims,
        RootProverOracle, RootVerifierOracle,
    },
};
use jolt_field::Zero;

use super::support::Case;
use crate::common::Recording;

struct CheckedOracle<'a, O> {
    inner: O,
    image: &'a [F],
    digits: Vec<u8>,
    calls: Vec<&'static str>,
}
impl<O: RootProverOracle<F>> RootProverOracle<F> for CheckedOracle<'_, O> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.calls.push("image");
        self.inner.bind_image(layout, channel)
    }
    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.calls.push("response");
        self.digits = digits.to_vec();
        self.inner.commit_response(layout, digits, channel)
    }
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<F>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.calls.push("discharge");
        check_transparent_evaluations(self.image, &self.digits, claims)?;
        self.inner.discharge(claims, channel)
    }
}

#[test]
fn reduction_claims_are_actual_table_evaluations_and_oracles_are_single_use() {
    let case = Case::<Digits2>::new(0);
    let image = flatten_image(&case.layout, &case.output.image).unwrap();
    let mut oracle = CheckedOracle {
        inner: case.prover.oracle(&case.output).unwrap(),
        image: &image,
        digits: Vec::new(),
        calls: Vec::new(),
    };
    let prepared = PreparedRoot::prepare(case.root.setup()).unwrap();
    let (proof, claims) = prove_root_reduction_bytes(
        &case.root,
        &prepared,
        Digits2::BASE,
        &case.source,
        &case.output.image,
        &case.point,
        case.value,
        &mut oracle,
    )
    .unwrap();
    assert_eq!(oracle.calls, ["image", "response", "discharge"]);
    assert_eq!(oracle.digits.len(), case.layout.witness_len());
    check_transparent_evaluations(&image, &oracle.digits, &claims).unwrap();
    case.verify(&proof).unwrap();
    let mut transcript = Recording::new(channel::new_root_prover().unwrap());
    assert_failed_prover(
        &mut oracle.inner,
        &case.layout,
        &oracle.digits,
        &claims,
        &mut transcript,
    );
    let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
    let verified = verify_root_reduction_bytes(
        &case.root,
        Digits2::BASE,
        &case.point,
        case.value,
        &mut verifier,
        &proof,
    )
    .unwrap();
    assert_eq!(verified, claims);
    assert_failed_verifier(&mut verifier, &case.layout, &claims, &mut transcript);
}

fn activity<S>(channel: &Recording<S>) -> (usize, usize, usize) {
    (
        channel.public.len(),
        channel.message_calls,
        channel.draws.len(),
    )
}

fn assert_failed_prover<O: RootProverOracle<F>, S: ClearChannel>(
    oracle: &mut O,
    layout: &LoweredRootLayout,
    digits: &[u8],
    claims: &RootEvaluationClaims<F>,
    channel: &mut Recording<S>,
) {
    let before = activity(channel);
    for _ in 0..2 {
        assert!(matches!(
            oracle.bind_image(layout, channel),
            Err(AkitaError::InvalidProof)
        ));
        assert!(matches!(
            oracle.commit_response(layout, digits, channel),
            Err(AkitaError::InvalidProof)
        ));
        assert!(matches!(
            oracle.discharge(claims, channel),
            Err(AkitaError::InvalidProof)
        ));
        assert_eq!(activity(channel), before);
    }
}

fn assert_failed_verifier<O: RootVerifierOracle<F>, S: ClearChannel>(
    oracle: &mut O,
    layout: &LoweredRootLayout,
    claims: &RootEvaluationClaims<F>,
    channel: &mut Recording<S>,
) {
    let before = activity(channel);
    for _ in 0..2 {
        assert!(matches!(
            oracle.bind_image(layout, channel),
            Err(AkitaError::InvalidProof)
        ));
        assert!(matches!(
            oracle.bind_response(layout, channel),
            Err(AkitaError::InvalidProof)
        ));
        assert!(matches!(
            oracle.discharge(claims, channel),
            Err(AkitaError::InvalidProof)
        ));
        assert_eq!(activity(channel), before);
    }
}

fn zero_claims(layout: &LoweredRootLayout) -> RootEvaluationClaims<F> {
    RootEvaluationClaims {
        image_point: vec![F::zero(); layout.image_log_len()],
        image_value: F::zero(),
        response_point: vec![F::zero(); layout.witness_log_len()],
        response_value: F::zero(),
    }
}

#[test]
fn wrong_layout_image_binding_permanently_fails_both_oracles() {
    let case = Case::<Digits2>::new(0);
    let wrong_layout = LoweredRootLayout::new(
        case.root.setup(),
        case.root.shape(),
        akita_labinius_pcs::config::Digits1::BASE,
    )
    .unwrap();
    assert_ne!(wrong_layout, case.layout);
    let claims = zero_claims(&case.layout);
    let digits = vec![0; case.layout.witness_len()];
    let mut state = Recording::new(channel::new_root_prover().unwrap());
    let mut prover = case.prover.oracle(&case.output).unwrap();
    assert!(matches!(
        prover.bind_image(&wrong_layout, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(activity(&state), (0, 0, 0));
    assert_failed_prover(&mut prover, &case.layout, &digits, &claims, &mut state);
    let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
    assert!(matches!(
        verifier.bind_image(&wrong_layout, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(activity(&state), (0, 0, 0));
    assert_failed_verifier(&mut verifier, &case.layout, &claims, &mut state);
}

struct RejectMessages<S>(S);
impl<S: ClearChannel> ClearChannel for RejectMessages<S> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.0.public(bytes)
    }
    fn message(&mut self, _: &mut [u8]) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidProof)
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        self.0.challenge_block()
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        self.0.fold_challenges(sampler, label, count)
    }
}

#[test]
fn response_channel_errors_permanently_fail_both_oracles() {
    let case = Case::<Digits2>::new(0);
    let claims = zero_claims(&case.layout);
    let digits = vec![0; case.layout.witness_len()];
    let mut state = Recording::new(RejectMessages(channel::new_root_prover().unwrap()));
    let mut prover = case.prover.oracle(&case.output).unwrap();
    prover.bind_image(&case.layout, &mut state).unwrap();
    assert!(matches!(
        prover.commit_response(&case.layout, &digits, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(state.message_calls, 1);
    assert_failed_prover(&mut prover, &case.layout, &digits, &claims, &mut state);
    let mut state = Recording::new(RejectMessages(channel::new_root_prover().unwrap()));
    let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
    verifier.bind_image(&case.layout, &mut state).unwrap();
    assert!(matches!(
        verifier.bind_response(&case.layout, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(state.message_calls, 1);
    assert_failed_verifier(&mut verifier, &case.layout, &claims, &mut state);
}

#[test]
fn initial_out_of_order_calls_permanently_fail_both_oracles() {
    let case = Case::<Digits2>::new(0);
    let claims = zero_claims(&case.layout);
    let digits = vec![0; case.layout.witness_len()];
    for discharge_first in [false, true] {
        let mut state = Recording::new(channel::new_root_prover().unwrap());
        let mut prover = case.prover.oracle(&case.output).unwrap();
        let result = if discharge_first {
            prover.discharge(&claims, &mut state)
        } else {
            prover.commit_response(&case.layout, &digits, &mut state)
        };
        assert!(matches!(result, Err(AkitaError::InvalidProof)));
        assert_eq!(activity(&state), (0, 0, 0));
        assert_failed_prover(&mut prover, &case.layout, &digits, &claims, &mut state);
        let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
        let result = if discharge_first {
            verifier.discharge(&claims, &mut state)
        } else {
            verifier.bind_response(&case.layout, &mut state)
        };
        assert!(matches!(result, Err(AkitaError::InvalidProof)));
        assert_eq!(activity(&state), (0, 0, 0));
        assert_failed_verifier(&mut verifier, &case.layout, &claims, &mut state);
    }
}

#[test]
fn duplicate_image_binding_permanently_fails_both_oracles() {
    let case = Case::<Digits2>::new(0);
    let claims = zero_claims(&case.layout);
    let digits = vec![0; case.layout.witness_len()];
    let mut state = Recording::new(channel::new_root_prover().unwrap());
    let mut prover = case.prover.oracle(&case.output).unwrap();
    prover.bind_image(&case.layout, &mut state).unwrap();
    let before = activity(&state);
    assert!(matches!(
        prover.bind_image(&case.layout, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(activity(&state), before);
    assert_failed_prover(&mut prover, &case.layout, &digits, &claims, &mut state);
    let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
    verifier.bind_image(&case.layout, &mut state).unwrap();
    let before = activity(&state);
    assert!(matches!(
        verifier.bind_image(&case.layout, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(activity(&state), before);
    assert_failed_verifier(&mut verifier, &case.layout, &claims, &mut state);
}

#[test]
fn duplicate_response_permanently_fails_both_oracles_before_valid_discharge() {
    let case = Case::<Digits2>::new(0);
    let image = flatten_image(&case.layout, &case.output.image).unwrap();
    let mut claims = zero_claims(&case.layout);
    claims.image_value = image[0];
    let digits = vec![0; case.layout.witness_len()];
    let mut healthy = channel::new_root_prover().unwrap();
    let mut oracle = case.prover.oracle(&case.output).unwrap();
    oracle.bind_image(&case.layout, &mut healthy).unwrap();
    oracle
        .commit_response(&case.layout, &digits, &mut healthy)
        .unwrap();
    oracle.discharge(&claims, &mut healthy).unwrap();
    let proof = healthy.narg_string().to_vec();

    let mut state = Recording::new(channel::new_root_prover().unwrap());
    let mut prover = case.prover.oracle(&case.output).unwrap();
    prover.bind_image(&case.layout, &mut state).unwrap();
    prover
        .commit_response(&case.layout, &digits, &mut state)
        .unwrap();
    let before = activity(&state);
    assert!(matches!(
        prover.commit_response(&case.layout, &digits, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(activity(&state), before);
    assert_failed_prover(&mut prover, &case.layout, &digits, &claims, &mut state);

    let mut state = Recording::new(channel::new_root_verifier(&proof).unwrap());
    let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
    verifier.bind_image(&case.layout, &mut state).unwrap();
    verifier.bind_response(&case.layout, &mut state).unwrap();
    let before = activity(&state);
    assert!(matches!(
        verifier.bind_response(&case.layout, &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(activity(&state), before);
    assert_failed_verifier(&mut verifier, &case.layout, &claims, &mut state);
}

#[test]
fn invalid_response_table_length_permanently_fails_prover() {
    let case = Case::<Digits2>::new(0);
    let mut state = Recording::new(channel::new_root_prover().unwrap());
    let mut prover = case.prover.oracle(&case.output).unwrap();
    prover.bind_image(&case.layout, &mut state).unwrap();
    let before = activity(&state);
    assert!(matches!(
        prover.commit_response(&case.layout, &[0], &mut state),
        Err(AkitaError::InvalidProof)
    ));
    assert_eq!(activity(&state), before);
    assert_failed_prover(
        &mut prover,
        &case.layout,
        &vec![0; case.layout.witness_len()],
        &zero_claims(&case.layout),
        &mut state,
    );
}

#[test]
fn response_binding_reads_commitment_before_any_coefficient_challenge() {
    let case = Case::<Digits2>::new(0);
    let proof = case.prove();
    let regions = case.regions(&proof);
    let (_, prefix, _) = *regions
        .iter()
        .find(|(name, _, _)| *name == "W length")
        .unwrap();
    let (_, start, len) = *regions
        .iter()
        .find(|(name, _, _)| *name == "W commitment")
        .unwrap();
    let frame = &proof[prefix..start + len];
    let challenges = |frame: &[u8]| {
        let state = akita_transcript::new_verifier_channel(b"oracle-order/v1", b"", frame).unwrap();
        let mut record = Recording::new(state);
        let mut oracle = case.verifier.oracle(&case.output.committed_group).unwrap();
        oracle.bind_image(&case.layout, &mut record).unwrap();
        oracle.bind_response(&case.layout, &mut record).unwrap();
        assert_eq!(record.message_calls, 2);
        assert_eq!(record.messages[0].len(), 8);
        assert_eq!(record.messages[1].len(), len);
        assert!(record.draws.is_empty());
        (0..8)
            .map(|_| record.challenge_block().unwrap())
            .collect::<Vec<_>>()
    };
    let honest = challenges(frame);
    let mut changed = frame.to_vec();
    // The final coefficient's low byte remains canonical after this mutation.
    changed[frame.len() - 16] ^= 1;
    let altered = challenges(&changed);
    for (before, after) in honest.iter().zip(&altered) {
        assert_ne!(before, after);
    }
}
