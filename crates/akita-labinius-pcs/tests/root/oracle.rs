use akita_error::AkitaError;
use akita_labinius_pcs::{
    config::{DigitConfig, Digits2},
    F,
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
    let (proof, claims) = prove_root_reduction_bytes(
        &case.root,
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
    let mut transcript = channel::new_root_prover().unwrap();
    assert!(oracle
        .inner
        .bind_image(&case.layout, &mut transcript)
        .is_err());
    assert!(oracle
        .inner
        .commit_response(&case.layout, &oracle.digits, &mut transcript)
        .is_err());
    assert!(oracle.inner.discharge(&claims, &mut transcript).is_err());
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
    assert!(verifier.bind_image(&case.layout, &mut transcript).is_err());
    assert!(verifier
        .bind_response(&case.layout, &mut transcript)
        .is_err());
    assert!(verifier.discharge(&claims, &mut transcript).is_err());
}

#[test]
fn out_of_order_calls_and_repeated_bindings_reject() {
    let case = Case::<Digits2>::new(0);
    let claims = RootEvaluationClaims {
        image_point: vec![F::zero(); case.layout.image_log_len()],
        image_value: F::zero(),
        response_point: vec![F::zero(); case.layout.witness_log_len()],
        response_value: F::zero(),
    };
    let mut state = channel::new_root_prover().unwrap();
    let mut prover = case.prover.oracle(&case.output).unwrap();
    assert!(prover
        .commit_response(&case.layout, &[], &mut state)
        .is_err());
    assert!(prover.discharge(&claims, &mut state).is_err());
    let mut prover = case.prover.oracle(&case.output).unwrap();
    prover.bind_image(&case.layout, &mut state).unwrap();
    assert!(prover.bind_image(&case.layout, &mut state).is_err());
    assert!(prover.discharge(&claims, &mut state).is_err());
    let mut prover = case.prover.oracle(&case.output).unwrap();
    prover.bind_image(&case.layout, &mut state).unwrap();
    assert!(prover
        .commit_response(&case.layout, &[0], &mut state)
        .is_err());
    let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
    assert!(verifier.bind_response(&case.layout, &mut state).is_err());
    assert!(verifier.discharge(&claims, &mut state).is_err());
    let mut verifier = case.verifier.oracle(&case.output.committed_group).unwrap();
    verifier.bind_image(&case.layout, &mut state).unwrap();
    assert!(verifier.bind_image(&case.layout, &mut state).is_err());
    assert!(verifier.discharge(&claims, &mut state).is_err());
}

#[test]
fn response_binding_reads_commitment_before_any_coefficient_challenge() {
    use crate::common::Recording;
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
