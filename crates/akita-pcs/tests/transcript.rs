#![allow(missing_docs)]

use akita_algebra::Fp64;
use akita_transcript::{labels, Blake2bTranscript, KeccakTranscript, Transcript};

type F = Fp64<4294967197>;

fn sample_schedule<T: Transcript<F>>(transcript: &mut T) -> F {
    transcript.append_bytes(labels::ABSORB_COMMITMENT, b"commitment-a");
    transcript.append_bytes(labels::ABSORB_COMMITMENT, b"commitment-b");
    transcript.append_serde(labels::ABSORB_EVALUATION_CLAIMS, &42u64);
    let rho = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

    transcript.append_bytes(labels::ABSORB_RING_SWITCH_MESSAGE, b"ring-switch");
    let zeta = transcript.challenge_scalar(labels::CHALLENGE_RING_SWITCH);

    transcript.append_field(labels::ABSORB_SUMCHECK_ROUND, &(rho + zeta));
    let r = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);

    transcript.append_field(labels::ABSORB_STOP_CONDITION, &r);
    transcript.challenge_scalar(labels::CHALLENGE_STOP_CONDITION)
}

#[test]
fn transcript_is_deterministic_for_identical_schedule() {
    let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_schedule(&mut t1);
    let c2 = sample_schedule(&mut t2);
    assert_eq!(c1, c2);
}

#[test]
fn transcript_differs_when_label_changes() {
    let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    t1.append_bytes(labels::ABSORB_COMMITMENT, b"same-bytes");
    t2.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"same-bytes");
    let c1 = t1.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = t2.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    assert_ne!(c1, c2);
}

#[test]
fn transcript_differs_when_absorb_order_changes() {
    let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    t1.append_bytes(labels::ABSORB_COMMITMENT, b"A");
    t1.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"B");

    t2.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"B");
    t2.append_bytes(labels::ABSORB_COMMITMENT, b"A");

    let c1 = t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let c2 = t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_ne!(c1, c2);
}

#[test]
fn transcript_reset_restores_domain_state() {
    let mut t = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    t.append_bytes(labels::ABSORB_COMMITMENT, b"before-reset");
    let _ = t.challenge_scalar(labels::CHALLENGE_STOP_CONDITION);

    t.reset(labels::DOMAIN_AKITA_PROTOCOL);
    let after_reset = sample_schedule(&mut t);

    let mut fresh = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let fresh_challenge = sample_schedule(&mut fresh);
    assert_eq!(after_reset, fresh_challenge);
}

#[test]
fn keccak_transcript_is_deterministic_for_identical_schedule() {
    let mut t1 = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_schedule(&mut t1);
    let c2 = sample_schedule(&mut t2);
    assert_eq!(c1, c2);
}

#[test]
fn keccak_transcript_differs_when_label_changes() {
    let mut t1 = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    t1.append_bytes(labels::ABSORB_COMMITMENT, b"same-bytes");
    t2.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"same-bytes");
    let c1 = t1.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = t2.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    assert_ne!(c1, c2);
}

#[test]
fn keccak_transcript_differs_when_absorb_order_changes() {
    let mut t1 = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    t1.append_bytes(labels::ABSORB_COMMITMENT, b"A");
    t1.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"B");

    t2.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"B");
    t2.append_bytes(labels::ABSORB_COMMITMENT, b"A");

    let c1 = t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let c2 = t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_ne!(c1, c2);
}

#[test]
fn keccak_transcript_reset_restores_domain_state() {
    let mut t = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    t.append_bytes(labels::ABSORB_COMMITMENT, b"before-reset");
    let _ = t.challenge_scalar(labels::CHALLENGE_STOP_CONDITION);

    t.reset(labels::DOMAIN_AKITA_PROTOCOL);
    let after_reset = sample_schedule(&mut t);

    let mut fresh = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let fresh_challenge = sample_schedule(&mut fresh);
    assert_eq!(after_reset, fresh_challenge);
}

#[test]
fn blake2b_and_keccak_diverge_on_same_schedule() {
    let mut blake = Blake2bTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut keccak = KeccakTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let b = sample_schedule(&mut blake);
    let k = sample_schedule(&mut keccak);
    assert_ne!(b, k);
}
