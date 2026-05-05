#![allow(missing_docs)]

use akita_field::{ExtField, Fp2, Fp32, Fp4, Fp64, NegOneNr, UnitNr};
use akita_transcript::{
    append_ext_field, labels, sample_ext_challenge, Blake2bTranscript, KeccakTranscript, Transcript,
};

type F = Fp64<4294967197>;
type Base = Fp32<251>;
type BaseFp2 = Fp2<Base, NegOneNr>;
type BaseFp4 = Fp4<Base, NegOneNr, UnitNr>;

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

#[test]
fn extension_challenge_sampling_is_deterministic() {
    let mut t1 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_ext_challenge::<Base, BaseFp2, _>(&mut t1, labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = sample_ext_challenge::<Base, BaseFp2, _>(&mut t2, labels::CHALLENGE_SUMCHECK_ROUND);
    assert_eq!(c1, c2);
}

#[test]
fn degree_one_extension_challenge_uses_scalar_label() {
    let mut ext = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut scalar = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_ext_challenge::<Base, Base, _>(&mut ext, labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = scalar.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    assert_eq!(c1, c2);
}

#[test]
fn quartic_extension_challenge_sampling_is_deterministic() {
    let mut t1 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_ext_challenge::<Base, BaseFp4, _>(&mut t1, labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = sample_ext_challenge::<Base, BaseFp4, _>(&mut t2, labels::CHALLENGE_SUMCHECK_ROUND);
    assert_eq!(c1, c2);
}

#[test]
fn extension_challenge_sampling_does_not_project_to_base_field() {
    let mut transcript = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let challenge =
        sample_ext_challenge::<Base, BaseFp4, _>(&mut transcript, labels::CHALLENGE_SUMCHECK_ROUND);
    let limbs = challenge.to_base_vec();

    assert_eq!(limbs.len(), 4);
    assert!(limbs[1..]
        .iter()
        .any(|limb: &Base| *limb != Base::from_u64(0)));
}

#[test]
fn append_ext_field_is_coordinate_order_sensitive() {
    let x = BaseFp2::new(Base::from_u64(1), Base::from_u64(2));
    let y = BaseFp2::new(Base::from_u64(2), Base::from_u64(1));

    let mut tx = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut ty = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    append_ext_field::<Base, BaseFp2, _>(&mut tx, labels::ABSORB_EVALUATION_CLAIMS, &x);
    append_ext_field::<Base, BaseFp2, _>(&mut ty, labels::ABSORB_EVALUATION_CLAIMS, &y);

    let cx = tx.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let cy = ty.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_ne!(cx, cy);
}

#[test]
fn append_degree_one_ext_field_uses_scalar_label() {
    let x = Base::from_u64(7);

    let mut ext = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut scalar = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    append_ext_field::<Base, Base, _>(&mut ext, labels::ABSORB_EVALUATION_CLAIMS, &x);
    scalar.append_field(labels::ABSORB_EVALUATION_CLAIMS, &x);

    let c1 = ext.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let c2 = scalar.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_eq!(c1, c2);
}

#[test]
fn append_fp4_ext_field_is_coordinate_order_sensitive() {
    let x = BaseFp4::new(
        BaseFp2::new(Base::from_u64(1), Base::from_u64(2)),
        BaseFp2::new(Base::from_u64(3), Base::from_u64(4)),
    );
    let y = BaseFp4::new(
        BaseFp2::new(Base::from_u64(1), Base::from_u64(2)),
        BaseFp2::new(Base::from_u64(4), Base::from_u64(3)),
    );

    let mut tx = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut ty = Blake2bTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    append_ext_field::<Base, BaseFp4, _>(&mut tx, labels::ABSORB_EVALUATION_CLAIMS, &x);
    append_ext_field::<Base, BaseFp4, _>(&mut ty, labels::ABSORB_EVALUATION_CLAIMS, &y);

    let cx = tx.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let cy = ty.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_ne!(cx, cy);
}
