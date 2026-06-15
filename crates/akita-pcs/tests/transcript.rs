#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_field::{ExtField, Fp32, Fp64, FpExt2, FpExt4, NegOneNr};
use akita_transcript::{
    append_ext_field, labels, sample_ext_challenge, AkitaTranscript, Transcript,
};

type F = Fp64<4294967197>;
type Base = Fp32<251>;
type BaseFpExt2 = FpExt2<Base, NegOneNr>;
type BaseFpExt4 = FpExt4<Base>;

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
    let mut t1 = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_schedule(&mut t1);
    let c2 = sample_schedule(&mut t2);
    assert_eq!(c1, c2);
}

#[test]
fn production_transcript_ignores_label_changes() {
    let mut t1 = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

    t1.append_bytes(labels::ABSORB_COMMITMENT, b"same-bytes");
    t2.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"same-bytes");
    let c1 = t1.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = t2.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    assert_eq!(c1, c2);
}

#[test]
fn transcript_differs_when_absorb_order_changes() {
    let mut t1 = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);

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
    let mut t = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    t.append_bytes(labels::ABSORB_COMMITMENT, b"before-reset");
    let _ = t.challenge_scalar(labels::CHALLENGE_STOP_CONDITION);

    t.reset(labels::DOMAIN_AKITA_PROTOCOL);
    let after_reset = sample_schedule(&mut t);

    let mut fresh = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let fresh_challenge = sample_schedule(&mut fresh);
    assert_eq!(after_reset, fresh_challenge);
}

#[test]
fn transcript_differs_when_session_label_changes() {
    let mut t1 = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<F>::new(b"akita-transcript/other-session");

    let c1 = sample_schedule(&mut t1);
    let c2 = sample_schedule(&mut t2);
    assert_ne!(c1, c2);
}

#[test]
fn extension_challenge_sampling_is_deterministic() {
    let mut t1 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_ext_challenge::<Base, BaseFpExt2, _>(&mut t1, labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = sample_ext_challenge::<Base, BaseFpExt2, _>(&mut t2, labels::CHALLENGE_SUMCHECK_ROUND);
    assert_eq!(c1, c2);
}

#[test]
fn degree_one_extension_challenge_uses_scalar_label() {
    let mut ext = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut scalar = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_ext_challenge::<Base, Base, _>(&mut ext, labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = scalar.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    assert_eq!(c1, c2);
}

#[test]
fn quartic_extension_challenge_sampling_is_deterministic() {
    let mut t1 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut t2 = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);

    let c1 = sample_ext_challenge::<Base, BaseFpExt4, _>(&mut t1, labels::CHALLENGE_SUMCHECK_ROUND);
    let c2 = sample_ext_challenge::<Base, BaseFpExt4, _>(&mut t2, labels::CHALLENGE_SUMCHECK_ROUND);
    assert_eq!(c1, c2);
}

#[test]
fn extension_challenge_sampling_does_not_project_to_base_field() {
    let mut transcript = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let challenge = sample_ext_challenge::<Base, BaseFpExt4, _>(
        &mut transcript,
        labels::CHALLENGE_SUMCHECK_ROUND,
    );
    let limbs = challenge.to_base_vec();

    assert_eq!(limbs.len(), 4);
    assert!(limbs[1..]
        .iter()
        .any(|limb: &Base| *limb != Base::from_u64(0)));
}

#[test]
fn append_ext_field_is_coordinate_order_sensitive() {
    let x = BaseFpExt2::new(Base::from_u64(1), Base::from_u64(2));
    let y = BaseFpExt2::new(Base::from_u64(2), Base::from_u64(1));

    let mut tx = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut ty = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    append_ext_field::<Base, BaseFpExt2, _>(&mut tx, labels::ABSORB_EVALUATION_CLAIMS, &x);
    append_ext_field::<Base, BaseFpExt2, _>(&mut ty, labels::ABSORB_EVALUATION_CLAIMS, &y);

    let cx = tx.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let cy = ty.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_ne!(cx, cy);
}

#[test]
fn append_degree_one_ext_field_uses_scalar_label() {
    let x = Base::from_u64(7);

    let mut ext = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut scalar = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    append_ext_field::<Base, Base, _>(&mut ext, labels::ABSORB_EVALUATION_CLAIMS, &x);
    scalar.append_field(labels::ABSORB_EVALUATION_CLAIMS, &x);

    let c1 = ext.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let c2 = scalar.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_eq!(c1, c2);
}

#[test]
fn append_fp_ext4_ext_field_is_coordinate_order_sensitive() {
    let x = BaseFpExt4::new([
        Base::from_u64(1),
        Base::from_u64(2),
        Base::from_u64(3),
        Base::from_u64(4),
    ]);
    let y = BaseFpExt4::new([
        Base::from_u64(1),
        Base::from_u64(2),
        Base::from_u64(4),
        Base::from_u64(3),
    ]);

    let mut tx = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut ty = AkitaTranscript::<Base>::new(labels::DOMAIN_AKITA_PROTOCOL);
    append_ext_field::<Base, BaseFpExt4, _>(&mut tx, labels::ABSORB_EVALUATION_CLAIMS, &x);
    append_ext_field::<Base, BaseFpExt4, _>(&mut ty, labels::ABSORB_EVALUATION_CLAIMS, &y);

    let cx = tx.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    let cy = ty.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);
    assert_ne!(cx, cy);
}

#[test]
fn append_fp_ext4_uses_univariate_limb_order() {
    let x = BaseFpExt4::new([
        Base::from_u64(1),
        Base::from_u64(2),
        Base::from_u64(3),
        Base::from_u64(4),
    ]);

    assert_eq!(
        <BaseFpExt4 as ExtField<Base>>::to_base_vec(&x),
        vec![
            Base::from_u64(1),
            Base::from_u64(2),
            Base::from_u64(3),
            Base::from_u64(4)
        ]
    );
}
