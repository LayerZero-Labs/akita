#![allow(missing_docs)]

use hachi_pcs::algebra::Fp64;
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{Blake2bTranscript, Transcript};

type F = Fp64<4294967197>;

#[test]
fn label_namespace_does_not_include_dory_literals() {
    let banned = ["vmv_", "beta", "alpha", "gamma", "final_e", "dory"];
    for label in labels::all_labels() {
        let text = std::str::from_utf8(label).expect("labels must be valid utf8 literals");
        for needle in &banned {
            assert!(
                !text.contains(needle),
                "label `{text}` must not contain banned token `{needle}`"
            );
        }
    }
}

fn run_hachi_schedule<T: Transcript<F>>(transcript: &mut T) -> (F, F, F) {
    transcript.append_bytes(labels::ABSORB_COMMITMENT, b"C");
    transcript.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"O");
    let c_linear_relation = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

    transcript.append_bytes(labels::ABSORB_RING_SWITCH_MESSAGE, b"RS");
    let c_ring_switch = transcript.challenge_scalar(labels::CHALLENGE_RING_SWITCH);

    transcript.append_bytes(labels::ABSORB_SUMCHECK_ROUND, b"SC1");
    let c_sumcheck = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
    transcript.append_bytes(labels::ABSORB_STOP_CONDITION, b"STOP");
    let _ = transcript.challenge_scalar(labels::CHALLENGE_STOP_CONDITION);

    (c_linear_relation, c_ring_switch, c_sumcheck)
}

#[test]
fn schedule_is_replayable_with_hachi_labels() {
    let mut prover = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let mut verifier = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    assert_eq!(
        run_hachi_schedule(&mut prover),
        run_hachi_schedule(&mut verifier)
    );
}

#[test]
fn schedule_detects_reordered_round_messages() {
    let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);

    t1.append_bytes(labels::ABSORB_COMMITMENT, b"C");
    t1.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"O");
    let a = t1.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

    t2.append_bytes(labels::ABSORB_EVALUATION_CLAIMS, b"O");
    t2.append_bytes(labels::ABSORB_COMMITMENT, b"C");
    let b = t2.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

    assert_ne!(a, b);
}
