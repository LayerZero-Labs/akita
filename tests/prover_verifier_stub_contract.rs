#![allow(missing_docs)]

use hachi_pcs::algebra::Fp64;
use hachi_pcs::error::HachiError;
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{
    verify_opening_stub, Blake2bTranscript, RingCommitment, RingOpenProof, RingOpening, Transcript,
};

type F = Fp64<4294967197>;
const D: usize = 64;

#[test]
fn verifier_stub_returns_placeholder_error() {
    let mut t = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let commitment = RingCommitment::<F, D> { u: Vec::new() };
    let proof = RingOpenProof::<F, D> {
        opening: RingOpening {
            s: Vec::new(),
            t_hat: Vec::new(),
        },
    };
    let err = verify_opening_stub(&mut t, &commitment, &proof).unwrap_err();
    match err {
        HachiError::InvalidInput(msg) => assert!(msg.contains("stub")),
        other => panic!("unexpected error: {other:?}"),
    }
}
