#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

//! Digit-range wire-preservation epoch captured from baseline commit
//! `bc959ef34572aee143ba0114094b0b4212b4e111`.
//!
//! These fixtures are stable only for changes that claim to preserve this epoch's wire and
//! transcript behavior. An intentional protocol-changing PR must regenerate the affected entries
//! atomically, document the expected old/new deltas, and thereby establish a new protocol epoch.

use akita_field::{CanonicalBytes, Prime128Offset275};
use akita_prover::DigitRangeProver;
use akita_transcript::{labels, AkitaTranscript, LoggingTranscript};
use akita_types::{DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
use akita_verifier::AkitaStage1Verifier;
use std::sync::Arc;

mod common;

use common::{protocol_epoch_digest, serialize_stage1_payload, serialize_transcript_events};

type F = Prime128Offset275;

struct DigitRangeProtocolEpoch {
    basis: usize,
    proof_len: usize,
    event_count: usize,
    proof_digest: &'static str,
    event_digest: &'static str,
    output_point_digest: &'static str,
    range_image_eval: &'static str,
}

const DIGIT_RANGE_PROTOCOL_EPOCH: &[DigitRangeProtocolEpoch] = &[
    DigitRangeProtocolEpoch {
        basis: 4,
        proof_len: 144,
        event_count: 9,
        proof_digest: "abd1266b50d20cfe7b9a3ddf83e9b544",
        event_digest: "4491fc78622c42a3b932e6636f0fc667",
        output_point_digest: "7d9fafb86c4b931cffd679891031586b",
        range_image_eval: "a3597f88c2a199b05fe5c285c9366d15",
    },
    DigitRangeProtocolEpoch {
        basis: 8,
        proof_len: 272,
        event_count: 9,
        proof_digest: "9cda2cf6600a1cc46240993b5cf15b92",
        event_digest: "41f764d51da50603e671b3b9c4a57a8e",
        output_point_digest: "a1a125181986353fb9d84f93764b21d0",
        range_image_eval: "0092bf8c55d7731e323a60f3f9b603c7",
    },
    DigitRangeProtocolEpoch {
        basis: 16,
        proof_len: 432,
        event_count: 21,
        proof_digest: "3e0cc8bac0d1349a16c8e7d58fb0f3ee",
        event_digest: "250eb155de29e174e4e385bb21533a3a",
        output_point_digest: "a740f51168e899129c346b13f027e4ea",
        range_image_eval: "119a3282dc19d6bd68b08929aa82b8f4",
    },
    DigitRangeProtocolEpoch {
        basis: 32,
        proof_len: 592,
        event_count: 23,
        proof_digest: "fd0d48a7b9b1cf9e772df377a0c67849",
        event_digest: "f171889729c52a1829dc8949e1898c20",
        output_point_digest: "d7178d608a5edb274e90e2b583104b78",
        range_image_eval: "7335696cb66eb923716ac7c085344918",
    },
    DigitRangeProtocolEpoch {
        basis: 64,
        proof_len: 816,
        event_count: 39,
        proof_digest: "efb26d21239c4bff5d221216ab092e79",
        event_digest: "c4f7688506a87b26db7ce8e547c47e8e",
        output_point_digest: "97aa7de0a6605ff6eeeea0ec6afbe514",
        range_image_eval: "b62ebc70f19c7bcd71fa9c466d5c86b9",
    },
];

fn digit_witness(basis: usize, live_block_count: usize, low_variable_count: usize) -> Arc<[i8]> {
    let half_basis = basis / 2;
    (0..live_block_count * (1usize << low_variable_count))
        .map(|index| i8::try_from(index % half_basis).expect("small digit"))
        .collect::<Vec<_>>()
        .into()
}

fn field_vector_bytes(values: &[F]) -> Vec<u8> {
    values
        .iter()
        .flat_map(CanonicalBytes::to_bytes_le_vec)
        .collect()
}

fn canonical_hex(value: &F) -> String {
    value
        .to_bytes_le_vec()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn digit_range_proof_matches_protocol_epoch() {
    let high_variable_count = 3;
    let low_variable_count = 1;
    for expected in DIGIT_RANGE_PROTOCOL_EPOCH {
        let live_block_count = if expected.basis == 16 { 6 } else { 5 };
        let digit_witness = digit_witness(expected.basis, live_block_count, low_variable_count);
        let raw_point = [
            F::from_u64(u64::try_from(expected.basis + 3).unwrap()),
            F::from_u64(u64::try_from(expected.basis + 5).unwrap()),
            F::from_u64(u64::try_from(expected.basis + 7).unwrap()),
            F::from_u64(u64::try_from(expected.basis + 9).unwrap()),
        ];
        let equality_point = DigitRangeEqualityPoint::from_column_then_ring_challenges(
            &raw_point,
            high_variable_count,
            low_variable_count,
        )
        .expect("protocol-epoch equality point");
        let domain = FlatBooleanDomain::new(
            digit_witness.len(),
            high_variable_count + low_variable_count,
        )
        .expect("protocol-epoch domain");
        let prover = DigitRangeProver::new(
            digit_witness,
            DigitRangePlan::new(expected.basis).expect("protocol-epoch basis"),
            domain,
            equality_point.clone(),
        )
        .expect("protocol-epoch prover");
        let mut transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL));
        let (proof, output_point) = prover.prove(&mut transcript).expect("protocol-epoch proof");
        let proof_bytes = serialize_stage1_payload(&proof);
        let event_bytes = serialize_transcript_events(transcript.events());
        let verifier = AkitaStage1Verifier::new(
            equality_point,
            DigitRangePlan::new(expected.basis).expect("protocol-epoch basis"),
        );
        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL));
        let verified_point = verifier
            .verify(&proof, &mut verifier_transcript)
            .expect("protocol-epoch verification");

        assert_eq!(
            proof_bytes.len(),
            expected.proof_len,
            "basis {}",
            expected.basis
        );
        assert_eq!(
            transcript.events().len(),
            expected.event_count,
            "basis {}",
            expected.basis
        );
        assert_eq!(
            protocol_epoch_digest::<F>(&proof_bytes),
            expected.proof_digest,
            "proof bytes changed for basis {}",
            expected.basis
        );
        assert_eq!(
            protocol_epoch_digest::<F>(&event_bytes),
            expected.event_digest,
            "transcript events changed for basis {}",
            expected.basis
        );
        assert_eq!(
            protocol_epoch_digest::<F>(&field_vector_bytes(&output_point)),
            expected.output_point_digest,
            "output point changed for basis {}",
            expected.basis
        );
        assert_eq!(
            canonical_hex(&proof.range_image_evaluation),
            expected.range_image_eval,
            "range-image evaluation changed for basis {}",
            expected.basis
        );
        assert_eq!(verified_point, output_point, "basis {}", expected.basis);
        assert_eq!(
            verifier_transcript.events(),
            transcript.events(),
            "prover/verifier transcript events differ for basis {}",
            expected.basis
        );
    }
}
