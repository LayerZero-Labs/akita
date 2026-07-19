#![allow(missing_docs)]
#![cfg(feature = "logging-transcript")]

//! F1 wire-preservation oracle epoch, captured from the literal #311 head
//! `bc959ef34572aee143ba0114094b0b4212b4e111`.
//!
//! These fixtures are stable only for changes that claim to preserve this epoch's wire and
//! transcript behavior. An intentional protocol-changing PR must regenerate the affected entries
//! atomically, document the expected old/new deltas, and thereby establish a new oracle epoch.

use akita_field::{CanonicalBytes, Prime128Offset275};
use akita_prover::DigitRangeProver;
use akita_serialization::{AkitaSerialize, Compress};
use akita_transcript::{labels, AkitaTranscript, LoggingTranscript, Transcript, TranscriptEvent};
use akita_types::{AkitaStage1Proof, DigitRangeEqualityPoint, DigitRangePlan, FlatBooleanDomain};
use akita_verifier::AkitaStage1Verifier;
use std::sync::Arc;

type F = Prime128Offset275;

struct OracleEpochEntry {
    basis: usize,
    proof_len: usize,
    event_count: usize,
    proof_digest: &'static str,
    event_digest: &'static str,
    output_point_digest: &'static str,
    range_image_eval: &'static str,
}

const PR311_ORACLE_EPOCH: &[OracleEpochEntry] = &[
    OracleEpochEntry {
        basis: 4,
        proof_len: 144,
        event_count: 9,
        proof_digest: "c61248b8bf8c61deea2524d26db69185",
        event_digest: "5ddaed9b348525b4f9d2ef345f672ffb",
        output_point_digest: "e783caec7cbe466efe4f1c34c184cd00",
        range_image_eval: "a3597f88c2a199b05fe5c285c9366d15",
    },
    OracleEpochEntry {
        basis: 8,
        proof_len: 272,
        event_count: 9,
        proof_digest: "5da8770dddfb285c8208fd9aae0b4f6b",
        event_digest: "50334f5bdbb07670569bc029f37ef121",
        output_point_digest: "7b8ff2e661c4f5704694ff7a07c16f7d",
        range_image_eval: "0092bf8c55d7731e323a60f3f9b603c7",
    },
    OracleEpochEntry {
        basis: 16,
        proof_len: 432,
        event_count: 21,
        proof_digest: "cace743ea85f297bc72a1f796a6d9f8c",
        event_digest: "3220b489f953f08382a3eb9805176043",
        output_point_digest: "98834bf262f55f45cb47b9da459ba46d",
        range_image_eval: "119a3282dc19d6bd68b08929aa82b8f4",
    },
    OracleEpochEntry {
        basis: 32,
        proof_len: 592,
        event_count: 23,
        proof_digest: "ea696003d2f1dedad5b27b813b9d9322",
        event_digest: "30c6da082fa039f7c642be9a09efe4fc",
        output_point_digest: "0cb5f64dbe2e9bf5260ac5811c36476f",
        range_image_eval: "7335696cb66eb923716ac7c085344918",
    },
    OracleEpochEntry {
        basis: 64,
        proof_len: 816,
        event_count: 39,
        proof_digest: "26d7092d9f07de2be3da47e10cbc232b",
        event_digest: "14621564d2b7e9863e29c6e5b9cf2c04",
        output_point_digest: "1e8623c7cac587fd9b45c1fe86dc4dd8",
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

fn serialize_stage1(proof: &AkitaStage1Proof<F>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for stage in &proof.stages {
        stage
            .sumcheck_proof
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .expect("serialize sumcheck");
        for claim in &stage.child_claims {
            claim
                .serialize_with_mode(&mut bytes, Compress::Yes)
                .expect("serialize child claim");
        }
    }
    proof
        .s_claim
        .serialize_with_mode(&mut bytes, Compress::Yes)
        .expect("serialize range-image claim");
    bytes
}

fn serialize_events(events: &[TranscriptEvent]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for event in events {
        match event {
            TranscriptEvent::Preamble {
                bytes_digest,
                bytes_len,
            } => {
                bytes.push(0);
                bytes.extend_from_slice(bytes_digest);
                bytes.extend_from_slice(&u64::try_from(*bytes_len).unwrap().to_le_bytes());
            }
            TranscriptEvent::Absorb {
                label,
                bytes_digest,
                bytes_len,
            } => {
                bytes.push(1);
                bytes.extend_from_slice(&u64::try_from(label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(label);
                bytes.extend_from_slice(bytes_digest);
                bytes.extend_from_slice(&u64::try_from(*bytes_len).unwrap().to_le_bytes());
            }
            TranscriptEvent::Squeeze { label, len } => {
                bytes.push(2);
                bytes.extend_from_slice(&u64::try_from(label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(label);
                bytes.extend_from_slice(&u64::try_from(*len).unwrap().to_le_bytes());
            }
            TranscriptEvent::Wire {
                label,
                bytes_digest,
                bytes_len,
            } => {
                bytes.push(3);
                bytes.extend_from_slice(&u64::try_from(label.len()).unwrap().to_le_bytes());
                bytes.extend_from_slice(label);
                bytes.extend_from_slice(bytes_digest);
                bytes.extend_from_slice(&u64::try_from(*bytes_len).unwrap().to_le_bytes());
            }
        }
    }
    bytes
}

fn oracle_digest(payload: &[u8]) -> String {
    let mut transcript = AkitaTranscript::<F>::new(b"akita/digit-range-f1-oracle-digest");
    transcript.append_bytes(labels::ABSORB_PROVER_V, payload);
    transcript
        .challenge_scalar(labels::CHALLENGE_SUMCHECK_BATCH)
        .to_bytes_le_vec()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
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
fn stage1_matches_f1_pr311_oracle_epoch() {
    let high_variable_count = 3;
    let low_variable_count = 1;
    for expected in PR311_ORACLE_EPOCH {
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
        .expect("oracle equality point");
        let domain = FlatBooleanDomain::new(
            digit_witness.len(),
            high_variable_count + low_variable_count,
        )
        .expect("oracle domain");
        let prover = DigitRangeProver::new(
            digit_witness,
            DigitRangePlan::new(expected.basis).expect("oracle basis"),
            domain,
            equality_point.clone(),
        )
        .expect("oracle prover");
        let mut transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL));
        let (proof, output_point) = prover.prove(&mut transcript).expect("oracle proof");
        let proof_bytes = serialize_stage1(&proof);
        let event_bytes = serialize_events(transcript.events());
        let verifier = AkitaStage1Verifier::new(
            equality_point,
            DigitRangePlan::new(expected.basis).expect("oracle basis"),
        );
        let mut verifier_transcript =
            LoggingTranscript::wrap(AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL));
        let verified_point = verifier
            .verify(&proof, &mut verifier_transcript)
            .expect("oracle verification");

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
            oracle_digest(&proof_bytes),
            expected.proof_digest,
            "proof bytes changed for basis {}",
            expected.basis
        );
        assert_eq!(
            oracle_digest(&event_bytes),
            expected.event_digest,
            "transcript events changed for basis {}",
            expected.basis
        );
        assert_eq!(
            oracle_digest(&field_vector_bytes(&output_point)),
            expected.output_point_digest,
            "output point changed for basis {}",
            expected.basis
        );
        assert_eq!(
            canonical_hex(&proof.s_claim),
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
