//! Native Spongefish transcript: prover/verifier agreement and binding.
//!
//! A fuzzed operation sequence (public data, prover messages, challenges) is
//! executed by a prover and replayed by a verifier over the emitted argument
//! string. Oracles: every received value and challenge matches, EOF holds,
//! and any single-byte change to the argument string makes replay fail or
//! diverge. A different session, instance, or public message changes the
//! first challenge that follows it.

use crate::input::Reader;
use crate::{gen, stats};
use akita_config::proof_optimized::{fp32, fp64};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};

type F = akita_config::proof_optimized::fp128::Field;
type Ext4 = fp32::ExtensionField;
type Ext2 = fp64::ExtensionField;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("akita-fuzz/transcript");

#[derive(Clone, Debug, PartialEq)]
enum Op {
    PublicBytes(Vec<u8>),
    PublicFields(Vec<F>),
    SendField(F),
    SendExt4(Ext4),
    SendBytes(Vec<u8>),
    ChallengeField,
    ChallengeExt2,
}

#[derive(Debug, PartialEq)]
enum Seen {
    Field(F),
    Ext4(Ext4),
    Ext2(Ext2),
    Bytes(Vec<u8>),
}

fn ops(reader: &mut Reader<'_>) -> Vec<Op> {
    let count = 1 + usize::from(reader.u8() % 24);
    (0..count)
        .map(|_| match reader.u8() % 7 {
            0 => {
                let len = usize::from(reader.u8() % 80);
                Op::PublicBytes(reader.take(len).to_vec())
            }
            1 => {
                let len = usize::from(reader.u8() % 6);
                Op::PublicFields(
                    (0..len)
                        .map(|_| gen::scalar(reader, gen::Domain::Full))
                        .collect(),
                )
            }
            2 => Op::SendField(gen::scalar(reader, gen::Domain::Full)),
            3 => Op::SendExt4(gen::ext_scalar::<fp32::Field, Ext4>(reader)),
            4 => {
                let len = usize::from(reader.u8() % 80);
                Op::SendBytes(reader.take(len).to_vec())
            }
            5 => Op::ChallengeField,
            _ => Op::ChallengeExt2,
        })
        .collect()
}

fn new_prover(session: &[u8], instance: &[u8]) -> ProverTranscript<Blake2b512> {
    let mut state = ProverTranscript::new(&PROTOCOL, session);
    state.public_bytes(instance);
    state
}

fn new_verifier<'a>(
    session: &[u8],
    instance: &[u8],
    proof: &'a [u8],
) -> VerifierTranscript<'a, Blake2b512> {
    let mut state = VerifierTranscript::new(&PROTOCOL, session, proof);
    state.public_bytes(instance);
    state
}

fn prove(state: &mut ProverTranscript<Blake2b512>, ops: &[Op]) -> Vec<Seen> {
    let mut seen = Vec::new();
    for op in ops {
        match op {
            Op::PublicBytes(bytes) => state.public_bytes(bytes),
            Op::PublicFields(values) => state.public_all(values),
            Op::SendField(value) => state.send(value),
            Op::SendExt4(value) => state.send(value),
            Op::SendBytes(bytes) => state.send_bytes(bytes),
            Op::ChallengeField => seen.push(Seen::Field(state.challenge())),
            Op::ChallengeExt2 => seen.push(Seen::Ext2(state.challenge())),
        }
    }
    seen
}

/// Replay; `None` means the verifier rejected somewhere.
fn replay(
    mut state: VerifierTranscript<'_, Blake2b512>,
    ops: &[Op],
) -> Option<(Vec<Seen>, Vec<Seen>)> {
    let mut received = Vec::new();
    let mut challenges = Vec::new();
    for op in ops {
        match op {
            Op::PublicBytes(bytes) => state.public_bytes(bytes),
            Op::PublicFields(values) => state.public_all(values),
            Op::SendField(_) => received.push(Seen::Field(state.receive::<F>().ok()?)),
            Op::SendExt4(_) => received.push(Seen::Ext4(state.receive::<Ext4>().ok()?)),
            Op::SendBytes(bytes) => {
                received.push(Seen::Bytes(state.receive_bytes(bytes.len()).ok()?.to_vec()))
            }
            Op::ChallengeField => challenges.push(Seen::Field(state.challenge())),
            Op::ChallengeExt2 => challenges.push(Seen::Ext2(state.challenge())),
        }
    }
    state.finish().ok()?;
    Some((received, challenges))
}

fn sent(ops: &[Op]) -> Vec<Seen> {
    ops.iter()
        .filter_map(|op| match op {
            Op::SendField(value) => Some(Seen::Field(*value)),
            Op::SendExt4(value) => Some(Seen::Ext4(*value)),
            Op::SendBytes(bytes) => Some(Seen::Bytes(bytes.clone())),
            _ => None,
        })
        .collect()
}

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    let session_len = usize::from(reader.u8() % 40);
    let session = reader.take(session_len).to_vec();
    let instance_len = usize::from(reader.u8() % 40);
    let instance = reader.take(instance_len).to_vec();
    let ops = ops(&mut reader);

    let mut prover = new_prover(&session, &instance);
    let challenges = prove(&mut prover, &ops);
    let proof = prover.finish();
    let verifier = new_verifier(&session, &instance, &proof);
    let (received, replayed) = replay(verifier, &ops).expect("honest transcript must replay");
    assert_eq!(received, sent(&ops), "received prover messages");
    assert_eq!(replayed, challenges, "replayed challenges");
    stats::count("transcript_honest");

    if !proof.is_empty() {
        let mut tampered = proof.clone();
        let offset = reader.u32() as usize % tampered.len();
        tampered[offset] ^= reader.u8().max(1);
        let verifier = new_verifier(&session, &instance, &tampered);
        if let Some((received_t, replayed_t)) = replay(verifier, &ops) {
            assert!(
                received_t != received || replayed_t != replayed,
                "a modified argument string replayed to identical messages and challenges"
            );
        }
        stats::count("transcript_tampered");
    }

    let first_challenge = ops
        .iter()
        .position(|op| matches!(op, Op::ChallengeField | Op::ChallengeExt2));
    if first_challenge.is_some() && !challenges.is_empty() {
        let mut other_session = session.clone();
        other_session.push(reader.u8());
        let mut other = new_prover(&other_session, &instance);
        let other_challenges = prove(&mut other, &ops);
        assert_ne!(
            other_challenges[0], challenges[0],
            "session does not separate challenges"
        );
    }
}
