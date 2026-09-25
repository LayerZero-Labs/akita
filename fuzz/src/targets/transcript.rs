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
use akita_transcript::{
    native_prover_ext_challenge, native_verifier_ext_challenge, new_native_prover,
    new_native_verifier, public_native_bytes_prover, public_native_bytes_verifier,
    public_native_fields_prover, public_native_fields_verifier, receive_native_bytes,
    receive_native_extension, receive_native_field, send_native_bytes, send_native_extension,
    send_native_field, NativeProverState, NativeVerifierState, ProtocolSiteId,
};

type F = akita_config::proof_optimized::fp128::Field;
type Ext4 = fp32::ExtensionField;
type Ext2 = fp64::ExtensionField;

#[derive(Clone, Debug, PartialEq)]
enum Op {
    PublicBytes(ProtocolSiteId, Vec<u8>),
    PublicFields(ProtocolSiteId, Vec<F>),
    SendField(F),
    SendExt4(Ext4),
    SendBytes(Vec<u8>),
    ChallengeField(ProtocolSiteId),
    ChallengeExt2(ProtocolSiteId),
}

#[derive(Debug, PartialEq)]
enum Seen {
    Field(F),
    Ext4(Ext4),
    Ext2(Ext2),
    Bytes(Vec<u8>),
}

fn site(reader: &mut Reader<'_>) -> ProtocolSiteId {
    ProtocolSiteId {
        family: u32::from(reader.u8()),
        invocation: u32::from(reader.u8()),
        round: u32::from(reader.u8()),
        ..ProtocolSiteId::default()
    }
}

fn ops(reader: &mut Reader<'_>) -> Vec<Op> {
    let count = 1 + usize::from(reader.u8() % 24);
    (0..count)
        .map(|_| match reader.u8() % 7 {
            0 => {
                let site = site(reader);
                let len = usize::from(reader.u8() % 80);
                Op::PublicBytes(site, reader.take(len).to_vec())
            }
            1 => {
                let site = site(reader);
                let len = usize::from(reader.u8() % 6);
                Op::PublicFields(
                    site,
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
            5 => Op::ChallengeField(site(reader)),
            _ => Op::ChallengeExt2(site(reader)),
        })
        .collect()
}

fn challenge_context(state_site: ProtocolSiteId) -> akita_transcript::ProtocolContextRecord {
    akita_transcript::ProtocolContextRecord::new(
        state_site.to_bytes(),
        akita_transcript::ProtocolMessageKind::Challenge as u32,
        0,
        0,
        akita_transcript::native_field_challenge_bytes::<F>(),
    )
}

fn prove(state: &mut NativeProverState, ops: &[Op]) -> Vec<Seen> {
    let mut seen = Vec::new();
    for op in ops {
        match op {
            Op::PublicBytes(site, bytes) => {
                public_native_bytes_prover(state, *site, bytes).expect("bounded public bytes")
            }
            Op::PublicFields(site, values) => {
                public_native_fields_prover(state, *site, values).expect("public fields")
            }
            Op::SendField(value) => send_native_field(state, *value),
            Op::SendExt4(value) => send_native_extension::<fp32::Field, Ext4>(state, *value),
            Op::SendBytes(bytes) => send_native_bytes(state, bytes),
            Op::ChallengeField(site) => {
                akita_transcript::prover_context(state, challenge_context(*site));
                seen.push(Seen::Field(
                    akita_transcript::native_prover_field_challenge(state).expect("challenge"),
                ));
            }
            Op::ChallengeExt2(site) => seen.push(Seen::Ext2(
                native_prover_ext_challenge::<fp64::Field, Ext2>(state, *site).expect("challenge"),
            )),
        }
    }
    seen
}

/// Replay; `None` means the verifier rejected somewhere.
fn replay(mut state: NativeVerifierState<'_>, ops: &[Op]) -> Option<(Vec<Seen>, Vec<Seen>)> {
    let mut received = Vec::new();
    let mut challenges = Vec::new();
    for op in ops {
        match op {
            Op::PublicBytes(site, bytes) => {
                public_native_bytes_verifier(&mut state, *site, bytes).ok()?
            }
            Op::PublicFields(site, values) => {
                public_native_fields_verifier(&mut state, *site, values).ok()?
            }
            Op::SendField(_) => {
                received.push(Seen::Field(receive_native_field::<F>(&mut state).ok()?))
            }
            Op::SendExt4(_) => received.push(Seen::Ext4(
                receive_native_extension::<fp32::Field, Ext4>(&mut state).ok()?,
            )),
            Op::SendBytes(bytes) => received.push(Seen::Bytes(
                receive_native_bytes(&mut state, bytes.len()).ok()?,
            )),
            Op::ChallengeField(site) => {
                akita_transcript::verifier_context(&mut state, challenge_context(*site));
                challenges.push(Seen::Field(
                    akita_transcript::native_verifier_field_challenge(&mut state).ok()?,
                ));
            }
            Op::ChallengeExt2(site) => challenges.push(Seen::Ext2(
                native_verifier_ext_challenge::<fp64::Field, Ext2>(&mut state, *site).ok()?,
            )),
        }
    }
    state.check_eof().ok()?;
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

    let Ok(mut prover) = new_native_prover(&session, &instance) else {
        return;
    };
    let challenges = prove(&mut prover, &ops);
    let proof = prover.narg_string().to_vec();
    let verifier =
        new_native_verifier(&session, &instance, &proof).expect("prover accepted this domain");
    let (received, replayed) = replay(verifier, &ops).expect("honest transcript must replay");
    assert_eq!(received, sent(&ops), "received prover messages");
    assert_eq!(replayed, challenges, "replayed challenges");
    stats::count("transcript_honest");

    if !proof.is_empty() {
        let mut tampered = proof.clone();
        let offset = reader.u32() as usize % tampered.len();
        tampered[offset] ^= reader.u8().max(1);
        let verifier = new_native_verifier(&session, &instance, &tampered).expect("same domain");
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
        .position(|op| matches!(op, Op::ChallengeField(_) | Op::ChallengeExt2(_)));
    if first_challenge.is_some() && !challenges.is_empty() {
        let mut other_session = session.clone();
        other_session.push(reader.u8());
        if let Ok(mut other) = new_native_prover(&other_session, &instance) {
            let other_challenges = prove(&mut other, &ops);
            assert_ne!(
                other_challenges[0], challenges[0],
                "session does not separate challenges"
            );
        }
    }
}
