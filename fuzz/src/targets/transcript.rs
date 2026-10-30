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
    exchange_extension_group, ext_challenge, new_prover_channel, new_verifier_channel,
    public_bytes, public_fields, receive_byte_group, receive_field_group, send_byte_group,
    send_field_group, ProtocolSiteId, ProverChannel, VerifierChannel,
};
use jolt_field::Zero;

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

fn prove(state: &mut ProverChannel, ops: &[Op]) -> Vec<Seen> {
    let mut seen = Vec::new();
    for op in ops {
        match op {
            Op::PublicBytes(site, bytes) => {
                public_bytes(state, *site, bytes).expect("bounded public bytes")
            }
            Op::PublicFields(site, values) => {
                public_fields(state, *site, values).expect("public fields")
            }
            Op::SendField(value) => {
                send_field_group(state, ProtocolSiteId::default(), &[*value]).expect("field group")
            }
            Op::SendExt4(value) => exchange_extension_group::<fp32::Field, Ext4, _>(
                state,
                ProtocolSiteId::default(),
                &mut [*value],
            )
            .expect("extension group"),
            Op::SendBytes(bytes) => {
                send_byte_group(state, ProtocolSiteId::default(), bytes).expect("byte group")
            }
            Op::ChallengeField(site) => seen.push(Seen::Field(
                ext_challenge::<F, _>(state, *site).expect("challenge"),
            )),
            Op::ChallengeExt2(site) => seen.push(Seen::Ext2(
                ext_challenge::<Ext2, _>(state, *site).expect("challenge"),
            )),
        }
    }
    seen
}

/// Replay; `None` means the verifier rejected somewhere.
fn replay(mut state: VerifierChannel<'_>, ops: &[Op]) -> Option<(Vec<Seen>, Vec<Seen>)> {
    let mut received = Vec::new();
    let mut challenges = Vec::new();
    for op in ops {
        match op {
            Op::PublicBytes(site, bytes) => public_bytes(&mut state, *site, bytes).ok()?,
            Op::PublicFields(site, values) => public_fields(&mut state, *site, values).ok()?,
            Op::SendField(_) => {
                let values =
                    receive_field_group::<F>(&mut state, ProtocolSiteId::default(), 1).ok()?;
                received.push(Seen::Field(*values.first()?));
            }
            Op::SendExt4(_) => {
                let mut values = [Ext4::zero()];
                exchange_extension_group::<fp32::Field, _, _>(
                    &mut state,
                    ProtocolSiteId::default(),
                    &mut values,
                )
                .ok()?;
                received.push(Seen::Ext4(values[0]));
            }
            Op::SendBytes(bytes) => received.push(Seen::Bytes(
                receive_byte_group(&mut state, ProtocolSiteId::default(), bytes.len()).ok()?,
            )),
            Op::ChallengeField(site) => {
                challenges.push(Seen::Field(ext_challenge::<F, _>(&mut state, *site).ok()?))
            }
            Op::ChallengeExt2(site) => challenges.push(Seen::Ext2(
                ext_challenge::<Ext2, _>(&mut state, *site).ok()?,
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

    let mut prover = new_prover_channel(&session, &instance);
    let challenges = prove(&mut prover, &ops);
    let proof = prover.narg_string().to_vec();
    let verifier = new_verifier_channel(&session, &instance, &proof);
    let (received, replayed) = replay(verifier, &ops).expect("honest transcript must replay");
    assert_eq!(received, sent(&ops), "received prover messages");
    assert_eq!(replayed, challenges, "replayed challenges");
    stats::count("transcript_honest");

    if !proof.is_empty() {
        let mut tampered = proof.clone();
        let offset = reader.u32() as usize % tampered.len();
        tampered[offset] ^= reader.u8().max(1);
        let verifier = new_verifier_channel(&session, &instance, &tampered);
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
        let mut other = new_prover_channel(&other_session, &instance);
        let other_challenges = prove(&mut other, &ops);
        assert_ne!(
            other_challenges[0], challenges[0],
            "session does not separate challenges"
        );
    }
}
