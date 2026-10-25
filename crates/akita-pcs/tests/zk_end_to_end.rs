//! Zero knowledge end to end (Whiteout W5d, `specs/whiteout-akita.md` §5.5).
//!
//! A synthetic statement with masked claims: committed groups opened at a
//! point, each claim sent masked before Akita, as Jolt's stage-8 claims are.
//! One transcript carries, in order:
//!
//! 1. the lattice pad commitment `t_p`, compressed by the response prover;
//! 2. the masked claims;
//! 3. Akita's masked prefix (levels `0..=cut_level`);
//! 4. the masked auxiliary block;
//! 5. jolt-zk's Σ-protocol over Akita's residual rows and `ResidualClaim`,
//!    whose response statement [`AkitaResponseProver`] proves.
//!
//! The pads are committed by jolt-zk's `LatticePadCommitment`, and the same
//! pads mask the messages. jolt-zk builds a `CommittedPads` bound to that
//! commitment only through `ZkBackend` (W6), so the masking channel here is
//! `CommittedPads::uncommitted` over the committed pads and the verifier's
//! view is `CommittedPadsView::counting()`, checked to end at the layout's
//! pad count.

#![allow(missing_docs)]

mod common;
#[path = "zk_end_to_end/harness.rs"]
mod harness;
#[path = "masked_prefix/instance.rs"]
mod instance;
#[path = "dynamic_backends/private_cpu.rs"]
mod private_cpu;

use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::zk::ZK_CUT_LEVEL;
use akita_params::ProtocolSiteId;
use akita_pcs::zk::RESPONSE_SITE;
use akita_types::{AkitaSponge, PROOF_STREAM_PROTOCOL};
use harness::*;
use instance::*;
use jolt_algebra::{CanonicalBytes, CanonicalDecode, Field, One};
use jolt_transcript::{
    Channel, ProverTranscript, SiteId, TranscriptEvent, TranscriptOp, VerifierTranscript,
    ATTEMPT_NONCE_LEN,
};
use jolt_zk::masking::{CommittedPads, ZkError};
use jolt_zk::whiteout::WhiteoutError;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashSet;
use std::ops::Range;

/// What a received byte is to the verifier (HT1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// A masked slot, m̂ = m + p.
    Masked(MaskedKind),
    /// An image hidden by its own fresh randomness: `t_p`, a D image, a
    /// hidden next-witness commitment (u_a at the cut).
    RandomizedCommitment,
    /// Public, a function of the setup and of challenges: proof-of-work
    /// nonces, Stage 3 (claim, rounds, setup-prefix evaluation).
    PublicChallengeDerived,
    /// `cm(t_D)` and the attempt nonce.
    Sigma,
    /// The response proof.
    Response,
}

/// The masked message classes of the synthetic statement and the prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MaskedKind {
    Claim,
    Stage1Round,
    RangeImage,
    Stage2Round,
    WitnessEval,
    Auxiliary,
}

/// Where the verifier is in the proof, from the harness's site labels.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Start,
    PadCommitment,
    Statement,
    Sigma,
    Response,
}

/// One received message: its bytes, class, and pads if masked.
#[derive(Clone, Debug)]
struct Received {
    bytes: Range<usize>,
    class: Class,
    pads: Option<Range<usize>>,
}

/// The allowlist: the class every site of an honest zero-knowledge proof
/// may carry, and whether it must arrive masked. A site outside it, a masked
/// message at a clear site, or a private message without its masking is an
/// error naming the message.
fn class_of(
    phase: Phase,
    site: SiteId,
    masked: bool,
    grinding: &[ProtocolSiteId],
) -> Result<Class, String> {
    use akita_params::transcript_site::*;
    use akita_params::SumcheckProtocol;
    let class = match phase {
        Phase::Start => return Err("a message before the pad commitment".into()),
        Phase::PadCommitment => Class::RandomizedCommitment,
        Phase::Sigma => Class::Sigma,
        Phase::Response => Class::Response,
        Phase::Statement if site == CLAIMS_SITE => Class::Masked(MaskedKind::Claim),
        Phase::Statement if site == AUXILIARY_SITE => Class::Masked(MaskedKind::Auxiliary),
        Phase::Statement => {
            let id = ProtocolSiteId::from_bytes(site.0);
            match id.family {
                SITE_FAMILY_SUMCHECK if id.invocation == SumcheckProtocol::Stage1.tag() => {
                    Class::Masked(MaskedKind::Stage1Round)
                }
                SITE_FAMILY_SUMCHECK if id.invocation == SumcheckProtocol::Stage2.tag() => {
                    Class::Masked(MaskedKind::Stage2Round)
                }
                SITE_FAMILY_SUMCHECK if id.invocation == SumcheckProtocol::Stage3.tag() => {
                    Class::PublicChallengeDerived
                }
                SITE_FAMILY_STAGE1 => Class::Masked(MaskedKind::RangeImage),
                SITE_FAMILY_STAGE2 => Class::Masked(MaskedKind::WitnessEval),
                SITE_FAMILY_STAGE3 => Class::PublicChallengeDerived,
                SITE_FAMILY_OPENING_PAYLOAD | SITE_FAMILY_NEXT_WITNESS => {
                    Class::RandomizedCommitment
                }
                // Proof-of-work nonces: exactly the schedule's grinding
                // sites (`GrindingSite::site_id`; the detail carries widths).
                _ if grinding.contains(&ProtocolSiteId { detail: 0, ..id }) => {
                    Class::PublicChallengeDerived
                }
                _ => return Err(format!("site {id:?} is outside the allowlist")),
            }
        }
    };
    match (class, masked) {
        (Class::Masked(_), true) => Ok(class),
        (Class::Masked(kind), false) => Err(format!("{kind:?} message reached a clear channel")),
        (_, true) => Err(format!("{class:?} message arrived masked")),
        (_, false) => Ok(class),
    }
}

/// HT1's classifier: every received byte of `proof_len` in exactly one
/// class, in order, from the verifier's event log.
fn classify(
    events: &[TranscriptEvent],
    proof_len: usize,
    grinding: &[ProtocolSiteId],
) -> Result<Vec<Received>, String> {
    let mut phase = Phase::Start;
    let mut received = Vec::new();
    let mut cursor = 0;
    for (index, event) in events.iter().enumerate() {
        phase = match event.site {
            site if site == PAD_SITE => Phase::PadCommitment,
            site if site == CLAIMS_SITE && phase == Phase::PadCommitment => Phase::Statement,
            site if site == SIGMA_SITE => Phase::Sigma,
            site if site == RESPONSE_SITE => Phase::Response,
            _ => phase,
        };
        if event.op != TranscriptOp::Message {
            continue;
        }
        let bytes = event.narg.clone().ok_or("a message without bytes")?;
        if bytes.start != cursor {
            return Err(format!(
                "bytes {cursor}..{} belong to no message",
                bytes.start
            ));
        }
        cursor = bytes.end;
        let pads = match events.get(index + 1) {
            Some(TranscriptEvent {
                op: TranscriptOp::Masked { first_pad },
                len,
                ..
            }) => Some(*first_pad..first_pad + len),
            _ => None,
        };
        let class = class_of(phase, event.site, pads.is_some(), grinding)
            .map_err(|error| format!("bytes {bytes:?}: {error}"))?;
        received.push(Received { bytes, class, pads });
    }
    if cursor != proof_len {
        return Err(format!("bytes {cursor}..{proof_len} belong to no message"));
    }
    Ok(received)
}

/// The masked messages' pad ranges, in order.
fn masked_pads(received: &[Received]) -> Vec<Range<usize>> {
    received
        .iter()
        .filter_map(|message| message.pads.clone())
        .collect()
}

/// The field elements of every masked message.
fn masked_values(proof: &[u8], received: &[Received]) -> Vec<F> {
    received
        .iter()
        .filter(|message| message.pads.is_some())
        .flat_map(|message| field_elements(&proof[message.bytes.clone()]))
        .collect()
}

fn field_elements(bytes: &[u8]) -> Vec<F> {
    bytes
        .chunks_exact(<F as CanonicalBytes>::NUM_BYTES)
        .map(|chunk| F::from_bytes_le_checked(chunk).unwrap())
        .collect()
}

/// The `WhiteoutError` inside a failure, if any.
fn whiteout(failure: &Failure) -> Option<&WhiteoutError> {
    match failure {
        Failure::Zk(ZkError::Whiteout(source)) => source.downcast_ref(),
        _ => None,
    }
}

/// Honest proofs of both statement shapes verify end to end; a flipped Σ
/// byte and a flipped response-proof byte are rejected.
#[test]
fn honest_proofs_verify_end_to_end() {
    on_large_stack(|| {
        fn check<C: CommitmentConfig<Field = F, ExtField = F>>(harness: &Harness<C>, seed: u64) {
            let proved = harness.prove(seed, Cheat::default()).unwrap();
            let events = harness.verify(&proved.proof).unwrap();
            let received =
                classify(&events, proved.proof.len(), &harness.grinding_sites()).unwrap();
            for class in [Class::Sigma, Class::Response] {
                let message = received.iter().find(|m| m.class == class).unwrap();
                let mut proof = proved.proof.clone();
                proof[message.bytes.start] ^= 1;
                assert!(
                    harness.verify(&proof).is_err(),
                    "{class:?} byte flip accepted"
                );
            }
        }
        check(&single(), 1);
        check(&multi(), 2);
    });
}

/// HT1, the public-site allowlist: every byte the verifier receives is in
/// exactly one class (masked slot, randomized commitment, public
/// challenge-derived data, Σ message, response proof), every masked class
/// arrives masked, and nothing else does. Both statement shapes; the
/// multi-group one runs Stage 3 at level 0 in the clear.
///
/// Positive control: the same verifier with the statement claims routed
/// through the clear transcript instead of the masked channel is flagged.
#[test]
fn every_received_byte_is_in_one_public_class() {
    on_large_stack(|| {
        fn check<C: CommitmentConfig<Field = F, ExtField = F>>(harness: &Harness<C>, seed: u64) {
            let proved = harness.prove(seed, Cheat::default()).unwrap();
            let events = harness.verify(&proved.proof).unwrap();
            let received =
                classify(&events, proved.proof.len(), &harness.grinding_sites()).unwrap();
            let kinds = received
                .iter()
                .filter_map(|message| match message.class {
                    Class::Masked(kind) => Some(kind),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for kind in [
                MaskedKind::Claim,
                MaskedKind::Stage1Round,
                MaskedKind::RangeImage,
                MaskedKind::Stage2Round,
                MaskedKind::WitnessEval,
                MaskedKind::Auxiliary,
            ] {
                assert!(kinds.contains(&kind), "no {kind:?} message");
            }
            for class in [
                Class::RandomizedCommitment,
                Class::PublicChallengeDerived,
                Class::Sigma,
                Class::Response,
            ] {
                assert!(
                    received.iter().any(|message| message.class == class),
                    "no {class:?} message"
                );
            }

            // Positive control: one private site routed clear.
            let events = harness.verify_with_clear_claims(&proved.proof);
            let error =
                classify(&events, proved.proof.len(), &harness.grinding_sites()).unwrap_err();
            assert!(
                error.contains("Claim message reached a clear channel"),
                "{error}"
            );
        }
        check(&single(), 3);
        let multi = multi();
        check(&multi, 4);
        let proved = multi.prove(5, Cheat::default()).unwrap();
        let events = multi.verify(&proved.proof).unwrap();
        let stage3 = events
            .iter()
            .filter(|event| {
                event.op == TranscriptOp::Message
                    && akita_params::ProtocolSiteId::from_bytes(event.site.0).family
                        == akita_params::transcript_site::SITE_FAMILY_STAGE3
            })
            .count();
        assert!(stage3 > 0, "the multi-group row runs no Stage 3");
    });
}

/// Coverage (W3-b) over the full synthetic prefix: with the auxiliary values
/// recomputed consistently, shifting any one message slot by one breaks a
/// residual row, and so does shifting any auxiliary slot alone; so every pad
/// slot is constrained. Both statement shapes.
#[test]
fn shifting_any_slot_breaks_a_row() {
    on_large_stack(|| {
        fn check<C: CommitmentConfig<Field = F, ExtField = F>>(harness: &Harness<C>, seed: u64) {
            let proved = harness.prove(seed, Cheat::default()).unwrap();
            let mut transcript = VerifierTranscript::<AkitaSponge>::new(
                &PROOF_STREAM_PROTOCOL,
                SESSION,
                &proved.proof,
            );
            let Replayed {
                prefix,
                masked_auxiliary,
                relation,
                ..
            } = harness
                .replay(&mut transcript, proved.pad_image.len())
                .unwrap();
            let witness = [proved.residual.witness_eval];
            let pads = &proved.pads;
            assert_eq!(relation.first_violation(pads, &witness).unwrap(), None);
            for slot in 0..harness.layout.num_pads {
                let mut shifted = pads.clone();
                shifted[slot] -= F::one();
                if slot < harness.layout.auxiliary.start {
                    consistent_auxiliary_pads(
                        &prefix,
                        &harness.layout,
                        &masked_auxiliary,
                        &mut shifted,
                    );
                }
                assert!(
                    relation
                        .first_violation(&shifted, &witness)
                        .unwrap()
                        .is_some(),
                    "shifting slot {slot} breaks no row"
                );
            }
        }
        check(&single(), 6);
        check(&multi(), 7);
    });
}

/// HT3, pad uniqueness and freshness:
///
/// - the masked messages take every pad exactly once, in order, and end at
///   the layout's pad count, on both roles' event logs;
/// - two proofs of one witness with different seeds (fresh root commitments,
///   pads, hiding randomness and Σ coins) share no pad, masked value,
///   commitment image element, randomness vector or Σ nonce.
#[test]
fn pads_are_used_once_and_never_reused() {
    on_large_stack(|| {
        let first = single_seeded(8);
        let second = single_seeded(9);
        let proofs = [
            (&first, first.prove(10, Cheat::default()).unwrap()),
            (&second, second.prove(11, Cheat::default()).unwrap()),
        ];
        let mut seen = Vec::new();
        for (harness, proved) in &proofs {
            let events = harness.verify(&proved.proof).unwrap();
            let received =
                classify(&events, proved.proof.len(), &harness.grinding_sites()).unwrap();
            let mut next = 0;
            for pads in masked_pads(&received) {
                assert_eq!(pads.start, next, "pads {pads:?} out of order");
                next = pads.end;
            }
            assert_eq!(next, harness.layout.num_pads);
            seen.push(received);
        }
        let [(_, a), (_, b)] = &proofs;
        let disjoint = |x: &[F], y: &[F], what: &str| {
            let x = x.iter().collect::<HashSet<_>>();
            assert!(y.iter().all(|value| !x.contains(value)), "{what} reused");
        };
        disjoint(&a.pads, &b.pads, "a pad");
        disjoint(
            &masked_values(&a.proof, &seen[0]),
            &masked_values(&b.proof, &seen[1]),
            "a masked value",
        );
        let images = |proved: &ZkProof, received: &[Received]| {
            received
                .iter()
                .filter(|message| message.class == Class::RandomizedCommitment)
                .flat_map(|message| field_elements(&proved.proof[message.bytes.clone()]))
                .collect::<Vec<_>>()
        };
        disjoint(
            &images(a, &seen[0]),
            &images(b, &seen[1]),
            "a commitment image element",
        );
        disjoint(&a.pad_image, &b.pad_image, "a pad image element");
        assert_ne!(a.residual.r_a, b.residual.r_a, "r_a reused");
        assert_ne!(a.pad_randomness, b.pad_randomness, "r_p reused");
        let nonce = |proved: &ZkProof, received: &[Received]| {
            received
                .iter()
                .filter(|message| message.class == Class::Sigma)
                .find(|message| message.bytes.len() == ATTEMPT_NONCE_LEN)
                .map(|message| proved.proof[message.bytes.clone()].to_vec())
                .unwrap()
        };
        assert_ne!(nonce(a, &seen[0]), nonce(b, &seen[1]), "Σ nonce reused");
        assert_ne!(
            format!("{:?}", first.instance.groups[0].commitment),
            format!("{:?}", second.instance.groups[0].commitment),
            "root commitment reused"
        );
    });
}

/// HT3, a forced fold overflow: the backend rejects the one-shot fold
/// response at a masked level, the prover returns `FoldOverflow` after that
/// one probe (W2-b), and the
/// rerun draws everything fresh. The rerun shares no pad, masked value or
/// randomized image with the aborted attempt's transcript. (The root
/// commitment is the statement and is the same in both.)
#[test]
fn a_fold_overflow_rerun_reuses_nothing() {
    on_large_stack(|| {
        use akita_cpu_backend::{CpuBackend, GroupContext};
        use akita_prover::{prove_masked_prefix, SelectedProverOpeningData};
        use akita_types::{OpeningClaims, PolynomialGroupClaims};
        use private_cpu::PrivateCpu;

        let harness = single();
        let instance = &harness.instance;
        let private = PrivateCpu::new(
            CpuBackend::new(instance.setup.expanded.clone())
                .unwrap()
                .with_hiding_rng(StdRng::seed_from_u64(12)),
        );
        let group = &instance.groups[0];
        let (_, poly) = dense_poly(NV);
        let (commitment, handle) = private
            .commit_in(
                instance.scheme.schedules(),
                vec![poly],
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .unwrap();
        let opening_data = || {
            SelectedProverOpeningData::from_committed_claims::<Cfg>(
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    group.point.clone(),
                    vec![group.opening],
                    commitment.clone(),
                )
                .unwrap()])
                .unwrap(),
                vec![handle.clone()],
                instance.scheme.schedules(),
            )
            .unwrap()
        };
        let resolved = instance
            .scheme
            .schedules()
            .resolve_selection(opening_data().selection())
            .unwrap();
        let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening_data().opening_layout(),
        )
        .unwrap();
        let prefixes = private.prefixes(&instance.setup, &ids);
        let mut rng = StdRng::seed_from_u64(13);
        // One attempt with fresh pads: its pads, its transcript, and the
        // prover's event log.
        let mut attempt = |reject: Option<usize>| {
            let pads: Vec<F> = (0..harness.layout.num_pads)
                .map(|_| F::random(&mut rng))
                .collect();
            let mut masking =
                CommittedPads::uncommitted(harness.layout.clone(), pads.clone()).unwrap();
            let mut transcript =
                ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
            transcript.site(PAD_SITE);
            transcript.site(CLAIMS_SITE);
            let claims = masking.send(&[group.opening], &mut transcript).unwrap();
            let first_probe = private.fold_probes.get();
            private
                .reject_fold_probe
                .set(reject.map(|level| first_probe + level));
            let result = prove_masked_prefix::<Cfg, _, _, _>(
                instance.setup.expanded.descriptor(),
                instance.scheme.schedules(),
                &private,
                &prefixes,
                opening_data(),
                &own_pads(claims),
                &mut masking,
                &mut transcript,
                harness.response(),
                BASIS,
            );
            private.reject_fold_probe.set(None);
            (
                result.map(|_| ()),
                private.fold_probes.get() - first_probe,
                pads,
                transcript.narg().to_vec(),
                transcript.events().to_vec(),
            )
        };
        for level in 0..=ZK_CUT_LEVEL {
            let (aborted, probes, aborted_pads, aborted_bytes, aborted_events) =
                attempt(Some(level as usize));
            assert_eq!(aborted, Err(AkitaError::FoldOverflow { level }));
            // One-shot (W2-b): the rejected response is that level's only
            // probe, and the proof stops there.
            assert_eq!(probes, level as usize + 1);
            let (rerun, _, rerun_pads, rerun_bytes, rerun_events) = attempt(None);
            rerun.unwrap();
            let values = |bytes: &[u8], events: &[TranscriptEvent], masked: bool| {
                events
                    .iter()
                    .enumerate()
                    .filter(|(_, event)| event.op == TranscriptOp::Message)
                    .filter(|(index, _)| {
                        matches!(
                            events.get(index + 1).map(|event| event.op),
                            Some(TranscriptOp::Masked { .. })
                        ) == masked
                    })
                    .filter_map(|(_, event)| event.narg.clone())
                    .filter(|range| range.len() >= 16)
                    .flat_map(|range| field_elements(&bytes[range]))
                    .collect::<HashSet<_>>()
            };
            let reused = |a: &HashSet<F>, b: &HashSet<F>, what: &str| {
                assert!(a.is_disjoint(b), "level {level}: the rerun reuses a {what}");
            };
            reused(
                &aborted_pads.iter().copied().collect(),
                &rerun_pads.iter().copied().collect(),
                "pad",
            );
            reused(
                &values(&aborted_bytes, &aborted_events, true),
                &values(&rerun_bytes, &rerun_events, true),
                "masked value",
            );
            // The clear field messages before the abort are the masked
            // levels' randomized images (D images, next-witness
            // commitments) and Stage-3 values, which this row has none of.
            let images = values(&aborted_bytes, &aborted_events, false);
            assert!(!images.is_empty());
            reused(
                &images,
                &values(&rerun_bytes, &rerun_events, false),
                "randomized image element",
            );
        }
    });
}

/// HT4, a consistent cheat for each masked message class: the prover
/// commits pads that decode one message to one more than its true value and
/// recomputes the auxiliary products to follow it. The Σ prover's per-row
/// check refuses; with it skipped the response prover's statement check
/// refuses; with both skipped (`UncheckedResponseProver`, whose rounds stay
/// consistent) the prover sends a forged response proof for the false
/// statement over Akita's real rows, and `verify_sigma` rejects it. The
/// honest proof with the masked value shifted in its bytes is rejected too,
/// though that alone is only Fiat–Shamir binding.
#[test]
fn a_consistent_cheat_fails_at_every_layer() {
    on_large_stack(|| {
        let harness = single();
        let honest = harness.prove(14, Cheat::default()).unwrap();
        let events = harness.verify(&honest.proof).unwrap();
        let received = classify(&events, honest.proof.len(), &harness.grinding_sites()).unwrap();
        for kind in [
            MaskedKind::Claim,
            MaskedKind::Stage1Round,
            MaskedKind::RangeImage,
            MaskedKind::Stage2Round,
            MaskedKind::WitnessEval,
        ] {
            let message = received
                .iter()
                .find(|message| message.class == Class::Masked(kind))
                .unwrap();
            let slot = message.pads.clone().unwrap().start;
            // The same shift seen only by the verifier: the masked value
            // plus one in the honest proof's bytes.
            let mut proof = honest.proof.clone();
            let element = message.bytes.start..message.bytes.start + 16;
            let shifted = field_elements(&proof[element.clone()])[0] + F::one();
            shifted.to_bytes_le(&mut proof[element]);
            assert!(
                harness.verify(&proof).is_err(),
                "{kind:?}: a shifted masked value verifies"
            );
            let cheat = Cheat {
                shifted_slot: Some(slot),
                ..Cheat::default()
            };
            let error = harness.prove(15, cheat).err().unwrap();
            assert!(
                matches!(
                    whiteout(&error),
                    Some(WhiteoutError::ResidualUnsatisfied { .. })
                ),
                "{kind:?}: {error:?}"
            );
            let error = harness
                .prove(
                    15,
                    Cheat {
                        unchecked_sigma: true,
                        ..cheat
                    },
                )
                .err()
                .unwrap();
            assert!(
                matches!(whiteout(&error), Some(WhiteoutError::ResponseRejected(_))),
                "{kind:?}: {error:?}"
            );
            let forged = harness
                .prove(
                    15,
                    Cheat {
                        unchecked_sigma: true,
                        unchecked_response: true,
                        ..cheat
                    },
                )
                .unwrap();
            let error = harness.verify(&forged.proof).err().unwrap();
            assert!(
                matches!(whiteout(&error), Some(WhiteoutError::ResponseRejected(_))),
                "{kind:?}: {error:?}"
            );
        }
    });
}

/// A47, Item 3 reads u_a's target from `ξ_a`: a prover whose `cm(u_a)`
/// honestly compresses an image the residual does not open, stated
/// consistently as the Σ target. Its chain rows hold; Item 3 fails. The Σ
/// prover refuses, the response prover refuses, and the verifier rejects a
/// forged response proof. The same unchecked path over the honest
/// `cm(u_a)` verifies, so the rejection is u_a's compressed target, not the
/// unchecked provers.
#[test]
fn item_3_reads_u_a_from_its_compression_witness() {
    on_large_stack(|| {
        let harness = single();
        let unchecked = Cheat {
            unchecked_sigma: true,
            unchecked_response: true,
            ..Cheat::default()
        };
        let control = harness.prove(16, unchecked).unwrap();
        harness.verify(&control.proof).unwrap();
        let cheat = Cheat {
            shifted_residual: true,
            ..Cheat::default()
        };
        let error = harness.prove(16, cheat).err().unwrap();
        assert!(
            matches!(
                whiteout(&error),
                Some(WhiteoutError::ResidualOpeningMismatch)
            ),
            "{error:?}"
        );
        let error = harness
            .prove(
                16,
                Cheat {
                    unchecked_sigma: true,
                    ..cheat
                },
            )
            .err()
            .unwrap();
        assert!(
            matches!(whiteout(&error), Some(WhiteoutError::ResponseRejected(_))),
            "{error:?}"
        );
        let forged = harness
            .prove(
                16,
                Cheat {
                    shifted_residual: true,
                    ..unchecked
                },
            )
            .unwrap();
        let error = harness.verify(&forged.proof).err().unwrap();
        assert!(
            matches!(whiteout(&error), Some(WhiteoutError::ResponseRejected(_))),
            "{error:?}"
        );
    });
}

/// A47, u_a's chain rows: the cut sends `cm(u_a)` with its last
/// (192-coefficient) chunk's payload moved, and `e`'s `ξ_a` is built from
/// that compression, so u_a, `ξ_a` and Item 3 are honest and only that
/// chunk's `K₂` row fails. Through the unchecked provers the verifier
/// rejects at the fused sumcheck, where the chain rows are batched.
#[test]
fn a_forged_chain_on_u_a_is_rejected_at_the_fused_sumcheck() {
    on_large_stack(|| {
        let harness = single();
        let forged = harness
            .prove(
                17,
                Cheat {
                    forged_residual_chain: true,
                    unchecked_sigma: true,
                    unchecked_response: true,
                    ..Cheat::default()
                },
            )
            .unwrap();
        let error = harness.verify(&forged.proof).err().unwrap();
        assert_eq!(
            whiteout(&error),
            Some(&WhiteoutError::ResponseRejected("fused sumcheck")),
            "{error:?}"
        );
    });
}
