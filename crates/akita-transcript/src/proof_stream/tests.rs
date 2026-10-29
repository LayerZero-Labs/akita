use super::*;
use jolt_field::{CanonicalBytes, FpExt4, Prime32Offset99 as F, Ring, Zero};

type E = FpExt4<F>;

fn site(family: u32, stage: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family,
        stage,
        ..ProtocolSiteId::default()
    }
}

#[test]
fn site_bytes_round_trip() {
    let site = ProtocolSiteId {
        family: 1,
        invocation: 2,
        level: 3,
        stage: 4,
        round: 5,
        group: 6,
        limb: 7,
        detail: u32::MAX,
    };
    let encoded = site.to_bytes();
    assert_eq!(encoded[..4], 1u32.to_le_bytes());
    assert_eq!(encoded[28..], u32::MAX.to_le_bytes());
    assert_eq!(ProtocolSiteId::from_bytes(encoded), site);
}

#[test]
fn record_size_overflow_is_rejected_before_receipt() {
    let site = ProtocolSiteId::default();
    let kind = ProtocolMessageKind::ProofAtoms;
    assert!(matches!(
        group_record(site, kind, usize::MAX.checked_mul(4), 4),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        group_record(site, kind, Some(usize::MAX), 4),
        Err(AkitaError::InvalidInput(_))
    ));
    let mut verifier = new_verifier_channel(b"count-overflow", &[]);
    assert!(matches!(
        receive_field_group::<F>(&mut verifier, site, usize::MAX),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(verifier.check_eof().is_err());
}

/// A prover that absorbs `absorbed` as one public message, for comparing the
/// sponge state a helper leaves against a hand-built absorbed layout.
fn reference_prover(session: &[u8], absorbed: &[u8]) -> ProverChannel {
    let mut reference = new_prover_channel(session);
    reference.public_message_as(absorbed, |bytes| bytes);
    reference
}

#[test]
fn rejected_verifier_draws_no_challenge_message() {
    let copy = |message: &[u8]| Ok(message.to_vec());
    let mut verifier = new_verifier_channel(b"fold-message", &[]);
    let message = verifier.challenge_message(40, copy).unwrap();
    assert_eq!(message.len(), 40);
    let _ = reject(&mut verifier);
    assert!(matches!(
        verifier.challenge_message(40, copy),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn failed_challenge_message_decode_poisons_the_verifier() {
    let mut verifier = new_verifier_channel(b"fold-message", &[]);
    let result = verifier.challenge_message(40, |_| {
        Err::<(), _>(AkitaError::InvalidInput("pool ran out".into()))
    });
    assert!(matches!(result, Err(AkitaError::InvalidProof)));
    assert!(verifier.check_eof().is_err());
}

#[test]
fn public_absorption_never_extends_the_argument_string() {
    let mut prover = new_prover_channel(b"public-size");
    crate::ProverTranscript::send(&mut prover, &[7u8; 3]);
    let before = prover.narg_string().to_vec();
    public_bytes(&mut prover, site(1, 0), b"public fold payload").unwrap();
    assert_eq!(prover.narg_string(), before);
}

#[test]
fn session_and_statement_are_framed_separately() {
    let mut left = new_prover_channel(b"ab");
    bind_instance(&mut left, b"c").unwrap();
    let mut right = new_prover_channel(b"a");
    bind_instance(&mut right, b"bc").unwrap();
    assert_ne!(
        left.verifier_message::<[u8; 32]>(),
        right.verifier_message::<[u8; 32]>()
    );
}

#[test]
fn tags_are_the_padded_protocol_names() {
    let digest = match jolt_transcript::DIGEST_HASH {
        jolt_transcript::DigestHash::Sha3_256 => "sha3",
        jolt_transcript::DigestHash::Keccak256 => "keccak",
    };
    let padded = |tag: &[u8; 32], name: &str| {
        assert_eq!(&tag[..name.len()], name.as_bytes());
        assert!(tag[name.len()..].iter().all(|&byte| byte == 0));
    };
    assert_eq!(PROTOCOL_VERSION, 8);
    padded(
        &PROTOCOL_TAG,
        &format!("akita-pcs/transcript/v{PROTOCOL_VERSION}/{digest}"),
    );
    padded(&STANDALONE_SESSION_TAG, "akita-pcs/standalone-session/v8");
    padded(&COMMITMENT_DIGEST_TAG, "akita-pcs/commitment-digest/v8");
    assert_eq!(&CONTEXT_DOMAIN[..27], b"akita-pcs/native-context/v8");
}

/// The standalone session and an embedding caller's session reach different
/// states after the same Akita instance binding.
#[test]
fn standalone_and_embedded_sessions_differ() {
    let mut standalone = new_prover_channel(b"caller");
    bind_instance(&mut standalone, b"descriptor").unwrap();
    let caller_session =
        spongefish::derive_session_id::<jolt_transcript::Shake128>(b"test/embedding-caller");
    let mut embedded = ProverChannel::new(
        &caller_session,
        &jolt_transcript::BytesMsg(b"caller".to_vec()),
    );
    bind_instance(&mut embedded, b"descriptor").unwrap();
    assert_ne!(
        standalone.verifier_message::<[u8; 32]>(),
        embedded.verifier_message::<[u8; 32]>()
    );
}

#[test]
fn bind_instance_absorbs_the_tag_then_a_framed_descriptor() {
    let mut bound = new_prover_channel(b"bind");
    bind_instance(&mut bound, b"abc").unwrap();
    let mut absorbed = PROTOCOL_TAG.to_vec();
    absorbed.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0, b'a', b'b', b'c', 0, 0, 0, 0, 0]);
    assert_eq!(
        bound.verifier_message::<[u8; 32]>(),
        reference_prover(b"bind", &absorbed).verifier_message::<[u8; 32]>()
    );
}

#[test]
fn commitment_digest_hashes_the_tag_then_the_bytes() {
    use sha3::Digest;

    let expected = jolt_transcript::DigestHasher::new()
        .chain_update(COMMITMENT_DIGEST_TAG)
        .chain_update(b"rows")
        .finalize();
    assert_eq!(
        CommitmentDigest::of_canonical_bytes(b"rows")
            .as_bytes()
            .as_slice(),
        expected.as_slice()
    );
    assert_ne!(
        CommitmentDigest::of_canonical_bytes(b"rows"),
        CommitmentDigest::of_canonical_bytes(b"rowz")
    );
}

#[test]
fn field_group_sends_wire_atoms_and_absorbs_a_padded_word() {
    let values = [F::from_u64(42), F::from_u64(43), F::from_u64(44)];
    let mut prover = new_prover_channel(b"fields");
    send_field_group(&mut prover, site(SITE_FAMILY_OPENING_PAYLOAD, 0), &values).unwrap();
    let proof = prover.narg_string().to_vec();
    let mut absorbed = proof.clone();
    assert_eq!(absorbed.len(), 12);
    absorbed.extend_from_slice(&[0; 4]);
    let challenge = prover.verifier_message::<[u8; 32]>();
    assert_eq!(
        challenge,
        reference_prover(b"fields", &absorbed).verifier_message::<[u8; 32]>()
    );

    let mut verifier = new_verifier_channel(b"fields", &proof);
    assert_eq!(
        receive_field_group::<F>(&mut verifier, site(SITE_FAMILY_OPENING_PAYLOAD, 0), 3).unwrap(),
        values
    );
    assert_eq!(verifier.verifier_message::<[u8; 32]>(), challenge);
    verifier.check_eof().unwrap();
}

#[test]
fn byte_group_sends_its_bytes_and_absorbs_them_padded_to_a_word() {
    for len in [0usize, 1, 7, 8, 9, 63, 64, 65, 1025] {
        let bytes = (0..len)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(7))
            .collect::<Vec<_>>();
        let next = *b"next-msg";
        let mut grouped = new_prover_channel(b"bytes");
        send_byte_group(&mut grouped, site(SITE_FAMILY_TERMINAL, 3), &bytes).unwrap();
        public_bytes(&mut grouped, site(SITE_FAMILY_TERMINAL, 4), &next).unwrap();
        let grouped_challenge = grouped.verifier_message::<[u8; 32]>();
        assert_eq!(grouped.narg_string(), bytes);

        let mut padded = bytes.clone();
        padded.resize(len.next_multiple_of(8), 0);
        padded.extend_from_slice(&next);
        assert_eq!(
            grouped_challenge,
            reference_prover(b"bytes", &padded).verifier_message::<[u8; 32]>()
        );
        // The padding is absorbed: it moves the next message to a word
        // boundary, which unpadded bytes do not unless the length is already
        // a whole number of words.
        let mut unpadded = bytes.clone();
        unpadded.extend_from_slice(&next);
        let unpadded = reference_prover(b"bytes", &unpadded).verifier_message::<[u8; 32]>();
        assert_eq!(grouped_challenge == unpadded, len % 8 == 0);

        let proof = grouped.narg_string().to_vec();
        let mut verifier = new_verifier_channel(b"bytes", &proof);
        assert_eq!(
            receive_byte_group(&mut verifier, site(SITE_FAMILY_TERMINAL, 3), len).unwrap(),
            bytes
        );
        public_bytes(&mut verifier, site(SITE_FAMILY_TERMINAL, 4), &next).unwrap();
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), grouped_challenge);
        verifier.check_eof().unwrap();
    }
}

#[test]
fn public_groups_absorb_one_padded_word_sequence() {
    let values = [F::from_u64(1), F::from_u64(2), F::from_u64(3)];
    let mut prover = new_prover_channel(b"public");
    public_fields(&mut prover, site(1, 0), &values).unwrap();
    public_bytes(&mut prover, site(1, 1), b"abc").unwrap();
    assert!(prover.narg_string().is_empty());
    let mut absorbed = Vec::new();
    for value in values {
        absorbed.extend_from_slice(&value.to_bytes_le_vec());
    }
    absorbed.extend_from_slice(&[0; 4]);
    absorbed.extend_from_slice(b"abc\0\0\0\0\0");
    assert_eq!(
        prover.verifier_message::<[u8; 32]>(),
        reference_prover(b"public", &absorbed).verifier_message::<[u8; 32]>()
    );
}

#[test]
fn nonce_is_sent_as_leb128_and_absorbed_as_a_word() {
    let mut prover = new_prover_channel(b"nonce");
    send_nonce(&mut prover, 300);
    assert_eq!(prover.narg_string(), [0xac, 0x02]);
    let challenge = prover.verifier_message::<[u8; 32]>();
    assert_eq!(
        challenge,
        reference_prover(b"nonce", &300u64.to_le_bytes()).verifier_message::<[u8; 32]>()
    );

    let proof = prover.narg_string().to_vec();
    let mut verifier = new_verifier_channel(b"nonce", &proof);
    assert_eq!(receive_nonce(&mut verifier).unwrap(), 300);
    assert_eq!(verifier.verifier_message::<[u8; 32]>(), challenge);
    verifier.check_eof().unwrap();

    let mut verifier = new_verifier_channel(b"nonce", &[0x80, 0x00]);
    assert!(matches!(
        receive_nonce(&mut verifier),
        Err(AkitaError::InvalidProof)
    ));
    assert!(verifier.check_eof().is_err());
}

#[test]
fn extension_groups_and_challenges_replay() {
    let public = [E::from_u64(3), E::from_u64(5)];
    let private = [E::from_u64(8), E::from_u64(13)];
    let mut prover = new_prover_channel(b"groups");
    public_extensions::<F, E, _>(&mut prover, site(27, 1), &public).unwrap();
    let mut sent = private;
    exchange_extension_group::<F, E, _>(&mut prover, site(27, 2), &mut sent).unwrap();
    let prover_one = ext_challenge::<E, _>(&mut prover, site(27, 3)).unwrap();
    let prover_many = ext_challenges::<E, _>(&mut prover, site(27, 4), 5).unwrap();
    let proof = prover.narg_string().to_vec();
    assert_eq!(proof.len(), private.len() * 4 * 4);
    assert_eq!(prover_many.len(), 5);

    let mut verifier = new_verifier_channel(b"groups", &proof);
    public_extensions::<F, E, _>(&mut verifier, site(27, 1), &public).unwrap();
    let mut received = [E::zero(); 2];
    exchange_extension_group::<F, E, _>(&mut verifier, site(27, 2), &mut received).unwrap();
    assert_eq!(received, private);
    assert_eq!(
        ext_challenge::<E, _>(&mut verifier, site(27, 3)).unwrap(),
        prover_one
    );
    assert_eq!(
        ext_challenges::<E, _>(&mut verifier, site(27, 4), 5).unwrap(),
        prover_many
    );
    verifier.check_eof().unwrap();
}

#[test]
fn noncanonical_atoms_poison_the_verifier() {
    let mut verifier = new_verifier_channel(b"poison", &[u8::MAX; 4]);
    assert!(receive_field_group::<F>(&mut verifier, site(1, 0), 1).is_err());
    assert!(ext_challenge::<F, _>(&mut verifier, site(1, 0)).is_err());
    assert!(verifier.check_eof().is_err());

    let encoded = ExtensionAtom::<F, E>::new(E::from_u64(9))
        .encode()
        .as_ref()
        .to_vec();
    let mut noncanonical = encoded.clone();
    noncanonical[4..8].fill(u8::MAX);
    for proof in [&encoded[..encoded.len() - 1], &noncanonical[..]] {
        let mut verifier = new_verifier_channel(b"poison", proof);
        let mut slot = [E::zero()];
        assert!(exchange_extension_group::<F, E, _>(&mut verifier, site(1, 0), &mut slot).is_err());
        assert!(verifier.check_eof().is_err());
    }
}

#[test]
fn bounded_payload_rejects_over_maximum_input_and_proofs() {
    let mut prover = new_prover_channel(b"bounded");
    assert!(matches!(
        send_bounded_bytes(&mut prover, site(SITE_FAMILY_TERMINAL, 7), &[1; 9], 8),
        Err(AkitaError::InvalidInput(_))
    ));

    let mut prover = new_prover_channel(b"bounded");
    crate::ProverTranscript::send(&mut prover, &9u32.to_le_bytes());
    let proof = prover.narg_string().to_vec();
    let mut verifier = new_verifier_channel(b"bounded", &proof);
    assert!(receive_bounded_bytes(&mut verifier, site(SITE_FAMILY_TERMINAL, 7), 8).is_err());
    assert!(verifier.prover_message::<u32>().is_err());
    assert!(verifier.check_eof().is_err());
}

#[test]
fn bounded_payload_round_trips() {
    let target = site(SITE_FAMILY_TERMINAL, 7);
    let mut prover = new_prover_channel(b"bounded");
    send_bounded_bytes(&mut prover, target, &[4, 5, 6], 8).unwrap();
    let proof = prover.narg_string().to_vec();
    assert_eq!(proof, [3, 0, 0, 0, 4, 5, 6]);
    let challenge = prover.verifier_message::<[u8; 32]>();
    let absorbed = [3, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0];
    assert_eq!(
        challenge,
        reference_prover(b"bounded", &absorbed).verifier_message::<[u8; 32]>()
    );
    let mut verifier = new_verifier_channel(b"bounded", &proof);
    assert_eq!(
        receive_bounded_bytes(&mut verifier, target, 8).unwrap(),
        [4, 5, 6]
    );
    assert_eq!(verifier.verifier_message::<[u8; 32]>(), challenge);
    verifier.check_eof().unwrap();
}

#[cfg(feature = "prover")]
#[test]
fn fold_preview_matches_live_prover_and_verifier() {
    use crate::{ProverTranscript, VerifierTranscript};
    const WORK_BITS: u32 = 6;
    let nonce_record = ProtocolContextRecord::new(
        site(SITE_FAMILY_FOLD_CHALLENGE, 0).to_bytes(),
        ProtocolMessageKind::FoldResponseNonce as u32,
        1,
        nonce_max_bytes(12) as u64,
        0,
    );
    let lens = [40usize, 304];
    let message_record = |len: usize| {
        ProtocolContextRecord::new(
            site(SITE_FAMILY_FOLD_CHALLENGE, 1).to_bytes(),
            ProtocolMessageKind::Challenge as u32,
            0,
            0,
            len as u64,
        )
    };
    let copy = |message: &[u8]| Ok(message.to_vec());
    // An arbitrary 12-bit candidate of at least 128, so its LEB128 encoding
    // takes two bytes, drawn from an unrelated transcript.
    let draw = new_prover_channel(b"fold-preview/candidate").verifier_message::<[u8; 32]>();
    let nonce = (u32::from(u16::from_le_bytes([draw[0], draw[1]])) & 0x0fff) | 0x80;
    let mut prover = new_prover_channel(b"fold-preview");
    let mut preview = FoldPreview::new(&prover, nonce);
    preview.grind(WORK_BITS).unwrap();
    let work = preview.take_work().unwrap();
    assert_eq!(work.nonce(), nonce);
    let previewed = [0, 1].map(|i| preview.challenge_message(lens[i], copy).unwrap());
    // One live prover searches again; the other commits the preview's work.
    // Both must produce the same argument and challenges.
    let mut searched = new_prover_channel(b"fold-preview");
    for (state, work) in [(&mut searched, None), (&mut prover, Some(work))] {
        prover_context(state, nonce_record);
        send_nonce(state, nonce);
        match work {
            None => state.grind(WORK_BITS).unwrap(),
            Some(work) => work.commit(state).unwrap(),
        }
    }
    let [searched_live, live] = [&mut searched, &mut prover].map(|state| {
        [0, 1].map(|i| {
            state.context(message_record(lens[i]));
            state.challenge_message(lens[i], copy).unwrap()
        })
    });
    assert_eq!(previewed, live);
    assert_eq!(searched_live, live);
    let proof = prover.narg_string().to_vec();
    assert_eq!(searched.narg_string(), proof.as_slice());
    assert_eq!(&proof[..2], NonceAtom::new(nonce).wire().as_ref());
    assert!(proof.len() > 2);

    let mut verifier = new_verifier_channel(b"fold-preview", &proof);
    verifier_context(&mut verifier, nonce_record);
    assert_eq!(receive_nonce(&mut verifier).unwrap(), nonce);
    verifier.check_grind(WORK_BITS).unwrap();
    let replayed = [0, 1].map(|i| {
        verifier.context(message_record(lens[i]));
        verifier.challenge_message(lens[i], copy).unwrap()
    });
    assert_eq!(replayed, live);
    verifier.check_eof().unwrap();
}

#[cfg(feature = "prover")]
#[test]
fn fold_preview_zero_bit_work_is_a_no_op() {
    let prover = new_prover_channel(b"fold-preview");
    let mut preview = FoldPreview::new(&prover, 3);
    preview.grind(0).unwrap();
    let work = preview.take_work().unwrap();
    assert!(work.argument.is_empty());
    assert_eq!(sponge_fingerprint(&work.sponge), work.start);
}

#[cfg(feature = "prover")]
#[test]
fn fold_work_rejects_a_diverged_live_channel() {
    let mut prover = new_prover_channel(b"fold-preview");
    let mut preview = FoldPreview::new(&prover, 3);
    preview.grind(6).unwrap();
    let work = preview.take_work().unwrap();
    // The live channel sends a different nonce than the preview absorbed.
    send_nonce(&mut prover, 4);
    let before = prover.narg_string().to_vec();
    assert!(matches!(
        work.commit(&mut prover),
        Err(AkitaError::Internal(_))
    ));
    assert_eq!(prover.narg_string(), before.as_slice());
}

#[test]
fn transcript_has_a_known_answer() {
    let mut prover = new_prover_channel(b"kat/session");
    bind_instance(&mut prover, b"kat/descriptor").unwrap();
    public_bytes(&mut prover, site(1, 0), b"public-message").unwrap();
    send_field_group(&mut prover, site(1, 1), &[F::from_u64(7)]).unwrap();
    let challenges = ext_challenges::<E, _>(&mut prover, site(1, 1), 2).unwrap();
    let root = prover.verifier_message::<[u8; 32]>();
    let rendered = format!("{challenges:?} {root:?}");
    // The protocol tag names the digest hash, so it changes every output.
    #[cfg(feature = "transcript-blake2b")]
    let expected = match DIGEST_HASH {
        DigestHash::Sha3_256 => "[FpExt4 { coeffs: [Fp32(2078537693), Fp32(3820402426), Fp32(718280182), Fp32(1765994312)] }, FpExt4 { coeffs: [Fp32(1910168669), Fp32(3377925487), Fp32(3968304959), Fp32(357759526)] }] [250, 104, 138, 7, 199, 56, 28, 85, 254, 234, 61, 62, 140, 134, 28, 240, 47, 181, 8, 185, 61, 226, 128, 237, 214, 108, 14, 174, 68, 103, 47, 171]",
        DigestHash::Keccak256 => "[FpExt4 { coeffs: [Fp32(811873273), Fp32(2718273516), Fp32(3402320926), Fp32(3699840647)] }, FpExt4 { coeffs: [Fp32(827200401), Fp32(1290266527), Fp32(1624312049), Fp32(220072360)] }] [155, 151, 11, 84, 153, 41, 114, 44, 164, 119, 49, 142, 183, 246, 0, 119, 141, 191, 96, 92, 161, 158, 138, 244, 233, 39, 69, 137, 95, 126, 156, 209]",
    };
    #[cfg(feature = "transcript-keccak")]
    let expected = match DIGEST_HASH {
        DigestHash::Sha3_256 => "[FpExt4 { coeffs: [Fp32(1538086018), Fp32(2356685772), Fp32(939000672), Fp32(3159938612)] }, FpExt4 { coeffs: [Fp32(1640874697), Fp32(1035798056), Fp32(3581879581), Fp32(891095212)] }] [54, 72, 90, 56, 239, 211, 128, 30, 12, 80, 158, 176, 19, 198, 95, 96, 222, 120, 215, 81, 16, 223, 98, 231, 87, 39, 214, 184, 63, 131, 16, 17]",
        DigestHash::Keccak256 => "[FpExt4 { coeffs: [Fp32(219517373), Fp32(1460299387), Fp32(770638488), Fp32(354388703)] }, FpExt4 { coeffs: [Fp32(2315217163), Fp32(3996746789), Fp32(1649955996), Fp32(903286857)] }] [45, 189, 119, 79, 141, 42, 244, 166, 109, 51, 121, 195, 225, 65, 69, 175, 160, 233, 76, 180, 209, 158, 20, 150, 31, 42, 189, 172, 221, 107, 54, 59]",
    };
    assert_eq!(rendered, expected);
}

#[test]
fn commitment_digest_has_a_known_answer() {
    let digest = CommitmentDigest::of_canonical_bytes(b"kat/commitment");
    let rendered = format!("{:?}", digest.as_bytes());
    #[cfg(not(feature = "keccak256-digests"))]
    let expected = "[176, 17, 105, 186, 141, 114, 82, 250, 9, 177, 167, 124, 5, 5, 89, 33, 111, 209, 20, 247, 89, 5, 169, 150, 223, 1, 26, 197, 206, 26, 216, 200]";
    #[cfg(feature = "keccak256-digests")]
    let expected = "[231, 43, 247, 28, 199, 65, 243, 106, 211, 102, 169, 32, 222, 237, 118, 31, 29, 18, 4, 66, 30, 243, 72, 227, 153, 236, 242, 73, 22, 152, 224, 17]";
    assert_eq!(rendered, expected);
    assert_ne!(
        digest,
        CommitmentDigest::of_canonical_bytes(b"kat/commitmenu")
    );
    assert_ne!(
        digest,
        CommitmentDigest::of_canonical_bytes(b"kat/commitment\0")
    );
}

/// An empty group makes no absorb call, so it leaves the sponge unchanged
/// at the start and directly after a squeeze, on every sponge backend.
#[test]
fn empty_groups_make_no_absorb_call() {
    let reference = |squeeze_first: bool| {
        let mut state = new_prover_channel(b"empty");
        if squeeze_first {
            let _ = state.verifier_message::<[u8; 32]>();
        }
        state.verifier_message::<[u8; 32]>()
    };
    for squeeze_first in [false, true] {
        let mut state = new_prover_channel(b"empty");
        if squeeze_first {
            let _ = state.verifier_message::<[u8; 32]>();
        }
        send_atoms::<FieldAtom<F>>(&mut state, []);
        public_fields::<F, _>(&mut state, site(2, 0), &[]).unwrap();
        public_extensions::<F, E, _>(&mut state, site(2, 1), &[]).unwrap();
        assert!(state.narg_string().is_empty());
        assert_eq!(
            state.verifier_message::<[u8; 32]>(),
            reference(squeeze_first)
        );

        let mut verifier = new_verifier_channel(b"empty", &[]);
        if squeeze_first {
            let _ = verifier.verifier_message::<[u8; 32]>();
        }
        assert!(receive_atoms::<FieldAtom<F>>(&mut verifier, 0)
            .unwrap()
            .is_empty());
        assert_eq!(
            verifier.verifier_message::<[u8; 32]>(),
            reference(squeeze_first)
        );
        verifier.check_eof().unwrap();
    }
}
