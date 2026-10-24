use super::*;
use jolt_field::{FpExt4, Prime32Offset99 as F, Ring, Zero};

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
    let mut verifier = new_verifier_channel(b"count-overflow", b"fixture", &[]);
    assert!(matches!(
        receive_field_group::<F>(&mut verifier, site, usize::MAX),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(verifier.check_eof().is_err());
}

/// A prover that absorbs `absorbed` as one public message, for comparing the
/// sponge state a helper leaves against a hand-built absorbed layout.
fn reference_prover(session: &[u8], absorbed: &[u8]) -> ProverChannel {
    let mut reference = new_prover_channel(session, b"fixture");
    reference.public_message_as(absorbed, |bytes| bytes);
    reference
}

#[test]
fn rejected_verifier_draws_no_challenge_message() {
    let copy = |message: &[u8]| Ok(message.to_vec());
    let mut verifier = new_verifier_channel(b"fold-message", b"fixture", &[]);
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
    let mut verifier = new_verifier_channel(b"fold-message", b"fixture", &[]);
    let result = verifier.challenge_message(40, |_| {
        Err::<(), _>(AkitaError::InvalidInput("pool ran out".into()))
    });
    assert!(matches!(result, Err(AkitaError::InvalidProof)));
    assert!(verifier.check_eof().is_err());
}

#[test]
fn public_absorption_never_extends_the_argument_string() {
    let mut prover = new_prover_channel(b"public-size", b"fixture");
    prover.prover_message(&[7u8; 3]);
    let before = prover.narg_string().to_vec();
    public_bytes(&mut prover, site(1, 0), b"public fold payload").unwrap();
    assert_eq!(prover.narg_string(), before);
}

#[test]
fn session_and_statement_are_framed_separately() {
    let mut left = new_prover_channel(b"ab", b"c");
    let mut right = new_prover_channel(b"a", b"bc");
    assert_ne!(
        left.verifier_message::<[u8; 32]>(),
        right.verifier_message::<[u8; 32]>()
    );
}

#[test]
fn session_tag_is_the_padded_protocol_name() {
    let name = format!("akita-pcs/transcript/v{PROTOCOL_VERSION}");
    assert_eq!(PROTOCOL_VERSION, 8);
    assert_eq!(SESSION_TAG.len(), 32);
    assert_eq!(&SESSION_TAG[..name.len()], name.as_bytes());
    assert!(SESSION_TAG[name.len()..].iter().all(|&byte| byte == 0));
    assert_eq!(&CONTEXT_DOMAIN[..27], b"akita-pcs/native-context/v8");
}

#[test]
fn field_group_sends_wire_atoms_and_absorbs_a_padded_word() {
    let values = [F::from_u64(42), F::from_u64(43), F::from_u64(44)];
    let mut prover = new_prover_channel(b"fields", b"fixture");
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

    let mut verifier = new_verifier_channel(b"fields", b"fixture", &proof);
    assert_eq!(
        receive_field_group::<F>(&mut verifier, site(SITE_FAMILY_OPENING_PAYLOAD, 0), 3).unwrap(),
        values
    );
    assert_eq!(verifier.verifier_message::<[u8; 32]>(), challenge);
    verifier.check_eof().unwrap();
}

#[test]
fn byte_group_sends_its_bytes_and_absorbs_them_padded_to_a_word() {
    for len in [0, 1, 7, 8, 9, 63, 64, 65, 1025] {
        let bytes = (0..len)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(7))
            .collect::<Vec<_>>();
        let mut grouped = new_prover_channel(b"bytes", b"fixture");
        send_byte_group(&mut grouped, site(SITE_FAMILY_TERMINAL, 3), &bytes).unwrap();
        let grouped_challenge = grouped.verifier_message::<[u8; 32]>();
        assert_eq!(grouped.narg_string(), bytes);

        let mut padded = bytes.clone();
        padded.resize(len.next_multiple_of(8), 0);
        assert_eq!(
            grouped_challenge,
            reference_prover(b"bytes", &padded).verifier_message::<[u8; 32]>()
        );
        // The padding is absorbed: unpadded bytes give a different sponge
        // unless the length is already a whole number of words.
        let unpadded = reference_prover(b"bytes", &bytes).verifier_message::<[u8; 32]>();
        assert_eq!(grouped_challenge == unpadded, len % 8 == 0);

        let proof = grouped.narg_string().to_vec();
        let mut verifier = new_verifier_channel(b"bytes", b"fixture", &proof);
        assert_eq!(
            receive_byte_group(&mut verifier, site(SITE_FAMILY_TERMINAL, 3), len).unwrap(),
            bytes
        );
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), grouped_challenge);
        verifier.check_eof().unwrap();
    }
}

#[test]
fn public_groups_absorb_one_padded_word_sequence() {
    let values = [F::from_u64(1), F::from_u64(2), F::from_u64(3)];
    let mut prover = new_prover_channel(b"public", b"fixture");
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
    let mut prover = new_prover_channel(b"nonce", b"fixture");
    send_nonce(&mut prover, 300);
    assert_eq!(prover.narg_string(), [0xac, 0x02]);
    let challenge = prover.verifier_message::<[u8; 32]>();
    assert_eq!(
        challenge,
        reference_prover(b"nonce", &300u64.to_le_bytes()).verifier_message::<[u8; 32]>()
    );

    let proof = prover.narg_string().to_vec();
    let mut verifier = new_verifier_channel(b"nonce", b"fixture", &proof);
    assert_eq!(receive_nonce(&mut verifier).unwrap(), 300);
    assert_eq!(verifier.verifier_message::<[u8; 32]>(), challenge);
    verifier.check_eof().unwrap();

    let mut verifier = new_verifier_channel(b"nonce", b"fixture", &[0x80, 0x00]);
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
    let mut prover = new_prover_channel(b"groups", b"fixture");
    public_extensions::<F, E, _>(&mut prover, site(27, 1), &public).unwrap();
    let mut sent = private;
    exchange_extension_group::<F, E, _>(&mut prover, site(27, 2), &mut sent).unwrap();
    let prover_one = ext_challenge::<E, _>(&mut prover, site(27, 3)).unwrap();
    let prover_many = ext_challenges::<E, _>(&mut prover, site(27, 4), 5).unwrap();
    let proof = prover.narg_string().to_vec();
    assert_eq!(proof.len(), private.len() * 4 * 4);
    assert_eq!(prover_many.len(), 5);

    let mut verifier = new_verifier_channel(b"groups", b"fixture", &proof);
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
    let mut verifier = new_verifier_channel(b"poison", b"fixture", &[u8::MAX; 4]);
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
        let mut verifier = new_verifier_channel(b"poison", b"fixture", proof);
        let mut slot = [E::zero()];
        assert!(exchange_extension_group::<F, E, _>(&mut verifier, site(1, 0), &mut slot).is_err());
        assert!(verifier.check_eof().is_err());
    }
}

#[test]
fn bounded_payload_rejects_over_maximum_input_and_proofs() {
    let mut prover = new_prover_channel(b"bounded", b"fixture");
    assert!(matches!(
        send_bounded_bytes(&mut prover, site(SITE_FAMILY_TERMINAL, 7), &[1; 9], 8),
        Err(AkitaError::InvalidInput(_))
    ));

    let mut prover = new_prover_channel(b"bounded", b"fixture");
    prover.prover_message(&9u32);
    let proof = prover.narg_string().to_vec();
    let mut verifier = new_verifier_channel(b"bounded", b"fixture", &proof);
    assert!(receive_bounded_bytes(&mut verifier, site(SITE_FAMILY_TERMINAL, 7), 8).is_err());
    assert!(verifier.prover_message::<u32>().is_err());
    assert!(verifier.check_eof().is_err());
}

#[test]
fn bounded_payload_round_trips() {
    let target = site(SITE_FAMILY_TERMINAL, 7);
    let mut prover = new_prover_channel(b"bounded", b"fixture");
    send_bounded_bytes(&mut prover, target, &[4, 5, 6], 8).unwrap();
    let proof = prover.narg_string().to_vec();
    assert_eq!(proof, [3, 0, 0, 0, 4, 5, 6]);
    let challenge = prover.verifier_message::<[u8; 32]>();
    let absorbed = [3, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0];
    assert_eq!(
        challenge,
        reference_prover(b"bounded", &absorbed).verifier_message::<[u8; 32]>()
    );
    let mut verifier = new_verifier_channel(b"bounded", b"fixture", &proof);
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
    let draw =
        new_prover_channel(b"fold-preview/candidate", b"fixture").verifier_message::<[u8; 32]>();
    let nonce = (u32::from(u16::from_le_bytes([draw[0], draw[1]])) & 0x0fff) | 0x80;
    let mut prover = new_prover_channel(b"fold-preview", b"fixture");
    let mut preview = FoldPreview::new(&prover, nonce);
    preview.grind(WORK_BITS).unwrap();
    let work = preview.take_work().unwrap();
    assert_eq!(work.nonce(), nonce);
    let previewed = [0, 1].map(|i| preview.challenge_message(lens[i], copy).unwrap());
    // One live prover searches again; the other commits the preview's work.
    // Both must produce the same argument and challenges.
    let mut searched = new_prover_channel(b"fold-preview", b"fixture");
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

    let mut verifier = new_verifier_channel(b"fold-preview", b"fixture", &proof);
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
    let prover = new_prover_channel(b"fold-preview", b"fixture");
    let mut preview = FoldPreview::new(&prover, 3);
    preview.grind(0).unwrap();
    let work = preview.take_work().unwrap();
    assert!(work.argument.is_empty());
    assert_eq!(sponge_fingerprint(&work.sponge), work.start);
}

#[cfg(feature = "prover")]
#[test]
fn fold_work_rejects_a_diverged_live_channel() {
    let mut prover = new_prover_channel(b"fold-preview", b"fixture");
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
    let mut prover = new_prover_channel(b"kat/session", b"kat/instance");
    public_bytes(&mut prover, site(1, 0), b"public-message").unwrap();
    send_field_group(&mut prover, site(1, 1), &[F::from_u64(7)]).unwrap();
    let challenges = ext_challenges::<E, _>(&mut prover, site(1, 1), 2).unwrap();
    let root = prover.verifier_message::<[u8; 32]>();
    let rendered = format!("{challenges:?} {root:?}");
    #[cfg(feature = "transcript-blake2b")]
    let expected = "[FpExt4 { coeffs: [Fp32(2296752781), Fp32(3059908793), Fp32(163358452), Fp32(2723312530)] }, FpExt4 { coeffs: [Fp32(2502126249), Fp32(3189731935), Fp32(3152105171), Fp32(824640072)] }] [98, 125, 201, 141, 153, 47, 101, 191, 117, 29, 12, 58, 241, 63, 212, 69, 33, 96, 155, 121, 58, 0, 53, 70, 233, 251, 121, 199, 80, 159, 172, 29]";
    #[cfg(feature = "transcript-keccak")]
    let expected = "[FpExt4 { coeffs: [Fp32(2650355531), Fp32(1950568104), Fp32(649483776), Fp32(1708084815)] }, FpExt4 { coeffs: [Fp32(1259572985), Fp32(1159280445), Fp32(2772604853), Fp32(24547)] }] [85, 83, 10, 250, 166, 246, 86, 55, 11, 212, 159, 173, 193, 164, 90, 147, 131, 202, 9, 62, 149, 120, 59, 138, 169, 75, 42, 229, 153, 52, 139, 25]";
    assert_eq!(rendered, expected);
}
