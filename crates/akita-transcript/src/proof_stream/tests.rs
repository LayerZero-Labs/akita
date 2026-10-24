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
    let mut verifier = new_verifier_channel(b"count-overflow", &[]);
    assert!(matches!(
        receive_field_group::<F>(&mut verifier, site, usize::MAX),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(verifier.check_eof().is_err());
}

#[test]
fn rejected_verifier_draws_no_fold_root() {
    let record = ProtocolContextRecord::new(
        site(SITE_FAMILY_FOLD_CHALLENGE, 1).to_bytes(),
        ProtocolMessageKind::Challenge as u32,
        1,
        7,
        FOLD_CHALLENGE_SEED_LEN as u64,
    );
    let mut verifier = new_verifier_channel(b"fold-root", &[]);
    assert!(fold_root(&mut verifier, record, b"payload").is_ok());
    let _ = reject(&mut verifier);
    assert!(matches!(
        fold_root(&mut verifier, record, b"payload"),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn public_absorption_never_extends_the_argument_string() {
    let mut prover = new_prover_channel(b"public-size");
    prover.prover_message(&[7u8; 3]);
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
fn protocol_tag_names_the_backend_and_codecs() {
    #[cfg(feature = "transcript-blake2b")]
    let backend: &[u8] = b"/blake2b512/";
    #[cfg(feature = "transcript-keccak")]
    let backend: &[u8] = b"/keccak-f1600/";
    let contains = |tag: &[u8], needle: &[u8]| tag.windows(needle.len()).any(|w| w == needle);
    assert!(contains(PROTOCOL_TAG, b"/v8/"));
    assert!(contains(PROTOCOL_TAG, backend));
    assert!(contains(PROTOCOL_TAG, b"jolt-pooled-rejection-tv161"));
    assert!(contains(PROTOCOL_TAG, b"jolt-sponge-pow-leb128"));
    assert!(contains(COMMITMENT_DIGEST_TAG, b"/v8/"));
    assert!(contains(
        COMMITMENT_DIGEST_TAG,
        &backend[..backend.len() - 1]
    ));
    assert!(contains(STANDALONE_SESSION_TAG, b"/v8"));
    assert_ne!(PROTOCOL_TAG, STANDALONE_SESSION_TAG);
    assert_ne!(PROTOCOL_TAG, COMMITMENT_DIGEST_TAG);
}

/// The standalone session and an embedding caller's session reach different
/// states after the same Akita instance binding.
#[test]
fn standalone_and_embedded_sessions_differ() {
    let mut standalone = new_prover_channel(b"caller");
    bind_instance(&mut standalone, b"descriptor").unwrap();
    let caller_session = spongefish::derive_session_id::<spongefish::instantiations::TurboShake128>(
        b"test/embedding-caller",
    );
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
fn field_round_trip_and_eof() {
    let mut prover = new_prover_channel(b"session");
    send_atoms(&mut prover, &[FieldAtom::new(F::from_u64(42))]);
    let proof = prover.narg_string().to_vec();

    let mut verifier = new_verifier_channel(b"session", &proof);
    let received = receive_atoms::<FieldAtom<F>>(&mut verifier, 1).unwrap();
    assert_eq!(received[0].into_inner(), F::from_u64(42));
    verifier.check_eof().unwrap();
}

#[test]
fn byte_group_is_one_message_equal_to_bytewise_messages() {
    for len in [0, 1, 63, 64, 65, 1025] {
        let bytes = (0..len)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(7))
            .collect::<Vec<_>>();
        let mut grouped = new_prover_channel(b"bytes");
        send_byte_group(&mut grouped, site(SITE_FAMILY_TERMINAL, 3), &bytes).unwrap();
        let grouped_challenge = grouped.verifier_message::<[u8; 32]>();

        let mut bytewise = new_prover_channel(b"bytes");
        for &byte in &bytes {
            bytewise.prover_message(&byte);
        }
        assert_eq!(grouped.narg_string(), bytes);
        assert_eq!(grouped.narg_string(), bytewise.narg_string());
        assert_eq!(grouped_challenge, bytewise.verifier_message::<[u8; 32]>());

        let proof = grouped.narg_string().to_vec();
        let mut verifier = new_verifier_channel(b"bytes", &proof);
        assert_eq!(
            receive_byte_group(&mut verifier, site(SITE_FAMILY_TERMINAL, 3), len).unwrap(),
            bytes
        );
        assert_eq!(verifier.verifier_message::<[u8; 32]>(), grouped_challenge);
        verifier.check_eof().unwrap();
    }
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
    assert!(receive_atoms::<FieldAtom<F>>(&mut verifier, 1).is_err());
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
        assert!(receive_atoms::<ExtensionAtom<F, E>>(&mut verifier, 1).is_err());
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
    prover.prover_message(&9u32);
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
    let mut verifier = new_verifier_channel(b"bounded", &proof);
    assert_eq!(
        receive_bounded_bytes(&mut verifier, target, 8).unwrap(),
        [4, 5, 6]
    );
    verifier.check_eof().unwrap();
}

#[cfg(feature = "prover")]
#[test]
fn fold_preview_matches_live_prover_and_verifier() {
    let nonce_record = ProtocolContextRecord::new(
        site(SITE_FAMILY_FOLD_CHALLENGE, 0).to_bytes(),
        ProtocolMessageKind::FoldResponseNonce as u32,
        1,
        nonce_max_bytes(12) as u64,
        0,
    );
    let root_record = ProtocolContextRecord::new(
        site(SITE_FAMILY_FOLD_CHALLENGE, 1).to_bytes(),
        ProtocolMessageKind::Challenge as u32,
        3,
        3,
        FOLD_CHALLENGE_SEED_LEN as u64,
    );
    let payloads = [[5u8, 8, 13], [21, 34, 55]];
    // An arbitrary 12-bit candidate of at least 128, so its LEB128 encoding
    // takes two bytes, drawn from an unrelated transcript.
    let draw = new_prover_channel(b"fold-preview/candidate").verifier_message::<[u8; 32]>();
    let nonce = (u32::from(u16::from_le_bytes([draw[0], draw[1]])) & 0x0fff) | 0x80;
    let mut prover = new_prover_channel(b"fold-preview");
    let mut preview = FoldPreview::new(&prover, nonce);
    let previewed = payloads.map(|payload| preview.fold_root(&payload));
    prover_context(&mut prover, nonce_record);
    prover.prover_message(&NonceAtom::new(nonce));
    let live = payloads.map(|payload| fold_root(&mut prover, root_record, &payload).unwrap());
    assert_eq!(previewed, live);
    let proof = prover.narg_string().to_vec();
    assert_eq!(proof.len(), 2);

    let mut verifier = new_verifier_channel(b"fold-preview", &proof);
    verifier_context(&mut verifier, nonce_record);
    assert_eq!(
        verifier.prover_message::<NonceAtom>().unwrap().into_inner(),
        nonce
    );
    let replayed = payloads.map(|payload| fold_root(&mut verifier, root_record, &payload).unwrap());
    assert_eq!(replayed, live);
    verifier.check_eof().unwrap();
}

#[test]
fn transcript_has_a_known_answer() {
    let mut prover = new_prover_channel(b"kat/session");
    bind_instance(&mut prover, b"kat/descriptor").unwrap();
    public_bytes(&mut prover, site(1, 0), b"public-message").unwrap();
    send_atoms(&mut prover, &[FieldAtom::new(F::from_u64(7))]);
    let challenges = ext_challenges::<E, _>(&mut prover, site(1, 1), 2).unwrap();
    let root = prover.verifier_message::<[u8; 32]>();
    let rendered = format!("{challenges:?} {root:?}");
    #[cfg(feature = "transcript-blake2b")]
    let expected = "[FpExt4 { coeffs: [Fp32(2444502442), Fp32(1020706786), Fp32(3029891290), Fp32(3025947715)] }, FpExt4 { coeffs: [Fp32(1070114574), Fp32(1970817519), Fp32(224082722), Fp32(770304463)] }] [3, 143, 220, 150, 220, 142, 193, 180, 65, 74, 106, 189, 19, 197, 132, 39, 132, 219, 173, 190, 4, 3, 55, 236, 158, 251, 206, 38, 153, 35, 203, 29]";
    #[cfg(feature = "transcript-keccak")]
    let expected = "[FpExt4 { coeffs: [Fp32(3285395900), Fp32(819767650), Fp32(2583361512), Fp32(4091088760)] }, FpExt4 { coeffs: [Fp32(4054547815), Fp32(580125663), Fp32(3922885649), Fp32(2086965421)] }] [52, 78, 141, 126, 202, 209, 105, 188, 79, 222, 230, 162, 1, 182, 186, 227, 6, 167, 249, 234, 197, 205, 90, 187, 224, 12, 11, 32, 141, 24, 22, 102]";
    assert_eq!(rendered, expected);
}

#[test]
fn commitment_digest_has_a_known_answer() {
    let digest = CommitmentDigest::of_canonical_bytes(b"kat/commitment");
    let rendered = format!("{:?}", digest.as_bytes());
    #[cfg(feature = "transcript-blake2b")]
    let expected = "[114, 13, 95, 252, 128, 239, 233, 115, 147, 217, 160, 119, 135, 8, 129, 235, 24, 255, 67, 192, 219, 32, 197, 51, 70, 117, 103, 99, 89, 95, 250, 228]";
    #[cfg(feature = "transcript-keccak")]
    let expected = "[165, 210, 69, 154, 136, 128, 20, 10, 186, 18, 41, 186, 227, 72, 194, 41, 129, 150, 70, 0, 27, 114, 114, 251, 183, 140, 215, 116, 24, 172, 81, 160]";
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

#[test]
fn atom_group_is_one_message_equal_to_atomwise_messages() {
    let values: Vec<FieldAtom<F>> = (0..5).map(|i| FieldAtom::new(F::from_u64(i + 3))).collect();
    let mut grouped = new_prover_channel(b"group");
    send_atoms(&mut grouped, &values);
    let mut atomwise = new_prover_channel(b"group");
    for value in &values {
        atomwise.prover_message(value);
    }
    assert_eq!(grouped.narg_string(), atomwise.narg_string());
    assert_eq!(
        grouped.verifier_message::<[u8; 32]>(),
        atomwise.verifier_message::<[u8; 32]>()
    );

    let proof = atomwise.narg_string().to_vec();
    let mut verifier = new_verifier_channel(b"group", &proof);
    let received = receive_atoms::<FieldAtom<F>>(&mut verifier, values.len()).unwrap();
    assert_eq!(received, values);
    verifier.check_eof().unwrap();
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
        send_atoms::<FieldAtom<F>>(&mut state, &[]);
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
