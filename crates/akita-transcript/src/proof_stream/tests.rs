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

#[test]
fn rejected_verifier_draws_no_fold_root() {
    let record = ProtocolContextRecord::new(
        site(SITE_FAMILY_FOLD_CHALLENGE, 1).to_bytes(),
        ProtocolMessageKind::Challenge as u32,
        1,
        7,
        FOLD_CHALLENGE_SEED_LEN as u64,
    );
    let mut verifier = new_verifier_channel(b"fold-root", b"fixture", &[]);
    assert!(fold_root(&mut verifier, record, b"payload").is_ok());
    let _ = reject(&mut verifier);
    assert!(matches!(
        fold_root(&mut verifier, record, b"payload"),
        Err(AkitaError::InvalidProof)
    ));
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
fn session_tag_names_the_backend_and_codecs() {
    #[cfg(feature = "transcript-blake2b")]
    let backend: &[u8] = b"/blake2b512/";
    #[cfg(feature = "transcript-keccak")]
    let backend: &[u8] = b"/keccak-f1600/";
    let contains = |needle: &[u8]| SESSION_TAG.windows(needle.len()).any(|w| w == needle);
    assert!(contains(b"/v8/"));
    assert!(contains(backend));
    assert!(contains(b"jolt-pooled-rejection-tv161"));
    assert!(contains(b"jolt-sponge-pow-leb128"));
}

#[test]
fn field_round_trip_and_eof() {
    let mut prover = new_prover_channel(b"session", b"instance");
    send_field(&mut prover, F::from_u64(42));
    let proof = prover.narg_string().to_vec();

    let mut verifier = new_verifier_channel(b"session", b"instance", &proof);
    assert_eq!(receive_field::<F>(&mut verifier).unwrap(), F::from_u64(42));
    verifier.check_eof().unwrap();
}

#[test]
fn byte_group_is_one_message_equal_to_bytewise_messages() {
    for len in [0, 1, 63, 64, 65, 1025] {
        let bytes = (0..len)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(7))
            .collect::<Vec<_>>();
        let mut grouped = new_prover_channel(b"bytes", b"instance");
        send_byte_group(&mut grouped, site(SITE_FAMILY_TERMINAL, 3), &bytes).unwrap();
        let grouped_challenge = grouped.verifier_message::<[u8; 32]>();

        let mut bytewise = new_prover_channel(b"bytes", b"instance");
        for &byte in &bytes {
            bytewise.prover_message(&byte);
        }
        assert_eq!(grouped.narg_string(), bytes);
        assert_eq!(grouped.narg_string(), bytewise.narg_string());
        assert_eq!(grouped_challenge, bytewise.verifier_message::<[u8; 32]>());

        let proof = grouped.narg_string().to_vec();
        let mut verifier = new_verifier_channel(b"bytes", b"instance", &proof);
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
    assert!(receive_field::<F>(&mut verifier).is_err());
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
        assert!(receive_extension::<F, E>(&mut verifier).is_err());
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
    let mut verifier = new_verifier_channel(b"bounded", b"fixture", &proof);
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
    let draw =
        new_prover_channel(b"fold-preview/candidate", b"fixture").verifier_message::<[u8; 32]>();
    let nonce = (u32::from(u16::from_le_bytes([draw[0], draw[1]])) & 0x0fff) | 0x80;
    let mut prover = new_prover_channel(b"fold-preview", b"fixture");
    let mut preview = FoldPreview::new(&prover, nonce);
    let previewed = payloads.map(|payload| preview.fold_root(&payload));
    prover_context(&mut prover, nonce_record);
    prover.prover_message(&NonceAtom::new(nonce));
    let live = payloads.map(|payload| fold_root(&mut prover, root_record, &payload).unwrap());
    assert_eq!(previewed, live);
    let proof = prover.narg_string().to_vec();
    assert_eq!(proof.len(), 2);

    let mut verifier = new_verifier_channel(b"fold-preview", b"fixture", &proof);
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
    let mut prover = new_prover_channel(b"kat/session", b"kat/instance");
    public_bytes(&mut prover, site(1, 0), b"public-message").unwrap();
    send_field(&mut prover, F::from_u64(7));
    let challenges = ext_challenges::<E, _>(&mut prover, site(1, 1), 2).unwrap();
    let root = prover.verifier_message::<[u8; 32]>();
    let rendered = format!("{challenges:?} {root:?}");
    #[cfg(feature = "transcript-blake2b")]
    let expected = "[FpExt4 { coeffs: [Fp32(175589814), Fp32(1706661519), Fp32(2395894747), Fp32(3103557429)] }, FpExt4 { coeffs: [Fp32(539557180), Fp32(1506485249), Fp32(1583258172), Fp32(1897420848)] }] [189, 102, 0, 49, 154, 152, 65, 90, 28, 5, 214, 223, 51, 171, 136, 148, 141, 35, 128, 39, 11, 75, 235, 56, 187, 239, 14, 213, 137, 114, 160, 238]";
    #[cfg(feature = "transcript-keccak")]
    let expected = "[FpExt4 { coeffs: [Fp32(3180819226), Fp32(3648517247), Fp32(3845021235), Fp32(1306707338)] }, FpExt4 { coeffs: [Fp32(2824915371), Fp32(2110316638), Fp32(2186556993), Fp32(673535351)] }] [113, 69, 72, 143, 203, 149, 26, 138, 64, 123, 84, 231, 10, 179, 142, 121, 189, 188, 216, 222, 141, 128, 158, 143, 84, 184, 174, 26, 116, 126, 240, 230]";
    assert_eq!(rendered, expected);
}
