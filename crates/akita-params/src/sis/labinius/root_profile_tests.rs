use super::*;

#[test]
fn closed_profile_tags_and_exact_parameters() {
    for profile in LabiniusRootProfile::ALL {
        assert_eq!(
            LabiniusRootProfile::from_tag(profile.tag()).unwrap(),
            profile
        );
        assert_eq!(profile.scalar_ring(), BinaryScalarRing::Cyclotomic243);
        assert_eq!(profile.ring_degree(), LabiniusRingDegree::D648);
        assert_eq!(
            profile.coefficient_prime(),
            LabiniusCoefficientPrime::P128OffsetA7F7
        );
        assert_eq!(profile.response_interval(), (-32_768, 32_767));
        let challenge = profile.challenge_profile().unwrap();
        assert_eq!(challenge.coefficient_l1_bound(), 46);
        assert_eq!(challenge.multiplication_linf_operator_bound(), 92);
        assert_eq!(
            profile.identity_bytes().unwrap(),
            profile.identity_bytes().unwrap()
        );
    }
    for tag in 2..=u8::MAX {
        assert!(matches!(
            LabiniusRootProfile::from_tag(tag),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}

#[test]
fn identity_binds_each_field_independently() {
    let p = LabiniusRootProfile::ALL[0];
    let c = p.challenge_profile().unwrap();
    let mut digest = LABINIUS_WIDTH_TABLE_DIGEST;
    digest[0] ^= 1;
    let original = p.identity_bytes().unwrap();
    let altered_challenge = BinaryChallengeProfile::bounded_weight(p.scalar_ring(), 45).unwrap();
    let variants = [
        encode_identity(
            1,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            p.response_interval(),
            None,
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            BinaryScalarRing::Cyclotomic729,
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            p.response_interval(),
            None,
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            LabiniusCoefficientPrime::P64Offset23703,
            p.ring_degree(),
            c.identity_bytes(),
            p.response_interval(),
            None,
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            LabiniusRingDegree::D324,
            c.identity_bytes(),
            p.response_interval(),
            None,
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            altered_challenge.identity_bytes(),
            p.response_interval(),
            None,
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            (-32_767, 32_767),
            None,
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            (-32_768, 32_768),
            None,
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            p.response_interval(),
            None,
            digest,
        ),
    ];
    for variant in variants {
        assert_ne!(variant, original);
    }
}

#[test]
fn root_profile_identity_matches_hand_assembled_stable_fixture() {
    let prime = 340282366920938463463374607427473266697u128;
    assert_eq!(prime, u128::MAX - (1u128 << 32) + 22538);
    let challenge =
        BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 46).unwrap();
    let digest: Vec<u8> = "0586ee6294846d118d1e6ef364f10c40f8e5678f135f11dcde42b0fb61a89bfd"
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    assert_eq!(digest.len(), 32);

    let mut expected = b"akita/labinius/root-profile/v1\0".to_vec();
    expected.extend_from_slice(&[0, 0]); // Profile tag, then scalar-ring tag.
    expected.extend_from_slice(&prime.to_le_bytes());
    expected.extend_from_slice(&648u32.to_le_bytes());
    for &byte in challenge.identity_bytes() {
        expected.extend_from_slice(&[1, byte]);
    }
    expected.push(0);
    expected.extend_from_slice(&(-32768i128).to_le_bytes());
    expected.extend_from_slice(&32767i128.to_le_bytes());
    expected.extend_from_slice(&digest);

    assert_eq!(
        LabiniusRootProfile::D648P128BoundedW46Delta16
            .identity_bytes()
            .unwrap(),
        expected
    );
}

#[test]
fn existing_identity_digest_is_the_base_commit_literal() {
    use sha3::{Digest, Sha3_256};
    let old = LabiniusRootProfile::D648P128BoundedW46Delta16;
    let new = LabiniusRootProfile::D648P128Q28BoundedW46Delta16;
    let digest: [u8; 32] = Sha3_256::digest(old.identity_bytes().unwrap()).into();
    assert_eq!(
        digest,
        [
            207, 31, 247, 89, 41, 150, 77, 88, 189, 124, 4, 204, 43, 83, 250, 0, 17, 166, 14, 148,
            32, 194, 221, 202, 124, 103, 52, 68, 224, 75, 21, 112
        ]
    );
    assert_ne!(old.identity_bytes().unwrap(), new.identity_bytes().unwrap());
    assert_eq!(old.tag(), 0);
    assert_eq!(new.tag(), 1);
    assert_eq!(
        old.commitment_modulus(),
        LabiniusCommitmentModulus::CoefficientPrime
    );
    assert_eq!(
        new.commitment_modulus(),
        LabiniusCommitmentModulus::Q28Offset2103
    );
    assert_eq!(new.commitment_modulus().small_modulus(), Some(268_433_353));
    assert_eq!(new.commitment_modulus().label(), "q28-2103");
    assert_eq!(old.commitment_modulus().tag(), 0);
    assert_eq!(new.commitment_modulus().tag(), 1);
    assert_eq!(old.commitment_modulus().small_modulus(), None);
}

#[test]
fn small_identity_digest_is_a_literal() {
    // Independent of the generated table digest constant: regenerating the
    // small-modulus table changes this identity and must fail here.
    use sha3::{Digest, Sha3_256};
    let identity = LabiniusRootProfile::D648P128Q28BoundedW46Delta16
        .identity_bytes()
        .unwrap();
    let digest: [u8; 32] = Sha3_256::digest(identity).into();
    assert_eq!(
        digest,
        [
            154, 226, 234, 231, 114, 38, 215, 0, 157, 72, 77, 215, 193, 215, 124, 124, 78, 205,
            190, 116, 77, 217, 48, 56, 132, 231, 232, 157, 188, 49, 143, 49
        ]
    );
}

#[test]
fn small_identity_binds_modulus_and_its_separate_table_digest() {
    let old = LabiniusRootProfile::D648P128BoundedW46Delta16
        .identity_bytes()
        .unwrap();
    let new = LabiniusRootProfile::D648P128Q28BoundedW46Delta16
        .identity_bytes()
        .unwrap();
    let mut expected = old;
    expected[31] = 1; // Profile tag follows the 31-byte domain.
    expected.truncate(expected.len() - 32);
    expected.push(1);
    expected.extend_from_slice(&268_433_353u32.to_le_bytes());
    expected.extend_from_slice(&LABINIUS_SMALL_MODULUS_WIDTH_TABLE_DIGEST);
    assert_eq!(new, expected);
}
