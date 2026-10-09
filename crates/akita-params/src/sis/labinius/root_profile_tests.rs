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
    for tag in 1..=u8::MAX {
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
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            BinaryScalarRing::Cyclotomic729,
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            p.response_interval(),
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            LabiniusCoefficientPrime::P64Offset23703,
            p.ring_degree(),
            c.identity_bytes(),
            p.response_interval(),
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            LabiniusRingDegree::D324,
            c.identity_bytes(),
            p.response_interval(),
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            altered_challenge.identity_bytes(),
            p.response_interval(),
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            (-32_767, 32_767),
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            (-32_768, 32_768),
            LABINIUS_WIDTH_TABLE_DIGEST,
        ),
        encode_identity(
            0,
            p.scalar_ring(),
            p.coefficient_prime(),
            p.ring_degree(),
            c.identity_bytes(),
            p.response_interval(),
            digest,
        ),
    ];
    for variant in variants {
        assert_ne!(variant, original);
    }
}
