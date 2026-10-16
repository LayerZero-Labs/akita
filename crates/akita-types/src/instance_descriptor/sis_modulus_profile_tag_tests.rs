use super::*;

#[test]
fn sis_modulus_profile_tags_round_trip_without_moving() {
    for (profile, tag) in [
        (SisModulusProfileId::Q32Offset99, 0u8),
        (SisModulusProfileId::Q64Offset59, 1),
        (SisModulusProfileId::Q128OffsetA7F7, 2),
        (SisModulusProfileId::Q128Offset275, 4),
    ] {
        let mut bytes = Vec::new();
        encode_sis_modulus_profile(profile, &mut bytes, Compress::No).expect("encode");
        assert_eq!(bytes, [tag]);
        let decoded = decode_sis_modulus_profile(&bytes[..], Compress::No, Validate::Yes);
        assert_eq!(decoded.expect("decode"), profile);
    }
}
