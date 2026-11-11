use super::*;

#[test]
fn sis_modulus_profile_tags_round_trip_without_moving() {
    for (profile, tag) in [
        (SisModulusProfileId::Q32Offset99, 0u8),
        (SisModulusProfileId::Q64Offset59, 1),
        (SisModulusProfileId::Q128Offset6A23801F, 2),
        (SisModulusProfileId::Q128Offset275, 4),
    ] {
        assert_eq!(sis_modulus_profile_tag(profile), tag);
        let decoded =
            decode_sis_modulus_profile(std::io::Cursor::new(u64::from(tag).to_le_bytes()));
        assert_eq!(decoded.expect("decode"), profile);
    }
}
