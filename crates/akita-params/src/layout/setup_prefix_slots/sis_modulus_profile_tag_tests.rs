use super::*;

#[test]
fn setup_prefix_modulus_profile_wire_tags_round_trip_without_moving() {
    for (profile, tag) in [
        (SisModulusProfileId::Q32Offset99, 0u8),
        (SisModulusProfileId::Q64Offset59, 1),
        (SisModulusProfileId::Q128OffsetA7F7, 2),
        (SisModulusProfileId::Q128Offset275, 4),
    ] {
        let mut wire = Vec::new();
        serialize_sis_modulus_profile(profile, &mut wire).expect("serialize profile");
        assert_eq!(wire, vec![tag]);
        assert_eq!(
            deserialize_sis_modulus_profile(wire.as_slice()).expect("deserialize profile"),
            profile
        );
    }
    // The slot of the retired Q16 profile stays rejected.
    assert!(deserialize_sis_modulus_profile(&[3u8][..]).is_err());
}
