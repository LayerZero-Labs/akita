//! Executable field capabilities are additive and separate from protocol policy.

use akita_error::AkitaError;
use akita_types::{
    compiled_field_tier_enabled, dispatch_for_field, validate_compiled_field, ProtocolDispatchSlot,
    ProtocolRingDispatchTierId, RingRole,
};
use jolt_field::Prime64Offset59;

#[test]
fn compiled_capabilities_match_positive_features() {
    assert_eq!(
        compiled_field_tier_enabled(ProtocolRingDispatchTierId::Fp32),
        cfg!(feature = "field-fp32")
    );
    assert_eq!(
        compiled_field_tier_enabled(ProtocolRingDispatchTierId::Fp64),
        cfg!(feature = "field-fp64")
    );
    assert_eq!(
        compiled_field_tier_enabled(ProtocolRingDispatchTierId::Fp128),
        cfg!(feature = "field-fp128")
    );
}

#[cfg(feature = "field-fp64")]
#[test]
fn fp64_keeps_every_protocol_ring_degree() {
    assert!(validate_compiled_field::<Prime64Offset59>().is_ok());
    assert_fp64_dispatches(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        &[64, 128, 256, 512, 1024, 2048],
    );
    assert_fp64_dispatches(ProtocolDispatchSlot::Role(RingRole::Outer), &[64, 128, 256]);
    assert_fp64_dispatches(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        &[64, 128, 256],
    );
    assert_fp64_dispatches(
        ProtocolDispatchSlot::Ntt,
        &[32, 64, 128, 256, 512, 1024, 2048],
    );
    assert_fp64_dispatches(ProtocolDispatchSlot::Compression, &[16, 32]);
}

#[cfg(feature = "field-fp64")]
fn assert_fp64_dispatches(slot: ProtocolDispatchSlot, dimensions: &[usize]) {
    for &dimension in dimensions {
        let result: Result<usize, AkitaError> = match slot {
            ProtocolDispatchSlot::Role(RingRole::Inner) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                Prime64Offset59,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Role(RingRole::Outer) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                Prime64Offset59,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Role(RingRole::Opening) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                Prime64Offset59,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Ntt => {
                dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime64Offset59, dimension, |D| {
                    Ok(D)
                })
            }
            ProtocolDispatchSlot::Compression => dispatch_for_field!(
                ProtocolDispatchSlot::Compression,
                Prime64Offset59,
                dimension,
                |D| Ok(D)
            ),
        };
        assert_eq!(result.unwrap(), dimension);
    }
}

#[test]
fn disabled_fields_report_the_required_feature() {
    #[cfg(not(feature = "field-fp32"))]
    assert!(validate_compiled_field::<jolt_field::Prime32Offset99>()
        .unwrap_err()
        .to_string()
        .contains("field-fp32"));
    #[cfg(not(feature = "field-fp64"))]
    assert!(validate_compiled_field::<Prime64Offset59>()
        .unwrap_err()
        .to_string()
        .contains("field-fp64"));
    #[cfg(not(feature = "field-fp128"))]
    assert!(validate_compiled_field::<jolt_field::Prime128OffsetA7F7>()
        .unwrap_err()
        .to_string()
        .contains("field-fp128"));
}

#[cfg(not(any(
    feature = "field-fp32",
    feature = "field-fp64",
    feature = "field-fp128"
)))]
#[test]
fn zero_field_build_has_no_implicit_full_support() {
    let result: Result<usize, AkitaError> =
        dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime64Offset59, 32, |D| Ok(D));
    assert!(result.unwrap_err().to_string().contains("field-fp64"));
}
