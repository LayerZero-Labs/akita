//! External macro caller: restricted bodies cannot typecheck unused degrees.
use akita_error::AkitaError;
use akita_types::dispatch::{inner_ring_dim_supported_for_tier, validate_compiled_dispatch};
use akita_types::ProtocolRingDispatchTierId;
use akita_types::{dispatch_for_field, ProtocolDispatchSlot, RingRole};
use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

#[test]
fn protocol_policy_is_independent_of_executable_profile() {
    assert!(inner_ring_dim_supported_for_tier(
        ProtocolRingDispatchTierId::Fp64,
        2048
    ));
    assert!(inner_ring_dim_supported_for_tier(
        ProtocolRingDispatchTierId::Fp128,
        64
    ));
    assert_eq!(
        akita_types::compression_ring_dimensions(akita_types::SisModulusProfileId::Q64Offset59),
        [32, 16]
    );
}

#[test]
fn common_supported_request_agrees_with_dispatch() {
    validate_compiled_dispatch::<Prime64Offset59>(ProtocolDispatchSlot::Role(RingRole::Inner), 128)
        .unwrap();
    let result: Result<usize, AkitaError> = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Prime64Offset59,
        128,
        |D| Ok(D)
    );
    assert_eq!(result.unwrap(), 128);
}

#[cfg(feature = "dispatch-aerie")]
mod restricted {
    use super::*;
    struct Degree<const D: usize>;
    trait RoleKernel {
        fn degree() -> usize;
    }
    impl RoleKernel for Degree<128> {
        fn degree() -> usize {
            128
        }
    }
    trait CompressionKernel {
        fn degree() -> usize;
    }
    impl CompressionKernel for Degree<16> {
        fn degree() -> usize {
            16
        }
    }
    impl CompressionKernel for Degree<32> {
        fn degree() -> usize {
            32
        }
    }

    #[test]
    fn unused_arms_are_excluded_before_typechecking() {
        // An ordinary runtime if would still require implementations for
        // unwanted degrees and fail to compile.
        let inner: Result<usize, AkitaError> = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Prime64Offset59,
            128,
            |D| Ok(<Degree<D> as RoleKernel>::degree())
        );
        let outer: Result<usize, AkitaError> = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            Prime64Offset59,
            128,
            |D| Ok(<Degree<D> as RoleKernel>::degree())
        );
        let opening: Result<usize, AkitaError> = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            Prime64Offset59,
            128,
            |D| Ok(<Degree<D> as RoleKernel>::degree())
        );
        let ntt: Result<usize, AkitaError> =
            dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime64Offset59, 128, |D| Ok(
                <Degree<D> as RoleKernel>::degree()
            ));
        assert_eq!(
            [
                inner.unwrap(),
                outer.unwrap(),
                opening.unwrap(),
                ntt.unwrap()
            ],
            [128; 4]
        );
        for d in [16, 32] {
            let compressed: Result<usize, AkitaError> =
                dispatch_for_field!(ProtocolDispatchSlot::Compression, Prime64Offset59, d, |D| {
                    Ok(<Degree<D> as CompressionKernel>::degree())
                });
            assert_eq!(compressed.unwrap(), d);
        }
    }

    #[test]
    fn capability_checks_cover_all_slots_and_reject_foreign_requests() {
        for slot in [
            ProtocolDispatchSlot::Role(RingRole::Inner),
            ProtocolDispatchSlot::Role(RingRole::Outer),
            ProtocolDispatchSlot::Role(RingRole::Opening),
            ProtocolDispatchSlot::Ntt,
            ProtocolDispatchSlot::Compression,
        ] {
            for d in [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
                let expected = match slot {
                    ProtocolDispatchSlot::Compression => [16, 32].contains(&d),
                    _ => d == 128,
                };
                assert_eq!(
                    validate_compiled_dispatch::<Prime64Offset59>(slot, d).is_ok(),
                    expected,
                    "{slot:?} degree {d}"
                );
                if expected {
                    assert!(akita_types::dispatch::slot_dim_supported_for_tier(
                        ProtocolRingDispatchTierId::Fp64,
                        slot,
                        d
                    ));
                }
                assert!(validate_compiled_dispatch::<Prime32Offset99>(slot, d).is_err());
                assert!(validate_compiled_dispatch::<Prime128OffsetA7F7>(slot, d).is_err());
            }
        }
        let bad_degree: Result<usize, AkitaError> = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            Prime64Offset59,
            64,
            |D| Ok(D)
        );
        assert!(bad_degree
            .unwrap_err()
            .to_string()
            .contains("dispatch-aerie"));
        let bad_field: Result<usize, AkitaError> =
            dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime32Offset99, 128, |D| Ok(D));
        assert!(bad_field.unwrap_err().to_string().contains("Fp64"));
    }
}

#[cfg(not(feature = "dispatch-aerie"))]
#[test]
fn default_retains_other_fields_and_degrees() {
    let fp32: Result<usize, AkitaError> = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        Prime32Offset99,
        2048,
        |D| Ok(D)
    );
    let fp128: Result<usize, AkitaError> =
        dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime128OffsetA7F7, 16, |D| Ok(D));
    assert_eq!(fp32.unwrap(), 2048);
    assert_eq!(fp128.unwrap(), 16);
    validate_compiled_dispatch::<Prime32Offset99>(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        2048,
    )
    .unwrap();
}
