//! Executable field capabilities are additive and separate from protocol policy.

#[cfg(feature = "field-fp64")]
use akita_algebra::{ntt::tables::Q64_PRIMES, CyclotomicRing};
use akita_error::AkitaError;
use akita_types::dispatch::slot_dim_supported_for_tier;
#[cfg(feature = "field-fp64")]
use akita_types::{
    centered_quotient_requires_i16_tail, centered_quotient_requires_i16_tail_for_field,
    prepare_ntt_cache, select_crt_ntt_params, FlatMatrix, NttCacheMode, ProtocolCrtNttParams,
    SisModulusProfileId,
};
use akita_types::{
    compiled_field_tier_enabled, dispatch_for_field, validate_compiled_field, ProtocolDispatchSlot,
    ProtocolRingDispatchTierId, RingRole,
};
use jolt_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};

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

#[test]
fn canonical_protocol_policy_is_not_pruned() {
    for (tier, slot, dimensions) in [
        (
            ProtocolRingDispatchTierId::Fp128,
            ProtocolDispatchSlot::Ntt,
            &[16usize, 32, 64, 128, 256, 512, 1024][..],
        ),
        (
            ProtocolRingDispatchTierId::Fp64,
            ProtocolDispatchSlot::Ntt,
            &[32usize, 64, 128, 256, 512, 1024, 2048][..],
        ),
        (
            ProtocolRingDispatchTierId::Fp32,
            ProtocolDispatchSlot::Ntt,
            &[64usize, 128, 256, 512, 1024, 2048][..],
        ),
    ] {
        for &dimension in dimensions {
            assert!(slot_dim_supported_for_tier(tier, slot, dimension));
        }
    }
}

#[cfg(feature = "field-fp32")]
#[test]
fn fp32_keeps_every_protocol_ring_degree() {
    assert!(validate_compiled_field::<Prime32Offset99>().is_ok());
    assert_dispatches_fp32(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        &[64, 128, 256, 512, 1024, 2048],
    );
    assert_dispatches_fp32(ProtocolDispatchSlot::Role(RingRole::Outer), &[64, 128, 256]);
    assert_dispatches_fp32(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        &[64, 128, 256],
    );
    assert_dispatches_fp32(ProtocolDispatchSlot::Ntt, &[64, 128, 256, 512, 1024, 2048]);
    assert_dispatches_fp32(ProtocolDispatchSlot::Compression, &[32, 64]);
}

#[cfg(feature = "field-fp32")]
fn assert_dispatches_fp32(slot: ProtocolDispatchSlot, dimensions: &[usize]) {
    for &dimension in dimensions {
        let result: Result<usize, AkitaError> = match slot {
            ProtocolDispatchSlot::Role(RingRole::Inner) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                Prime32Offset99,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Role(RingRole::Outer) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                Prime32Offset99,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Role(RingRole::Opening) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                Prime32Offset99,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Ntt => {
                dispatch_for_field!(ProtocolDispatchSlot::Ntt, Prime32Offset99, dimension, |D| {
                    Ok(D)
                })
            }
            ProtocolDispatchSlot::Compression => dispatch_for_field!(
                ProtocolDispatchSlot::Compression,
                Prime32Offset99,
                dimension,
                |D| Ok(D)
            ),
        };
        assert_eq!(result.unwrap(), dimension);
    }
}

#[cfg(feature = "field-fp64")]
#[test]
fn fp64_keeps_every_protocol_ring_degree() {
    assert!(validate_compiled_field::<Prime64Offset59>().is_ok());
    assert_dispatches_fp64(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        &[64, 128, 256, 512, 1024, 2048],
    );
    assert_dispatches_fp64(ProtocolDispatchSlot::Role(RingRole::Outer), &[64, 128, 256]);
    assert_dispatches_fp64(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        &[64, 128, 256],
    );
    assert_dispatches_fp64(
        ProtocolDispatchSlot::Ntt,
        &[32, 64, 128, 256, 512, 1024, 2048],
    );
    assert_dispatches_fp64(ProtocolDispatchSlot::Compression, &[16, 32]);
}

#[cfg(feature = "field-fp64")]
#[test]
fn fp64_only_build_keeps_q64_crt_cache_and_tail_policy() {
    const D: usize = 64;
    let ProtocolCrtNttParams::Q64(params) =
        select_crt_ntt_params::<Prime64Offset59, D>().expect("Q64 parameters")
    else {
        panic!("Fp64 must select the Q64 CRT profile");
    };
    assert_eq!(params.primes, Q64_PRIMES);

    let matrix = FlatMatrix::from_ring_slice(&[CyclotomicRing::<Prime64Offset59, D>::zero()]);
    let cache = prepare_ntt_cache(
        matrix.ring_view::<D>(1, 1).expect("matrix view"),
        NttCacheMode::BothTransforms,
    )
    .expect("Q64 cache");
    let base = cache.q64_base().expect("Q64 cache representation");
    assert_eq!(base.params().primes, Q64_PRIMES);
    assert_eq!(base.negacyclic().expect("negacyclic cache").len(), 1);
    assert_eq!(base.cyclic().expect("cyclic cache").len(), 1);

    for rhs_abs_bound in [1, 1 << 15, 1_000_000, u64::from(u32::MAX)] {
        assert_eq!(
            centered_quotient_requires_i16_tail(SisModulusProfileId::Q64Offset59, D, rhs_abs_bound,),
            centered_quotient_requires_i16_tail_for_field::<Prime64Offset59, D>(rhs_abs_bound),
        );
    }
}

#[cfg(feature = "field-fp64")]
fn assert_dispatches_fp64(slot: ProtocolDispatchSlot, dimensions: &[usize]) {
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

#[cfg(feature = "field-fp128")]
#[test]
fn fp128_keeps_every_protocol_ring_degree() {
    assert!(validate_compiled_field::<Prime128OffsetA7F7>().is_ok());
    assert_dispatches_fp128(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        &[64, 128, 256, 512, 1024],
    );
    assert_dispatches_fp128(ProtocolDispatchSlot::Role(RingRole::Outer), &[64, 128, 256]);
    assert_dispatches_fp128(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        &[64, 128, 256],
    );
    assert_dispatches_fp128(
        ProtocolDispatchSlot::Ntt,
        &[16, 32, 64, 128, 256, 512, 1024],
    );
    assert_dispatches_fp128(ProtocolDispatchSlot::Compression, &[8, 16]);
}

#[cfg(feature = "field-fp128")]
fn assert_dispatches_fp128(slot: ProtocolDispatchSlot, dimensions: &[usize]) {
    for &dimension in dimensions {
        let result: Result<usize, AkitaError> = match slot {
            ProtocolDispatchSlot::Role(RingRole::Inner) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                Prime128OffsetA7F7,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Role(RingRole::Outer) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                Prime128OffsetA7F7,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Role(RingRole::Opening) => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                Prime128OffsetA7F7,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Ntt => dispatch_for_field!(
                ProtocolDispatchSlot::Ntt,
                Prime128OffsetA7F7,
                dimension,
                |D| Ok(D)
            ),
            ProtocolDispatchSlot::Compression => dispatch_for_field!(
                ProtocolDispatchSlot::Compression,
                Prime128OffsetA7F7,
                dimension,
                |D| Ok(D)
            ),
        };
        assert_eq!(result.unwrap(), dimension);
    }
}

#[cfg(not(feature = "field-fp32"))]
#[test]
fn disabled_fp32_is_rejected_clearly() {
    let error = validate_compiled_field::<Prime32Offset99>().unwrap_err();
    assert!(error.to_string().contains("field-fp32"));
}

#[cfg(not(feature = "field-fp64"))]
#[test]
fn disabled_fp64_is_rejected_clearly() {
    let error = validate_compiled_field::<Prime64Offset59>().unwrap_err();
    assert!(error.to_string().contains("field-fp64"));
}

#[cfg(not(feature = "field-fp128"))]
#[test]
fn disabled_fp128_is_rejected_clearly() {
    let error = validate_compiled_field::<Prime128OffsetA7F7>().unwrap_err();
    assert!(error.to_string().contains("field-fp128"));
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
    let error = result.unwrap_err();
    assert!(error.to_string().contains("field-fp64"));
}
