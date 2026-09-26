use super::*;
use akita_serialization::{Valid, DEFAULT_MAX_SEQUENCE_LEN};
use jolt_field::{Prime128Offset275, Prime128OffsetA7F7, Zero};

type F = Prime128OffsetA7F7;

#[test]
fn ring_vec_checked_views_reject_invalid_storage() {
    let empty = RingVec::<F>::from_coeffs(Vec::new());
    assert!(empty.as_single_ring::<64>().is_err());
    assert!(empty
        .as_ring_slice::<64>()
        .expect("empty ring slice")
        .is_empty());
    assert!(empty.as_single_ring::<0>().is_err());
    assert!(empty.as_ring_slice::<0>().is_err());

    let undersized = RingVec::from_coeffs(vec![F::zero(); 63]);
    assert!(undersized.as_single_ring::<64>().is_err());
    assert!(undersized.as_ring_slice::<64>().is_err());

    let mismatched =
        RingVec::from_coeffs_with_ring_dim(vec![F::zero(); 64], 32).expect("stored ring");
    assert!(mismatched.as_single_ring::<64>().is_err());
    assert!(mismatched.as_ring_slice::<64>().is_err());

    let valid =
        RingVec::from_coeffs_with_ring_dim(vec![F::zero(); 64], 64).expect("valid ring storage");
    assert!(valid.as_single_ring::<64>().is_ok());
    assert_eq!(valid.as_ring_slice::<64>().expect("valid slice").len(), 1);
}

#[test]
fn direct_witness_shape_rejects_oversized_allocations() {
    let err = TerminalResponseShape {
        layout: TailSegmentLayout {
            ring_dimension: 64,
            groups: vec![TailSegmentGroupLayout {
                z_coords: 1,
                e_field_elems: DEFAULT_MAX_SEQUENCE_LEN + 1,
                t_field_elems: 0,
                z_linf_cap: Some(1),
                z_payload_bytes: 1,
                z_rice_low_bits: 0,
            }],
            logical_num_elems: DEFAULT_MAX_SEQUENCE_LEN + 1,
        },
    }
    .check()
    .unwrap_err();
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

#[test]
fn flat_ring_vec_deserialization_rejects_shape_before_allocation() {
    let coeffs = DEFAULT_MAX_SEQUENCE_LEN + 1;

    let err = RingVec::<Prime128Offset275>::deserialize_compressed(&[][..], &coeffs)
        .expect_err("shape exceeds cap");
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

#[test]
fn flat_ring_vec_checked_decoders_reject_zero_dimension() {
    let flat = RingVec::<Prime128Offset275>::from_coeffs(vec![]);

    assert!(!flat.can_decode_single(0));
    assert!(!flat.can_decode_vec(0));
}
