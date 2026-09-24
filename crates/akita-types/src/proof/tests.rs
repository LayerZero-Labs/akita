use super::*;
use akita_serialization::Valid;
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
    assert!(flat.try_to_single::<0>().is_err());
    assert!(flat.try_to_vec::<0>().is_err());
}

#[test]
fn level_shape_validation_checks_extension_opening_reduction() {
    let oversized = LevelProofShape {
        extension_opening_reduction: Some(ExtensionOpeningReductionShape::standard(
            DEFAULT_MAX_SEQUENCE_LEN + 1,
            1,
            1,
        )),
        opening_payload_coeffs: 1,
        stage1_stages: Vec::new(),
        stage1_norm: None,
        stage2_sumcheck_proof: Vec::new(),
        stage3_sumcheck: None,
        next_witness_binding: NextWitnessBindingShape::OuterPayload { coeffs: 1 },
    };

    let err = oversized.check().unwrap_err();
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));

    let wrong_degree = LevelProofShape {
        extension_opening_reduction: Some(ExtensionOpeningReductionShape {
            partials: 1,
            final_claims: 1,
            sumcheck: vec![EXTENSION_OPENING_REDUCTION_DEGREE + 1],
        }),
        ..oversized
    };

    let err = wrong_degree.check().unwrap_err();
    assert!(matches!(err, SerializationError::InvalidData(_)));
}

#[test]
fn level_shape_deserialization_rejects_vector_length_before_allocation() {
    let mut bytes = Vec::new();
    false.serialize_compressed(&mut bytes).unwrap(); // extension_opening_reduction
    0usize.serialize_compressed(&mut bytes).unwrap(); // opening_payload_coeffs
    (MAX_PROOF_SHAPE_SEQUENCE_LEN as u64 + 1)
        .serialize_compressed(&mut bytes)
        .unwrap(); // stage1_stages

    let err = LevelProofShape::deserialize_compressed(&bytes[..], &())
        .expect_err("oversized shape vector must be rejected before allocation");
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

#[test]
fn l2_shape_deserialization_rejects_rounds_before_allocation() {
    let mut bytes = Vec::new();
    0usize.serialize_compressed(&mut bytes).unwrap(); // subclaims
    1usize.serialize_compressed(&mut bytes).unwrap(); // virtual evaluations
    (MAX_PROOF_SHAPE_SEQUENCE_LEN as u64 + 1)
        .serialize_compressed(&mut bytes)
        .unwrap(); // sumcheck round count

    let err = PhysicalL2NormProofWireShape::deserialize_compressed(&bytes[..], &())
        .expect_err("oversized L2 round vector must be rejected before allocation");
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}
