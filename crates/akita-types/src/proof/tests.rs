use super::wire::extension_opening_reduction_serialized_size;
use super::*;
use akita_algebra::CompressedUniPoly;
use akita_field::Prime128Offset275;
use akita_serialization::Valid;
use akita_sumcheck::SumcheckProof;

type F = Prime128Offset275;

#[test]
fn direct_witness_shape_rejects_oversized_allocations() {
    let err = CleartextWitnessShape::FieldElements(DEFAULT_MAX_SEQUENCE_LEN + 1)
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

    let err = FlatRingVec::<Prime128Offset275>::deserialize_compressed(&[][..], &coeffs)
        .expect_err("shape exceeds cap");
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

#[test]
fn flat_ring_vec_checked_decoders_reject_zero_dimension() {
    let flat = FlatRingVec::<Prime128Offset275>::from_coeffs(vec![]);

    assert!(!flat.can_decode_single(0));
    assert!(!flat.can_decode_vec(0));
    assert!(flat.try_to_single::<0>().is_err());
    assert!(flat.try_to_vec::<0>().is_err());
    assert!(flat.as_ring_slice::<0>().is_err());
    assert!(flat.try_to_ring_commitment::<0>().is_err());
}

#[test]
fn batched_proof_shape_validation_recurses_into_witness_shapes() {
    let shape = AkitaBatchedProofShape::ZeroFold {
        witness_shapes: vec![CleartextWitnessShape::FieldElements(
            DEFAULT_MAX_SEQUENCE_LEN + 1,
        )],
    };

    let err = shape.check().unwrap_err();
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

#[test]
fn level_shape_validation_checks_extension_opening_reduction() {
    let oversized = LevelProofShape {
        extension_opening_reduction: Some(ExtensionOpeningReductionShape::standard(
            DEFAULT_MAX_SEQUENCE_LEN + 1,
            1,
        )),
        v_coeffs: 1,
        stage1_stages: Vec::new(),
        stage2_sumcheck_proof: Vec::new(),
        stage3_sumcheck: None,
        next_commit_coeffs: 1,
    };

    let err = oversized.check().unwrap_err();
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));

    let wrong_degree = LevelProofShape {
        extension_opening_reduction: Some(ExtensionOpeningReductionShape {
            partials: 1,
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
    0usize.serialize_compressed(&mut bytes).unwrap(); // v_coeffs
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
fn terminal_shape_deserialization_validates_shape() {
    let mut bytes = Vec::new();
    false.serialize_compressed(&mut bytes).unwrap(); // extension_opening_reduction
    (MAX_PROOF_SHAPE_SEQUENCE_LEN as u64 + 1)
        .serialize_compressed(&mut bytes)
        .unwrap(); // stage2_sumcheck

    let err = TerminalLevelProofShape::deserialize_compressed(&bytes[..], &())
        .expect_err("oversized terminal sumcheck shape must be rejected");
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

#[test]
fn terminal_level_proof_deserialization_validates_context_shape() {
    let shape = TerminalLevelProofShape {
        extension_opening_reduction: None,
        stage2_sumcheck: vec![0; MAX_PROOF_SHAPE_SEQUENCE_LEN + 1],
        final_witness: CleartextWitnessShape::FieldElements(0),
    };

    let err = TerminalLevelProof::<F, F>::deserialize_compressed(&[][..], &shape)
        .expect_err("oversized terminal proof context shape must be rejected");
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

fn tiny_stage1() -> AkitaStage1Proof<F> {
    AkitaStage1Proof {
        stages: Vec::new(),
        s_claim: F::zero(),
    }
}

fn tiny_stage2<const D: usize>() -> AkitaStage2Proof<F, F> {
    AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
        sumcheck_proof: SumcheckProof {
            round_polys: Vec::new(),
        },
        next_w_commitment: FlatRingVec::from_ring_elems(&[CyclotomicRing::<F, D>::zero()])
            .into_compact(),
        next_w_eval: F::zero(),
    })
}

fn tiny_reduction() -> ExtensionOpeningReductionProof<F> {
    ExtensionOpeningReductionProof {
        partials: vec![F::zero(), F::one()],
        sumcheck: SumcheckProof {
            round_polys: vec![CompressedUniPoly {
                coeffs_except_linear_term: vec![F::zero(), F::one()],
            }],
        },
    }
}

#[test]
fn extension_opening_reduction_none_is_zero_proof_wire_bytes() {
    const D: usize = 8;
    let without_reduction = AkitaLevelProof::new::<D>(
        vec![CyclotomicRing::<F, D>::zero()],
        tiny_stage1(),
        tiny_stage2::<D>(),
    );
    assert!(without_reduction.extension_opening_reduction().is_none());
    assert!(without_reduction
        .shape()
        .extension_opening_reduction
        .is_none());

    let mut bytes = Vec::new();
    without_reduction
        .serialize_uncompressed(&mut bytes)
        .expect("serialize proof without extension-opening reduction");
    assert_eq!(bytes.len(), without_reduction.serialized_size(Compress::No));

    let decoded =
        AkitaLevelProof::<F, F>::deserialize_uncompressed(&*bytes, &without_reduction.shape())
            .expect("deserialize proof without extension-opening reduction");
    assert!(decoded.extension_opening_reduction().is_none());
    assert_eq!(decoded, without_reduction);

    let with_reduction = AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
        Some(tiny_reduction()),
        vec![CyclotomicRing::<F, D>::zero()],
        tiny_stage1(),
        SumcheckProof {
            round_polys: Vec::new(),
        },
        FlatRingVec::from_ring_elems(&[CyclotomicRing::<F, D>::zero()]).into_compact(),
        F::zero(),
    );
    let reduction_bytes = extension_opening_reduction_serialized_size(
        with_reduction.extension_opening_reduction(),
        Compress::No,
    );
    assert!(reduction_bytes > 0);
    assert_eq!(
        with_reduction.serialized_size(Compress::No)
            - without_reduction.serialized_size(Compress::No),
        reduction_bytes
    );

    let mut bytes_with_reduction = Vec::new();
    with_reduction
        .serialize_uncompressed(&mut bytes_with_reduction)
        .expect("serialize proof with extension-opening reduction");
    let decoded_with_reduction = AkitaLevelProof::<F, F>::deserialize_uncompressed(
        &*bytes_with_reduction,
        &with_reduction.shape(),
    )
    .expect("deserialize proof with extension-opening reduction");
    assert_eq!(decoded_with_reduction, with_reduction);
}

fn tiny_terminal_stage2() -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: Vec::new(),
    }
}

#[test]
fn terminal_level_proof_serde_round_trip() {
    let final_witness = CleartextWitnessProof::FieldElements(FlatRingVec::from_coeffs(vec![
        F::one(),
        -F::one(),
        F::zero(),
        F::from_u64(2),
    ]));

    let without_reduction = TerminalLevelProof::new_with_extension_opening_reduction(
        None,
        tiny_terminal_stage2(),
        final_witness.clone(),
        7,
    );
    assert!(without_reduction.extension_opening_reduction.is_none());
    assert!(without_reduction
        .shape()
        .extension_opening_reduction
        .is_none());
    assert_eq!(
        AkitaBatchedRootProof::new_terminal(without_reduction.clone())
            .fold_grind_nonce()
            .expect("terminal root has fold nonce"),
        7
    );

    let mut bytes = Vec::new();
    without_reduction
        .serialize_uncompressed(&mut bytes)
        .expect("serialize terminal proof without extension-opening reduction");
    assert_eq!(bytes.len(), without_reduction.serialized_size(Compress::No));

    let decoded =
        TerminalLevelProof::<F, F>::deserialize_uncompressed(&*bytes, &without_reduction.shape())
            .expect("deserialize terminal proof without extension-opening reduction");
    assert_eq!(decoded, without_reduction);

    let with_reduction = TerminalLevelProof::new_with_extension_opening_reduction(
        Some(tiny_reduction()),
        tiny_terminal_stage2(),
        final_witness,
        0,
    );
    let mut bytes_with_reduction = Vec::new();
    with_reduction
        .serialize_uncompressed(&mut bytes_with_reduction)
        .expect("serialize terminal proof with extension-opening reduction");
    let decoded_with_reduction = TerminalLevelProof::<F, F>::deserialize_uncompressed(
        &*bytes_with_reduction,
        &with_reduction.shape(),
    )
    .expect("deserialize terminal proof with extension-opening reduction");
    assert_eq!(decoded_with_reduction, with_reduction);

    with_reduction
        .shape()
        .check()
        .expect("terminal shape with reduction passes Valid::check()");
}
