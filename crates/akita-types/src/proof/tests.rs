use super::*;
use akita_algebra::CompressedUniPoly;
use akita_serialization::Valid;
use akita_sumcheck::SumcheckProof;
use akita_transcript::{labels, AkitaTranscript, Transcript};
use jolt_field::{One, Prime128Offset275, Prime128OffsetA7F7, Zero};
use rand::SeedableRng;

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

/// Local reproduction of the (deleted) typed `RingSliceSerializer`: serialize a
/// borrowed slice of ring elements with no length header, each ring element via
/// its own `serialize_with_mode`. This is the reference encoding the S4 flat
/// absorber must remain byte-identical to.
struct TypedRingSliceSerializer<'a, const D: usize>(&'a [CyclotomicRing<F, D>]);

impl<const D: usize> AkitaSerialize for TypedRingSliceSerializer<'_, D> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for ring in self.0 {
            ring.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.iter().map(|r| r.serialized_size(compress)).sum()
    }
}

/// Helper: absorb `ring_elems` via the legacy typed encoding (reproduced above)
/// and return the challenge bytes squeezed immediately afterwards.
fn typed_challenge<const D: usize>(
    ring_elems: &[CyclotomicRing<F, D>],
    label: &[u8],
    challenge_label: &[u8],
    challenge_len: usize,
) -> Vec<u8>
where
    F: CanonicalEncoding,
{
    let mut t = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    t.append_serde(label, &TypedRingSliceSerializer(ring_elems));
    t.challenge_bytes(challenge_label, challenge_len)
}

/// Helper: absorb the same ring elements via the D-free flat path and return
/// the challenge bytes squeezed immediately afterwards.
fn flat_challenge<const D: usize>(
    ring_elems: &[CyclotomicRing<F, D>],
    label: &[u8],
    challenge_label: &[u8],
    challenge_len: usize,
) -> Vec<u8>
where
    F: AkitaSerialize + CanonicalEncoding,
{
    let mut t = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let rv = RingVec::from_ring_elems(ring_elems);
    rv.append_flat_to_transcript(label, D, &mut t)
        .expect("well-formed flat absorption must succeed");
    t.challenge_bytes(challenge_label, challenge_len)
}

/// Prove that the D-free flat transcript absorber produces a byte-identical
/// transcript state to the legacy typed ring-slice encoding (reproduced by
/// `TypedRingSliceSerializer`), for D ∈ {32, 64, 128, 256} and a fixed number
/// of ring elements.
///
/// Both paths absorb the same field-element bytes in the same order (no
/// length header, coefficient-major within each ring element). The comparison
/// is via the first 64 challenge bytes squeezed after absorption — any
/// divergence in the absorbed stream would produce a different challenge.
#[test]
fn flat_absorption_byte_identical_to_typed() {
    const N_RINGS: usize = 3;
    const CHALLENGE_LABEL: &[u8] = b"test_challenge";
    const ABSORB_LABEL: &[u8] = b"commitment";
    const CHALLENGE_LEN: usize = 64;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0xdead_beef_cafe_1234);

    // D = 32
    {
        const D: usize = 32;
        let elems: Vec<CyclotomicRing<F, D>> = (0..N_RINGS)
            .map(|_| CyclotomicRing::<F, D>::random(&mut rng))
            .collect();
        let typed = typed_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        let flat = flat_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        assert_eq!(
            typed, flat,
            "D=32: flat absorption must be byte-identical to typed path"
        );
    }

    // D = 64
    {
        const D: usize = 64;
        let elems: Vec<CyclotomicRing<F, D>> = (0..N_RINGS)
            .map(|_| CyclotomicRing::<F, D>::random(&mut rng))
            .collect();
        let typed = typed_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        let flat = flat_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        assert_eq!(
            typed, flat,
            "D=64: flat absorption must be byte-identical to typed path"
        );
    }

    // D = 128
    {
        const D: usize = 128;
        let elems: Vec<CyclotomicRing<F, D>> = (0..N_RINGS)
            .map(|_| CyclotomicRing::<F, D>::random(&mut rng))
            .collect();
        let typed = typed_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        let flat = flat_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        assert_eq!(
            typed, flat,
            "D=128: flat absorption must be byte-identical to typed path"
        );
    }

    // D = 256
    {
        const D: usize = 256;
        let elems: Vec<CyclotomicRing<F, D>> = (0..N_RINGS)
            .map(|_| CyclotomicRing::<F, D>::random(&mut rng))
            .collect();
        let typed = typed_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        let flat = flat_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);
        assert_eq!(
            typed, flat,
            "D=256: flat absorption must be byte-identical to typed path"
        );
    }
}

/// Prove that the free-function form `append_flat_coefficients` also matches
/// the typed path, and that `RingView::append_flat_to_transcript` does too.
#[test]
fn flat_absorption_free_fn_and_ring_view_match_typed() {
    const D: usize = 64;
    const N_RINGS: usize = 4;
    const ABSORB_LABEL: &[u8] = b"commitment";
    const CHALLENGE_LABEL: &[u8] = b"ch";
    const CHALLENGE_LEN: usize = 32;

    let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234_5678_9abc_def0);

    let elems: Vec<CyclotomicRing<F, D>> = (0..N_RINGS)
        .map(|_| CyclotomicRing::<F, D>::random(&mut rng))
        .collect();

    // Typed reference.
    let typed = typed_challenge::<D>(&elems, ABSORB_LABEL, CHALLENGE_LABEL, CHALLENGE_LEN);

    // Free function `append_flat_coefficients`.
    let flat_coeffs: Vec<F> = elems
        .iter()
        .flat_map(|r| r.coefficients().iter().copied())
        .collect();
    let free_fn = {
        let mut t = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
        append_flat_coefficients(ABSORB_LABEL, &flat_coeffs, D, &mut t)
            .expect("free fn flat absorption must succeed");
        t.challenge_bytes(CHALLENGE_LABEL, CHALLENGE_LEN)
    };
    assert_eq!(
        typed, free_fn,
        "append_flat_coefficients must match typed path"
    );

    // `RingView::append_flat_to_transcript`.
    let ring_view = {
        let mut t = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
        let rv = RingVec::from_ring_elems(&elems);
        let view = rv.view().expect("ring_dim = D is valid");
        view.append_flat_to_transcript(ABSORB_LABEL, &mut t)
            .expect("ring view invariants hold in test");
        t.challenge_bytes(CHALLENGE_LABEL, CHALLENGE_LEN)
    };
    assert_eq!(
        typed, ring_view,
        "RingView::append_flat_to_transcript must match typed path"
    );
}
