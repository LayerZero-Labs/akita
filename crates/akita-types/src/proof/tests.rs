use super::wire::extension_opening_reduction_serialized_size;
use super::*;
use akita_algebra::CompressedUniPoly;
use akita_field::{Prime128Offset275, RandomSampling};
use akita_serialization::Valid;
use akita_sumcheck::SumcheckProof;
use akita_transcript::{labels, AkitaTranscript, Transcript};
use rand::SeedableRng;

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
        )),
        v_coeffs: 1,
        stage1_stages: Vec::new(),
        stage2_sumcheck_proof: Vec::new(),
        stage3_sumcheck: None,
        next_witness_binding: NextWitnessBindingShape::OuterCommitment { coeffs: 1 },
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

fn tiny_stage1() -> AkitaStage1Proof<F> {
    AkitaStage1Proof {
        stages: Vec::new(),
        s_claim: F::zero(),
    }
}

fn tiny_stage2<const D: usize>() -> AkitaStage2Proof<F, F> {
    AkitaStage2Proof {
        sumcheck_proof: SumcheckProof {
            round_polys: Vec::new(),
        },
        next_witness_binding: NextWitnessBinding::OuterCommitment(
            RingVec::from_ring_elems(&[CyclotomicRing::<F, D>::zero()]).into_compact(),
        ),
        next_w_eval: F::zero(),
    }
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
    let without_reduction = FoldLevelProof::new::<D>(
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
        FoldLevelProof::<F, F>::deserialize_uncompressed(&*bytes, &without_reduction.shape())
            .expect("deserialize proof without extension-opening reduction");
    assert!(decoded.extension_opening_reduction().is_none());
    assert_eq!(decoded, without_reduction);

    let with_reduction = FoldLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
        Some(tiny_reduction()),
        vec![CyclotomicRing::<F, D>::zero()],
        tiny_stage1(),
        SumcheckProof {
            round_polys: Vec::new(),
        },
        RingVec::from_ring_elems(&[CyclotomicRing::<F, D>::zero()]).into_compact(),
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
    let decoded_with_reduction = FoldLevelProof::<F, F>::deserialize_uncompressed(
        &*bytes_with_reduction,
        &with_reduction.shape(),
    )
    .expect("deserialize proof with extension-opening reduction");
    assert_eq!(decoded_with_reduction, with_reduction);
}

#[test]
fn terminal_inner_state_omits_outer_commitment_from_tag_free_proof_wire() {
    const D: usize = 8;
    let outer = FoldLevelProof::new::<D>(
        vec![CyclotomicRing::<F, D>::zero()],
        tiny_stage1(),
        tiny_stage2::<D>(),
    );
    let outer_commitment_bytes = outer
        .next_w_commitment()
        .expect("ordinary recursive edge carries an outer commitment")
        .serialized_size(Compress::No);

    let mut terminal_inner = outer.clone();
    terminal_inner.stage2_mut().next_witness_binding = NextWitnessBinding::TerminalInnerState;
    assert_eq!(terminal_inner.next_w_commitment(), None);
    assert_eq!(
        outer.serialized_size(Compress::No) - terminal_inner.serialized_size(Compress::No),
        outer_commitment_bytes,
        "the schedule-selected terminal-inner proof body must remove exactly the outer-u bytes"
    );

    let mut bytes = Vec::new();
    terminal_inner
        .serialize_uncompressed(&mut bytes)
        .expect("serialize terminal-inner edge without a proof-body tag");
    assert_eq!(bytes.len(), terminal_inner.serialized_size(Compress::No));

    let shape = terminal_inner.shape();
    assert_eq!(
        shape.next_witness_binding,
        NextWitnessBindingShape::TerminalInnerState
    );
    let decoded = FoldLevelProof::<F, F>::deserialize_uncompressed(&bytes[..], &shape)
        .expect("shape-driven deserialize terminal-inner edge");
    assert_eq!(decoded, terminal_inner);
}

#[test]
fn terminal_level_proof_serde_round_trip() {
    let final_witness = CleartextWitnessProof::FieldElements(RingVec::from_coeffs(vec![
        F::one(),
        -F::one(),
        F::zero(),
        F::from_u64(2),
    ]));

    let without_reduction =
        TerminalLevelProof::new_with_extension_opening_reduction(None, final_witness.clone(), 7);
    assert!(without_reduction.extension_opening_reduction.is_none());
    assert!(without_reduction
        .shape()
        .extension_opening_reduction
        .is_none());
    assert_eq!(without_reduction.fold_grind_nonce, 7);

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

#[test]
fn direct_terminal_relation_proof_serde_round_trip() {
    let final_witness =
        CleartextWitnessProof::FieldElements(RingVec::from_coeffs(vec![F::one(), -F::one()]));
    let proof = TerminalLevelProof {
        extension_opening_reduction: None,
        fold_grind_nonce: 3,
        final_witness,
    };

    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize direct terminal proof");
    assert_eq!(bytes.len(), proof.serialized_size(Compress::No));
    let decoded = TerminalLevelProof::<F, F>::deserialize_uncompressed(&bytes[..], &proof.shape())
        .expect("deserialize direct terminal proof");
    assert_eq!(decoded, proof);

    let mut shape_bytes = Vec::new();
    proof
        .shape()
        .serialize_uncompressed(&mut shape_bytes)
        .expect("serialize direct terminal shape");
    let decoded_shape = TerminalLevelProofShape::deserialize_uncompressed(&shape_bytes[..], &())
        .expect("deserialize direct terminal shape");
    assert_eq!(decoded_shape, proof.shape());
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
    F: CanonicalField,
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
    F: AkitaSerialize + CanonicalField,
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
