use super::*;
use crate::SisModulusProfileId;
use akita_challenges::SparseChallengeConfig;
use akita_field::CanonicalField;
use akita_field::Prime128OffsetA7F7;

type F = Prime128OffsetA7F7;

fn test_lp() -> LevelParams {
    let mut params = LevelParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(8, 32, 2, 3, 3)
    .expect("tail segment test params");
    let key = crate::sis::SisTableKey {
        policy: params.a_key.security_policy(),
        table_digest: params.a_key.sis_table_key().table_digest,
        modulus_profile: params.a_key.sis_modulus_profile(),
        role: crate::sis::SisMatrixRole::A,
        ring_dimension: 64,
        coeff_linf_bound: *crate::sis::COEFF_LINF_BUCKETS
            .last()
            .expect("nonempty SIS buckets"),
    };
    params.a_key = crate::sis::AjtaiKeyParams::try_new_with_min_rank(key, params.a_key.col_len())
        .expect("secure terminal test matrix");
    params
}

fn scalar_group_layout(
    lp: &LevelParams,
    num_w_vectors: usize,
    num_t_vectors: usize,
    num_z_segments: usize,
    field_bits: u32,
) -> Result<TailSegmentLayout, AkitaError> {
    TerminalResponseShape::from_groups(
        lp,
        field_bits,
        [(
            lp as &dyn LevelParamsLike,
            num_w_vectors,
            num_t_vectors,
            num_z_segments,
        )],
    )
    .map(|shape| shape.layout)
}

#[test]
fn recompose_and_split_digits_round_trip() {
    let digits = vec![-2i8, 1, 0];
    let value = test_support::recompose_balanced_i8_digits(&digits, 3);
    let back = test_support::balanced_digits_from_i64(value, digits.len(), 3);
    assert_eq!(back, digits);
}

#[test]
fn terminal_decoder_separates_coding_scale_from_sis_admission() {
    let coding_scale = 7;
    let admissible_cap = 63;
    let values = [20, -20];
    let (rice_low_bits, zigzag_w) =
        tail_golomb_rice_z_params_from_caps(coding_scale, admissible_cap).unwrap();
    let payload = golomb_rice_encode_vec(&values, rice_low_bits, zigzag_w).unwrap();
    assert_eq!(
        decode_terminal_z_golomb_payload(
            &payload,
            values.len(),
            coding_scale,
            admissible_cap,
            None,
        )
        .unwrap(),
        values
    );
    assert!(decode_terminal_z_golomb_payload(
        &payload,
        values.len(),
        coding_scale,
        coding_scale,
        None,
    )
    .is_err());
}

#[test]
fn terminal_response_z_budget_uses_golomb_rate_not_packed_digit_width() {
    let lp = test_lp();
    let field_bits = F::modulus_bits();
    let layout = scalar_group_layout(&lp, 1, 1, 1, field_bits).unwrap();
    let z_bytes = terminal_response_z_payload_bytes(&lp, &layout, 1).unwrap();
    let group = layout.groups[0];
    let depth_fold = lp.num_digits_fold(1, field_bits).unwrap();
    let packed_z = crate::layout::proof_size::packed_digits_bytes(
        group.z_coords.saturating_mul(depth_fold),
        8,
    );
    assert_ne!(z_bytes, packed_z);
}

#[test]
fn direct_terminal_layout_contains_only_z_e_t_planes() {
    let lp = test_lp();
    let field_bits = F::modulus_bits();
    let layout = TerminalResponseShape::from_groups(
        &lp,
        field_bits,
        [(&lp as &dyn LevelParamsLike, 1usize, 1usize, 1usize)],
    )
    .expect("direct terminal layout")
    .layout;
    assert_eq!(layout.groups.len(), 1);
    assert_eq!(layout.logical_num_elems % lp.ring_dimension, 0);
}

#[test]
fn direct_terminal_builder_constructs_z_e_t_segments() {
    let lp = test_lp();
    let field_bits = F::modulus_bits();
    let layout = TerminalResponseShape::from_groups(
        &lp,
        field_bits,
        [(&lp as &dyn LevelParamsLike, 1usize, 1usize, 1usize)],
    )
    .expect("direct terminal layout")
    .layout;
    let group_layout = layout.groups[0];
    let e_folded = RingVec::from_coeffs(vec![F::zero(); group_layout.e_field_elems]);
    let recomposed_inner_rows = vec![RingVec::from_coeffs(vec![
        F::zero();
        group_layout.t_field_elems
    ])];
    let z_folded_centered_flat = vec![0i32; group_layout.z_coords];
    let group = TerminalResponseGroupParts {
        params: &lp,
        num_w_vectors: 1,
        num_t_vectors: 1,
        num_z_segments: 1,
        e_folded: &e_folded,
        recomposed_inner_rows: &recomposed_inner_rows,
        z_folded_centered_flat: &z_folded_centered_flat,
    };
    let witness = build_terminal_response_from_groups(lp.ring_dimension, &[group], &lp)
        .expect("direct terminal witness");

    assert_eq!(witness.layout, layout);
}

#[test]
fn terminal_golomb_grind_covers_terminal_layout() {
    let lp = test_lp();
    let layout = scalar_group_layout(&lp, 1, 1, 1, F::modulus_bits()).unwrap();
    let shape = TerminalResponseShape { layout };

    let terminal = terminal_golomb_grind_tail_t_vectors(
        &lp,
        RelationMatrixRowLayout::WithoutCommitmentBlocks,
        Some(&shape),
    )
    .unwrap();
    assert!(terminal.is_some());
    assert_eq!(
        terminal_golomb_grind_tail_t_vectors(
            &lp,
            RelationMatrixRowLayout::WithDBlock,
            Some(&shape),
        )
        .unwrap(),
        None
    );
}

#[test]
fn terminal_response_wire_round_trip_with_scheduled_z_budget() {
    use akita_field::CanonicalField;
    use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};

    let lp = test_lp();
    let field_bits = F::modulus_bits();
    let layout = scalar_group_layout(&lp, 1, 1, 1, field_bits).unwrap();
    let scheduled_z_bytes = terminal_response_z_payload_bytes(&lp, &layout, 1).unwrap();
    assert!(
        scheduled_z_bytes > 16,
        "test expects scheduled z budget to exceed a tight payload"
    );
    let (rice_low_bits, zigzag_w_z) = tail_golomb_rice_z_params(&lp, 1).unwrap();
    let centered = [[-3i32, 0, 1, 2, -1, 4, 0, 0]; 2];
    let z_payload = test_support::encode_z_segment_from_centered(
        &centered,
        1,
        lp.num_digits_inner,
        rice_low_bits,
        zigzag_w_z,
    )
    .unwrap();
    assert!(z_payload.len() < scheduled_z_bytes);
    let group = layout.groups[0];
    let witness = TerminalResponse {
        layout: layout.clone(),
        z_payloads: vec![z_payload],
        e_fields: RingVec::from_coeffs(vec![F::zero(); group.e_field_elems]),
        t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
    };
    let scheduled_shape = TerminalResponseShape { layout };
    let mut bytes = Vec::new();
    witness
        .serialize_with_mode(&mut bytes, Compress::No)
        .expect("serialize segment witness");
    let decoded = TerminalResponse::<F>::deserialize_with_mode(
        &bytes[..],
        Compress::No,
        Validate::Yes,
        &scheduled_shape,
    )
    .expect("deserialize with scheduled z budget");
    assert_eq!(decoded, witness);
}

#[test]
fn terminal_e_absorb_matches_emitted_field_segment() {
    let lp = test_lp();
    let layout = scalar_group_layout(&lp, 1, 1, 1, F::modulus_bits()).unwrap();
    let group = layout.groups[0];
    let e_fields = RingVec::from_coeffs(
        (0..group.e_field_elems)
            .map(|index| F::from_canonical_u128_reduced(index as u128 + 1))
            .collect(),
    );
    let witness = TerminalResponse {
        layout: layout.clone(),
        z_payloads: vec![vec![0]],
        e_fields: e_fields.clone(),
        t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
    };

    assert_eq!(
        witness.terminal_transcript_parts().unwrap().e_folded,
        raw_field_segment_bytes(&e_fields).unwrap(),
    );
}

#[test]
fn terminal_transcript_parts_separate_t_state_from_z_response() {
    let lp = test_lp();
    let layout = scalar_group_layout(&lp, 1, 1, 1, F::modulus_bits()).unwrap();
    let group = layout.groups[0];
    let t_fields = RingVec::from_coeffs(
        (0..group.t_field_elems)
            .map(|index| F::from_canonical_u128_reduced(index as u128 + 9))
            .collect(),
    );
    let z = vec![3, 1, 4, 1];
    let witness = TerminalResponse {
        layout,
        z_payloads: vec![z.clone()],
        e_fields: RingVec::from_coeffs(vec![F::one(); group.e_field_elems]),
        t_fields: t_fields.clone(),
    };

    let parts = witness.terminal_transcript_parts().unwrap();
    assert_eq!(parts.response, z);
}

#[test]
fn decode_terminal_z_rejects_coefficient_above_fold_cap() {
    use crate::golomb_rice::golomb_rice_encode_vec;

    let lp = test_lp();
    let cap = lp.fold_witness_linf_cap_for_claims(1).unwrap();
    let (rice_low_bits, zigzag_w) = tail_golomb_rice_z_params(&lp, 1).unwrap();
    let over_cap = cap as i64 + 1;
    let payload =
        golomb_rice_encode_vec(&[over_cap], rice_low_bits, zigzag_w).expect("zigzag covers cap+1");
    assert!(decode_terminal_z_golomb_payload(&payload, 1, cap, cap, None,).is_err());
}

#[test]
fn decode_terminal_z_rejects_trailing_zero_byte_padding() {
    use crate::golomb_rice::golomb_rice_encode_vec;

    let lp = test_lp();
    let (rice_low_bits, zigzag_w) = tail_golomb_rice_z_params(&lp, 1).unwrap();
    let mut payload = golomb_rice_encode_vec(&[-2i64, 1, 0], rice_low_bits, zigzag_w).unwrap();
    payload.push(0x00);
    let cap = lp.fold_witness_linf_cap_for_claims(1).unwrap();
    assert!(decode_terminal_z_golomb_payload(&payload, 3, cap, cap, None,).is_err());
}

#[test]
fn terminal_layout_validation_rejects_overflow_without_panicking() {
    let layout = TailSegmentLayout {
        ring_dimension: 64,
        log_basis_open: 6,
        groups: vec![
            TailSegmentGroupLayout {
                z_coords: 1,
                e_field_elems: usize::MAX,
                t_field_elems: 1,
                z_payload_bytes: 1,
            },
            TailSegmentGroupLayout {
                z_coords: 1,
                e_field_elems: 1,
                t_field_elems: usize::MAX,
                z_payload_bytes: usize::MAX,
            },
        ],
        logical_num_elems: 1,
    };
    let result = std::panic::catch_unwind(|| layout.check());
    assert!(result.is_ok(), "malformed proof shape must not panic");
    assert!(result.unwrap().is_err());
}

#[test]
fn terminal_layout_decode_rejects_oversized_group_count_before_allocation() {
    use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};

    let mut bytes = Vec::new();
    64usize
        .serialize_with_mode(&mut bytes, Compress::No)
        .unwrap();
    6u32.serialize_with_mode(&mut bytes, Compress::No).unwrap();
    (super::super::MAX_PROOF_SHAPE_SEQUENCE_LEN as u64 + 1)
        .serialize_with_mode(&mut bytes, Compress::No)
        .unwrap();
    let err =
        TailSegmentLayout::deserialize_with_mode(&bytes[..], Compress::No, Validate::Yes, &())
            .expect_err("oversized terminal group vector must be rejected");
    assert!(matches!(
        err,
        SerializationError::LengthLimitExceeded { .. }
    ));
}

#[path = "test_support.rs"]
mod test_support;
