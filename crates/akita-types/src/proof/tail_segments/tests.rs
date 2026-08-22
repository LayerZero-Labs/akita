use super::*;
use crate::SisModulusProfileId;
use akita_challenges::SparseChallengeConfig;
use jolt_field::CanonicalEncoding;
use jolt_field::{One, Prime128OffsetA7F7, Zero};

type F = Prime128OffsetA7F7;
const TEST_ADMISSION_CAP: u128 = 127;

fn test_lp() -> CommittedGroupParams {
    let mut params = CommittedGroupParams::params_only(
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
        policy: params.inner().matrix.security_policy(),
        table_digest: params
            .inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        modulus_profile: params.inner().matrix.sis_modulus_profile(),
        role: crate::sis::SisMatrixRole::Inner,
        ring_dimension: 64,
        coeff_linf_bound: *crate::sis::COEFF_LINF_BUCKETS
            .last()
            .expect("nonempty SIS buckets"),
    };
    params.own_group_mut().profile.inner.matrix =
        crate::sis::InnerCommitMatrixParams::try_new_with_min_rank(
            key,
            params.inner().matrix.input_width(),
        )
        .expect("secure terminal test matrix");
    params
}

fn scalar_group_layout(
    lp: &CommittedGroupParams,
    num_w_vectors: usize,
    num_t_vectors: usize,
    num_z_segments: usize,
    field_bits: u32,
) -> Result<TailSegmentLayout, AkitaError> {
    TerminalResponseShape::from_groups(
        lp,
        field_bits,
        [(
            lp.final_group_scalar().expect("scalar final group"),
            num_w_vectors,
            num_t_vectors,
            num_z_segments,
            TEST_ADMISSION_CAP,
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
fn terminal_decoder_uses_one_coding_and_linf_cap() {
    let cap = 7;
    let values = [6, -6];
    let rice_low_bits = wire_rice_low_bits(cap);
    let zigzag_w = golomb_rice_zigzag_width(cap);
    let payload = golomb_rice_encode_vec(&values, rice_low_bits, zigzag_w).unwrap();
    let group = TailSegmentGroupLayout {
        z_coords: values.len(),
        e_field_elems: 0,
        t_field_elems: 0,
        z_linf_cap: Some(cap),
        z_rice_low_bits: rice_low_bits,
        z_payload_bytes: payload.len(),
    };
    assert_eq!(
        decode_terminal_z_golomb_payload(&payload, &group).unwrap(),
        values.map(|value| value as i16)
    );
}

#[test]
fn terminal_decoder_without_linf_cap_uses_only_the_signed_wire_range() {
    let values = [1_000, -1_000];
    let wire_abs_bound = i16::MAX as u128;
    let rice_low_bits = 7;
    let zigzag_w = golomb_rice_zigzag_width(wire_abs_bound);
    let payload = golomb_rice_encode_vec(&values, rice_low_bits, zigzag_w).unwrap();
    let group = TailSegmentGroupLayout {
        z_coords: values.len(),
        e_field_elems: 0,
        t_field_elems: 0,
        z_linf_cap: None,
        z_rice_low_bits: rice_low_bits,
        z_payload_bytes: payload.len(),
    };
    assert_eq!(
        decode_terminal_z_golomb_payload(&payload, &group).unwrap(),
        values.map(|value| value as i16)
    );
}

#[test]
fn terminal_decoder_rejects_coefficient_outside_i16() {
    let cap = u128::from(u16::MAX);
    let value = i64::from(i16::MAX) + 1;
    let rice_low_bits = wire_rice_low_bits(cap);
    let zigzag_w = golomb_rice_zigzag_width(cap);
    let payload = golomb_rice_encode_vec(&[value], rice_low_bits, zigzag_w).unwrap();
    let group = TailSegmentGroupLayout {
        z_coords: 1,
        e_field_elems: 0,
        t_field_elems: 0,
        z_linf_cap: Some(cap),
        z_rice_low_bits: rice_low_bits,
        z_payload_bytes: payload.len(),
    };
    assert!(decode_terminal_z_golomb_payload(&payload, &group).is_err());
}

#[test]
fn terminal_response_z_budget_uses_golomb_rate_not_packed_digit_width() {
    let lp = test_lp();
    let field_bits = F::MODULUS_BITS;
    let cap = 31;
    let layout = TerminalResponseShape::from_groups(
        &lp,
        field_bits,
        [(
            lp.final_group_scalar().expect("scalar final group"),
            1usize,
            1usize,
            1usize,
            cap,
        )],
    )
    .unwrap()
    .layout;
    let z_bytes = terminal_response_z_payload_bytes(&layout);
    let group = layout.groups[0];
    assert_eq!(z_bytes, z_payload_budget_from_cap(group.z_coords, cap));
    let depth_fold = lp.num_digits_fold();
    let packed_z = crate::layout::proof_size::packed_digits_bytes(
        group.z_coords.saturating_mul(depth_fold),
        8,
    );
    assert_ne!(z_bytes, packed_z);
}

#[test]
fn direct_terminal_layout_contains_only_z_e_t_planes() {
    let lp = test_lp();
    let field_bits = F::MODULUS_BITS;
    let layout = TerminalResponseShape::from_groups(
        &lp,
        field_bits,
        [(
            lp.final_group_scalar().expect("scalar final group"),
            1usize,
            1usize,
            1usize,
            TEST_ADMISSION_CAP,
        )],
    )
    .expect("direct terminal layout")
    .layout;
    assert_eq!(layout.groups.len(), 1);
    assert_eq!(layout.logical_num_elems % lp.d_a(), 0);
}

#[test]
fn direct_terminal_builder_constructs_z_e_t_segments() {
    let lp = test_lp();
    let field_bits = F::MODULUS_BITS;
    let layout = TerminalResponseShape::from_groups(
        &lp,
        field_bits,
        [(
            lp.final_group_scalar().expect("scalar final group"),
            1usize,
            1usize,
            1usize,
            TEST_ADMISSION_CAP,
        )],
    )
    .expect("direct terminal layout")
    .layout;
    let group_layout = layout.groups[0];
    let e_folded = RingVec::from_coeffs(vec![F::zero(); group_layout.e_field_elems]);
    let recomposed_inner_rows = RingVec::from_coeffs(vec![F::zero(); group_layout.t_field_elems]);
    let z_folded_centered_flat = vec![0i32; group_layout.z_coords];
    let group = TerminalResponseGroupParts {
        params: lp.final_group_scalar().expect("scalar final group"),
        num_w_vectors: 1,
        num_t_vectors: 1,
        num_z_segments: 1,
        e_folded: &e_folded,
        recomposed_inner_rows: &recomposed_inner_rows,
        z_folded_centered_flat: &z_folded_centered_flat,
    };
    let scheduled_shape = TerminalResponseShape {
        layout: layout.clone(),
    };
    let witness = build_terminal_response_from_groups(lp.d_a(), &[group], &lp, &scheduled_shape)
        .expect("direct terminal witness");

    assert_eq!(witness.layout, layout);
}

#[test]
fn terminal_response_wire_round_trip_with_scheduled_z_budget() {
    use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};
    use jolt_field::CanonicalEncoding;

    let lp = test_lp();
    let field_bits = F::MODULUS_BITS;
    let layout = scalar_group_layout(&lp, 1, 1, 1, field_bits).unwrap();
    let scheduled_z_bytes = terminal_response_z_payload_bytes(&layout);
    assert!(
        scheduled_z_bytes > 16,
        "test expects scheduled z budget to exceed a tight payload"
    );
    let group = layout.groups[0];
    let rice_low_bits = group.z_rice_low_bits;
    let zigzag_w_z = golomb_rice_zigzag_width(group.z_linf_cap.expect("Linf fixture cap"));
    let centered = [[-3i32, 0, 1, 2, -1, 4, 0, 0]; 2];
    let z_payload = test_support::encode_z_segment_from_centered(
        &centered,
        1,
        lp.inner().digits.num_digits,
        rice_low_bits,
        zigzag_w_z,
    )
    .unwrap();
    assert!(z_payload.len() < scheduled_z_bytes);
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
    let layout = scalar_group_layout(&lp, 1, 1, 1, F::MODULUS_BITS).unwrap();
    let group = layout.groups[0];
    let e_fields = RingVec::from_coeffs(
        (0..group.e_field_elems)
            .map(|index| F::from_u128_reduced(index as u128 + 1))
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
    let layout = scalar_group_layout(&lp, 1, 1, 1, F::MODULUS_BITS).unwrap();
    let group = layout.groups[0];
    let t_fields = RingVec::from_coeffs(
        (0..group.t_field_elems)
            .map(|index| F::from_u128_reduced(index as u128 + 9))
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

    let cap = TEST_ADMISSION_CAP;
    let rice_low_bits = wire_rice_low_bits(cap);
    let zigzag_w = golomb_rice_zigzag_width(cap);
    // For cap = 2^k - 1, the signed zigzag width admits -2^k but not +2^k.
    // Use the representable negative endpoint to reach the decoder's explicit
    // magnitude check without weakening the production encoder.
    let over_cap = -(cap as i64) - 1;
    let payload =
        golomb_rice_encode_vec(&[over_cap], rice_low_bits, zigzag_w).expect("zigzag covers -cap-1");
    let group = TailSegmentGroupLayout {
        z_coords: 1,
        e_field_elems: 0,
        t_field_elems: 0,
        z_linf_cap: Some(cap),
        z_rice_low_bits: rice_low_bits,
        z_payload_bytes: payload.len(),
    };
    assert!(decode_terminal_z_golomb_payload(&payload, &group).is_err());
}

#[test]
fn decode_terminal_z_rejects_trailing_zero_byte_padding() {
    use crate::golomb_rice::golomb_rice_encode_vec;

    let cap = TEST_ADMISSION_CAP;
    let rice_low_bits = wire_rice_low_bits(cap);
    let zigzag_w = golomb_rice_zigzag_width(cap);
    let mut payload = golomb_rice_encode_vec(&[-2i64, 1, 0], rice_low_bits, zigzag_w).unwrap();
    payload.push(0x00);
    let group = TailSegmentGroupLayout {
        z_coords: 3,
        e_field_elems: 0,
        t_field_elems: 0,
        z_linf_cap: Some(cap),
        z_rice_low_bits: rice_low_bits,
        z_payload_bytes: payload.len(),
    };
    assert!(decode_terminal_z_golomb_payload(&payload, &group).is_err());
}

#[test]
fn terminal_layout_validation_rejects_overflow_without_panicking() {
    let layout = TailSegmentLayout {
        ring_dimension: 64,
        groups: vec![
            TailSegmentGroupLayout {
                z_coords: 1,
                e_field_elems: usize::MAX,
                t_field_elems: 1,
                z_linf_cap: Some(1),
                z_payload_bytes: 1,
                z_rice_low_bits: 0,
            },
            TailSegmentGroupLayout {
                z_coords: 1,
                e_field_elems: 1,
                t_field_elems: usize::MAX,
                z_linf_cap: Some(1),
                z_payload_bytes: usize::MAX,
                z_rice_low_bits: 0,
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

/// Terminal A matrix pinned to one audited coefficient bucket.
///
/// The default [`test_lp`] fixture uses the largest bucket, whose certified
/// capacity is far above the terminal wire limit, so it can only exercise the
/// clamp. A small bucket puts the SIS bound in charge instead.
fn terminal_matrix_with_bucket(bucket: u128) -> crate::sis::InnerCommitMatrixParams {
    let base = test_lp();
    let key = crate::sis::SisTableKey {
        policy: base.inner().matrix.security_policy(),
        table_digest: base
            .inner()
            .matrix
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        modulus_profile: base.inner().matrix.sis_modulus_profile(),
        role: crate::sis::SisMatrixRole::Inner,
        ring_dimension: 64,
        coeff_linf_bound: bucket,
    };
    crate::sis::InnerCommitMatrixParams::try_new_with_min_rank(
        key,
        base.inner().matrix.input_width(),
    )
    .expect("terminal test matrix for the requested bucket")
}

#[test]
fn certified_terminal_cap_applies_the_wire_representation_limit() {
    let lp = test_lp();
    let raw = crate::sis::max_response_linf_for_role_a_collision(
        lp.inner()
            .matrix
            .coeff_linf_bound()
            .expect("L infinity route"),
        crate::sis::FoldChallengeNorms::new(&lp.fold_challenge_config()).l1_norm,
    )
    .expect("raw SIS capacity");
    assert!(
        raw > crate::sis::TERMINAL_RESPONSE_WIRE_LINF_LIMIT,
        "fixture must exercise the clamp; raw capacity was {raw}"
    );
    let cap = crate::sis::certified_terminal_response_linf_cap(
        &lp.inner().matrix,
        &lp.fold_challenge_config(),
    )
    .expect("certified terminal cap");
    assert_eq!(
        cap,
        crate::sis::TERMINAL_RESPONSE_WIRE_LINF_LIMIT,
        "a cap the terminal z wire cannot encode is not a usable cap"
    );
}

#[test]
fn certified_terminal_cap_is_priced_by_the_supplied_challenge_family() {
    let matrix = terminal_matrix_with_bucket(2047);
    let light = crate::sis::certified_terminal_response_linf_cap(
        &matrix,
        &SparseChallengeConfig::pm1_only(3),
    )
    .expect("light challenge cap");
    let heavy = crate::sis::certified_terminal_response_linf_cap(
        &matrix,
        &SparseChallengeConfig::pm1_only(6),
    )
    .expect("heavy challenge cap");
    assert!(
        light < crate::sis::TERMINAL_RESPONSE_WIRE_LINF_LIMIT,
        "bucket must leave the SIS bound in charge; got {light}"
    );
    assert!(
        heavy < light,
        "a heavier challenge family must price the same matrix more conservatively: {heavy} vs {light}"
    );
}

#[test]
fn terminal_cap_has_exactly_one_implementation() {
    // The schedule-side method must not re-derive the cap. If these ever
    // disagree the split-brain this consolidation removed has returned.
    for bucket in [2047u128, 8191, 67_108_863] {
        let matrix = terminal_matrix_with_bucket(bucket);
        for weight in [3usize, 6, 11] {
            let sparse = SparseChallengeConfig::pm1_only(weight);
            let mut lp = test_lp();
            lp.own_group_mut().profile.inner.matrix = matrix;
            lp.own_group_mut().opening.fold_challenge_config = sparse;
            let terminal = crate::TerminalFoldParams::from_expanded_group(lp);
            assert_eq!(
                terminal
                    .certified_response_linf_cap()
                    .expect("schedule-side cap"),
                crate::sis::certified_terminal_response_linf_cap(&matrix, &sparse)
                    .expect("single-authority cap"),
                "bucket {bucket}, challenge weight {weight}"
            );
        }
    }
}

#[path = "test_support.rs"]
mod test_support;
