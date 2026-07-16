use super::*;
use crate::SisModulusFamily;
use akita_challenges::SparseChallengeConfig;
use jolt_field::CanonicalField;
use jolt_field::Prime128OffsetA7F7;

type F = Prime128OffsetA7F7;

fn test_lp() -> LevelParams {
    LevelParams::params_only(
        SisModulusFamily::Q128,
        8,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(3, 2, 2, 3, 0)
    .expect("tail segment test params")
}

fn scalar_group_layout(
    lp: &LevelParams,
    num_w_vectors: usize,
    num_t_vectors: usize,
    num_z_segments: usize,
    num_commitment_groups: usize,
    field_bits: u32,
) -> Result<TailSegmentLayout, AkitaError> {
    tail_segment_layout_from_groups(
        lp,
        [(
            lp as &dyn LevelParamsLike,
            num_w_vectors,
            num_t_vectors,
            num_z_segments,
        )],
        num_commitment_groups,
        field_bits,
    )
}

#[test]
fn recompose_and_split_digits_round_trip() {
    let digits = vec![-2i8, 1, 0];
    let value = recompose_balanced_i8_digits(&digits, 3);
    let back = balanced_digits_from_i64(value, digits.len(), 3);
    assert_eq!(back, digits);
}

#[test]
fn segment_typed_z_budget_uses_golomb_rate_not_packed_digit_width() {
    let lp = test_lp();
    let field_bits = F::modulus_bits();
    let layout = scalar_group_layout(&lp, 1, 1, 1, 1, field_bits).unwrap();
    let z_bytes = segment_typed_z_payload_bytes(&lp, &layout, 1).unwrap();
    let group = layout.groups[0];
    let depth_fold = lp.num_digits_fold(1, field_bits).unwrap();
    let packed_z = crate::layout::proof_size::packed_digits_bytes(
        group.z_coords.saturating_mul(depth_fold),
        8,
    );
    assert_ne!(z_bytes, packed_z);
}

#[test]
fn segment_typed_wire_round_trip_with_scheduled_z_budget() {
    use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};
    use jolt_field::CanonicalField;

    let lp = test_lp();
    let field_bits = F::modulus_bits();
    let layout = scalar_group_layout(&lp, 1, 1, 1, 1, field_bits).unwrap();
    let scheduled_z_bytes = segment_typed_z_payload_bytes(&lp, &layout, 1).unwrap();
    assert!(
        scheduled_z_bytes > 16,
        "test expects scheduled z budget to exceed a tight payload"
    );
    let (rice_low_bits, zigzag_w_z) = tail_golomb_rice_z_params(&lp, 1).unwrap();
    let centered = [[-3i32, 0, 1, 2, -1, 4, 0, 0]; 2];
    let z_payload = encode_z_segment_from_centered(
        &centered,
        1,
        lp.num_digits_commit,
        rice_low_bits,
        zigzag_w_z,
    )
    .unwrap();
    assert!(z_payload.len() < scheduled_z_bytes);
    let group = layout.groups[0];
    let witness = SegmentTypedWitness {
        layout: layout.clone(),
        z_payloads: vec![z_payload],
        e_fields: RingVec::from_coeffs(vec![F::zero(); group.e_field_elems]),
        t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
        r_fields: RingVec::from_coeffs(vec![F::zero(); layout.r_field_elems]),
    };
    let scheduled_shape = SegmentTypedWitnessShape { layout };
    let mut bytes = Vec::new();
    witness
        .serialize_with_mode(&mut bytes, Compress::No)
        .expect("serialize segment witness");
    let decoded = SegmentTypedWitness::<F>::deserialize_with_mode(
        &bytes[..],
        Compress::No,
        Validate::Yes,
        &scheduled_shape,
    )
    .expect("deserialize with scheduled z budget");
    assert_eq!(decoded, witness);
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
    assert!(decode_terminal_z_golomb_payload(&payload, 1, &lp, 1, None).is_err());
}

#[test]
fn decode_terminal_z_rejects_trailing_zero_byte_padding() {
    use crate::golomb_rice::golomb_rice_encode_vec;

    let lp = test_lp();
    let (rice_low_bits, zigzag_w) = tail_golomb_rice_z_params(&lp, 1).unwrap();
    let mut payload = golomb_rice_encode_vec(&[-2i64, 1, 0], rice_low_bits, zigzag_w).unwrap();
    payload.push(0x00);
    assert!(decode_terminal_z_golomb_payload(&payload, 3, &lp, 1, None).is_err());
}

#[test]
fn expand_segment_typed_rejects_inadmissible_z_payload() {
    use crate::golomb_rice::golomb_rice_encode_vec;

    let lp = test_lp();
    let field_bits = F::modulus_bits();
    let layout = scalar_group_layout(&lp, 1, 1, 1, 1, field_bits).unwrap();
    let (rice_low_bits, zigzag_w) = tail_golomb_rice_z_params(&lp, 1).unwrap();
    let cap = lp.fold_witness_linf_cap_for_claims(1).unwrap();
    let z_payload = golomb_rice_encode_vec(&[cap as i64 + 1], rice_low_bits, zigzag_w).unwrap();
    let group = layout.groups[0];
    let witness = SegmentTypedWitness {
        layout: layout.clone(),
        z_payloads: vec![z_payload],
        e_fields: RingVec::from_coeffs(vec![F::zero(); group.e_field_elems]),
        t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
        r_fields: RingVec::from_coeffs(vec![F::zero(); layout.r_field_elems]),
    };
    assert!(expand_segment_typed_to_i8_digits::<8, F>(&witness, &lp, 1).is_err());
}
