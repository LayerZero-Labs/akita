use akita_algebra::poly::Poly;
use akita_algebra::{CyclotomicRing, VectorModule};
use akita_field::{Fp32, Fp64, FpExt2, FpExt4, Prime128Offset275};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, SerializationError};

use super::fixtures::NR;

#[test]
fn serialization_round_trip_fp32() {
    type F = Fp32<251>;
    let val = F::from_u64(42);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F::deserialize_compressed(&buf[..], &()).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn serialization_round_trip_fp64() {
    type F = Fp64<4294967197>;
    let val = F::from_u64(123456789);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F::deserialize_compressed(&buf[..], &()).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn serialization_round_trip_fp128() {
    type F = Prime128Offset275;
    let val = F::from_u64(999999999);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F::deserialize_compressed(&buf[..], &()).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn serialization_round_trip_ext() {
    type F = Fp32<251>;
    type F2 = FpExt2<F, NR>;
    let val = F2::new(F::from_u64(3), F::from_u64(7));
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F2::deserialize_compressed(&buf[..], &()).unwrap();
    assert!(val == restored);
}

#[test]
fn serialization_round_trip_fp_ext4() {
    type F = Fp32<251>;
    type F4 = FpExt4<F>;

    let val = F4::new([
        F::from_u64(5),
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(9),
    ]);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = F4::deserialize_compressed(&buf[..], &()).unwrap();
    assert!(val == restored);
}

#[test]
fn serialization_round_trip_vector_module() {
    type F = Fp32<251>;
    let val = VectorModule::<F, 3>([F::from_u64(1), F::from_u64(2), F::from_u64(3)]);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = VectorModule::<F, 3>::deserialize_compressed(&buf[..], &()).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn serialization_round_trip_poly() {
    type F = Fp32<251>;

    let val = Poly::<F, 4>([
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(29),
    ]);
    let mut buf = Vec::new();
    val.serialize_compressed(&mut buf).unwrap();
    let restored = Poly::<F, 4>::deserialize_compressed(&buf[..], &()).unwrap();
    assert_eq!(val, restored);
}

#[test]
fn deserialize_checked_rejects_non_canonical_field_elements() {
    type F32 = Fp32<251>;
    let bad32 = 251u32.to_le_bytes();
    let err32 = F32::deserialize_compressed(&bad32[..], &()).unwrap_err();
    assert!(matches!(err32, SerializationError::InvalidData(_)));
    let unchecked32 = F32::deserialize_compressed_unchecked(&bad32[..], &()).unwrap();
    assert_eq!(unchecked32, F32::zero());

    type F64 = Fp64<4294967197>;
    let bad64 = 4294967197u64.to_le_bytes();
    let err64 = F64::deserialize_compressed(&bad64[..], &()).unwrap_err();
    assert!(matches!(err64, SerializationError::InvalidData(_)));
    let unchecked64 = F64::deserialize_compressed_unchecked(&bad64[..], &()).unwrap();
    assert_eq!(unchecked64, F64::zero());

    type F128 = Prime128Offset275;
    const P275: u128 = 0xfffffffffffffffffffffffffffffeedu128;
    let bad128 = P275.to_le_bytes();
    let err128 = F128::deserialize_compressed(&bad128[..], &()).unwrap_err();
    assert!(matches!(err128, SerializationError::InvalidData(_)));
    let unchecked128 = F128::deserialize_compressed_unchecked(&bad128[..], &()).unwrap();
    assert_eq!(unchecked128, F128::zero());

    type F128b = Prime128Offset275;
    const P275B: u128 = 0xfffffffffffffffffffffffffffffeedu128;
    let bad275 = P275B.to_le_bytes();
    let err275 = F128b::deserialize_compressed(&bad275[..], &()).unwrap_err();
    assert!(matches!(err275, SerializationError::InvalidData(_)));
    let unchecked275 = F128b::deserialize_compressed_unchecked(&bad275[..], &()).unwrap();
    assert_eq!(unchecked275, F128b::zero());
}

#[test]
fn cyclotomic_ring_serialization_round_trip() {
    type F = Fp32<251>;
    type R = CyclotomicRing<F, 4>;

    let a = R::from_coefficients([
        F::from_u64(3),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(42),
    ]);
    let mut buf = Vec::new();
    a.serialize_compressed(&mut buf).unwrap();
    let restored = R::deserialize_compressed(&buf[..], &()).unwrap();
    assert_eq!(a, restored);
}
