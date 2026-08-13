//! `serde` for the prime fields, as fixed-width canonical little-endian bytes.
//!
//! An element encodes as the little-endian bytes of its canonical representative
//! (`Fp32` → `[u8; 4]`, `Fp64` → `[u8; 8]`, `Fp128` → `[u8; 16]`); decode rejects
//! non-canonical values (`val >= P`) rather than reducing. Encoding the byte
//! array rather than the storage integer keeps the length value-independent
//! under formats that varint-encode integers, matches Jolt's `JoltProof`
//! convention, and agrees with
//! [`AkitaSerialize`](akita_serialization::AkitaSerialize) for these types.
//!
//! This is a host and tooling surface. Verifier-reachable decoding of Akita
//! containers stays on
//! [`AkitaDeserialize`](akita_serialization::AkitaDeserialize), which is the
//! only path that bounds container lengths.

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use super::{Fp128, Fp32, Fp64};
use crate::CanonicalField;

macro_rules! impl_prime_serde {
    ($ty:ident<$p:ident: $p_ty:ty>) => {
        impl<const $p: $p_ty> Serialize for $ty<$p> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                (self.to_canonical_u128() as $p_ty)
                    .to_le_bytes()
                    .serialize(serializer)
            }
        }

        impl<'de, const $p: $p_ty> Deserialize<'de> for $ty<$p> {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let bytes = <[u8; size_of::<$p_ty>()]>::deserialize(deserializer)?;
                let raw = <$p_ty>::from_le_bytes(bytes);
                Self::from_canonical_u128_checked(u128::from(raw)).ok_or_else(|| {
                    de::Error::custom(format_args!(
                        concat!(stringify!($ty), " value {} is not a canonical residue"),
                        raw
                    ))
                })
            }
        }
    };
}

impl_prime_serde!(Fp32<P: u32>);
impl_prime_serde!(Fp64<P: u64>);
impl_prime_serde!(Fp128<P: u128>);

#[cfg(test)]
mod tests {
    use akita_serialization::AkitaSerialize;
    use serde::de::DeserializeOwned;

    use super::*;

    type F32 = Fp32<4_294_967_291>;
    type F64 = Fp64<18_446_744_073_709_551_557>;
    type F128 = Fp128<0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feed>;

    fn encode<T: Serialize>(value: T) -> Vec<u8> {
        postcard::to_stdvec(&value).expect("encoding cannot fail")
    }

    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    fn fp32(value: u128) -> F32 {
        F32::from_canonical_u128_checked(value).expect("value is canonical")
    }

    fn fp64(value: u128) -> F64 {
        F64::from_canonical_u128_checked(value).expect("value is canonical")
    }

    fn fp128(value: u128) -> F128 {
        F128::from_canonical_u128_checked(value).expect("value is canonical")
    }

    #[test]
    fn fp32_encodes_as_four_canonical_le_bytes() {
        assert_eq!(encode(fp32(200)), [0xc8, 0x00, 0x00, 0x00]);
        assert_eq!(encode(fp32(4_294_967_290)), [0xfa, 0xff, 0xff, 0xff]);
    }

    #[test]
    fn fp64_encodes_as_eight_canonical_le_bytes() {
        assert_eq!(
            encode(fp64(200)),
            [0xc8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
        assert_eq!(
            encode(fp64(18_446_744_073_709_551_556)),
            [0xc4, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]
        );
    }

    #[test]
    fn fp128_encodes_as_sixteen_canonical_le_bytes() {
        let mut small = [0x00; 16];
        small[0] = 0xc8;
        assert_eq!(encode(fp128(200)), small);

        let mut large = [0xff; 16];
        large[0] = 0xec;
        large[1] = 0xfe;
        assert_eq!(
            encode(fp128(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feec)),
            large
        );
    }

    #[test]
    fn prime_fields_round_trip() {
        for value in [0, 1, 200, 4_294_967_290] {
            assert_eq!(decode::<F32>(&encode(fp32(value))).unwrap(), fp32(value));
        }
        for value in [0, 1, 200, 4_294_967_296, 18_446_744_073_709_551_556] {
            assert_eq!(decode::<F64>(&encode(fp64(value))).unwrap(), fp64(value));
        }
        for value in [
            0,
            1,
            200,
            18_446_744_073_709_551_616,
            0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feec,
        ] {
            assert_eq!(decode::<F128>(&encode(fp128(value))).unwrap(), fp128(value));
        }
    }

    #[test]
    fn non_canonical_and_truncated_encodings_are_rejected() {
        assert!(decode::<F32>(&encode(4_294_967_291u32.to_le_bytes())).is_err());
        assert!(decode::<F64>(&encode(18_446_744_073_709_551_557u64.to_le_bytes())).is_err());
        assert!(decode::<F128>(&encode(
            0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feedu128.to_le_bytes()
        ))
        .is_err());
        assert!(decode::<F128>(&encode(u128::MAX.to_le_bytes())).is_err());
        assert!(decode::<F128>(&encode(fp128(200))[..15]).is_err());
    }

    #[test]
    fn serde_bytes_match_akita_serialize_bytes() {
        fn assert_agrees<T: Serialize + AkitaSerialize>(value: T) {
            let mut akita = Vec::new();
            value
                .serialize_uncompressed(&mut akita)
                .expect("encoding cannot fail");
            assert_eq!(encode(&value), akita);
        }

        assert_agrees(fp32(4_294_967_290));
        assert_agrees(fp64(18_446_744_073_709_551_556));
        assert_agrees(fp128(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feec));
    }
}
