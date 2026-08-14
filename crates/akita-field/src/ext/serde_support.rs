//! `serde` for the extension fields, as base-coefficient arrays.
//!
//! An element encodes as its `[F; K]` coefficients in the same basis order as
//! [`AkitaSerialize`](akita_serialization::AkitaSerialize), so canonicality
//! follows from the base field's decode. As with the prime fields,
//! verifier-reachable decoding of Akita containers stays on
//! [`AkitaDeserialize`](akita_serialization::AkitaDeserialize), which is the
//! only path that bounds container lengths.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{FpExt2, FpExt2Config, FpExt4, FpExt8};
use crate::FieldCore;

macro_rules! impl_ext_serde {
    ($ty:ident $(, $cfg:ident: $bound:path)?; $k:literal; |$coeffs:ident| $new:expr) => {
        impl<F: FieldCore + Serialize $(, $cfg: $bound)?> Serialize for $ty<F $(, $cfg)?> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.coeffs.serialize(serializer)
            }
        }

        impl<'de, F: FieldCore + Deserialize<'de> $(, $cfg: $bound)?> Deserialize<'de>
            for $ty<F $(, $cfg)?>
        {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let $coeffs = <[F; $k]>::deserialize(deserializer)?;
                Ok($new)
            }
        }
    };
}

impl_ext_serde!(FpExt2, C: FpExt2Config<F>; 2; |coeffs| Self::new(coeffs[0], coeffs[1]));
impl_ext_serde!(FpExt4; 4; |coeffs| Self::new(coeffs));
impl_ext_serde!(FpExt8; 8; |coeffs| Self::new(coeffs));

#[cfg(test)]
mod tests {
    use serde::de::DeserializeOwned;

    use super::*;
    use crate::ext::Ext2;
    use crate::prime::Fp128;
    use crate::CanonicalField;

    type F = Fp128<0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feed>;
    type E2 = Ext2<F>;
    type E4 = FpExt4<F>;
    type E8 = FpExt8<F>;

    const BASE_WIDTH: usize = 16;

    fn encode<T: Serialize>(value: T) -> Vec<u8> {
        postcard::to_stdvec(&value).expect("encoding cannot fail")
    }

    fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    fn base(value: u128) -> F {
        F::from_canonical_u128_checked(value).expect("value is canonical")
    }

    fn assert_layout(bytes: &[u8], coeffs: &[u128]) {
        assert_eq!(bytes.len(), coeffs.len() * BASE_WIDTH);
        for (index, coeff) in coeffs.iter().enumerate() {
            let start = index * BASE_WIDTH;
            assert_eq!(&bytes[start..start + BASE_WIDTH], coeff.to_le_bytes());
        }
    }

    #[test]
    fn coefficients_encode_in_basis_order_at_fixed_width() {
        assert_layout(&encode(E2::new(base(1), base(2))), &[1, 2]);
        assert_layout(
            &encode(E4::new([base(1), base(2), base(3), base(4)])),
            &[1, 2, 3, 4],
        );
        assert_layout(
            &encode(E8::new([
                base(1),
                base(2),
                base(3),
                base(4),
                base(5),
                base(6),
                base(7),
                base(8),
            ])),
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );
    }

    #[test]
    fn extension_fields_round_trip() {
        let e2 = E2::new(base(3), base(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feec));
        assert_eq!(decode::<E2>(&encode(e2)).unwrap(), e2);

        let e4 = E4::new([base(0), base(1), base(1 << 100), base(7)]);
        assert_eq!(decode::<E4>(&encode(e4)).unwrap(), e4);

        let e8 = E8::new([
            base(0),
            base(1),
            base(2),
            base(1 << 64),
            base(4),
            base(5),
            base(6),
            base(0xffff_ffff_ffff_ffff_ffff_ffff_ffff_feec),
        ]);
        assert_eq!(decode::<E8>(&encode(e8)).unwrap(), e8);
    }

    #[test]
    fn non_canonical_and_short_coefficient_arrays_are_rejected() {
        let mut bytes = encode(E2::new(base(1), base(2)));
        assert!(decode::<E2>(&bytes[..BASE_WIDTH]).is_err());

        bytes[BASE_WIDTH..].copy_from_slice(&u128::MAX.to_le_bytes());
        assert!(decode::<E2>(&bytes).is_err());
    }
}
