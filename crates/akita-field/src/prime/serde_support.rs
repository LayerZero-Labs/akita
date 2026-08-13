//! `serde` for the prime fields, over the canonical residue.
//!
//! An element encodes as its canonical representative in the field's storage
//! width; decode rejects non-canonical values (`val >= P`) rather than reducing.
//!
//! This is a host and tooling surface. Akita's protocol wire format is
//! [`AkitaSerialize`](akita_serialization::AkitaSerialize) /
//! [`AkitaDeserialize`](akita_serialization::AkitaDeserialize), and
//! verifier-reachable decoding stays there: a serde format bounds sequence
//! lengths only if its consumer configured a limit, so these impls cannot make
//! the container guarantee that `AkitaDeserialize` does.

use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use super::{Fp128, Fp32, Fp64};
use crate::CanonicalField;

macro_rules! impl_prime_serde {
    ($ty:ident<$p:ident: $p_ty:ty>) => {
        impl<const $p: $p_ty> Serialize for $ty<$p> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                (self.to_canonical_u128() as $p_ty).serialize(serializer)
            }
        }

        impl<'de, const $p: $p_ty> Deserialize<'de> for $ty<$p> {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let raw = <$p_ty as Deserialize>::deserialize(deserializer)?;
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
