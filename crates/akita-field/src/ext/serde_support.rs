//! `serde` for the extension fields, as base-coefficient arrays.
//!
//! An element encodes as its `[F; K]` coefficients in the same basis order as
//! [`AkitaSerialize`](akita_serialization::AkitaSerialize), so canonicality is
//! enforced by the base field's own decode. As with the prime fields, this is a
//! host and tooling surface: verifier-reachable decoding stays on
//! [`AkitaDeserialize`](akita_serialization::AkitaDeserialize), which is the
//! only path that bounds container lengths.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{FpExt2, FpExt2Config, FpExt4, FpExt8};
use crate::FieldCore;

/// Implements serde for one extension arity. `$cfg` is the extension-config
/// parameter, which only `FpExt2` carries.
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
