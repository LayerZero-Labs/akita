use std::{
    fmt,
    hash::{Hash, Hasher},
    iter::{Product, Sum},
    ops::{Add, Sub},
};

use jolt_field as jf;
use num_traits::{One, Zero};

use crate::{
    fields::{
        AccumPair, Fp128, Fp128MulU64Accum, Fp128ProductAccum, Fp128x8i32, Fp2, Fp2Config, Fp32,
        Fp32x2i32, Fp4, Fp4Config, Fp64, Fp64ProductAccum, Fp64x4i32,
    },
    CanonicalField, FieldCore,
};
use akita_serialization::Valid;

macro_rules! impl_prime_jolt_traits {
    ($ty:ident<$p:ident: $p_ty:ty>, $canon:ident, $bytes:expr, $fixed_bytes:literal) => {
        impl<const $p: $p_ty> Zero for $ty<$p> {
            #[inline]
            fn zero() -> Self {
                Self::default()
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.to_canonical_u128() == 0
            }
        }

        impl<const $p: $p_ty> One for $ty<$p> {
            #[inline]
            fn one() -> Self {
                if $p > 1 {
                    Self::$canon(1)
                } else {
                    Self::zero()
                }
            }
        }

        impl<const $p: $p_ty> fmt::Display for $ty<$p> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.to_canonical_u128())
            }
        }

        impl<const $p: $p_ty> Hash for $ty<$p> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.to_canonical_u128().hash(state);
            }
        }

        impl<const $p: $p_ty> Sum for $ty<$p> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), |acc, x| acc + x)
            }
        }

        impl<'a, const $p: $p_ty> Sum<&'a Self> for $ty<$p> {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), |acc, x| acc + *x)
            }
        }

        impl<const $p: $p_ty> Product for $ty<$p> {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::one(), |acc, x| acc * x)
            }
        }

        impl<'a, const $p: $p_ty> Product<&'a Self> for $ty<$p> {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::one(), |acc, x| acc * *x)
            }
        }

        impl<const $p: $p_ty> jf::AdditiveGroup for $ty<$p> {}
        impl<const $p: $p_ty> jf::RingCore for $ty<$p> {}
        impl<const $p: $p_ty> jf::FieldCore for $ty<$p> {}
        impl<const $p: $p_ty> jf::MulPow2 for $ty<$p> {}
        impl<const $p: $p_ty> jf::MulPrimitiveInt for $ty<$p> {}
        impl<const $p: $p_ty> jf::WithAccumulator for $ty<$p> {
            type Accumulator = jf::NaiveAccumulator<Self>;
        }
        impl<const $p: $p_ty> jf::FixedByteSize for $ty<$p> {
            const NUM_BYTES: usize = $bytes;
        }

        impl<const $p: $p_ty> jf::CanonicalBytes for $ty<$p> {
            #[inline]
            fn to_bytes_le(&self, out: &mut [u8]) {
                assert_eq!(out.len(), <Self as jf::FixedByteSize>::NUM_BYTES);
                out.copy_from_slice(
                    &self.to_canonical_u128().to_le_bytes()
                        [..<Self as jf::FixedByteSize>::NUM_BYTES],
                );
            }
        }

        impl<const $p: $p_ty> jf::ReducingBytes for $ty<$p> {
            #[inline]
            fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
                reduce_le_bytes_mod_order(bytes)
            }
        }

        impl<const $p: $p_ty> jf::TranscriptChallenge for $ty<$p> {
            #[inline]
            fn from_challenge_bytes(bytes: &[u8]) -> Self {
                <Self as jf::ReducingBytes>::from_le_bytes_mod_order(bytes)
            }
        }

        impl<const $p: $p_ty> jf::FixedBytes<$fixed_bytes> for $ty<$p> {}

        impl<const $p: $p_ty> jf::CanonicalBitLength for $ty<$p> {
            #[inline]
            fn num_bits(&self) -> u32 {
                let value = self.to_canonical_u128();
                u128::BITS - value.leading_zeros()
            }
        }

        impl<const $p: $p_ty> jf::CanonicalU64 for $ty<$p> {
            #[inline]
            fn to_canonical_u64_checked(&self) -> Option<u64> {
                self.to_canonical_u128().try_into().ok()
            }
        }
    };
}

fn reduce_le_bytes_mod_order<F: FieldCore + jf::FromPrimitiveInt>(bytes: &[u8]) -> F {
    bytes.iter().rev().fold(F::zero(), |acc, &byte| {
        acc * F::from_u64(256) + F::from_u64(byte as u64)
    })
}

impl_prime_jolt_traits!(Fp32<P: u32>, from_canonical_u32, 4, 4);
impl_prime_jolt_traits!(Fp64<P: u64>, from_canonical_u64, 8, 8);
impl_prime_jolt_traits!(Fp128<P: u128>, from_canonical_u128, 16, 16);

impl<F: FieldCore, C: Fp2Config<F>> Zero for Fp2<F, C> {
    #[inline]
    fn zero() -> Self {
        Self::new(F::zero(), F::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }
}

impl<F: FieldCore, C: Fp2Config<F>> One for Fp2<F, C> {
    #[inline]
    fn one() -> Self {
        Self::new(F::one(), F::zero())
    }
}

impl<F: FieldCore, C: Fp2Config<F>> fmt::Display for Fp2<F, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.c0, self.c1)
    }
}

impl<F: FieldCore, C: Fp2Config<F>> Hash for Fp2<F, C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.c0.hash(state);
        self.c1.hash(state);
    }
}

impl<F: FieldCore, C: Fp2Config<F>> Sum for Fp2<F, C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore, C: Fp2Config<F>> Sum<&'a Self> for Fp2<F, C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore, C: Fp2Config<F>> Product for Fp2<F, C> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore, C: Fp2Config<F>> Product<&'a Self> for Fp2<F, C> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore, C: Fp2Config<F>> jf::AdditiveGroup for Fp2<F, C> {}
impl<F: FieldCore + Valid, C: Fp2Config<F>> jf::FieldCore for Fp2<F, C> {}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Zero for Fp4<F, C2, C4> {
    #[inline]
    fn zero() -> Self {
        Self::new(Fp2::zero(), Fp2::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> One for Fp4<F, C2, C4> {
    #[inline]
    fn one() -> Self {
        Self::new(Fp2::one(), Fp2::zero())
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> fmt::Display for Fp4<F, C2, C4> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.c0, self.c1)
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Hash for Fp4<F, C2, C4> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.c0.hash(state);
        self.c1.hash(state);
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Sum for Fp4<F, C2, C4> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Sum<&'a Self> for Fp4<F, C2, C4> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + *x)
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Product for Fp4<F, C2, C4> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> Product<&'a Self>
    for Fp4<F, C2, C4>
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * *x)
    }
}

impl<F: FieldCore, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> jf::AdditiveGroup for Fp4<F, C2, C4> {}
impl<F: FieldCore + Valid, C2: Fp2Config<F>, C4: Fp4Config<F, C2>> jf::FieldCore
    for Fp4<F, C2, C4>
{
}

macro_rules! impl_wide_additive {
    ($ty:ty, $zero:expr) => {
        impl Zero for $ty {
            #[inline]
            fn zero() -> Self {
                $zero
            }

            #[inline]
            fn is_zero(&self) -> bool {
                *self == Self::zero()
            }
        }

        impl<'a> Add<&'a Self> for $ty {
            type Output = Self;

            #[inline]
            fn add(self, rhs: &'a Self) -> Self::Output {
                self + *rhs
            }
        }

        impl<'a> Sub<&'a Self> for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: &'a Self) -> Self::Output {
                self - *rhs
            }
        }

        impl jf::AdditiveGroup for $ty {}
    };
}

impl_wide_additive!(Fp32x2i32, Fp32x2i32([0; 2]));
impl_wide_additive!(Fp64x4i32, Fp64x4i32([0; 4]));
impl_wide_additive!(Fp128x8i32, Fp128x8i32([0; 8]));
impl_wide_additive!(Fp64ProductAccum, Fp64ProductAccum([0; 2]));
impl_wide_additive!(Fp128MulU64Accum, Fp128MulU64Accum([0; 3]));
impl_wide_additive!(Fp128ProductAccum, Fp128ProductAccum([0; 4]));

impl<A: jf::AdditiveGroup> Zero for AccumPair<A> {
    #[inline]
    fn zero() -> Self {
        Self(A::zero(), A::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

impl<'a, A: jf::AdditiveGroup> Add<&'a Self> for AccumPair<A> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &'a Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, A: jf::AdditiveGroup> Sub<&'a Self> for AccumPair<A> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &'a Self) -> Self::Output {
        self - *rhs
    }
}

impl<A: jf::AdditiveGroup> jf::AdditiveGroup for AccumPair<A> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{Fp32, Fp64, Prime128Offset275};
    use jolt_field::{
        AdditiveAccumulator, CanonicalBitLength, CanonicalU64, MulPow2, MulPrimitiveInt,
        ReducingBytes, RingAccumulator, TranscriptChallenge, WithAccumulator,
    };
    use jolt_transcript::{Blake2bTranscript, KeccakTranscript, Transcript};

    fn assert_byte_traits<F, const N: usize>(value: F, expected: [u8; N])
    where
        F: CanonicalField
            + jf::CanonicalBytes
            + jf::ReducingBytes
            + jf::TranscriptChallenge
            + jf::FixedByteSize
            + jf::FixedBytes<N>
            + jf::CanonicalBitLength
            + jf::CanonicalU64
            + fmt::Debug
            + Eq,
    {
        let encoded = value.to_bytes_array();
        assert_eq!(encoded, expected);
        assert_eq!(F::from_bytes_array(&encoded), value);
        assert_eq!(F::from_challenge_bytes(&encoded), value);
        assert_eq!(F::from_le_bytes_mod_order(&encoded), value);
        assert_eq!(
            value.num_bits(),
            u128::BITS - value.to_canonical_u128().leading_zeros()
        );
    }

    #[test]
    fn prime_fields_satisfy_jolt_byte_capabilities() {
        type F32 = Fp32<251>;
        type F64 = Fp64<4294967197>;
        type F128 = Prime128Offset275;

        assert_byte_traits::<F32, 4>(F32::from_u64(42), 42u32.to_le_bytes());
        assert_byte_traits::<F64, 8>(F64::from_u64(42), 42u64.to_le_bytes());
        assert_byte_traits::<F128, 16>(
            F128::from_canonical_u128(0x0102_0304_0506_0708),
            0x0102_0304_0506_0708u128.to_le_bytes(),
        );

        assert_eq!(F32::from_le_bytes_mod_order(&[255, 0]), F32::from_u64(4));
        assert_eq!(F32::from_challenge_bytes(&[255, 0]), F32::from_u64(4));
        assert_eq!(F32::zero().num_bits(), 0);
        assert_eq!(F64::from_u64(7).to_canonical_u64_checked(), Some(7));
        assert_eq!(
            F128::from_canonical_u128(1u128 << 65).to_canonical_u64_checked(),
            None
        );
        assert_eq!(F32::from_u64(3).mul_pow_2(4), F32::from_u64(48));
        assert_eq!(F64::from_u64(9).mul_u64(7), F64::from_u64(63));

        let mut acc = <F64 as WithAccumulator>::Accumulator::default();
        acc.fmadd(F64::from_u64(9), F64::from_u64(7));
        acc.add(F64::from_u64(2));
        assert_eq!(acc.reduce(), F64::from_u64(65));
    }

    #[test]
    fn jolt_digest_transcripts_accept_akita_fields() {
        type F = Fp64<4294967197>;

        let mut blake_a = Blake2bTranscript::<F>::new(b"akita-compat");
        let mut blake_b = Blake2bTranscript::<F>::new(b"akita-compat");
        let mut keccak = KeccakTranscript::<F>::new(b"akita-compat");

        for transcript in [&mut blake_a, &mut blake_b] {
            transcript.append(&F::from_u64(42));
            transcript.append_bytes(b"payload");
        }
        keccak.append(&F::from_u64(42));
        keccak.append_bytes(b"payload");

        let blake_challenge = blake_a.challenge();
        assert_eq!(blake_challenge, blake_b.challenge());
        let _: F = keccak.challenge();
    }
}
