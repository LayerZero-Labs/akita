//! Jolt interop: implements `jolt_field`'s slim trait hierarchy plus capability
//! traits for Akita field types, forwarding to the native [`crate`] definitions.
//!
//! This is the only module that names `jolt_field`. It carries no algebra of its
//! own: marker impls are empty and every method delegates to the native impl, so
//! Jolt callers observe identical behavior. Native traits are referenced through
//! fully-qualified `crate::` paths and are deliberately never `use`d here: that
//! keeps the `#[cfg(test)]` module's `use super::*` from pulling a native trait
//! into scope where it would collide with the `jolt_field` trait of the same name
//! on a concrete field type.
//!
//! The accumulator traits are nativized (see [`crate::AdditiveAccumulator`] etc.),
//! but the Jolt `WithAccumulator` impl deliberately keeps Jolt's own
//! `jf::NaiveAccumulator<Self>` as its associated accumulator: it satisfies the
//! Jolt accumulator-trait bounds out of the box and there is no Jolt-side consumer
//! that would benefit from routing through the native accumulator instead.

use jolt_field as jf;
use rand_core::RngCore;

use crate::unreduced::{
    AccumPair, Fp128MulU64Accum, Fp128ProductAccum, Fp128x8i32, Fp32ProductAccum, Fp32x2i32,
    Fp64ProductAccum, Fp64x4i32, FpExt2Fp64ProductAccum, RingSubfieldFpExt4Fp32ProductAccum,
};
use crate::{
    Fp128, Fp32, Fp64, FpExt2, FpExt2Config, PowerBasisFpExt4, PowerBasisFpExt4Config,
    RingSubfieldFpExt4, RingSubfieldFpExt8, TowerBasisFpExt4, TowerBasisFpExt4Config,
};

macro_rules! impl_prime_jolt_traits {
    ($ty:ident<$p:ident: $p_ty:ty>, $fixed_bytes:literal) => {
        impl<const $p: $p_ty> jf::AdditiveGroup for $ty<$p> {}
        impl<const $p: $p_ty> jf::RingCore for $ty<$p> {}

        impl<const $p: $p_ty> jf::Invertible for $ty<$p> {
            #[inline]
            fn inverse(&self) -> Option<Self> {
                <Self as crate::Invertible>::inverse(self)
            }
        }

        impl<const $p: $p_ty> jf::FieldCore for $ty<$p> {}

        impl<const $p: $p_ty> jf::FromPrimitiveInt for $ty<$p> {
            #[inline]
            fn from_u64(v: u64) -> Self {
                <Self as crate::FromPrimitiveInt>::from_u64(v)
            }
            #[inline]
            fn from_i64(v: i64) -> Self {
                <Self as crate::FromPrimitiveInt>::from_i64(v)
            }
            #[inline]
            fn from_u128(v: u128) -> Self {
                <Self as crate::FromPrimitiveInt>::from_u128(v)
            }
            #[inline]
            fn from_i128(v: i128) -> Self {
                <Self as crate::FromPrimitiveInt>::from_i128(v)
            }
        }

        impl<const $p: $p_ty> jf::RandomSampling for $ty<$p> {
            #[inline]
            fn random<R: RngCore>(rng: &mut R) -> Self {
                <Self as crate::RandomSampling>::random(rng)
            }
        }

        impl<const $p: $p_ty> jf::MulPow2 for $ty<$p> {}
        impl<const $p: $p_ty> jf::MulPrimitiveInt for $ty<$p> {}

        impl<const $p: $p_ty> jf::FixedByteSize for $ty<$p> {
            const NUM_BYTES: usize = <Self as crate::FixedByteSize>::NUM_BYTES;
        }

        impl<const $p: $p_ty> jf::CanonicalBytes for $ty<$p> {
            #[inline(always)]
            fn to_bytes_le(&self, out: &mut [u8]) {
                <Self as crate::CanonicalBytes>::to_bytes_le(self, out)
            }
        }

        impl<const $p: $p_ty> jf::ReducingBytes for $ty<$p> {
            #[inline(always)]
            fn from_le_bytes_mod_order(bytes: &[u8]) -> Self {
                <Self as crate::ReducingBytes>::from_le_bytes_mod_order(bytes)
            }
        }

        impl<const $p: $p_ty> jf::TranscriptChallenge for $ty<$p> {
            #[inline(always)]
            fn from_challenge_bytes(bytes: &[u8]) -> Self {
                <Self as crate::TranscriptChallenge>::from_challenge_bytes(bytes)
            }
        }

        impl<const $p: $p_ty> jf::FixedBytes<$fixed_bytes> for $ty<$p> {}

        impl<const $p: $p_ty> jf::CanonicalBitLength for $ty<$p> {
            #[inline]
            fn num_bits(&self) -> u32 {
                <Self as crate::CanonicalBitLength>::num_bits(self)
            }
        }

        impl<const $p: $p_ty> jf::CanonicalU64 for $ty<$p> {
            #[inline]
            fn to_canonical_u64_checked(&self) -> Option<u64> {
                <Self as crate::CanonicalU64>::to_canonical_u64_checked(self)
            }
        }

        impl<const $p: $p_ty> jf::WithAccumulator for $ty<$p> {
            type Accumulator = jf::NaiveAccumulator<Self>;
        }
    };
}

impl_prime_jolt_traits!(Fp32<P: u32>, 4);
impl_prime_jolt_traits!(Fp64<P: u64>, 8);
impl_prime_jolt_traits!(Fp128<P: u128>, 16);

// Extension fields forward every Jolt trait to the native impl, gated on the
// native trait being available for the same instantiation (`where Self: ...`).
// This tracks the native bounds (the `*MulBackend` requirements live there)
// without restating them and stays correct as those bounds evolve.
macro_rules! forward_ext_jolt_traits {
    ([$($g:tt)*], $ty:ty) => {
        impl<$($g)*> jf::AdditiveGroup for $ty where $ty: crate::AdditiveGroup {}

        impl<$($g)*> jf::RingCore for $ty
        where
            $ty: crate::RingCore,
        {
            #[inline(always)]
            fn square(&self) -> Self {
                <Self as crate::RingCore>::square(self)
            }
        }

        impl<$($g)*> jf::Invertible for $ty
        where
            $ty: crate::Invertible,
        {
            #[inline]
            fn inverse(&self) -> Option<Self> {
                <Self as crate::Invertible>::inverse(self)
            }
        }

        impl<$($g)*> jf::FieldCore for $ty where $ty: crate::FieldCore {}

        impl<$($g)*> jf::FromPrimitiveInt for $ty
        where
            $ty: crate::FromPrimitiveInt,
        {
            #[inline]
            fn from_u64(v: u64) -> Self {
                <Self as crate::FromPrimitiveInt>::from_u64(v)
            }
            #[inline]
            fn from_i64(v: i64) -> Self {
                <Self as crate::FromPrimitiveInt>::from_i64(v)
            }
            #[inline]
            fn from_u128(v: u128) -> Self {
                <Self as crate::FromPrimitiveInt>::from_u128(v)
            }
            #[inline]
            fn from_i128(v: i128) -> Self {
                <Self as crate::FromPrimitiveInt>::from_i128(v)
            }
        }

        impl<$($g)*> jf::RandomSampling for $ty
        where
            $ty: crate::RandomSampling,
        {
            #[inline]
            fn random<R: RngCore>(rng: &mut R) -> Self {
                <Self as crate::RandomSampling>::random(rng)
            }
        }
    };
}

forward_ext_jolt_traits!([F: crate::FieldCore, C: FpExt2Config<F>], FpExt2<F, C>);
forward_ext_jolt_traits!(
    [F: crate::FieldCore, C2: FpExt2Config<F>, C4: TowerBasisFpExt4Config<F, C2>],
    TowerBasisFpExt4<F, C2, C4>
);
forward_ext_jolt_traits!(
    [F: crate::FieldCore, C: PowerBasisFpExt4Config<F>],
    PowerBasisFpExt4<F, C>
);
forward_ext_jolt_traits!([F: crate::FieldCore], RingSubfieldFpExt4<F>);
forward_ext_jolt_traits!([F: crate::FieldCore], RingSubfieldFpExt8<F>);

// --- Wide accumulators ------------------------------------------------------

impl jf::AdditiveGroup for Fp32x2i32 {}
impl jf::AdditiveGroup for Fp64x4i32 {}
impl jf::AdditiveGroup for Fp128x8i32 {}
impl jf::AdditiveGroup for Fp32ProductAccum {}
impl jf::AdditiveGroup for Fp64ProductAccum {}
impl jf::AdditiveGroup for Fp128MulU64Accum {}
impl jf::AdditiveGroup for Fp128ProductAccum {}
impl jf::AdditiveGroup for RingSubfieldFpExt4Fp32ProductAccum {}
impl jf::AdditiveGroup for FpExt2Fp64ProductAccum {}

impl<A: crate::AdditiveGroup> jf::AdditiveGroup for AccumPair<A> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalField, Fp32, Fp64, Prime128Offset275};
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
            + std::fmt::Debug
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
