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
    ($ty:ident<$p:ident: $p_ty:ty>, $canon:ident) => {
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
    };
}

impl_prime_jolt_traits!(Fp32<P: u32>, from_canonical_u32);
impl_prime_jolt_traits!(Fp64<P: u64>, from_canonical_u64);
impl_prime_jolt_traits!(Fp128<P: u128>, from_canonical_u128);

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
