//! Count operations at the Field API, with unchanged underlying field arithmetic.

use super::counts;
use jolt_field::{ExtField, Field, Prime128Offset6A23801F as F, Ring};
use rand::RngCore;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub(super) struct Counted(pub(super) F);

impl Display for Counted {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

jolt_field::impl_ring_ops!(impl[] Counted {
    add(a, b): { counts::add(); Counted(a.0 + b.0) },
    sub(a, b): { counts::sub(); Counted(a.0 - b.0) },
    mul(a, b): { counts::mul(); Counted(a.0 * b.0) },
    neg(a): { counts::neg(); Counted(-a.0) },
    zero: Counted(F::from_u64(0)),
    one: Counted(F::from_u64(1)),
});

impl Ring for Counted {
    fn from_u64(v: u64) -> Self {
        Self(F::from_u64(v))
    }
    fn from_i64(v: i64) -> Self {
        Self(F::from_i64(v))
    }
    fn from_u128(v: u128) -> Self {
        Self(F::from_u128(v))
    }
    fn from_i128(v: i128) -> Self {
        Self(F::from_i128(v))
    }
}

impl Field for Counted {
    fn inverse(&self) -> Option<Self> {
        counts::inverse();
        self.0.inverse().map(Self)
    }
    fn random<R: RngCore>(rng: &mut R) -> Self {
        Self(F::random(rng))
    }
}

impl ExtField<F> for Counted {
    const DEGREE: usize = 1;
    fn lift_base(x: F) -> Self {
        Self(x)
    }
    fn mul_base(self, x: F) -> Self {
        counts::mul();
        Self(self.0 * x)
    }
    fn from_base_fn<G: FnMut(usize) -> F>(mut f: G) -> Self {
        counts::import();
        Self(f(0))
    }
    fn base_coefficient(&self, index: usize) -> F {
        assert_eq!(index, 0);
        counts::export();
        self.0
    }
    fn frobenius_pow(self, _power: usize) -> Self {
        self
    }
}
