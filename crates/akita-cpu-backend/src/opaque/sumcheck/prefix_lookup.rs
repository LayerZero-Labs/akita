//! Lookup helpers shared by the Stage 1 and Stage 2 two-round prefix caches.
//!
//! When the stage-specific prefix gate fires, the first two rounds of each
//! stage's sumcheck can be collapsed into a single bivariate evaluation over
//! a 4-value inner-dimension quad. The prover builds a local compressed grid
//! and immediately reconstructs the two ordinary sumcheck round messages from
//! it. Those reconstructed messages are then passed to the normal generic
//! sumcheck drivers and serialized as ordinary `SumcheckProof` or
//! `EqFactoredSumcheckProof` rounds.
//!
//! The prefix grids are not part of the public proof object or
//! verifier API. They are transient backend caches used to avoid
//! expensive scans over compact witness tables before the witness is folded to
//! round 2.
//!
//! Point semantics for the evaluation domains:
//!
//! - Finite points are ordinary evaluations of the bilinear multilinear
//!   extension over the quad.
//! - `Infinity` means "take the leading coefficient in that coordinate".

use jolt_field::{Field, Unreduced};

pub(crate) const LOOKUP_PREFIX_INF: i64 = i64::MIN;

pub(crate) const fn lookup_bilinear_coeffs_from_quad(quad: [i64; 4]) -> [i64; 4] {
    let [t00, t10, t01, t11] = quad;
    [t00, t10 - t00, t01 - t00, t11 - t10 - t01 + t00]
}

pub(crate) const fn lookup_bilinear_eval_on_prefix_points(quad: [i64; 4], x: i64, y: i64) -> i64 {
    let [a, b, c, d] = lookup_bilinear_coeffs_from_quad(quad);
    let x_is_inf = x == LOOKUP_PREFIX_INF;
    let y_is_inf = y == LOOKUP_PREFIX_INF;
    if !x_is_inf && !y_is_inf {
        a + x * (b + y * d) + y * c
    } else if x_is_inf && !y_is_inf {
        b + y * d
    } else if !x_is_inf && y_is_inf {
        c + x * d
    } else {
        d
    }
}

#[inline]
pub(crate) fn accum_lookup_vector_signed<E: Field + Unreduced, const N: usize>(
    pos: &mut [E::SmallProduct; N],
    neg: &mut [E::SmallProduct; N],
    coeff: E,
    values: &[i64; N],
) {
    for (idx, &value) in values.iter().enumerate() {
        if value > 0 {
            pos[idx] += coeff.mul_u64_unreduced(value as u64);
        } else if value < 0 {
            neg[idx] += coeff.mul_u64_unreduced(value.unsigned_abs());
        }
    }
}

#[inline]
pub(crate) fn linear_eq_eval<E: Field>(tau: E, x: E) -> E {
    tau * x + (E::one() - tau) * (E::one() - x)
}

#[inline]
pub(crate) fn quadratic_coeffs_from_01_inf<E: Field>(at_zero: E, at_one: E, at_inf: E) -> [E; 3] {
    [at_zero, at_one - at_zero - at_inf, at_inf]
}

#[inline]
pub(crate) fn eval_quadratic_from_coeffs<E: Field>(coeffs: [E; 3], x: E) -> E {
    coeffs[0] + x * (coeffs[1] + x * coeffs[2])
}

#[inline]
fn linear_eq_coeffs<E: Field>(tau: E) -> [E; 2] {
    [E::one() - tau, tau + tau - E::one()]
}

#[inline]
pub(crate) fn scale_quadratic_coeffs<E: Field>(coeffs: [E; 3], scale: E) -> [E; 3] {
    [scale * coeffs[0], scale * coeffs[1], scale * coeffs[2]]
}

#[inline]
pub(crate) fn add_quadratic_coeffs<E: Field>(lhs: [E; 3], rhs: [E; 3]) -> [E; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

#[inline]
pub(crate) fn mul_linear_by_quadratic_coeffs<E: Field>(tau: E, quad: [E; 3]) -> [E; 4] {
    let [l0, l1] = linear_eq_coeffs(tau);
    [
        l0 * quad[0],
        l0 * quad[1] + l1 * quad[0],
        l0 * quad[2] + l1 * quad[1],
        l1 * quad[2],
    ]
}

#[cfg(test)]
pub(crate) mod test_support {
    use jolt_field::Field;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum PrefixPoint<E: Field> {
        Finite(E),
        Infinity,
    }

    #[inline]
    pub(crate) fn bilinear_coeffs_from_quad<E: Field>(quad: [E; 4]) -> [E; 4] {
        let [t00, t10, t01, t11] = quad;
        [t00, t10 - t00, t01 - t00, t11 - t10 - t01 + t00]
    }

    #[inline]
    pub(crate) fn bilinear_eval<E: Field>(quad: [E; 4], x: E, y: E) -> E {
        let [a, b, c, d] = bilinear_coeffs_from_quad(quad);
        a + x * (b + y * d) + y * c
    }

    #[inline]
    pub(crate) fn bilinear_eval_on_prefix_points<E: Field>(
        quad: [E; 4],
        x: PrefixPoint<E>,
        y: PrefixPoint<E>,
    ) -> E {
        let [a, b, c, d] = bilinear_coeffs_from_quad(quad);
        match (x, y) {
            (PrefixPoint::Finite(x), PrefixPoint::Finite(y)) => a + x * (b + y * d) + y * c,
            (PrefixPoint::Infinity, PrefixPoint::Finite(y)) => b + y * d,
            (PrefixPoint::Finite(x), PrefixPoint::Infinity) => c + x * d,
            (PrefixPoint::Infinity, PrefixPoint::Infinity) => d,
        }
    }
}
