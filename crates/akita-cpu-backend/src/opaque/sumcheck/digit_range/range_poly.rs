//! The Stage-1 polynomial `Q(x) = prod_{k < b/2} (x - k(k+1))`.

use jolt_field::{Field, Ring, Unreduced};
use jolt_poly::OmittedConstantPoly;

/// Nonconstant round coefficients `q_1..q_d` of the range polynomial. The
/// eq-factored driver recovers `q_0` from the running claim, so no kernel
/// computes it.
pub(crate) const MAX_DIRECT_RANGE_COEFFICIENTS: usize = 4;

/// What the first of a round's per-pair sums holds; every later sum is a
/// Taylor term divided by its integer factor.
#[derive(Clone, Copy)]
pub(crate) enum LinearSum {
    /// The linear Taylor term, divided by its factor like the others.
    Taylor,
    /// `Q(right) - Q(left)`, the sum of all nonconstant coefficients, which
    /// table-driven rounds read without computing the linear term.
    RangeDifference,
}

#[derive(Clone)]
pub(crate) struct RangePoly {
    degree_q: usize,
}

impl RangePoly {
    pub(crate) const fn new(basis: usize) -> Self {
        assert!(
            matches!(basis, 4 | 8),
            "direct range prover requires basis 4 or 8"
        );
        Self {
            degree_q: basis / 2,
        }
    }

    /// Number of nonconstant round coefficients each pair contributes.
    pub(crate) const fn num_coefficients(&self) -> usize {
        self.degree_q
    }

    /// Integer factor of each nonconstant Taylor coefficient of
    /// `Q(left + X * delta)`, which the per-pair kernels leave out and
    /// [`round_poly_from_sums`](Self::round_poly_from_sums) applies once.
    pub(crate) fn taylor_factors(&self) -> &'static [u64] {
        match self.degree_q {
            2 => &[2, 1],
            4 => &[4, 6, 4, 1],
            _ => unreachable!("direct range leaf only supports quadratic and quartic checks"),
        }
    }

    /// Reduce a round's accumulated per-pair sums into its nonconstant
    /// coefficients.
    pub(crate) fn round_poly_from_sums<E: Field + Ring + Unreduced>(
        &self,
        sums: &TaylorSums<E>,
        linear: LinearSum,
    ) -> OmittedConstantPoly<E> {
        let mut coefficients: Vec<E> = sums
            .iter()
            .zip(self.taylor_factors())
            .map(|(&sum, &factor)| E::from_u64(factor) * E::reduce_product(sum))
            .collect();
        if let LinearSum::RangeDifference = linear {
            let higher: E = coefficients[1..].iter().copied().sum();
            coefficients[0] = E::reduce_product(sums[0]) - higher;
        }
        OmittedConstantPoly::new(coefficients)
    }
}

/// Nonconstant Taylor sums of a round, before
/// [`RangePoly::round_poly_from_sums`] scales them.
pub(crate) type TaylorSums<E> = [<E as Unreduced>::Product; MAX_DIRECT_RANGE_COEFFICIENTS];

/// Round-3 terms of one octet class.
///
/// With `value` the class's folded value after round 2 and
/// `shifted = value - 5`, a pair with this class on the left and difference
/// `delta` has quartic Taylor terms `(shifted^2 - 7) delta^2`,
/// `shifted delta^3` and `delta^4`, before their integer factors. Quadratic
/// checks use neither `shifted` nor `second`.
#[derive(Clone, Copy)]
pub(crate) struct OctetClassTerms<E> {
    pub(crate) value: E,
    shifted: E,
    /// `shifted^2 - 7`.
    second: E,
}

impl RangePoly {
    /// Evaluate Q at a field element, multiplying its roots in order.
    #[inline]
    pub(crate) fn eval<E: Field + Ring>(&self, x: E) -> E {
        (0..self.degree_q).fold(E::one(), |value, k| {
            let k = k as i64;
            value * (x - E::from_i64(k * (k + 1)))
        })
    }

    /// Integer evaluation for the compile-time prefix lookup tables.
    pub(crate) const fn eval_i64(&self, x: i64) -> i64 {
        match self.degree_q {
            2 => x * (x - 2),
            4 => x * (x - 2) * (x - 6) * (x - 12),
            _ => unreachable!(),
        }
    }

    /// Precompute the value-dependent higher Taylor terms of an octet class.
    pub(crate) fn class_terms<E: Field + Ring>(&self, value: E) -> OctetClassTerms<E> {
        let shifted = value - E::from_u64(5);
        OctetClassTerms {
            value,
            shifted,
            second: shifted.square() - E::from_u64(7),
        }
    }

    /// Add `weight` times the nonconstant Taylor terms of `Q(left + X * delta)`,
    /// each divided by its integer factor from
    /// [`RangePoly::taylor_factors`], to `sums`.
    ///
    /// With `b = left - 5`, the quartic range polynomial is
    /// `Q = b^4 - 42 b^2 - 64 b + 105`, so its terms at `left` are
    /// `4((b^2 - 21) b - 16) delta`, `6(b^2 - 7) delta^2`, `4 b delta^3` and
    /// `delta^4`. Every term carries a power of `delta`, so the weight rides on
    /// the powers `weight * delta^k`: five multiplications, one squaring and four
    /// unreduced products. The quadratic terms are `2(left - 1) delta` and
    /// `delta^2`.
    #[inline(always)]
    pub(crate) fn accumulate_entry_terms<E: Field + Ring + Unreduced>(
        &self,
        sums: &mut TaylorSums<E>,
        left_range_image: E,
        range_image_delta: E,
        weight: E,
    ) {
        let weighted_delta = weight * range_image_delta;
        match self.degree_q {
            2 => {
                sums[0] += (left_range_image - E::one()).mul_unreduced(weighted_delta);
                sums[1] += range_image_delta.mul_unreduced(weighted_delta);
            }
            4 => {
                let shifted = left_range_image - E::from_u64(5);
                let shifted_squared = shifted.square();
                let weighted_delta_squared = weighted_delta * range_image_delta;
                let weighted_delta_cubed = weighted_delta_squared * range_image_delta;
                sums[0] += ((shifted_squared - E::from_u64(21)) * shifted - E::from_u64(16))
                    .mul_unreduced(weighted_delta);
                sums[1] += (shifted_squared - E::from_u64(7)).mul_unreduced(weighted_delta_squared);
                sums[2] += shifted.mul_unreduced(weighted_delta_cubed);
                sums[3] += range_image_delta.mul_unreduced(weighted_delta_cubed);
            }
            _ => unreachable!("direct range leaf only supports quadratic and quartic checks"),
        }
    }

    /// Add `weight` times the round-3 terms of one octet pair above the linear one.
    #[inline(always)]
    pub(crate) fn accumulate_octet_pair_terms<E: Field + Ring + Unreduced>(
        &self,
        sums: &mut TaylorSums<E>,
        left: &OctetClassTerms<E>,
        right: &OctetClassTerms<E>,
        weight: E,
    ) {
        let delta = right.value - left.value;
        let delta_squared = delta.square();
        if self.degree_q == 2 {
            sums[1] += delta_squared.mul_unreduced(weight);
        } else {
            let weighted_delta_squared = weight * delta_squared;
            sums[1] += left.second.mul_unreduced(weighted_delta_squared);
            sums[2] += (left.shifted * delta).mul_unreduced(weighted_delta_squared);
            sums[3] += delta_squared.mul_unreduced(weighted_delta_squared);
        }
    }
}
