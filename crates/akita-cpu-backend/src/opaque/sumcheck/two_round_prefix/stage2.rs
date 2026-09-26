use super::common::*;
use akita_sumcheck::reduce_signed_accum;
use jolt_field::{Field, Ring, Unreduced, Zero};
use jolt_poly::UnivariatePoly;

/// Range-image grid of the first two stage-2 rounds.
///
/// Each witness quad `[w00, w10, w01, w11]` spans the first two coefficient
/// variables. The grid holds, for every point of `{0, 1, Infinity}^2`, the sum
/// over quads of the equality weight of the variables after the quad times the
/// quad's local range-image value `W (W + 1)`, where `Infinity` takes the
/// leading coefficient in that coordinate. It is stored as one quadratic in the
/// first variable per value of the second.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2PrefixCache<E: Field> {
    norm_x_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
    batching_coeff: E,
}

impl<E: Field + Unreduced> Stage2PrefixCache<E> {
    /// Build the grid from the equality weight of every quad digit class.
    ///
    /// Class `d0 | d1 << bits | d2 << 2 bits | d3 << 3 bits` holds quads whose
    /// digits are `d_i - b / 2`, with `bits = log2(b)`.
    ///
    /// # Panics
    ///
    /// Panics if `b` is not 4 or 8, or if `histogram` does not have one entry
    /// per class.
    pub(crate) fn from_norm_histogram(
        histogram: &[E],
        b: usize,
        tau0: E,
        tau1: E,
        batching_coeff: E,
    ) -> Self {
        let table: &[[i64; STAGE2_PREFIX_POINT_COUNT]] = match b {
            4 => &STAGE2_B4_NORM_LOOKUP_TABLE,
            8 => &STAGE2_B8_NORM_LOOKUP_TABLE,
            _ => unreachable!("unsupported stage-2 prefix basis"),
        };
        assert_eq!(histogram.len(), table.len());
        let mut pos = [E::SmallProduct::zero(); STAGE2_PREFIX_POINT_COUNT];
        let mut neg = [E::SmallProduct::zero(); STAGE2_PREFIX_POINT_COUNT];
        for (&weight, values) in histogram.iter().zip(table) {
            if !weight.is_zero() {
                accum_lookup_vector_signed(&mut pos, &mut neg, weight, values);
            }
        }
        let grid: [E; STAGE2_PREFIX_POINT_COUNT] =
            std::array::from_fn(|point| reduce_signed_accum::<E>(pos[point], neg[point]));
        Self {
            norm_x_row_coeffs: std::array::from_fn(|y| {
                quadratic_coeffs_from_01_inf(grid[y], grid[3 + y], grid[6 + y])
            }),
            tau0,
            tau1,
            batching_coeff,
        }
    }
}

impl<E: Field + Ring> Stage2PrefixCache<E> {
    /// Range-image message of round 0, including the batching coefficient.
    pub(crate) fn round0_norm_poly(&self) -> UnivariatePoly<E> {
        let norm_q = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.norm_x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.norm_x_row_coeffs[1], self.tau1),
        );
        let coeffs = mul_linear_by_quadratic_coeffs(self.tau0, norm_q)
            .map(|coeff| self.batching_coeff * coeff);
        UnivariatePoly::new(coeffs.to_vec())
    }

    /// Range-image message of round 1 after challenge `r0`, including the
    /// batching coefficient.
    pub(crate) fn round1_norm_poly(&self, r0: E) -> UnivariatePoly<E> {
        let norm_y_values: [E; 3] =
            std::array::from_fn(|y| eval_quadratic_from_coeffs(self.norm_x_row_coeffs[y], r0));
        let norm_q =
            quadratic_coeffs_from_01_inf(norm_y_values[0], norm_y_values[1], norm_y_values[2]);
        let scale = self.batching_coeff * linear_eq_eval(self.tau0, r0);
        let coeffs = mul_linear_by_quadratic_coeffs(self.tau1, norm_q).map(|coeff| scale * coeff);
        UnivariatePoly::new(coeffs.to_vec())
    }
}
