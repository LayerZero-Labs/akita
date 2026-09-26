//! Two-round prefix cache for the Stage 2 range-image norm term.
//!
//! Stage 2 (`b = 4` or `8`): domain `{0, 1, Infinity}^2`, 9-point range-image
//! grid built from an equality-weighted histogram of witness quad digit
//! classes. The relation term is linear in the witness and does not use the
//! grid.

use crate::opaque::sumcheck::prefix_lookup::*;
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
pub(super) struct Stage2PrefixCache<E: Field> {
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
    pub(super) fn from_norm_histogram(
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
    pub(super) fn round0_norm_poly(&self) -> UnivariatePoly<E> {
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
    pub(super) fn round1_norm_poly(&self, r0: E) -> UnivariatePoly<E> {
        let norm_y_values: [E; 3] =
            std::array::from_fn(|y| eval_quadratic_from_coeffs(self.norm_x_row_coeffs[y], r0));
        let norm_q =
            quadratic_coeffs_from_01_inf(norm_y_values[0], norm_y_values[1], norm_y_values[2]);
        let scale = self.batching_coeff * linear_eq_eval(self.tau0, r0);
        let coeffs = mul_linear_by_quadratic_coeffs(self.tau1, norm_q).map(|coeff| scale * coeff);
        UnivariatePoly::new(coeffs.to_vec())
    }
}

const STAGE2_B4_W_VALUES: [i64; 4] = [-2, -1, 0, 1];

const STAGE2_B8_W_VALUES: [i64; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];

const STAGE2_PREFIX_POINT_COUNT: usize = 9;

const STAGE2_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE2_PREFIX_POINT_COUNT] = [
    (0, 0),
    (0, 1),
    (0, LOOKUP_PREFIX_INF),
    (1, 0),
    (1, 1),
    (1, LOOKUP_PREFIX_INF),
    (LOOKUP_PREFIX_INF, 0),
    (LOOKUP_PREFIX_INF, 1),
    (LOOKUP_PREFIX_INF, LOOKUP_PREFIX_INF),
];

#[inline(always)]
const fn stage2_b4_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 2) | (digits[2] << 4) | (digits[3] << 6)
}

#[inline(always)]
const fn stage2_b8_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 3) | (digits[2] << 6) | (digits[3] << 9)
}

const fn stage2_local_norm_raw_eval_i64(w_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let w_eval = lookup_bilinear_eval_on_prefix_points(w_quad, x, y);
    if x == LOOKUP_PREFIX_INF || y == LOOKUP_PREFIX_INF {
        w_eval * w_eval
    } else {
        w_eval * (w_eval + 1)
    }
}

const fn build_stage2_b4_norm_lookup_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] {
    let mut table = [[0i64; STAGE2_PREFIX_POINT_COUNT]; 256];
    let mut d0 = 0usize;
    while d0 < 4 {
        let mut d1 = 0usize;
        while d1 < 4 {
            let mut d2 = 0usize;
            while d2 < 4 {
                let mut d3 = 0usize;
                while d3 < 4 {
                    let quad = [
                        STAGE2_B4_W_VALUES[d0],
                        STAGE2_B4_W_VALUES[d1],
                        STAGE2_B4_W_VALUES[d2],
                        STAGE2_B4_W_VALUES[d3],
                    ];
                    let table_idx = stage2_b4_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE2_PREFIX_POINT_COUNT {
                        let (x, y) = STAGE2_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage2_local_norm_raw_eval_i64(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

static STAGE2_B4_NORM_LOOKUP_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] =
    build_stage2_b4_norm_lookup_table();

const fn build_stage2_b8_norm_lookup_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] {
    let mut table = [[0i64; STAGE2_PREFIX_POINT_COUNT]; 4096];
    let mut d0 = 0usize;
    while d0 < 8 {
        let mut d1 = 0usize;
        while d1 < 8 {
            let mut d2 = 0usize;
            while d2 < 8 {
                let mut d3 = 0usize;
                while d3 < 8 {
                    let quad = [
                        STAGE2_B8_W_VALUES[d0],
                        STAGE2_B8_W_VALUES[d1],
                        STAGE2_B8_W_VALUES[d2],
                        STAGE2_B8_W_VALUES[d3],
                    ];
                    let table_idx = stage2_b8_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE2_PREFIX_POINT_COUNT {
                        let (x, y) = STAGE2_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage2_local_norm_raw_eval_i64(quad, x, y);
                        point_idx += 1;
                    }
                    d3 += 1;
                }
                d2 += 1;
            }
            d1 += 1;
        }
        d0 += 1;
    }
    table
}

static STAGE2_B8_NORM_LOOKUP_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] =
    build_stage2_b8_norm_lookup_table();

#[cfg(test)]
mod tests;
