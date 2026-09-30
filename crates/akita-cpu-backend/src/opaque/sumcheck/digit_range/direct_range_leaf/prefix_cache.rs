//! Two-round prefix cache for the Stage 1 range check: rounds 0 and 1 from
//! one grid over quad digit classes.
//!
//! Stage 1 (`b = 4`): domain `{0, 1, Infinity}^2`, 9-point internal grid with
//! the four Boolean corners omitted (5 cached values).
//!
//! Stage 1 (`b = 8`): domain `{0, 1, -1, 2, Infinity}^2`, 25-point internal
//! grid with the four Boolean corners omitted (21 cached values).

use crate::opaque::sumcheck::digit_range::range_poly::RangePoly;
use crate::opaque::sumcheck::prefix_lookup::*;
use akita_sumcheck::reduce_signed_accum;
use jolt_field::{Field, Ring, Unreduced, Zero};
use jolt_poly::OmittedConstantPoly;
use jolt_poly::UnivariatePoly;

/// Internal stage-1 first-two-round interpolation grid.
///
/// This is built and consumed inside the prover to reconstruct ordinary
/// eq-factored sumcheck round messages; it is not serialized in the Akita proof.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage1PrefixGrid<E: Field> {
    evals_except_boolean_core: Vec<E>,
}

#[inline]
fn stage1_full_grid_index(x_idx: usize, y_idx: usize) -> usize {
    x_idx * 5 + y_idx
}

#[inline]
fn stage1_is_boolean_corner(x_idx: usize, y_idx: usize) -> bool {
    x_idx < 2 && y_idx < 2
}

#[inline]
fn stage1_quartic_coeffs_from_prefix_values<E: Field + Ring>(values: [E; 5]) -> [E; 5] {
    let [at_0, at_1, at_neg_1, at_2, at_inf] = values;
    let two_inv = E::from_u64(2)
        .inverse()
        .expect("stage1 prefix interpolation requires 2 to be invertible");
    let three_inv = E::from_u64(3)
        .inverse()
        .expect("stage1 prefix interpolation requires 3 to be invertible");

    let a0 = at_0;
    let a4 = at_inf;
    let rhs_at_1 = at_1 - a0 - a4;
    let rhs_at_neg_1 = at_neg_1 - a0 - a4;
    let a2 = (rhs_at_1 + rhs_at_neg_1) * two_inv;
    let a1_plus_a3 = (rhs_at_1 - rhs_at_neg_1) * two_inv;
    let rhs_at_2 = at_2 - a0 - E::from_u64(16) * a4;
    let a1_plus_4a3 = rhs_at_2 * two_inv - E::from_u64(2) * a2;
    let a3 = (a1_plus_4a3 - a1_plus_a3) * three_inv;
    let a1 = a1_plus_a3 - a3;
    [a0, a1, a2, a3, a4]
}

#[inline]
fn stage1_eval_quartic_from_prefix_values<E: Field + Ring>(values: [E; 5], x: E) -> E {
    let [a0, a1, a2, a3, a4] = stage1_quartic_coeffs_from_prefix_values(values);
    a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
}

#[inline]
fn eval_stage1_biquartic_from_full_grid<E: Field + Ring>(full_grid: [E; 25], x: E, y: E) -> E {
    let x_rows = std::array::from_fn(|x_idx| {
        stage1_eval_quartic_from_prefix_values(
            [
                full_grid[stage1_full_grid_index(x_idx, 0)],
                full_grid[stage1_full_grid_index(x_idx, 1)],
                full_grid[stage1_full_grid_index(x_idx, 2)],
                full_grid[stage1_full_grid_index(x_idx, 3)],
                full_grid[stage1_full_grid_index(x_idx, 4)],
            ],
            y,
        )
    });
    stage1_eval_quartic_from_prefix_values(x_rows, x)
}

/// Build the cache for the first two stage-1 rounds.
///
/// `quad_class_weights[c]` is the sum of `eq(tau0[2..], q)` over the live quads
/// `q` of class `c`, where a quad's class packs the classes `k` of its four
/// range images `k(k+1)`, first entry lowest. Quads of class zero contribute
/// nothing, so their weight may be omitted.
#[tracing::instrument(skip_all, name = "two_round_prefix::build_stage1_prefix_cache")]
pub(super) fn build_stage1_prefix_cache<E: Field + Ring + Unreduced>(
    quad_class_weights: &[E],
    tau0: &[E],
    b: usize,
) -> Option<Stage1PrefixCache<E>> {
    Stage1PrefixCache::new(&build_stage1_prefix_grid(quad_class_weights, b), tau0, b)
}

fn build_stage1_prefix_grid<E: Field + Ring + Unreduced>(
    quad_class_weights: &[E],
    b: usize,
) -> Stage1PrefixGrid<E> {
    let evals_except_boolean_core = match b {
        4 => accumulate_prefix_lookup_rows(quad_class_weights, &STAGE1_B4_PREFIX_LOOKUP_TABLE),
        8 => accumulate_prefix_lookup_rows(quad_class_weights, &STAGE1_B8_PREFIX_LOOKUP_TABLE),
        _ => unreachable!("unsupported stage-1 two-round prefix basis"),
    };
    Stage1PrefixGrid {
        evals_except_boolean_core,
    }
}

fn accumulate_prefix_lookup_rows<E: Field + Unreduced, const N: usize>(
    weights: &[E],
    table: &[[i64; N]],
) -> Vec<E> {
    debug_assert_eq!(weights.len(), table.len());
    let mut pos = [E::SmallProduct::zero(); N];
    let mut neg = [E::SmallProduct::zero(); N];
    for (&weight, row) in weights.iter().zip(table) {
        if !weight.is_zero() {
            accum_lookup_vector_signed(&mut pos, &mut neg, weight, row);
        }
    }
    pos.into_iter()
        .zip(neg)
        .map(|(pos, neg)| reduce_signed_accum::<E>(pos, neg))
        .collect()
}

/// Cache needed to reconstruct the first two ordinary stage-1 round messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stage1B4PrefixCache<E: Field> {
    x_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stage1B8PrefixCache<E: Field> {
    full_grid: [E; 25],
    tau0: E,
    tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Stage1PrefixCache<E: Field> {
    B4(Stage1B4PrefixCache<E>),
    B8(Stage1B8PrefixCache<E>),
}

impl<E: Field + Ring> Stage1PrefixCache<E> {
    fn new(proof: &Stage1PrefixGrid<E>, tau0: &[E], b: usize) -> Option<Self> {
        if tau0.len() < 2 {
            return None;
        }

        match b {
            4 => {
                if proof.evals_except_boolean_core.len() != STAGE1_B4_PREFIX_EVAL_COUNT {
                    return None;
                }
                let mut full_grid = [E::zero(); 9];
                for (payload_idx, &grid_idx) in STAGE1_B4_NONBOOLEAN_GRID_INDICES.iter().enumerate()
                {
                    full_grid[grid_idx] = proof.evals_except_boolean_core[payload_idx];
                }
                let x_row_coeffs = std::array::from_fn(|y_idx| {
                    quadratic_coeffs_from_01_inf(
                        full_grid[y_idx],
                        full_grid[3 + y_idx],
                        full_grid[6 + y_idx],
                    )
                });
                Some(Self::B4(Stage1B4PrefixCache {
                    x_row_coeffs,
                    tau0: tau0[0],
                    tau1: tau0[1],
                }))
            }
            8 => {
                if proof.evals_except_boolean_core.len() != STAGE1_PREFIX_EVAL_COUNT {
                    return None;
                }

                let mut full_grid = [E::zero(); 25];
                let mut payload_idx = 0usize;
                for x_idx in 0..5 {
                    for y_idx in 0..5 {
                        if stage1_is_boolean_corner(x_idx, y_idx) {
                            continue;
                        }
                        full_grid[stage1_full_grid_index(x_idx, y_idx)] =
                            proof.evals_except_boolean_core[payload_idx];
                        payload_idx += 1;
                    }
                }

                Some(Self::B8(Stage1B8PrefixCache {
                    full_grid,
                    tau0: tau0[0],
                    tau1: tau0[1],
                }))
            }
            _ => None,
        }
    }

    #[cfg(test)]
    fn b4_test_data(&self) -> Option<([[E; 3]; 3], E, E)> {
        match self {
            Self::B4(state) => Some((state.x_row_coeffs, state.tau0, state.tau1)),
            Self::B8(_) => None,
        }
    }

    #[cfg(test)]
    fn b8_test_data(&self) -> Option<([E; 25], E, E)> {
        match self {
            Self::B4(_) => None,
            Self::B8(state) => Some((state.full_grid, state.tau0, state.tau1)),
        }
    }

    pub(super) fn reconstruct_round0_eq_poly(&self) -> OmittedConstantPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round0_eq_poly(),
            Self::B8(state) => state.reconstruct_round0_eq_poly(),
        }
    }

    pub(super) fn reconstruct_round1_eq_poly(&self, r0: E) -> OmittedConstantPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round1_eq_poly(r0),
            Self::B8(state) => state.reconstruct_round1_eq_poly(r0),
        }
    }
}

impl<E: Field + Ring> Stage1B4PrefixCache<E> {
    fn reconstruct_round0_eq_poly(&self) -> OmittedConstantPoly<E> {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.x_row_coeffs[1], self.tau1),
        );
        OmittedConstantPoly::from_q_coefficients(q_x.into())
    }

    fn reconstruct_round1_eq_poly(&self, r0: E) -> OmittedConstantPoly<E> {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(self.x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        OmittedConstantPoly::from_q_coefficients(q_y.into())
    }
}

impl<E: Field + Ring> Stage1B8PrefixCache<E> {
    fn reconstruct_round0_eq_poly(&self) -> OmittedConstantPoly<E> {
        let l1_at_0 = E::one() - self.tau1;
        let l1_at_1 = self.tau1;
        let evals: Vec<E> = (0..=4u64)
            .map(|x_raw| {
                let x = E::from_u64(x_raw);
                let q_x0 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::zero());
                let q_x1 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::one());
                l1_at_0 * q_x0 + l1_at_1 * q_x1
            })
            .collect();
        interpolate_eq_factored_q_poly(&evals, STAGE1_B8_Q_POLY_DEGREE)
    }

    fn reconstruct_round1_eq_poly(&self, r0: E) -> OmittedConstantPoly<E> {
        let evals: Vec<E> = (0..=4u64)
            .map(|y_raw| {
                let y = E::from_u64(y_raw);
                eval_stage1_biquartic_from_full_grid(self.full_grid, r0, y)
            })
            .collect();
        interpolate_eq_factored_q_poly(&evals, STAGE1_B8_Q_POLY_DEGREE)
    }
}

/// Number of cached evaluations in the stage-1 `b = 4` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,Infinity}^2`.
const STAGE1_B4_PREFIX_EVAL_COUNT: usize = 5;

const STAGE1_B4_NONBOOLEAN_GRID_INDICES: [usize; STAGE1_B4_PREFIX_EVAL_COUNT] = [2, 5, 6, 7, 8];

/// Number of cached evaluations in the stage-1 `b = 8` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,-1,2,Infinity}^2`.
const STAGE1_PREFIX_EVAL_COUNT: usize = 21;

const STAGE1_B8_Q_POLY_DEGREE: usize = 4;

const STAGE1_B4_S_VALUES: [i64; 2] = [0, 2];

const STAGE1_B8_S_VALUES: [i64; 4] = [0, 2, 6, 12];

const fn pow_i64(mut base: i64, mut exp: usize) -> i64 {
    let mut out = 1i64;
    while exp > 0 {
        if exp & 1 == 1 {
            out *= base;
        }
        exp >>= 1;
        if exp > 0 {
            base *= base;
        }
    }
    out
}

const fn stage1_b4_local_norm_raw_eval_i64(s_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let [_, bx, cy, dxy] = lookup_bilinear_coeffs_from_quad(s_quad);
    let x_is_inf = x == LOOKUP_PREFIX_INF;
    let y_is_inf = y == LOOKUP_PREFIX_INF;
    if !x_is_inf && !y_is_inf {
        RangePoly::new(4).eval_i64(lookup_bilinear_eval_on_prefix_points(s_quad, x, y))
    } else if x_is_inf && !y_is_inf {
        let linear = bx + y * dxy;
        linear * linear
    } else if !x_is_inf && y_is_inf {
        let linear = cy + x * dxy;
        linear * linear
    } else {
        dxy * dxy
    }
}

const fn stage1_b8_local_norm_raw_eval_i64(s_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let [_, bx, cy, dxy] = lookup_bilinear_coeffs_from_quad(s_quad);
    let x_is_inf = x == LOOKUP_PREFIX_INF;
    let y_is_inf = y == LOOKUP_PREFIX_INF;
    if !x_is_inf && !y_is_inf {
        RangePoly::new(8).eval_i64(lookup_bilinear_eval_on_prefix_points(s_quad, x, y))
    } else if x_is_inf && !y_is_inf {
        pow_i64(bx + y * dxy, RangePoly::new(8).num_coefficients())
    } else if !x_is_inf && y_is_inf {
        pow_i64(cy + x * dxy, RangePoly::new(8).num_coefficients())
    } else {
        pow_i64(dxy, RangePoly::new(8).num_coefficients())
    }
}

const STAGE1_B4_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE1_B4_PREFIX_EVAL_COUNT] = [
    (0, LOOKUP_PREFIX_INF),
    (1, LOOKUP_PREFIX_INF),
    (LOOKUP_PREFIX_INF, 0),
    (LOOKUP_PREFIX_INF, 1),
    (LOOKUP_PREFIX_INF, LOOKUP_PREFIX_INF),
];

const fn stage1_lookup_points_i64() -> [(i64, i64); STAGE1_PREFIX_EVAL_COUNT] {
    let coords = [0i64, 1, -1, 2, LOOKUP_PREFIX_INF];
    let mut out = [(0i64, 0i64); STAGE1_PREFIX_EVAL_COUNT];
    let mut out_idx = 0usize;
    let mut x_idx = 0usize;
    while x_idx < 5 {
        let mut y_idx = 0usize;
        while y_idx < 5 {
            if !(x_idx < 2 && y_idx < 2) {
                out[out_idx] = (coords[x_idx], coords[y_idx]);
                out_idx += 1;
            }
            y_idx += 1;
        }
        x_idx += 1;
    }
    out
}

const STAGE1_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE1_PREFIX_EVAL_COUNT] =
    stage1_lookup_points_i64();

#[inline(always)]
const fn stage1_b4_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 1) | (digits[2] << 2) | (digits[3] << 3)
}

#[inline(always)]
const fn stage1_b8_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 2) | (digits[2] << 4) | (digits[3] << 6)
}

const fn build_stage1_b4_prefix_lookup_table() -> [[i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16] {
    let mut table = [[0i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16];
    let mut d0 = 0usize;
    while d0 < 2 {
        let mut d1 = 0usize;
        while d1 < 2 {
            let mut d2 = 0usize;
            while d2 < 2 {
                let mut d3 = 0usize;
                while d3 < 2 {
                    let quad = [
                        STAGE1_B4_S_VALUES[d0],
                        STAGE1_B4_S_VALUES[d1],
                        STAGE1_B4_S_VALUES[d2],
                        STAGE1_B4_S_VALUES[d3],
                    ];
                    let table_idx = stage1_b4_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE1_B4_PREFIX_EVAL_COUNT {
                        let (x, y) = STAGE1_B4_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage1_b4_local_norm_raw_eval_i64(quad, x, y);
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

static STAGE1_B4_PREFIX_LOOKUP_TABLE: [[i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16] =
    build_stage1_b4_prefix_lookup_table();

const fn build_stage1_b8_prefix_lookup_table() -> [[i64; STAGE1_PREFIX_EVAL_COUNT]; 256] {
    let mut table = [[0i64; STAGE1_PREFIX_EVAL_COUNT]; 256];
    let mut d0 = 0usize;
    while d0 < 4 {
        let mut d1 = 0usize;
        while d1 < 4 {
            let mut d2 = 0usize;
            while d2 < 4 {
                let mut d3 = 0usize;
                while d3 < 4 {
                    let quad = [
                        STAGE1_B8_S_VALUES[d0],
                        STAGE1_B8_S_VALUES[d1],
                        STAGE1_B8_S_VALUES[d2],
                        STAGE1_B8_S_VALUES[d3],
                    ];
                    let table_idx = stage1_b8_lookup_index_from_digits([d0, d1, d2, d3]);
                    let mut point_idx = 0usize;
                    while point_idx < STAGE1_PREFIX_EVAL_COUNT {
                        let (x, y) = STAGE1_PREFIX_LOOKUP_POINTS_I64[point_idx];
                        table[table_idx][point_idx] = stage1_b8_local_norm_raw_eval_i64(quad, x, y);
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

static STAGE1_B8_PREFIX_LOOKUP_TABLE: [[i64; STAGE1_PREFIX_EVAL_COUNT]; 256] =
    build_stage1_b8_prefix_lookup_table();

fn interpolate_eq_factored_q_poly<E: Field + Ring>(
    evals: &[E],
    degree: usize,
) -> OmittedConstantPoly<E> {
    let mut q_poly = UnivariatePoly::from_evals(evals);
    q_poly.trim_trailing_zeros();
    let mut q_coeffs = q_poly.into_coefficients();
    q_coeffs.resize(degree + 1, E::zero());
    OmittedConstantPoly::from_q_coefficients(q_coeffs)
}

#[cfg(test)]
mod tests;
