use crate::opaque::sumcheck::digit_range::range_poly::RangePoly;
use jolt_field::Unreduced;
use jolt_field::{Field, Ring};
use jolt_poly::{OmittedConstantPoly, UnivariatePoly};

/// Number of cached evaluations in the stage-1 `b = 4` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,Infinity}^2`.
pub(crate) const STAGE1_B4_PREFIX_EVAL_COUNT: usize = 5;

pub(crate) const STAGE1_B4_NONBOOLEAN_GRID_INDICES: [usize; STAGE1_B4_PREFIX_EVAL_COUNT] =
    [2, 5, 6, 7, 8];

/// Number of cached evaluations in the stage-1 `b = 8` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,-1,2,Infinity}^2`.
pub(crate) const STAGE1_PREFIX_EVAL_COUNT: usize = 21;
pub(crate) const STAGE1_B8_Q_POLY_DEGREE: usize = 4;

pub(crate) const LOOKUP_PREFIX_INF: i64 = i64::MIN;
pub(crate) const STAGE1_B4_S_VALUES: [i64; 2] = [0, 2];
pub(crate) const STAGE1_B8_S_VALUES: [i64; 4] = [0, 2, 6, 12];
pub(crate) const STAGE2_B4_W_VALUES: [i64; 4] = [-2, -1, 0, 1];
pub(crate) const STAGE2_B8_W_VALUES: [i64; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
pub(crate) const STAGE2_PREFIX_POINT_COUNT: usize = 9;

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

pub(crate) const fn pow_i64(mut base: i64, mut exp: usize) -> i64 {
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

pub(crate) const fn stage1_b4_local_norm_raw_eval_i64(s_quad: [i64; 4], x: i64, y: i64) -> i64 {
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

pub(crate) const fn stage1_b8_local_norm_raw_eval_i64(s_quad: [i64; 4], x: i64, y: i64) -> i64 {
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

pub(crate) const STAGE1_B4_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE1_B4_PREFIX_EVAL_COUNT] = [
    (0, LOOKUP_PREFIX_INF),
    (1, LOOKUP_PREFIX_INF),
    (LOOKUP_PREFIX_INF, 0),
    (LOOKUP_PREFIX_INF, 1),
    (LOOKUP_PREFIX_INF, LOOKUP_PREFIX_INF),
];

pub(crate) const fn stage1_lookup_points_i64() -> [(i64, i64); STAGE1_PREFIX_EVAL_COUNT] {
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

pub(crate) const STAGE1_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE1_PREFIX_EVAL_COUNT] =
    stage1_lookup_points_i64();

#[inline(always)]
pub(crate) const fn stage1_b4_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 1) | (digits[2] << 2) | (digits[3] << 3)
}

#[inline(always)]
pub(crate) const fn stage1_b8_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 2) | (digits[2] << 4) | (digits[3] << 6)
}

pub(crate) const fn build_stage1_b4_prefix_lookup_table() -> [[i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16]
{
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

pub(crate) static STAGE1_B4_PREFIX_LOOKUP_TABLE: [[i64; STAGE1_B4_PREFIX_EVAL_COUNT]; 16] =
    build_stage1_b4_prefix_lookup_table();

pub(crate) const fn build_stage1_b8_prefix_lookup_table() -> [[i64; STAGE1_PREFIX_EVAL_COUNT]; 256]
{
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

pub(crate) static STAGE1_B8_PREFIX_LOOKUP_TABLE: [[i64; STAGE1_PREFIX_EVAL_COUNT]; 256] =
    build_stage1_b8_prefix_lookup_table();

pub(crate) const STAGE2_PREFIX_LOOKUP_POINTS_I64: [(i64, i64); STAGE2_PREFIX_POINT_COUNT] = [
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
pub(crate) const fn stage2_b4_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 2) | (digits[2] << 4) | (digits[3] << 6)
}

#[inline(always)]
pub(crate) const fn stage2_b8_lookup_index_from_digits(digits: [usize; 4]) -> usize {
    digits[0] | (digits[1] << 3) | (digits[2] << 6) | (digits[3] << 9)
}

pub(crate) const fn stage2_local_norm_raw_eval_i64(w_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let w_eval = lookup_bilinear_eval_on_prefix_points(w_quad, x, y);
    if x == LOOKUP_PREFIX_INF || y == LOOKUP_PREFIX_INF {
        w_eval * w_eval
    } else {
        w_eval * (w_eval + 1)
    }
}

pub(crate) const fn build_stage2_b4_norm_lookup_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] {
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

pub(crate) static STAGE2_B4_NORM_LOOKUP_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] =
    build_stage2_b4_norm_lookup_table();

pub(crate) const fn build_stage2_b8_norm_lookup_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096]
{
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

pub(crate) static STAGE2_B8_NORM_LOOKUP_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] =
    build_stage2_b8_norm_lookup_table();

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

pub(crate) fn interpolate_eq_factored_q_poly<E: Field + Ring>(
    evals: &[E],
    degree: usize,
) -> OmittedConstantPoly<E> {
    let mut q_poly = UnivariatePoly::from_evals(evals);
    q_poly.trim_trailing_zeros();
    let mut q_coeffs = q_poly.into_coefficients();
    q_coeffs.resize(degree + 1, E::zero());
    OmittedConstantPoly::from_q_coefficients(q_coeffs)
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
pub(crate) fn linear_eq_coeffs<E: Field>(tau: E) -> [E; 2] {
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
