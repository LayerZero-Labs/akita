//! Prover-internal two-round-prefix kernels for Akita stages 1 and 2.
//!
//! When the stage-specific prefix gate fires, the first two rounds of each
//! stage's sumcheck can be collapsed into a single bivariate evaluation over
//! a 4-value inner-dimension quad. The prover builds a local compressed grid
//! and immediately reconstructs the two ordinary sumcheck round messages from
//! it. Those reconstructed messages are then passed to the normal generic
//! sumcheck drivers and serialized as ordinary `SumcheckProof` or
//! `EqFactoredSumcheckProof` rounds.
//!
//! The bivariate-skip grids in this module are not part of the public proof
//! object or verifier API. They are transient prover-side payloads used to avoid
//! expensive scans over compact witness tables before the witness is folded to
//! round 2.
//!
//! Point semantics for the evaluation domains:
//!
//! - Finite points are ordinary evaluations of the bilinear multilinear
//!   extension over the quad.
//! - `Infinity` means "take the leading coefficient in that coordinate".
//!
//! Stage 1 (`b = 4`): domain `{0, 1, Infinity}^2`, 9-point internal grid with
//! the four Boolean corners omitted (5 cached values).
//!
//! Stage 1 (`b = 8`): domain `{0, 1, -1, 2, Infinity}^2`, 25-point internal
//! grid with the four Boolean corners omitted (21 cached values).
//!
//! Stage 2 (`b = 8`): domain `{0, 1, Infinity}^2`, 9-point internal grid. The
//! norm and relation families each cache a compressed grid with one Boolean
//! corner omitted (8 values each), recovered via the known claim before ordinary
//! round polynomials are emitted.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::fields::HasUnreducedOps;
use akita_field::parallel::*;
use akita_field::{AdditiveGroup, FieldCore, FromSmallInt};
use akita_sumcheck::{reduce_signed_accum, EqFactoredUniPoly, UniPoly};
#[cfg(test)]
use akita_types::range_check_eval_from_s;

/// Point in a small evaluation domain used by the 2-round prefix kernels.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrefixPoint<E: FieldCore> {
    Finite(E),
    Infinity,
}

/// Candidate stage-1 domain `{1, -1, 2, Infinity}`.
#[cfg(test)]
pub(crate) fn stage1_prefix_points<E: FieldCore + FromSmallInt>() -> [PrefixPoint<E>; 4] {
    [
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

/// Safe full stage-1 fallback domain `{0, 1, -1, 2, Infinity}`.
#[cfg(test)]
pub(crate) fn stage1_full_prefix_points<E: FieldCore + FromSmallInt>() -> [PrefixPoint<E>; 5] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

/// Number of cached evaluations in the stage-1 `b = 4` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,Infinity}^2`.
pub(crate) const STAGE1_B4_PREFIX_EVAL_COUNT: usize = 5;

const STAGE1_B4_NONBOOLEAN_GRID_INDICES: [usize; STAGE1_B4_PREFIX_EVAL_COUNT] = [2, 5, 6, 7, 8];

/// Number of cached evaluations in the stage-1 `b = 8` two-round-prefix grid
/// after omitting the four Boolean corners from `{0,1,-1,2,Infinity}^2`.
pub(crate) const STAGE1_PREFIX_EVAL_COUNT: usize = 21;
const STAGE1_B8_Q_POLY_DEGREE: usize = 4;

/// Internal stage-1 first-two-round bivariate-skip payload.
///
/// This is built and consumed inside the prover to reconstruct ordinary
/// eq-factored sumcheck round messages; it is not serialized in the Akita proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1BivariateSkipProof<E: FieldCore> {
    pub evals_except_boolean_core: Vec<E>,
}

#[inline]
fn stage1_full_grid_index(x_idx: usize, y_idx: usize) -> usize {
    x_idx * 5 + y_idx
}

#[inline]
fn stage1_is_boolean_corner(x_idx: usize, y_idx: usize) -> bool {
    x_idx < 2 && y_idx < 2
}

const LOOKUP_PREFIX_INF: i64 = i64::MIN;
const STAGE1_B4_S_VALUES: [i64; 2] = [0, 2];
const STAGE1_B8_S_VALUES: [i64; 4] = [0, 2, 6, 12];
const STAGE2_B4_W_VALUES: [i64; 4] = [-2, -1, 0, 1];
const STAGE2_B8_W_VALUES: [i64; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
const STAGE2_PREFIX_POINT_COUNT: usize = 9;
const STAGE2_COMPRESSED_POINT_COUNT: usize = STAGE2_PREFIX_POINT_COUNT - 1;
const STAGE2_COMPRESSED_POINT_INDICES_BY_OMITTED_CORNER: [[usize; STAGE2_COMPRESSED_POINT_COUNT];
    4] = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [0, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 5, 6, 7, 8],
];

const fn lookup_bilinear_coeffs_from_quad(quad: [i64; 4]) -> [i64; 4] {
    let [t00, t10, t01, t11] = quad;
    [t00, t10 - t00, t01 - t00, t11 - t10 - t01 + t00]
}

const fn lookup_bilinear_eval_on_prefix_points(quad: [i64; 4], x: i64, y: i64) -> i64 {
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

const fn stage1_b8_range_check_from_s(s: i64) -> i64 {
    s * (s - 2) * (s - 6) * (s - 12)
}

const fn stage1_b4_range_check_from_s(s: i64) -> i64 {
    s * (s - 2)
}

const fn stage1_b4_local_norm_raw_eval_i64(s_quad: [i64; 4], x: i64, y: i64) -> i64 {
    let [_, bx, cy, dxy] = lookup_bilinear_coeffs_from_quad(s_quad);
    let x_is_inf = x == LOOKUP_PREFIX_INF;
    let y_is_inf = y == LOOKUP_PREFIX_INF;
    if !x_is_inf && !y_is_inf {
        stage1_b4_range_check_from_s(lookup_bilinear_eval_on_prefix_points(s_quad, x, y))
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
        stage1_b8_range_check_from_s(lookup_bilinear_eval_on_prefix_points(s_quad, x, y))
    } else if x_is_inf && !y_is_inf {
        pow_i64(bx + y * dxy, 4)
    } else if !x_is_inf && y_is_inf {
        pow_i64(cy + x * dxy, 4)
    } else {
        pow_i64(dxy, 4)
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

const fn compress_stage2_lookup_values(
    values: [i64; STAGE2_PREFIX_POINT_COUNT],
    omitted_idx: usize,
) -> [i64; STAGE2_COMPRESSED_POINT_COUNT] {
    let mut out = [0i64; STAGE2_COMPRESSED_POINT_COUNT];
    let mut src_idx = 0usize;
    let mut dst_idx = 0usize;
    while src_idx < STAGE2_PREFIX_POINT_COUNT {
        if src_idx != omitted_idx {
            out[dst_idx] = values[src_idx];
            dst_idx += 1;
        }
        src_idx += 1;
    }
    out
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

const fn build_stage2_b4_relation_weight_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] {
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
                        table[table_idx][point_idx] =
                            lookup_bilinear_eval_on_prefix_points(quad, x, y);
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

static STAGE2_B4_RELATION_WEIGHT_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 256] =
    build_stage2_b4_relation_weight_table();

const fn build_stage2_b4_relation_weight_compressed_table(
) -> [[i64; STAGE2_COMPRESSED_POINT_COUNT]; 256] {
    let mut table = [[0i64; STAGE2_COMPRESSED_POINT_COUNT]; 256];
    let mut table_idx = 0usize;
    while table_idx < 256 {
        table[table_idx] =
            compress_stage2_lookup_values(STAGE2_B4_RELATION_WEIGHT_TABLE[table_idx], 0);
        table_idx += 1;
    }
    table
}

static STAGE2_B4_RELATION_WEIGHT_COMPRESSED_TABLE: [[i64; STAGE2_COMPRESSED_POINT_COUNT]; 256] =
    build_stage2_b4_relation_weight_compressed_table();

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

const fn build_stage2_b8_relation_weight_table() -> [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] {
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
                        table[table_idx][point_idx] =
                            lookup_bilinear_eval_on_prefix_points(quad, x, y);
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

static STAGE2_B8_RELATION_WEIGHT_TABLE: [[i64; STAGE2_PREFIX_POINT_COUNT]; 4096] =
    build_stage2_b8_relation_weight_table();

const fn build_stage2_b8_relation_weight_compressed_table(
) -> [[i64; STAGE2_COMPRESSED_POINT_COUNT]; 4096] {
    let mut table = [[0i64; STAGE2_COMPRESSED_POINT_COUNT]; 4096];
    let mut table_idx = 0usize;
    while table_idx < 4096 {
        table[table_idx] =
            compress_stage2_lookup_values(STAGE2_B8_RELATION_WEIGHT_TABLE[table_idx], 0);
        table_idx += 1;
    }
    table
}

static STAGE2_B8_RELATION_WEIGHT_COMPRESSED_TABLE: [[i64; STAGE2_COMPRESSED_POINT_COUNT]; 4096] =
    build_stage2_b8_relation_weight_compressed_table();

#[inline]
fn accum_lookup_vector_signed<E: FieldCore + HasUnreducedOps, const N: usize>(
    pos: &mut [E::MulU64Accum; N],
    neg: &mut [E::MulU64Accum; N],
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
fn accum_lookup_vector_signed_selected<
    E: FieldCore + HasUnreducedOps,
    const N: usize,
    const M: usize,
>(
    pos: &mut [E::MulU64Accum; N],
    neg: &mut [E::MulU64Accum; N],
    coeff: E,
    values: &[i64; M],
    selected_indices: &[usize; N],
) {
    for (dst_idx, &src_idx) in selected_indices.iter().enumerate() {
        let value = values[src_idx];
        if value > 0 {
            pos[dst_idx] += coeff.mul_u64_unreduced(value as u64);
        } else if value < 0 {
            neg[dst_idx] += coeff.mul_u64_unreduced(value.unsigned_abs());
        }
    }
}

#[inline]
fn accum_pointwise_signed<E: FieldCore + HasUnreducedOps, const N: usize>(
    pos: &mut [E::MulU64Accum; N],
    neg: &mut [E::MulU64Accum; N],
    coeffs: &[E; N],
    weights: &[i64; N],
) {
    for (idx, (&coeff, &weight)) in coeffs.iter().zip(weights.iter()).enumerate() {
        if weight > 0 {
            pos[idx] += coeff.mul_u64_unreduced(weight as u64);
        } else if weight < 0 {
            neg[idx] += coeff.mul_u64_unreduced(weight.unsigned_abs());
        }
    }
}

#[inline(always)]
pub(super) fn stage1_b4_s_digit_from_compact_s(s: i16) -> usize {
    match s {
        0 => 0,
        2 => 1,
        other => unreachable!("unexpected compact s value {other}"),
    }
}

#[inline(always)]
pub(super) fn stage1_b8_s_digit_from_compact_s(s: i16) -> usize {
    match s {
        0 => 0,
        2 => 1,
        6 => 2,
        12 => 3,
        other => unreachable!("unexpected compact s value {other}"),
    }
}

#[inline(always)]
pub(super) fn stage2_b4_w_digit(w: i8) -> usize {
    let w = i32::from(w);
    debug_assert!((-2..=1).contains(&w));
    (w + 2) as usize
}

#[inline(always)]
pub(super) fn stage2_b8_w_digit(w: i8) -> usize {
    let w = i32::from(w);
    debug_assert!((-4..=3).contains(&w));
    (w + 4) as usize
}

#[inline]
fn linear_eq_eval<E: FieldCore>(tau: E, x: E) -> E {
    tau * x + (E::one() - tau) * (E::one() - x)
}

#[inline]
fn stage2_relation_m_point_values_compressed<E: FieldCore>(
    m_quad: [E; 4],
) -> [E; STAGE2_COMPRESSED_POINT_COUNT] {
    let m00 = m_quad[0];
    let m10 = m_quad[1];
    let m01 = m_quad[2];
    let m11 = m_quad[3];
    [
        m01,
        m01 - m00,
        m10,
        m11,
        m11 - m10,
        m10 - m00,
        m11 - m01,
        m11 - m10 - m01 + m00,
    ]
}

#[inline]
fn stage1_quartic_coeffs_from_prefix_values<E: FieldCore + FromSmallInt>(values: [E; 5]) -> [E; 5] {
    let [at_0, at_1, at_neg_1, at_2, at_inf] = values;
    let two_inv = E::from_u64(2)
        .inv()
        .expect("stage1 prefix interpolation requires 2 to be invertible");
    let three_inv = E::from_u64(3)
        .inv()
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
fn stage1_eval_quartic_from_prefix_values<E: FieldCore + FromSmallInt>(values: [E; 5], x: E) -> E {
    let [a0, a1, a2, a3, a4] = stage1_quartic_coeffs_from_prefix_values(values);
    a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
}

#[inline]
fn eval_stage1_biquartic_from_full_grid<E: FieldCore + FromSmallInt>(
    full_grid: [E; 25],
    x: E,
    y: E,
) -> E {
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

/// Whether stage 1 has enough leading y-rounds to use the 2-round prefix path.
#[inline]
pub(crate) fn can_use_stage1_two_round_prefix(ring_bits: usize, b: usize) -> bool {
    ring_bits >= 2 && matches!(b, 4 | 8)
}

/// Build the stage-1 first-two-round bivariate-skip payload from the compact
/// witness columns at the start of stage 1.
///
/// Returns `None` when there are fewer than two leading y-rounds to batch.
#[tracing::instrument(
    skip_all,
    name = "two_round_prefix::build_stage1_bivariate_skip_proof_from_compact"
)]
#[cfg(test)]
pub(crate) fn build_stage1_bivariate_skip_proof_from_compact<
    E: FieldCore + FromSmallInt + HasUnreducedOps,
>(
    w_compact: &[i8],
    tau0: &[E],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Option<Stage1BivariateSkipProof<E>> {
    let y_len = 1usize << ring_bits;
    assert_eq!(w_compact.len(), live_x_cols * y_len);
    let s_compact = w_compact
        .iter()
        .map(|&w| {
            let w = i32::from(w);
            (w * (w + 1)) as i16
        })
        .collect::<Vec<_>>();
    build_stage1_bivariate_skip_proof_from_s_compact(
        &s_compact,
        tau0,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    )
}

/// Build the stage-1 first-two-round bivariate-skip payload from the compact
/// `s = w(w+1)` table already materialized by the prover.
#[tracing::instrument(
    skip_all,
    name = "two_round_prefix::build_stage1_bivariate_skip_proof_from_s_compact"
)]
pub(crate) fn build_stage1_bivariate_skip_proof_from_s_compact<
    E: FieldCore + FromSmallInt + HasUnreducedOps,
>(
    s_compact: &[i16],
    tau0: &[E],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Option<Stage1BivariateSkipProof<E>> {
    if !can_use_stage1_two_round_prefix(ring_bits, b) {
        return None;
    }

    let y_len = 1usize << ring_bits;
    assert_eq!(s_compact.len(), live_x_cols * y_len);
    assert_eq!(tau0.len(), col_bits + ring_bits);

    let eq_y_suffix = EqPolynomial::evals(&tau0[2..ring_bits]);
    let eq_x = EqPolynomial::evals(&tau0[ring_bits..]);
    let y_quads = y_len / 4;
    debug_assert!(eq_y_suffix.len() >= y_quads);
    debug_assert!(eq_x.len() >= live_x_cols);

    let evals_except_boolean_core = match b {
        4 => {
            let (pos, neg) = cfg_fold_reduce!(
                0..live_x_cols,
                || {
                    (
                        [E::MulU64Accum::ZERO; STAGE1_B4_PREFIX_EVAL_COUNT],
                        [E::MulU64Accum::ZERO; STAGE1_B4_PREFIX_EVAL_COUNT],
                    )
                },
                |(mut pos, mut neg), x_col| {
                    let col = &s_compact[x_col * y_len..(x_col + 1) * y_len];
                    let eq_x_weight = eq_x[x_col];
                    for (y_quad, &eq_y_weight) in eq_y_suffix.iter().take(y_quads).enumerate() {
                        let base = 4 * y_quad;
                        let lookup_idx = stage1_b4_lookup_index_from_digits([
                            stage1_b4_s_digit_from_compact_s(col[base]),
                            stage1_b4_s_digit_from_compact_s(col[base + 1]),
                            stage1_b4_s_digit_from_compact_s(col[base + 2]),
                            stage1_b4_s_digit_from_compact_s(col[base + 3]),
                        ]);
                        let weight = eq_x_weight * eq_y_weight;
                        accum_lookup_vector_signed(
                            &mut pos,
                            &mut neg,
                            weight,
                            &STAGE1_B4_PREFIX_LOOKUP_TABLE[lookup_idx],
                        );
                    }
                    (pos, neg)
                },
                |(mut pos_a, mut neg_a), (pos_b, neg_b)| {
                    for (dst, src) in pos_a.iter_mut().zip(pos_b.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in neg_a.iter_mut().zip(neg_b.iter()) {
                        *dst += *src;
                    }
                    (pos_a, neg_a)
                }
            );
            (0..STAGE1_B4_PREFIX_EVAL_COUNT)
                .map(|idx| reduce_signed_accum::<E>(pos[idx], neg[idx]))
                .collect()
        }
        8 => {
            let (pos, neg) = cfg_fold_reduce!(
                0..live_x_cols,
                || {
                    (
                        [E::MulU64Accum::ZERO; STAGE1_PREFIX_EVAL_COUNT],
                        [E::MulU64Accum::ZERO; STAGE1_PREFIX_EVAL_COUNT],
                    )
                },
                |(mut pos, mut neg), x_col| {
                    let col = &s_compact[x_col * y_len..(x_col + 1) * y_len];
                    let eq_x_weight = eq_x[x_col];
                    for (y_quad, &eq_y_weight) in eq_y_suffix.iter().take(y_quads).enumerate() {
                        let base = 4 * y_quad;
                        let lookup_idx = stage1_b8_lookup_index_from_digits([
                            stage1_b8_s_digit_from_compact_s(col[base]),
                            stage1_b8_s_digit_from_compact_s(col[base + 1]),
                            stage1_b8_s_digit_from_compact_s(col[base + 2]),
                            stage1_b8_s_digit_from_compact_s(col[base + 3]),
                        ]);
                        let weight = eq_x_weight * eq_y_weight;
                        accum_lookup_vector_signed(
                            &mut pos,
                            &mut neg,
                            weight,
                            &STAGE1_B8_PREFIX_LOOKUP_TABLE[lookup_idx],
                        );
                    }
                    (pos, neg)
                },
                |(mut pos_a, mut neg_a), (pos_b, neg_b)| {
                    for (dst, src) in pos_a.iter_mut().zip(pos_b.iter()) {
                        *dst += *src;
                    }
                    for (dst, src) in neg_a.iter_mut().zip(neg_b.iter()) {
                        *dst += *src;
                    }
                    (pos_a, neg_a)
                }
            );
            (0..STAGE1_PREFIX_EVAL_COUNT)
                .map(|idx| reduce_signed_accum::<E>(pos[idx], neg[idx]))
                .collect()
        }
        _ => unreachable!("unsupported stage-1 two-round prefix basis"),
    };

    Some(Stage1BivariateSkipProof {
        evals_except_boolean_core,
    })
}

#[cfg(test)]
fn stage1_storage_vector_from_quad<E: FieldCore + FromSmallInt>(quad: [E; 4], b: usize) -> Vec<E> {
    let points = stage1_full_prefix_points::<E>();
    let mut out = Vec::with_capacity(STAGE1_PREFIX_EVAL_COUNT);
    for x_idx in 0..5 {
        for y_idx in 0..5 {
            if stage1_is_boolean_corner(x_idx, y_idx) {
                continue;
            }
            out.push(stage1_local_norm_raw_eval(
                quad,
                points[x_idx],
                points[y_idx],
                b,
            ));
        }
    }
    out
}

/// State needed to reconstruct the first two ordinary stage-1 round messages
/// from the internal bivariate-skip payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1B4BivariateSkipState<E: FieldCore> {
    x_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1B8BivariateSkipState<E: FieldCore> {
    full_grid: [E; 25],
    tau0: E,
    tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Stage1BivariateSkipState<E: FieldCore> {
    B4(Stage1B4BivariateSkipState<E>),
    B8(Stage1B8BivariateSkipState<E>),
}

impl<E: FieldCore + FromSmallInt> Stage1BivariateSkipState<E> {
    pub(crate) fn new(proof: &Stage1BivariateSkipProof<E>, tau0: &[E], b: usize) -> Option<Self> {
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
                Some(Self::B4(Stage1B4BivariateSkipState {
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

                Some(Self::B8(Stage1B8BivariateSkipState {
                    full_grid,
                    tau0: tau0[0],
                    tau1: tau0[1],
                }))
            }
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn reconstruct_round0_poly(&self) -> UniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round0_poly(),
            Self::B8(state) => state.reconstruct_round0_poly(),
        }
    }

    #[cfg(test)]
    pub(crate) fn reconstruct_round1_poly(&self, r0: E) -> UniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round1_poly(r0),
            Self::B8(state) => state.reconstruct_round1_poly(r0),
        }
    }

    pub(crate) fn reconstruct_round0_eq_poly(&self) -> EqFactoredUniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round0_eq_poly(),
            Self::B8(state) => state.reconstruct_round0_eq_poly(),
        }
    }

    pub(crate) fn reconstruct_round1_eq_poly(&self, r0: E) -> EqFactoredUniPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round1_eq_poly(r0),
            Self::B8(state) => state.reconstruct_round1_eq_poly(r0),
        }
    }
}

impl<E: FieldCore + FromSmallInt> Stage1B4BivariateSkipState<E> {
    #[cfg(test)]
    fn reconstruct_round0_poly(&self) -> UniPoly<E> {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.x_row_coeffs[1], self.tau1),
        );
        coeff_array_to_poly(mul_linear_by_quadratic_coeffs(self.tau0, q_x))
    }

    #[cfg(test)]
    fn reconstruct_round1_poly(&self, r0: E) -> UniPoly<E> {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(self.x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        let round0_eq = linear_eq_eval(self.tau0, r0);
        let coeffs = mul_linear_by_quadratic_coeffs(self.tau1, q_y).map(|coeff| round0_eq * coeff);
        coeff_array_to_poly(coeffs)
    }

    fn reconstruct_round0_eq_poly(&self) -> EqFactoredUniPoly<E> {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.x_row_coeffs[1], self.tau1),
        );
        EqFactoredUniPoly::from_q_coeffs(q_x.into())
    }

    fn reconstruct_round1_eq_poly(&self, r0: E) -> EqFactoredUniPoly<E> {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(self.x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        EqFactoredUniPoly::from_q_coeffs(q_y.into())
    }
}

impl<E: FieldCore + FromSmallInt> Stage1B8BivariateSkipState<E> {
    #[cfg(test)]
    fn reconstruct_round0_poly(&self) -> UniPoly<E> {
        let l1_at_0 = E::one() - self.tau1;
        let l1_at_1 = self.tau1;
        let evals: Vec<E> = (0..=5u64)
            .map(|x_raw| {
                let x = E::from_u64(x_raw);
                let q_x0 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::zero());
                let q_x1 = eval_stage1_biquartic_from_full_grid(self.full_grid, x, E::one());
                linear_eq_eval(self.tau0, x) * (l1_at_0 * q_x0 + l1_at_1 * q_x1)
            })
            .collect();
        UniPoly::from_evals(&evals)
    }

    #[cfg(test)]
    fn reconstruct_round1_poly(&self, r0: E) -> UniPoly<E> {
        let l0_at_r0 = linear_eq_eval(self.tau0, r0);
        let evals: Vec<E> = (0..=5u64)
            .map(|y_raw| {
                let y = E::from_u64(y_raw);
                l0_at_r0
                    * linear_eq_eval(self.tau1, y)
                    * eval_stage1_biquartic_from_full_grid(self.full_grid, r0, y)
            })
            .collect();
        UniPoly::from_evals(&evals)
    }

    fn reconstruct_round0_eq_poly(&self) -> EqFactoredUniPoly<E> {
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

    fn reconstruct_round1_eq_poly(&self, r0: E) -> EqFactoredUniPoly<E> {
        let evals: Vec<E> = (0..=4u64)
            .map(|y_raw| {
                let y = E::from_u64(y_raw);
                eval_stage1_biquartic_from_full_grid(self.full_grid, r0, y)
            })
            .collect();
        interpolate_eq_factored_q_poly(&evals, STAGE1_B8_Q_POLY_DEGREE)
    }
}

fn interpolate_eq_factored_q_poly<E: FieldCore + FromSmallInt>(
    evals: &[E],
    degree: usize,
) -> EqFactoredUniPoly<E> {
    let mut q_coeffs = UniPoly::from_evals(evals).coeffs;
    q_coeffs.resize(degree + 1, E::zero());
    EqFactoredUniPoly::from_q_coeffs(q_coeffs)
}

/// Proposed reduced stage-2 domain `{1, Infinity}`.
#[cfg(test)]
pub(crate) fn stage2_reduced_prefix_points<E: FieldCore + FromSmallInt>() -> [PrefixPoint<E>; 2] {
    [PrefixPoint::Finite(E::one()), PrefixPoint::Infinity]
}

/// Safe full stage-2 fallback domain `{0, 1, Infinity}`.
#[cfg(test)]
pub(crate) fn stage2_full_prefix_points<E: FieldCore + FromSmallInt>() -> [PrefixPoint<E>; 3] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Infinity,
    ]
}

/// Return the bilinear coefficients for a quad ordered as `[t00, t10, t01, t11]`.
#[inline]
#[cfg(test)]
pub(crate) fn bilinear_coeffs_from_quad<E: FieldCore>(quad: [E; 4]) -> [E; 4] {
    let [t00, t10, t01, t11] = quad;
    [t00, t10 - t00, t01 - t00, t11 - t10 - t01 + t00]
}

/// Evaluate the bilinear multilinear extension of a quad at ordinary field
/// points `(x, y)`.
#[inline]
#[cfg(test)]
pub(crate) fn bilinear_eval<E: FieldCore>(quad: [E; 4], x: E, y: E) -> E {
    let [a, b, c, d] = bilinear_coeffs_from_quad(quad);
    a + x * (b + y * d) + y * c
}

/// Evaluate a quad on a small domain where `Infinity` means "leading
/// coefficient in that coordinate".
#[inline]
#[cfg(test)]
pub(crate) fn bilinear_eval_on_prefix_points<E: FieldCore>(
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

/// Evaluate the stage-1 candidate storage contribution used by the original
/// `{1, -1, 2, Infinity}^2` proposal.
#[inline]
#[cfg(test)]
pub(crate) fn stage1_local_norm_eval<E: FieldCore + FromSmallInt>(
    s_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
    b: usize,
) -> E {
    let s_eval = bilinear_eval_on_prefix_points(s_quad, x, y);
    range_check_eval_from_s(s_eval, b)
}

/// Evaluate the raw stage-1 full-domain polynomial on
/// `{0, 1, -1, 2, Infinity}^2`.
///
/// At `Infinity`, we take the leading coefficient in that coordinate of the
/// composed range-check polynomial `range_check(s(X, Y))`, rather than first
/// evaluating `s` at `Infinity` and then applying the range check.
#[inline]
#[cfg(test)]
pub(crate) fn stage1_local_norm_raw_eval<E: FieldCore + FromSmallInt>(
    s_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
    b: usize,
) -> E {
    let [_, bx, cy, dxy] = bilinear_coeffs_from_quad(s_quad);
    let degree = b / 2;
    let pow = |base: E| {
        let mut out = E::one();
        for _ in 0..degree {
            out = out * base;
        }
        out
    };

    match (x, y) {
        (PrefixPoint::Finite(x), PrefixPoint::Finite(y)) => {
            range_check_eval_from_s(bilinear_eval(s_quad, x, y), b)
        }
        (PrefixPoint::Infinity, PrefixPoint::Finite(y)) => pow(bx + y * dxy),
        (PrefixPoint::Finite(x), PrefixPoint::Infinity) => pow(cy + x * dxy),
        (PrefixPoint::Infinity, PrefixPoint::Infinity) => pow(dxy),
    }
}

/// Evaluate the stage-2 local norm candidate used by the proposed reduced
/// `{1, Infinity}^2` storage: evaluate the bilinear witness first, then apply
/// `w (w + 1)`.
#[inline]
#[cfg(test)]
pub(crate) fn stage2_local_norm_candidate_eval<E: FieldCore>(
    w_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    w_eval * (w_eval + E::one())
}

/// Evaluate the raw degree-`(2,2)` stage-2 norm polynomial on the safe full
/// `{0, 1, Infinity}^2` fallback domain.
///
/// At `Infinity`, we take the leading coefficient in that coordinate of
/// `w(X, Y) * (w(X, Y) + 1)`, so the linear `+w` term drops out.
#[inline]
#[cfg(test)]
pub(crate) fn stage2_local_norm_raw_eval<E: FieldCore>(
    w_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    match (x, y) {
        (PrefixPoint::Finite(_), PrefixPoint::Finite(_)) => w_eval * (w_eval + E::one()),
        _ => w_eval * w_eval,
    }
}

/// Evaluate the stage-2 local relation contribution for one witness quad, one
/// local bilinear factor quad, and one fixed scalar factor.
#[inline]
#[cfg(test)]
pub(crate) fn stage2_local_relation_eval<E: FieldCore>(
    w_quad: [E; 4],
    local_factor_quad: [E; 4],
    fixed_factor: E,
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    fixed_factor
        * bilinear_eval_on_prefix_points(w_quad, x, y)
        * bilinear_eval_on_prefix_points(local_factor_quad, x, y)
}

/// Boolean corner in the `{0, 1}^2` sub-grid of the stage-2 full domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BooleanCorner {
    ZeroZero,
    ZeroOne,
    OneZero,
    OneOne,
}

impl BooleanCorner {
    pub(crate) const ALL: [Self; 4] = [Self::ZeroZero, Self::ZeroOne, Self::OneZero, Self::OneOne];
    #[cfg(test)]
    pub(crate) const DEFAULT_STAGE2_NORM: Self = Self::ZeroZero;
    pub(crate) const DEFAULT_STAGE2_RELATION: Self = Self::ZeroZero;

    #[inline]
    pub(crate) fn default_norm_order() -> [Self; 4] {
        Self::ALL
    }

    #[inline]
    fn boolean_index(self) -> usize {
        match self {
            Self::ZeroZero => 0,
            Self::ZeroOne => 1,
            Self::OneZero => 2,
            Self::OneOne => 3,
        }
    }

    #[inline]
    fn grid_index(self) -> usize {
        match self {
            Self::ZeroZero => 0,
            Self::ZeroOne => 1,
            Self::OneZero => 3,
            Self::OneOne => 4,
        }
    }
}

/// Internal compressed stage-2 `{0, 1, Infinity}^2` grid with one omitted
/// Boolean corner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2CompressedGrid<E: FieldCore> {
    pub omitted_corner: BooleanCorner,
    pub evals_except_corner: [E; 8],
}

impl<E: FieldCore> Stage2CompressedGrid<E> {
    #[cfg(test)]
    pub(crate) fn from_full_grid(full_grid: [E; 9], omitted_corner: BooleanCorner) -> Self {
        let omitted_idx = omitted_corner.grid_index();
        let mut out_idx = 0usize;
        let evals_except_corner = std::array::from_fn(|_| {
            while out_idx == omitted_idx {
                out_idx += 1;
            }
            let value = full_grid[out_idx];
            out_idx += 1;
            value
        });
        Self {
            omitted_corner,
            evals_except_corner,
        }
    }

    pub(crate) fn reconstruct_with_corner_value(&self, omitted_value: E) -> [E; 9] {
        let omitted_idx = self.omitted_corner.grid_index();
        let mut src_idx = 0usize;
        std::array::from_fn(|dst_idx| {
            if dst_idx == omitted_idx {
                omitted_value
            } else {
                let value = self.evals_except_corner[src_idx];
                src_idx += 1;
                value
            }
        })
    }
}

/// Internal stage-2 first-two-round bivariate-skip payload.
///
/// This payload is built and consumed inside the prover to reconstruct ordinary
/// stage-2 sumcheck round messages; it is not serialized in the Akita proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2BivariateSkipProof<E: FieldCore> {
    pub norm: Stage2CompressedGrid<E>,
    pub relation: Stage2CompressedGrid<E>,
}

/// Return the stage-2 full-domain grid in row-major `x`-major order over
/// `{0, 1, Infinity}^2`.
#[cfg(test)]
pub(crate) fn stage2_full_grid_values<E: FieldCore + FromSmallInt>(
    mut eval: impl FnMut(PrefixPoint<E>, PrefixPoint<E>) -> E,
) -> [E; 9] {
    let points = stage2_full_prefix_points::<E>();
    std::array::from_fn(|idx| {
        let x = points[idx / 3];
        let y = points[idx % 3];
        eval(x, y)
    })
}

/// Evaluate a quadratic from its values at `{0, 1, Infinity}`.
#[inline]
#[cfg(test)]
pub(crate) fn eval_quadratic_from_01_inf<E: FieldCore>(
    at_zero: E,
    at_one: E,
    at_inf: E,
    x: PrefixPoint<E>,
) -> E {
    match x {
        PrefixPoint::Infinity => at_inf,
        PrefixPoint::Finite(x) => {
            let linear = at_one - at_zero - at_inf;
            at_zero + x * (linear + x * at_inf)
        }
    }
}

#[inline]
pub(crate) fn quadratic_coeffs_from_01_inf<E: FieldCore>(
    at_zero: E,
    at_one: E,
    at_inf: E,
) -> [E; 3] {
    [at_zero, at_one - at_zero - at_inf, at_inf]
}

#[inline]
fn eval_quadratic_from_coeffs<E: FieldCore>(coeffs: [E; 3], x: E) -> E {
    coeffs[0] + x * (coeffs[1] + x * coeffs[2])
}

#[inline]
fn linear_eq_coeffs<E: FieldCore>(tau: E) -> [E; 2] {
    [E::one() - tau, tau + tau - E::one()]
}

#[inline]
fn scale_quadratic_coeffs<E: FieldCore>(coeffs: [E; 3], scale: E) -> [E; 3] {
    [scale * coeffs[0], scale * coeffs[1], scale * coeffs[2]]
}

#[inline]
fn add_quadratic_coeffs<E: FieldCore>(lhs: [E; 3], rhs: [E; 3]) -> [E; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

#[inline]
#[cfg(test)]
fn coeff_array_to_poly<E: FieldCore, const N: usize>(coeffs: [E; N]) -> UniPoly<E> {
    UniPoly::from_coeffs(coeffs.to_vec())
}

#[inline]
fn mul_linear_by_quadratic_coeffs<E: FieldCore>(tau: E, quad: [E; 3]) -> [E; 4] {
    let [l0, l1] = linear_eq_coeffs(tau);
    [
        l0 * quad[0],
        l0 * quad[1] + l1 * quad[0],
        l0 * quad[2] + l1 * quad[1],
        l1 * quad[2],
    ]
}

/// Evaluate a biquadratic from its full `{0, 1, Infinity}^2` grid.
#[inline]
#[cfg(test)]
pub(crate) fn eval_biquadratic_from_full_grid<E: FieldCore>(
    full_grid: [E; 9],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let q_y0 = eval_quadratic_from_01_inf(full_grid[0], full_grid[3], full_grid[6], x);
    let q_y1 = eval_quadratic_from_01_inf(full_grid[1], full_grid[4], full_grid[7], x);
    let q_yinf = eval_quadratic_from_01_inf(full_grid[2], full_grid[5], full_grid[8], x);
    eval_quadratic_from_01_inf(q_y0, q_y1, q_yinf, y)
}

/// Return the local claim weights for the four Boolean corners of the stage-2
/// norm half, ordered as `[(0,0), (0,1), (1,0), (1,1)]`.
#[inline]
pub(crate) fn stage2_norm_corner_weights_from_linear_evals<E: FieldCore>(
    l0_at_0: E,
    l0_at_1: E,
    l1_at_0: E,
    l1_at_1: E,
) -> [E; 4] {
    [
        l0_at_0 * l1_at_0,
        l0_at_0 * l1_at_1,
        l0_at_1 * l1_at_0,
        l0_at_1 * l1_at_1,
    ]
}

/// Return the local claim weights for the four Boolean corners of the stage-2
/// norm half when the two local eq factors are `eq(tau0, X)` and `eq(tau1, Y)`.
#[inline]
pub(crate) fn stage2_norm_corner_weights_from_taus<E: FieldCore>(tau0: E, tau1: E) -> [E; 4] {
    stage2_norm_corner_weights_from_linear_evals(E::one() - tau0, tau0, E::one() - tau1, tau1)
}

/// Choose the default omitted corner for stage-2 norm compression, preferring
/// `(0,0)` when its claim weight is nonzero.
#[inline]
pub(crate) fn default_stage2_norm_omitted_corner<E: FieldCore>(
    corner_weights: [E; 4],
) -> BooleanCorner {
    for corner in BooleanCorner::default_norm_order() {
        if !corner_weights[corner.boolean_index()].is_zero() {
            return corner;
        }
    }
    unreachable!("at least one Boolean-corner weight must be nonzero");
}

/// Recover a full stage-2 grid from an omitted-corner compression and a
/// weighted Boolean-corner claim relation.
pub(crate) fn recover_stage2_grid_from_corner_claim<E: FieldCore>(
    compressed: &Stage2CompressedGrid<E>,
    corner_weights: [E; 4],
    claim: E,
) -> Option<[E; 9]> {
    let omitted_weight = corner_weights[compressed.omitted_corner.boolean_index()];
    let omitted_weight_inv = omitted_weight.inv()?;
    let mut full_grid = compressed.reconstruct_with_corner_value(E::zero());
    let known_sum = BooleanCorner::ALL
        .iter()
        .copied()
        .filter(|corner| *corner != compressed.omitted_corner)
        .fold(E::zero(), |acc, corner| {
            acc + corner_weights[corner.boolean_index()] * full_grid[corner.grid_index()]
        });
    let omitted_value = (claim - known_sum) * omitted_weight_inv;
    full_grid[compressed.omitted_corner.grid_index()] = omitted_value;
    Some(full_grid)
}

/// Recover a full stage-2 relation grid from its default `(0,0)` omission.
#[inline]
pub(crate) fn recover_stage2_relation_grid_from_claim<E: FieldCore>(
    compressed: &Stage2CompressedGrid<E>,
    relation_claim: E,
) -> [E; 9] {
    recover_stage2_grid_from_corner_claim(compressed, [E::one(); 4], relation_claim)
        .expect("relation corner weights are all one")
}

/// Recover a full stage-2 norm grid from an omitted Boolean corner and the
/// weighted local norm claim.
#[inline]
pub(crate) fn recover_stage2_norm_grid_from_claim<E: FieldCore>(
    compressed: &Stage2CompressedGrid<E>,
    corner_weights: [E; 4],
    norm_claim: E,
) -> Option<[E; 9]> {
    recover_stage2_grid_from_corner_claim(compressed, corner_weights, norm_claim)
}

/// Whether stage 2 has enough y-rounds to use the 2-round prefix path.
#[inline]
pub(crate) fn can_use_stage2_two_round_prefix(ring_bits: usize, b: usize) -> bool {
    ring_bits >= 2 && matches!(b, 4 | 8)
}

/// Build the stage-2 first-two-round bivariate-skip payload from the compact
/// witness table at the start of stage 2.
///
/// Returns `None` when there are fewer than two y-rounds to batch.
#[tracing::instrument(
    skip_all,
    name = "two_round_prefix::build_stage2_bivariate_skip_proof_from_compact"
)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_stage2_bivariate_skip_proof_from_compact<
    E: FieldCore + FromSmallInt + HasUnreducedOps,
>(
    w_compact: &[i8],
    alpha_evals_y: &[E],
    m_evals_x: &[E],
    r_stage1: &[E],
    b: usize,
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Option<Stage2BivariateSkipProof<E>> {
    if !can_use_stage2_two_round_prefix(ring_bits, b) {
        return None;
    }

    let y_len = 1usize << ring_bits;
    assert_eq!(alpha_evals_y.len(), y_len);
    assert_eq!(w_compact.len(), live_x_cols * y_len);
    assert_eq!(m_evals_x.len(), 1usize << col_bits);
    assert_eq!(r_stage1.len(), col_bits + ring_bits);

    let eq_y_suffix = EqPolynomial::evals(&r_stage1[2..ring_bits]);
    let eq_x = EqPolynomial::evals(&r_stage1[ring_bits..]);
    let y_quads = y_len >> 2;
    debug_assert_eq!(eq_y_suffix.len(), y_quads);
    let norm_omitted_corner = default_stage2_norm_omitted_corner(
        stage2_norm_corner_weights_from_taus(r_stage1[0], r_stage1[1]),
    );
    let norm_point_indices =
        &STAGE2_COMPRESSED_POINT_INDICES_BY_OMITTED_CORNER[norm_omitted_corner.boolean_index()];
    let alpha_point_values_by_quad: Vec<[E; STAGE2_COMPRESSED_POINT_COUNT]> = (0..y_quads)
        .map(|y_quad| {
            let base = 4 * y_quad;
            let alpha_quad = std::array::from_fn(|offset| alpha_evals_y[base + offset]);
            stage2_relation_m_point_values_compressed(alpha_quad)
        })
        .collect();

    let w_digit_fn: fn(i8) -> usize = match b {
        4 => stage2_b4_w_digit,
        8 => stage2_b8_w_digit,
        _ => unreachable!("unsupported stage-2 two-round prefix basis"),
    };
    let lookup_index_fn: fn([usize; 4]) -> usize = match b {
        4 => stage2_b4_lookup_index_from_digits,
        8 => stage2_b8_lookup_index_from_digits,
        _ => unreachable!(),
    };
    let norm_table: &[[i64; STAGE2_PREFIX_POINT_COUNT]] = match b {
        4 => &STAGE2_B4_NORM_LOOKUP_TABLE,
        8 => &STAGE2_B8_NORM_LOOKUP_TABLE,
        _ => unreachable!(),
    };
    let rel_table: &[[i64; STAGE2_COMPRESSED_POINT_COUNT]] = match b {
        4 => &STAGE2_B4_RELATION_WEIGHT_COMPRESSED_TABLE,
        8 => &STAGE2_B8_RELATION_WEIGHT_COMPRESSED_TABLE,
        _ => unreachable!(),
    };

    let (norm_pos, norm_neg, rel_accum) = cfg_fold_reduce!(
        0..live_x_cols,
        || {
            (
                [E::MulU64Accum::ZERO; STAGE2_COMPRESSED_POINT_COUNT],
                [E::MulU64Accum::ZERO; STAGE2_COMPRESSED_POINT_COUNT],
                [E::ProductAccum::ZERO; STAGE2_COMPRESSED_POINT_COUNT],
            )
        },
        |(mut norm_pos, mut norm_neg, mut rel_accum), x_idx| {
            let column = &w_compact[x_idx * y_len..(x_idx + 1) * y_len];
            let eq_x_weight = eq_x[x_idx];
            let m_val = m_evals_x[x_idx];
            let mut x_rel_pos = [E::MulU64Accum::ZERO; STAGE2_COMPRESSED_POINT_COUNT];
            let mut x_rel_neg = [E::MulU64Accum::ZERO; STAGE2_COMPRESSED_POINT_COUNT];
            for (y_quad, &eq_y_weight) in eq_y_suffix.iter().enumerate() {
                let base = 4 * y_quad;
                let lookup_idx = lookup_index_fn([
                    w_digit_fn(column[base]),
                    w_digit_fn(column[base + 1]),
                    w_digit_fn(column[base + 2]),
                    w_digit_fn(column[base + 3]),
                ]);
                let norm_weight = eq_y_weight * eq_x_weight;
                accum_lookup_vector_signed_selected(
                    &mut norm_pos,
                    &mut norm_neg,
                    norm_weight,
                    &norm_table[lookup_idx],
                    norm_point_indices,
                );
                accum_pointwise_signed(
                    &mut x_rel_pos,
                    &mut x_rel_neg,
                    &alpha_point_values_by_quad[y_quad],
                    &rel_table[lookup_idx],
                );
            }
            for idx in 0..STAGE2_COMPRESSED_POINT_COUNT {
                let x_rel = reduce_signed_accum::<E>(x_rel_pos[idx], x_rel_neg[idx]);
                rel_accum[idx] += m_val.mul_to_product_accum(x_rel);
            }
            (norm_pos, norm_neg, rel_accum)
        },
        |(mut norm_pos_a, mut norm_neg_a, mut rel_accum_a),
         (norm_pos_b, norm_neg_b, rel_accum_b)| {
            for (dst, src) in norm_pos_a.iter_mut().zip(norm_pos_b.iter()) {
                *dst += *src;
            }
            for (dst, src) in norm_neg_a.iter_mut().zip(norm_neg_b.iter()) {
                *dst += *src;
            }
            for (dst, src) in rel_accum_a.iter_mut().zip(rel_accum_b.iter()) {
                *dst += *src;
            }
            (norm_pos_a, norm_neg_a, rel_accum_a)
        }
    );
    let norm_evals_except_corner: [E; STAGE2_COMPRESSED_POINT_COUNT] =
        std::array::from_fn(|idx| reduce_signed_accum::<E>(norm_pos[idx], norm_neg[idx]));
    let relation_evals_except_corner: [E; STAGE2_COMPRESSED_POINT_COUNT] =
        std::array::from_fn(|idx| E::reduce_product_accum(rel_accum[idx]));
    Some(Stage2BivariateSkipProof {
        norm: Stage2CompressedGrid {
            omitted_corner: norm_omitted_corner,
            evals_except_corner: norm_evals_except_corner,
        },
        relation: Stage2CompressedGrid {
            omitted_corner: BooleanCorner::DEFAULT_STAGE2_RELATION,
            evals_except_corner: relation_evals_except_corner,
        },
    })
}

/// State needed to reconstruct the first two ordinary stage-2 round messages
/// from the internal bivariate-skip payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2BivariateSkipState<E: FieldCore> {
    norm_x_row_coeffs: [[E; 3]; 3],
    relation_x_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
    batching_coeff: E,
}

impl<E: FieldCore> Stage2BivariateSkipState<E> {
    pub(crate) fn new(
        proof: &Stage2BivariateSkipProof<E>,
        r_stage1: &[E],
        s_claim: E,
        relation_claim: E,
        batching_coeff: E,
    ) -> Option<Self> {
        if r_stage1.len() < 2 {
            return None;
        }
        let tau0 = r_stage1[0];
        let tau1 = r_stage1[1];
        let norm_full_grid = recover_stage2_norm_grid_from_claim(
            &proof.norm,
            stage2_norm_corner_weights_from_taus(tau0, tau1),
            s_claim,
        )?;
        let relation_full_grid =
            recover_stage2_relation_grid_from_claim(&proof.relation, relation_claim);
        let norm_x_row_coeffs = std::array::from_fn(|y_idx| {
            quadratic_coeffs_from_01_inf(
                norm_full_grid[y_idx],
                norm_full_grid[3 + y_idx],
                norm_full_grid[6 + y_idx],
            )
        });
        let relation_x_row_coeffs = std::array::from_fn(|y_idx| {
            quadratic_coeffs_from_01_inf(
                relation_full_grid[y_idx],
                relation_full_grid[3 + y_idx],
                relation_full_grid[6 + y_idx],
            )
        });
        Some(Self {
            norm_x_row_coeffs,
            relation_x_row_coeffs,
            tau0,
            tau1,
            batching_coeff,
        })
    }
}

impl<E: FieldCore + FromSmallInt> Stage2BivariateSkipState<E> {
    #[inline]
    pub(crate) fn reconstruct_round0_polys(&self) -> (UniPoly<E>, UniPoly<E>) {
        let norm_q = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.norm_x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.norm_x_row_coeffs[1], self.tau1),
        );
        let mut norm_coeffs = mul_linear_by_quadratic_coeffs(self.tau0, norm_q);
        for coeff in &mut norm_coeffs {
            *coeff = self.batching_coeff * *coeff;
        }
        let relation_coeffs =
            add_quadratic_coeffs(self.relation_x_row_coeffs[0], self.relation_x_row_coeffs[1]);
        (
            UniPoly::from_coeffs(norm_coeffs.to_vec()),
            UniPoly::from_coeffs(relation_coeffs.to_vec()),
        )
    }

    #[inline]
    pub(crate) fn reconstruct_round1_polys(&self, r0: E) -> (UniPoly<E>, UniPoly<E>) {
        let norm_y_values: [E; 3] = std::array::from_fn(|y_idx| {
            eval_quadratic_from_coeffs(self.norm_x_row_coeffs[y_idx], r0)
        });
        let norm_q =
            quadratic_coeffs_from_01_inf(norm_y_values[0], norm_y_values[1], norm_y_values[2]);
        let round0_eq = linear_eq_eval(self.tau0, r0);
        let mut norm_coeffs = mul_linear_by_quadratic_coeffs(self.tau1, norm_q);
        for coeff in &mut norm_coeffs {
            *coeff = self.batching_coeff * round0_eq * *coeff;
        }
        let relation_y_values: [E; 3] = std::array::from_fn(|y_idx| {
            eval_quadratic_from_coeffs(self.relation_x_row_coeffs[y_idx], r0)
        });
        let relation_coeffs = quadratic_coeffs_from_01_inf(
            relation_y_values[0],
            relation_y_values[1],
            relation_y_values[2],
        );
        (
            UniPoly::from_coeffs(norm_coeffs.to_vec()),
            UniPoly::from_coeffs(relation_coeffs.to_vec()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::akita_stage1::advance_stage1_claim;
    use crate::sumcheck::akita_stage1::AkitaStage1Prover;
    use akita_algebra::Prime128Offset275;
    use akita_serialization::{AkitaDeserialize, AkitaSerialize};
    use akita_sumcheck::EqFactoredSumcheckInstanceProver;
    use akita_types::reorder_stage1_coords;
    use std::collections::HashMap;

    type F = Prime128Offset275;

    fn gaussian_rank(mut rows: Vec<Vec<F>>) -> usize {
        rows.retain(|row| row.iter().any(|x| !x.is_zero()));
        if rows.is_empty() {
            return 0;
        }

        let num_cols = rows[0].len();
        let mut rank = 0usize;
        let mut col = 0usize;
        while rank < rows.len() && col < num_cols {
            let Some(pivot_row) = (rank..rows.len()).find(|&r| !rows[r][col].is_zero()) else {
                col += 1;
                continue;
            };
            rows.swap(rank, pivot_row);
            let pivot_inv = rows[rank][col].inv().expect("pivot must be invertible");
            for entry in &mut rows[rank] {
                *entry *= pivot_inv;
            }
            let pivot_snapshot = rows[rank].clone();
            for (row_idx, row) in rows.iter_mut().enumerate() {
                if row_idx == rank || row[col].is_zero() {
                    continue;
                }
                let factor = row[col];
                for (entry, &pivot_entry) in row.iter_mut().zip(pivot_snapshot.iter()) {
                    *entry -= factor * pivot_entry;
                }
            }
            rank += 1;
            col += 1;
        }
        rank
    }

    fn vec_key(vals: &[F]) -> String {
        format!("{vals:?}")
    }

    fn stage2_norm_round_values(w_quad: [F; 4], tau0: F, tau1: F, r0: F) -> Vec<F> {
        let l0 = |x: F| tau0 * x + (F::one() - tau0) * (F::one() - x);
        let l1 = |y: F| tau1 * y + (F::one() - tau1) * (F::one() - y);
        let q = |x: F, y: F| {
            let w = bilinear_eval(w_quad, x, y);
            w * (w + F::one())
        };

        let mut out = Vec::new();
        for x in 0..=3u64 {
            let x = F::from_u64(x);
            out.push(l0(x) * (l1(F::zero()) * q(x, F::zero()) + l1(F::one()) * q(x, F::one())));
        }
        for y in 0..=3u64 {
            let y = F::from_u64(y);
            out.push(l1(y) * l0(r0) * q(r0, y));
        }
        out
    }

    fn stage2_relation_round_values(w_quad: [F; 4], m_quad: [F; 4], r0: F) -> Vec<F> {
        let relation = |x: F, y: F| bilinear_eval(w_quad, x, y) * bilinear_eval(m_quad, x, y);
        let mut out = Vec::new();
        for x in 0..=2u64 {
            let x = F::from_u64(x);
            out.push(relation(x, F::zero()) + relation(x, F::one()));
        }
        for y in 0..=2u64 {
            let y = F::from_u64(y);
            out.push(relation(r0, y));
        }
        out
    }

    fn stage2_norm_claim_from_full_grid(full_grid: [F; 9], corner_weights: [F; 4]) -> F {
        BooleanCorner::ALL
            .iter()
            .copied()
            .fold(F::zero(), |acc, corner| {
                acc + corner_weights[corner.boolean_index()] * full_grid[corner.grid_index()]
            })
    }

    fn stage2_relation_claim_from_full_grid(full_grid: [F; 9]) -> F {
        stage2_norm_claim_from_full_grid(full_grid, [F::one(); 4])
    }

    fn stage2_norm_round_values_from_full_grid(
        full_grid: [F; 9],
        tau0: F,
        tau1: F,
        r0: F,
    ) -> Vec<F> {
        let l0_at = |x: PrefixPoint<F>| match x {
            PrefixPoint::Finite(x) => tau0 * x + (F::one() - tau0) * (F::one() - x),
            PrefixPoint::Infinity => tau0,
        };
        let l1_0 = F::one() - tau1;
        let l1_1 = tau1;
        let mut out = Vec::new();
        for x in [F::zero(), F::one(), F::from_u64(2), F::from_u64(3)] {
            let x_point = PrefixPoint::Finite(x);
            let q_x0 =
                eval_biquadratic_from_full_grid(full_grid, x_point, PrefixPoint::Finite(F::zero()));
            let q_x1 =
                eval_biquadratic_from_full_grid(full_grid, x_point, PrefixPoint::Finite(F::one()));
            out.push(l0_at(x_point) * (l1_0 * q_x0 + l1_1 * q_x1));
        }
        for y in [F::zero(), F::one(), F::from_u64(2), F::from_u64(3)] {
            let y_point = PrefixPoint::Finite(y);
            let q_r0_y =
                eval_biquadratic_from_full_grid(full_grid, PrefixPoint::Finite(r0), y_point);
            let l1_y = tau1 * y + (F::one() - tau1) * (F::one() - y);
            out.push(l1_y * l0_at(PrefixPoint::Finite(r0)) * q_r0_y);
        }
        out
    }

    fn stage2_relation_round_values_from_full_grid(full_grid: [F; 9], r0: F) -> Vec<F> {
        let mut out = Vec::new();
        for x in [F::zero(), F::one(), F::from_u64(2)] {
            let q_x0 = eval_biquadratic_from_full_grid(
                full_grid,
                PrefixPoint::Finite(x),
                PrefixPoint::Finite(F::zero()),
            );
            let q_x1 = eval_biquadratic_from_full_grid(
                full_grid,
                PrefixPoint::Finite(x),
                PrefixPoint::Finite(F::one()),
            );
            out.push(q_x0 + q_x1);
        }
        for y in [F::zero(), F::one(), F::from_u64(2)] {
            out.push(eval_biquadratic_from_full_grid(
                full_grid,
                PrefixPoint::Finite(r0),
                PrefixPoint::Finite(y),
            ));
        }
        out
    }

    fn tensor_values<E: FieldCore, const NX: usize, const NY: usize>(
        xs: [PrefixPoint<E>; NX],
        ys: [PrefixPoint<E>; NY],
        mut eval: impl FnMut(PrefixPoint<E>, PrefixPoint<E>) -> E,
    ) -> Vec<E> {
        let mut out = Vec::with_capacity(NX * NY);
        for &x in &xs {
            for &y in &ys {
                out.push(eval(x, y));
            }
        }
        out
    }

    fn stage1_norm_round_values(s_quad: [F; 4], tau0: F, tau1: F, r0: F, b: usize) -> Vec<F> {
        let l0 = |x: F| tau0 * x + (F::one() - tau0) * (F::one() - x);
        let l1 = |y: F| tau1 * y + (F::one() - tau1) * (F::one() - y);
        let q = |x: F, y: F| range_check_eval_from_s(bilinear_eval(s_quad, x, y), b);

        let mut out = Vec::new();
        for x in 0..=5u64 {
            let x = F::from_u64(x);
            out.push(l0(x) * (l1(F::zero()) * q(x, F::zero()) + l1(F::one()) * q(x, F::one())));
        }
        for y in 0..=5u64 {
            let y = F::from_u64(y);
            out.push(l0(r0) * l1(y) * q(r0, y));
        }
        out
    }

    fn build_stage1_bivariate_skip_proof_from_compact_reference(
        w_compact: &[i8],
        tau0: &[F],
        b: usize,
        live_x_cols: usize,
        _col_bits: usize,
        ring_bits: usize,
    ) -> Option<Stage1BivariateSkipProof<F>> {
        if !can_use_stage1_two_round_prefix(ring_bits, b) {
            return None;
        }

        let y_len = 1usize << ring_bits;
        let eq_y_suffix = EqPolynomial::evals(&tau0[2..ring_bits]);
        let eq_x = EqPolynomial::evals(&tau0[ring_bits..]);
        let points = stage1_full_prefix_points::<F>();
        let y_quads = y_len / 4;
        let mut evals_except_boolean_core = Vec::with_capacity(STAGE1_PREFIX_EVAL_COUNT);

        for x_idx in 0..5 {
            for y_idx in 0..5 {
                if stage1_is_boolean_corner(x_idx, y_idx) {
                    continue;
                }
                let mut accum = F::zero();
                let x = points[x_idx];
                let y = points[y_idx];
                for x_col in 0..live_x_cols {
                    let col = &w_compact[x_col * y_len..(x_col + 1) * y_len];
                    let eq_x_weight = eq_x[x_col];
                    for (y_quad, &eq_y_weight) in eq_y_suffix.iter().enumerate().take(y_quads) {
                        let base = 4 * y_quad;
                        let s_quad = std::array::from_fn(|offset| {
                            let w = i64::from(col[base + offset]);
                            F::from_i64(w * (w + 1))
                        });
                        accum +=
                            eq_x_weight * eq_y_weight * stage1_local_norm_raw_eval(s_quad, x, y, b);
                    }
                }
                evals_except_boolean_core.push(accum);
            }
        }

        Some(Stage1BivariateSkipProof {
            evals_except_boolean_core,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_stage2_bivariate_skip_proof_from_compact_reference(
        w_compact: &[i8],
        alpha_evals_y: &[F],
        m_evals_x: &[F],
        r_stage1: &[F],
        b: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Option<Stage2BivariateSkipProof<F>> {
        if !can_use_stage2_two_round_prefix(ring_bits, b) {
            return None;
        }

        let y_len = 1usize << ring_bits;
        assert_eq!(m_evals_x.len(), 1usize << col_bits);
        let eq_y_suffix = EqPolynomial::evals(&r_stage1[2..ring_bits]);
        let eq_x = EqPolynomial::evals(&r_stage1[ring_bits..]);
        let points = stage2_full_prefix_points::<F>();
        let y_quads = y_len >> 2;
        let mut norm_full = [F::zero(); 9];
        let mut relation_full = [F::zero(); 9];

        for x_idx in 0..live_x_cols {
            let column = &w_compact[x_idx * y_len..(x_idx + 1) * y_len];
            let m_val = m_evals_x[x_idx];
            let eq_x_weight = eq_x[x_idx];
            for (y_quad, &eq_y_weight) in eq_y_suffix.iter().enumerate().take(y_quads) {
                let base = 4 * y_quad;
                let w_quad =
                    std::array::from_fn(|offset| F::from_i64(column[base + offset] as i64));
                let alpha_quad = std::array::from_fn(|offset| alpha_evals_y[base + offset]);
                let norm_weight = eq_y_weight * eq_x_weight;
                for idx in 0..9 {
                    let x = points[idx / 3];
                    let y = points[idx % 3];
                    norm_full[idx] += norm_weight * stage2_local_norm_raw_eval(w_quad, x, y);
                    relation_full[idx] +=
                        stage2_local_relation_eval(w_quad, alpha_quad, m_val, x, y);
                }
            }
        }

        let norm_omitted_corner = default_stage2_norm_omitted_corner(
            stage2_norm_corner_weights_from_taus(r_stage1[0], r_stage1[1]),
        );
        Some(Stage2BivariateSkipProof {
            norm: Stage2CompressedGrid::from_full_grid(norm_full, norm_omitted_corner),
            relation: Stage2CompressedGrid::from_full_grid(
                relation_full,
                BooleanCorner::DEFAULT_STAGE2_RELATION,
            ),
        })
    }

    #[test]
    fn stage1_b8_lookup_table_matches_raw_evals() {
        let points = stage1_full_prefix_points::<F>();
        for (d0, &s00) in STAGE1_B8_S_VALUES.iter().enumerate() {
            for (d1, &s10) in STAGE1_B8_S_VALUES.iter().enumerate() {
                for (d2, &s01) in STAGE1_B8_S_VALUES.iter().enumerate() {
                    for (d3, &s11) in STAGE1_B8_S_VALUES.iter().enumerate() {
                        let lookup = &STAGE1_B8_PREFIX_LOOKUP_TABLE
                            [stage1_b8_lookup_index_from_digits([d0, d1, d2, d3])];
                        let quad = [
                            F::from_i64(s00),
                            F::from_i64(s10),
                            F::from_i64(s01),
                            F::from_i64(s11),
                        ];
                        let mut point_idx = 0usize;
                        for x_idx in 0..5 {
                            for y_idx in 0..5 {
                                if stage1_is_boolean_corner(x_idx, y_idx) {
                                    continue;
                                }
                                assert_eq!(
                                    F::from_i64(lookup[point_idx]),
                                    stage1_local_norm_raw_eval(
                                        quad,
                                        points[x_idx],
                                        points[y_idx],
                                        8,
                                    ),
                                );
                                point_idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage2_b8_norm_lookup_table_matches_raw_evals() {
        let points = stage2_full_prefix_points::<F>();
        for w00 in -4i64..=3 {
            for w10 in -4i64..=3 {
                for w01 in -4i64..=3 {
                    for w11 in -4i64..=3 {
                        let lookup = &STAGE2_B8_NORM_LOOKUP_TABLE
                            [stage2_b8_lookup_index_from_digits([
                                (w00 + 4) as usize,
                                (w10 + 4) as usize,
                                (w01 + 4) as usize,
                                (w11 + 4) as usize,
                            ])];
                        let quad = [
                            F::from_i64(w00),
                            F::from_i64(w10),
                            F::from_i64(w01),
                            F::from_i64(w11),
                        ];
                        for point_idx in 0..STAGE2_PREFIX_POINT_COUNT {
                            let x = points[point_idx / 3];
                            let y = points[point_idx % 3];
                            assert_eq!(
                                F::from_i64(lookup[point_idx]),
                                stage2_local_norm_raw_eval(quad, x, y),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage2_b8_relation_weight_table_matches_prefix_w_evals() {
        let points = stage2_full_prefix_points::<F>();
        for w00 in -4i64..=3 {
            for w10 in -4i64..=3 {
                for w01 in -4i64..=3 {
                    for w11 in -4i64..=3 {
                        let lookup = &STAGE2_B8_RELATION_WEIGHT_TABLE
                            [stage2_b8_lookup_index_from_digits([
                                (w00 + 4) as usize,
                                (w10 + 4) as usize,
                                (w01 + 4) as usize,
                                (w11 + 4) as usize,
                            ])];
                        let quad = [
                            F::from_i64(w00),
                            F::from_i64(w10),
                            F::from_i64(w01),
                            F::from_i64(w11),
                        ];
                        for point_idx in 0..STAGE2_PREFIX_POINT_COUNT {
                            let x = points[point_idx / 3];
                            let y = points[point_idx % 3];
                            assert_eq!(
                                F::from_i64(lookup[point_idx]),
                                bilinear_eval_on_prefix_points(quad, x, y),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage1_bivariate_skip_proof_builder_matches_reference() {
        let col_bits = 3;
        let ring_bits = 2;
        let w_compact: Vec<i8> = (0..(5usize << ring_bits))
            .map(|i| ((3 * i + 1) % 8) as i8 - 4)
            .collect();
        let tau0_raw = vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let tau0 = reorder_stage1_coords(&tau0_raw, col_bits, ring_bits);
        assert_eq!(
            build_stage1_bivariate_skip_proof_from_compact(
                &w_compact, &tau0, 8, 5, col_bits, ring_bits
            ),
            build_stage1_bivariate_skip_proof_from_compact_reference(
                &w_compact, &tau0, 8, 5, col_bits, ring_bits,
            ),
        );
    }

    #[test]
    fn stage2_bivariate_skip_proof_builder_matches_reference() {
        let w_compact = vec![1, -2, 0, 2, 1, -1, 2, 1, 0, 2];
        let alpha_evals_y = [F::from_u64(3), F::from_u64(5)];
        let m_evals_x = [
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
            F::from_u64(23),
            F::from_u64(29),
            F::from_u64(31),
        ];
        let r_stage1 = [
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        assert_eq!(
            build_stage2_bivariate_skip_proof_from_compact(
                &w_compact,
                &alpha_evals_y,
                &m_evals_x,
                &r_stage1,
                8,
                5,
                3,
                1,
            ),
            build_stage2_bivariate_skip_proof_from_compact_reference(
                &w_compact,
                &alpha_evals_y,
                &m_evals_x,
                &r_stage1,
                8,
                5,
                3,
                1,
            ),
        );
    }

    #[test]
    fn stage2_bivariate_skip_proof_builder_matches_reference_large_odd_randomized() {
        let live_x_cols = 34_519usize;
        let col_bits = 16usize;
        let ring_bits = 6usize;
        let y_len = 1usize << ring_bits;
        let w_compact: Vec<i8> = (0..(live_x_cols * y_len))
            .map(|i| ((i * 37 + 11) % 8) as i8 - 4)
            .collect();
        let alpha_evals_y: Vec<F> = (0..y_len)
            .map(|i| {
                F::from_u64(
                    (i as u64)
                        .wrapping_mul(0x9e37_79b9)
                        .wrapping_add(0x1234_5678),
                )
            })
            .collect();
        let m_evals_x: Vec<F> = (0..(1usize << col_bits))
            .map(|i| {
                F::from_u64(
                    (i as u64)
                        .wrapping_mul(0x85eb_ca6b)
                        .wrapping_add(0xc2b2_ae35),
                )
            })
            .collect();
        let r_stage1: Vec<F> = (0..(col_bits + ring_bits))
            .map(|i| {
                F::from_u64(
                    (i as u64)
                        .wrapping_mul(0x27d4_eb2d)
                        .wrapping_add(0x1656_67b1),
                )
            })
            .collect();
        assert_eq!(
            build_stage2_bivariate_skip_proof_from_compact(
                &w_compact,
                &alpha_evals_y,
                &m_evals_x,
                &r_stage1,
                8,
                live_x_cols,
                col_bits,
                ring_bits,
            ),
            build_stage2_bivariate_skip_proof_from_compact_reference(
                &w_compact,
                &alpha_evals_y,
                &m_evals_x,
                &r_stage1,
                8,
                live_x_cols,
                col_bits,
                ring_bits,
            ),
        );
    }

    #[test]
    fn stage1_candidate_omits_11_via_zero_check() {
        let points = stage1_prefix_points::<F>();
        let one = points[0];
        let valid_s = [0i64, 2, 6, 12];
        for &s00 in &valid_s {
            for &s10 in &valid_s {
                for &s01 in &valid_s {
                    for &s11 in &valid_s {
                        let quad = [
                            F::from_i64(s00),
                            F::from_i64(s10),
                            F::from_i64(s01),
                            F::from_i64(s11),
                        ];
                        assert_eq!(
                            stage1_local_norm_eval(quad, one, one, 8),
                            F::zero(),
                            "stage1 local zero-check should vanish at (1,1)"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stage1_candidate_storage_family_has_rank_15() {
        let [one, neg_one, two, inf] = stage1_prefix_points::<F>();
        let storage_points = [
            (one, neg_one),
            (one, two),
            (one, inf),
            (neg_one, one),
            (neg_one, neg_one),
            (neg_one, two),
            (neg_one, inf),
            (two, one),
            (two, neg_one),
            (two, two),
            (two, inf),
            (inf, one),
            (inf, neg_one),
            (inf, two),
            (inf, inf),
        ];
        let valid_s = [0i64, 2, 6, 12];
        let mut rows = Vec::new();
        for &s00 in &valid_s {
            for &s10 in &valid_s {
                for &s01 in &valid_s {
                    for &s11 in &valid_s {
                        let quad = [
                            F::from_i64(s00),
                            F::from_i64(s10),
                            F::from_i64(s01),
                            F::from_i64(s11),
                        ];
                        rows.push(
                            storage_points
                                .iter()
                                .map(|&(x, y)| stage1_local_norm_eval(quad, x, y, 8))
                                .collect(),
                        );
                    }
                }
            }
        }
        assert_eq!(gaussian_rank(rows), 15);
    }

    #[test]
    fn stage1_full_domain_omits_boolean_core_via_zero_check() {
        let points = stage1_full_prefix_points::<F>();
        let valid_s = [0i64, 2, 6, 12];
        for &s00 in &valid_s {
            for &s10 in &valid_s {
                for &s01 in &valid_s {
                    for &s11 in &valid_s {
                        let quad = [
                            F::from_i64(s00),
                            F::from_i64(s10),
                            F::from_i64(s01),
                            F::from_i64(s11),
                        ];
                        for &(x_idx, y_idx) in &[(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                            assert_eq!(
                                stage1_local_norm_raw_eval(quad, points[x_idx], points[y_idx], 8),
                                F::zero(),
                                "stage1 local zero-check should vanish on the Boolean core",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage1_full_storage_family_has_rank_21() {
        let points = stage1_full_prefix_points::<F>();
        let mut storage_points = Vec::new();
        for x_idx in 0..5 {
            for y_idx in 0..5 {
                if stage1_is_boolean_corner(x_idx, y_idx) {
                    continue;
                }
                storage_points.push((points[x_idx], points[y_idx]));
            }
        }

        let valid_s = [0i64, 2, 6, 12];
        let mut rows = Vec::new();
        for &s00 in &valid_s {
            for &s10 in &valid_s {
                for &s01 in &valid_s {
                    for &s11 in &valid_s {
                        let quad = [
                            F::from_i64(s00),
                            F::from_i64(s10),
                            F::from_i64(s01),
                            F::from_i64(s11),
                        ];
                        rows.push(
                            storage_points
                                .iter()
                                .map(|&(x, y)| stage1_local_norm_raw_eval(quad, x, y, 8))
                                .collect(),
                        );
                    }
                }
            }
        }
        assert_eq!(gaussian_rank(rows), 21);
    }

    #[test]
    fn stage1_storage_domain_matches_local_round_messages() {
        let tau0 = F::from_u64(7);
        let tau1 = F::from_u64(11);
        let r0 = F::from_u64(13);
        let valid_s = [0i64, 2, 6, 12];

        for &s00 in &valid_s {
            for &s10 in &valid_s {
                for &s01 in &valid_s {
                    for &s11 in &valid_s {
                        let quad = [
                            F::from_i64(s00),
                            F::from_i64(s10),
                            F::from_i64(s01),
                            F::from_i64(s11),
                        ];
                        let proof = Stage1BivariateSkipProof {
                            evals_except_boolean_core: stage1_storage_vector_from_quad(quad, 8),
                        };
                        let skip_state = Stage1BivariateSkipState::new(&proof, &[tau0, tau1], 8)
                            .expect("stage1 bivariate-skip state should build");
                        let round_values = stage1_norm_round_values(quad, tau0, tau1, r0, 8);
                        assert_eq!(
                            skip_state.reconstruct_round0_poly(),
                            UniPoly::from_evals(&round_values[..6])
                        );
                        assert_eq!(
                            skip_state.reconstruct_round1_poly(r0),
                            UniPoly::from_evals(&round_values[6..])
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stage1_bivariate_skip_proof_reconstructs_first_two_rounds() {
        let b = 8;
        let live_x_cols = 5;
        let col_bits = 3;
        let ring_bits = 2;
        let w_compact: Vec<i8> = (0..(live_x_cols << ring_bits))
            .map(|i| ((5 * i + 3) % b) as i8 - (b / 2) as i8)
            .collect();
        let tau0_raw = vec![
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let tau0 = reorder_stage1_coords(&tau0_raw, col_bits, ring_bits);

        let proof = build_stage1_bivariate_skip_proof_from_compact(
            &w_compact,
            &tau0,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        )
        .expect("stage1 bivariate-skip payload should be available");
        let skip_state = Stage1BivariateSkipState::new(&proof, &tau0, b)
            .expect("stage1 bivariate-skip state should build");

        let mut prover =
            AkitaStage1Prover::<F>::new(&w_compact, &tau0, b, live_x_cols, col_bits, ring_bits);
        let round0 = prover.compute_round_eq_factored(0);
        assert_eq!(skip_state.reconstruct_round0_eq_poly(), round0);

        let r0 = F::from_u64(9);
        let _ = advance_stage1_claim(&prover, F::zero(), F::one(), &round0, r0);
        prover.ingest_challenge(0, r0);

        let round1 = prover.compute_round_eq_factored(1);
        assert_eq!(skip_state.reconstruct_round1_eq_poly(r0), round1);
    }

    #[test]
    fn stage1_b8_reconstructed_eq_polys_keep_degree4_storage_width() {
        let state = Stage1B8BivariateSkipState {
            full_grid: [F::zero(); 25],
            tau0: F::from_u64(3),
            tau1: F::from_u64(5),
        };

        for poly in [
            state.reconstruct_round0_eq_poly(),
            state.reconstruct_round1_eq_poly(F::from_u64(7)),
        ] {
            assert_eq!(
                poly.coeffs_except_linear_term.len(),
                EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(STAGE1_B8_Q_POLY_DEGREE)
            );
            assert_eq!(
                poly.coeffs_except_linear_term,
                vec![
                    F::zero();
                    EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(STAGE1_B8_Q_POLY_DEGREE)
                ]
            );

            let mut bytes = Vec::new();
            poly.serialize_uncompressed(&mut bytes)
                .expect("eq-factored poly should serialize");
            let decoded = EqFactoredUniPoly::<F>::deserialize_uncompressed(
                &bytes[..],
                &STAGE1_B8_Q_POLY_DEGREE,
            )
            .expect("eq-factored poly should deserialize at degree 4");
            assert_eq!(decoded, poly);
        }
    }

    #[test]
    fn stage2_default_norm_omitted_corner_prefers_00() {
        let weights = stage2_norm_corner_weights_from_taus(F::from_u64(7), F::from_u64(11));
        assert_eq!(
            default_stage2_norm_omitted_corner(weights),
            BooleanCorner::DEFAULT_STAGE2_NORM
        );
    }

    #[test]
    fn stage2_default_norm_omitted_corner_falls_back_when_00_is_zero() {
        let weights = stage2_norm_corner_weights_from_taus(F::one(), F::from_u64(11));
        assert_eq!(
            default_stage2_norm_omitted_corner(weights),
            BooleanCorner::OneZero
        );

        let weights = stage2_norm_corner_weights_from_taus(F::from_u64(7), F::one());
        assert_eq!(
            default_stage2_norm_omitted_corner(weights),
            BooleanCorner::ZeroOne
        );

        let weights = stage2_norm_corner_weights_from_taus(F::one(), F::one());
        assert_eq!(
            default_stage2_norm_omitted_corner(weights),
            BooleanCorner::OneOne
        );
    }

    #[test]
    fn stage2_norm_reduced_domain_has_round_message_collision() {
        let reduced = stage2_reduced_prefix_points::<F>();
        let tau0 = F::from_u64(7);
        let tau1 = F::from_u64(11);
        let r0 = F::from_u64(13);

        let mut seen: HashMap<String, Vec<F>> = HashMap::new();
        let mut found_collision = false;
        for w00 in -4i64..=3 {
            for w10 in -4i64..=3 {
                for w01 in -4i64..=3 {
                    for w11 in -4i64..=3 {
                        let quad = [
                            F::from_i64(w00),
                            F::from_i64(w10),
                            F::from_i64(w01),
                            F::from_i64(w11),
                        ];
                        let storage = tensor_values(reduced, reduced, |x, y| {
                            stage2_local_norm_candidate_eval(quad, x, y)
                        });
                        let target = stage2_norm_round_values(quad, tau0, tau1, r0);
                        let key = vec_key(&storage);
                        if let Some(existing) = seen.get(&key) {
                            if *existing != target {
                                found_collision = true;
                                break;
                            }
                        } else {
                            seen.insert(key, target);
                        }
                    }
                    if found_collision {
                        break;
                    }
                }
                if found_collision {
                    break;
                }
            }
            if found_collision {
                break;
            }
        }
        assert!(
            found_collision,
            "reduced stage-2 norm domain should not uniquely determine local round messages"
        );
    }

    #[test]
    fn stage2_relation_reduced_domain_has_round_message_collision() {
        let reduced = stage2_reduced_prefix_points::<F>();
        let r0 = F::from_u64(13);
        let alpha = F::one();
        let bit = [F::zero(), F::one()];

        let mut seen: HashMap<String, Vec<F>> = HashMap::new();
        let mut found_collision = false;
        for &w00 in &bit {
            for &w10 in &bit {
                for &w01 in &bit {
                    for &w11 in &bit {
                        let w_quad = [w00, w10, w01, w11];
                        for &m00 in &bit {
                            for &m10 in &bit {
                                for &m01 in &bit {
                                    for &m11 in &bit {
                                        let m_quad = [m00, m10, m01, m11];
                                        let storage = tensor_values(reduced, reduced, |x, y| {
                                            stage2_local_relation_eval(w_quad, m_quad, alpha, x, y)
                                        });
                                        let target =
                                            stage2_relation_round_values(w_quad, m_quad, r0);
                                        let key = vec_key(&storage);
                                        if let Some(existing) = seen.get(&key) {
                                            if *existing != target {
                                                found_collision = true;
                                                break;
                                            }
                                        } else {
                                            seen.insert(key, target);
                                        }
                                    }
                                    if found_collision {
                                        break;
                                    }
                                }
                                if found_collision {
                                    break;
                                }
                            }
                            if found_collision {
                                break;
                            }
                        }
                        if found_collision {
                            break;
                        }
                    }
                    if found_collision {
                        break;
                    }
                }
                if found_collision {
                    break;
                }
            }
            if found_collision {
                break;
            }
        }
        assert!(
            found_collision,
            "reduced stage-2 relation domain should not uniquely determine local round messages"
        );
    }

    #[test]
    fn stage2_norm_full_domain_matches_local_round_messages() {
        let full = stage2_full_prefix_points::<F>();
        let tau0 = F::from_u64(7);
        let tau1 = F::from_u64(11);
        let r0 = F::from_u64(13);

        let mut seen: HashMap<String, Vec<F>> = HashMap::new();
        for w00 in -4i64..=3 {
            for w10 in -4i64..=3 {
                for w01 in -4i64..=3 {
                    for w11 in -4i64..=3 {
                        let quad = [
                            F::from_i64(w00),
                            F::from_i64(w10),
                            F::from_i64(w01),
                            F::from_i64(w11),
                        ];
                        let storage = tensor_values(full, full, |x, y| {
                            stage2_local_norm_raw_eval(quad, x, y)
                        });
                        let target = stage2_norm_round_values(quad, tau0, tau1, r0);
                        let key = vec_key(&storage);
                        if let Some(existing) = seen.get(&key) {
                            assert_eq!(
                                existing, &target,
                                "full stage-2 norm domain lost information for a compact quad"
                            );
                        } else {
                            seen.insert(key, target);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage2_norm_8_point_reconstruction_matches_full_grid_and_round_messages() {
        let tau_choices = [F::zero(), F::one(), F::from_u64(2), F::from_u64(7)];
        let r0 = F::from_u64(13);

        for &tau0 in &tau_choices {
            for &tau1 in &tau_choices {
                let corner_weights = stage2_norm_corner_weights_from_taus(tau0, tau1);
                for w00 in -4i64..=3 {
                    for w10 in -4i64..=3 {
                        for w01 in -4i64..=3 {
                            for w11 in -4i64..=3 {
                                let quad = [
                                    F::from_i64(w00),
                                    F::from_i64(w10),
                                    F::from_i64(w01),
                                    F::from_i64(w11),
                                ];
                                let full_grid = stage2_full_grid_values(|x, y| {
                                    stage2_local_norm_raw_eval(quad, x, y)
                                });
                                let norm_claim =
                                    stage2_norm_claim_from_full_grid(full_grid, corner_weights);
                                let omitted_corner =
                                    default_stage2_norm_omitted_corner(corner_weights);
                                let compressed =
                                    Stage2CompressedGrid::from_full_grid(full_grid, omitted_corner);
                                let recovered = recover_stage2_norm_grid_from_claim(
                                    &compressed,
                                    corner_weights,
                                    norm_claim,
                                )
                                .expect("selected norm corner should be recoverable");

                                assert_eq!(
                                    recovered, full_grid,
                                    "norm full-grid reconstruction mismatch for quad={quad:?}, tau0={tau0:?}, tau1={tau1:?}"
                                );
                                assert_eq!(
                                    stage2_norm_round_values_from_full_grid(recovered, tau0, tau1, r0),
                                    stage2_norm_round_values(quad, tau0, tau1, r0),
                                    "norm round reconstruction mismatch for quad={quad:?}, tau0={tau0:?}, tau1={tau1:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage2_relation_full_domain_matches_local_round_messages() {
        let full = stage2_full_prefix_points::<F>();
        let r0 = F::from_u64(13);
        let alpha = F::one();
        let bit = [F::zero(), F::one()];

        let mut seen: HashMap<String, Vec<F>> = HashMap::new();
        for &w00 in &bit {
            for &w10 in &bit {
                for &w01 in &bit {
                    for &w11 in &bit {
                        let w_quad = [w00, w10, w01, w11];
                        for &m00 in &bit {
                            for &m10 in &bit {
                                for &m01 in &bit {
                                    for &m11 in &bit {
                                        let m_quad = [m00, m10, m01, m11];
                                        let storage = tensor_values(full, full, |x, y| {
                                            stage2_local_relation_eval(w_quad, m_quad, alpha, x, y)
                                        });
                                        let target =
                                            stage2_relation_round_values(w_quad, m_quad, r0);
                                        let key = vec_key(&storage);
                                        if let Some(existing) = seen.get(&key) {
                                            assert_eq!(
                                                existing, &target,
                                                "full stage-2 relation domain lost information"
                                            );
                                        } else {
                                            seen.insert(key, target);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage2_relation_8_point_reconstruction_matches_full_grid_and_round_messages() {
        let r0 = F::from_u64(13);
        let alpha_choices = [F::zero(), F::one(), F::from_u64(3)];
        let bit = [F::zero(), F::one()];

        for &alpha in &alpha_choices {
            for &w00 in &bit {
                for &w10 in &bit {
                    for &w01 in &bit {
                        for &w11 in &bit {
                            let w_quad = [w00, w10, w01, w11];
                            for &m00 in &bit {
                                for &m10 in &bit {
                                    for &m01 in &bit {
                                        for &m11 in &bit {
                                            let m_quad = [m00, m10, m01, m11];
                                            let full_grid = stage2_full_grid_values(|x, y| {
                                                stage2_local_relation_eval(
                                                    w_quad, m_quad, alpha, x, y,
                                                )
                                            });
                                            let relation_claim =
                                                stage2_relation_claim_from_full_grid(full_grid);
                                            let compressed = Stage2CompressedGrid::from_full_grid(
                                                full_grid,
                                                BooleanCorner::DEFAULT_STAGE2_RELATION,
                                            );
                                            let recovered = recover_stage2_relation_grid_from_claim(
                                                &compressed,
                                                relation_claim,
                                            );

                                            assert_eq!(
                                                recovered, full_grid,
                                                "relation full-grid reconstruction mismatch"
                                            );
                                            assert_eq!(
                                                stage2_relation_round_values_from_full_grid(
                                                    recovered, r0
                                                ),
                                                stage2_relation_round_values(w_quad, m_quad, r0)
                                                    .into_iter()
                                                    .map(|value| alpha * value)
                                                    .collect::<Vec<_>>(),
                                                "relation round reconstruction mismatch"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
