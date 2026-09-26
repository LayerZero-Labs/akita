use super::common::*;
use super::stage1::*;
use super::stage2::*;
use crate::opaque::sumcheck::digit_range::range_poly::RangePoly;
use crate::opaque::LowBasisRangeCheckProver;
use akita_algebra::eq_poly::EqPolynomial;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_sumcheck::EqFactoredSumcheckInstanceProver;
use akita_types::DigitRangeEqualityPoint;
use jolt_field::{Field, One, Prime128Offset275, Ring, Unreduced, Zero};
use jolt_poly::{OmittedConstantPoly, UnivariatePoly};
use std::collections::HashMap;

type F = Prime128Offset275;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixPoint<E: Field> {
    Finite(E),
    Infinity,
}

fn stage1_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 4] {
    [
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

fn stage1_full_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 5] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

fn stage2_reduced_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 2] {
    [PrefixPoint::Finite(E::one()), PrefixPoint::Infinity]
}

fn stage2_full_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 3] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Infinity,
    ]
}

#[inline]
fn bilinear_coeffs_from_quad<E: Field>(quad: [E; 4]) -> [E; 4] {
    let [t00, t10, t01, t11] = quad;
    [t00, t10 - t00, t01 - t00, t11 - t10 - t01 + t00]
}

#[inline]
fn bilinear_eval<E: Field>(quad: [E; 4], x: E, y: E) -> E {
    let [a, b, c, d] = bilinear_coeffs_from_quad(quad);
    a + x * (b + y * d) + y * c
}

#[inline]
fn bilinear_eval_on_prefix_points<E: Field>(
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

#[inline]
fn stage1_local_norm_eval<E: Field + Ring>(
    s_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
    b: usize,
) -> E {
    RangePoly::new(b).eval(bilinear_eval_on_prefix_points(s_quad, x, y))
}

#[inline]
fn stage1_local_norm_raw_eval<E: Field + Ring>(
    s_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
    b: usize,
) -> E {
    let [_, bx, cy, dxy] = bilinear_coeffs_from_quad(s_quad);
    let degree = RangePoly::new(b).num_coefficients();
    let pow = |base: E| {
        let mut out = E::one();
        for _ in 0..degree {
            out *= base;
        }
        out
    };

    match (x, y) {
        (PrefixPoint::Finite(x), PrefixPoint::Finite(y)) => {
            RangePoly::new(b).eval(bilinear_eval(s_quad, x, y))
        }
        (PrefixPoint::Infinity, PrefixPoint::Finite(y)) => pow(bx + y * dxy),
        (PrefixPoint::Finite(x), PrefixPoint::Infinity) => pow(cy + x * dxy),
        (PrefixPoint::Infinity, PrefixPoint::Infinity) => pow(dxy),
    }
}

#[inline]
fn stage2_local_norm_candidate_eval<E: Field>(
    w_quad: [E; 4],
    x: PrefixPoint<E>,
    y: PrefixPoint<E>,
) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    w_eval * (w_eval + E::one())
}

#[inline]
fn stage2_local_norm_raw_eval<E: Field>(w_quad: [E; 4], x: PrefixPoint<E>, y: PrefixPoint<E>) -> E {
    let w_eval = bilinear_eval_on_prefix_points(w_quad, x, y);
    match (x, y) {
        (PrefixPoint::Finite(_), PrefixPoint::Finite(_)) => w_eval * (w_eval + E::one()),
        _ => w_eval * w_eval,
    }
}

/// Build the stage-1 first-two-round prefix grid from the compact witness by
/// summing the round-2 equality weight of every quad into its quad class.
///
/// `w_compact` is the flat live prefix of the witness table and `tau0` the
/// stage-1 point in binding order.
fn build_stage1_prefix_grid_from_m_compact<E: Field + Ring + Unreduced>(
    w_compact: &[i8],
    tau0: &[E],
    b: usize,
) -> Stage1PrefixGrid<E> {
    let class_bits = b / 4;
    let quad_weights =
        EqPolynomial::evals(&tau0[2..]).expect("stage-1 prefix dimensions are prevalidated");
    let mut quad_class_weights = vec![E::zero(); 1usize << (4 * class_bits)];
    for (quad, digits) in w_compact.chunks(4).enumerate() {
        let class = digits
            .iter()
            .enumerate()
            .fold(0usize, |class, (offset, &w)| {
                class | (usize::from((w ^ (w >> 7)) as u8) << (class_bits * offset))
            });
        quad_class_weights[class] += quad_weights[quad];
    }
    build_stage1_prefix_grid(&quad_class_weights, b)
}

fn stage1_storage_vector_from_quad<E: Field + Ring>(quad: [E; 4], b: usize) -> Vec<E> {
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

fn reconstruct_stage1_round0_poly<E: Field + Ring>(
    cache: &Stage1PrefixCache<E>,
) -> UnivariatePoly<E> {
    if let Some((x_row_coeffs, tau0, tau1)) = cache.b4_test_data() {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(x_row_coeffs[0], E::one() - tau1),
            scale_quadratic_coeffs(x_row_coeffs[1], tau1),
        );
        return UnivariatePoly::new(mul_linear_by_quadratic_coeffs(tau0, q_x).to_vec());
    }

    let (full_grid, tau0, tau1) = cache
        .b8_test_data()
        .expect("cache must contain one stage-1 basis");
    let l1_at_0 = E::one() - tau1;
    let l1_at_1 = tau1;
    let evals: Vec<E> = (0..=5u64)
        .map(|x_raw| {
            let x = E::from_u64(x_raw);
            let q_x0 = eval_stage1_biquartic_from_full_grid(full_grid, x, E::zero());
            let q_x1 = eval_stage1_biquartic_from_full_grid(full_grid, x, E::one());
            linear_eq_eval(tau0, x) * (l1_at_0 * q_x0 + l1_at_1 * q_x1)
        })
        .collect();
    let mut polynomial = UnivariatePoly::from_evals(&evals);
    polynomial.trim_trailing_zeros();
    polynomial
}

fn reconstruct_stage1_round1_poly<E: Field + Ring>(
    cache: &Stage1PrefixCache<E>,
    r0: E,
) -> UnivariatePoly<E> {
    if let Some((x_row_coeffs, tau0, tau1)) = cache.b4_test_data() {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        let round0_eq = linear_eq_eval(tau0, r0);
        let coeffs = mul_linear_by_quadratic_coeffs(tau1, q_y).map(|coeff| round0_eq * coeff);
        return UnivariatePoly::new(coeffs.to_vec());
    }

    let (full_grid, tau0, tau1) = cache
        .b8_test_data()
        .expect("cache must contain one stage-1 basis");
    let l0_at_r0 = linear_eq_eval(tau0, r0);
    let evals: Vec<E> = (0..=5u64)
        .map(|y_raw| {
            let y = E::from_u64(y_raw);
            l0_at_r0
                * linear_eq_eval(tau1, y)
                * eval_stage1_biquartic_from_full_grid(full_grid, r0, y)
        })
        .collect();
    let mut polynomial = UnivariatePoly::from_evals(&evals);
    polynomial.trim_trailing_zeros();
    polynomial
}

fn packed(witness: &[i8]) -> crate::sources::packed_digits::PackedSignedDigits {
    crate::sources::packed_digits::PackedSignedDigits::from_i8_digits_auto(witness.to_vec())
}

fn ordered_equality_point(
    challenges: &[F],
    column_variables: usize,
    ring_variables: usize,
) -> Vec<F> {
    DigitRangeEqualityPoint::from_column_then_ring_challenges(
        challenges,
        column_variables,
        ring_variables,
    )
    .expect("valid test point")
    .into_coordinates()
}

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
        let pivot_inv = rows[rank][col].inverse().expect("pivot must be invertible");
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

fn tensor_values<E: Field, const NX: usize, const NY: usize>(
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
    let q = |x: F, y: F| RangePoly::new(b).eval(bilinear_eval(s_quad, x, y));

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

fn build_stage1_prefix_grid_from_m_compact_reference(
    w_compact: &[i8],
    tau0: &[F],
    b: usize,
    live_x_cols: usize,
    _col_bits: usize,
    ring_bits: usize,
) -> Stage1PrefixGrid<F> {
    let y_len = 1usize << ring_bits;
    let eq_y_suffix = EqPolynomial::evals(&tau0[2..ring_bits])
        .expect("stage-1 reference two-round prefix dimensions are prevalidated");
    let eq_x = EqPolynomial::evals(&tau0[ring_bits..])
        .expect("stage-1 reference x-prefix dimensions are prevalidated");
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

    Stage1PrefixGrid {
        evals_except_boolean_core,
    }
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
                                stage1_local_norm_raw_eval(quad, points[x_idx], points[y_idx], 8,),
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
                    let lookup =
                        &STAGE2_B8_NORM_LOOKUP_TABLE[stage2_b8_lookup_index_from_digits([
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
fn stage2_norm_histogram_matches_local_round_messages() {
    let tau0 = F::from_u64(1_234_567);
    let tau1 = F::from_u64(7_654_321);
    let batching_coeff = F::from_u64(424_242);
    let r0 = F::from_u64(98_765);
    for b in [4usize, 8] {
        let bits = b.trailing_zeros();
        let half = (b / 2) as i64;
        let mut histogram = vec![F::zero(); b.pow(4)];
        let mut round0 = [F::zero(); 4];
        let mut round1 = [F::zero(); 4];
        for sample in 0..97u64 {
            let digits: [usize; 4] = std::array::from_fn(|i| {
                ((sample * (7 + 2 * i as u64) + i as u64) % b as u64) as usize
            });
            let class = digits
                .iter()
                .enumerate()
                .fold(0usize, |class, (i, &d)| class | (d << (i as u32 * bits)));
            let weight = F::from_u64(1_000 + 31 * sample);
            histogram[class] += weight;
            let quad = digits.map(|d| F::from_i64(d as i64 - half));
            let values = stage2_norm_round_values(quad, tau0, tau1, r0);
            for point in 0..4 {
                round0[point] += weight * values[point];
                round1[point] += weight * values[4 + point];
            }
        }
        let cache =
            Stage2PrefixCache::from_norm_histogram(&histogram, b, tau0, tau1, batching_coeff);
        let poly0 = cache.round0_norm_poly();
        let poly1 = cache.round1_norm_poly(r0);
        for point in 0..4u64 {
            let x = F::from_u64(point);
            assert_eq!(
                poly0.evaluate(x),
                batching_coeff * round0[point as usize],
                "b={b} round 0 at {point}"
            );
            assert_eq!(
                poly1.evaluate(x),
                batching_coeff * round1[point as usize],
                "b={b} round 1 at {point}"
            );
        }
    }
}

#[test]
fn stage1_prefix_proof_builder_matches_reference() {
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
    let tau0 = ordered_equality_point(&tau0_raw, col_bits, ring_bits);
    assert_eq!(
        build_stage1_prefix_grid_from_m_compact(&w_compact, &tau0, 8),
        build_stage1_prefix_grid_from_m_compact_reference(
            &w_compact, &tau0, 8, 5, col_bits, ring_bits,
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
                    let proof = Stage1PrefixGrid {
                        evals_except_boolean_core: stage1_storage_vector_from_quad(quad, 8),
                    };
                    let cache = Stage1PrefixCache::new(&proof, &[tau0, tau1], 8)
                        .expect("stage1 prefix state should build");
                    let round_values = stage1_norm_round_values(quad, tau0, tau1, r0, 8);
                    let mut round0 = UnivariatePoly::from_evals(&round_values[..6]);
                    round0.trim_trailing_zeros();
                    let mut round1 = UnivariatePoly::from_evals(&round_values[6..]);
                    round1.trim_trailing_zeros();
                    assert_eq!(reconstruct_stage1_round0_poly(&cache), round0);
                    assert_eq!(reconstruct_stage1_round1_poly(&cache, r0), round1);
                }
            }
        }
    }
}

#[test]
fn stage1_prefix_proof_reconstructs_first_two_rounds() {
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
    let tau0 = ordered_equality_point(&tau0_raw, col_bits, ring_bits);

    let proof = build_stage1_prefix_grid_from_m_compact(&w_compact, &tau0, b);
    let cache = Stage1PrefixCache::new(&proof, &tau0, b).expect("stage1 prefix state should build");

    let mut prover = LowBasisRangeCheckProver::<F>::new(
        packed(&w_compact),
        &tau0,
        akita_types::DigitRangePlan::new(b).unwrap(),
        live_x_cols,
        col_bits,
        ring_bits,
    )
    .unwrap();
    let round0 = prover.compute_round_eq_factored(0);
    assert_eq!(cache.reconstruct_round0_eq_poly(), round0);

    let r0 = F::from_u64(9);
    prover.ingest_challenge(0, r0);

    let round1 = prover.compute_round_eq_factored(1);
    assert_eq!(cache.reconstruct_round1_eq_poly(r0), round1);
}

#[test]
fn stage1_b8_reconstructed_eq_polys_keep_degree4_storage_width() {
    let state = Stage1B8PrefixCache {
        full_grid: [F::zero(); 25],
        tau0: F::from_u64(3),
        tau1: F::from_u64(5),
    };

    for poly in [
        state.reconstruct_round0_eq_poly(),
        state.reconstruct_round1_eq_poly(F::from_u64(7)),
    ] {
        assert_eq!(poly.coefficients().len(), STAGE1_B8_Q_POLY_DEGREE);
        assert_eq!(
            poly.coefficients(),
            vec![F::zero(); STAGE1_B8_Q_POLY_DEGREE]
        );

        let mut bytes = Vec::new();
        poly.serialize_uncompressed(&mut bytes)
            .expect("eq-factored poly should serialize");
        let decoded = OmittedConstantPoly::<F>::deserialize_uncompressed(
            &bytes[..],
            &STAGE1_B8_Q_POLY_DEGREE,
        )
        .expect("eq-factored poly should deserialize at degree 4");
        assert_eq!(decoded, poly);
    }
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
                    let storage =
                        tensor_values(full, full, |x, y| stage2_local_norm_raw_eval(quad, x, y));
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
