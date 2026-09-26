use super::common::*;
#[cfg(test)]
use akita_algebra::eq_poly::EqPolynomial;
use akita_sumcheck::reduce_signed_accum;
use jolt_field::{Field, Ring, Unreduced, Zero};
use jolt_poly::OmittedConstantPoly;
#[cfg(test)]
use jolt_poly::UnivariatePoly;

/// Candidate stage-1 domain `{1, -1, 2, Infinity}`.
#[cfg(test)]
pub(crate) fn stage1_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 4] {
    [
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

/// Safe full stage-1 fallback domain `{0, 1, -1, 2, Infinity}`.
#[cfg(test)]
pub(crate) fn stage1_full_prefix_points<E: Field + Ring>() -> [PrefixPoint<E>; 5] {
    [
        PrefixPoint::Finite(E::zero()),
        PrefixPoint::Finite(E::one()),
        PrefixPoint::Finite(E::zero() - E::one()),
        PrefixPoint::Finite(E::from_u64(2)),
        PrefixPoint::Infinity,
    ]
}

/// Internal stage-1 first-two-round interpolation grid.
///
/// This is built and consumed inside the prover to reconstruct ordinary
/// eq-factored sumcheck round messages; it is not serialized in the Akita proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Stage1PrefixGrid<E: Field> {
    pub evals_except_boolean_core: Vec<E>,
}

#[inline]
pub(crate) fn stage1_full_grid_index(x_idx: usize, y_idx: usize) -> usize {
    x_idx * 5 + y_idx
}

#[inline]
pub(crate) fn stage1_is_boolean_corner(x_idx: usize, y_idx: usize) -> bool {
    x_idx < 2 && y_idx < 2
}

#[inline]
pub(crate) fn stage1_quartic_coeffs_from_prefix_values<E: Field + Ring>(values: [E; 5]) -> [E; 5] {
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
pub(crate) fn stage1_eval_quartic_from_prefix_values<E: Field + Ring>(values: [E; 5], x: E) -> E {
    let [a0, a1, a2, a3, a4] = stage1_quartic_coeffs_from_prefix_values(values);
    a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)))
}

#[inline]
pub(crate) fn eval_stage1_biquartic_from_full_grid<E: Field + Ring>(
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

/// Build the stage-1 first-two-round prefix grid from the compact witness by
/// summing the round-2 equality weight of every quad into its quad class.
///
/// `w_compact` is the flat live prefix of the witness table and `tau0` the
/// stage-1 point in binding order.
#[cfg(test)]
pub(crate) fn build_stage1_prefix_grid_from_m_compact<E: Field + Ring + Unreduced>(
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

/// Build the cache for the first two stage-1 rounds.
///
/// `quad_class_weights[c]` is the sum of `eq(tau0[2..], q)` over the live quads
/// `q` of class `c`, where a quad's class packs the classes `k` of its four
/// range images `k(k+1)`, first entry lowest. Quads of class zero contribute
/// nothing, so their weight may be omitted.
#[tracing::instrument(skip_all, name = "two_round_prefix::build_stage1_prefix_cache")]
pub(crate) fn build_stage1_prefix_cache<E: Field + Ring + Unreduced>(
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

#[cfg(test)]
pub(crate) fn stage1_storage_vector_from_quad<E: Field + Ring>(quad: [E; 4], b: usize) -> Vec<E> {
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

/// Cache needed to reconstruct the first two ordinary stage-1 round messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1B4PrefixCache<E: Field> {
    x_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage1B8PrefixCache<E: Field> {
    pub(crate) full_grid: [E; 25],
    pub(crate) tau0: E,
    pub(crate) tau1: E,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Stage1PrefixCache<E: Field> {
    B4(Stage1B4PrefixCache<E>),
    B8(Stage1B8PrefixCache<E>),
}

impl<E: Field + Ring> Stage1PrefixCache<E> {
    pub(super) fn new(proof: &Stage1PrefixGrid<E>, tau0: &[E], b: usize) -> Option<Self> {
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
    pub(crate) fn reconstruct_round0_poly(&self) -> UnivariatePoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round0_poly(),
            Self::B8(state) => state.reconstruct_round0_poly(),
        }
    }

    #[cfg(test)]
    pub(crate) fn reconstruct_round1_poly(&self, r0: E) -> UnivariatePoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round1_poly(r0),
            Self::B8(state) => state.reconstruct_round1_poly(r0),
        }
    }

    pub(crate) fn reconstruct_round0_eq_poly(&self) -> OmittedConstantPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round0_eq_poly(),
            Self::B8(state) => state.reconstruct_round0_eq_poly(),
        }
    }

    pub(crate) fn reconstruct_round1_eq_poly(&self, r0: E) -> OmittedConstantPoly<E> {
        match self {
            Self::B4(state) => state.reconstruct_round1_eq_poly(r0),
            Self::B8(state) => state.reconstruct_round1_eq_poly(r0),
        }
    }
}

impl<E: Field + Ring> Stage1B4PrefixCache<E> {
    #[cfg(test)]
    fn reconstruct_round0_poly(&self) -> UnivariatePoly<E> {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.x_row_coeffs[1], self.tau1),
        );
        coeff_array_to_poly(mul_linear_by_quadratic_coeffs(self.tau0, q_x))
    }

    #[cfg(test)]
    fn reconstruct_round1_poly(&self, r0: E) -> UnivariatePoly<E> {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(self.x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        let round0_eq = linear_eq_eval(self.tau0, r0);
        let coeffs = mul_linear_by_quadratic_coeffs(self.tau1, q_y).map(|coeff| round0_eq * coeff);
        coeff_array_to_poly(coeffs)
    }

    pub(crate) fn reconstruct_round0_eq_poly(&self) -> OmittedConstantPoly<E> {
        let q_x = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.x_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.x_row_coeffs[1], self.tau1),
        );
        OmittedConstantPoly::from_q_coefficients(q_x.into())
    }

    pub(crate) fn reconstruct_round1_eq_poly(&self, r0: E) -> OmittedConstantPoly<E> {
        let y_values: [E; 3] =
            std::array::from_fn(|y_idx| eval_quadratic_from_coeffs(self.x_row_coeffs[y_idx], r0));
        let q_y = quadratic_coeffs_from_01_inf(y_values[0], y_values[1], y_values[2]);
        OmittedConstantPoly::from_q_coefficients(q_y.into())
    }
}

impl<E: Field + Ring> Stage1B8PrefixCache<E> {
    #[cfg(test)]
    fn reconstruct_round0_poly(&self) -> UnivariatePoly<E> {
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
        let mut polynomial = UnivariatePoly::from_evals(&evals);
        polynomial.trim_trailing_zeros();
        polynomial
    }

    #[cfg(test)]
    fn reconstruct_round1_poly(&self, r0: E) -> UnivariatePoly<E> {
        let l0_at_r0 = linear_eq_eval(self.tau0, r0);
        let evals: Vec<E> = (0..=5u64)
            .map(|y_raw| {
                let y = E::from_u64(y_raw);
                l0_at_r0
                    * linear_eq_eval(self.tau1, y)
                    * eval_stage1_biquartic_from_full_grid(self.full_grid, r0, y)
            })
            .collect();
        let mut polynomial = UnivariatePoly::from_evals(&evals);
        polynomial.trim_trailing_zeros();
        polynomial
    }

    pub(crate) fn reconstruct_round0_eq_poly(&self) -> OmittedConstantPoly<E> {
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

    pub(crate) fn reconstruct_round1_eq_poly(&self, r0: E) -> OmittedConstantPoly<E> {
        let evals: Vec<E> = (0..=4u64)
            .map(|y_raw| {
                let y = E::from_u64(y_raw);
                eval_stage1_biquartic_from_full_grid(self.full_grid, r0, y)
            })
            .collect();
        interpolate_eq_factored_q_poly(&evals, STAGE1_B8_Q_POLY_DEGREE)
    }
}
