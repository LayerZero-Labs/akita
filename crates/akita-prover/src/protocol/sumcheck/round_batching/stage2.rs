use super::common::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::parallel::*;
use akita_field::unreduced::HasUnreducedOps;
use akita_field::{FieldCore, FromPrimitiveInt, Zero};
use akita_sumcheck::{reduce_signed_accum, UniPoly};

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
    pub(crate) fn boolean_index(self) -> usize {
        match self {
            Self::ZeroZero => 0,
            Self::ZeroOne => 1,
            Self::OneZero => 2,
            Self::OneOne => 3,
        }
    }

    #[inline]
    pub(crate) fn grid_index(self) -> usize {
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
pub(crate) struct OmittedCornerEvaluationGrid<E: FieldCore> {
    pub omitted_corner: BooleanCorner,
    pub evals_except_corner: [E; 8],
}

impl<E: FieldCore> OmittedCornerEvaluationGrid<E> {
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
pub(crate) struct Stage2RoundBatchGrid<E: FieldCore> {
    pub norm: OmittedCornerEvaluationGrid<E>,
    pub relation: OmittedCornerEvaluationGrid<E>,
}

/// Return the stage-2 full-domain grid in first-local-round-major order over
/// the first-two-round local batch domain.
#[cfg(test)]
pub(crate) fn stage2_full_grid_values<E: FieldCore + FromPrimitiveInt>(
    mut eval: impl FnMut(RoundBatchPoint<E>, RoundBatchPoint<E>) -> E,
) -> [E; 9] {
    let points = stage2_initial_batch_points::<E>();
    std::array::from_fn(|idx| {
        let first_round_point = points[idx / 3];
        let second_round_point = points[idx % 3];
        eval(first_round_point, second_round_point)
    })
}

/// Evaluate a biquadratic from its full first-two-round local batch grid.
#[inline]
#[cfg(test)]
pub(crate) fn eval_biquadratic_from_full_grid<E: FieldCore>(
    full_grid: [E; 9],
    first_round_point: RoundBatchPoint<E>,
    second_round_point: RoundBatchPoint<E>,
) -> E {
    let at_second_zero =
        eval_quadratic_from_01_inf(full_grid[0], full_grid[3], full_grid[6], first_round_point);
    let at_second_one =
        eval_quadratic_from_01_inf(full_grid[1], full_grid[4], full_grid[7], first_round_point);
    let at_second_infinity =
        eval_quadratic_from_01_inf(full_grid[2], full_grid[5], full_grid[8], first_round_point);
    eval_quadratic_from_01_inf(
        at_second_zero,
        at_second_one,
        at_second_infinity,
        second_round_point,
    )
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
/// norm half from the two local eq factors bound by `tau0` and `tau1`.
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
    compressed: &OmittedCornerEvaluationGrid<E>,
    corner_weights: [E; 4],
    claim: E,
) -> Option<[E; 9]> {
    let omitted_weight = corner_weights[compressed.omitted_corner.boolean_index()];
    let omitted_weight_inv = omitted_weight.inverse()?;
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
    compressed: &OmittedCornerEvaluationGrid<E>,
    relation_claim: E,
) -> [E; 9] {
    recover_stage2_grid_from_corner_claim(compressed, [E::one(); 4], relation_claim)
        .expect("relation corner weights are all one")
}

/// Flat Stage-2 witness layout plus the optional local embedding used by the
/// first-two-round batch optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stage2InitialRoundBatchLayout {
    pub live_len: usize,
    pub num_vars: usize,
    pub live_tiles: usize,
    pub tile_bits: usize,
    pub lane_bits: usize,
}

impl Stage2InitialRoundBatchLayout {
    pub(crate) fn new(
        live_len: usize,
        num_vars: usize,
        live_tiles: usize,
        tile_bits: usize,
        lane_bits: usize,
    ) -> Option<Self> {
        let lane_len = 1usize.checked_shl(u32::try_from(lane_bits).ok()?)?;
        let tile_capacity = 1usize.checked_shl(u32::try_from(tile_bits).ok()?)?;
        let domain_len = 1usize.checked_shl(u32::try_from(num_vars).ok()?)?;
        if live_tiles == 0
            || live_tiles > tile_capacity
            || live_len != live_tiles.checked_mul(lane_len)?
            || num_vars != tile_bits.checked_add(lane_bits)?
            || live_len > domain_len
        {
            return None;
        }
        Some(Self {
            live_len,
            num_vars,
            live_tiles,
            tile_bits,
            lane_bits,
        })
    }

    #[inline]
    fn lane_len(self) -> usize {
        1usize << self.lane_bits
    }
}

/// Whether stage 2 has enough lane-local rounds to use the 2-round batch path.
#[inline]
pub(crate) fn can_use_stage2_initial_round_batch(lane_bits: usize, b: usize) -> bool {
    lane_bits >= 2 && matches!(b, 4 | 8)
}

/// Build the stage-2 first-two-round bivariate-skip payload from the compact
/// witness table at the start of stage 2.
///
/// Returns `None` when the flat layout has no valid local two-round embedding.
#[tracing::instrument(
    skip_all,
    name = "round_batching::build_stage2_initial_round_batch_grid"
)]
pub(crate) fn build_stage2_initial_round_batch_grid<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
>(
    w_compact: &[i8],
    relation_weight_evals: &[E],
    stage1_point: &[E],
    b: usize,
    layout: Stage2InitialRoundBatchLayout,
) -> Option<Stage2RoundBatchGrid<E>> {
    if !can_use_stage2_initial_round_batch(layout.lane_bits, b) {
        return None;
    }

    let lane_len = layout.lane_len();
    assert_eq!(relation_weight_evals.len(), layout.live_len);
    assert_eq!(w_compact.len(), layout.live_len);
    assert_eq!(stage1_point.len(), layout.num_vars);

    let eq_lane_suffix = EqPolynomial::evals(&stage1_point[2..layout.lane_bits])
        .expect("stage-2 two-round batch dimensions are prevalidated");
    let eq_tile = EqPolynomial::evals(&stage1_point[layout.lane_bits..])
        .expect("stage-2 tile dimensions are prevalidated");
    let lane_quads = lane_len >> 2;
    debug_assert_eq!(eq_lane_suffix.len(), lane_quads);
    let norm_omitted_corner = default_stage2_norm_omitted_corner(
        stage2_norm_corner_weights_from_taus(stage1_point[0], stage1_point[1]),
    );
    let norm_point_indices =
        &STAGE2_COMPRESSED_POINT_INDICES_BY_OMITTED_CORNER[norm_omitted_corner.boolean_index()];

    let w_digit_fn: fn(i8) -> usize = match b {
        4 => stage2_b4_w_digit,
        8 => stage2_b8_w_digit,
        _ => unreachable!("unsupported stage-2 initial round batch basis"),
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
        0..layout.live_tiles,
        || {
            (
                [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
                [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
                [E::ProductAccum::zero(); STAGE2_COMPRESSED_POINT_COUNT],
            )
        },
        |(mut norm_pos, mut norm_neg, mut rel_accum), tile_idx| {
            let column = &w_compact[tile_idx * lane_len..(tile_idx + 1) * lane_len];
            let weight_column =
                &relation_weight_evals[tile_idx * lane_len..(tile_idx + 1) * lane_len];
            let eq_tile_weight = eq_tile[tile_idx];
            let mut tile_rel_pos = [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT];
            let mut tile_rel_neg = [E::MulU64Accum::zero(); STAGE2_COMPRESSED_POINT_COUNT];
            for (lane_quad, &eq_lane_weight) in eq_lane_suffix.iter().enumerate() {
                let base = 4 * lane_quad;
                let lookup_idx = lookup_index_fn([
                    w_digit_fn(column[base]),
                    w_digit_fn(column[base + 1]),
                    w_digit_fn(column[base + 2]),
                    w_digit_fn(column[base + 3]),
                ]);
                let norm_weight = eq_lane_weight * eq_tile_weight;
                accum_lookup_vector_signed_selected(
                    &mut norm_pos,
                    &mut norm_neg,
                    norm_weight,
                    &norm_table[lookup_idx],
                    norm_point_indices,
                );
                let weight_quad = std::array::from_fn(|offset| weight_column[base + offset]);
                let weight_point_values = stage2_relation_m_point_values_compressed(weight_quad);
                accum_pointwise_signed(
                    &mut tile_rel_pos,
                    &mut tile_rel_neg,
                    &weight_point_values,
                    &rel_table[lookup_idx],
                );
            }
            for idx in 0..STAGE2_COMPRESSED_POINT_COUNT {
                let tile_rel = reduce_signed_accum::<E>(tile_rel_pos[idx], tile_rel_neg[idx]);
                rel_accum[idx] += E::one().mul_to_product_accum(tile_rel);
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
    Some(Stage2RoundBatchGrid {
        norm: OmittedCornerEvaluationGrid {
            omitted_corner: norm_omitted_corner,
            evals_except_corner: norm_evals_except_corner,
        },
        relation: OmittedCornerEvaluationGrid {
            omitted_corner: BooleanCorner::DEFAULT_STAGE2_RELATION,
            evals_except_corner: relation_evals_except_corner,
        },
    })
}

/// State needed to reconstruct the first two ordinary stage-2 round messages
/// from the internal bivariate-skip payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stage2RoundBatchState<E: FieldCore> {
    norm_first_round_row_coeffs: [[E; 3]; 3],
    relation_first_round_row_coeffs: [[E; 3]; 3],
    tau0: E,
    tau1: E,
    batching_coeff: E,
}

impl<E: FieldCore> Stage2RoundBatchState<E> {
    pub(crate) fn new(
        proof: &Stage2RoundBatchGrid<E>,
        stage1_point: &[E],
        s_claim: E,
        relation_claim: E,
        batching_coeff: E,
    ) -> Option<Self> {
        if stage1_point.len() < 2 {
            return None;
        }
        let tau0 = stage1_point[0];
        let tau1 = stage1_point[1];
        let norm_full_grid = recover_stage2_grid_from_corner_claim(
            &proof.norm,
            stage2_norm_corner_weights_from_taus(tau0, tau1),
            s_claim,
        )?;
        let relation_full_grid =
            recover_stage2_relation_grid_from_claim(&proof.relation, relation_claim);
        let norm_first_round_row_coeffs = std::array::from_fn(|row_idx| {
            quadratic_coeffs_from_01_inf(
                norm_full_grid[row_idx],
                norm_full_grid[3 + row_idx],
                norm_full_grid[6 + row_idx],
            )
        });
        let relation_first_round_row_coeffs = std::array::from_fn(|row_idx| {
            quadratic_coeffs_from_01_inf(
                relation_full_grid[row_idx],
                relation_full_grid[3 + row_idx],
                relation_full_grid[6 + row_idx],
            )
        });
        Some(Self {
            norm_first_round_row_coeffs,
            relation_first_round_row_coeffs,
            tau0,
            tau1,
            batching_coeff,
        })
    }
}

impl<E: FieldCore + FromPrimitiveInt> Stage2RoundBatchState<E> {
    #[inline]
    pub(crate) fn reconstruct_round0_polys(&self) -> (UniPoly<E>, UniPoly<E>) {
        let norm_q = add_quadratic_coeffs(
            scale_quadratic_coeffs(self.norm_first_round_row_coeffs[0], E::one() - self.tau1),
            scale_quadratic_coeffs(self.norm_first_round_row_coeffs[1], self.tau1),
        );
        let mut norm_coeffs = mul_linear_by_quadratic_coeffs(self.tau0, norm_q);
        for coeff in &mut norm_coeffs {
            *coeff = self.batching_coeff * *coeff;
        }
        let relation_coeffs = add_quadratic_coeffs(
            self.relation_first_round_row_coeffs[0],
            self.relation_first_round_row_coeffs[1],
        );
        (
            UniPoly::from_coeffs(norm_coeffs.to_vec()),
            UniPoly::from_coeffs(relation_coeffs.to_vec()),
        )
    }

    #[inline]
    pub(crate) fn reconstruct_round1_polys(&self, r0: E) -> (UniPoly<E>, UniPoly<E>) {
        let norm_second_round_values: [E; 3] = std::array::from_fn(|row_idx| {
            eval_quadratic_from_coeffs(self.norm_first_round_row_coeffs[row_idx], r0)
        });
        let norm_q = quadratic_coeffs_from_01_inf(
            norm_second_round_values[0],
            norm_second_round_values[1],
            norm_second_round_values[2],
        );
        let round0_eq = linear_eq_eval(self.tau0, r0);
        let mut norm_coeffs = mul_linear_by_quadratic_coeffs(self.tau1, norm_q);
        for coeff in &mut norm_coeffs {
            *coeff = self.batching_coeff * round0_eq * *coeff;
        }
        let relation_second_round_values: [E; 3] = std::array::from_fn(|row_idx| {
            eval_quadratic_from_coeffs(self.relation_first_round_row_coeffs[row_idx], r0)
        });
        let relation_coeffs = quadratic_coeffs_from_01_inf(
            relation_second_round_values[0],
            relation_second_round_values[1],
            relation_second_round_values[2],
        );
        (
            UniPoly::from_coeffs(norm_coeffs.to_vec()),
            UniPoly::from_coeffs(relation_coeffs.to_vec()),
        )
    }
}
