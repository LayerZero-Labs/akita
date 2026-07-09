use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::scalar_powers;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::layout::CommitmentRingDims;
use crate::{
    SetupContributionGroupInputs, SetupContributionPlanInputs, SetupContributionStatic,
    WitnessChunkLayout,
};

const POSSIBLE_CARRIES: usize = 2;

/// Succinct evaluator for the setup-index weight multilinear extension.
///
/// For a setup-index point `rho`, this evaluates the same polynomial as the
/// materialized packed setup weight vector:
///
/// ```text
/// setup_index_weight~(rho)
///   = sum_g D_g(rho) + B_g(rho) + A_g(rho).
/// ```
///
/// For each role, a base setup index is decomposed as
///
/// ```text
/// base_idx = lane + ratio_role * (col + width_role * row),
/// ```
///
/// where `ratio_role = d_role / setup_ring_dim`. The lane contribution is the
/// corresponding alpha-projection scale, the row contribution is
/// `eq(tau1, row_start + row)`, and the column contribution is evaluated from
/// the chunk offsets and gadget formulas directly.
#[derive(Clone)]
pub struct SetupIndexWeightEvaluator<E> {
    tau1: Vec<E>,
    x_challenges: Vec<E>,
    groups: Vec<SetupContributionGroupInputs>,
    d_row_start: usize,
    d_rows: usize,
    d_physical_cols: usize,
    a_projection: SetupRoleProjection<E>,
    b_projection: SetupRoleProjection<E>,
    d_projection: SetupRoleProjection<E>,
    fold_gadget: Vec<E>,
    required: usize,
}

#[derive(Clone)]
struct SetupRoleProjection<E> {
    ratio: usize,
    scales: Vec<E>,
}

impl<E: FieldCore> SetupIndexWeightEvaluator<E> {
    /// Build a succinct evaluator for `setup_index_weight~`.
    ///
    /// `setup_ring_dim` is the base ring dimension used by the setup prefix
    /// being checked. For uniform dimensions this is the common role dimension;
    /// for mixed dimensions it should be the base dimension onto which A, B,
    /// and D are projected.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F>(
        inputs: &SetupContributionPlanInputs<E>,
        static_plan: &SetupContributionStatic<E>,
        groups: &[SetupContributionGroupInputs],
        tau1: &[E],
        x_challenges: &[E],
        fold_gadget: &[F],
        setup_ring_dim: usize,
        role_dims: CommitmentRingDims,
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: MulBase<F>,
    {
        if groups.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "setup-index weight evaluator requires at least one group".into(),
            ));
        }
        if setup_ring_dim == 0 || !setup_ring_dim.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "setup-index weight base ring dimension must be a non-zero power of two".into(),
            ));
        }
        validate_tau_domain(tau1, inputs.rows)?;

        let d_rows = static_plan.d_rows();
        let d_physical_cols = static_plan.d_physical_cols();
        let d_row_start = inputs.rows.checked_sub(d_rows).ok_or_else(|| {
            AkitaError::InvalidSetup("setup D rows exceed relation row count".into())
        })?;
        let a_projection = setup_role_projection(alpha, setup_ring_dim, role_dims.d_a(), "A")?;
        let b_projection = setup_role_projection(alpha, setup_ring_dim, role_dims.d_b(), "B")?;
        let d_projection = setup_role_projection(alpha, setup_ring_dim, role_dims.d_d(), "D")?;

        for group in groups {
            if fold_gadget.len() < group.depth_fold {
                return Err(AkitaError::InvalidSize {
                    expected: group.depth_fold,
                    actual: fold_gadget.len(),
                });
            }
        }

        let fold_gadget = fold_gadget
            .iter()
            .copied()
            .map(|fold| E::one().mul_base(fold))
            .collect::<Vec<_>>();
        let required = evaluator_required(
            d_rows,
            d_physical_cols,
            groups,
            a_projection.ratio,
            b_projection.ratio,
            d_projection.ratio,
        )?;

        Ok(Self {
            tau1: tau1.to_vec(),
            x_challenges: x_challenges.to_vec(),
            groups: groups.to_vec(),
            d_row_start,
            d_rows,
            d_physical_cols,
            a_projection,
            b_projection,
            d_projection,
            fold_gadget,
            required,
        })
    }

    /// Number of base setup positions covered by this evaluator.
    #[must_use]
    pub fn required(&self) -> usize {
        self.required
    }

    /// Whether the current verifier should prefer this evaluator over the
    /// packed setup-index path.
    ///
    /// The succinct formulas are already exact for tiled multi-chunk layouts,
    /// but the current per-chunk loop only wins consistently in the low-chunk
    /// regime. Multi-chunk callers can still use [`Self::evaluate`] directly
    /// for testing and benchmarking.
    #[must_use]
    pub fn prefers_succinct_path(&self) -> bool {
        self.groups.iter().all(|group| group.chunks.len() == 1)
    }

    /// Evaluate `setup_index_weight~(rho_setup_idx)`.
    ///
    /// Returns `Ok(None)` when the layout is valid but outside the succinct
    /// evaluator's current fast surface. Callers can then use the materialized
    /// packed path as a fallback.
    pub fn evaluate(&self, rho_setup_idx: &[E]) -> Result<Option<E>, AkitaError> {
        let setup_idx_bits = self.setup_idx_bits()?;
        if rho_setup_idx.len() != setup_idx_bits {
            return Err(AkitaError::InvalidSize {
                expected: setup_idx_bits,
                actual: rho_setup_idx.len(),
            });
        }

        let mut acc = E::zero();
        for group in &self.groups {
            let Some(d_value) = self.evaluate_d_role(group, rho_setup_idx)? else {
                return Ok(None);
            };
            let Some(b_value) = self.evaluate_b_role(group, rho_setup_idx)? else {
                return Ok(None);
            };
            let Some(a_value) = self.evaluate_a_role(group, rho_setup_idx)? else {
                return Ok(None);
            };
            acc += d_value + b_value + a_value;
        }
        Ok(Some(acc))
    }

    fn setup_idx_bits(&self) -> Result<usize, AkitaError> {
        let setup_idx_len = self
            .required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup-index weight length overflow".into()))?;
        Ok(setup_idx_len.trailing_zeros() as usize)
    }

    fn evaluate_d_role(
        &self,
        group: &SetupContributionGroupInputs,
        rho_setup_idx: &[E],
    ) -> Result<Option<E>, AkitaError> {
        if self.d_rows == 0 || self.d_physical_cols == 0 {
            return Ok(Some(E::zero()));
        }
        let e_cols = checked_mul3(
            group.num_claims,
            group.num_blocks,
            group.depth_open,
            "setup D active width overflow",
        )?;
        if group.e_col_offset != 0 || self.d_physical_cols != e_cols {
            return Ok(None);
        }
        self.evaluate_role(
            rho_setup_idx,
            &self.d_projection,
            self.d_rows,
            self.d_physical_cols,
            self.d_row_start,
            |col_point| self.evaluate_e_columns(group, col_point),
        )
    }

    fn evaluate_b_role(
        &self,
        group: &SetupContributionGroupInputs,
        rho_setup_idx: &[E],
    ) -> Result<Option<E>, AkitaError> {
        if group.n_b == 0 {
            return Ok(Some(E::zero()));
        }
        let t_cols = group
            .num_claims
            .checked_mul(group.t_cols_per_vector)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
        self.evaluate_role(
            rho_setup_idx,
            &self.b_projection,
            group.n_b,
            t_cols,
            group.b_row_start,
            |col_point| self.evaluate_t_columns(group, col_point),
        )
    }

    fn evaluate_a_role(
        &self,
        group: &SetupContributionGroupInputs,
        rho_setup_idx: &[E],
    ) -> Result<Option<E>, AkitaError> {
        if group.n_a == 0 {
            return Ok(Some(E::zero()));
        }
        let z_cols = group
            .block_len
            .checked_mul(group.depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A width overflow".into()))?;
        self.evaluate_role(
            rho_setup_idx,
            &self.a_projection,
            group.n_a,
            z_cols,
            group.a_row_start,
            |col_point| self.evaluate_z_columns(group, col_point),
        )
    }

    fn evaluate_role<FN>(
        &self,
        rho_setup_idx: &[E],
        projection: &SetupRoleProjection<E>,
        rows: usize,
        width: usize,
        row_start: usize,
        column_eval: FN,
    ) -> Result<Option<E>, AkitaError>
    where
        FN: FnOnce(&[E]) -> Result<Option<E>, AkitaError>,
    {
        if rows == 0 || width == 0 {
            return Ok(Some(E::zero()));
        }
        if !width.is_power_of_two() {
            return Ok(None);
        }
        let lane_bits = projection.ratio.trailing_zeros() as usize;
        let width_bits = width.trailing_zeros() as usize;
        let split_bits = lane_bits
            .checked_add(width_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("setup-index role arity overflow".into()))?;
        if split_bits > rho_setup_idx.len() {
            return Err(AkitaError::InvalidProof);
        }

        let lane_point = &rho_setup_idx[..lane_bits];
        let logical_point = &rho_setup_idx[lane_bits..];
        let col_point = &logical_point[..width_bits];
        let row_point = &logical_point[width_bits..];
        let Some(col_eval) = column_eval(col_point)? else {
            return Ok(None);
        };
        let lane_eval = projection.lane_factor(lane_point);
        let row_eval = self.row_factor(row_point, row_start, rows);
        Ok(Some(lane_eval * row_eval * col_eval))
    }

    fn row_factor(&self, row_point: &[E], row_start: usize, rows: usize) -> E {
        (0..rows)
            .map(|row| {
                eq_eval_at_index(row_point, row) * eq_eval_at_index(&self.tau1, row_start + row)
            })
            .sum()
    }

    fn evaluate_e_columns(
        &self,
        group: &SetupContributionGroupInputs,
        col_point: &[E],
    ) -> Result<Option<E>, AkitaError> {
        if !chunks_tile_blocks(group)? {
            return Ok(None);
        }
        let expected_width = checked_mul3(
            group.num_claims,
            group.num_blocks,
            group.depth_open,
            "setup D active width overflow",
        )?;
        if expected_width != self.d_physical_cols {
            return Ok(None);
        }

        let mut cursor = 0usize;
        let Some(digit_point) = take_axis_point(col_point, &mut cursor, group.depth_open)? else {
            return Ok(None);
        };
        let Some(block_point) = take_axis_point(col_point, &mut cursor, group.num_blocks)? else {
            return Ok(None);
        };
        let Some(claim_point) = take_axis_point(col_point, &mut cursor, group.num_claims)? else {
            return Ok(None);
        };
        if cursor != col_point.len() {
            return Ok(None);
        }

        let chunk_bits = group.blocks_per_chunk.trailing_zeros() as usize;
        if chunk_bits > block_point.len() || chunk_bits > self.x_challenges.len() {
            return Err(AkitaError::InvalidProof);
        }
        let block_low = &block_point[..chunk_bits];
        let block_high = &block_point[chunk_bits..];
        let x_low = &self.x_challenges[..chunk_bits];
        let x_high = &self.x_challenges[chunk_bits..];
        let low_mask = group.blocks_per_chunk - 1;
        let chunk_eq = eq_axis_table(block_high, group.chunks.len())?;
        let digit_eq = eq_axis_table(digit_point, group.depth_open)?;
        let claim_eq = eq_axis_table(claim_point, group.num_claims)?;

        let mut acc = E::zero();
        for (chunk_idx, chunk) in group.chunks.iter().enumerate() {
            let chunk_factor = chunk_eq[chunk_idx];
            let low = shifted_eq_carry_sums(block_low, x_low, chunk.offset_e & low_mask)?;
            let offset_high = chunk.offset_e >> chunk_bits;
            for (digit, &digit_factor) in digit_eq.iter().enumerate() {
                for (claim, &claim_factor) in claim_eq.iter().enumerate() {
                    let query_factor = chunk_factor * digit_factor * claim_factor;
                    for (carry, &low_factor) in low.iter().enumerate() {
                        let high_idx = digit
                            .checked_mul(group.num_claims)
                            .and_then(|idx| idx.checked_add(claim))
                            .and_then(|idx| idx.checked_add(carry))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("setup D high index overflow".into())
                            })?;
                        acc += query_factor
                            * low_factor
                            * eq_eval_at_index(x_high, offset_high + high_idx);
                    }
                }
            }
        }
        Ok(Some(acc))
    }

    fn evaluate_t_columns(
        &self,
        group: &SetupContributionGroupInputs,
        col_point: &[E],
    ) -> Result<Option<E>, AkitaError> {
        if !chunks_tile_blocks(group)? {
            return Ok(None);
        }
        let compound_per_block = group
            .n_a
            .checked_mul(group.depth_open)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B compound width overflow".into()))?;
        let expected_cols_per_vector = compound_per_block
            .checked_mul(group.num_blocks)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup B columns-per-vector overflow".into())
            })?;
        if group.t_cols_per_vector != expected_cols_per_vector {
            return Ok(None);
        }

        let mut cursor = 0usize;
        let Some(digit_point) = take_axis_point(col_point, &mut cursor, group.depth_open)? else {
            return Ok(None);
        };
        let Some(a_point) = take_axis_point(col_point, &mut cursor, group.n_a)? else {
            return Ok(None);
        };
        let Some(block_point) = take_axis_point(col_point, &mut cursor, group.num_blocks)? else {
            return Ok(None);
        };
        let Some(vector_point) = take_axis_point(col_point, &mut cursor, group.num_claims)? else {
            return Ok(None);
        };
        if cursor != col_point.len() {
            return Ok(None);
        }

        let chunk_bits = group.blocks_per_chunk.trailing_zeros() as usize;
        if chunk_bits > block_point.len() || chunk_bits > self.x_challenges.len() {
            return Err(AkitaError::InvalidProof);
        }
        let block_low = &block_point[..chunk_bits];
        let block_high = &block_point[chunk_bits..];
        let x_low = &self.x_challenges[..chunk_bits];
        let x_high = &self.x_challenges[chunk_bits..];
        let low_mask = group.blocks_per_chunk - 1;
        let chunk_eq = eq_axis_table(block_high, group.chunks.len())?;
        let digit_eq = eq_axis_table(digit_point, group.depth_open)?;
        let a_eq = eq_axis_table(a_point, group.n_a)?;
        let vector_eq = eq_axis_table(vector_point, group.num_claims)?;

        let mut acc = E::zero();
        for (chunk_idx, chunk) in group.chunks.iter().enumerate() {
            let chunk_factor = chunk_eq[chunk_idx];
            let low = shifted_eq_carry_sums(block_low, x_low, chunk.offset_t & low_mask)?;
            let offset_high = chunk.offset_t >> chunk_bits;
            for (digit, &digit_factor) in digit_eq.iter().enumerate() {
                for (a_idx, &a_factor) in a_eq.iter().enumerate() {
                    let compound = digit
                        .checked_add(group.depth_open.checked_mul(a_idx).ok_or_else(|| {
                            AkitaError::InvalidSetup("setup B compound index overflow".into())
                        })?)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("setup B compound index overflow".into())
                        })?;
                    for (vector, &vector_factor) in vector_eq.iter().enumerate() {
                        let query_factor = chunk_factor * digit_factor * a_factor * vector_factor;
                        for (carry, &low_factor) in low.iter().enumerate() {
                            let high_idx = compound
                                .checked_mul(group.num_claims)
                                .and_then(|idx| idx.checked_add(vector))
                                .and_then(|idx| idx.checked_add(carry))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup B high index overflow".into())
                                })?;
                            acc += query_factor
                                * low_factor
                                * eq_eval_at_index(x_high, offset_high + high_idx);
                        }
                    }
                }
            }
        }
        Ok(Some(acc))
    }

    fn evaluate_z_columns(
        &self,
        group: &SetupContributionGroupInputs,
        col_point: &[E],
    ) -> Result<Option<E>, AkitaError> {
        let mut cursor = 0usize;
        let Some(dc_point) = take_axis_point(col_point, &mut cursor, group.depth_commit)? else {
            return Ok(None);
        };
        let Some(block_point) = take_axis_point(col_point, &mut cursor, group.block_len)? else {
            return Ok(None);
        };
        if cursor != col_point.len() {
            return Ok(None);
        }

        let z_bits = group.block_len.trailing_zeros() as usize;
        if z_bits > self.x_challenges.len() {
            return Err(AkitaError::InvalidProof);
        }
        let x_low = &self.x_challenges[..z_bits];
        let x_high = &self.x_challenges[z_bits..];
        let low_mask = group.block_len - 1;
        let dc_eq = eq_axis_table(dc_point, group.depth_commit)?;

        let mut acc = E::zero();
        for chunk in &group.chunks {
            let low = shifted_eq_carry_sums(block_point, x_low, chunk.offset_z & low_mask)?;
            let offset_high = chunk.offset_z >> z_bits;
            for (dc, &dc_factor) in dc_eq.iter().enumerate() {
                for (df, &fold) in self.fold_gadget.iter().enumerate().take(group.depth_fold) {
                    for (carry, &low_factor) in low.iter().enumerate() {
                        let high_idx = df
                            .checked_add(group.depth_fold.checked_mul(dc).ok_or_else(|| {
                                AkitaError::InvalidSetup("setup A high index overflow".into())
                            })?)
                            .and_then(|idx| idx.checked_add(carry))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("setup A high index overflow".into())
                            })?;
                        acc -= dc_factor
                            * low_factor
                            * eq_eval_at_index(x_high, offset_high + high_idx)
                            * fold;
                    }
                }
            }
        }
        Ok(Some(acc))
    }
}

impl<E: FieldCore> SetupRoleProjection<E> {
    fn lane_factor(&self, lane_point: &[E]) -> E {
        self.scales
            .iter()
            .enumerate()
            .map(|(lane, &scale)| eq_eval_at_index(lane_point, lane) * scale)
            .sum()
    }
}

fn setup_role_projection<E: FieldCore>(
    alpha: E,
    setup_ring_dim: usize,
    role_dim: usize,
    role: &'static str,
) -> Result<SetupRoleProjection<E>, AkitaError> {
    if role_dim == 0 || !role_dim.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "{role} setup-index weight ring dimension must be a non-zero power of two"
        )));
    }
    if !role_dim.is_multiple_of(setup_ring_dim) {
        return Err(AkitaError::InvalidSetup(format!(
            "{role} setup-index weight ring dimension does not decompose over base setup ring"
        )));
    }
    let ratio = role_dim / setup_ring_dim;
    if ratio == 0 || !ratio.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "{role} setup-index weight projection ratio must be a non-zero power of two"
        )));
    }

    let role_pows = scalar_powers(alpha, role_dim);
    let base_pows = scalar_powers(alpha, setup_ring_dim);
    let mut scales = Vec::with_capacity(ratio);
    for lane in 0..ratio {
        let offset = lane * setup_ring_dim;
        let scale = role_pows[offset];
        for idx in 0..setup_ring_dim {
            if role_pows[offset + idx] != scale * base_pows[idx] {
                return Err(AkitaError::InvalidSetup(format!(
                    "{role} setup-index weight alpha powers do not decompose over base setup ring"
                )));
            }
        }
        scales.push(scale);
    }
    Ok(SetupRoleProjection { ratio, scales })
}

fn evaluator_required(
    d_rows: usize,
    d_physical_cols: usize,
    groups: &[SetupContributionGroupInputs],
    a_ratio: usize,
    b_ratio: usize,
    d_ratio: usize,
) -> Result<usize, AkitaError> {
    let mut required = d_rows
        .checked_mul(d_physical_cols)
        .and_then(|width| width.checked_mul(d_ratio))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D base footprint overflow".into()))?;
    for group in groups {
        let t_cols = group
            .num_claims
            .checked_mul(group.t_cols_per_vector)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
        let z_cols = group
            .block_len
            .checked_mul(group.depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A width overflow".into()))?;
        let b_required = group
            .n_b
            .checked_mul(t_cols)
            .and_then(|width| width.checked_mul(b_ratio))
            .ok_or_else(|| AkitaError::InvalidSetup("setup B base footprint overflow".into()))?;
        let a_required = group
            .n_a
            .checked_mul(z_cols)
            .and_then(|width| width.checked_mul(a_ratio))
            .ok_or_else(|| AkitaError::InvalidSetup("setup A base footprint overflow".into()))?;
        required = required.max(b_required).max(a_required);
    }
    Ok(required)
}

fn validate_tau_domain<E: FieldCore>(tau1: &[E], rows: usize) -> Result<(), AkitaError> {
    if tau1.len() < usize::BITS as usize && rows > (1usize << tau1.len()) {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: 1usize << tau1.len(),
        });
    }
    Ok(())
}

fn chunks_tile_blocks(group: &SetupContributionGroupInputs) -> Result<bool, AkitaError> {
    if group.chunks.is_empty()
        || group.blocks_per_chunk == 0
        || !group.blocks_per_chunk.is_power_of_two()
    {
        return Ok(false);
    }
    let covered_blocks = group
        .chunks
        .len()
        .checked_mul(group.blocks_per_chunk)
        .ok_or_else(|| AkitaError::InvalidSetup("setup chunk block coverage overflow".into()))?;
    if covered_blocks != group.num_blocks {
        return Ok(false);
    }
    Ok(group
        .chunks
        .iter()
        .enumerate()
        .all(|(idx, chunk)| chunk_is_contiguous(idx, group.blocks_per_chunk, chunk)))
}

fn chunk_is_contiguous(
    chunk_idx: usize,
    blocks_per_chunk: usize,
    chunk: &WitnessChunkLayout,
) -> bool {
    chunk.global_block_base == chunk_idx * blocks_per_chunk
}

fn take_axis_point<'a, E>(
    point: &'a [E],
    cursor: &mut usize,
    len: usize,
) -> Result<Option<&'a [E]>, AkitaError> {
    if len == 0 || !len.is_power_of_two() {
        return Ok(None);
    }
    let bits = len.trailing_zeros() as usize;
    let end = cursor
        .checked_add(bits)
        .ok_or_else(|| AkitaError::InvalidSetup("setup column arity overflow".into()))?;
    if end > point.len() {
        return Err(AkitaError::InvalidProof);
    }
    let axis = &point[*cursor..end];
    *cursor = end;
    Ok(Some(axis))
}

fn eq_axis_table<E: FieldCore>(point: &[E], len: usize) -> Result<Vec<E>, AkitaError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup-index weight axis length must be a non-zero power of two".into(),
        ));
    }
    let bits = len.trailing_zeros() as usize;
    if point.len() != bits {
        return Err(AkitaError::InvalidProof);
    }
    let table = EqPolynomial::evals(point)?;
    if table.len() < len {
        return Err(AkitaError::InvalidProof);
    }
    Ok(table)
}

fn shifted_eq_carry_sums<E: FieldCore>(
    r_low: &[E],
    x_low: &[E],
    offset_low: usize,
) -> Result<[E; POSSIBLE_CARRIES], AkitaError> {
    if r_low.len() != x_low.len() {
        return Err(AkitaError::InvalidSize {
            expected: r_low.len(),
            actual: x_low.len(),
        });
    }
    let bits = r_low.len();
    if bits >= usize::BITS as usize {
        return Err(AkitaError::InvalidSize {
            expected: usize::BITS as usize - 1,
            actual: bits,
        });
    }
    let low_len = 1usize << bits;
    if offset_low >= low_len {
        return Err(AkitaError::InvalidInput(
            "setup-index weight low offset exceeds peeled block".into(),
        ));
    }

    let mut state = [E::one(), E::zero()];
    for bit in 0..bits {
        let offset_bit = (offset_low >> bit) & 1;
        let r = r_low[bit];
        let x = x_low[bit];
        let mut next = [E::zero(), E::zero()];
        for (carry_in, &state_factor) in state.iter().enumerate() {
            for u_bit in 0..=1usize {
                let sum = u_bit + offset_bit + carry_in;
                let y_bit = sum & 1;
                let carry_out = sum >> 1;
                debug_assert!(carry_out < POSSIBLE_CARRIES);
                let r_factor = if u_bit == 1 { r } else { E::one() - r };
                let x_factor = if y_bit == 1 { x } else { E::one() - x };
                next[carry_out] += state_factor * r_factor * x_factor;
            }
        }
        state = next;
    }
    Ok(state)
}

fn checked_mul3(a: usize, b: usize, c: usize, context: &'static str) -> Result<usize, AkitaError> {
    a.checked_mul(b)
        .and_then(|ab| ab.checked_mul(c))
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}
