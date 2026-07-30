use super::*;
use akita_algebra::{offset_eq::eval_affine_digit_intervals, ring::scalar_powers_with_stride};

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Contract one group's structured E/T/Z terms from the same checked spans
    /// that define its setup contribution.
    pub fn evaluate_structured_group<F>(
        &self,
        group_id: usize,
        block_challenges: &[E],
        opening_a_evals: &[E],
        alpha: E,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let group = self
            .groups
            .iter()
            .find(|group| group.group_id == group_id)
            .ok_or(AkitaError::InvalidProof)?;
        let expected_blocks = group
            .num_claims
            .checked_mul(group.num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
        if block_challenges.len() != expected_blocks
            || opening_a_evals.len() != group.num_positions_per_block
        {
            return Err(AkitaError::InvalidProof);
        }
        if let Some(column_slices) = group.column_eq_slices() {
            return self.evaluate_structured_group_from_slices::<F>(
                group,
                column_slices,
                block_challenges,
                opening_a_evals,
                alpha,
            );
        }

        let opening_gadget = crate::gadget_row_scalars::<F>(group.depth_open, group.log_basis_open);
        let commitment_gadget =
            crate::gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
        let witness_gadget =
            crate::gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
        let witness_gadget_ext = witness_gadget
            .iter()
            .copied()
            .map(|weight| E::one().mul_base(weight))
            .collect::<Vec<_>>();
        let lane_powers = self.group_relation_lane_powers(group, alpha)?;
        let inner_lane_powers = &lane_powers[0];
        let (outer_subcolumns, opening_subcolumns) =
            SetupProjectionGeometry::a_carrier_subcolumn_counts(group.role_dims)?;

        if group.num_live_blocks == 0 {
            return Err(AkitaError::InvalidSetup(
                "structured setup group has no live blocks".into(),
            ));
        }
        let low_len = group
            .num_live_blocks
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("structured block domain overflow".into()))?;
        let claim_factors = block_challenges
            .chunks_exact(group.num_live_blocks)
            .map(|exact| -> Result<Vec<E>, AkitaError> {
                let mut padded = vec![E::zero(); low_len];
                padded
                    .get_mut(..exact.len())
                    .ok_or(AkitaError::InvalidProof)?
                    .copy_from_slice(exact);
                Ok(padded)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if claim_factors.len() != group.num_claims {
            return Err(AkitaError::InvalidProof);
        }

        let mut evaluation = E::zero();
        evaluation += cfg_fold_reduce!(
            0..group.d_spans.len(),
            || Ok(E::zero()),
            |acc: Result<E, AkitaError>, span_index| {
                let mut acc = acc?;
                let span = group
                    .d_spans
                    .get(span_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let digit = span.setup_column_start % group.depth_open;
                let semantic = span.setup_column_start / group.depth_open;
                let opening_subcolumn = semantic % opening_subcolumns;
                if opening_subcolumn != 0 {
                    return Ok(acc);
                }
                ensure_projection_partition(span, opening_subcolumns, inner_lane_powers)?;
                let block_claim = semantic / opening_subcolumns;
                let claim = block_claim / group.num_live_blocks;
                let block_start = block_claim % group.num_live_blocks;
                let interval = evaluate_single_factor_row(
                    self.relation_address.point(),
                    self.relation_address.equality_window(),
                    span.relation_lane_start,
                    block_start,
                    span,
                    inner_lane_powers,
                    claim_factors.get(claim).ok_or(AkitaError::InvalidProof)?,
                )?;
                acc += group.consistency_weight
                    * interval
                        .mul_base(*opening_gadget.get(digit).ok_or(AkitaError::InvalidProof)?);
                Ok(acc)
            },
            |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
        )?;
        evaluation += cfg_fold_reduce!(
            0..group.b_spans.len(),
            || Ok(E::zero()),
            |acc: Result<E, AkitaError>, span_index| {
                let mut acc = acc?;
                let span = group
                    .b_spans
                    .get(span_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let outer_subcolumn = span.setup_column_start % outer_subcolumns;
                if outer_subcolumn != 0 {
                    return Ok(acc);
                }
                ensure_projection_partition(span, outer_subcolumns, inner_lane_powers)?;
                let semantic = span.setup_column_start / outer_subcolumns;
                let digit = semantic % group.depth_commit;
                let row_and_block = semantic / group.depth_commit;
                let a_row = row_and_block % group.n_a;
                let block_claim = row_and_block / group.n_a;
                let claim = block_claim / group.num_live_blocks;
                let block_start = block_claim % group.num_live_blocks;
                let interval = evaluate_single_factor_row(
                    self.relation_address.point(),
                    self.relation_address.equality_window(),
                    span.relation_lane_start,
                    block_start,
                    span,
                    inner_lane_powers,
                    claim_factors.get(claim).ok_or(AkitaError::InvalidProof)?,
                )?;
                acc += *group
                    .a_row_weights
                    .get(a_row)
                    .ok_or(AkitaError::InvalidProof)?
                    * interval.mul_base(
                        *commitment_gadget
                            .get(digit)
                            .ok_or(AkitaError::InvalidProof)?,
                    );
                Ok(acc)
            },
            |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
        )?;
        // Each canonical A span is one coarse fold family for a physical
        // witness unit. Projection lanes and fold-gadget powers remain
        // independent digit weights inside the shared affine traversal.
        if group.fold_gadget.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "setup A fold family has no gadget digits".into(),
            ));
        }
        let dense_a_families = witness_gadget.len().is_power_of_two()
            && group
                .a_families
                .iter()
                .all(|family| family.setup_column_start == 0 && family.setup_column_stride == 1);
        if dense_a_families {
            evaluation -= evaluate_dense_a_families(
                self.relation_address.point(),
                &group.a_families,
                inner_lane_powers,
                opening_a_evals,
                &witness_gadget_ext,
                &group.fold_gadget,
            )? * group.consistency_weight;
        } else {
            evaluation += cfg_fold_reduce!(
                0..group.a_families.len(),
                || Ok(E::zero()),
                |acc: Result<E, AkitaError>, family_index| {
                    let mut acc = acc?;
                    let family = group
                        .a_families
                        .get(family_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    ensure_a_family(family, inner_lane_powers, &group.fold_gadget)?;
                    for occurrence in family.occurrences() {
                        let (setup_column, relation_lane_start) = occurrence?;
                        let position = setup_column / group.depth_witness;
                        let witness_digit = setup_column % group.depth_witness;
                        let mut lane_equality = E::zero();
                        for (fold_digit, &fold) in group.fold_gadget.iter().enumerate() {
                            let fold_start = fold_digit
                                .checked_mul(family.fold_lane_stride)
                                .and_then(|offset| relation_lane_start.checked_add(offset))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup A fold lane overflow".into())
                                })?;
                            lane_equality += evaluate_lane_segment(
                                self.relation_address.equality_window(),
                                fold_start,
                                inner_lane_powers,
                            )? * fold;
                        }
                        acc -= lane_equality
                            * group.consistency_weight
                            * *opening_a_evals
                                .get(position)
                                .ok_or(AkitaError::InvalidProof)?
                            * E::one().mul_base(
                                *witness_gadget
                                    .get(witness_digit)
                                    .ok_or(AkitaError::InvalidProof)?,
                            );
                    }
                    Ok(acc)
                },
                |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
            )?;
        }
        Ok(evaluation)
    }

    fn evaluate_structured_group_from_slices<F>(
        &self,
        group: &SetupContributionGroupPlan<E>,
        (e_eq_slice, t_eq_slice, z_eq_slice): (&[E], &[E], &[E]),
        block_challenges: &[E],
        opening_a_evals: &[E],
        alpha: E,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let (outer_subcolumns, opening_subcolumns) =
            SetupProjectionGeometry::a_carrier_subcolumn_counts(group.role_dims)?;
        let block_claims = group
            .num_claims
            .checked_mul(group.num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
        let e_stride = opening_subcolumns
            .checked_mul(group.depth_open)
            .ok_or_else(|| AkitaError::InvalidSetup("structured E stride overflow".into()))?;
        let t_stride = group
            .n_a
            .checked_mul(group.depth_commit)
            .and_then(|stride| stride.checked_mul(outer_subcolumns))
            .ok_or_else(|| AkitaError::InvalidSetup("structured T stride overflow".into()))?;
        let expected_e = block_claims
            .checked_mul(e_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("structured E width overflow".into()))?;
        let expected_t = block_claims
            .checked_mul(t_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("structured T width overflow".into()))?;
        let expected_z = group
            .num_positions_per_block
            .checked_mul(group.depth_witness)
            .ok_or_else(|| AkitaError::InvalidSetup("structured Z width overflow".into()))?;
        if e_eq_slice.len() != expected_e
            || t_eq_slice.len() != expected_t
            || z_eq_slice.len() != expected_z
        {
            return Err(AkitaError::InvalidProof);
        }

        let opening_gadget = crate::gadget_row_scalars::<F>(group.depth_open, group.log_basis_open);
        let commitment_gadget =
            crate::gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
        let witness_gadget =
            crate::gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
        let opening_gadget = opening_gadget
            .into_iter()
            .map(|weight| E::one().mul_base(weight))
            .collect::<Vec<_>>();
        let commitment_gadget = commitment_gadget
            .into_iter()
            .map(|weight| E::one().mul_base(weight))
            .collect::<Vec<_>>();
        let witness_gadget = witness_gadget
            .into_iter()
            .map(|weight| E::one().mul_base(weight))
            .collect::<Vec<_>>();
        let opening_scales =
            scalar_powers_with_stride(alpha, group.role_dims.d_d(), opening_subcolumns)?;
        let outer_scales =
            scalar_powers_with_stride(alpha, group.role_dims.d_b(), outer_subcolumns)?;
        if opening_scales.len() != opening_subcolumns || outer_scales.len() != outer_subcolumns {
            return Err(AkitaError::InvalidSetup(
                "structured setup projection scale count mismatch".into(),
            ));
        }
        let mut t_weights = Vec::new();
        t_weights.try_reserve_exact(t_stride).map_err(|_| {
            AkitaError::InvalidSetup("structured T weight allocation failed".into())
        })?;
        for &row_weight in group.a_row_weights.iter() {
            for &gadget in &commitment_gadget {
                for &scale in &outer_scales {
                    t_weights.push(row_weight * gadget * scale);
                }
            }
        }
        if t_weights.len() != t_stride {
            return Err(AkitaError::InvalidSetup(
                "structured T contraction weight count mismatch".into(),
            ));
        }

        let mut evaluation = cfg_fold_reduce!(
            0..block_claims,
            || Ok(E::zero()),
            |acc: Result<E, AkitaError>, block_claim| {
                let mut acc = acc?;
                let challenge = *block_challenges
                    .get(block_claim)
                    .ok_or(AkitaError::InvalidProof)?;
                let e_start = block_claim
                    .checked_mul(e_stride)
                    .ok_or(AkitaError::InvalidProof)?;
                let t_start = block_claim
                    .checked_mul(t_stride)
                    .ok_or(AkitaError::InvalidProof)?;
                let mut e_weight = E::zero();
                for (subcolumn, &scale) in opening_scales.iter().enumerate() {
                    for (digit, &gadget) in opening_gadget.iter().enumerate() {
                        let offset = subcolumn
                            .checked_mul(group.depth_open)
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|offset| e_start.checked_add(offset))
                            .ok_or(AkitaError::InvalidProof)?;
                        e_weight += *e_eq_slice.get(offset).ok_or(AkitaError::InvalidProof)?
                            * scale
                            * gadget;
                    }
                }
                acc += challenge * group.consistency_weight * e_weight;

                let t_end = t_start
                    .checked_add(t_stride)
                    .ok_or(AkitaError::InvalidProof)?;
                let t_block = t_eq_slice
                    .get(t_start..t_end)
                    .ok_or(AkitaError::InvalidProof)?;
                let t_weight = t_block
                    .iter()
                    .zip(&t_weights)
                    .map(|(&equality, &weight)| equality * weight)
                    .sum::<E>();
                Ok(acc + challenge * t_weight)
            },
            |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
        )?;

        for (position, &opening_a) in opening_a_evals.iter().enumerate() {
            for (digit, &gadget) in witness_gadget.iter().enumerate() {
                let column = position
                    .checked_mul(group.depth_witness)
                    .and_then(|base| base.checked_add(digit))
                    .ok_or(AkitaError::InvalidProof)?;
                evaluation += *z_eq_slice.get(column).ok_or(AkitaError::InvalidProof)?
                    * group.consistency_weight
                    * opening_a
                    * gadget;
            }
        }
        Ok(evaluation)
    }

    pub(super) fn group_relation_lane_powers(
        &self,
        group: &SetupContributionGroupPlan<E>,
        alpha: E,
    ) -> Result<[Vec<E>; 3], AkitaError> {
        let common = self
            .relation_address_geometry
            .relation_coefficient_block_len();
        Ok([
            relation_lane_powers(alpha, group.role_dims.d_a(), common)?,
            relation_lane_powers(alpha, group.role_dims.d_b(), common)?,
            relation_lane_powers(alpha, group.role_dims.d_d(), common)?,
        ])
    }
}

/// Contract compatible dense A fold families through one shared affine
/// recurrence.
fn evaluate_dense_a_families<E: FieldCore>(
    address_point: &[E],
    spans: &[SetupContributionSpan],
    lane_powers: &[E],
    opening_a_evals: &[E],
    witness_gadget: &[E],
    fold_gadget: &[E],
) -> Result<E, AkitaError> {
    let Some(first) = spans.first() else {
        return Ok(E::zero());
    };
    if !witness_gadget.len().is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup A span families are not dense affine rectangles".into(),
        ));
    }
    for span in spans {
        ensure_a_family(span, lane_powers, fold_gadget)?;
        if span.setup_column_start != 0
            || span.setup_column_stride != 1
            || span.occurrence_count != first.occurrence_count
            || span.relation_lane_stride != first.relation_lane_stride
            || span.fold_lane_stride != first.fold_lane_stride
        {
            return Err(AkitaError::InvalidSetup(
                "setup A span families do not share one affine geometry".into(),
            ));
        }
    }
    let digit_count = first
        .fold_count
        .checked_sub(1)
        .and_then(|last| last.checked_mul(first.fold_lane_stride))
        .and_then(|offset| offset.checked_add(first.relation_lane_count))
        .ok_or_else(|| AkitaError::InvalidSetup("setup A digit width overflow".into()))?;
    let mut digit_weights = vec![E::zero(); digit_count];
    for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
        for (lane, &lane_power) in lane_powers.iter().enumerate() {
            let digit = fold_digit
                .checked_mul(first.fold_lane_stride)
                .and_then(|offset| offset.checked_add(lane))
                .ok_or_else(|| AkitaError::InvalidSetup("setup A digit width overflow".into()))?;
            let weight = digit_weights
                .get_mut(digit)
                .ok_or(AkitaError::InvalidProof)?;
            *weight += fold * lane_power;
        }
    }
    let base_offsets = spans
        .iter()
        .map(|span| span.relation_lane_start)
        .collect::<Vec<_>>();
    eval_affine_digit_intervals(
        address_point,
        &base_offsets,
        0,
        first.occurrence_count,
        first.relation_lane_stride,
        &digit_weights,
        opening_a_evals,
        witness_gadget,
    )
}

fn ensure_a_family<E: FieldCore>(
    span: &SetupContributionSpan,
    lane_powers: &[E],
    fold_gadget: &[E],
) -> Result<(), AkitaError> {
    if span.relation_lane_count != lane_powers.len()
        || span.fold_count != fold_gadget.len()
        || span.fold_lane_stride < span.relation_lane_count
    {
        return Err(AkitaError::InvalidSetup(
            "setup A span is not one coarse fold family".into(),
        ));
    }
    Ok(())
}

// A span contained in one low-factor row has no high rows to summarize. Reuse
// the plan's equality tables; longer spans retain the compact carry recurrence.
fn evaluate_single_factor_row<E: FieldCore>(
    address_point: &[E],
    equality_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    relation_lane_start: usize,
    outer_start: usize,
    span: &SetupContributionSpan,
    digit_weights: &[E],
    low_weights: &[E],
) -> Result<E, AkitaError> {
    let outer_end = outer_start
        .checked_add(span.occurrence_count)
        .ok_or_else(|| AkitaError::InvalidSetup("structured outer window overflow".into()))?;
    if outer_end > low_weights.len() {
        return eval_affine_digit_intervals(
            address_point,
            &[relation_lane_start],
            outer_start,
            span.occurrence_count,
            span.relation_lane_stride,
            digit_weights,
            &[E::one()],
            low_weights,
        );
    }

    (0..span.occurrence_count).try_fold(E::zero(), |sum, occurrence| {
        let address = span
            .relation_lane_stride
            .checked_mul(occurrence)
            .and_then(|offset| relation_lane_start.checked_add(offset))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("structured relation address overflow".into())
            })?;
        let factor = *low_weights
            .get(outer_start + occurrence)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(sum + evaluate_lane_segment(equality_window, address, digit_weights)? * factor)
    })
}

fn relation_lane_powers<E: FieldCore>(
    alpha: E,
    role_dimension: usize,
    common_coeff_count: usize,
) -> Result<Vec<E>, AkitaError> {
    let lane_count = role_dimension
        .checked_div(common_coeff_count)
        .filter(|count| *count != 0 && role_dimension.is_multiple_of(common_coeff_count))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "relation role dimension does not decompose over the common coefficient count"
                    .into(),
            )
        })?;
    scalar_powers_with_stride(alpha, common_coeff_count, lane_count)
}

fn ensure_projection_partition<E: FieldCore>(
    span: &SetupContributionSpan,
    subcolumns: usize,
    inner_lane_powers: &[E],
) -> Result<(), AkitaError> {
    if inner_lane_powers.is_empty()
        || span
            .relation_lane_count
            .checked_mul(subcolumns)
            .filter(|&count| count == inner_lane_powers.len())
            .is_none()
        || !span
            .relation_lane_start
            .is_multiple_of(inner_lane_powers.len())
    {
        return Err(AkitaError::InvalidSetup(
            "setup projection spans do not cover one inner relation lane block".into(),
        ));
    }
    Ok(())
}

pub(super) fn ensure_lane_count<E: FieldCore>(
    span: &SetupContributionSpan,
    lane_powers: &[E],
) -> Result<(), AkitaError> {
    if span.relation_lane_count != lane_powers.len() {
        return Err(AkitaError::InvalidSetup(
            "setup contribution span disagrees with relation lane geometry".into(),
        ));
    }
    Ok(())
}

fn evaluate_lane_segment<E: FieldCore>(
    equality_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    lane_start: usize,
    lane_powers: &[E],
) -> Result<E, AkitaError> {
    lane_powers
        .iter()
        .copied()
        .enumerate()
        .try_fold(E::zero(), |sum, (lane, power)| {
            let address = lane_start
                .checked_add(lane)
                .ok_or_else(|| AkitaError::InvalidSetup("relation lane address overflow".into()))?;
            Ok(sum + equality_window.eval(address) * power)
        })
}
