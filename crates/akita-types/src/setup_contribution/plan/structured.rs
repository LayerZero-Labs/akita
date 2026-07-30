use super::*;
use akita_algebra::{offset_eq::eval_affine_digit_interval, ring::scalar_powers};

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
        let lane_powers = self.group_relation_lane_powers(group, alpha);
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
                    &self.address_point,
                    &self.eq_window,
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
                    &self.address_point,
                    &self.eq_window,
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
        // The canonical A spans are emitted as one complete fold-digit family
        // per witness unit. Contract a dense family in one affine pass so all
        // fold digits share the high-row traversal.
        let a_family_width = group.fold_gadget.len();
        if a_family_width == 0 || !group.a_spans.len().is_multiple_of(a_family_width) {
            return Err(AkitaError::InvalidSetup(
                "setup A spans do not form complete fold families".into(),
            ));
        }
        let a_family_count = group.a_spans.len() / a_family_width;
        evaluation += cfg_fold_reduce!(
            0..a_family_count,
            || Ok(E::zero()),
            |acc: Result<E, AkitaError>, family_index| {
                let mut acc = acc?;
                let family_start = family_index.checked_mul(a_family_width).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup A span family overflow".into())
                })?;
                let family_end = family_start.checked_add(a_family_width).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup A span family overflow".into())
                })?;
                let family = group
                    .a_spans
                    .get(family_start..family_end)
                    .ok_or(AkitaError::InvalidProof)?;
                let dense = witness_gadget.len().is_power_of_two()
                    && family
                        .iter()
                        .all(|span| span.setup_column_start == 0 && span.setup_column_stride == 1);
                if dense {
                    acc -= evaluate_dense_a_family(
                        &self.address_point,
                        family,
                        inner_lane_powers,
                        opening_a_evals,
                        &witness_gadget_ext,
                        &group.fold_gadget,
                    )? * group.consistency_weight;
                } else {
                    for span in family {
                        ensure_lane_count(span, inner_lane_powers)?;
                        let fold = *group
                            .fold_gadget
                            .get(span.fold_digit.ok_or(AkitaError::InvalidProof)?)
                            .ok_or(AkitaError::InvalidProof)?;
                        for occurrence in span.occurrences() {
                            let (setup_column, relation_lane_start) = occurrence?;
                            let position = setup_column / group.depth_witness;
                            let witness_digit = setup_column % group.depth_witness;
                            let lane_equality = evaluate_lane_segment(
                                &self.eq_window,
                                relation_lane_start,
                                inner_lane_powers,
                            )?;
                            acc -= lane_equality
                                * group.consistency_weight
                                * *opening_a_evals
                                    .get(position)
                                    .ok_or(AkitaError::InvalidProof)?
                                * fold
                                * E::one().mul_base(
                                    *witness_gadget
                                        .get(witness_digit)
                                        .ok_or(AkitaError::InvalidProof)?,
                                );
                        }
                    }
                }
                Ok(acc)
            },
            |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
        )?;
        Ok(evaluation)
    }

    pub(super) fn group_relation_lane_powers(
        &self,
        group: &SetupContributionGroupPlan<E>,
        alpha: E,
    ) -> [Vec<E>; 3] {
        let common = self
            .relation_address_geometry
            .relation_coefficient_block_len();
        [
            relation_lane_powers(alpha, group.role_dims.d_a(), common),
            relation_lane_powers(alpha, group.role_dims.d_b(), common),
            relation_lane_powers(alpha, group.role_dims.d_d(), common),
        ]
    }
}

/// Contract one dense canonical A fold family as a single affine rectangle.
fn evaluate_dense_a_family<E: FieldCore>(
    address_point: &[E],
    spans: &[SetupContributionSpan],
    lane_powers: &[E],
    opening_a_evals: &[E],
    witness_gadget: &[E],
    fold_gadget: &[E],
) -> Result<E, AkitaError> {
    let first = spans
        .first()
        .ok_or_else(|| AkitaError::InvalidSetup("setup A span family must be non-empty".into()))?;
    if !witness_gadget.len().is_power_of_two()
        || first.setup_column_start != 0
        || first.setup_column_stride != 1
    {
        return Err(AkitaError::InvalidSetup(
            "setup A span family is not a dense affine rectangle".into(),
        ));
    }

    let mut digit_count = 0usize;
    for span in spans {
        ensure_lane_count(span, lane_powers)?;
        if span.setup_column_start != first.setup_column_start
            || span.setup_column_stride != first.setup_column_stride
            || span.relation_lane_stride != first.relation_lane_stride
            || span.occurrence_count != first.occurrence_count
        {
            return Err(AkitaError::InvalidSetup(
                "setup A span family has inconsistent affine geometry".into(),
            ));
        }
        let lane_offset = span
            .relation_lane_start
            .checked_sub(first.relation_lane_start)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup A span family is not address ordered".into())
            })?;
        digit_count = digit_count.max(
            lane_offset
                .checked_add(lane_powers.len())
                .ok_or_else(|| AkitaError::InvalidSetup("setup A digit width overflow".into()))?,
        );
    }
    if digit_count > first.relation_lane_stride {
        return Err(AkitaError::InvalidSetup(
            "setup A digit family exceeds its affine stride".into(),
        ));
    }

    let mut digit_weights = vec![E::zero(); digit_count];
    for span in spans {
        let fold = *fold_gadget
            .get(span.fold_digit.ok_or(AkitaError::InvalidProof)?)
            .ok_or(AkitaError::InvalidProof)?;
        let lane_offset = span
            .relation_lane_start
            .checked_sub(first.relation_lane_start)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup A span family is not address ordered".into())
            })?;
        for (lane, &lane_power) in lane_powers.iter().enumerate() {
            let digit = lane_offset
                .checked_add(lane)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A digit width overflow".into()))?;
            let weight = digit_weights
                .get_mut(digit)
                .ok_or(AkitaError::InvalidProof)?;
            *weight += fold * lane_power;
        }
    }
    eval_affine_digit_interval(
        address_point,
        first.relation_lane_start,
        0,
        first.occurrence_count,
        first.relation_lane_stride,
        &digit_weights,
        opening_a_evals,
        witness_gadget,
    )
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
        return eval_affine_digit_interval(
            address_point,
            relation_lane_start,
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
) -> Vec<E> {
    scalar_powers(alpha, role_dimension)
        .into_iter()
        .step_by(common_coeff_count)
        .collect()
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
