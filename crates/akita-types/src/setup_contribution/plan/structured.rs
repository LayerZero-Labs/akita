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
        let high = [E::one()];
        for span in &group.d_spans {
            let digit = span.setup_column_start % group.depth_open;
            let semantic = span.setup_column_start / group.depth_open;
            let opening_subcolumn = semantic % opening_subcolumns;
            if opening_subcolumn != 0 {
                continue;
            }
            ensure_projection_partition(span, opening_subcolumns, inner_lane_powers)?;
            let block_claim = semantic / opening_subcolumns;
            let claim = block_claim / group.num_live_blocks;
            let block_start = block_claim % group.num_live_blocks;
            let interval = eval_affine_digit_interval(
                &self.address_point,
                span.relation_lane_start,
                block_start,
                span.occurrence_count,
                span.relation_lane_stride,
                inner_lane_powers,
                &high,
                claim_factors.get(claim).ok_or(AkitaError::InvalidProof)?,
            )?;
            evaluation += group.consistency_weight
                * interval.mul_base(*opening_gadget.get(digit).ok_or(AkitaError::InvalidProof)?);
        }

        for span in &group.b_spans {
            let outer_subcolumn = span.setup_column_start % outer_subcolumns;
            if outer_subcolumn != 0 {
                continue;
            }
            ensure_projection_partition(span, outer_subcolumns, inner_lane_powers)?;
            let semantic = span.setup_column_start / outer_subcolumns;
            let digit = semantic % group.depth_commit;
            let row_and_block = semantic / group.depth_commit;
            let a_row = row_and_block % group.n_a;
            let block_claim = row_and_block / group.n_a;
            let claim = block_claim / group.num_live_blocks;
            let block_start = block_claim % group.num_live_blocks;
            let interval = eval_affine_digit_interval(
                &self.address_point,
                span.relation_lane_start,
                block_start,
                span.occurrence_count,
                span.relation_lane_stride,
                inner_lane_powers,
                &high,
                claim_factors.get(claim).ok_or(AkitaError::InvalidProof)?,
            )?;
            evaluation += *group
                .a_row_weights
                .get(a_row)
                .ok_or(AkitaError::InvalidProof)?
                * interval.mul_base(
                    *commitment_gadget
                        .get(digit)
                        .ok_or(AkitaError::InvalidProof)?,
                );
        }

        for span in &group.a_spans {
            ensure_lane_count(span, inner_lane_powers)?;
            let fold = *group
                .fold_gadget
                .get(span.fold_digit.ok_or(AkitaError::InvalidProof)?)
                .ok_or(AkitaError::InvalidProof)?;
            for occurrence in span.occurrences() {
                let (setup_column, relation_lane_start) = occurrence?;
                let position = setup_column / group.depth_witness;
                let witness_digit = setup_column % group.depth_witness;
                let lane_equality =
                    evaluate_lane_segment(&self.eq_window, relation_lane_start, inner_lane_powers)?;
                evaluation -= lane_equality
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
        Ok(evaluation)
    }

    pub(super) fn group_relation_lane_powers(
        &self,
        group: &SetupContributionGroupPlan<E>,
        alpha: E,
    ) -> [Vec<E>; 3] {
        let common = self
            .relation_address_geometry
            .common_relation_witness_coeff_count();
        [
            relation_lane_powers(alpha, group.role_dims.d_a(), common),
            relation_lane_powers(alpha, group.role_dims.d_b(), common),
            relation_lane_powers(alpha, group.role_dims.d_d(), common),
        ]
    }
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
