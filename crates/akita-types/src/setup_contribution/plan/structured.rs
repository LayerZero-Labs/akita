use super::*;
use akita_algebra::{offset_eq::eval_affine_digit_interval, ring::scalar_powers};

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Contract one group's structured E/T/Z relation terms from its canonical
    /// setup-contribution spans.
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

        let consistency_weight = self
            .eq_tau1
            .first()
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let opening_gadget = crate::gadget_row_scalars::<F>(group.depth_open, group.log_basis_open);
        let commitment_gadget =
            crate::gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
        let witness_gadget =
            crate::gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
        let inner_lane_powers =
            relation_lane_powers(alpha, self.common_coeff_count, self.inner_lane_count)?;
        let (outer_subcolumns, opening_subcolumns) =
            SetupProjectionGeometry::witness_subcolumn_ratios(
                self.projection_geometry.role_dims(),
            )?;

        let low_len = group.num_live_blocks.next_power_of_two();
        let mut claim_factors = Vec::with_capacity(group.num_claims);
        for claim in 0..group.num_claims {
            let start = claim
                .checked_mul(group.num_live_blocks)
                .ok_or(AkitaError::InvalidProof)?;
            let end = start
                .checked_add(group.num_live_blocks)
                .ok_or(AkitaError::InvalidProof)?;
            let exact = block_challenges
                .get(start..end)
                .ok_or(AkitaError::InvalidProof)?;
            let mut padded = vec![E::zero(); low_len];
            padded
                .get_mut(..exact.len())
                .ok_or(AkitaError::InvalidProof)?
                .copy_from_slice(exact);
            claim_factors.push(padded);
        }

        let mut evaluation = E::zero();
        let high = [E::one()];
        for span in &group.d_spans {
            let native = span.setup_start / group.depth_open;
            let digit = span.setup_start % group.depth_open;
            let subcolumn = native % opening_subcolumns;
            let block_claim = native / opening_subcolumns;
            let claim = block_claim / group.num_live_blocks;
            let block_start = block_claim % group.num_live_blocks;
            let factors = claim_factors.get(claim).ok_or(AkitaError::InvalidProof)?;
            let lane_start = subcolumn
                .checked_mul(self.opening_lane_count)
                .ok_or(AkitaError::InvalidProof)?;
            let lane_end = lane_start
                .checked_add(self.opening_lane_count)
                .ok_or(AkitaError::InvalidProof)?;
            let lane_powers = inner_lane_powers
                .get(lane_start..lane_end)
                .ok_or(AkitaError::InvalidProof)?;
            let interval = eval_affine_digit_interval(
                &self.x_challenges,
                span.witness_start,
                block_start,
                span.len,
                span.witness_stride,
                lane_powers,
                &high,
                factors,
            )?;
            evaluation += consistency_weight
                * interval.mul_base(*opening_gadget.get(digit).ok_or(AkitaError::InvalidProof)?);
        }

        for span in &group.b_spans {
            let subcolumn = span.setup_start % outer_subcolumns;
            let semantic = span.setup_start / outer_subcolumns;
            let digit = semantic % group.depth_commit;
            let row_and_block = semantic / group.depth_commit;
            let a_row = row_and_block % group.n_a;
            let block_claim = row_and_block / group.n_a;
            let claim = block_claim / group.num_live_blocks;
            let block_start = block_claim % group.num_live_blocks;
            let factors = claim_factors.get(claim).ok_or(AkitaError::InvalidProof)?;
            let row_weight = *self
                .eq_tau1
                .get(group.a_row_start + a_row)
                .ok_or(AkitaError::InvalidProof)?;
            let lane_start = subcolumn
                .checked_mul(self.outer_lane_count)
                .ok_or(AkitaError::InvalidProof)?;
            let lane_end = lane_start
                .checked_add(self.outer_lane_count)
                .ok_or(AkitaError::InvalidProof)?;
            let lane_powers = inner_lane_powers
                .get(lane_start..lane_end)
                .ok_or(AkitaError::InvalidProof)?;
            let interval = eval_affine_digit_interval(
                &self.x_challenges,
                span.witness_start,
                block_start,
                span.len,
                span.witness_stride,
                lane_powers,
                &high,
                factors,
            )?;
            evaluation += row_weight
                * interval.mul_base(
                    *commitment_gadget
                        .get(digit)
                        .ok_or(AkitaError::InvalidProof)?,
                );
        }

        for span in &group.a_spans {
            let fold_digit = span.fold_digit.ok_or(AkitaError::InvalidProof)?;
            let fold = *self
                .fold_gadget
                .get(fold_digit)
                .ok_or(AkitaError::InvalidProof)?;
            for offset in 0..span.len {
                let address = span
                    .witness_start
                    .checked_add(
                        offset
                            .checked_mul(span.witness_stride)
                            .ok_or(AkitaError::InvalidProof)?,
                    )
                    .ok_or(AkitaError::InvalidProof)?;
                let lane_equality =
                    evaluate_lane_segment(&self.eq_window, address, &inner_lane_powers)?;
                let position = offset / group.depth_witness;
                let witness_digit = offset % group.depth_witness;
                let opening = *opening_a_evals
                    .get(position)
                    .ok_or(AkitaError::InvalidProof)?;
                evaluation -= lane_equality
                    * consistency_weight
                    * opening
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
}

pub(super) fn relation_lane_powers<E: FieldCore>(
    alpha: E,
    common_coeff_count: usize,
    lane_count: usize,
) -> Result<Vec<E>, AkitaError> {
    let role_dim = common_coeff_count
        .checked_mul(lane_count)
        .ok_or_else(|| AkitaError::InvalidSetup("relation role dimension overflow".into()))?;
    Ok(scalar_powers(alpha, role_dim)
        .into_iter()
        .step_by(common_coeff_count)
        .collect())
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
