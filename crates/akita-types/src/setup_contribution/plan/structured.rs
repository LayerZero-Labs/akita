use super::*;
use akita_algebra::ring::scalar_powers_with_stride;

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Contract one group's structured E/T/Z terms through its canonical
    /// relation-column tensors.
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
        let block_claims = group
            .num_claims
            .checked_mul(group.num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
        if block_challenges.len() != block_claims
            || opening_a_evals.len() != group.num_positions_per_block
        {
            return Err(AkitaError::InvalidProof);
        }

        let opening_gadget = extension_gadget::<F, E>(group.depth_open, group.log_basis_open);
        let commitment_gadget = extension_gadget::<F, E>(group.depth_commit, group.log_basis_outer);
        let witness_gadget = extension_gadget::<F, E>(group.depth_witness, group.log_basis_inner);
        let (outer_subcolumns, opening_subcolumns) =
            SetupProjectionGeometry::a_carrier_subcolumn_counts(group.role_dims)?;
        let opening_scales = (opening_subcolumns != 1)
            .then(|| scalar_powers_with_stride(alpha, group.role_dims.d_d(), opening_subcolumns))
            .transpose()?;
        let outer_scales = (outer_subcolumns != 1)
            .then(|| scalar_powers_with_stride(alpha, group.role_dims.d_b(), outer_subcolumns))
            .transpose()?;

        let e_stride = checked_product(
            opening_subcolumns,
            group.depth_open,
            "structured E stride overflow",
        )?;
        let t_stride = group
            .n_a
            .checked_mul(group.depth_commit)
            .and_then(|stride| stride.checked_mul(outer_subcolumns))
            .ok_or_else(|| AkitaError::InvalidSetup("structured T stride overflow".into()))?;
        let mut e_weights = vec![
            E::zero();
            block_claims.checked_mul(e_stride).ok_or_else(|| {
                AkitaError::InvalidSetup("structured E width overflow".into())
            })?
        ];
        let mut t_weights = vec![
            E::zero();
            block_claims.checked_mul(t_stride).ok_or_else(|| {
                AkitaError::InvalidSetup("structured T width overflow".into())
            })?
        ];
        for (block_claim, &block_challenge) in block_challenges.iter().enumerate() {
            let e_start = block_claim
                .checked_mul(e_stride)
                .ok_or(AkitaError::InvalidProof)?;
            if let Some(opening_scales) = &opening_scales {
                for (subcolumn, &scale) in opening_scales.iter().enumerate() {
                    for (digit, &gadget) in opening_gadget.iter().enumerate() {
                        let column = subcolumn
                            .checked_mul(group.depth_open)
                            .and_then(|offset| offset.checked_add(digit))
                            .and_then(|offset| e_start.checked_add(offset))
                            .ok_or(AkitaError::InvalidProof)?;
                        e_weights[column] =
                            block_challenge * group.consistency_weight * scale * gadget;
                    }
                }
            } else {
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    let column = e_start.checked_add(digit).ok_or(AkitaError::InvalidProof)?;
                    e_weights[column] = block_challenge * group.consistency_weight * gadget;
                }
            }

            let t_start = block_claim
                .checked_mul(t_stride)
                .ok_or(AkitaError::InvalidProof)?;
            for (a_row, &row_weight) in group.a_row_weights.iter().enumerate() {
                let row_start = a_row
                    .checked_mul(outer_subcolumns)
                    .and_then(|offset| offset.checked_mul(group.depth_commit))
                    .and_then(|offset| t_start.checked_add(offset))
                    .ok_or(AkitaError::InvalidProof)?;
                if let Some(outer_scales) = &outer_scales {
                    for (subcolumn, &scale) in outer_scales.iter().enumerate() {
                        let subcolumn_start = subcolumn
                            .checked_mul(group.depth_commit)
                            .and_then(|offset| row_start.checked_add(offset))
                            .ok_or(AkitaError::InvalidProof)?;
                        for (digit, &gadget) in commitment_gadget.iter().enumerate() {
                            let column = subcolumn_start
                                .checked_add(digit)
                                .ok_or(AkitaError::InvalidProof)?;
                            t_weights[column] = block_challenge * row_weight * gadget * scale;
                        }
                    }
                } else {
                    for (digit, &gadget) in commitment_gadget.iter().enumerate() {
                        let column = row_start
                            .checked_add(digit)
                            .ok_or(AkitaError::InvalidProof)?;
                        t_weights[column] = block_challenge * row_weight * gadget;
                    }
                }
            }
        }

        let z_cols = group
            .num_positions_per_block
            .checked_mul(group.depth_witness)
            .ok_or_else(|| AkitaError::InvalidSetup("structured Z width overflow".into()))?;
        let mut z_weights = vec![E::zero(); z_cols];
        for (position, &opening_a) in opening_a_evals.iter().enumerate() {
            for (digit, &gadget) in witness_gadget.iter().enumerate() {
                let column = position
                    .checked_mul(group.depth_witness)
                    .and_then(|offset| offset.checked_add(digit))
                    .ok_or(AkitaError::InvalidProof)?;
                z_weights[column] = group.consistency_weight * opening_a * gadget;
            }
        }

        Ok(
            self.contract_role_tensor_weights(group.d_ratio, &group.d_tensors, &e_weights, alpha)?
                + self.contract_role_tensor_weights(
                    group.b_ratio,
                    &group.b_tensors,
                    &t_weights,
                    alpha,
                )?
                + self.contract_role_tensor_weights(
                    group.a_ratio,
                    &group.a_tensors,
                    &z_weights,
                    alpha,
                )?,
        )
    }
}

fn extension_gadget<F, E>(depth: usize, log_basis: u32) -> Vec<E>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    crate::gadget_row_scalars::<F>(depth, log_basis)
        .into_iter()
        .map(|weight| E::one().mul_base(weight))
        .collect()
}

fn checked_product(lhs: usize, rhs: usize, context: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}
