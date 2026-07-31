use super::*;
use akita_algebra::{
    offset_eq::{eval_affine_digit_intervals, EqPairTensorFamily},
    ring::scalar_powers_with_stride,
};

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
        if self
            .direct_scan_alpha
            .is_some_and(|prepared| prepared != alpha)
        {
            return Err(AkitaError::InvalidInput(
                "structured relation alpha disagrees with direct setup weights".into(),
            ));
        }
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
            SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims)?;
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
        let opening_scales = (opening_subcolumns != 1)
            .then(|| scalar_powers_with_stride(alpha, group.role_dims.d_d(), opening_subcolumns))
            .transpose()?;
        let outer_scales = (outer_subcolumns != 1)
            .then(|| scalar_powers_with_stride(alpha, group.role_dims.d_b(), outer_subcolumns))
            .transpose()?;
        let e_len = block_claims
            .checked_mul(e_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("structured E width overflow".into()))?;
        let t_len = block_claims
            .checked_mul(t_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("structured T width overflow".into()))?;
        let z_cols = group
            .num_positions_per_block
            .checked_mul(group.depth_witness)
            .ok_or_else(|| AkitaError::InvalidSetup("structured Z width overflow".into()))?;

        if let Some(weights) = &group.direct_scan_weights {
            if weights.e.len() != e_len
                || weights.t.len() != t_len
                || weights.z.len() != z_cols
                || group.a_row_weights.len() != group.n_a
            {
                return Err(AkitaError::InvalidProof);
            }
            let projected_opening_gadget = opening_scales.as_ref().map(|scales| {
                scales
                    .iter()
                    .flat_map(|&scale| opening_gadget.iter().map(move |&gadget| scale * gadget))
                    .collect::<Vec<_>>()
            });
            let direct_opening_gadget = projected_opening_gadget
                .as_deref()
                .unwrap_or(&opening_gadget);
            let projected_commitment_gadget = outer_scales.as_ref().map(|scales| {
                scales
                    .iter()
                    .flat_map(|&scale| commitment_gadget.iter().map(move |&gadget| scale * gadget))
                    .collect::<Vec<_>>()
            });
            let direct_commitment_gadget = projected_commitment_gadget
                .as_deref()
                .unwrap_or(&commitment_gadget);
            let t_row_stride = checked_product(
                outer_subcolumns,
                group.depth_commit,
                "structured T row stride overflow",
            )?;
            if direct_opening_gadget.len() != e_stride
                || direct_commitment_gadget.len() != t_row_stride
            {
                return Err(AkitaError::InvalidProof);
            }
            let et = cfg_fold_reduce!(
                0..block_claims,
                || Ok(E::zero()),
                |acc: Result<E, AkitaError>, block_claim| {
                    let e_start = block_claim
                        .checked_mul(e_stride)
                        .ok_or(AkitaError::InvalidProof)?;
                    let e_eq =
                        checked_slice(&weights.e, e_start, e_stride, "structured direct E slice")?;
                    let e = e_eq
                        .iter()
                        .zip(direct_opening_gadget)
                        .fold(E::zero(), |sum, (&eq, &gadget)| sum + eq * gadget);

                    let t_start = block_claim
                        .checked_mul(t_stride)
                        .ok_or(AkitaError::InvalidProof)?;
                    let t_eq =
                        checked_slice(&weights.t, t_start, t_stride, "structured direct T slice")?;
                    let t = t_eq
                        .chunks_exact(t_row_stride)
                        .zip(group.a_row_weights.iter())
                        .fold(E::zero(), |sum, (row, &row_weight)| {
                            sum + row_weight
                                * row
                                    .iter()
                                    .zip(direct_commitment_gadget)
                                    .fold(E::zero(), |inner, (&eq, &gadget)| inner + eq * gadget)
                        });
                    let block_challenge = *block_challenges
                        .get(block_claim)
                        .ok_or(AkitaError::InvalidProof)?;
                    Ok(acc? + block_challenge * (group.consistency_weight * e + t))
                },
                |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
            )?;
            let z = cfg_fold_reduce!(
                0..group.num_positions_per_block,
                || Ok(E::zero()),
                |acc: Result<E, AkitaError>, position| {
                    let start = position
                        .checked_mul(group.depth_witness)
                        .ok_or(AkitaError::InvalidProof)?;
                    let eq = checked_slice(
                        &weights.z,
                        start,
                        group.depth_witness,
                        "structured direct Z slice",
                    )?;
                    let inner = eq
                        .iter()
                        .zip(&witness_gadget)
                        .fold(E::zero(), |sum, (&eq, &gadget)| sum + eq * gadget);
                    Ok(acc?
                        + group.consistency_weight
                            * *opening_a_evals
                                .get(position)
                                .ok_or(AkitaError::InvalidProof)?
                            * inner)
                },
                |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
            )?;
            return Ok(et + z);
        }

        let point = self.relation_address.point();
        let base_ring_dim = self.projection_geometry.base_ring_dim();
        let one = [E::one()];
        let opening_low = opening_scales.as_deref().unwrap_or(&one);
        let outer_low = outer_scales.as_deref().unwrap_or(&one);
        let projected_digits = |gadget: &[E], ratio: usize| -> Result<Option<Vec<E>>, AkitaError> {
            if ratio == 1 {
                return Ok(None);
            }
            let lanes = scalar_powers_with_stride(alpha, base_ring_dim, ratio)?;
            Ok(Some(
                gadget
                    .iter()
                    .flat_map(|&digit| lanes.iter().map(move |&lane| digit * lane))
                    .collect(),
            ))
        };
        let projected_opening = projected_digits(&opening_gadget, group.d_ratio)?;
        let opening_digits = projected_opening.as_deref().unwrap_or(&opening_gadget);
        let projected_commitment = projected_digits(&commitment_gadget, group.b_ratio)?;
        let commitment_digits = projected_commitment
            .as_deref()
            .unwrap_or(&commitment_gadget);

        if group.num_claims == 0
            || !group.d_tensors.len().is_multiple_of(group.num_claims)
            || (group.n_b != 0 && group.b_tensors.len() != group.d_tensors.len())
        {
            return Err(AkitaError::InvalidSetup(
                "structured role tensor families disagree".into(),
            ));
        }
        let unit_count = group.d_tensors.len() / group.num_claims;
        if group.a_tensors.len() != usize::from(group.n_a != 0) * unit_count {
            return Err(AkitaError::InvalidSetup(
                "structured A tensor families disagree".into(),
            ));
        }

        let t_high_weights = (0..group.num_claims)
            .map(|claim| {
                let block_start = claim
                    .checked_mul(group.num_live_blocks)
                    .ok_or(AkitaError::InvalidProof)?;
                let challenges = checked_slice(
                    block_challenges,
                    block_start,
                    group.num_live_blocks,
                    "structured T block factors",
                )?;
                Ok(challenges
                    .iter()
                    .flat_map(|&challenge| {
                        group.a_row_weights.iter().map(move |&row| challenge * row)
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;

        let mut evaluation = E::zero();
        let mut global_block_start = 0usize;
        for unit in 0..unit_count {
            let first_d = group
                .d_tensors
                .get(
                    unit.checked_mul(group.num_claims)
                        .ok_or(AkitaError::InvalidProof)?,
                )
                .ok_or(AkitaError::InvalidProof)?;
            let d_coordinates = tensor_coordinate_count(first_d)?;
            let unit_blocks = d_coordinates
                .checked_div(e_stride)
                .filter(|&count| count != 0 && d_coordinates.is_multiple_of(e_stride))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("structured D tensor is not a complete unit".into())
                })?;
            let unit_end = global_block_start
                .checked_add(unit_blocks)
                .ok_or(AkitaError::InvalidProof)?;
            if unit_end > group.num_live_blocks {
                return Err(AkitaError::InvalidProof);
            }

            for claim in 0..group.num_claims {
                let family_index = unit
                    .checked_mul(group.num_claims)
                    .and_then(|base| base.checked_add(claim))
                    .ok_or(AkitaError::InvalidProof)?;
                let d_tensor = group
                    .d_tensors
                    .get(family_index)
                    .ok_or(AkitaError::InvalidProof)?;
                if tensor_coordinate_count(d_tensor)? != d_coordinates {
                    return Err(AkitaError::InvalidSetup(
                        "structured D claim tensors disagree".into(),
                    ));
                }
                let claim_start = claim
                    .checked_mul(group.num_live_blocks)
                    .ok_or(AkitaError::InvalidProof)?;
                let claim_challenges = checked_slice(
                    block_challenges,
                    claim_start,
                    group.num_live_blocks,
                    "structured E block factors",
                )?;
                let e_outer_start = global_block_start
                    .checked_mul(opening_subcolumns)
                    .ok_or(AkitaError::InvalidProof)?;
                let e_live_len = unit_blocks
                    .checked_mul(opening_subcolumns)
                    .ok_or(AkitaError::InvalidProof)?;
                evaluation += group.consistency_weight
                    * eval_affine_digit_intervals(
                        point,
                        &[d_tensor.right_offset],
                        e_outer_start,
                        e_live_len,
                        opening_digits.len(),
                        1,
                        opening_digits,
                        claim_challenges,
                        opening_low,
                    )?;

                if group.n_b != 0 {
                    let b_tensor = group
                        .b_tensors
                        .get(family_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    let expected_b = unit_blocks
                        .checked_mul(t_stride)
                        .ok_or(AkitaError::InvalidProof)?;
                    if tensor_coordinate_count(b_tensor)? != expected_b {
                        return Err(AkitaError::InvalidSetup(
                            "structured B tensor is not a complete unit".into(),
                        ));
                    }
                    let t_outer_start = global_block_start
                        .checked_mul(group.n_a)
                        .and_then(|start| start.checked_mul(outer_subcolumns))
                        .ok_or(AkitaError::InvalidProof)?;
                    let t_live_len = unit_blocks
                        .checked_mul(group.n_a)
                        .and_then(|len| len.checked_mul(outer_subcolumns))
                        .ok_or(AkitaError::InvalidProof)?;
                    evaluation += eval_affine_digit_intervals(
                        point,
                        &[b_tensor.right_offset],
                        t_outer_start,
                        t_live_len,
                        commitment_digits.len(),
                        1,
                        commitment_digits,
                        t_high_weights.get(claim).ok_or(AkitaError::InvalidProof)?,
                        outer_low,
                    )?;
                }
            }
            global_block_start = unit_end;
        }
        if global_block_start != group.num_live_blocks {
            return Err(AkitaError::InvalidSetup(
                "structured D tensors do not cover every live block".into(),
            ));
        }

        let projection_lanes = (group.a_ratio != 1)
            .then(|| scalar_powers_with_stride(alpha, base_ring_dim, group.a_ratio))
            .transpose()?;
        let fold_digits = group
            .fold_gadget
            .iter()
            .flat_map(|&fold| {
                projection_lanes
                    .as_deref()
                    .unwrap_or(&one)
                    .iter()
                    .map(move |&lane| -(fold * lane))
            })
            .collect::<Vec<_>>();
        let a_coordinates = z_cols
            .checked_mul(group.fold_gadget.len())
            .ok_or(AkitaError::InvalidProof)?;
        for tensor in &group.a_tensors {
            if tensor_coordinate_count(tensor)? != a_coordinates {
                return Err(AkitaError::InvalidSetup(
                    "structured A tensor is not a complete unit".into(),
                ));
            }
        }
        let a_base_offsets = group
            .a_tensors
            .iter()
            .map(|tensor| tensor.right_offset)
            .collect::<Vec<_>>();
        let z = if witness_gadget.len().is_power_of_two() {
            eval_affine_digit_intervals(
                point,
                &a_base_offsets,
                0,
                z_cols,
                fold_digits.len(),
                1,
                &fold_digits,
                opening_a_evals,
                &witness_gadget,
            )?
        } else {
            opening_a_evals.iter().enumerate().try_fold(
                E::zero(),
                |sum, (position, &opening_a)| {
                    let position_offset = position
                        .checked_mul(group.depth_witness)
                        .and_then(|offset| offset.checked_mul(fold_digits.len()))
                        .ok_or(AkitaError::InvalidProof)?;
                    let position_bases = a_base_offsets
                        .iter()
                        .map(|base| {
                            base.checked_add(position_offset)
                                .ok_or(AkitaError::InvalidProof)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(sum
                        + opening_a
                            * eval_affine_digit_intervals(
                                point,
                                &position_bases,
                                0,
                                group.depth_witness,
                                fold_digits.len(),
                                1,
                                &fold_digits,
                                &witness_gadget,
                                &one,
                            )?)
                },
            )?
        };
        Ok(evaluation + group.consistency_weight * z)
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

fn tensor_coordinate_count<E: FieldCore>(
    tensor: &EqPairTensorFamily<E>,
) -> Result<usize, AkitaError> {
    tensor.axes.iter().try_fold(1usize, |count, axis| {
        count
            .checked_mul(axis.len)
            .ok_or_else(|| AkitaError::InvalidSetup("structured tensor size overflow".into()))
    })
}
