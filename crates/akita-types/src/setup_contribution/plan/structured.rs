use super::*;
use akita_algebra::{
    offset_eq::{
        eval_affine_digit_intervals, EqPairTensorAxis, EqPairTensorFamily, EqPairTensorWeights,
    },
    ring::scalar_powers_with_stride,
};
use std::collections::BTreeMap;

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
        let opening_low = opening_scales.as_deref().unwrap_or(&[]);
        let outer_low = outer_scales.as_deref().unwrap_or(&[]);
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
            || group.num_live_blocks == 0
            || !group.d_tensors.len().is_multiple_of(group.num_claims)
        {
            return Err(AkitaError::InvalidSetup(
                "structured role tensor families disagree".into(),
            ));
        }
        if (group.n_b == 0 && !group.b_tensors.is_empty())
            || (group.n_b != 0 && group.b_tensors.len() != group.d_tensors.len())
        {
            return Err(AkitaError::InvalidSetup(
                "structured B tensor families disagree".into(),
            ));
        }

        let mut b_by_unit = BTreeMap::new();
        for tensor in &group.b_tensors {
            let key = canonical_unit_tensor_key(
                tensor,
                t_stride,
                group.b_ratio,
                group.num_claims,
                group.num_live_blocks,
                "structured B tensor",
            )?;
            if b_by_unit.insert(key, tensor).is_some() {
                return Err(AkitaError::InvalidSetup(
                    "structured B tensor unit is duplicated".into(),
                ));
            }
        }
        let mut unit_families = Vec::with_capacity(group.d_tensors.len());
        for tensor in &group.d_tensors {
            let key = canonical_unit_tensor_key(
                tensor,
                e_stride,
                group.d_ratio,
                group.num_claims,
                group.num_live_blocks,
                "structured D tensor",
            )?;
            let b_tensor = if group.n_b == 0 {
                None
            } else {
                Some(b_by_unit.remove(&key).ok_or_else(|| {
                    AkitaError::InvalidSetup("structured D/B tensor units disagree".into())
                })?)
            };
            unit_families.push((key, tensor, b_tensor));
        }
        if !b_by_unit.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "structured B tensor has no matching D unit".into(),
            ));
        }
        unit_families.sort_unstable_by_key(|(key, _, _)| *key);
        let mut claim_coverage = vec![0usize; group.num_claims];
        for (key, _, _) in &unit_families {
            let (claim, block_start, unit_blocks) = *key;
            let cursor = claim_coverage
                .get_mut(claim)
                .ok_or(AkitaError::InvalidProof)?;
            if *cursor != block_start {
                return Err(AkitaError::InvalidSetup(
                    "structured tensor units do not form a block partition".into(),
                ));
            }
            *cursor = cursor
                .checked_add(unit_blocks)
                .ok_or(AkitaError::InvalidProof)?;
        }
        if claim_coverage
            .iter()
            .any(|&covered| covered != group.num_live_blocks)
        {
            return Err(AkitaError::InvalidSetup(
                "structured tensors do not cover every live block".into(),
            ));
        }
        let partition = unit_families
            .iter()
            .take_while(|((claim, _, _), _, _)| *claim == 0)
            .map(|((_, block_start, unit_blocks), _, _)| (*block_start, *unit_blocks))
            .collect::<Vec<_>>();
        if partition.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "structured tensor partition is empty".into(),
            ));
        }
        for claim in 1..group.num_claims {
            if !unit_families
                .iter()
                .filter(|((family_claim, _, _), _, _)| *family_claim == claim)
                .map(|((_, block_start, unit_blocks), _, _)| (*block_start, *unit_blocks))
                .eq(partition.iter().copied())
            {
                return Err(AkitaError::InvalidSetup(
                    "structured claims disagree on the chunk partition".into(),
                ));
            }
        }
        let unit_count = partition.len();
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

        let evaluation = cfg_fold_reduce!(
            0..unit_families.len(),
            || Ok(E::zero()),
            |acc: Result<E, AkitaError>, family_index| {
                let (key, d_tensor, b_tensor) = unit_families
                    .get(family_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let (claim, global_block_start, unit_blocks) = *key;
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
                let mut contribution = group.consistency_weight
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
                        &[],
                    )?;

                if let Some(b_tensor) = b_tensor {
                    let t_outer_start = global_block_start
                        .checked_mul(group.n_a)
                        .and_then(|start| start.checked_mul(outer_subcolumns))
                        .ok_or(AkitaError::InvalidProof)?;
                    let t_live_len = unit_blocks
                        .checked_mul(group.n_a)
                        .and_then(|len| len.checked_mul(outer_subcolumns))
                        .ok_or(AkitaError::InvalidProof)?;
                    contribution += eval_affine_digit_intervals(
                        point,
                        &[b_tensor.right_offset],
                        t_outer_start,
                        t_live_len,
                        commitment_digits.len(),
                        1,
                        commitment_digits,
                        t_high_weights.get(claim).ok_or(AkitaError::InvalidProof)?,
                        outer_low,
                        &[],
                    )?;
                }
                Ok(acc? + contribution)
            },
            |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
        )?;

        let projection_lanes = (group.a_ratio != 1)
            .then(|| scalar_powers_with_stride(alpha, base_ring_dim, group.a_ratio))
            .transpose()?;
        let fold_digits = if let Some(lanes) = &projection_lanes {
            group
                .fold_gadget
                .iter()
                .flat_map(|&fold| lanes.iter().map(move |&lane| -(fold * lane)))
                .collect::<Vec<_>>()
        } else {
            group.fold_gadget.iter().map(|&fold| -fold).collect()
        };
        let fold_weights = group
            .fold_gadget
            .iter()
            .map(|&fold| -fold)
            .collect::<Vec<E>>();
        for tensor in &group.a_tensors {
            let expected = EqPairTensorFamily::new(
                0,
                tensor.right_offset,
                E::one(),
                vec![
                    EqPairTensorAxis::unit(z_cols, 1, fold_digits.len()),
                    EqPairTensorAxis::dense(0, group.a_ratio, fold_weights.clone()),
                ],
            )?;
            if tensor != &expected {
                return Err(AkitaError::InvalidSetup(
                    "structured A tensor disagrees with canonical geometry".into(),
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
                &[],
            )?
        } else {
            let weighted_base_count = checked_product(
                opening_a_evals.len(),
                a_base_offsets.len(),
                "structured weighted A base count overflow",
            )?;
            let mut weighted_bases = Vec::new();
            let mut base_scales = Vec::new();
            weighted_bases
                .try_reserve_exact(weighted_base_count)
                .map_err(|_| {
                    AkitaError::InvalidSetup("structured weighted A bases are too large".into())
                })?;
            base_scales
                .try_reserve_exact(weighted_base_count)
                .map_err(|_| {
                    AkitaError::InvalidSetup("structured weighted A scales are too large".into())
                })?;
            for (position, &opening_a) in opening_a_evals.iter().enumerate() {
                let position_offset = position
                    .checked_mul(group.depth_witness)
                    .and_then(|offset| offset.checked_mul(fold_digits.len()))
                    .ok_or(AkitaError::InvalidProof)?;
                for &base in &a_base_offsets {
                    weighted_bases.push(
                        base.checked_add(position_offset)
                            .ok_or(AkitaError::InvalidProof)?,
                    );
                    base_scales.push(opening_a);
                }
            }
            eval_affine_digit_intervals(
                point,
                &weighted_bases,
                0,
                group.depth_witness,
                fold_digits.len(),
                1,
                &fold_digits,
                &witness_gadget,
                &[],
                &base_scales,
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

fn canonical_unit_tensor_key<E: FieldCore>(
    tensor: &EqPairTensorFamily<E>,
    semantic_stride: usize,
    role_ratio: usize,
    num_claims: usize,
    num_live_blocks: usize,
    context: &'static str,
) -> Result<(usize, usize, usize), AkitaError> {
    let coordinates = tensor_coordinate_count(tensor)?;
    let unit_blocks = coordinates
        .checked_div(semantic_stride)
        .filter(|&count| count != 0 && coordinates.is_multiple_of(semantic_stride))
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} is not a complete unit")))?;
    let semantic_start = tensor
        .left_offset
        .checked_div(semantic_stride)
        .filter(|_| tensor.left_offset.is_multiple_of(semantic_stride))
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{context} is not column aligned")))?;
    let claim = semantic_start / num_live_blocks;
    let block_start = semantic_start % num_live_blocks;
    if claim >= num_claims
        || block_start
            .checked_add(unit_blocks)
            .is_none_or(|end| end > num_live_blocks)
    {
        return Err(AkitaError::InvalidSetup(format!(
            "{context} is outside the group block domain"
        )));
    }
    let expected = EqPairTensorFamily::new(
        tensor.left_offset,
        tensor.right_offset,
        E::one(),
        vec![EqPairTensorAxis {
            len: coordinates,
            left_stride: 1,
            right_stride: role_ratio,
            weights: EqPairTensorWeights::Unit,
        }],
    )?;
    if tensor != &expected {
        return Err(AkitaError::InvalidSetup(format!(
            "{context} disagrees with canonical affine geometry"
        )));
    }
    Ok((claim, block_start, unit_blocks))
}
