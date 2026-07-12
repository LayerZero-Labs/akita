use super::*;
use crate::backend::{RingSwitchQuotientView, RingSwitchRelationView};
use crate::compute::{
    OperationCtx, RingSwitchProveBackend, RingSwitchQuotientKernel, RingSwitchQuotientPlan,
    RingSwitchRelationKernel, RingSwitchRelationPlan,
};
use crate::protocol::ring_switch::PreparedRingSwitchGroup;
use crate::validation::validate_i8_setup_log_basis;
use akita_types::{LevelParams, RelationGroupId, RelationRowId, RelationRowPlan};

/// Add only the high-half quotient contribution of `challenge * ring`.
///
/// Skips the first `D - pos` coefficients per challenge term that cannot
/// contribute (degree < D), cutting iteration count roughly in half.
#[inline(always)]
fn add_sparse_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    challenge: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let c = F::from_i64(coeff as i64);
        let p = pos as usize;
        for s in (D - p)..D {
            quotient[p + s - D] += c * rc[s];
        }
    }
}

#[inline(always)]
fn add_tensor_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    left: &SparseChallenge,
    right: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&left_pos, &left_coeff) in left.positions.iter().zip(left.coeffs.iter()) {
        for (&right_pos, &right_coeff) in right.positions.iter().zip(right.coeffs.iter()) {
            let degree = left_pos as usize + right_pos as usize;
            let (pos, sign) = if degree < D {
                (degree, 1i64)
            } else {
                (degree - D, -1i64)
            };
            let coeff = sign * i64::from(left_coeff) * i64::from(right_coeff);
            let c = F::from_i64(coeff);
            for s in (D - pos)..D {
                quotient[pos + s - D] += c * rc[s];
            }
        }
    }
}

fn parallel_high_half_accumulate<F, R, const D: usize>(
    challenges: &Challenges,
    ring_fn: R,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + Send + Sync,
    R: Fn(usize) -> Option<CyclotomicRing<F, D>> + Sync,
{
    let tensor_blocks_per_claim = match challenges {
        Challenges::Tensor { factored } => {
            factored.validate::<D>()?;
            Some(factored.blocks_per_claim()?)
        }
        Challenges::Sparse { .. } => None,
    };
    let total = match challenges {
        Challenges::Tensor { factored } => factored.total_blocks()?,
        Challenges::Sparse { .. } => challenges.logical_len(),
    };
    let out = cfg_fold_reduce!(
        0..total,
        || vec![F::zero(); D],
        |mut acc: Vec<F>, i: usize| {
            let Some(ring) = ring_fn(i) else {
                return acc;
            };
            match challenges {
                Challenges::Sparse {
                    challenges: sparse, ..
                } => add_sparse_ring_product_high_half::<F, D>(&mut acc, &sparse[i], &ring),
                Challenges::Tensor { factored } => {
                    let blocks_per_claim = tensor_blocks_per_claim.unwrap_or(0);
                    let claim_idx = i / blocks_per_claim;
                    let local_idx = i % blocks_per_claim;
                    let left_idx = claim_idx * factored.left_len + (local_idx / factored.right_len);
                    let right_idx =
                        claim_idx * factored.right_len + (local_idx % factored.right_len);
                    add_tensor_ring_product_high_half::<F, D>(
                        &mut acc,
                        &factored.left[left_idx],
                        &factored.right[right_idx],
                        &ring,
                    );
                }
            }
            acc
        },
        |mut a: Vec<F>, b: Vec<F>| {
            for (ai, bi) in a.iter_mut().zip(b.iter()) {
                *ai += *bi;
            }
            a
        }
    );
    Ok(out)
}

/// Relation quotient `r` returned by [`compute_multi_group_relation_quotient`].
pub(crate) type RelationQuotientOutput<F, const D: usize> = Vec<CyclotomicRing<F, D>>;

fn quotient_from_cyclic_and_reduced<F: FieldCore + HalvingField, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    reduced: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc_c = cyclic.coefficients();
    let red_c = reduced.coefficients();
    let quotient = std::array::from_fn(|k| (cyc_c[k] - red_c[k]).half());
    CyclotomicRing::from_coefficients(quotient)
}

fn add_cyclic_ring_product<F: FieldCore, const D: usize>(
    acc: &mut [F; D],
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) {
    let lhs_coeffs = lhs.coefficients();
    let rhs_coeffs = rhs.coefficients();
    for (i, &a) in lhs_coeffs.iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs_coeffs.iter().enumerate() {
            if !b.is_zero() {
                acc[(i + j) % D] += a * b;
            }
        }
    }
}

fn add_cyclic_scalar_ring_product<F: FieldCore, const D: usize>(
    acc: &mut [F; D],
    scalar: F,
    rhs: &CyclotomicRing<F, D>,
) {
    for (idx, &coeff) in rhs.coefficients().iter().enumerate() {
        if !coeff.is_zero() {
            acc[idx] += scalar * coeff;
        }
    }
}

fn centered_i32_ring<F: FieldCore + FromPrimitiveInt, const D: usize>(
    coeffs: &[i32; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(coeffs[idx] as i64)))
}

fn cyclic_consistency_z_product<F, const D: usize>(
    ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
    z_folded_centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    log_basis: u32,
) -> Result<(CyclotomicRing<F, D>, CyclotomicRing<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let inner_width = block_len
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("z inner width overflow".to_string()))?;
    if inner_width == 0 || z_folded_centered.len() != inner_width {
        return Err(AkitaError::InvalidInput(format!(
            "ring-multiplier z layout mismatch: z_folded_len={} block_len={} depth_commit={} expected={}",
            z_folded_centered.len(),
            block_len,
            depth_commit,
            inner_width
        )));
    }
    let g_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let mut cyclic = [F::zero(); D];
    let mut reduced = CyclotomicRing::<F, D>::zero();

    {
        if ring_multiplier_point.a_len() < block_len {
            return Err(AkitaError::InvalidInput(format!(
                "ring-multiplier a length mismatch: actual={} expected_at_least={block_len}",
                ring_multiplier_point.a_len()
            )));
        }
        for block_idx in 0..block_len {
            let mut z_block = CyclotomicRing::<F, D>::zero();
            for (digit_idx, &g) in g_commit.iter().enumerate() {
                let z_idx = block_idx * depth_commit + digit_idx;
                z_block += centered_i32_ring::<F, D>(&z_folded_centered[z_idx]).scale(&g);
            }
            if let Some(scalar) = ring_multiplier_point.a_constant_coeff(block_idx) {
                add_cyclic_scalar_ring_product::<F, D>(&mut cyclic, scalar, &z_block);
                reduced += z_block.scale(&scalar);
            } else {
                let a_rings = ring_multiplier_point
                    .a_rings_trusted::<D>()?
                    .ok_or(AkitaError::InvalidProof)?;
                let multiplier = a_rings.get(block_idx).ok_or(AkitaError::InvalidProof)?;
                add_cyclic_ring_product::<F, D>(&mut cyclic, multiplier, &z_block);
                reduced += *multiplier * z_block;
            }
        }
    }

    Ok((CyclotomicRing::from_coefficients(cyclic), reduced))
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_multi_group_relation_quotient")]
pub(crate) fn compute_multi_group_relation_quotient<F, B, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, B>,
    lp: &LevelParams,
    opening_batch: &akita_types::OpeningClaimsLayout,
    groups: &[PreparedRingSwitchGroup<'_, F, D>],
    group_ring_multiplier_points: &[&RingMultiplierOpeningPoint<F>],
    group_challenges: &[Challenges],
    e_hat_concat: &[[i8; D]],
    y: &[CyclotomicRing<F, D>],
    row_plan: &RelationRowPlan,
) -> Result<RelationQuotientOutput<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField,
    B: RingSwitchProveBackend<F, D>,
{
    lp.reject_multi_group_multi_chunk("multi-group relation quotient")?;
    lp.validate_root_opening_batch(opening_batch)?;
    if groups.len() != opening_batch.num_groups()
        || group_ring_multiplier_points.len() != opening_batch.num_groups()
        || group_challenges.len() != opening_batch.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    let d_family = row_plan.family(RelationRowId::D).ok();
    let n_d_active = d_family.map_or(0, |family| family.rows().len());
    let num_rows = row_plan.trace_row();
    if y.len() != num_rows {
        return Err(AkitaError::InvalidProof);
    }
    let d_start = d_family.map_or(num_rows, |family| family.rows().start());
    let mut result = vec![CyclotomicRing::<F, D>::zero(); num_rows];
    let mut d_cyclic_rows: Option<Vec<CyclotomicRing<F, D>>> = None;
    let order = opening_batch.root_group_order()?;

    for (order_pos, &group_index) in order.iter().enumerate() {
        let group = groups.get(group_index).ok_or(AkitaError::InvalidProof)?;
        let ring_multiplier_point = group_ring_multiplier_points
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let challenges = group_challenges
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let e_folded = &group.e_folded;
        let recomposed_inner_rows = &group.recomposed_inner_rows;
        let group_layout = opening_batch.group_layout(group_index)?;
        let log_basis = group.params.log_basis();
        let num_digits_open = group.params.num_digits_open();
        let n_a = group.params.a_rows_len();
        let n_b = group.params.b_rows_len();
        let blocks_per_claim = group.params.num_blocks();
        let inner_width = group.params.a_col_len();
        validate_i8_setup_log_basis(log_basis, "for multi-group relation quotient")?;
        if group_layout.num_polynomials() == 0 {
            return Err(AkitaError::InvalidProof);
        }
        let expected_blocks = group_layout
            .num_polynomials()
            .checked_mul(blocks_per_claim)
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.logical_len() != expected_blocks
            || e_folded.len() != expected_blocks
            || recomposed_inner_rows.len() != expected_blocks
            || group.e_hat.typed_planes::<D>()?.len()
                != expected_blocks
                    .checked_mul(num_digits_open)
                    .ok_or(AkitaError::InvalidProof)?
        {
            return Err(AkitaError::InvalidProof);
        }
        if group.z_centered.len() != inner_width {
            return Err(AkitaError::InvalidProof);
        }
        let expected_t_hat_block_digits = n_a
            .checked_mul(num_digits_open)
            .ok_or(AkitaError::InvalidProof)?;
        if group.t_hat.block_count() != expected_blocks
            || group
                .t_hat
                .block_sizes()
                .iter()
                .any(|&size| size != expected_t_hat_block_digits)
        {
            return Err(AkitaError::InvalidProof);
        }

        let n_d_for_group = if order_pos == 0 { n_d_active } else { 0 };
        let e_hat_for_group = if order_pos == 0 { e_hat_concat } else { &[] };
        let mut z_segments = group.z_centered.chunks(inner_width);
        let first_z_segment = z_segments.next().ok_or(AkitaError::InvalidProof)?;
        let relation_rows = RingSwitchRelationKernel::relation_rows(
            backend,
            prepared,
            RingSwitchRelationView {
                e_hat: e_hat_for_group,
                t_hat: group.t_hat.typed_planes::<D>()?,
                z_segment: first_z_segment,
                z_folded_centered_inf_norm: group.z_inf,
            },
            RingSwitchRelationPlan {
                n_d: n_d_for_group,
                n_b,
                n_a,
                log_basis,
            },
        )?;
        if relation_rows.d_cyclic.len() != n_d_for_group
            || relation_rows.b_cyclic.len() != n_b
            || relation_rows.a_quotients.len() != n_a
        {
            return Err(AkitaError::InvalidProof);
        }
        if n_d_for_group != 0 {
            d_cyclic_rows = Some(relation_rows.d_cyclic);
        }
        let mut a_quotients = relation_rows.a_quotients;
        for z_segment in z_segments {
            let segment_rows = RingSwitchQuotientKernel::quotient_rows(
                backend,
                prepared,
                RingSwitchQuotientView {
                    z_segment,
                    z_folded_centered_inf_norm: group.z_inf,
                },
                RingSwitchQuotientPlan { n_a },
            )?;
            if segment_rows.len() != n_a {
                return Err(AkitaError::InvalidProof);
            }
            for (dst, src) in a_quotients.iter_mut().zip(segment_rows.into_iter()) {
                *dst += src;
            }
        }

        let consistency_z_quotient = if ring_multiplier_point.is_constant() {
            CyclotomicRing::<F, D>::zero()
        } else {
            let (consistency_z_cyclic, consistency_z_reduced) = cyclic_consistency_z_product::<F, D>(
                ring_multiplier_point,
                &group.z_centered,
                group.params.block_len(),
                group.params.num_digits_commit(),
                log_basis,
            )?;
            quotient_from_cyclic_and_reduced(&consistency_z_cyclic, &consistency_z_reduced)
        };
        let quotient = parallel_high_half_accumulate::<F, _, D>(challenges, |i| Some(e_folded[i]))?;
        let mut quotient = CyclotomicRing::from_slice(&quotient);
        quotient -= consistency_z_quotient;
        let consistency_row = row_plan.family(RelationRowId::Consistency)?.rows().start();
        result[consistency_row] += quotient;

        let final_group_index = opening_batch.root_final_group_index()?;
        let relation_group = if group_index == final_group_index {
            RelationGroupId::Current
        } else {
            RelationGroupId::Precommitted { index: group_index }
        };
        let a_range = row_plan
            .family(RelationRowId::A {
                group: relation_group,
            })?
            .rows()
            .range();
        if a_range.len() != n_a {
            return Err(AkitaError::InvalidProof);
        }
        for (a_idx, row_idx) in a_range.enumerate() {
            let mut quotient = parallel_high_half_accumulate::<F, _, D>(challenges, |i| {
                let claim_idx = i / blocks_per_claim;
                let block_idx = i % blocks_per_claim;
                let inner_idx = claim_idx * blocks_per_claim + block_idx;
                recomposed_inner_rows[inner_idx].get(a_idx).copied()
            })?;
            let a_q = a_quotients[a_idx].coefficients();
            for k in 0..D {
                quotient[k] -= a_q[k];
            }
            result[row_idx] = CyclotomicRing::from_slice(&quotient);
        }

        let b_range = row_plan
            .family(RelationRowId::B {
                group: relation_group,
            })?
            .rows()
            .range();
        if b_range.len() != n_b {
            return Err(AkitaError::InvalidProof);
        }
        for (commit_idx, row_idx) in b_range.enumerate() {
            let cyclic = relation_rows
                .b_cyclic
                .get(commit_idx)
                .ok_or(AkitaError::InvalidProof)?;
            result[row_idx] = quotient_from_cyclic_and_reduced(cyclic, &y[row_idx]);
        }
    }

    // Terminal layout (`WithoutDBlock`) drops the D rows from M entirely, so no
    // group produces D-cyclic rows and `d_start == num_rows`.
    if n_d_active == 0 {
        if d_cyclic_rows.is_some() {
            return Err(AkitaError::InvalidProof);
        }
    } else {
        let d_cyclic_rows = d_cyclic_rows.ok_or(AkitaError::InvalidProof)?;
        if d_cyclic_rows.len() != n_d_active {
            return Err(AkitaError::InvalidProof);
        }
        for (d_idx, cyclic) in d_cyclic_rows.iter().enumerate() {
            let row_idx = d_start.checked_add(d_idx).ok_or(AkitaError::InvalidProof)?;
            result[row_idx] = quotient_from_cyclic_and_reduced(cyclic, &y[row_idx]);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_support::tensor_oracle_challenges;
    use akita_challenges::SparseChallenge;
    use akita_field::Prime128OffsetA7F7 as F;

    fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            F::from_u64(offset + idx as u64 + 1)
        }))
    }

    fn sparse_challenge_as_ring<const D: usize>(
        challenge: &SparseChallenge,
    ) -> CyclotomicRing<F, D> {
        let mut coeffs = [F::zero(); D];
        for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
            coeffs[pos as usize] += F::from_i64(i64::from(coeff));
        }
        CyclotomicRing::from_coefficients(coeffs)
    }

    fn add_ring_product_reference_high_half<const D: usize>(
        quotient: &mut [F],
        challenge: &CyclotomicRing<F, D>,
        ring: &CyclotomicRing<F, D>,
    ) {
        let rc = ring.coefficients();
        for (p, &c) in challenge.coefficients().iter().enumerate() {
            for s in (D - p)..D {
                quotient[p + s - D] += c * rc[s];
            }
        }
    }

    #[test]
    fn tensor_high_half_streaming_matches_ring_multiplication_reference() {
        const D: usize = 8;
        let tensor = tensor_oracle_challenges::<D>();
        let rings = (0..tensor.total_blocks().unwrap())
            .map(|idx| (idx != 3).then(|| ring::<D>(10 * idx as u64)))
            .collect::<Vec<_>>();
        let challenges = Challenges::Tensor {
            factored: tensor.clone(),
        };

        let got = parallel_high_half_accumulate::<F, _, D>(&challenges, |idx| rings[idx]).unwrap();
        let mut expected = vec![F::zero(); D];
        for (idx, ring) in rings
            .iter()
            .enumerate()
            .take(tensor.total_blocks().unwrap())
        {
            if let Some(ring) = ring {
                let (_, _, left, right) = tensor.factors_for_logical_block(idx).unwrap();
                let challenge =
                    sparse_challenge_as_ring::<D>(left) * sparse_challenge_as_ring::<D>(right);
                add_ring_product_reference_high_half::<D>(&mut expected, &challenge, ring);
            }
        }

        assert_eq!(got, expected);
    }
}
