use super::*;
use crate::backend::{RingSwitchQuotientView, RingSwitchRelationView};
use crate::compute::{
    OperationCtx, RingSwitchProveBackend, RingSwitchQuotientKernel, RingSwitchQuotientPlan,
    RingSwitchRelationKernel, RingSwitchRelationPlan, RuntimeRingSwitchProveBackend,
};
use crate::protocol::ring_switch::PreparedRingSwitchGroup;
use crate::validation::validate_i8_setup_log_basis;
use akita_types::{CommitmentRingDims, LevelParams, RelationMatrixRowLayout, RingVec};

#[inline]
fn accumulate_small_signed<F: FieldCore + FromPrimitiveInt>(dst: &mut F, value: F, coeff: i64) {
    match coeff {
        1 => *dst += value,
        -1 => *dst -= value,
        2 => {
            *dst += value;
            *dst += value;
        }
        -2 => {
            *dst -= value;
            *dst -= value;
        }
        _ => *dst += value * F::from_i64(coeff),
    }
}

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
        let p = pos as usize;
        for s in (D - p)..D {
            accumulate_small_signed(&mut quotient[p + s - D], rc[s], i64::from(coeff));
        }
    }
}

#[inline(always)]
fn add_tensor_ring_product_high_half<F: FieldCore + CanonicalField, const D: usize>(
    quotient: &mut [F],
    fold_high: &SparseChallenge,
    fold_low: &SparseChallenge,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (&high_pos, &high_coeff) in fold_high.positions.iter().zip(fold_high.coeffs.iter()) {
        for (&low_pos, &low_coeff) in fold_low.positions.iter().zip(fold_low.coeffs.iter()) {
            let degree = high_pos as usize + low_pos as usize;
            let (pos, sign) = if degree < D {
                (degree, 1i64)
            } else {
                (degree - D, -1i64)
            };
            let coeff = sign * i64::from(high_coeff) * i64::from(low_coeff);
            for s in (D - pos)..D {
                accumulate_small_signed(&mut quotient[pos + s - D], rc[s], coeff);
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
                    let high_idx =
                        claim_idx * factored.fold_high_len() + (local_idx / factored.fold_low_len);
                    let low_idx =
                        claim_idx * factored.fold_low_len + (local_idx % factored.fold_low_len);
                    add_tensor_ring_product_high_half::<F, D>(
                        &mut acc,
                        &factored.fold_high[high_idx],
                        &factored.fold_low[low_idx],
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

#[derive(Clone)]
pub(crate) struct RelationQuotientRow<F: FieldCore> {
    ring_dim: usize,
    coeffs: Vec<F>,
}

/// Relation quotient `r` returned by [`compute_multi_group_relation_quotient`].
///
/// Each row retains the native dimension of its relation family. This is the
/// D-free orchestration boundary between the role-local quotient kernels and
/// the flat recursive witness.
#[derive(Clone)]
pub(crate) struct RelationQuotientOutput<F: FieldCore> {
    rows: Vec<RelationQuotientRow<F>>,
}

impl<F: FieldCore> RelationQuotientOutput<F> {
    fn from_slots(slots: Vec<Option<RelationQuotientRow<F>>>) -> Result<Self, AkitaError> {
        let mut rows = Vec::with_capacity(slots.len());
        for (index, row) in slots.into_iter().enumerate() {
            rows.push(row.ok_or_else(|| {
                AkitaError::InvalidInput(format!("relation quotient row {index} was not built"))
            })?);
        }
        Ok(Self { rows })
    }

    fn row_from_ring<const D: usize>(ring: CyclotomicRing<F, D>) -> RelationQuotientRow<F> {
        RelationQuotientRow {
            ring_dim: D,
            coeffs: ring.coefficients().to_vec(),
        }
    }

    pub(crate) fn rows(&self) -> &[RelationQuotientRow<F>] {
        &self.rows
    }

    pub(crate) fn into_padded_ring_vec<const D: usize>(self) -> Result<RingVec<F>, AkitaError> {
        let mut coeffs = Vec::with_capacity(self.rows.len() * D);
        for row in self.rows {
            if row.coeffs.len() > D || !D.is_multiple_of(row.ring_dim) {
                return Err(AkitaError::InvalidSize {
                    expected: D,
                    actual: row.coeffs.len(),
                });
            }
            coeffs.extend_from_slice(&row.coeffs);
            coeffs.resize(coeffs.len() + (D - row.coeffs.len()), F::zero());
        }
        Ok(RingVec::from_coeffs(coeffs))
    }
}

impl<F: FieldCore> RelationQuotientRow<F> {
    pub(crate) fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    pub(crate) fn coeffs(&self) -> &[F] {
        &self.coeffs
    }
}

fn ring_from_flat_y<F: FieldCore, const D: usize>(
    y: &RingVec<F>,
    offset: usize,
) -> Result<CyclotomicRing<F, D>, AkitaError> {
    let end = offset.checked_add(D).ok_or(AkitaError::InvalidProof)?;
    let coeffs: [F; D] = y
        .coeffs()
        .get(offset..end)
        .ok_or(AkitaError::InvalidProof)?
        .try_into()
        .map_err(|_| AkitaError::InvalidProof)?;
    Ok(CyclotomicRing::from_coefficients(coeffs))
}

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
    fold_position_count: usize,
    depth_commit: usize,
    log_basis: u32,
) -> Result<(CyclotomicRing<F, D>, CyclotomicRing<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
{
    let inner_width = fold_position_count
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("z inner width overflow".to_string()))?;
    if inner_width == 0 || z_folded_centered.len() != inner_width {
        return Err(AkitaError::InvalidInput(format!(
            "ring-multiplier z layout mismatch: z_folded_len={} fold_position_count={} depth_commit={} expected={}",
            z_folded_centered.len(),
            fold_position_count,
            depth_commit,
            inner_width
        )));
    }
    let g_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    let mut cyclic = [F::zero(); D];
    let mut reduced = CyclotomicRing::<F, D>::zero();

    {
        if ring_multiplier_point.position_len() < fold_position_count {
            return Err(AkitaError::InvalidInput(format!(
                "ring-multiplier a length mismatch: actual={} expected_at_least={fold_position_count}",
                ring_multiplier_point.position_len()
            )));
        }
        for block_idx in 0..fold_position_count {
            let mut z_block = CyclotomicRing::<F, D>::zero();
            for (digit_idx, &g) in g_commit.iter().enumerate() {
                let z_idx = block_idx * depth_commit + digit_idx;
                z_block += centered_i32_ring::<F, D>(&z_folded_centered[z_idx]).scale(&g);
            }
            if let Some(scalar) = ring_multiplier_point.position_constant_coeff(block_idx) {
                add_cyclic_scalar_ring_product::<F, D>(&mut cyclic, scalar, &z_block);
                reduced += z_block.scale(&scalar);
            } else {
                let a_rings = ring_multiplier_point
                    .position_rings_trusted::<D>()?
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
    e_hat_concat: &DigitBlocks,
    y: &RingVec<F>,
    role_dims: CommitmentRingDims,
    relation_matrix_row_layout: RelationMatrixRowLayout,
) -> Result<RelationQuotientOutput<F>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HalvingField,
    B: RuntimeRingSwitchProveBackend<F>
        + RingSwitchProveBackend<F, 16>
        + RingSwitchProveBackend<F, D>,
{
    lp.reject_multi_group_multi_chunk("multi-group relation quotient")?;
    lp.validate_opening_batch(opening_batch)?;
    if groups.len() != opening_batch.num_groups()
        || group_ring_multiplier_points.len() != opening_batch.num_groups()
        || group_challenges.len() != opening_batch.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    let n_d_active = lp.n_d_active_for(relation_matrix_row_layout);
    let num_rows =
        lp.relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)?;
    let d_start = num_rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let rhs_layout =
        akita_types::relation_rhs_layout_for(lp, opening_batch, relation_matrix_row_layout)?;
    let expected_y_len = akita_types::relation_rhs_coeff_len(role_dims, &rhs_layout)?;
    if y.coeff_len() != expected_y_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_y_len,
            actual: y.coeff_len(),
        });
    }
    let mut result: Vec<Option<RelationQuotientRow<F>>> = vec![None; num_rows];
    let order = opening_batch.root_group_order()?;

    // The consistency and every A row are native A-role quotients.
    // B and D rows are dispatched independently below.
    let mut y_offset = role_dims.d_a();

    for &group_index in &order {
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
        let blocks_per_claim = group.params.live_fold_count();
        let inner_width = group.params.a_col_len();
        validate_i8_setup_log_basis(log_basis, "for multi-group relation quotient")?;
        if group_layout.num_polynomials() == 0 {
            return Err(AkitaError::InvalidProof);
        }
        let expected_blocks = group_layout
            .num_polynomials()
            .checked_mul(blocks_per_claim)
            .ok_or(AkitaError::InvalidProof)?;
        let opening_ratio = role_dims
            .d_a()
            .checked_div(role_dims.d_d())
            .filter(|ratio| *ratio != 0 && ratio.is_power_of_two())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "opening role dimension must divide the A-role witness width".into(),
                )
            })?;
        let expected_e_planes = expected_blocks
            .checked_mul(num_digits_open)
            .and_then(|n| n.checked_mul(opening_ratio))
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.logical_len() != expected_blocks
            || e_folded.len() != expected_blocks
            || recomposed_inner_rows.len() != expected_blocks
            || group.e_hat.total_planes() != expected_e_planes
            || group.e_hat.digit_stride() != role_dims.d_d()
        {
            return Err(AkitaError::InvalidInput(format!(
                "relation quotient group shape mismatch: challenges={} e_folded={} recomposed={} e_planes={} e_stride={} expected_blocks={} expected_e_planes={} expected_d_d={}",
                challenges.logical_len(),
                e_folded.len(),
                recomposed_inner_rows.len(),
                group.e_hat.total_planes(),
                group.e_hat.digit_stride(),
                expected_blocks,
                expected_e_planes,
                role_dims.d_d(),
            )));
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

        let mut z_segments = group.z_centered.chunks(inner_width);
        let first_z_segment = z_segments.next().ok_or(AkitaError::InvalidProof)?;
        let relation_rows = RingSwitchRelationKernel::relation_rows(
            backend,
            prepared,
            RingSwitchRelationView {
                e_hat: &[],
                t_hat: &[],
                z_segment: first_z_segment,
                z_folded_centered_inf_norm: group.z_inf,
            },
            RingSwitchRelationPlan {
                n_d: 0,
                n_b: 0,
                n_a,
                log_basis,
            },
        )
        .map_err(|err| AkitaError::InvalidInput(format!("A quotient rows failed: {err:?}")))?;
        if !relation_rows.d_cyclic.is_empty()
            || !relation_rows.b_cyclic.is_empty()
            || relation_rows.a_quotients.len() != n_a
        {
            return Err(AkitaError::InvalidProof);
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
                group.params.fold_position_count(),
                group.params.num_digits_commit(),
                log_basis,
            )?;
            quotient_from_cyclic_and_reduced(&consistency_z_cyclic, &consistency_z_reduced)
        };
        let quotient = parallel_high_half_accumulate::<F, _, D>(challenges, |i| Some(e_folded[i]))?;
        let mut quotient = CyclotomicRing::from_slice(&quotient);
        quotient -= consistency_z_quotient;
        match &mut result[0] {
            Some(row) => {
                if row.ring_dim != D || row.coeffs.len() != D {
                    return Err(AkitaError::InvalidProof);
                }
                for (dst, src) in row.coeffs.iter_mut().zip(quotient.coefficients()) {
                    *dst += *src;
                }
            }
            slot @ None => *slot = Some(RelationQuotientOutput::row_from_ring(quotient)),
        }

        let a_range = lp.a_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
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
            result[row_idx] = Some(RelationQuotientOutput::row_from_ring(
                CyclotomicRing::<F, D>::from_slice(&quotient),
            ));
        }

        y_offset = y_offset
            .checked_add(
                n_a.checked_mul(role_dims.d_a())
                    .ok_or(AkitaError::InvalidProof)?,
            )
            .ok_or(AkitaError::InvalidProof)?;

        let b_range =
            lp.commitment_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
        if b_range.len() != n_b {
            return Err(AkitaError::InvalidProof);
        }
        akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            F,
            role_dims.d_b(),
            |D_B| {
                let (t_hat_rows, remainder) = group.t_hat.digits().as_chunks::<D_B>();
                if !remainder.is_empty() {
                    return Err(AkitaError::InvalidProof);
                }
                let b_rows = RingSwitchRelationKernel::relation_rows(
                    backend,
                    prepared,
                    RingSwitchRelationView {
                        e_hat: &[],
                        t_hat: t_hat_rows,
                        z_segment: &[],
                        z_folded_centered_inf_norm: 0,
                    },
                    RingSwitchRelationPlan {
                        n_d: 0,
                        n_b,
                        n_a: 0,
                        log_basis,
                    },
                )
                .map_err(|err| {
                    AkitaError::InvalidInput(format!("B quotient rows failed: {err:?}"))
                })?;
                if b_rows.b_cyclic.len() != n_b
                    || !b_rows.d_cyclic.is_empty()
                    || !b_rows.a_quotients.is_empty()
                {
                    return Err(AkitaError::InvalidProof);
                }
                for (commit_idx, row_idx) in b_range.clone().enumerate() {
                    let reduced = ring_from_flat_y::<F, D_B>(y, y_offset + commit_idx * D_B)?;
                    result[row_idx] = Some(RelationQuotientOutput::row_from_ring(
                        quotient_from_cyclic_and_reduced(
                            b_rows
                                .b_cyclic
                                .get(commit_idx)
                                .ok_or(AkitaError::InvalidProof)?,
                            &reduced,
                        ),
                    ));
                }
                Ok::<(), AkitaError>(())
            }
        )?;
        y_offset = y_offset
            .checked_add(
                n_b.checked_mul(role_dims.d_b())
                    .ok_or(AkitaError::InvalidProof)?,
            )
            .ok_or(AkitaError::InvalidProof)?;
    }

    if d_start != num_rows - n_d_active {
        return Err(AkitaError::InvalidProof);
    }
    if n_d_active != 0 {
        akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            role_dims.d_d(),
            |D_D| {
                let d_rows = RingSwitchRelationKernel::relation_rows(
                    backend,
                    prepared,
                    RingSwitchRelationView {
                        e_hat: e_hat_concat.typed_planes::<D_D>()?,
                        t_hat: &[],
                        z_segment: &[],
                        z_folded_centered_inf_norm: 0,
                    },
                    RingSwitchRelationPlan {
                        n_d: n_d_active,
                        n_b: 0,
                        n_a: 0,
                        log_basis: lp.log_basis,
                    },
                )
                .map_err(|err| {
                    AkitaError::InvalidInput(format!("D quotient rows failed: {err:?}"))
                })?;
                if d_rows.d_cyclic.len() != n_d_active
                    || !d_rows.b_cyclic.is_empty()
                    || !d_rows.a_quotients.is_empty()
                {
                    return Err(AkitaError::InvalidProof);
                }
                for (d_idx, cyclic) in d_rows.d_cyclic.iter().enumerate() {
                    let row_idx = d_start.checked_add(d_idx).ok_or(AkitaError::InvalidProof)?;
                    let reduced = ring_from_flat_y::<F, D_D>(y, y_offset + d_idx * D_D)?;
                    result[row_idx] = Some(RelationQuotientOutput::row_from_ring(
                        quotient_from_cyclic_and_reduced(cyclic, &reduced),
                    ));
                }
                Ok::<(), AkitaError>(())
            }
        )?;
        y_offset = y_offset
            .checked_add(
                n_d_active
                    .checked_mul(role_dims.d_d())
                    .ok_or(AkitaError::InvalidProof)?,
            )
            .ok_or(AkitaError::InvalidProof)?;
    }
    if y_offset != y.coeff_len() {
        return Err(AkitaError::InvalidProof);
    }
    RelationQuotientOutput::from_slots(result)
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
                let (_, _, fold_high, fold_low) = tensor.factors_for_logical_block(idx).unwrap();
                let challenge = sparse_challenge_as_ring::<D>(fold_high)
                    * sparse_challenge_as_ring::<D>(fold_low);
                add_ring_product_reference_high_half::<D>(&mut expected, &challenge, ring);
            }
        }

        assert_eq!(got, expected);
    }
}
