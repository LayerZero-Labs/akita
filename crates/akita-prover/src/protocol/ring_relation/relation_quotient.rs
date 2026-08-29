use super::*;
use crate::api::commitment::for_each_outer_slice_input;
use crate::backend::RingSwitchRelationView;
use crate::compute::{
    OperationCtx, RingSwitchProveBackend, RingSwitchRelationKernel, RingSwitchRelationPlan,
    RuntimeRingSwitchProveBackend,
};
use crate::protocol::ring_relation::{CompressionSourceId, CompressionWitnessMaterialization};
use crate::protocol::ring_switch::PreparedRingSwitchGroup;
use crate::validation::validate_i8_setup_log_basis;
use akita_types::{
    CommittedGroupParams, OpeningFamily, RelationRowGeometry, RingRelationGroupOpening, RingVec,
};

#[cfg(test)]
std::thread_local! {
    static MULTI_GROUP_QUOTIENT_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_multi_group_quotient_calls() {
    MULTI_GROUP_QUOTIENT_CALLS.set(0);
}

#[cfg(test)]
pub(crate) fn multi_group_quotient_calls() -> usize {
    MULTI_GROUP_QUOTIENT_CALLS.get()
}

#[inline]
fn accumulate_small_signed<F: Field + Ring>(dst: &mut F, value: F, coeff: i64) {
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
fn add_sparse_ring_product_high_half<
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    const D: usize,
>(
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

fn parallel_high_half_accumulate<F, R, const D: usize>(
    challenges: &Challenges,
    ring_fn: R,
) -> Result<Vec<F>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Send + Sync,
    R: Fn(usize) -> Option<CyclotomicRing<F, D>> + Sync,
{
    let total = challenges.len();
    let out = cfg_fold_reduce!(
        0..total,
        || vec![F::zero(); D],
        |mut acc: Vec<F>, i: usize| {
            let Some(ring) = ring_fn(i) else {
                return acc;
            };
            add_sparse_ring_product_high_half::<F, D>(&mut acc, &challenges.as_slice()[i], &ring);
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
pub(crate) struct RelationQuotientRow<F: Field> {
    geometry: RelationRowGeometry,
    coeffs: Vec<F>,
}

/// Relation quotient `r` returned by [`compute_multi_group_relation_quotient`].
///
/// Each row retains the native dimension of its relation family. This is the
/// D-free orchestration boundary between the role-local quotient kernels and
/// the flat recursive witness.
#[derive(Clone)]
pub(crate) struct RelationQuotientOutput<F: Field> {
    rows: Vec<RelationQuotientRow<F>>,
}

impl<F: Field> RelationQuotientOutput<F> {
    fn from_slots(slots: Vec<Option<RelationQuotientRow<F>>>) -> Result<Self, AkitaError> {
        let mut rows = Vec::with_capacity(slots.len());
        for (index, row) in slots.into_iter().enumerate() {
            rows.push(row.ok_or_else(|| {
                AkitaError::InvalidInput(format!("relation quotient row {index} was not built"))
            })?);
        }
        Ok(Self { rows })
    }

    fn row_from_ring<const D: usize>(
        ring: CyclotomicRing<F, D>,
    ) -> Result<RelationQuotientRow<F>, AkitaError> {
        Ok(RelationQuotientRow {
            geometry: RelationRowGeometry::native(D)?,
            coeffs: ring.coefficients().to_vec(),
        })
    }

    fn from_physical_coordinates(
        geometry: RelationRowGeometry,
        coeffs: Vec<F>,
    ) -> Result<RelationQuotientRow<F>, AkitaError> {
        if coeffs.len() != geometry.physical_coefficient_width() {
            return Err(AkitaError::InvalidSize {
                expected: geometry.physical_coefficient_width(),
                actual: coeffs.len(),
            });
        }
        Ok(RelationQuotientRow { geometry, coeffs })
    }

    pub(crate) fn rows(&self) -> &[RelationQuotientRow<F>] {
        &self.rows
    }
}

impl<F: Field> RelationQuotientRow<F> {
    pub(crate) fn geometry(&self) -> RelationRowGeometry {
        self.geometry
    }

    pub(crate) fn coeffs(&self) -> &[F] {
        &self.coeffs
    }
}

fn ring_from_flat_y<F: Field, const D: usize>(
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

pub(super) fn quotient_from_cyclic_and_reduced<F: Field, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    reduced: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc_c = cyclic.coefficients();
    let red_c = reduced.coefficients();
    let quotient = std::array::from_fn(|k| (cyc_c[k] - red_c[k]).half());
    CyclotomicRing::from_coefficients(quotient)
}

fn centered_i32_ring<F: Field + Ring, const D: usize>(coeffs: &[i32; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(coeffs[idx] as i64)))
}

fn consistency_z_product_high_half<F, const D: usize>(
    ring_multiplier_point: &RingMultiplierOpeningPoint<F>,
    z_folded_centered: &[[i32; D]],
    num_positions_per_block: usize,
    depth_commit: usize,
    log_basis: u32,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
{
    let inner_width = num_positions_per_block
        .checked_mul(depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("z inner width overflow".to_string()))?;
    if inner_width == 0 || z_folded_centered.len() != inner_width {
        return Err(AkitaError::InvalidInput(format!(
            "ring-multiplier z layout mismatch: z_folded_len={} num_positions_per_block={} depth_commit={} expected={}",
            z_folded_centered.len(),
            num_positions_per_block,
            depth_commit,
            inner_width
        )));
    }
    let g_commit = gadget_row_scalars::<F>(depth_commit, log_basis);
    if ring_multiplier_point.position_len() < num_positions_per_block {
        return Err(AkitaError::InvalidInput(format!(
            "ring-multiplier a length mismatch: actual={} expected_at_least={num_positions_per_block}",
            ring_multiplier_point.position_len()
        )));
    }
    let mut high_half = [F::zero(); D];
    for block_idx in 0..num_positions_per_block {
        let mut z_block = CyclotomicRing::<F, D>::zero();
        for (digit_idx, &g) in g_commit.iter().enumerate() {
            let z_idx = block_idx * depth_commit + digit_idx;
            z_block += centered_i32_ring::<F, D>(&z_folded_centered[z_idx]).scale(&g);
        }
        ring_multiplier_point.accumulate_position_product_high_half(
            block_idx,
            &z_block,
            &mut high_half,
        )?;
    }
    Ok(CyclotomicRing::from_coefficients(high_half))
}

fn compute_group_a_relation_quotients<F, B, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, B>,
    group: &PreparedRingSwitchGroup<F>,
    group_opening: &RingRelationGroupOpening<F>,
) -> Result<(RelationQuotientRow<F>, Vec<RelationQuotientRow<F>>), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    B: RingSwitchProveBackend<F, D>,
{
    if group.role_dims.d_a() != D {
        return Err(AkitaError::InvalidSize {
            expected: group.role_dims.d_a(),
            actual: D,
        });
    }
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    let n_a = group.params.a_rows_len();
    let inner_width = group.params.a_col_len();
    let log_basis_outer = group.params.log_basis_outer();
    let log_basis_open = group.params.log_basis_open();
    let challenges = group_opening.ambient_a_challenges();
    let recomposed_inner_rows = group.recomposed_inner_rows.as_ring_slice::<D>()?;
    let (z_centered, z_remainder) = group.z_centered.as_chunks::<D>();
    if !z_remainder.is_empty() || z_centered.len() != inner_width {
        return Err(AkitaError::InvalidProof);
    }

    let relation_rows = RingSwitchRelationKernel::relation_rows(
        backend,
        prepared,
        RingSwitchRelationView {
            e_hat: &[],
            t_hat: &[],
            z_segment: z_centered,
            z_folded_centered_inf_norm: group.z_inf,
        },
        RingSwitchRelationPlan {
            n_d: 0,
            n_b: 0,
            n_a,
            log_basis_open,
            log_basis_outer,
        },
    )
    .map_err(|err| AkitaError::InvalidInput(format!("A quotient rows failed: {err:?}")))?;
    if !relation_rows.d_negacyclic.is_empty()
        || !relation_rows.d_cyclic.is_empty()
        || !relation_rows.b_cyclic.is_empty()
        || relation_rows.a_quotients.len() != n_a
    {
        return Err(AkitaError::InvalidProof);
    }
    let a_quotients = relation_rows.a_quotients;

    let consistency_quotient = match &group.folded_opening {
        OpeningFamily::EvaluationTrace(e_folded)
            if group_opening.coefficient_packing_geometry().is_none() =>
        {
            let ring_multiplier_point = group_opening.evaluation_trace_multiplier_point()?;
            let e_folded = e_folded.as_ring_slice::<D>()?;
            let consistency_z_quotient = if ring_multiplier_point.is_constant() {
                CyclotomicRing::<F, D>::zero()
            } else {
                consistency_z_product_high_half::<F, D>(
                    ring_multiplier_point,
                    z_centered,
                    group.params.num_positions_per_block(),
                    group.params.num_digits_inner(),
                    group.params.log_basis_inner(),
                )?
            };
            let quotient =
                parallel_high_half_accumulate::<F, _, D>(challenges, |i| e_folded.get(i).copied())?;
            let mut consistency_quotient = CyclotomicRing::from_slice(&quotient);
            consistency_quotient -= consistency_z_quotient;
            RelationQuotientOutput::row_from_ring(consistency_quotient)?
        }
        OpeningFamily::SubringCoefficientPacking(product)
            if Some(product.geometry()) == group_opening.coefficient_packing_geometry() =>
        {
            let geometry = product.geometry();
            RelationQuotientOutput::from_physical_coordinates(
                RelationRowGeometry::new(
                    geometry.challenge_subring_dimension(),
                    geometry.extension_degree(),
                )?,
                product.quotient_high_half_base_field_coordinates().to_vec(),
            )?
        }
        _ => {
            return Err(AkitaError::InvalidSetup(
                "relation quotient opening method and witness disagree".into(),
            ));
        }
    };

    let num_live_blocks_per_claim = group.params.num_live_blocks();
    let mut a_rows = Vec::with_capacity(n_a);
    for (a_idx, a_q) in a_quotients.iter().enumerate() {
        let mut quotient = parallel_high_half_accumulate::<F, _, D>(challenges, |i| {
            let claim_idx = i / num_live_blocks_per_claim;
            let block_idx = i % num_live_blocks_per_claim;
            let inner_idx = claim_idx * num_live_blocks_per_claim + block_idx;
            recomposed_inner_rows
                .get(inner_idx.checked_mul(n_a)?.checked_add(a_idx)?)
                .copied()
        })?;
        for (dst, src) in quotient.iter_mut().zip(a_q.coefficients()) {
            *dst -= *src;
        }
        a_rows.push(RelationQuotientOutput::row_from_ring(
            CyclotomicRing::<F, D>::from_slice(&quotient),
        )?);
    }
    Ok((consistency_quotient, a_rows))
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_multi_group_relation_quotient")]
pub(crate) fn compute_multi_group_relation_quotient<F, B>(
    ring_switch_ctx: &OperationCtx<'_, F, B>,
    lp: &CommittedGroupParams,
    opening_batch: &akita_types::OpeningClaimsLayout,
    groups: &[PreparedRingSwitchGroup<F>],
    group_openings: &[RingRelationGroupOpening<F>],
    extension_degree: usize,
    d_quotients: &RingVec<F>,
    y: &RingVec<F>,
    compression: Option<&CompressionWitnessMaterialization<F>>,
) -> Result<RelationQuotientOutput<F>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    B: RuntimeRingSwitchProveBackend<F>,
{
    #[cfg(test)]
    MULTI_GROUP_QUOTIENT_CALLS.with(|calls| calls.set(calls.get() + 1));

    lp.validate_opening_batch(opening_batch)?;
    if groups.len() != opening_batch.num_groups()
        || group_openings.len() != opening_batch.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    let relation_geometry =
        akita_types::RelationWitnessGeometry::for_level(lp, opening_batch, extension_degree)?;
    let rhs_layout = relation_geometry.rhs_layout();
    let row_families = rhs_layout.row_families()?;
    let num_rows = row_families.len();
    let n_d_active = lp.open().matrix.output_rank();
    let d_start = row_families
        .iter()
        .position(|row| matches!(row, akita_types::RelationRowFamily::Opening { .. }))
        .ok_or(AkitaError::InvalidProof)?;
    let expected_y_len = akita_types::relation_rhs_coeff_len(rhs_layout)?;
    if y.coeff_len() != expected_y_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_y_len,
            actual: y.coeff_len(),
        });
    }
    let ordinary_rhs_len = row_families
        .iter()
        .take_while(|row| {
            !matches!(
                row,
                akita_types::RelationRowFamily::CompressionF { .. }
                    | akita_types::RelationRowFamily::CompressionH { .. }
            )
        })
        .try_fold(0usize, |length, row| {
            length
                .checked_add(row.geometry().physical_coefficient_width())
                .ok_or(AkitaError::InvalidProof)
        })?;
    if compression.is_some()
        && y.coeffs()
            .get(..ordinary_rhs_len)
            .ok_or(AkitaError::InvalidProof)?
            .iter()
            .any(|coefficient| !coefficient.is_zero())
    {
        return Err(AkitaError::InvalidProof);
    }
    let mut result: Vec<Option<RelationQuotientRow<F>>> = vec![None; num_rows];
    let order = opening_batch.root_group_order()?;
    if order.len() != rhs_layout.groups.len() {
        return Err(AkitaError::InvalidProof);
    }

    // Every group owns a native consistency/A/B block. The shared D tail is
    // level-owned and follows all group blocks.
    let mut y_offset = 0usize;

    for (&group_index, group_rows) in order.iter().zip(&rhs_layout.groups) {
        let group_dims = group_rows.role_dims;
        let group = groups.get(group_index).ok_or(AkitaError::InvalidProof)?;
        if group.role_dims != group_dims {
            return Err(AkitaError::InvalidProof);
        }
        let consistency_row = lp.consistency_row_index(opening_batch, group_index)?;
        y_offset = y_offset
            .checked_add(group_rows.opening_geometry.physical_coefficient_width())
            .ok_or(AkitaError::InvalidProof)?;
        let group_opening = group_openings
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let challenges = group_opening.ambient_a_challenges();
        let group_layout = opening_batch.group_layout(group_index)?;
        let log_basis_outer = group.params.log_basis_outer();
        let log_basis_open = group.params.log_basis_open();
        let num_digits_outer = group.params.num_digits_outer();
        let num_digits_open = group.params.num_digits_open();
        let n_a = group.params.a_rows_len();
        let physical_n_b = group.params.b_rows_len();
        let n_b = group.params.logical_b_rows_len()?;
        let num_live_blocks_per_claim = group.params.num_live_blocks();
        let inner_width = group.params.a_col_len();
        validate_i8_setup_log_basis(log_basis_outer, "for multi-group relation quotient")?;
        validate_i8_setup_log_basis(log_basis_open, "for multi-group relation quotient")?;
        if group_layout.num_polynomials() == 0 {
            return Err(AkitaError::InvalidProof);
        }
        let expected_blocks = group_layout
            .num_polynomials()
            .checked_mul(num_live_blocks_per_claim)
            .ok_or(AkitaError::InvalidProof)?;
        let opening_width = group_rows.opening_geometry.physical_coefficient_width();
        let opening_ratio = opening_width
            .checked_div(group_dims.d_d())
            .filter(|ratio| *ratio != 0 && ratio.is_power_of_two())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "current A-width relation witness cannot carry the opening role".into(),
                )
            })?;
        let expected_e_planes = expected_blocks
            .checked_mul(num_digits_open)
            .and_then(|n| n.checked_mul(opening_ratio))
            .ok_or(AkitaError::InvalidProof)?;
        let expected_e_coeffs = expected_blocks
            .checked_mul(opening_width)
            .ok_or(AkitaError::InvalidProof)?;
        let expected_z_coeffs = inner_width
            .checked_mul(group_dims.d_a())
            .ok_or(AkitaError::InvalidProof)?;
        let expected_recomposed_coeffs = n_a
            .checked_mul(group_dims.d_a())
            .ok_or(AkitaError::InvalidProof)?;
        let folded_opening_is_valid = match &group.folded_opening {
            OpeningFamily::EvaluationTrace(e_folded)
                if group_opening.coefficient_packing_geometry().is_none() =>
            {
                e_folded.coeff_len() == expected_e_coeffs
            }
            OpeningFamily::SubringCoefficientPacking(product) => {
                Some(product.geometry()) == group_opening.coefficient_packing_geometry()
                    && product.reduced_base_field_coordinates().len() == opening_width
                    && product.quotient_high_half_base_field_coordinates().len() == opening_width
            }
            _ => false,
        };
        if challenges.len() != expected_blocks
            || !folded_opening_is_valid
            || group.e_hat.total_planes() != expected_e_planes
            || group.e_hat.digit_stride() != group_dims.d_d()
        {
            return Err(AkitaError::InvalidInput(format!(
                "relation quotient group shape mismatch: challenges={} recomposed={} e_planes={} e_stride={} expected_blocks={} expected_e_planes={} expected_d_d={}",
                challenges.len(),
                group.recomposed_inner_rows.coeff_len() / expected_recomposed_coeffs,
                group.e_hat.total_planes(),
                group.e_hat.digit_stride(),
                expected_blocks,
                expected_e_planes,
                group_dims.d_d(),
            )));
        }
        if group.z_centered.len() != expected_z_coeffs
            || group.recomposed_inner_rows.coeff_len()
                != expected_blocks
                    .checked_mul(expected_recomposed_coeffs)
                    .ok_or(AkitaError::InvalidProof)?
        {
            return Err(AkitaError::InvalidProof);
        }
        let outer_ratio = group_dims
            .d_a()
            .checked_div(group_dims.d_b())
            .filter(|ratio| *ratio != 0 && ratio.is_power_of_two())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "B-role ring dimension must divide the A-role ring dimension".into(),
                )
            })?;
        let expected_t_hat_block_digits = n_a
            .checked_mul(outer_ratio)
            .and_then(|n| n.checked_mul(num_digits_outer))
            .ok_or(AkitaError::InvalidProof)?;
        if group.t_hat.block_count() != expected_blocks
            || group.t_hat.digit_stride() != group_dims.d_b()
            || group
                .t_hat
                .block_sizes()
                .iter()
                .any(|&size| size != expected_t_hat_block_digits)
        {
            return Err(AkitaError::InvalidProof);
        }
        let slice_geometry = akita_types::CommitmentSliceGeometry::try_new(
            group.params.outer_slice_count(),
            num_live_blocks_per_claim,
            group_layout.num_polynomials(),
            n_a,
            num_digits_outer,
            group_dims.d_a(),
            group_dims.d_b(),
        )?;

        let (consistency_quotient, a_quotients) = akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_dims.d_a(),
            |D_A| {
                compute_group_a_relation_quotients::<F, B, D_A>(
                    ring_switch_ctx,
                    group,
                    group_opening,
                )
            }
        )?;
        if result
            .get(consistency_row)
            .ok_or(AkitaError::InvalidProof)?
            .is_some()
        {
            return Err(AkitaError::InvalidProof);
        }
        result[consistency_row] = Some(consistency_quotient);

        let a_range = lp.a_row_range(opening_batch, group_index)?;
        if a_range.len() != n_a || a_quotients.len() != n_a {
            return Err(AkitaError::InvalidProof);
        }
        for (row_idx, quotient) in a_range.zip(a_quotients) {
            result[row_idx] = Some(quotient);
        }

        y_offset = y_offset
            .checked_add(
                n_a.checked_mul(group_dims.d_a())
                    .ok_or(AkitaError::InvalidProof)?,
            )
            .ok_or(AkitaError::InvalidProof)?;

        let b_range = lp.commitment_row_range(opening_batch, group_index)?;
        if b_range.len() != n_b {
            return Err(AkitaError::InvalidProof);
        }
        let b_coeff_len = n_b
            .checked_mul(group_dims.d_b())
            .ok_or(AkitaError::InvalidProof)?;
        let b_end = y_offset
            .checked_add(b_coeff_len)
            .ok_or(AkitaError::InvalidProof)?;
        let recomposed_b = if let Some(compression) = compression {
            RingVec::from_coeffs(
                compression
                    .source(CompressionSourceId::Outer { group_index })?
                    .witness
                    .stages()
                    .first()
                    .ok_or(AkitaError::InvalidProof)?
                    .recompose::<F>()?,
            )
        } else {
            RingVec::from_coeffs(
                y.coeffs()
                    .get(y_offset..b_end)
                    .ok_or(AkitaError::InvalidProof)?
                    .to_vec(),
            )
        };
        akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            F,
            group_dims.d_b(),
            |D_B| {
                let t_hat_planes = group.t_hat.typed_planes::<D_B>()?;
                let planes_per_claim = num_live_blocks_per_claim
                    .checked_mul(expected_t_hat_block_digits)
                    .filter(|count| *count != 0)
                    .ok_or(AkitaError::InvalidProof)?;
                let mut b_cyclic = Vec::with_capacity(n_b);
                for_each_outer_slice_input::<D_B>(
                    t_hat_planes.chunks(planes_per_claim),
                    &slice_geometry,
                    |slice_input| {
                        let b_rows = RingSwitchRelationKernel::relation_rows(
                            backend,
                            prepared,
                            RingSwitchRelationView {
                                e_hat: &[],
                                t_hat: slice_input,
                                z_segment: &[],
                                z_folded_centered_inf_norm: 0,
                            },
                            RingSwitchRelationPlan {
                                n_d: 0,
                                n_b: physical_n_b,
                                n_a: 0,
                                log_basis_open,
                                log_basis_outer,
                            },
                        )
                        .map_err(|err| {
                            AkitaError::InvalidInput(format!("B quotient rows failed: {err:?}"))
                        })?;
                        if b_rows.b_cyclic.len() != physical_n_b
                            || !b_rows.d_negacyclic.is_empty()
                            || !b_rows.d_cyclic.is_empty()
                            || !b_rows.a_quotients.is_empty()
                        {
                            return Err(AkitaError::InvalidProof);
                        }
                        b_cyclic.extend(b_rows.b_cyclic);
                        Ok(())
                    },
                )?;
                if b_cyclic.len() != n_b {
                    return Err(AkitaError::InvalidProof);
                }
                for (commit_idx, row_idx) in b_range.clone().enumerate() {
                    let reduced = ring_from_flat_y::<F, D_B>(&recomposed_b, commit_idx * D_B)?;
                    result[row_idx] = Some(RelationQuotientOutput::row_from_ring(
                        quotient_from_cyclic_and_reduced(
                            b_cyclic.get(commit_idx).ok_or(AkitaError::InvalidProof)?,
                            &reduced,
                        ),
                    )?);
                }
                Ok::<(), AkitaError>(())
            }
        )?;
        y_offset = b_end;
    }

    if n_d_active != 0 {
        let d_coeff_len = n_d_active
            .checked_mul(rhs_layout.d_ring_dimension)
            .ok_or(AkitaError::InvalidProof)?;
        let d_end = y_offset
            .checked_add(d_coeff_len)
            .ok_or(AkitaError::InvalidProof)?;
        akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            rhs_layout.d_ring_dimension,
            |D_D| {
                let d_rows = d_quotients.as_ring_slice::<D_D>()?;
                if d_rows.len() != n_d_active {
                    return Err(AkitaError::InvalidProof);
                }
                for (d_idx, quotient) in d_rows.iter().enumerate() {
                    let row_idx = d_start.checked_add(d_idx).ok_or(AkitaError::InvalidProof)?;
                    result[row_idx] = Some(RelationQuotientOutput::row_from_ring(*quotient)?);
                }
                Ok::<(), AkitaError>(())
            }
        )?;
        y_offset = d_end;
    }
    for (row_index, family) in row_families.iter().enumerate() {
        let (source, map_index, geometry) = match *family {
            akita_types::RelationRowFamily::CompressionF {
                group_index,
                map_index,
                geometry,
            } => (
                CompressionSourceId::Outer { group_index },
                map_index,
                geometry,
            ),
            akita_types::RelationRowFamily::CompressionH {
                map_index,
                geometry,
            } => (CompressionSourceId::Opening, map_index, geometry),
            _ => continue,
        };
        if geometry.coordinate_plane_count() != 1 {
            return Err(AkitaError::InvalidSetup(
                "compression quotient requires one native coordinate plane".into(),
            ));
        }
        let ring_dim = geometry.polynomial_modulus_dimension();
        let compression = compression.ok_or(AkitaError::InvalidProof)?;
        let quotient = compression.source(source)?.quotient(map_index)?;
        if quotient.ring_dim() != ring_dim || quotient.coeff_len() != ring_dim {
            return Err(AkitaError::InvalidSize {
                expected: ring_dim,
                actual: quotient.coeff_len(),
            });
        }
        result[row_index] = Some(RelationQuotientRow {
            geometry,
            coeffs: quotient.coeffs().to_vec(),
        });
        let rhs_end = y_offset
            .checked_add(ring_dim)
            .ok_or(AkitaError::InvalidProof)?;
        let rhs_row = y
            .coeffs()
            .get(y_offset..rhs_end)
            .ok_or(AkitaError::InvalidProof)?;
        let source_witness = compression.source(source)?;
        if map_index + 1 == akita_types::COMPRESSION_MAP_COUNT {
            if rhs_row != source_witness.terminal.coefficients() {
                return Err(AkitaError::InvalidProof);
            }
        } else if rhs_row.iter().any(|coefficient| !coefficient.is_zero()) {
            return Err(AkitaError::InvalidProof);
        }
        y_offset = rhs_end;
    }
    if y_offset != y.coeff_len() {
        return Err(AkitaError::InvalidProof);
    }
    RelationQuotientOutput::from_slots(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallenge;
    use jolt_field::Prime128OffsetA7F7 as F;
    use jolt_field::Zero;

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
    fn sparse_high_half_streaming_matches_ring_multiplication_reference() {
        const D: usize = 8;
        let sparse = vec![
            SparseChallenge {
                positions: vec![0, 7].into(),
                coeffs: vec![1, -1].into(),
            },
            SparseChallenge {
                positions: vec![2, 4].into(),
                coeffs: vec![1, 2].into(),
            },
            SparseChallenge {
                positions: vec![1].into(),
                coeffs: vec![-1].into(),
            },
            SparseChallenge {
                positions: vec![3, 6].into(),
                coeffs: vec![1, 1].into(),
            },
        ];
        let rings = (0..sparse.len())
            .map(|idx| (idx != 3).then(|| ring::<D>(10 * idx as u64)))
            .collect::<Vec<_>>();
        let challenges = Challenges::from_sparse(sparse.clone(), sparse.len(), 1).unwrap();

        let got = parallel_high_half_accumulate::<F, _, D>(&challenges, |idx| rings[idx]).unwrap();
        let mut expected = vec![F::zero(); D];
        for (idx, ring) in rings.iter().enumerate() {
            if let Some(ring) = ring {
                let challenge = sparse_challenge_as_ring::<D>(&sparse[idx]);
                add_ring_product_reference_high_half::<D>(&mut expected, &challenge, ring);
            }
        }

        assert_eq!(got, expected);
    }

    #[test]
    fn physical_quotient_row_preserves_packing_planes_and_rejects_bad_width() {
        let geometry = RelationRowGeometry::new(64, 2).unwrap();
        let coordinates = (0..128)
            .map(|index| F::from_u64(index as u64 + 1))
            .collect::<Vec<_>>();
        let row = RelationQuotientOutput::from_physical_coordinates(geometry, coordinates.clone())
            .unwrap();
        assert_eq!(row.geometry(), geometry);
        assert_eq!(row.coeffs(), coordinates);
        assert!(
            RelationQuotientOutput::from_physical_coordinates(geometry, vec![F::zero(); 64],)
                .is_err()
        );
    }
}
