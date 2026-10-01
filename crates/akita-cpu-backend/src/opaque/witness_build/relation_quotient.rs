//! Consumer-owned construction of the ring-relation quotient witness R.

use super::compression_witness::{CompressionSourceId, CompressionWitnessMaterialization};
use super::finalize::PreparedRingSwitchGroup;
use crate::commitment::for_each_outer_slice_input;
use crate::opaque::RelationQuotientRow;
use crate::opaque::RingSwitchRelationView;
use crate::opaque::{
    OperationCtx, RingSwitchProveBackend, RingSwitchRelationKernel, RingSwitchRelationPlan,
    RuntimeRingSwitchProveBackend,
};
use crate::validation::validate_i8_setup_log_basis;
use akita_algebra::CyclotomicRing;
use akita_challenges::{Challenges, SparseChallenge};
use akita_error::AkitaError;
use akita_params::{CommittedGroupParams, RelationRowGeometry};
use akita_types::{OpeningFamily, RingRelationGroupOpening, RingVec};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field, Ring};

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

/// Batch block-major A-row products and an optional consistency product in
/// one challenge traversal. Per-lane scratch is bounded by `(rank + 1) * D`.
fn parallel_high_half_accumulate_a_rows<F, const D: usize>(
    challenges: &Challenges,
    relation_rows: &[CyclotomicRing<F, D>],
    row_rank: usize,
    consistency_rows: Option<&[CyclotomicRing<F, D>]>,
) -> Result<Vec<[F; D]>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring + Send + Sync,
{
    let challenge_count = challenges.len();
    let expected_relation_rows = challenge_count
        .checked_mul(row_rank)
        .ok_or(AkitaError::InvalidProof)?;
    if row_rank == 0
        || relation_rows.len() != expected_relation_rows
        || consistency_rows.is_some_and(|rows| rows.len() < challenge_count)
        || challenges.as_slice().iter().any(|challenge| {
            challenge.positions.len() != challenge.coeffs.len()
                || challenge
                    .positions
                    .iter()
                    .any(|&position| position as usize >= D)
        })
    {
        return Err(AkitaError::InvalidProof);
    }
    let output_rows = row_rank
        .checked_add(usize::from(consistency_rows.is_some()))
        .ok_or(AkitaError::InvalidProof)?;
    let output = cfg_fold_reduce!(
        0..challenge_count,
        || Ok::<_, AkitaError>(vec![[F::zero(); D]; output_rows]),
        |acc: Result<Vec<[F; D]>, AkitaError>, challenge_index| {
            acc.and_then(|mut acc| {
                let start = challenge_index
                    .checked_mul(row_rank)
                    .ok_or(AkitaError::InvalidProof)?;
                let end = start
                    .checked_add(row_rank)
                    .ok_or(AkitaError::InvalidProof)?;
                let rows = relation_rows
                    .get(start..end)
                    .ok_or(AkitaError::InvalidProof)?;
                let challenge = challenges
                    .as_slice()
                    .get(challenge_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let consistency = consistency_rows.and_then(|rows| rows.get(challenge_index));
                add_sparse_ring_products_high_half::<F, D>(&mut acc, challenge, rows, consistency);
                Ok(acc)
            })
        },
        |left: Result<Vec<[F; D]>, AkitaError>, right: Result<Vec<[F; D]>, AkitaError>| {
            match (left, right) {
                (Ok(mut left), Ok(right)) => {
                    for (left_row, right_row) in left.iter_mut().zip(right) {
                        for (left, right) in left_row.iter_mut().zip(right_row) {
                            *left += right;
                        }
                    }
                    Ok(left)
                }
                (Err(error), _) => Err(error),
                (_, Err(error)) => Err(error),
            }
        }
    )?;
    Ok(output)
}

/// Apply one sparse challenge to several rings while walking its positions and
/// nonzero coefficients only once.
#[inline(always)]
fn add_sparse_ring_products_high_half<F: Field + Ring, const D: usize>(
    outputs: &mut [[F; D]],
    challenge: &SparseChallenge,
    relation_rows: &[CyclotomicRing<F, D>],
    consistency: Option<&CyclotomicRing<F, D>>,
) {
    for (&position, &coefficient) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let position = position as usize;
        let coefficient = i64::from(coefficient);
        for (output, ring) in outputs
            .iter_mut()
            .zip(consistency.into_iter().chain(relation_rows))
        {
            if let (Some(destinations), Some(sources)) = (
                output.get_mut(..position),
                ring.coefficients().get(D - position..),
            ) {
                for (destination, &value) in destinations.iter_mut().zip(sources) {
                    accumulate_small_signed(destination, value, coefficient);
                }
            }
        }
    }
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
        RelationQuotientRow::new(
            RelationRowGeometry::native(D)?,
            ring.coefficients().to_vec(),
        )
    }

    fn from_physical_coordinates(
        geometry: RelationRowGeometry,
        coeffs: Vec<F>,
    ) -> Result<RelationQuotientRow<F>, AkitaError> {
        RelationQuotientRow::new(geometry, coeffs)
    }

    pub(crate) fn rows(&self) -> &[RelationQuotientRow<F>] {
        &self.rows
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

fn compute_group_a_relation_quotients<F, O, const D: usize>(
    opening_ctx: &OperationCtx<'_, F, O>,
    group: &PreparedRingSwitchGroup<F>,
    group_opening: &RingRelationGroupOpening<F>,
) -> Result<(RelationQuotientRow<F>, Vec<RelationQuotientRow<F>>), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    O: RingSwitchProveBackend<F, D>
        + crate::opaque::FoldRelationKernel<crate::opaque::CpuAcceptedFold<F>, F, D>,
{
    if group.role_dims.d_a() != D {
        return Err(AkitaError::InvalidSize {
            expected: group.role_dims.d_a(),
            actual: D,
        });
    }
    let n_a = group.params.a_rows_len();
    let inner_width = group.params.a_col_len();
    let log_basis_outer = group.params.log_basis_outer();
    let log_basis_open = group.params.log_basis_open();
    let challenges = group_opening.ambient_a_challenges();
    let recomposed_inner_rows = group.recomposed_inner_rows.as_ring_slice::<D>()?;
    if group.fold.response_coefficient_len() != inner_width * D {
        return Err(AkitaError::InvalidProof);
    }
    let (evaluation_trace, consistency_rows, packing_product) = match &group.folded_opening {
        OpeningFamily::EvaluationTrace(e_folded)
            if group_opening.coefficient_packing_geometry().is_none() =>
        {
            (
                Some((
                    group_opening.evaluation_trace_multiplier_point()?,
                    group.params.num_positions_per_block(),
                    group.params.num_digits_inner(),
                    group.params.log_basis_inner(),
                )),
                Some(e_folded.as_ring_slice::<D>()?),
                None,
            )
        }
        OpeningFamily::SubringCoefficientPacking(product)
            if Some(product.geometry()) == group_opening.coefficient_packing_geometry() =>
        {
            (None, None, Some(product))
        }
        _ => {
            return Err(AkitaError::InvalidSetup(
                "relation quotient opening method and witness disagree".into(),
            ))
        }
    };
    let plan = crate::opaque::ValidatedFoldRelationPlan::new(
        group.params.opening_method(),
        n_a,
        log_basis_open,
        log_basis_outer,
        evaluation_trace,
    )?;
    let derived = group
        .fold
        .relation::<O, D>(opening_ctx, &plan)
        .map_err(|err| AkitaError::InvalidInput(format!("A quotient rows failed: {err:?}")))?;
    let (a_quotients, z_consistency) = match derived {
        crate::opaque::FoldRelationOutput::EvaluationTrace {
            a_quotients,
            z_consistency_high_half,
        } => (a_quotients, Some(z_consistency_high_half)),
        crate::opaque::FoldRelationOutput::SubringCoefficientPacking { a_quotients } => {
            (a_quotients, None)
        }
    };

    if a_quotients.len() != n_a {
        return Err(AkitaError::InvalidProof);
    }
    let products = parallel_high_half_accumulate_a_rows::<F, D>(
        challenges,
        recomposed_inner_rows,
        n_a,
        consistency_rows,
    )?;
    let consistency_quotient = match packing_product {
        None => {
            let [consistency_product, ..] = products.as_slice() else {
                return Err(AkitaError::InvalidProof);
            };
            let mut coefficients = *consistency_product;
            for (destination, &source) in coefficients.iter_mut().zip(
                z_consistency
                    .as_ref()
                    .ok_or(AkitaError::InvalidProof)?
                    .coefficients(),
            ) {
                *destination -= source;
            }
            RelationQuotientOutput::row_from_ring(CyclotomicRing::from_coefficients(coefficients))?
        }
        Some(product) => {
            if z_consistency.is_some() {
                return Err(AkitaError::InvalidProof);
            }
            let geometry = product.geometry();
            RelationQuotientOutput::from_physical_coordinates(
                RelationRowGeometry::new(
                    geometry.challenge_subring_dimension(),
                    geometry.extension_degree(),
                )?,
                product.quotient_high_half_base_field_coordinates().to_vec(),
            )?
        }
    };

    let a_product_offset = usize::from(consistency_rows.is_some());
    let mut a_rows = Vec::with_capacity(n_a);
    for (a_idx, a_q) in a_quotients.iter().enumerate() {
        let product_index = a_product_offset
            .checked_add(a_idx)
            .ok_or(AkitaError::InvalidProof)?;
        let mut coefficients = *products
            .get(product_index)
            .ok_or(AkitaError::InvalidProof)?;
        for (destination, &source) in coefficients.iter_mut().zip(a_q.coefficients()) {
            *destination -= source;
        }
        a_rows.push(RelationQuotientOutput::row_from_ring(
            CyclotomicRing::<F, D>::from_coefficients(coefficients),
        )?);
    }
    Ok((consistency_quotient, a_rows))
}

#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_multi_group_relation_quotient")]
pub(crate) fn compute_multi_group_relation_quotient<F, O, B>(
    opening_ctx: &OperationCtx<'_, F, O>,
    ring_switch_ctx: &OperationCtx<'_, F, B>,
    lp: &CommittedGroupParams,
    opening_batch: &akita_params::OpeningClaimsLayout,
    groups: &[PreparedRingSwitchGroup<F>],
    group_openings: &[RingRelationGroupOpening<F>],
    extension_degree: usize,
    d_quotients: &RingVec<F>,
    y: &RingVec<F>,
    compression: Option<&CompressionWitnessMaterialization<F>>,
) -> Result<RelationQuotientOutput<F>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    O: RuntimeRingSwitchProveBackend<F>
        + crate::opaque::RuntimeFoldRelationBackend<F>
        + crate::opaque::FoldHandleBackend<F, AcceptedFold = crate::opaque::CpuAcceptedFold<F>>,
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
        akita_params::RelationWitnessGeometry::for_level(lp, opening_batch, extension_degree)?;
    let rhs_layout = relation_geometry.rhs_layout();
    let row_families = rhs_layout.row_families()?;
    let num_rows = row_families.len();
    let n_d_active = lp.open().matrix.output_rank();
    let d_start = row_families
        .iter()
        .position(|row| matches!(row, akita_params::RelationRowFamily::Opening { .. }))
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
                akita_params::RelationRowFamily::CompressionF { .. }
                    | akita_params::RelationRowFamily::CompressionH { .. }
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
        if group.fold.response_coefficient_len() != expected_z_coeffs
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
        let slice_geometry = akita_params::CommitmentSliceGeometry::try_new(
            group.params.outer_slice_count(),
            num_live_blocks_per_claim,
            group_layout.num_polynomials(),
            n_a,
            num_digits_outer,
            group_dims.d_a(),
            group_dims.d_b(),
        )?;

        let (consistency_quotient, a_quotients) = akita_params::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_dims.d_a(),
            |D_A| {
                compute_group_a_relation_quotients::<F, O, D_A>(opening_ctx, group, group_opening)
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
                    .witness()
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
        akita_params::dispatch_for_field!(
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
        akita_params::dispatch_for_field!(
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
            akita_params::RelationRowFamily::CompressionF {
                group_index,
                map_index,
                geometry,
            } => (
                CompressionSourceId::Outer { group_index },
                map_index,
                geometry,
            ),
            akita_params::RelationRowFamily::CompressionH {
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
        result[row_index] = Some(RelationQuotientRow::new(
            geometry,
            quotient.coeffs().to_vec(),
        )?);
        let rhs_end = y_offset
            .checked_add(ring_dim)
            .ok_or(AkitaError::InvalidProof)?;
        let rhs_row = y
            .coeffs()
            .get(y_offset..rhs_end)
            .ok_or(AkitaError::InvalidProof)?;
        let source_witness = compression.source(source)?;
        if map_index + 1 == akita_params::COMPRESSION_MAP_COUNT {
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
#[path = "relation_quotient/tests.rs"]
mod tests;
