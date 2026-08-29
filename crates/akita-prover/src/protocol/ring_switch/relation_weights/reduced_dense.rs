//! Dense prover weights for quotient-free reduced ring relations.
//!
//! This compiler retains every public multiplier in coefficient form until it
//! has passed through the shared negacyclic residue recurrence. It scatters
//! the resulting native kernels directly into the canonical `WitnessLayout`
//! ranges; it never constructs a relation matrix or a quotient-shaped event
//! stream.

use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::residue_kernel;
use akita_challenges::Challenges;
use akita_field::{ExtField, MulBaseUnreduced};
use akita_types::{
    dispatch_for_field, CommitmentSliceGeometry, RelationQuotientLayout,
    RingMultiplierOpeningPoint, RingRelationMode,
};

struct SetupColumnKernels<E> {
    batch_count: usize,
    column_count: usize,
    ring_dimension: usize,
    values: Vec<E>,
}

impl<E: Copy> SetupColumnKernels<E> {
    fn get(&self, batch: usize, column: usize) -> Result<&[E], AkitaError> {
        if batch >= self.batch_count || column >= self.column_count {
            return Err(AkitaError::InvalidProof);
        }
        let start = column
            .checked_mul(self.batch_count)
            .and_then(|offset| offset.checked_add(batch))
            .and_then(|index| index.checked_mul(self.ring_dimension))
            .ok_or(AkitaError::InvalidProof)?;
        self.values
            .get(start..start + self.ring_dimension)
            .ok_or(AkitaError::InvalidProof)
    }
}

fn evaluate_setup_column_kernels<F, E>(
    family: &SetupRows<'_, F>,
    columns: Range<usize>,
    row_weights: &[(usize, Vec<E>)],
    batch_count: usize,
    alpha: E,
) -> Result<SetupColumnKernels<E>, AkitaError>
where
    F: FieldCore,
    E: FieldCore + LiftBase<F>,
{
    if batch_count == 0
        || row_weights
            .iter()
            .any(|(_, weights)| weights.len() != batch_count)
    {
        return Err(AkitaError::InvalidSetup(
            "reduced setup column weight batches are malformed".into(),
        ));
    }
    let column_count = columns.len();
    let output_len = column_count
        .checked_mul(batch_count)
        .and_then(|len| len.checked_mul(family.ring_d))
        .ok_or_else(|| AkitaError::InvalidSetup("reduced setup kernel size overflow".into()))?;
    let mut values = vec![E::zero(); output_len];
    cfg_chunks_mut!(&mut values, batch_count * family.ring_d)
        .enumerate()
        .try_for_each(|(column_offset, output)| -> Result<(), AkitaError> {
            let column = columns
                .start
                .checked_add(column_offset)
                .ok_or_else(|| AkitaError::InvalidSetup("setup column offset overflow".into()))?;
            for (row, weights) in row_weights {
                let kernel = residue_kernel::<F, E>(family.ring_slice(*row, column)?, alpha)?;
                for (batch, &row_weight) in weights.iter().enumerate() {
                    if row_weight.is_zero() {
                        continue;
                    }
                    let destination = output
                        .get_mut(batch * family.ring_d..(batch + 1) * family.ring_d)
                        .ok_or(AkitaError::InvalidProof)?;
                    for (weight, &coefficient) in destination.iter_mut().zip(&kernel) {
                        *weight += row_weight * coefficient;
                    }
                }
            }
            Ok(())
        })?;
    Ok(SetupColumnKernels {
        batch_count,
        column_count,
        ring_dimension: family.ring_d,
        values,
    })
}

fn sparse_challenge_kernel<F, E>(
    challenges: &Challenges,
    index: usize,
    dimension: usize,
    alpha: E,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + LiftBase<F>,
{
    let challenge = challenges
        .as_slice()
        .get(index)
        .ok_or(AkitaError::InvalidProof)?;
    challenge.validate_dyn(dimension)?;
    let mut coefficients = vec![F::zero(); dimension];
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        *coefficients
            .get_mut(position as usize)
            .ok_or(AkitaError::InvalidProof)? = F::from_i64(i64::from(coefficient));
    }
    residue_kernel::<F, E>(&coefficients, alpha)
}

fn position_multiplier_kernels<F, E>(
    point: &RingMultiplierOpeningPoint<F>,
    position_count: usize,
    dimension: usize,
    alpha: E,
) -> Result<Vec<Vec<E>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    if let Some(base) = point.as_base() {
        if base.position_weights.len() != position_count {
            return Err(AkitaError::InvalidProof);
        }
        return base
            .position_weights
            .iter()
            .copied()
            .map(|scalar| {
                let mut coefficients = vec![F::zero(); dimension];
                coefficients[0] = scalar;
                residue_kernel::<F, E>(&coefficients, alpha)
            })
            .collect();
    }
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        dimension,
        |D| {
            let rings = point
                .materialize_position_rings::<D>()?
                .ok_or(AkitaError::InvalidProof)?;
            if rings.len() != position_count {
                return Err(AkitaError::InvalidProof);
            }
            rings
                .iter()
                .map(|ring| residue_kernel::<F, E>(ring.coefficients(), alpha))
                .collect()
        }
    )
}

fn add_scaled_kernel<E: FieldCore>(
    destination: &mut [E],
    physical_start: usize,
    kernel: &[E],
    scale: E,
) -> Result<(), AkitaError> {
    if scale.is_zero() {
        return Ok(());
    }
    let physical_end = physical_start
        .checked_add(kernel.len())
        .ok_or_else(|| AkitaError::InvalidSetup("reduced relation address overflow".into()))?;
    for (weight, &coefficient) in destination
        .get_mut(physical_start..physical_end)
        .ok_or(AkitaError::InvalidProof)?
        .iter_mut()
        .zip(kernel)
    {
        *weight += scale * coefficient;
    }
    Ok(())
}

/// Compile the complete padded ordinary and compression relation-weight MLE
/// for one reduced-evaluation fold.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "build_reduced_dense_relation_weights")]
pub(in super::super) fn build_reduced_dense_relation_weights<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    lp: &CommittedGroupParams,
    tau1: &[E],
    opening_source_len: usize,
    opening_ring_dim: usize,
    relation_plan: &RelationRangeImagePlan,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    if lp.ring_relation_mode != RingRelationMode::ReducedEvaluation {
        return Err(AkitaError::InvalidSetup(
            "dense reduced weights require reduced-evaluation mode".into(),
        ));
    }
    let opening_batch = instance.opening_batch();
    let relation_geometry =
        RelationWitnessGeometry::for_level(lp, opening_batch, instance.extension_degree())?;
    if (0..opening_batch.num_groups()).any(|group| {
        !matches!(
            relation_geometry.group_opening_method(group),
            Ok(OpeningMethod::EvaluationTrace)
        )
    }) {
        return Err(AkitaError::InvalidSetup(
            "reduced relation weights require evaluation-trace openings".into(),
        ));
    }
    let witness_layout = instance.segment_layout(lp, None)?;
    if !matches!(
        witness_layout.relation_quotient_layout(),
        RelationQuotientLayout::ReducedEvaluation
    ) || !witness_layout.r_rows().is_empty()
    {
        return Err(AkitaError::InvalidSetup(
            "reduced relation mode disagrees with its quotient-free witness layout".into(),
        ));
    }
    let physical_field_len = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;
    let domain = relation_plan.digit_witness_domain();
    if domain.domain_len() != physical_field_len
        || domain.live_len() != witness_layout.live_coeff_len()
        || relation_plan.witness_layout() != &witness_layout
        || relation_plan.relation_witness_geometry() != &relation_geometry
    {
        return Err(AkitaError::InvalidSetup(
            "reduced relation plan disagrees with the current witness".into(),
        ));
    }
    let row_families = relation_geometry.rhs_layout().row_families()?;
    let row_weights = EqPolynomial::evals_prefix(tau1, row_families.len())?;
    let n_d_active = lp.open().matrix.output_rank();
    let d_column_ranges = relation_d_column_ranges(lp, opening_batch, &relation_geometry)?;
    let d_physical_columns = d_column_ranges
        .iter()
        .map(|range| range.end)
        .max()
        .unwrap_or(0);
    let d_d = lp.role_dims().d_d();
    let d_view = setup
        .shared_matrix()
        .ring_view_dyn(n_d_active, d_physical_columns, d_d)?;
    let d_family = SetupRows {
        rows: (0..n_d_active)
            .map(|row| d_view.row_flat(row))
            .collect::<Result<Vec<_>, _>>()?,
        ring_d: d_d,
    };
    let d_start = row_families
        .iter()
        .position(|row| matches!(row, RelationRowFamily::Opening { .. }))
        .ok_or(AkitaError::InvalidProof)?;
    let mut dense = vec![E::zero(); physical_field_len];

    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims_geometry(opening_batch, group_index)?;
        let group_d_a = group_dims.d_a();
        let group_d_b = group_dims.d_b();
        let group_d_d = group_dims.d_d();
        let (b_ratio, _) = SetupProjectionGeometry::native_role_subcolumn_counts(group_dims)?;
        let opening_width = relation_geometry
            .group_opening_geometry(group_index)?
            .physical_coefficient_width();
        let d_ratio = opening_width
            .checked_div(group_d_d)
            .filter(|count| *count > 0 && opening_width.is_multiple_of(group_d_d))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("opening width does not factor the D role".into())
            })?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let units = witness_layout.units_for_group(group_index)?;
        let k_g = group_layout.num_polynomials();
        let challenges = instance.group_ambient_a_challenges(group_index)?;
        let ring_multiplier_point = instance.group_ring_multiplier_point(group_index)?;
        let total_blocks = k_g
            .checked_mul(group_lp.num_live_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.len() != total_blocks
            || ring_multiplier_point.position_len() != group_lp.num_positions_per_block()
            || ring_multiplier_point.fold_len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidProof);
        }
        let challenge_kernels = (0..total_blocks)
            .map(|index| sparse_challenge_kernel::<F, E>(challenges, index, group_d_a, alpha))
            .collect::<Result<Vec<_>, _>>()?;
        let opening_kernels = position_multiplier_kernels::<F, E>(
            ring_multiplier_point,
            group_lp.num_positions_per_block(),
            group_d_a,
            alpha,
        )?;

        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = group_lp.num_digits_fold();
        let n_a = group_lp.a_rows_len();
        let physical_n_b = group_lp.b_rows_len();
        let n_b = group_lp.logical_b_rows_len()?;
        let inner_width = group_lp.a_col_len();
        let num_live_blocks = group_lp.num_live_blocks();
        let num_positions = group_lp.num_positions_per_block();
        let slice_geometry = CommitmentSliceGeometry::try_new(
            group_lp.outer_slice_count(),
            num_live_blocks,
            k_g,
            n_a,
            depth_commit,
            group_d_a,
            group_d_b,
        )?;
        let b_width = slice_geometry.physical_input_width();
        let a_view = setup
            .shared_matrix()
            .ring_view_dyn(n_a, inner_width, group_d_a)?;
        let a_family = SetupRows {
            rows: (0..n_a)
                .map(|row| a_view.row_flat(row))
                .collect::<Result<Vec<_>, _>>()?,
            ring_d: group_d_a,
        };
        let b_view = setup
            .shared_matrix()
            .ring_view_dyn(physical_n_b, b_width, group_d_b)?;
        let b_family = SetupRows {
            rows: (0..physical_n_b)
                .map(|row| b_view.row_flat(row))
                .collect::<Result<Vec<_>, _>>()?,
            ring_d: group_d_b,
        };
        let a_range = matching_row_range(
            &row_families,
            |family| matches!(family, RelationRowFamily::Inner { group_index: group, .. } if *group == group_index),
        )?;
        let b_range = matching_row_range(
            &row_families,
            |family| matches!(family, RelationRowFamily::Outer { group_index: group, .. } if *group == group_index),
        )?;
        let consistency_row = row_families
            .iter()
            .position(|family| {
                matches!(family, RelationRowFamily::Consistency { group_index: group, opening_method: OpeningMethod::EvaluationTrace, .. } if *group == group_index)
            })
            .ok_or(AkitaError::InvalidProof)?;
        if a_range.end > row_weights.len()
            || b_range.end > row_weights.len()
            || b_range.len() != n_b
        {
            return Err(AkitaError::InvalidProof);
        }
        let consistency_weight = row_weights[consistency_row];
        let opening_gadget = gadget_row_scalars::<F>(depth_open, group_lp.log_basis_open())
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        let commitment_gadget = gadget_row_scalars::<F>(depth_commit, group_lp.log_basis_outer())
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        let witness_gadget = gadget_row_scalars::<F>(depth_witness, group_lp.log_basis_inner())
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        let fold_gadget = gadget_row_scalars::<F>(depth_fold, group_lp.log_basis_open())
            .into_iter()
            .map(E::lift_base)
            .collect::<Vec<_>>();

        let d_setup_start = d_column_ranges
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?
            .start;
        let d_setup_len = total_blocks
            .checked_mul(d_ratio)
            .and_then(|len| len.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        let d_setup_end = d_setup_start
            .checked_add(d_setup_len)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D extent overflow".into()))?;
        let d_row_weights = (0..n_d_active)
            .map(|row| {
                Ok((
                    row,
                    vec![*row_weights
                        .get(d_start + row)
                        .ok_or(AkitaError::InvalidProof)?],
                ))
            })
            .filter_map(|result| match result {
                Ok((_, ref weights)) if weights[0].is_zero() => None,
                other => Some(other),
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let d_setup_kernels = evaluate_setup_column_kernels(
            &d_family,
            d_setup_start..d_setup_end,
            &d_row_weights,
            1,
            alpha,
        )?;
        let slice_count = group_lp.outer_slice_count().get();
        let b_row_weights = (0..physical_n_b)
            .map(|row| {
                let weights = (0..slice_count)
                    .map(|slice| {
                        let logical = slice_geometry
                            .logical_row_index(slice, row, physical_n_b)?
                            .checked_add(b_range.start)
                            .ok_or(AkitaError::InvalidProof)?;
                        row_weights
                            .get(logical)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((row, weights))
            })
            .filter_map(|result| match result {
                Ok((_, ref weights)) if weights.iter().all(|weight| weight.is_zero()) => None,
                other => Some(other),
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let b_setup_kernels = evaluate_setup_column_kernels(
            &b_family,
            0..b_width,
            &b_row_weights,
            slice_count,
            alpha,
        )?;

        for claim in 0..k_g {
            for block in 0..num_live_blocks {
                let unit = witness_layout.unit_for_block(group_index, block)?;
                let challenge_index = claim
                    .checked_mul(num_live_blocks)
                    .and_then(|base| base.checked_add(block))
                    .ok_or(AkitaError::InvalidProof)?;
                let challenge_kernel = challenge_kernels
                    .get(challenge_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let (slice_index, slice_block) = slice_geometry.block_coordinates(block)?;
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    for subcolumn in 0..d_ratio {
                        let physical_start = unit.e_coefficient_index(
                            group_d_d, k_g, depth_open, claim, block, subcolumn, digit, 0,
                        )?;
                        let kernel_start = subcolumn * group_d_d;
                        add_scaled_kernel(
                            &mut dense,
                            physical_start,
                            challenge_kernel
                                .get(kernel_start..kernel_start + group_d_d)
                                .ok_or(AkitaError::InvalidProof)?,
                            consistency_weight * gadget,
                        )?;
                        let logical_block = claim * num_live_blocks + block;
                        let d_column = logical_block
                            .checked_mul(d_ratio)
                            .and_then(|base| base.checked_add(subcolumn))
                            .and_then(|base| base.checked_mul(depth_open))
                            .and_then(|base| base.checked_add(digit))
                            .ok_or(AkitaError::InvalidProof)?;
                        add_scaled_kernel(
                            &mut dense,
                            physical_start,
                            d_setup_kernels.get(0, d_column)?,
                            E::one(),
                        )?;
                    }
                }
                for a_row in 0..n_a {
                    let a_weight = row_weights[a_range.start + a_row];
                    for (digit, &gadget) in commitment_gadget.iter().enumerate() {
                        let block_claim = slice_geometry
                            .max_blocks_per_slice()
                            .checked_mul(claim)
                            .and_then(|base| base.checked_add(slice_block))
                            .ok_or(AkitaError::InvalidProof)?;
                        let row_block_claim = n_a
                            .checked_mul(block_claim)
                            .and_then(|base| base.checked_add(a_row))
                            .ok_or(AkitaError::InvalidProof)?;
                        for subcolumn in 0..b_ratio {
                            let local_column = row_block_claim
                                .checked_mul(b_ratio)
                                .and_then(|base| base.checked_add(subcolumn))
                                .and_then(|base| base.checked_mul(depth_commit))
                                .and_then(|base| base.checked_add(digit))
                                .ok_or(AkitaError::InvalidProof)?;
                            let physical_start = unit.t_coefficient_index(
                                group_d_a,
                                group_d_b,
                                k_g,
                                n_a,
                                depth_commit,
                                claim,
                                block,
                                a_row,
                                subcolumn,
                                digit,
                                0,
                            )?;
                            let kernel_start = subcolumn * group_d_b;
                            add_scaled_kernel(
                                &mut dense,
                                physical_start,
                                challenge_kernel
                                    .get(kernel_start..kernel_start + group_d_b)
                                    .ok_or(AkitaError::InvalidProof)?,
                                a_weight * gadget,
                            )?;
                            add_scaled_kernel(
                                &mut dense,
                                physical_start,
                                b_setup_kernels.get(slice_index, local_column)?,
                                E::one(),
                            )?;
                        }
                    }
                }
            }
        }

        let a_row_weights = (0..n_a)
            .map(|row| Ok((row, vec![row_weights[a_range.start + row]])))
            .filter_map(|result| match result {
                Ok((_, ref weights)) if weights[0].is_zero() => None,
                other => Some(other),
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let a_setup_kernels =
            evaluate_setup_column_kernels(&a_family, 0..inner_width, &a_row_weights, 1, alpha)?;
        for unit in units {
            for position in 0..num_positions {
                let opening_kernel = opening_kernels
                    .get(position)
                    .ok_or(AkitaError::InvalidProof)?;
                for (witness_digit, &witness_scale) in witness_gadget.iter().enumerate() {
                    let source_column = position
                        .checked_mul(depth_witness)
                        .and_then(|base| base.checked_add(witness_digit))
                        .ok_or(AkitaError::InvalidProof)?;
                    for (fold_digit, &fold_scale) in fold_gadget.iter().enumerate() {
                        let physical_start = unit.z_coefficient_index(
                            group_d_a,
                            num_positions,
                            depth_witness,
                            depth_fold,
                            position,
                            witness_digit,
                            fold_digit,
                            0,
                        )?;
                        add_scaled_kernel(
                            &mut dense,
                            physical_start,
                            opening_kernel,
                            -(consistency_weight * witness_scale * fold_scale),
                        )?;
                        add_scaled_kernel(
                            &mut dense,
                            physical_start,
                            a_setup_kernels.get(0, source_column)?,
                            -fold_scale,
                        )?;
                    }
                }
            }
        }
    }

    if lp.payload_mode.is_compressed() {
        let compression = akita_types::build_reduced_compression_relation_weights::<F, E>(
            alpha,
            lp,
            opening_batch,
            instance.extension_degree(),
            tau1,
            &witness_layout,
            opening_ring_dim,
            physical_field_len,
        )?;
        compression.accumulate_dense(setup, &mut dense)?;
    }
    if dense[domain.live_len()..]
        .iter()
        .any(|weight| !weight.is_zero())
    {
        return Err(AkitaError::InvalidSetup(
            "reduced relation weights are nonzero outside the live witness".into(),
        ));
    }
    Ok(dense)
}
