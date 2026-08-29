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
    dispatch_for_field, RelationQuotientLayout, RingMultiplierOpeningPoint, RingRelationMode,
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

struct ReducedGroupSink<'a, E> {
    dense: &'a mut [E],
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    challenge_kernels: Option<&'a [Vec<E>]>,
    opening_kernels: Option<&'a [Vec<E>]>,
    d_setup_kernels: Option<&'a SetupColumnKernels<E>>,
    b_setup_kernels: Option<&'a SetupColumnKernels<E>>,
    a_setup_kernels: Option<&'a SetupColumnKernels<E>>,
}

impl<E: FieldCore> RelationWeightSink<E> for ReducedGroupSink<'_, E> {
    fn add_e(&mut self, address: EAddress<E>) -> Result<(), AkitaError> {
        let kernel = self
            .challenge_kernels
            .ok_or(AkitaError::InvalidProof)?
            .get(address.challenge_index)
            .ok_or(AkitaError::InvalidProof)?;
        let kernel_start = address.role_subcolumn * self.plan.group_d_d;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            kernel
                .get(kernel_start..kernel_start + self.plan.group_d_d)
                .ok_or(AkitaError::InvalidProof)?,
            address.constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.d_setup_kernels
                .ok_or(AkitaError::InvalidProof)?
                .get(0, address.setup_column)?,
            E::one(),
        )
    }

    fn add_t(&mut self, address: TAddress<E>) -> Result<(), AkitaError> {
        let kernel = self
            .challenge_kernels
            .ok_or(AkitaError::InvalidProof)?
            .get(address.challenge_index)
            .ok_or(AkitaError::InvalidProof)?;
        let kernel_start = address.role_subcolumn * self.plan.group_d_b;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            kernel
                .get(kernel_start..kernel_start + self.plan.group_d_b)
                .ok_or(AkitaError::InvalidProof)?,
            address.constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.b_setup_kernels
                .ok_or(AkitaError::InvalidProof)?
                .get(address.slice_index, address.setup_column)?,
            E::one(),
        )
    }

    fn add_z(&mut self, address: ZAddress<E>) -> Result<(), AkitaError> {
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.opening_kernels
                .ok_or(AkitaError::InvalidProof)?
                .get(address.position)
                .ok_or(AkitaError::InvalidProof)?,
            address.constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.a_setup_kernels
                .ok_or(AkitaError::InvalidProof)?
                .get(0, address.setup_column)?,
            address.setup_scale,
        )
    }
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
    let compilation = RelationWeightCompilationPlan::new::<F>(
        lp,
        opening_batch,
        relation_plan,
        &row_families,
        &row_weights,
    )?;
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

    for group_plan in &compilation.groups {
        let group_index = group_plan.group_index;
        let group_lp = lp.group_params_geometry(opening_batch, group_index)?;
        let group_d_a = group_plan.group_d_a;
        let group_d_b = group_plan.group_d_b;
        let k_g = group_plan.num_claims;
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
        let n_a = group_plan.n_a;
        let physical_n_b = group_lp.b_rows_len();
        let inner_width = group_plan.inner_width;
        let slice_geometry = &group_plan.slice_geometry;
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
        let d_setup_start = d_column_ranges
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?
            .start;
        let d_setup_len = total_blocks
            .checked_mul(group_plan.d_ratio)
            .and_then(|len| len.checked_mul(group_plan.depth_open))
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
                        let logical = slice_geometry.logical_row_index(slice, row, physical_n_b)?;
                        group_plan
                            .b_row_weights
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

        {
            let mut et_sink = ReducedGroupSink {
                dense: &mut dense,
                plan: group_plan,
                challenge_kernels: Some(&challenge_kernels),
                opening_kernels: None,
                d_setup_kernels: Some(&d_setup_kernels),
                b_setup_kernels: Some(&b_setup_kernels),
                a_setup_kernels: None,
            };
            compile_group_et_addresses(group_plan, &witness_layout, &mut et_sink)?;
        }
        drop(challenge_kernels);
        drop(d_setup_kernels);
        drop(b_setup_kernels);

        let opening_kernels = position_multiplier_kernels::<F, E>(
            ring_multiplier_point,
            group_lp.num_positions_per_block(),
            group_d_a,
            alpha,
        )?;
        let a_row_weights = (0..n_a)
            .map(|row| Ok((row, vec![group_plan.a_row_weights[row]])))
            .filter_map(|result| match result {
                Ok((_, ref weights)) if weights[0].is_zero() => None,
                other => Some(other),
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let a_setup_kernels =
            evaluate_setup_column_kernels(&a_family, 0..inner_width, &a_row_weights, 1, alpha)?;
        let mut z_sink = ReducedGroupSink {
            dense: &mut dense,
            plan: group_plan,
            challenge_kernels: None,
            opening_kernels: Some(&opening_kernels),
            d_setup_kernels: None,
            b_setup_kernels: None,
            a_setup_kernels: Some(&a_setup_kernels),
        };
        compile_group_z_addresses(group_plan, &witness_layout, &mut z_sink)?;
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
