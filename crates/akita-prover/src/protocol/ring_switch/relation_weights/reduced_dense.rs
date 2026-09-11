//! Dense prover weights for quotient-free reduced ring relations.
//!
//! This compiler retains every public multiplier in coefficient form until it
//! has passed through the shared negacyclic residue recurrence. It scatters
//! the resulting native kernels directly into the canonical `WitnessLayout`
//! ranges; it never constructs a relation matrix or a quotient-shaped event
//! stream.

use super::*;
use akita_algebra::ring::ResidueKernelPoint;
use akita_challenges::Challenges;
use akita_types::{dispatch_for_field, RingMultiplierOpeningPoint};
use jolt_field::{ExtField, MulBaseUnreduced};

fn sparse_challenge_kernel<F, E>(
    challenges: &Challenges,
    index: usize,
    point: &ResidueKernelPoint<E>,
) -> Result<Vec<E>, AkitaError>
where
    F: Field,
    E: Field + ExtField<F>,
{
    let challenge = challenges
        .as_slice()
        .get(index)
        .ok_or(AkitaError::InvalidProof)?;
    if challenge.positions.len() != challenge.coeffs.len() {
        return Err(AkitaError::InvalidProof);
    }
    point.sparse_kernel(challenge.positions.iter().zip(&challenge.coeffs).map(
        |(&position, &coefficient)| {
            (
                position as usize,
                E::lift_base(F::from_i64(i64::from(coefficient))),
            )
        },
    ))
}

enum PositionMultiplierKernels<'a, F, E> {
    Base {
        position_weights: &'a [F],
        alpha_powers: &'a [E],
    },
    Subfield(Vec<Vec<E>>),
}

impl<F, E> PositionMultiplierKernels<'_, F, E>
where
    F: Field,
    E: Field + ExtField<F>,
{
    fn add_scaled(
        &self,
        destination: &mut [E],
        physical_start: usize,
        position: usize,
        scale: E,
    ) -> Result<(), AkitaError> {
        match self {
            Self::Base {
                position_weights,
                alpha_powers,
            } => add_scaled_kernel(
                destination,
                physical_start,
                alpha_powers,
                scale.mul_base(
                    *position_weights
                        .get(position)
                        .ok_or(AkitaError::InvalidProof)?,
                ),
            ),
            Self::Subfield(kernels) => add_scaled_kernel(
                destination,
                physical_start,
                kernels.get(position).ok_or(AkitaError::InvalidProof)?,
                scale,
            ),
        }
    }
}

fn position_multiplier_kernels<'a, F, E>(
    point: &'a RingMultiplierOpeningPoint<F>,
    position_count: usize,
    residue_point: &'a ResidueKernelPoint<E>,
) -> Result<PositionMultiplierKernels<'a, F, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    match point {
        RingMultiplierOpeningPoint::Base(base) => {
            if base.position_weights.len() != position_count {
                return Err(AkitaError::InvalidProof);
            }
            Ok(PositionMultiplierKernels::Base {
                position_weights: &base.position_weights,
                alpha_powers: residue_point.powers(),
            })
        }
        RingMultiplierOpeningPoint::Subfield(subfield) => dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            residue_point.dimension(),
            |D| {
                let rings = subfield.materialize_position_rings::<D>()?;
                if rings.len() != position_count {
                    return Err(AkitaError::InvalidProof);
                }
                let kernels = rings
                    .iter()
                    .map(|ring| residue_point.kernel(ring.coefficients()))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PositionMultiplierKernels::Subfield(kernels))
            }
        ),
    }
}

fn add_scaled_kernel<E: Field>(
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
    let destination = destination
        .get_mut(physical_start..physical_end)
        .ok_or(AkitaError::InvalidProof)?;
    if scale == E::one() {
        for (weight, &coefficient) in destination.iter_mut().zip(kernel) {
            *weight += coefficient;
        }
    } else if scale == -E::one() {
        for (weight, &coefficient) in destination.iter_mut().zip(kernel) {
            *weight -= coefficient;
        }
    } else {
        for (weight, &coefficient) in destination.iter_mut().zip(kernel) {
            *weight += scale * coefficient;
        }
    }
    Ok(())
}

struct ReducedEtSink<'a, E> {
    dense: &'a mut [E],
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    challenge_kernels: &'a [Vec<E>],
    d_setup_kernels: &'a SetupColumnValues<E>,
    b_setup_kernels: &'a SetupColumnValues<E>,
}

impl<E: Field> EtWeightSink<E> for ReducedEtSink<'_, E> {
    fn add_e(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        let kernel = self
            .challenge_kernels
            .get(challenge_index)
            .ok_or(AkitaError::InvalidProof)?;
        let kernel_start = role_subcolumn * self.plan.roles.d_d;
        add_scaled_kernel(
            self.dense,
            physical_start,
            kernel
                .get(kernel_start..kernel_start + self.plan.roles.d_d)
                .ok_or(AkitaError::InvalidProof)?,
            constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            physical_start,
            self.d_setup_kernels.get(0, setup_column)?,
            E::one(),
        )
    }

    fn add_t(
        &mut self,
        physical_start: usize,
        challenge_index: usize,
        role_subcolumn: usize,
        slice_index: usize,
        setup_column: usize,
        constraint_scale: E,
    ) -> Result<(), AkitaError> {
        let kernel = self
            .challenge_kernels
            .get(challenge_index)
            .ok_or(AkitaError::InvalidProof)?;
        let kernel_start = role_subcolumn * self.plan.roles.d_b;
        add_scaled_kernel(
            self.dense,
            physical_start,
            kernel
                .get(kernel_start..kernel_start + self.plan.roles.d_b)
                .ok_or(AkitaError::InvalidProof)?,
            constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            physical_start,
            self.b_setup_kernels.get(slice_index, setup_column)?,
            E::one(),
        )
    }
}

struct ReducedZSink<'a, F, E> {
    dense: &'a mut [E],
    opening_kernels: &'a PositionMultiplierKernels<'a, F, E>,
    a_setup_kernels: &'a SetupColumnValues<E>,
}

impl<F, E> ZWeightSink<E> for ReducedZSink<'_, F, E>
where
    F: Field,
    E: Field + ExtField<F>,
{
    fn add_z(
        &mut self,
        physical_start: usize,
        position: usize,
        setup_column: usize,
        constraint_scale: E,
        setup_scale: E,
    ) -> Result<(), AkitaError> {
        self.opening_kernels
            .add_scaled(self.dense, physical_start, position, constraint_scale)?;
        add_scaled_kernel(
            self.dense,
            physical_start,
            self.a_setup_kernels.get(0, setup_column)?,
            setup_scale,
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
) -> Result<crate::protocol::sumcheck::DenseRelationWeights<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F>,
{
    let opening_batch = instance.opening_batch();
    let compilation = {
        let _span = tracing::info_span!("reduced_weight_plan").entered();
        RelationWeightCompilation::new(
            Some(setup),
            instance,
            lp,
            tau1,
            opening_source_len,
            opening_ring_dim,
            relation_plan,
        )?
    };
    let setup_sources = compilation.setup_sources.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("reduced relation requires direct setup rows".into())
    })?;
    let mut dense = vec![E::zero(); compilation.physical_field_len];

    for group_plan in &compilation.plan.groups {
        let group_index = group_plan.group_index;
        let group_setup = setup_sources.group(group_index)?;
        let group_source = compilation.group_source(group_index)?;
        let group_d_a = group_plan.roles.d_a;
        let challenges = group_source.challenges;
        let a_residue_point = ResidueKernelPoint::new(alpha, group_d_a)?;
        let b_residue_point = ResidueKernelPoint::new(alpha, group_plan.roles.d_b)?;
        let d_residue_point = ResidueKernelPoint::new(alpha, group_plan.roles.d_d)?;
        let OpeningFamily::EvaluationTrace(ring_multiplier_point) = group_source.opening else {
            return Err(AkitaError::InvalidSetup(
                "reduced relation requires evaluation-trace openings".into(),
            ));
        };
        let total_blocks = challenges.len();
        let challenge_kernels = {
            let _span = tracing::info_span!("reduced_challenge_kernels").entered();
            (0..total_blocks)
                .map(|index| sparse_challenge_kernel::<F, E>(challenges, index, &a_residue_point))
                .collect::<Result<Vec<_>, _>>()?
        };
        let d_setup_kernels = {
            let _span = tracing::info_span!("reduced_d_setup_kernels").entered();
            contract_setup_residue_columns(
                &setup_sources.d,
                group_plan.rows.d_setup_range.clone(),
                &compilation.plan.d_row_weights,
                1,
                &d_residue_point,
            )?
        };
        let b_setup_kernels = {
            let _span = tracing::info_span!("reduced_b_setup_kernels").entered();
            contract_setup_residue_columns(
                &group_setup.b,
                0..group_plan.witness.b_width,
                &group_plan.rows.b_setup_row_weights,
                group_plan.witness.slice_count,
                &b_residue_point,
            )?
        };

        {
            let _span = tracing::info_span!("reduced_et_scatter").entered();
            let mut et_sink = ReducedEtSink {
                dense: &mut dense,
                plan: group_plan,
                challenge_kernels: &challenge_kernels,
                d_setup_kernels: &d_setup_kernels,
                b_setup_kernels: &b_setup_kernels,
            };
            compile_group_et_addresses(group_plan, &compilation.witness_layout, &mut et_sink)?;
        }
        drop(challenge_kernels);
        drop(d_setup_kernels);
        drop(b_setup_kernels);

        let opening_kernels = {
            let _span = tracing::info_span!("reduced_opening_kernels").entered();
            position_multiplier_kernels::<F, E>(
                ring_multiplier_point,
                group_plan.witness.num_positions,
                &a_residue_point,
            )?
        };
        let a_setup_kernels = {
            let _span = tracing::info_span!("reduced_a_setup_kernels").entered();
            contract_setup_residue_columns(
                &group_setup.a,
                0..group_plan.witness.inner_width,
                &group_plan.rows.a_setup_row_weights,
                1,
                &a_residue_point,
            )?
        };
        {
            let _span = tracing::info_span!("reduced_z_scatter").entered();
            let mut z_sink = ReducedZSink {
                dense: &mut dense,
                opening_kernels: &opening_kernels,
                a_setup_kernels: &a_setup_kernels,
            };
            compile_group_z_addresses(group_plan, &compilation.witness_layout, &mut z_sink)?;
        }
    }

    if lp.payload_mode.is_compressed() {
        let _span = tracing::info_span!("reduced_compression_weights").entered();
        let compression = akita_types::build_reduced_compression_relation_weights::<F, E>(
            alpha,
            lp,
            opening_batch,
            instance.extension_degree(),
            tau1,
            &compilation.witness_layout,
            opening_ring_dim,
            compilation.physical_field_len,
        )?;
        compression.accumulate_dense(setup, &mut dense)?;
    }
    crate::protocol::sumcheck::DenseRelationWeights::new(
        dense,
        compilation.witness_layout.live_coeff_len(),
    )
}
