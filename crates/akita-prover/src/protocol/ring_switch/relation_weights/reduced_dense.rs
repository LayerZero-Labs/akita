//! Dense prover weights for quotient-free reduced ring relations.
//!
//! This compiler retains every public multiplier in coefficient form until it
//! has passed through the shared negacyclic residue recurrence. It scatters
//! the resulting native kernels directly into the canonical `WitnessLayout`
//! ranges; it never constructs a relation matrix or a quotient-shaped event
//! stream.

use super::*;
use akita_algebra::ring::{residue_kernel, sparse_residue_kernel};
use akita_challenges::Challenges;
use akita_types::{dispatch_for_field, RingMultiplierOpeningPoint};
use jolt_field::{ExtField, MulBaseUnreduced};

fn sparse_challenge_kernel<F, E>(
    challenges: &Challenges,
    index: usize,
    dimension: usize,
    alpha: E,
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
    sparse_residue_kernel(
        dimension,
        challenge
            .positions
            .iter()
            .zip(&challenge.coeffs)
            .map(|(&position, &coefficient)| {
                (
                    position as usize,
                    E::lift_base(F::from_i64(i64::from(coefficient))),
                )
            }),
        alpha,
    )
}

fn position_multiplier_kernels<F, E>(
    point: &RingMultiplierOpeningPoint<F>,
    position_count: usize,
    dimension: usize,
    alpha: E,
) -> Result<Vec<Vec<E>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    match point {
        RingMultiplierOpeningPoint::Base(base) => {
            if base.position_weights.len() != position_count {
                return Err(AkitaError::InvalidProof);
            }
            let mut unit_coefficients = vec![F::zero(); dimension];
            *unit_coefficients
                .first_mut()
                .ok_or(AkitaError::InvalidProof)? = F::one();
            let unit_kernel = residue_kernel::<F, E>(&unit_coefficients, alpha)?;
            base.position_weights
                .iter()
                .copied()
                .map(|scalar| {
                    Ok(unit_kernel
                        .iter()
                        .copied()
                        .map(|coefficient| coefficient.mul_base(scalar))
                        .collect())
                })
                .collect()
        }
        RingMultiplierOpeningPoint::Subfield(subfield) => dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            dimension,
            |D| {
                let rings = subfield.materialize_position_rings::<D>()?;
                if rings.len() != position_count {
                    return Err(AkitaError::InvalidProof);
                }
                rings
                    .iter()
                    .map(|ring| residue_kernel::<F, E>(ring.coefficients(), alpha))
                    .collect()
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

struct ReducedEtSink<'a, E> {
    dense: &'a mut [E],
    plan: &'a compiler::RelationWeightGroupPlan<E>,
    challenge_kernels: &'a [Vec<E>],
    d_setup_kernels: &'a SetupColumnValues<E>,
    b_setup_kernels: &'a SetupColumnValues<E>,
}

impl<E: Field> EtWeightSink<E> for ReducedEtSink<'_, E> {
    fn add_e(&mut self, address: EAddress<E>) -> Result<(), AkitaError> {
        let kernel = self
            .challenge_kernels
            .get(address.challenge_index)
            .ok_or(AkitaError::InvalidProof)?;
        let kernel_start = address.role_subcolumn * self.plan.roles.d_d;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            kernel
                .get(kernel_start..kernel_start + self.plan.roles.d_d)
                .ok_or(AkitaError::InvalidProof)?,
            address.constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.d_setup_kernels.get(0, address.setup_column)?,
            E::one(),
        )
    }

    fn add_t(&mut self, address: TAddress<E>) -> Result<(), AkitaError> {
        let kernel = self
            .challenge_kernels
            .get(address.challenge_index)
            .ok_or(AkitaError::InvalidProof)?;
        let kernel_start = address.role_subcolumn * self.plan.roles.d_b;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            kernel
                .get(kernel_start..kernel_start + self.plan.roles.d_b)
                .ok_or(AkitaError::InvalidProof)?,
            address.constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.b_setup_kernels
                .get(address.slice_index, address.setup_column)?,
            E::one(),
        )
    }
}

struct ReducedZSink<'a, E> {
    dense: &'a mut [E],
    opening_kernels: &'a [Vec<E>],
    a_setup_kernels: &'a SetupColumnValues<E>,
}

impl<E: Field> ZWeightSink<E> for ReducedZSink<'_, E> {
    fn add_z(&mut self, address: ZAddress<E>) -> Result<(), AkitaError> {
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.opening_kernels
                .get(address.position)
                .ok_or(AkitaError::InvalidProof)?,
            address.constraint_scale,
        )?;
        add_scaled_kernel(
            self.dense,
            address.physical_start,
            self.a_setup_kernels.get(0, address.setup_column)?,
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
) -> Result<crate::protocol::sumcheck::DenseRelationWeights<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + MulBaseUnreduced<F>,
{
    let opening_batch = instance.opening_batch();
    let compilation = RelationWeightCompilation::new(
        Some(setup),
        instance,
        lp,
        tau1,
        opening_source_len,
        opening_ring_dim,
        relation_plan,
    )?;
    let setup_sources = compilation.setup_sources.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("reduced relation requires direct setup rows".into())
    })?;
    let mut dense = vec![E::zero(); compilation.physical_field_len];

    for group_plan in &compilation.plan.groups {
        let group_index = group_plan.group_index;
        let group_setup = setup_sources.group(group_index)?;
        let group_source = compilation.group_source(group_index)?;
        let group_d_a = group_plan.roles.d_a;
        let group_d_b = group_plan.roles.d_b;
        let group_d_d = group_plan.roles.d_d;
        let challenges = group_source.challenges;
        let OpeningFamily::EvaluationTrace(ring_multiplier_point) = group_source.opening else {
            return Err(AkitaError::InvalidSetup(
                "reduced relation requires evaluation-trace openings".into(),
            ));
        };
        let total_blocks = challenges.len();
        let challenge_kernels = (0..total_blocks)
            .map(|index| sparse_challenge_kernel::<F, E>(challenges, index, group_d_a, alpha))
            .collect::<Result<Vec<_>, _>>()?;
        let d_setup_kernels = contract_setup_columns(
            &setup_sources.d,
            group_plan.rows.d_setup_range.clone(),
            &compilation.plan.d_row_weights,
            1,
            group_d_d,
            |coefficients| residue_kernel::<F, E>(coefficients, alpha),
        )?;
        let b_setup_kernels = contract_setup_columns(
            &group_setup.b,
            0..group_plan.witness.b_width,
            &group_plan.rows.b_setup_row_weights,
            group_plan.witness.slice_count,
            group_d_b,
            |coefficients| residue_kernel::<F, E>(coefficients, alpha),
        )?;

        {
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

        let opening_kernels = position_multiplier_kernels::<F, E>(
            ring_multiplier_point,
            group_plan.witness.num_positions,
            group_d_a,
            alpha,
        )?;
        let a_setup_kernels = contract_setup_columns(
            &group_setup.a,
            0..group_plan.witness.inner_width,
            &group_plan.rows.a_setup_row_weights,
            1,
            group_d_a,
            |coefficients| residue_kernel::<F, E>(coefficients, alpha),
        )?;
        let mut z_sink = ReducedZSink {
            dense: &mut dense,
            opening_kernels: &opening_kernels,
            a_setup_kernels: &a_setup_kernels,
        };
        compile_group_z_addresses(group_plan, &compilation.witness_layout, &mut z_sink)?;
    }

    if lp.payload_mode.is_compressed() {
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
