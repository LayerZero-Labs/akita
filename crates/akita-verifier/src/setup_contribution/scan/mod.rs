mod group;
mod reduced;

use super::*;
use akita_algebra::cfg_try_fold_reduce;
use akita_algebra::ring::{eval_ring_at_pows_fast, scalar_powers};
use group::evaluate_base_ring_direct;
use reduced::evaluate_groups_reduced;

enum DirectScanKernel<'a, E: Field> {
    LiftedPower {
        base_powers: &'a [E],
        projections: &'a [[RoleProjection<E>; 3]],
        weights: &'a [DirectScanWeights<E>],
    },
    ReducedEvaluation {
        weights: &'a [ReducedDirectScanWeights<E>],
    },
}

impl<E: Field> DirectScan<E> {
    /// Evaluate the setup contribution by scanning the packed setup with the
    /// weights and partition prepared in this scan.
    ///
    /// # Errors
    ///
    /// Returns an error if the shared setup matrix is too small for this
    /// scan's plan.
    pub fn evaluate_direct<F>(&self, setup: &AkitaExpandedSetup<F>) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let plan = &self.plan;
        match &self.mode {
            DirectScanMode::Lifted { alpha, groups } => {
                let geometry = plan.projection_geometry();
                let alpha_pows_a = scalar_powers(*alpha, geometry.role_dims().d_a());
                let alpha_pows_b = scalar_powers(*alpha, geometry.role_dims().d_b());
                let alpha_pows_d = scalar_powers(*alpha, geometry.role_dims().d_d());
                evaluate_role_dims_direct(
                    plan,
                    setup,
                    &alpha_pows_a,
                    &alpha_pows_b,
                    &alpha_pows_d,
                    groups,
                    &self.partitions,
                )
            }
            DirectScanMode::Reduced { groups, .. } => {
                evaluate_reduced_direct(plan, setup, groups, &self.partitions)
            }
        }
    }
}

fn evaluate_reduced_direct<F, E>(
    plan: &SetupContributionPlan<E>,
    setup: &AkitaExpandedSetup<F>,
    weights: &[ReducedDirectScanWeights<E>],
    partitions: &[GroupScanPartition<E>],
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let base_d = plan.projection_geometry().base_ring_dim();
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        F,
        base_d,
        |BASE_D| {
            evaluate_direct_typed::<F, _, BASE_D>(
                plan,
                setup,
                DirectScanKernel::ReducedEvaluation { weights },
                partitions,
            )
        }
    )
}

fn evaluate_role_dims_direct<F, E>(
    plan: &SetupContributionPlan<E>,
    setup: &AkitaExpandedSetup<F>,
    alpha_pows_a: &[E],
    alpha_pows_b: &[E],
    alpha_pows_d: &[E],
    weights: &[DirectScanWeights<E>],
    partitions: &[GroupScanPartition<E>],
) -> Result<E, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let geometry = plan.projection_geometry();
    geometry.validate_alpha_power_lengths(
        alpha_pows_a.len(),
        alpha_pows_b.len(),
        alpha_pows_d.len(),
    )?;
    let base_d = geometry.base_ring_dim();
    let alpha = *alpha_pows_a.get(1).ok_or(AkitaError::InvalidProof)?;
    let base_pows_storage;
    let base_pows = if alpha_pows_a.len() == base_d {
        alpha_pows_a
    } else if alpha_pows_b.len() == base_d {
        alpha_pows_b
    } else if alpha_pows_d.len() == base_d {
        alpha_pows_d
    } else {
        base_pows_storage = scalar_powers(alpha, base_d);
        &base_pows_storage
    };
    let build_root_projection = |role: &'static str, powers: &[E], ratio: usize| {
        role_projection(powers, base_pows, ratio).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "{role} alpha powers do not decompose over the shared setup base"
            ))
        })
    };
    let root_projections = [
        build_root_projection("A", alpha_pows_a, geometry.a_ratio())?,
        build_root_projection("B", alpha_pows_b, geometry.b_ratio())?,
        build_root_projection("D", alpha_pows_d, geometry.d_ratio())?,
    ];
    let projections = if matches!(plan.groups(), [only] if only.role_dims() == geometry.role_dims())
    {
        vec![root_projections]
    } else {
        plan.groups()
            .iter()
            .map(|group| {
                let build = |role: &'static str, dimension: usize, ratio: usize| {
                    let powers = scalar_powers(alpha, dimension);
                    role_projection(&powers, base_pows, ratio).ok_or_else(|| {
                        AkitaError::InvalidSetup(format!(
                            "{role} alpha powers do not decompose over the shared setup base"
                        ))
                    })
                };
                Ok([
                    build("A", group.role_dims().d_a(), group.a_ratio())?,
                    build("B", group.role_dims().d_b(), group.b_ratio())?,
                    build("D", group.role_dims().d_d(), group.d_ratio())?,
                ])
            })
            .collect::<Result<Vec<_>, AkitaError>>()?
    };

    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        F,
        base_d,
        |BASE_D| {
            evaluate_direct_typed::<F, _, BASE_D>(
                plan,
                setup,
                DirectScanKernel::LiftedPower {
                    base_powers: base_pows,
                    projections: &projections,
                    weights,
                },
                partitions,
            )
        }
    )
}

fn evaluate_direct_typed<F, E, const BASE_D: usize>(
    plan: &SetupContributionPlan<E>,
    setup: &AkitaExpandedSetup<F>,
    kernel: DirectScanKernel<'_, E>,
    partitions: &[GroupScanPartition<E>],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let fused_groups = plan.groups().len() > 1;
    let reduced_evaluation = matches!(&kernel, DirectScanKernel::ReducedEvaluation { .. });
    let logical_group_rings = partitions.iter().fold(0usize, |sum, partition| {
        sum.saturating_add(partition.required)
    });
    let physical_ring_evaluations = if fused_groups || reduced_evaluation {
        plan.projection_geometry().required()
    } else {
        logical_group_rings
    };
    let jobs = if fused_groups || reduced_evaluation {
        plan.projection_geometry()
            .required()
            .div_ceil(super::segments::SETUP_SCAN_JOB_RINGS)
    } else {
        partitions
            .iter()
            .map(|partition| partition.segments.len())
            .sum::<usize>()
    };
    let _span = tracing::info_span!(
        "setup_contribution_scan",
        required = plan.projection_geometry().required(),
        groups = plan.groups().len(),
        logical_group_rings,
        physical_ring_evaluations,
        jobs,
        fused_groups,
        reduced_evaluation,
        base_d = BASE_D,
        final_a_ratio = plan.projection_geometry().a_ratio(),
        final_b_ratio = plan.projection_geometry().b_ratio(),
        final_d_ratio = plan.projection_geometry().d_ratio()
    )
    .entered();
    let required = plan.projection_geometry().required();
    let setup_len = setup.shared_matrix().num_field_elements() / BASE_D;
    if plan.projection_geometry().base_ring_dim() != BASE_D || required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected verifier layout".into(),
        ));
    }
    // The scan reads only the `required` leading rings.
    let setup_view = setup.shared_matrix().ring_view::<BASE_D>(1, required)?;
    let (base_powers, projections, weights) = match kernel {
        DirectScanKernel::LiftedPower {
            base_powers,
            projections,
            weights,
        } => (base_powers, projections, weights),
        DirectScanKernel::ReducedEvaluation { weights } => {
            return evaluate_groups_reduced(plan, &setup_view, weights, partitions);
        }
    };
    if base_powers.len() != BASE_D {
        return Err(AkitaError::InvalidSize {
            expected: BASE_D,
            actual: base_powers.len(),
        });
    }
    if fused_groups {
        return evaluate_groups_fused::<F, _, BASE_D>(
            plan,
            &setup_view,
            base_powers,
            projections,
            weights,
            partitions,
        );
    }
    if partitions.len() != plan.groups().len() || weights.len() != plan.groups().len() {
        return Err(AkitaError::InvalidSetup(
            "cached setup scan geometry is malformed".into(),
        ));
    }
    let mut acc = E::zero();
    for (((group, projection), weights), partition) in plan
        .groups()
        .iter()
        .zip(projections)
        .zip(weights)
        .zip(partitions)
    {
        acc += evaluate_base_ring_direct::<F, _, BASE_D>(
            group,
            &setup_view,
            weights,
            partition,
            base_powers,
            plan.d_weights(),
            &projection[0],
            &projection[1],
            &projection[2],
            plan.d_rows(),
            plan.d_physical_cols(),
        )?;
    }
    Ok(acc)
}

fn evaluate_groups_fused<F, E, const BASE_D: usize>(
    plan: &SetupContributionPlan<E>,
    setup_view: &RingMatrixView<'_, F, BASE_D>,
    base_pows: &[E],
    projections: &[[RoleProjection<E>; 3]],
    direct_weights: &[DirectScanWeights<E>],
    partitions: &[GroupScanPartition<E>],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let setup_flat = setup_view.as_slice();
    let required = plan.projection_geometry().required();
    if plan.d_weights().len() != plan.d_rows()
        || direct_weights.len() != plan.groups().len()
        || partitions.len() != plan.groups().len()
    {
        return Err(AkitaError::InvalidSetup(
            "cached setup scan geometry is malformed".into(),
        ));
    }
    let job_rings = super::segments::SETUP_SCAN_JOB_RINGS;
    let num_jobs = required.div_ceil(job_rings);
    cfg_try_fold_reduce!(
        0..num_jobs,
        E::zero,
        |acc, job| {
            let lo = job.checked_mul(job_rings).ok_or(AkitaError::InvalidProof)?;
            let hi = lo.saturating_add(job_rings).min(required);
            let setup = setup_flat.get(lo..hi).ok_or(AkitaError::InvalidProof)?;
            let mut weights = vec![E::zero(); setup.len()];
            for ((projection, direct), partition) in
                projections.iter().zip(direct_weights).zip(partitions)
            {
                let (e_eq_slice, t_eq_slice, z_eq_slice) =
                    (&direct.e[..], &direct.t[..], &direct.z[..]);
                let segments = &partition.segments;
                let first = segments.partition_point(|segment| segment.hi <= lo);
                for segment in segments.iter().skip(first) {
                    if segment.lo >= hi {
                        break;
                    }
                    let overlap = segment.lo.max(lo)..segment.hi.min(hi);
                    let weight_start = overlap
                        .start
                        .checked_sub(lo)
                        .ok_or(AkitaError::InvalidProof)?;
                    dispatch_segment_roles!(segment, Ok(()), |HAS_D, HAS_B, HAS_A| {
                        for_each_base_ring_segment_weight_typed::<E, HAS_D, HAS_B, HAS_A>(
                            overlap,
                            segment,
                            e_eq_slice,
                            t_eq_slice,
                            z_eq_slice,
                            &projection[2],
                            &projection[1],
                            &projection[0],
                            |offset, weight| {
                                let slot = weight_start
                                    .checked_add(offset)
                                    .and_then(|index| weights.get_mut(index))
                                    .ok_or(AkitaError::InvalidProof)?;
                                *slot += weight;
                                Ok(())
                            },
                        )
                    })?;
                }
            }
            let mut term = E::zero();
            for (ring, weight) in setup.iter().zip(weights) {
                if !weight.is_zero() {
                    term += eval_ring_at_pows_fast(ring, base_pows) * weight;
                }
            }
            Ok(acc + term)
        },
        |lhs, rhs| Ok(lhs + rhs)
    )
}
