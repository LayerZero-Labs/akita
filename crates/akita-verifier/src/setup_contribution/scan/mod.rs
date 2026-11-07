mod reduced;

use super::*;
use akita_algebra::cfg_try_fold_reduce;
use akita_algebra::ring::scalar_powers;
use kernels::{add_segment_weights, CanonicalProductSum, DelayedProductSum, ProductSum};
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
    let required = plan.projection_geometry().required();
    let logical_group_rings = partitions.iter().fold(0usize, |sum, partition| {
        sum.saturating_add(partition.required)
    });
    let _span = tracing::info_span!(
        "setup_contribution_scan",
        required,
        groups = plan.groups().len(),
        logical_group_rings,
        physical_ring_evaluations = required,
        jobs = required.div_ceil(super::segments::SETUP_SCAN_JOB_RINGS),
        fused_groups = plan.groups().len() > 1,
        reduced_evaluation = matches!(&kernel, DirectScanKernel::ReducedEvaluation { .. }),
        base_d = BASE_D,
        final_a_ratio = plan.projection_geometry().a_ratio(),
        final_b_ratio = plan.projection_geometry().b_ratio(),
        final_d_ratio = plan.projection_geometry().d_ratio()
    )
    .entered();
    let setup_len = setup.shared_matrix().num_field_elements() / BASE_D;
    if plan.projection_geometry().base_ring_dim() != BASE_D
        || required > setup_len
        || partitions
            .iter()
            .any(|partition| partition.required > required)
    {
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
    if projections.len() != plan.groups().len()
        || weights.len() != plan.groups().len()
        || partitions.len() != plan.groups().len()
    {
        return Err(AkitaError::InvalidSetup(
            "cached setup scan geometry is malformed".into(),
        ));
    }
    let groups = plan
        .groups()
        .iter()
        .zip(projections)
        .zip(weights)
        .zip(partitions)
        .map(|(((group, projection), weights), partition)| {
            ScanGroup::new(
                group,
                partition,
                weights,
                [&projection[0], &projection[1], &projection[2]],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    evaluate_weighted_scan(plan, &setup_view, base_powers, &groups)
}

/// One group's view of a weighted setup scan.
pub(super) struct ScanGroup<'a, E: Field> {
    partition: &'a GroupScanPartition<E>,
    weights: &'a DirectScanWeights<E>,
    /// A, B, and D role projections onto the base ring.
    projections: [&'a RoleProjection<E>; 3],
}

impl<'a, E: Field> ScanGroup<'a, E> {
    pub(super) fn new(
        group: &SetupContributionGroupPlan<E>,
        partition: &'a GroupScanPartition<E>,
        weights: &'a DirectScanWeights<E>,
        projections: [&'a RoleProjection<E>; 3],
    ) -> Result<Self, AkitaError> {
        if group.d_col_range().len() != weights.e.len() {
            return Err(AkitaError::InvalidSetup(
                "cached setup scan geometry is malformed".into(),
            ));
        }
        Ok(Self {
            partition,
            weights,
            projections,
        })
    }
}

/// `Σ_b w(b)·⟨ring_b, pows⟩` over the plan's base rings, where `w(b)` adds
/// every group's D/B/A weight at base ring `b`.
///
/// Each job assembles its rings' weights segment by segment, a block of
/// [`WEIGHT_BLOCK_RINGS`] at a time, then adds `w(b)·⟨ring_b, pows⟩` for every
/// nonzero-weight ring into one product sum. A job sums at most
/// `SETUP_SCAN_JOB_RINGS` products, inside the headroom `SUM_IS_EXACT` grants,
/// so the delayed sum equals the reduced one.
pub(super) fn evaluate_weighted_scan<F, E, const BASE_D: usize>(
    plan: &SetupContributionPlan<E>,
    setup_view: &RingMatrixView<'_, F, BASE_D>,
    pows: &[E],
    groups: &[ScanGroup<'_, E>],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    if E::SUM_IS_EXACT {
        evaluate_weighted_scan_with::<F, E, BASE_D, DelayedProductSum<E>>(
            plan, setup_view, pows, groups,
        )
    } else {
        evaluate_weighted_scan_with::<F, E, BASE_D, CanonicalProductSum<E>>(
            plan, setup_view, pows, groups,
        )
    }
}

/// Base rings whose weights a scan job assembles before contracting them.
///
/// The block's weight slots live on the stack and stay in the L1 cache: a
/// canonical fp128 slot is 16 bytes (2 KiB a block), and a delayed product sum
/// for the quartic fp32 or quadratic fp64 extension is 64 bytes (8 KiB).
const WEIGHT_BLOCK_RINGS: usize = 128;

fn evaluate_weighted_scan_with<F, E, const BASE_D: usize, A>(
    plan: &SetupContributionPlan<E>,
    setup_view: &RingMatrixView<'_, F, BASE_D>,
    pows: &[E],
    groups: &[ScanGroup<'_, E>],
) -> Result<E, AkitaError>
where
    F: Field,
    E: ExtField<F> + MulBaseUnreduced<F>,
    A: ProductSum<E>,
{
    let setup_flat = setup_view.as_slice();
    let required = plan.projection_geometry().required();
    let pows: &[E; BASE_D] = pows.try_into().map_err(|_| AkitaError::InvalidSize {
        expected: BASE_D,
        actual: pows.len(),
    })?;
    if plan.d_weights().len() != plan.d_rows() {
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
            let mut sum = A::zero();
            let mut slots: [A; WEIGHT_BLOCK_RINGS] = std::array::from_fn(|_| A::zero());
            for (block, block_setup) in setup.chunks(WEIGHT_BLOCK_RINGS).enumerate() {
                let block_lo = block
                    .checked_mul(WEIGHT_BLOCK_RINGS)
                    .and_then(|offset| offset.checked_add(lo))
                    .ok_or(AkitaError::InvalidProof)?;
                let block_hi = block_lo
                    .checked_add(block_setup.len())
                    .ok_or(AkitaError::InvalidProof)?;
                let slots = slots
                    .get_mut(..block_setup.len())
                    .ok_or(AkitaError::InvalidProof)?;
                for group in groups {
                    let segments = &group.partition.segments;
                    let first = segments.partition_point(|segment| segment.hi <= block_lo);
                    for segment in segments.iter().skip(first) {
                        if segment.lo >= block_hi {
                            break;
                        }
                        let overlap = segment.lo.max(block_lo)..segment.hi.min(block_hi);
                        let slots = slots
                            .get_mut(overlap.start - block_lo..overlap.end - block_lo)
                            .ok_or(AkitaError::InvalidProof)?;
                        add_segment_weights(
                            slots,
                            overlap,
                            segment,
                            group.weights,
                            group.projections,
                        )?;
                    }
                }
                // Rings with a zero weight contribute nothing.
                for (ring, slot) in block_setup.iter().zip(slots) {
                    let weight = std::mem::replace(slot, A::zero()).finish();
                    if !weight.is_zero() {
                        sum.add_product(weight, E::dot_base(pows, ring.coefficients()));
                    }
                }
            }
            Ok(acc + sum.finish())
        },
        |lhs, rhs| Ok(lhs + rhs)
    )
}
