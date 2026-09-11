use super::*;
use jolt_field::{Unreduced, Zero};

/// One distinct coefficient functional over the shared setup base rings.
///
/// Equal role functionals share a class, so their scalar weights are added
/// before evaluating a setup ring. Mixed role dimensions naturally form
/// separate classes and need no alternate scan implementation.
struct ReducedFunctionalClass<'a, E> {
    functional: &'a [E],
    ratio: usize,
}

impl<'a, E> ReducedFunctionalClass<'a, E> {
    fn chunk<const D: usize>(&self, base_idx: usize) -> Result<&'a [E], AkitaError> {
        let chunk = base_idx % self.ratio;
        let start = chunk.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidSetup("coefficient functional offset overflow".into())
        })?;
        let end = start.checked_add(D).ok_or_else(|| {
            AkitaError::InvalidSetup("coefficient functional extent overflow".into())
        })?;
        self.functional
            .get(start..end)
            .ok_or(AkitaError::InvalidProof)
    }
}

trait ProductSum<E: Unreduced> {
    fn zero() -> Self;
    fn add_product(&mut self, lhs: E, rhs: E);
    fn finish(self) -> E;
}

struct DelayedProductSum<E: Unreduced>(E::Product);

impl<E: Unreduced> ProductSum<E> for DelayedProductSum<E> {
    #[inline(always)]
    fn zero() -> Self {
        Self(E::Product::zero())
    }

    #[inline(always)]
    fn add_product(&mut self, lhs: E, rhs: E) {
        self.0 += lhs.mul_unreduced(rhs);
    }

    #[inline(always)]
    fn finish(self) -> E {
        E::reduce_product(self.0)
    }
}

struct CanonicalProductSum<E>(E);

impl<E: Unreduced> ProductSum<E> for CanonicalProductSum<E> {
    #[inline(always)]
    fn zero() -> Self {
        Self(E::zero())
    }

    #[inline(always)]
    fn add_product(&mut self, lhs: E, rhs: E) {
        self.0 += lhs * rhs;
    }

    #[inline(always)]
    fn finish(self) -> E {
        self.0
    }
}

struct ReducedScanGroup<'a, E> {
    e: &'a [E],
    t: &'a [E],
    z: &'a [E],
    role_ratios: [usize; 3],
    role_classes: [usize; 3],
}

type ReducedFunctionalClasses<'a, E> = (
    Vec<ReducedFunctionalClass<'a, E>>,
    Vec<ReducedScanGroup<'a, E>>,
);

impl<E: Field> SetupContributionPlan<E> {
    pub(super) fn evaluate_groups_reduced<F, const BASE_D: usize>(
        &self,
        setup_view: &RingMatrixView<'_, F, BASE_D>,
        weights: &[ReducedDirectScanWeights<E>],
    ) -> Result<E, AkitaError>
    where
        F: Field,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let (functional_classes, scan_groups) =
            self.reduced_functional_classes::<BASE_D>(weights)?;
        if E::SUM_IS_EXACT {
            self.evaluate_groups_reduced_with::<F, BASE_D, DelayedProductSum<E>>(
                setup_view,
                &functional_classes,
                &scan_groups,
            )
        } else {
            self.evaluate_groups_reduced_with::<F, BASE_D, CanonicalProductSum<E>>(
                setup_view,
                &functional_classes,
                &scan_groups,
            )
        }
    }

    fn reduced_functional_classes<'a, const BASE_D: usize>(
        &self,
        weights: &'a [ReducedDirectScanWeights<E>],
    ) -> Result<ReducedFunctionalClasses<'a, E>, AkitaError> {
        if self.groups.len() != weights.len() {
            return Err(AkitaError::InvalidSetup(
                "reduced setup scan group count is malformed".into(),
            ));
        }
        let capacity = checked::product([weights.len(), 3])
            .ok_or_else(|| AkitaError::InvalidSetup("reduced functional count overflow".into()))?;
        let mut classes = Vec::new();
        classes
            .try_reserve_exact(capacity)
            .map_err(|_| AkitaError::InvalidSetup("too many reduced functionals".into()))?;
        let mut scan_groups = Vec::new();
        scan_groups
            .try_reserve_exact(weights.len())
            .map_err(|_| AkitaError::InvalidSetup("too many reduced groups".into()))?;

        for (group, direct) in self.groups.iter().zip(weights) {
            let ratios = [group.a_ratio, group.b_ratio, group.d_ratio];
            let mut group_classes = [0usize; 3];
            for (role_index, (role, ratio)) in direct.roles.iter().zip(ratios).enumerate() {
                let expected_len = ratio.checked_mul(BASE_D).ok_or_else(|| {
                    AkitaError::InvalidSetup("reduced functional extent overflow".into())
                })?;
                if !ratio.is_power_of_two() || role.functional.len() != expected_len {
                    return Err(AkitaError::InvalidSetup(
                        "reduced functional does not match its setup projection".into(),
                    ));
                }
                let class = if let Some(class) =
                    classes
                        .iter()
                        .position(|candidate: &ReducedFunctionalClass<'_, E>| {
                            candidate.ratio == ratio
                                && candidate.functional == role.functional.as_ref()
                        }) {
                    class
                } else {
                    classes.push(ReducedFunctionalClass {
                        functional: &role.functional,
                        ratio,
                    });
                    classes.len() - 1
                };
                group_classes[role_index] = class;
            }
            scan_groups.push(ReducedScanGroup {
                e: &direct.weights.e,
                t: &direct.weights.t,
                z: &direct.weights.z,
                role_ratios: ratios,
                role_classes: group_classes,
            });
        }
        Ok((classes, scan_groups))
    }

    fn evaluate_groups_reduced_with<F, const BASE_D: usize, A>(
        &self,
        setup_view: &RingMatrixView<'_, F, BASE_D>,
        functional_classes: &[ReducedFunctionalClass<'_, E>],
        scan_groups: &[ReducedScanGroup<'_, E>],
    ) -> Result<E, AkitaError>
    where
        F: Field,
        E: ExtField<F> + MulBaseUnreduced<F>,
        A: ProductSum<E>,
    {
        let required = self.projection_geometry.required();
        if self.d_weights.len() != self.d_rows || scan_groups.len() != self.groups.len() {
            return Err(AkitaError::InvalidSetup(
                "cached setup scan geometry is malformed".into(),
            ));
        }
        let setup_flat = setup_view.as_slice();
        let job_rings = super::super::segments::SETUP_SCAN_JOB_RINGS;
        let num_jobs = required.div_ceil(job_rings);
        cfg_try_fold_reduce!(
            0..num_jobs,
            E::zero,
            |acc, job| {
                let lo = job.checked_mul(job_rings).ok_or(AkitaError::InvalidProof)?;
                let hi = lo
                    .checked_add(job_rings)
                    .ok_or(AkitaError::InvalidProof)?
                    .min(required);
                let setup = setup_flat.get(lo..hi).ok_or(AkitaError::InvalidProof)?;
                let mut segment_cursors = self
                    .groups
                    .iter()
                    .map(|group| group.segments.partition_point(|segment| segment.hi <= lo))
                    .collect::<Vec<_>>();
                let mut class_scalars = (0..functional_classes.len())
                    .map(|_| A::zero())
                    .collect::<Vec<_>>();
                let mut term = A::zero();
                for (offset, ring) in setup.iter().enumerate() {
                    let base_idx = lo.checked_add(offset).ok_or(AkitaError::InvalidProof)?;
                    for ((group, direct), cursor) in self
                        .groups
                        .iter()
                        .zip(scan_groups)
                        .zip(&mut segment_cursors)
                    {
                        while group
                            .segments
                            .get(*cursor)
                            .is_some_and(|segment| segment.hi <= base_idx)
                        {
                            *cursor += 1;
                        }
                        let Some(segment) = group.segments.get(*cursor) else {
                            continue;
                        };
                        if base_idx < segment.lo || base_idx >= segment.hi {
                            continue;
                        }
                        add_reduced_functional_products(
                            base_idx,
                            segment,
                            direct,
                            &mut class_scalars,
                        )?;
                    }
                    for (class, scalar) in functional_classes.iter().zip(&mut class_scalars) {
                        let scalar = std::mem::replace(scalar, A::zero()).finish();
                        if !scalar.is_zero() {
                            let functional = class.chunk::<BASE_D>(base_idx)?;
                            term.add_product(eval_ring_at_pows_fast(ring, functional), scalar);
                        }
                    }
                }
                Ok(acc + term.finish())
            },
            |lhs, rhs| Ok(lhs + rhs)
        )
    }
}

#[inline(always)]
fn add_reduced_functional_products<E, A>(
    base_idx: usize,
    segment: &GroupSetupSegment<E>,
    group: &ReducedScanGroup<'_, E>,
    output: &mut [A],
) -> Result<(), AkitaError>
where
    E: Field + Unreduced,
    A: ProductSum<E>,
{
    let [a_ratio, b_ratio, d_ratio] = group.role_ratios;
    let [a_class, b_class, d_class] = group.role_classes;
    if segment.has_d {
        let role_idx = projected_role_index(base_idx, d_ratio)?;
        let eq_idx = role_idx
            .checked_sub(segment.d_start_abs)
            .ok_or(AkitaError::InvalidProof)?;
        let equality = *group.e.get(eq_idx).ok_or(AkitaError::InvalidProof)?;
        output
            .get_mut(d_class)
            .ok_or(AkitaError::InvalidProof)?
            .add_product(segment.d_weight, equality);
    }
    if segment.has_b {
        let role_idx = projected_role_index(base_idx, b_ratio)?;
        let local = role_idx
            .checked_sub(segment.b_start_abs)
            .ok_or(AkitaError::InvalidProof)?;
        let output = output.get_mut(b_class).ok_or(AkitaError::InvalidProof)?;
        for term in segment.b_terms.iter() {
            let logical = term
                .logical_start
                .checked_add(local)
                .and_then(|index| group.t.get(index))
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            output.add_product(term.row_weight, logical);
        }
    }
    if segment.has_a {
        let role_idx = projected_role_index(base_idx, a_ratio)?;
        let eq_idx = role_idx
            .checked_sub(segment.a_start_abs)
            .ok_or(AkitaError::InvalidProof)?;
        let equality = *group.z.get(eq_idx).ok_or(AkitaError::InvalidProof)?;
        output
            .get_mut(a_class)
            .ok_or(AkitaError::InvalidProof)?
            .add_product(segment.a_row_weight, equality);
    }
    Ok(())
}

fn projected_role_index(base_idx: usize, ratio: usize) -> Result<usize, AkitaError> {
    if !ratio.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "setup role projection ratio must be a power of two".into(),
        ));
    }
    Ok(base_idx / ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{FpExt4, Prime32Offset99, Ring};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn extension(seed: u64) -> E {
        E::from_base_fn(|coordinate| F::from_u64(seed + coordinate as u64 * 17))
    }

    #[test]
    fn delayed_product_sum_matches_canonical_fp32() {
        const { assert!(E::SUM_IS_EXACT) };
        let mut delayed = DelayedProductSum::<E>::zero();
        let mut canonical = CanonicalProductSum::<E>::zero();
        for index in 0..4096 {
            let lhs = extension(3 * index + 1);
            let rhs = extension(5 * index + 7);
            delayed.add_product(lhs, rhs);
            canonical.add_product(lhs, rhs);
        }
        assert_eq!(delayed.finish(), canonical.finish());
    }
}
