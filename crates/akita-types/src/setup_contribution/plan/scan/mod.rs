use super::*;
use akita_algebra::ring::{eval_ring_at_pows_fast, scalar_powers};

struct CompiledSetupGroup<E> {
    required: usize,
    segments: Vec<GroupSetupSegment<E>>,
    d_eq: Vec<E>,
    b_eq: Vec<E>,
    a_eq: Vec<E>,
}

impl<E: FieldCore> SetupContributionPlan<E> {
    pub fn evaluate_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha: E,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let geometry = self.projection_geometry;
        let alpha_pows_a = scalar_powers(alpha, geometry.role_dims().d_a());
        let alpha_pows_b = scalar_powers(alpha, geometry.role_dims().d_b());
        let alpha_pows_d = scalar_powers(alpha, geometry.role_dims().d_d());
        let base_d = geometry.base_ring_dim();
        let base_pows = alpha_pows_d.get(..base_d).ok_or(AkitaError::InvalidProof)?;
        let a_projection = role_projection(&alpha_pows_a, base_pows, geometry.a_ratio())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "A alpha powers do not decompose over base dimension".into(),
                )
            })?;
        let b_projection = role_projection(&alpha_pows_b, base_pows, geometry.b_ratio())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "B alpha powers do not decompose over base dimension".into(),
                )
            })?;
        let d_projection = role_projection(&alpha_pows_d, base_pows, geometry.d_ratio())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "D alpha powers do not decompose over base dimension".into(),
                )
            })?;
        let compiled = self.compile_direct(alpha)?;

        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            base_d,
            |BASE_D| {
                self.evaluate_direct_typed::<F, BASE_D>(
                    setup,
                    base_pows,
                    &compiled,
                    &a_projection,
                    &b_projection,
                    &d_projection,
                )
            }
        )
    }

    fn compile_direct(&self, alpha: E) -> Result<Vec<CompiledSetupGroup<E>>, AkitaError> {
        let inner_lane_powers = super::structured::relation_lane_powers(
            alpha,
            self.common_coeff_count,
            self.inner_lane_count,
        )?;
        let outer_lane_powers = super::structured::relation_lane_powers(
            alpha,
            self.common_coeff_count,
            self.outer_lane_count,
        )?;
        let opening_lane_powers = super::structured::relation_lane_powers(
            alpha,
            self.common_coeff_count,
            self.opening_lane_count,
        )?;
        let d_weights = self
            .eq_tau1
            .get(self.d_row_start..self.d_row_start + self.d_rows)
            .ok_or(AkitaError::InvalidProof)?;

        self.groups
            .iter()
            .map(|group| {
                let d_eq = materialize_role_columns(
                    &group.d_spans,
                    group.d_col_range.len(),
                    &self.eq_window,
                    &opening_lane_powers,
                    None,
                )?;
                let b_eq = materialize_role_columns(
                    &group.b_spans,
                    group.t_cols,
                    &self.eq_window,
                    &outer_lane_powers,
                    None,
                )?;
                let a_eq = materialize_role_columns(
                    &group.a_spans,
                    group.z_cols,
                    &self.eq_window,
                    &inner_lane_powers,
                    Some(&self.fold_gadget),
                )?;
                let a_weights = self
                    .eq_tau1
                    .get(group.a_row_start..group.a_row_start + group.n_a)
                    .ok_or(AkitaError::InvalidProof)?;
                let b_weights = self
                    .eq_tau1
                    .get(group.b_row_start..group.b_row_start + group.n_b)
                    .ok_or(AkitaError::InvalidProof)?;
                let (required, segments) = super::segments::build_packed_segments(
                    group.d_col_range.start,
                    d_eq.len(),
                    group.t_cols,
                    group.z_cols,
                    group.n_a,
                    group.n_b,
                    a_weights,
                    b_weights,
                    d_weights,
                    self.d_rows,
                    self.d_physical_cols,
                    self.projection_geometry.a_ratio(),
                    self.projection_geometry.b_ratio(),
                    self.projection_geometry.d_ratio(),
                )?;
                Ok(CompiledSetupGroup {
                    required,
                    segments,
                    d_eq,
                    b_eq,
                    a_eq,
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_direct_typed<F, const BASE_D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        base_pows: &[E],
        groups: &[CompiledSetupGroup<E>],
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let required = self.projection_geometry.required();
        if base_pows.len() != BASE_D || groups.len() != self.groups.len() {
            return Err(AkitaError::InvalidProof);
        }
        let setup_len = setup.shared_matrix().total_ring_elements_at::<BASE_D>()?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        let setup_view = setup.shared_matrix().ring_view::<BASE_D>(1, setup_len)?;
        let setup_flat = setup_view.as_slice();
        let job_rings = super::segments::SETUP_SCAN_JOB_RINGS;
        let num_jobs = required.div_ceil(job_rings);
        let _span = tracing::info_span!(
            "setup_contribution_scan",
            required,
            groups = groups.len(),
            jobs = num_jobs,
            base_d = BASE_D,
        )
        .entered();
        cfg_try_fold_reduce!(
            0..num_jobs,
            E::zero,
            |acc, job| {
                let lo = job.checked_mul(job_rings).ok_or(AkitaError::InvalidProof)?;
                let hi = lo.saturating_add(job_rings).min(required);
                let setup = setup_flat.get(lo..hi).ok_or(AkitaError::InvalidProof)?;
                let mut weights = vec![E::zero(); setup.len()];
                for group in groups {
                    if group.required > required {
                        return Err(AkitaError::InvalidProof);
                    }
                    let first = group.segments.partition_point(|segment| segment.hi <= lo);
                    for segment in group.segments.iter().skip(first) {
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
                                &group.d_eq,
                                &group.b_eq,
                                &group.a_eq,
                                d_projection,
                                b_projection,
                                a_projection,
                                |offset, weight| {
                                    let slot = weights
                                        .get_mut(weight_start + offset)
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
}

pub(super) fn materialize_role_columns<E: FieldCore>(
    spans: &[SetupContributionSpan],
    column_count: usize,
    equality_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    lane_powers: &[E],
    fold_gadget: Option<&[E]>,
) -> Result<Vec<E>, AkitaError> {
    let mut weights = vec![E::zero(); column_count];
    for span in spans {
        let fold = match span.fold_digit {
            Some(digit) => Some(
                *fold_gadget
                    .and_then(|gadget| gadget.get(digit))
                    .ok_or(AkitaError::InvalidProof)?,
            ),
            None => None,
        };
        for offset in 0..span.len {
            let column = span
                .setup_start
                .checked_add(
                    offset
                        .checked_mul(span.setup_stride)
                        .ok_or(AkitaError::InvalidProof)?,
                )
                .ok_or(AkitaError::InvalidProof)?;
            let witness = span
                .witness_start
                .checked_add(
                    offset
                        .checked_mul(span.witness_stride)
                        .ok_or(AkitaError::InvalidProof)?,
                )
                .ok_or(AkitaError::InvalidProof)?;
            let equality = lane_powers.iter().copied().enumerate().try_fold(
                E::zero(),
                |sum, (lane, power)| {
                    let address = witness.checked_add(lane).ok_or(AkitaError::InvalidProof)?;
                    Ok(sum + equality_window.eval(address) * power)
                },
            )?;
            let slot = weights.get_mut(column).ok_or(AkitaError::InvalidProof)?;
            if let Some(fold) = fold {
                *slot -= equality * fold;
            } else {
                *slot += equality;
            }
        }
    }
    Ok(weights)
}
