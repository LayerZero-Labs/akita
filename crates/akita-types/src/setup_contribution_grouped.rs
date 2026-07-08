use super::{
    checked_add, checked_mul, checked_slice, push_role_boundaries, SetupContributionPlan,
    SetupContributionPlanInputs, POSSIBLE_CARRIES,
};
use crate::layout::flat_matrix::FlatRingMatrixView;
use crate::proof::AkitaExpandedSetup;
use crate::WitnessChunkLayout;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{eq_eval_at_index, high_eq_window};
use akita_algebra::ring::eval_flat_ring_at_pows_fast;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase, MulBaseUnreduced};

#[derive(Clone)]
pub struct SetupContributionGroupInputs {
    pub e_col_offset: usize,
    pub num_claims: usize,
    pub num_blocks: usize,
    pub block_len: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub log_basis: u32,
    pub n_a: usize,
    pub n_b: usize,
    pub t_cols_per_vector: usize,
    pub a_row_start: usize,
    pub b_row_start: usize,
    pub blocks_per_chunk: usize,
    pub chunks: Vec<WitnessChunkLayout>,
}

pub struct GroupedSetupContributionPlan<E> {
    pub(super) groups: Vec<GroupSetupContributionPlan<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
}

/// Tau1-derived grouped setup weights cached at ring-switch prepare time.
#[derive(Clone)]
pub struct GroupedSetupContributionStatic<E> {
    pub(super) groups: Vec<GroupSetupContributionStatic<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
    pub(super) d_weights: Vec<E>,
}

#[derive(Clone)]
pub(super) struct GroupSetupContributionStatic<E> {
    pub(super) e_col_offset: usize,
    pub(super) t_cols: usize,
    pub(super) z_cols: usize,
    pub(super) n_a: usize,
    pub(super) n_b: usize,
    pub(super) a_weights: Vec<E>,
    pub(super) b_weights: Vec<E>,
}

pub(super) struct GroupSetupContributionPlan<E> {
    pub(super) e_col_offset: usize,
    pub(super) t_cols: usize,
    pub(super) z_cols: usize,
    pub(super) n_a: usize,
    pub(super) n_b: usize,
    pub(super) e_eq_slice: Vec<E>,
    pub(super) t_eq_slice: Vec<E>,
    pub(super) z_eq_slice: Vec<E>,
    pub(super) a_weights: Vec<E>,
    pub(super) b_weights: Vec<E>,
    pub(super) d_weights: Vec<E>,
}

impl<E: FieldCore> SetupContributionPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_grouped<F>(
        inputs: &SetupContributionPlanInputs<E>,
        full_vec_randomness: &[E],
        eq_low: Option<&[E]>,
        z_block_low_eq: Option<&[E]>,
        fold_gadget: Option<&[F]>,
        groups: &[SetupContributionGroupInputs],
        d_row_start: usize,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<GroupedSetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let static_plan =
            Self::prepare_grouped_static(inputs, groups, d_row_start, d_rows, d_physical_cols)?;
        Self::finish_grouped_plan::<F>(
            &static_plan,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            fold_gadget,
            groups,
        )
    }

    pub fn prepare_grouped_static(
        inputs: &SetupContributionPlanInputs<E>,
        groups: &[SetupContributionGroupInputs],
        d_row_start: usize,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<GroupedSetupContributionStatic<E>, AkitaError> {
        let d_weights = if d_rows == 0 {
            Vec::new()
        } else {
            checked_slice(&inputs.eq_tau1, d_row_start, d_rows, "grouped D rows")?.to_vec()
        };
        let num_groups = groups.len();
        let static_groups = groups
            .iter()
            .map(|group| {
                validate_group_chunk_layout(group, num_groups)?;
                let t_cols =
                    checked_mul(group.num_claims, group.t_cols_per_vector, "grouped B width")?;
                let z_cols = checked_mul(group.block_len, group.depth_commit, "grouped Z range")?;
                let a_weights = checked_slice(
                    &inputs.eq_tau1,
                    group.a_row_start,
                    group.n_a,
                    "grouped A rows",
                )?
                .to_vec();
                let b_weights = checked_slice(
                    &inputs.eq_tau1,
                    group.b_row_start,
                    group.n_b,
                    "grouped B rows",
                )?
                .to_vec();
                Ok(GroupSetupContributionStatic {
                    e_col_offset: group.e_col_offset,
                    t_cols,
                    z_cols,
                    n_a: group.n_a,
                    n_b: group.n_b,
                    a_weights,
                    b_weights,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(GroupedSetupContributionStatic {
            groups: static_groups,
            d_rows,
            d_physical_cols,
            d_weights,
        })
    }

    pub fn finish_grouped_plan<F>(
        static_plan: &GroupedSetupContributionStatic<E>,
        full_vec_randomness: &[E],
        eq_low: Option<&[E]>,
        z_block_low_eq: Option<&[E]>,
        fold_gadget: Option<&[F]>,
        groups: &[SetupContributionGroupInputs],
    ) -> Result<GroupedSetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        if static_plan.groups.len() != groups.len() {
            return Err(AkitaError::InvalidSize {
                expected: groups.len(),
                actual: static_plan.groups.len(),
            });
        }
        let dynamic_groups = static_plan
            .groups
            .iter()
            .zip(groups)
            .map(|(static_group, group)| {
                let e_eq_slice = grouped_e_setup_weights(group, full_vec_randomness, eq_low)?;
                let t_eq_slice = grouped_t_setup_weights(group, full_vec_randomness, eq_low)?;
                let z_eq_slice = grouped_z_setup_weights::<F, E>(
                    group,
                    full_vec_randomness,
                    z_block_low_eq,
                    fold_gadget,
                )?;
                Ok(GroupSetupContributionPlan {
                    e_col_offset: static_group.e_col_offset,
                    t_cols: static_group.t_cols,
                    z_cols: static_group.z_cols,
                    n_a: static_group.n_a,
                    n_b: static_group.n_b,
                    e_eq_slice,
                    t_eq_slice,
                    z_eq_slice,
                    a_weights: static_group.a_weights.clone(),
                    b_weights: static_group.b_weights.clone(),
                    d_weights: static_plan.d_weights.clone(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(GroupedSetupContributionPlan {
            groups: dynamic_groups,
            d_rows: static_plan.d_rows,
            d_physical_cols: static_plan.d_physical_cols,
        })
    }
}

impl<E: FieldCore> SetupContributionPlan<E> {
    #[cfg(test)]
    pub(super) fn from_single_grouped(
        grouped: &GroupedSetupContributionPlan<E>,
    ) -> Result<Self, AkitaError> {
        if grouped.groups.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "flat setup contribution evaluation requires exactly one group".into(),
            ));
        }
        let group = &grouped.groups[0];
        if group.d_weights.len() != grouped.d_rows {
            return Err(AkitaError::InvalidSize {
                expected: grouped.d_rows,
                actual: group.d_weights.len(),
            });
        }
        if group.a_weights.len() != group.n_a {
            return Err(AkitaError::InvalidSize {
                expected: group.n_a,
                actual: group.a_weights.len(),
            });
        }
        if group.b_weights.len() != group.n_b {
            return Err(AkitaError::InvalidSize {
                expected: group.n_b,
                actual: group.b_weights.len(),
            });
        }
        if group.e_col_offset > grouped.d_physical_cols
            || group.e_eq_slice.len() > grouped.d_physical_cols - group.e_col_offset
        {
            return Err(AkitaError::InvalidSetup(
                "grouped D setup weights exceed physical D width".into(),
            ));
        }

        let d_stride = grouped.d_physical_cols;
        let b_stride = group.t_cols;
        let z_range = group.z_cols;
        if group.t_eq_slice.len() != b_stride {
            return Err(AkitaError::InvalidSize {
                expected: b_stride,
                actual: group.t_eq_slice.len(),
            });
        }
        if group.z_eq_slice.len() != z_range {
            return Err(AkitaError::InvalidSize {
                expected: z_range,
                actual: group.z_eq_slice.len(),
            });
        }

        let d_required = checked_mul(grouped.d_rows, d_stride, "D setup footprint")?;
        let b_required = checked_mul(group.n_b, b_stride, "B setup footprint")?;
        let a_required = checked_mul(group.n_a, z_range, "A setup footprint")?;
        let required = d_required.max(b_required).max(a_required);
        if required == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator requires a non-empty packed footprint".into(),
            ));
        }

        let mut e_eq_slice = vec![E::zero(); d_stride];
        let e_start = group.e_col_offset;
        let e_end = e_start + group.e_eq_slice.len();
        e_eq_slice[e_start..e_end].copy_from_slice(&group.e_eq_slice);

        let b_weights_by_row = group
            .b_weights
            .iter()
            .copied()
            .map(|weight| vec![weight])
            .collect();

        let mut endpoints = Vec::with_capacity(grouped.d_rows + group.n_b + group.n_a + 2);
        endpoints.push(0);
        endpoints.push(required);
        push_role_boundaries(&mut endpoints, grouped.d_rows, d_stride, "D")?;
        push_role_boundaries(&mut endpoints, group.n_b, b_stride, "B")?;
        push_role_boundaries(&mut endpoints, group.n_a, z_range, "A")?;
        endpoints.sort_unstable();
        endpoints.dedup();

        Ok(Self {
            required,
            d_stride,
            b_stride,
            z_range,
            d_required,
            b_required,
            a_required,
            e_eq_slice,
            t_eq_slice_per_group: vec![group.t_eq_slice.clone()],
            z_eq_slice: group.z_eq_slice.clone(),
            d_weights: group.d_weights.clone(),
            b_weights_by_row,
            a_weights: group.a_weights.clone(),
            endpoints,
        })
    }
}

impl<E: FieldCore> GroupedSetupContributionPlan<E> {
    pub fn evaluate_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let d_a = alpha_pows_a.len();
        let d_b = alpha_pows_b.len();
        let d_d = alpha_pows_d.len();
        if d_a == 0 || d_b == 0 || d_d == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup contribution role alpha powers must be non-empty".into(),
            ));
        }

        if self.groups.len() == 1 {
            self.evaluate_packed_direct(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d, d_a)
        } else {
            self.evaluate_direct_by_rows(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d, d_a)
        }
    }

    fn evaluate_packed_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
        d_a: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let mut acc = E::zero();
        for group in &self.groups {
            acc += group.evaluate_packed_direct(
                setup,
                alpha_pows_a,
                alpha_pows_b,
                alpha_pows_d,
                d_a,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(acc)
    }

    pub(super) fn evaluate_direct_by_rows<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
        d_a: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let d_d = alpha_pows_d.len();
        let d_b = alpha_pows_b.len();
        let mut acc = E::zero();
        if self.d_rows != 0 {
            let d_view =
                setup
                    .shared_matrix
                    .ring_view_dyn(self.d_rows, self.d_physical_cols, d_d)?;
            for group in &self.groups {
                for (row_idx, &row_weight) in group.d_weights.iter().enumerate() {
                    if row_weight.is_zero() {
                        continue;
                    }
                    let row = d_view.row_flat(row_idx)?;
                    acc += evaluate_weighted_setup_row::<F, E>(
                        row,
                        group.e_col_offset,
                        &group.e_eq_slice,
                        row_weight,
                        alpha_pows_d,
                    )?;
                }
            }
        }

        for group in &self.groups {
            let a_view = setup
                .shared_matrix
                .ring_view_dyn(group.n_a, group.z_cols, d_a)?;
            for (row_idx, &row_weight) in group.a_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = a_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    &group.z_eq_slice,
                    row_weight,
                    alpha_pows_a,
                )?;
            }

            let b_view = setup
                .shared_matrix
                .ring_view_dyn(group.n_b, group.t_cols, d_b)?;
            for (row_idx, &row_weight) in group.b_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = b_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    &group.t_eq_slice,
                    row_weight,
                    alpha_pows_b,
                )?;
            }
        }

        Ok(acc)
    }
}

impl<E: FieldCore> GroupSetupContributionPlan<E> {
    fn evaluate_packed_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
        d_a: usize,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let d_d = alpha_pows_d.len();
        let d_b = alpha_pows_b.len();
        let d_view = if d_rows != 0 && !self.e_eq_slice.is_empty() {
            Some(
                setup
                    .shared_matrix
                    .ring_view_dyn(d_rows, d_physical_cols, d_d)?,
            )
        } else {
            None
        };
        let b_view = setup
            .shared_matrix
            .ring_view_dyn(self.n_b, self.t_cols, d_b)?;
        let a_view = setup
            .shared_matrix
            .ring_view_dyn(self.n_a, self.z_cols, d_a)?;

        let (required, segments) = self.packed_segments(d_rows, d_physical_cols)?;
        let setup_len = setup.shared_matrix().total_ring_elements_at_dyn(d_a)?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected grouped verifier layout".into(),
            ));
        }

        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        packed_group_slice_inner_sum::<F, E, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            d_view.as_ref(),
                            d_physical_cols,
                            d_d,
                            &b_view,
                            self.t_cols,
                            &a_view,
                            self.z_cols,
                            d_a,
                            alpha_pows_a,
                            alpha_pows_b,
                            alpha_pows_d,
                            segment.d_start_abs,
                            segment.d_weight,
                            &self.e_eq_slice,
                            segment.b_start_abs,
                            segment.b_weight,
                            &self.t_eq_slice,
                            segment.a_start_abs,
                            segment.a_weight,
                            &self.z_eq_slice,
                        )
                    };
                }

                Ok(match (segment.has_d, segment.has_b, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    (false, false, false) => Ok(E::zero()),
                }?)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;

        Ok(segment_sums.into_iter().sum())
    }

    fn packed_segments(
        &self,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<(usize, Vec<GroupSetupSegment<E>>), AkitaError> {
        if self.d_weights.len() != d_rows {
            return Err(AkitaError::InvalidSize {
                expected: d_rows,
                actual: self.d_weights.len(),
            });
        }
        if self.a_weights.len() != self.n_a {
            return Err(AkitaError::InvalidSize {
                expected: self.n_a,
                actual: self.a_weights.len(),
            });
        }
        if self.b_weights.len() != self.n_b {
            return Err(AkitaError::InvalidSize {
                expected: self.n_b,
                actual: self.b_weights.len(),
            });
        }
        if self.t_eq_slice.len() != self.t_cols {
            return Err(AkitaError::InvalidSize {
                expected: self.t_cols,
                actual: self.t_eq_slice.len(),
            });
        }
        if self.z_eq_slice.len() != self.z_cols {
            return Err(AkitaError::InvalidSize {
                expected: self.z_cols,
                actual: self.z_eq_slice.len(),
            });
        }
        let e_end = checked_add(
            self.e_col_offset,
            self.e_eq_slice.len(),
            "grouped D setup footprint",
        )?;
        if e_end > d_physical_cols {
            return Err(AkitaError::InvalidSetup(
                "grouped D setup weights exceed physical D width".into(),
            ));
        }

        let d_required = checked_mul(d_rows, d_physical_cols, "grouped D setup footprint")?;
        let b_required = checked_mul(self.n_b, self.t_cols, "grouped B setup footprint")?;
        let a_required = checked_mul(self.n_a, self.z_cols, "grouped A setup footprint")?;
        let required = d_required.max(b_required).max(a_required);

        let mut endpoints = Vec::new();
        endpoints.push(0);
        endpoints.push(required);
        push_group_d_boundaries(
            &mut endpoints,
            d_rows,
            d_physical_cols,
            self.e_col_offset,
            self.e_eq_slice.len(),
        )?;
        push_role_boundaries(&mut endpoints, self.n_b, self.t_cols, "B")?;
        push_role_boundaries(&mut endpoints, self.n_a, self.z_cols, "A")?;
        endpoints.sort_unstable();
        endpoints.dedup();

        let segments = (0..endpoints.len().saturating_sub(1))
            .filter_map(|idx| {
                let lo = endpoints[idx];
                let hi = endpoints[idx + 1];
                if lo == hi {
                    return None;
                }

                let has_d =
                    if d_physical_cols == 0 || self.e_eq_slice.is_empty() || lo >= d_required {
                        false
                    } else {
                        let d_col = lo % d_physical_cols;
                        d_col >= self.e_col_offset && d_col < e_end
                    };
                let d_row = if has_d { lo / d_physical_cols } else { 0 };
                let d_start_abs = if has_d {
                    d_row * d_physical_cols + self.e_col_offset
                } else {
                    0
                };
                let d_weight = if has_d {
                    self.d_weights[d_row]
                } else {
                    E::zero()
                };

                let has_b = self.t_cols != 0 && lo < b_required;
                let b_row = if has_b { lo / self.t_cols } else { 0 };
                let b_start_abs = if has_b { b_row * self.t_cols } else { 0 };
                let b_weight = if has_b {
                    self.b_weights[b_row]
                } else {
                    E::zero()
                };

                let has_a = self.z_cols != 0 && lo < a_required;
                let a_row = if has_a { lo / self.z_cols } else { 0 };
                let a_start_abs = if has_a { a_row * self.z_cols } else { 0 };
                let a_weight = if has_a {
                    self.a_weights[a_row]
                } else {
                    E::zero()
                };

                if !has_d && !has_b && !has_a {
                    return None;
                }

                Some(GroupSetupSegment {
                    lo,
                    hi,
                    has_d,
                    d_start_abs,
                    d_weight,
                    has_b,
                    b_start_abs,
                    b_weight,
                    has_a,
                    a_start_abs,
                    a_weight,
                })
            })
            .collect();

        Ok((required, segments))
    }
}

struct GroupSetupSegment<E> {
    lo: usize,
    hi: usize,
    has_d: bool,
    d_start_abs: usize,
    d_weight: E,
    has_b: bool,
    b_start_abs: usize,
    b_weight: E,
    has_a: bool,
    a_start_abs: usize,
    a_weight: E,
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_group_slice_inner_sum<F, E, const HAS_D: bool, const HAS_B: bool, const HAS_A: bool>(
    range: std::ops::Range<usize>,
    d_view: Option<&FlatRingMatrixView<'_, F>>,
    d_physical_cols: usize,
    d_d: usize,
    b_view: &FlatRingMatrixView<'_, F>,
    t_cols: usize,
    a_view: &FlatRingMatrixView<'_, F>,
    z_cols: usize,
    d_a: usize,
    alpha_pows_a: &[E],
    alpha_pows_b: &[E],
    alpha_pows_d: &[E],
    d_start: usize,
    d_weight: E,
    e_eq: &[E],
    b_start: usize,
    b_weight: E,
    t_eq: &[E],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let d_b = b_view.ring_d();
    let mut acc = E::zero();
    for lambda in range {
        if HAS_D {
            let eq_w = d_weight * e_eq[lambda - d_start];
            if !eq_w.is_zero() {
                let d_view = d_view.ok_or_else(|| {
                    AkitaError::InvalidSetup("grouped packed D scan missing D view".into())
                })?;
                let d_row = lambda / d_physical_cols;
                let d_col = lambda % d_physical_cols;
                let row = d_view.row_flat(d_row)?;
                let coeff_start = checked_mul(d_col, d_d, "grouped packed D coeff start")?;
                let coeffs = checked_slice(row, coeff_start, d_d, "grouped packed D coeffs")?;
                acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_d) * eq_w;
            }
        }
        if HAS_B {
            let eq_w = b_weight * t_eq[lambda - b_start];
            if !eq_w.is_zero() {
                let b_row = lambda / t_cols;
                let b_col = lambda % t_cols;
                let row = b_view.row_flat(b_row)?;
                let coeff_start = checked_mul(b_col, d_b, "grouped packed B coeff start")?;
                let coeffs = checked_slice(row, coeff_start, d_b, "grouped packed B coeffs")?;
                acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_b) * eq_w;
            }
        }
        if HAS_A {
            let eq_w = a_weight * z_eq[lambda - a_start];
            if !eq_w.is_zero() {
                let a_row = lambda / z_cols;
                let a_col = lambda % z_cols;
                let row = a_view.row_flat(a_row)?;
                let coeff_start = checked_mul(a_col, d_a, "grouped packed A coeff start")?;
                let coeffs = checked_slice(row, coeff_start, d_a, "grouped packed A coeffs")?;
                acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_a) * eq_w;
            }
        }
    }
    Ok(acc)
}

fn validate_group_chunk_layout(
    group: &SetupContributionGroupInputs,
    num_groups: usize,
) -> Result<(), AkitaError> {
    if group.chunks.is_empty()
        || group.blocks_per_chunk == 0
        || !group.blocks_per_chunk.is_power_of_two()
    {
        return Err(AkitaError::InvalidSetup(
            "malformed grouped witness chunk layout".into(),
        ));
    }
    if checked_mul(
        group.chunks.len(),
        group.blocks_per_chunk,
        "grouped chunk block coverage",
    )? != group.num_blocks
    {
        return Err(AkitaError::InvalidSetup(
            "grouped witness chunk windows do not tile num_blocks".into(),
        ));
    }
    if group.chunks.len() > 1 && num_groups != 1 {
        return Err(AkitaError::InvalidSetup(
            "multi-chunk grouped setup contribution requires exactly one group".into(),
        ));
    }
    Ok(())
}

fn grouped_e_setup_weights<E: FieldCore>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
    eq_low: Option<&[E]>,
) -> Result<Vec<E>, AkitaError> {
    let block_bits = group.blocks_per_chunk.trailing_zeros() as usize;
    if block_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: block_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let eq_low_storage;
    let eq_low = if let Some(precomputed) = eq_low {
        precomputed
    } else {
        eq_low_storage = EqPolynomial::evals(&full_vec_randomness[..block_bits])?;
        &eq_low_storage
    };
    if eq_low.len() < group.blocks_per_chunk {
        return Err(AkitaError::InvalidSize {
            expected: group.blocks_per_chunk,
            actual: eq_low.len(),
        });
    }
    let high_challenges = &full_vec_randomness[block_bits..];
    let high_len = checked_mul(group.num_claims, group.depth_open, "grouped D high width")?;
    let eq_high_by_chunk: Vec<Vec<E>> = group
        .chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_e >> block_bits, high_len))
        .collect();
    let low_mask = group.blocks_per_chunk - 1;
    let total_blocks = checked_mul(group.num_claims, group.num_blocks, "grouped D blocks")?;
    let e_cols = checked_mul(total_blocks, group.depth_open, "grouped D columns")?;
    Ok(cfg_into_iter!(0..e_cols)
        .map(|local_col| {
            let flat_block = local_col / group.depth_open;
            let digit = local_col % group.depth_open;
            let claim_idx = flat_block / group.num_blocks;
            let global_block_idx = flat_block % group.num_blocks;
            let chunk_idx = global_block_idx / group.blocks_per_chunk;
            let block_idx = global_block_idx % group.blocks_per_chunk;
            let chunk = &group.chunks[chunk_idx];
            let eq_high = &eq_high_by_chunk[chunk_idx];
            let offset_low = chunk.offset_e & low_mask;
            let shifted = offset_low + block_idx;
            let low_idx = shifted & low_mask;
            let carry = shifted >> block_bits;
            let high_idx = digit * group.num_claims + claim_idx + carry;
            eq_low[low_idx] * eq_high[high_idx]
        })
        .collect())
}

fn grouped_t_setup_weights<E: FieldCore>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
    eq_low: Option<&[E]>,
) -> Result<Vec<E>, AkitaError> {
    let block_bits = group.blocks_per_chunk.trailing_zeros() as usize;
    if block_bits > full_vec_randomness.len() {
        return Err(AkitaError::InvalidSize {
            expected: block_bits,
            actual: full_vec_randomness.len(),
        });
    }
    let eq_low_storage;
    let eq_low = if let Some(precomputed) = eq_low {
        precomputed
    } else {
        eq_low_storage = EqPolynomial::evals(&full_vec_randomness[..block_bits])?;
        &eq_low_storage
    };
    if eq_low.len() < group.blocks_per_chunk {
        return Err(AkitaError::InvalidSize {
            expected: group.blocks_per_chunk,
            actual: eq_low.len(),
        });
    }
    let high_challenges = &full_vec_randomness[block_bits..];
    let high_len = checked_mul(
        checked_mul(group.num_claims, group.depth_open, "grouped B high width")?,
        group.n_a,
        "grouped B high width",
    )?;
    let eq_high_by_chunk: Vec<Vec<E>> = group
        .chunks
        .iter()
        .map(|chunk| high_eq_window(high_challenges, chunk.offset_t >> block_bits, high_len))
        .collect();
    let low_mask = group.blocks_per_chunk - 1;
    let t_compound_per_block =
        checked_mul(group.n_a, group.depth_open, "grouped B compound stride")?;
    let t_cols = checked_mul(group.num_claims, group.t_cols_per_vector, "grouped B width")?;
    Ok(cfg_into_iter!(0..t_cols)
        .map(|local_col| {
            let t_vector_idx = local_col / group.t_cols_per_vector;
            let phys_claim_offset = local_col % group.t_cols_per_vector;
            let global_block_idx = phys_claim_offset / t_compound_per_block;
            let chunk_idx = global_block_idx / group.blocks_per_chunk;
            let block_idx = global_block_idx % group.blocks_per_chunk;
            let chunk = &group.chunks[chunk_idx];
            let eq_high = &eq_high_by_chunk[chunk_idx];
            let offset_low = chunk.offset_t & low_mask;
            let compound = phys_claim_offset % t_compound_per_block;
            let shifted = offset_low + block_idx;
            let low_idx = shifted & low_mask;
            let carry = shifted >> block_bits;
            let high_idx = compound * group.num_claims + t_vector_idx + carry;
            eq_low[low_idx] * eq_high[high_idx]
        })
        .collect())
}

fn grouped_z_setup_weights<F, E>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
    z_block_low_eq: Option<&[E]>,
    fold_gadget: Option<&[F]>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: MulBase<F>,
{
    let fold_gadget_storage;
    let fold_gadget = if let Some(fold_gadget) = fold_gadget {
        if fold_gadget.len() < group.depth_fold {
            return Err(AkitaError::InvalidSize {
                expected: group.depth_fold,
                actual: fold_gadget.len(),
            });
        }
        fold_gadget
    } else {
        fold_gadget_storage = crate::gadget_row_scalars::<F>(group.depth_fold, group.log_basis);
        &fold_gadget_storage
    };
    let z_range = checked_mul(group.block_len, group.depth_commit, "grouped Z range")?;
    let mut z_eq_slice = vec![E::zero(); z_range];
    for chunk in &group.chunks {
        let per_chunk = grouped_z_setup_weights_for_offset::<F, E>(
            group,
            full_vec_randomness,
            z_block_low_eq,
            fold_gadget,
            chunk.offset_z,
            z_range,
        )?;
        for (dst, src) in z_eq_slice.iter_mut().zip(per_chunk) {
            *dst += src;
        }
    }
    Ok(z_eq_slice)
}

fn grouped_z_setup_weights_for_offset<F, E>(
    group: &SetupContributionGroupInputs,
    full_vec_randomness: &[E],
    z_block_low_eq: Option<&[E]>,
    fold_gadget: &[F],
    offset_z: usize,
    z_range: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: MulBase<F>,
{
    if group.block_len.is_power_of_two() {
        let z_bits = group.block_len.trailing_zeros() as usize;
        if z_bits > full_vec_randomness.len() {
            return Err(AkitaError::InvalidSize {
                expected: z_bits,
                actual: full_vec_randomness.len(),
            });
        }
        let eq_low_storage;
        let eq_low = if let Some(precomputed) = z_block_low_eq {
            precomputed
        } else {
            eq_low_storage = EqPolynomial::evals(&full_vec_randomness[..z_bits])?;
            &eq_low_storage
        };
        if eq_low.len() < group.block_len {
            return Err(AkitaError::InvalidSize {
                expected: group.block_len,
                actual: eq_low.len(),
            });
        }
        let high_challenges = &full_vec_randomness[z_bits..];
        let high_len = checked_mul(group.depth_commit, group.depth_fold, "grouped Z high width")?;
        let eq_high = high_eq_window(high_challenges, offset_z >> z_bits, high_len);
        let low_mask = group.block_len - 1;
        let offset_low = offset_z & low_mask;
        let s_per_dc_per_carry: Vec<[E; POSSIBLE_CARRIES]> = (0..group.depth_commit)
            .map(|dc| {
                let mut s = [E::zero(); POSSIBLE_CARRIES];
                for (carry_slot, slot) in s.iter_mut().enumerate() {
                    let mut acc = E::zero();
                    for (df, &fold) in fold_gadget.iter().enumerate().take(group.depth_fold) {
                        let high_idx = dc * group.depth_fold + df + carry_slot;
                        acc += eq_high[high_idx].mul_base(fold);
                    }
                    *slot = -acc;
                }
                s
            })
            .collect();
        Ok(cfg_into_iter!(0..z_range)
            .map(|k| {
                let block_idx = k / group.depth_commit;
                let dc = k % group.depth_commit;
                let shifted = offset_low + block_idx;
                let low_idx = shifted & low_mask;
                let carry = shifted >> z_bits;
                let low = eq_low[low_idx];
                let high = s_per_dc_per_carry[dc][carry];
                low * high
            })
            .collect())
    } else {
        let z_len = checked_mul(
            checked_mul(
                group.depth_commit,
                group.depth_fold,
                "grouped dense Z length",
            )?,
            group.block_len,
            "grouped dense Z length",
        )?;
        let low_bits = z_len
            .saturating_sub(1)
            .checked_next_power_of_two()
            .map(|p| p.trailing_zeros() as usize)
            .unwrap_or(0)
            .max(1)
            .min(full_vec_randomness.len());
        let low_mask = 1usize
            .checked_shl(
                u32::try_from(low_bits).map_err(|_| AkitaError::InvalidSize {
                    expected: usize::BITS as usize,
                    actual: low_bits,
                })?,
            )
            .ok_or_else(|| AkitaError::InvalidSetup("grouped dense Z eq width overflow".into()))?
            - 1;
        let eq_low = EqPolynomial::evals(&full_vec_randomness[..low_bits])?;
        let offset_low = offset_z & low_mask;
        let offset_high = offset_z >> low_bits;
        let max_high = checked_add(offset_z, z_len, "grouped dense Z end")?
            .checked_sub(1)
            .ok_or(AkitaError::InvalidProof)?
            >> low_bits;
        let eq_high: Vec<E> = (offset_high..=max_high)
            .map(|idx| eq_eval_at_index(&full_vec_randomness[low_bits..], idx))
            .collect();
        cfg_into_iter!(0..z_range)
            .map(|k| {
                let block_idx = k / group.depth_commit;
                let dc = k % group.depth_commit;
                let mut weight = E::zero();
                for (df, &fold) in fold_gadget.iter().enumerate().take(group.depth_fold) {
                    let x = checked_add(
                        block_idx,
                        checked_mul(
                            group.block_len,
                            checked_add(
                                df,
                                checked_mul(dc, group.depth_fold, "grouped dense Z dc")?,
                                "grouped dense Z df",
                            )?,
                            "grouped dense Z offset",
                        )?,
                        "grouped dense Z offset",
                    )?;
                    let shifted = checked_add(offset_low, x, "grouped dense Z low")?;
                    let low_idx = shifted & low_mask;
                    let high_carry = shifted >> low_bits;
                    let low = *eq_low.get(low_idx).ok_or(AkitaError::InvalidProof)?;
                    let high = *eq_high.get(high_carry).ok_or(AkitaError::InvalidProof)?;
                    weight -= (low * high).mul_base(fold);
                }
                Ok(weight)
            })
            .collect()
    }
}

fn evaluate_weighted_setup_row<Base, E>(
    row: &[Base],
    col_offset: usize,
    col_weights: &[E],
    row_weight: E,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    Base: FieldCore,
    E: ExtField<Base> + MulBaseUnreduced<Base>,
{
    let ring_d = alpha_pows.len();
    let mut acc = E::zero();
    for (col, &col_weight) in col_weights.iter().enumerate() {
        if col_weight.is_zero() {
            continue;
        }
        let setup_col = checked_add(col_offset, col, "weighted setup column")?;
        let coeff_start = checked_mul(setup_col, ring_d, "weighted setup coeff start")?;
        let coeffs = checked_slice(row, coeff_start, ring_d, "weighted setup coeffs")?;
        acc += row_weight * col_weight * eval_flat_ring_at_pows_fast::<Base, E>(coeffs, alpha_pows);
    }
    Ok(acc)
}

#[inline(always)]
fn push_group_d_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    active_col_start: usize,
    active_cols: usize,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let active_col_end = checked_add(active_col_start, active_cols, "grouped D active columns")?;
    let mut row_start = 0usize;
    for _ in 0..rows {
        let row_end = checked_add(row_start, stride, "packed D boundary")?;
        endpoints.push(row_end);
        if active_cols != 0 {
            endpoints.push(checked_add(
                row_start,
                active_col_start,
                "grouped D active boundary",
            )?);
            endpoints.push(checked_add(
                row_start,
                active_col_end,
                "grouped D active boundary",
            )?);
        }
        row_start = row_end;
    }
    Ok(())
}
