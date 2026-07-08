use super::{
    checked_add, checked_mul, checked_slice, push_role_boundaries, setup_e_col_weights,
    setup_t_col_weights, setup_z_col_weights_for_offset, SetupContributionPlanInputs,
};
use crate::layout::flat_matrix::FlatRingMatrixView;
use crate::layout::MRowLayout;
use crate::proof::AkitaExpandedSetup;
use crate::WitnessChunkLayout;
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

pub struct SetupContributionPlan<E> {
    pub(super) groups: Vec<SetupContributionGroupPlan<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
}

/// Tau1-derived grouped setup weights cached at ring-switch prepare time.
#[derive(Clone)]
pub struct SetupContributionStatic<E> {
    pub(super) groups: Vec<SetupContributionGroupStatic<E>>,
    pub(super) d_rows: usize,
    pub(super) d_physical_cols: usize,
    pub(super) d_weights: Vec<E>,
}

#[derive(Clone)]
pub(super) struct SetupContributionGroupStatic<E> {
    pub(super) e_col_offset: usize,
    pub(super) t_cols: usize,
    pub(super) z_cols: usize,
    pub(super) n_a: usize,
    pub(super) n_b: usize,
    pub(super) a_weights: Vec<E>,
    pub(super) b_weights: Vec<E>,
}

pub(super) struct SetupContributionGroupPlan<E> {
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
    ) -> Result<SetupContributionPlan<E>, AkitaError>
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
    ) -> Result<SetupContributionStatic<E>, AkitaError> {
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
                Ok(SetupContributionGroupStatic {
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
        Ok(SetupContributionStatic {
            groups: static_groups,
            d_rows,
            d_physical_cols,
            d_weights,
        })
    }

    pub fn finish_grouped_plan<F>(
        static_plan: &SetupContributionStatic<E>,
        full_vec_randomness: &[E],
        eq_low: Option<&[E]>,
        z_block_low_eq: Option<&[E]>,
        fold_gadget: Option<&[F]>,
        groups: &[SetupContributionGroupInputs],
    ) -> Result<SetupContributionPlan<E>, AkitaError>
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
                let e_eq_slice = setup_e_col_weights::<E>(
                    &group.chunks,
                    group.blocks_per_chunk,
                    group.num_blocks,
                    group.num_claims,
                    group.depth_open,
                    full_vec_randomness,
                    eq_low,
                )?;
                let t_eq_slice = setup_t_col_weights::<E>(
                    &group.chunks,
                    group.blocks_per_chunk,
                    group.depth_open,
                    group.n_a,
                    group.t_cols_per_vector,
                    group.num_claims,
                    group.num_claims,
                    0,
                    group.num_claims,
                    full_vec_randomness,
                    eq_low,
                )?;
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
                    fold_gadget_storage =
                        crate::gadget_row_scalars::<F>(group.depth_fold, group.log_basis);
                    &fold_gadget_storage
                };
                let z_range = checked_mul(group.block_len, group.depth_commit, "grouped Z range")?;
                let mut z_eq_slice = vec![E::zero(); z_range];
                for chunk in &group.chunks {
                    let per_chunk = setup_z_col_weights_for_offset::<F, E>(
                        group.block_len,
                        group.depth_commit,
                        group.depth_fold,
                        1,
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
                Ok(SetupContributionGroupPlan {
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
        Ok(SetupContributionPlan {
            groups: dynamic_groups,
            d_rows: static_plan.d_rows,
            d_physical_cols: static_plan.d_physical_cols,
        })
    }

    /// Single-group setup-contribution plan built from the flat inputs and a
    /// witness chunk layout. Reproduces the historical flat plan: one commitment
    /// group at `e_col_offset = 0` spanning the full `n_cols_e` D width.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        inputs: &SetupContributionPlanInputs<E>,
        full_vec_randomness: &[E],
        eq_low: Option<&[E]>,
        z_block_low_eq: Option<&[E]>,
        fold_gadget: &[F],
        chunk_layout: &crate::WitnessLayout,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        if inputs.num_groups != 1 || inputs.num_polys_per_group.len() != 1 {
            return Err(AkitaError::InvalidSetup(
                "single-group setup contribution requires exactly one commitment group".into(),
            ));
        }
        let n_d_active = match inputs.m_row_layout {
            MRowLayout::WithDBlock => inputs.n_d,
            MRowLayout::WithoutDBlock => 0,
        };
        let a_row_start = 1usize;
        let b_row_start = checked_add(a_row_start, inputs.n_a, "B row start")?;
        let d_row_start = checked_add(b_row_start, inputs.n_b, "D row start")?;
        let b_per_claim_e = checked_mul(inputs.num_blocks, inputs.depth_open, "e-hat claim width")?;
        let n_cols_e = checked_mul(inputs.num_claims, b_per_claim_e, "e-hat column width")?;
        let t_cols_per_vector = checked_mul(
            checked_mul(inputs.n_a, inputs.depth_open, "T stride")?,
            inputs.num_blocks,
            "T polynomial width",
        )?;
        let group = SetupContributionGroupInputs {
            e_col_offset: 0,
            num_claims: inputs.num_claims,
            num_blocks: inputs.num_blocks,
            block_len: inputs.block_len,
            depth_open: inputs.depth_open,
            depth_commit: inputs.depth_commit,
            depth_fold: inputs.depth_fold,
            log_basis: 0,
            n_a: inputs.n_a,
            n_b: inputs.n_b,
            t_cols_per_vector,
            a_row_start,
            b_row_start,
            blocks_per_chunk: chunk_layout.blocks_per_chunk,
            chunks: chunk_layout.chunks.clone(),
        };
        Self::prepare_grouped::<F>(
            inputs,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            Some(fold_gadget),
            std::slice::from_ref(&group),
            d_row_start,
            n_d_active,
            n_cols_e,
        )
    }

    /// Packed-scan footprint length: max over groups of each role's `rows * cols`.
    /// `D` rows/cols are plan-level (shared); `B`/`A` are per-group.
    pub fn required(&self) -> Result<usize, AkitaError> {
        let mut required = checked_mul(
            self.d_rows,
            self.d_physical_cols,
            "grouped D setup footprint",
        )?;
        for group in &self.groups {
            let b_required = checked_mul(group.n_b, group.t_cols, "grouped B setup footprint")?;
            let a_required = checked_mul(group.n_a, group.z_cols, "grouped A setup footprint")?;
            required = required.max(b_required).max(a_required);
        }
        Ok(required)
    }

    /// Dense per-position setup weights over the shared setup vector. Each group
    /// scatters its packed-segment weights into the shared footprint; overlapping
    /// positions (e.g. the shared D rows) accumulate additively. For a single
    /// group this equals the historical flat `bar_omega`.
    pub fn materialize_bar_omega(&self) -> Result<Vec<E>, AkitaError> {
        let required = self.required()?;
        let mut bar_omega = vec![E::zero(); required];
        for group in &self.groups {
            let (_, segments) = group.packed_segments(self.d_rows, self.d_physical_cols)?;
            let segment_values = cfg_into_iter!(segments)
                .map(|segment| {
                    let values = (segment.lo..segment.hi)
                        .map(|lambda| group.weight_at(lambda, &segment))
                        .collect::<Vec<_>>();
                    (segment.lo, values)
                })
                .collect::<Vec<_>>();
            for (lo, values) in segment_values {
                for (offset, value) in values.into_iter().enumerate() {
                    bar_omega[lo + offset] += value;
                }
            }
        }
        Ok(bar_omega)
    }

    /// Eq-weighted setup contribution `sum_lambda eq_lambda[lambda] * bar_omega[lambda]`
    /// without materializing `bar_omega`. `eq_lambda` must have length
    /// `required().next_power_of_two()`.
    pub fn evaluate_bar_omega_with_eq(&self, eq_lambda: &[E]) -> Result<E, AkitaError> {
        let required = self.required()?;
        let lambda_len = required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup omega lambda length overflow".into()))?;
        if eq_lambda.len() != lambda_len {
            return Err(AkitaError::InvalidSize {
                expected: lambda_len,
                actual: eq_lambda.len(),
            });
        }
        let mut acc = E::zero();
        for group in &self.groups {
            let (_, segments) = group.packed_segments(self.d_rows, self.d_physical_cols)?;
            let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
                .map(|idx| {
                    let segment = &segments[idx];
                    macro_rules! segment_sum {
                        ($has_d:literal, $has_b:literal, $has_a:literal) => {
                            group_bar_omega_segment_eval::<E, $has_d, $has_b, $has_a>(
                                segment.lo..segment.hi,
                                eq_lambda,
                                segment.d_start_abs,
                                segment.d_weight,
                                &group.e_eq_slice,
                                segment.b_start_abs,
                                segment.b_weight,
                                &group.t_eq_slice,
                                segment.a_start_abs,
                                segment.a_weight,
                                &group.z_eq_slice,
                            )
                        };
                    }
                    match (segment.has_d, segment.has_b, segment.has_a) {
                        (true, true, true) => segment_sum!(true, true, true),
                        (true, true, false) => segment_sum!(true, true, false),
                        (true, false, true) => segment_sum!(true, false, true),
                        (false, true, true) => segment_sum!(false, true, true),
                        (true, false, false) => segment_sum!(true, false, false),
                        (false, true, false) => segment_sum!(false, true, false),
                        (false, false, true) => segment_sum!(false, false, true),
                        (false, false, false) => E::zero(),
                    }
                })
                .collect();
            acc += segment_sums.into_iter().sum::<E>();
        }
        Ok(acc)
    }

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
        if d_a == d_b && d_b == d_d && alpha_pows_a == alpha_pows_b && alpha_pows_a == alpha_pows_d
        {
            return self.evaluate_uniform_direct(setup, alpha_pows_a, d_a);
        }
        if let Some(value) =
            self.evaluate_divisible_direct(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)?
        {
            return Ok(value);
        }

        self.evaluate_packed_direct(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d, d_a)
    }

    fn evaluate_divisible_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<Option<E>, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let base_d = alpha_pows_a
            .len()
            .min(alpha_pows_b.len())
            .min(alpha_pows_d.len());
        if base_d == 0
            || !alpha_pows_a.len().is_multiple_of(base_d)
            || !alpha_pows_b.len().is_multiple_of(base_d)
            || !alpha_pows_d.len().is_multiple_of(base_d)
        {
            return Ok(None);
        }
        let base_pows = if alpha_pows_a.len() == base_d {
            alpha_pows_a
        } else if alpha_pows_b.len() == base_d {
            alpha_pows_b
        } else {
            alpha_pows_d
        };
        let Some(a_scales) = alpha_chunk_scales(alpha_pows_a, base_pows) else {
            return Ok(None);
        };
        let Some(b_scales) = alpha_chunk_scales(alpha_pows_b, base_pows) else {
            return Ok(None);
        };
        let Some(d_scales) = alpha_chunk_scales(alpha_pows_d, base_pows) else {
            return Ok(None);
        };

        let required =
            self.required_divisible_base(a_scales.ratio(), b_scales.ratio(), d_scales.ratio())?;
        let setup_len = setup.shared_matrix().total_ring_elements_at_dyn(base_d)?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        let setup_view = setup.shared_matrix().ring_view_dyn(1, setup_len, base_d)?;
        let mut acc = E::zero();
        for group in &self.groups {
            acc += group.evaluate_divisible_packed_direct(
                &setup_view,
                base_pows,
                &a_scales,
                &b_scales,
                &d_scales,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(Some(acc))
    }

    fn required_divisible_base(
        &self,
        a_ratio: usize,
        b_ratio: usize,
        d_ratio: usize,
    ) -> Result<usize, AkitaError> {
        let mut required = checked_mul(
            checked_mul(
                self.d_rows,
                self.d_physical_cols,
                "grouped D setup footprint",
            )?,
            d_ratio,
            "grouped D base setup footprint",
        )?;
        for group in &self.groups {
            let b_required = checked_mul(group.n_b, group.t_cols, "grouped B setup footprint")?;
            let a_required = checked_mul(group.n_a, group.z_cols, "grouped A setup footprint")?;
            required = required
                .max(checked_mul(
                    b_required,
                    b_ratio,
                    "grouped B base setup footprint",
                )?)
                .max(checked_mul(
                    a_required,
                    a_ratio,
                    "grouped A base setup footprint",
                )?);
        }
        Ok(required)
    }

    fn evaluate_uniform_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows: &[E],
        ring_d: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let required = self.required()?;
        let setup_len = setup.shared_matrix().total_ring_elements_at_dyn(ring_d)?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        let setup_view = setup.shared_matrix().ring_view_dyn(1, setup_len, ring_d)?;
        let mut acc = E::zero();
        for group in &self.groups {
            acc += group.evaluate_uniform_packed_direct(
                &setup_view,
                alpha_pows,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(acc)
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

    #[cfg(test)]
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

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    #[allow(clippy::too_many_arguments)]
    fn evaluate_divisible_packed_direct<F>(
        &self,
        setup_view: &FlatRingMatrixView<'_, F>,
        base_pows: &[E],
        a_scales: &AlphaChunkScales<E>,
        b_scales: &AlphaChunkScales<E>,
        d_scales: &AlphaChunkScales<E>,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        self.packed_segments(d_rows, d_physical_cols)?;
        let d_required = checked_mul(d_rows, d_physical_cols, "grouped D setup footprint")?;
        let b_required = checked_mul(self.n_b, self.t_cols, "grouped B setup footprint")?;
        let a_required = checked_mul(self.n_a, self.z_cols, "grouped A setup footprint")?;
        let required = checked_mul(d_required, d_scales.ratio(), "grouped D base footprint")?
            .max(checked_mul(
                b_required,
                b_scales.ratio(),
                "grouped B base footprint",
            )?)
            .max(checked_mul(
                a_required,
                a_scales.ratio(),
                "grouped A base footprint",
            )?);
        if required > 0 {
            setup_view.elem(0, required - 1)?;
        }
        let d_scaled_weights = scaled_row_weights(&self.d_weights, d_scales.scales());
        let b_scaled_weights = scaled_row_weights(&self.b_weights, b_scales.scales());
        let a_scaled_weights = scaled_row_weights(&self.a_weights, a_scales.scales());

        Ok(cfg_fold_reduce!(
            0..required,
            E::zero,
            |mut acc, base_lambda| {
                let weight = self.divisible_base_weight_at(
                    base_lambda,
                    &a_scaled_weights,
                    &b_scaled_weights,
                    &d_scaled_weights,
                    a_scales,
                    b_scales,
                    d_scales,
                    d_physical_cols,
                    d_required,
                    b_required,
                    a_required,
                );
                if !weight.is_zero() {
                    let coeffs = setup_view.elem_in_band(0, base_lambda);
                    acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, base_pows) * weight;
                }
                acc
            },
            |lhs, rhs| lhs + rhs
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn divisible_base_weight_at(
        &self,
        base_lambda: usize,
        a_scaled_weights: &[E],
        b_scaled_weights: &[E],
        d_scaled_weights: &[E],
        a_scales: &AlphaChunkScales<E>,
        b_scales: &AlphaChunkScales<E>,
        d_scales: &AlphaChunkScales<E>,
        d_physical_cols: usize,
        d_required: usize,
        b_required: usize,
        a_required: usize,
    ) -> E {
        let mut weight = E::zero();

        let d_ratio = d_scales.ratio();
        if !self.e_eq_slice.is_empty() {
            let d_lambda = base_lambda >> d_scales.shift();
            if d_lambda < d_required {
                let d_col = d_lambda % d_physical_cols;
                let e_end = self.e_col_offset + self.e_eq_slice.len();
                if d_col >= self.e_col_offset && d_col < e_end {
                    let d_row = d_lambda / d_physical_cols;
                    let d_start_abs = d_row * d_physical_cols + self.e_col_offset;
                    let scaled_weight =
                        d_scaled_weights[d_row * d_ratio + (base_lambda & d_scales.mask())];
                    weight += scaled_weight * self.e_eq_slice[d_lambda - d_start_abs];
                }
            }
        }

        let b_ratio = b_scales.ratio();
        let b_lambda = base_lambda >> b_scales.shift();
        if b_lambda < b_required {
            let b_row = b_lambda / self.t_cols;
            let b_start_abs = b_row * self.t_cols;
            let scaled_weight = b_scaled_weights[b_row * b_ratio + (base_lambda & b_scales.mask())];
            weight += scaled_weight * self.t_eq_slice[b_lambda - b_start_abs];
        }

        let a_ratio = a_scales.ratio();
        let a_lambda = base_lambda >> a_scales.shift();
        if a_lambda < a_required {
            let a_row = a_lambda / self.z_cols;
            let a_start_abs = a_row * self.z_cols;
            let scaled_weight = a_scaled_weights[a_row * a_ratio + (base_lambda & a_scales.mask())];
            weight += scaled_weight * self.z_eq_slice[a_lambda - a_start_abs];
        }

        weight
    }

    fn evaluate_uniform_packed_direct<F>(
        &self,
        setup_view: &FlatRingMatrixView<'_, F>,
        alpha_pows: &[E],
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let (required, segments) = self.packed_segments(d_rows, d_physical_cols)?;
        if required > 0 {
            setup_view.elem(0, required - 1)?;
        }

        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        packed_uniform_group_slice_inner_sum::<F, E, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            setup_view,
                            alpha_pows,
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

                match (segment.has_d, segment.has_b, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    (false, false, false) => E::zero(),
                }
            })
            .collect();

        Ok(segment_sums.into_iter().sum())
    }

    #[allow(clippy::too_many_arguments)]
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

        let (_, segments) = self.packed_segments(d_rows, d_physical_cols)?;
        validate_packed_scan_access(
            d_rows,
            d_physical_cols,
            d_view.as_ref(),
            self.n_b,
            self.t_cols,
            &b_view,
            self.n_a,
            self.z_cols,
            &a_view,
            &segments,
        )?;

        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                macro_rules! segment_sum {
                    ($has_d:literal, $has_b:literal, $has_a:literal) => {
                        packed_group_slice_inner_sum::<F, E, $has_d, $has_b, $has_a>(
                            segment.lo..segment.hi,
                            d_view.as_ref(),
                            d_physical_cols,
                            &b_view,
                            self.t_cols,
                            &a_view,
                            self.z_cols,
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

                match (segment.has_d, segment.has_b, segment.has_a) {
                    (true, true, true) => segment_sum!(true, true, true),
                    (true, true, false) => segment_sum!(true, true, false),
                    (true, false, true) => segment_sum!(true, false, true),
                    (false, true, true) => segment_sum!(false, true, true),
                    (true, false, false) => segment_sum!(true, false, false),
                    (false, true, false) => segment_sum!(false, true, false),
                    (false, false, true) => segment_sum!(false, false, true),
                    (false, false, false) => E::zero(),
                }
            })
            .collect();

        Ok(segment_sums.into_iter().sum())
    }

    fn weight_at(&self, lambda: usize, segment: &GroupSetupSegment<E>) -> E {
        let mut weight = E::zero();
        if segment.has_d {
            weight += segment.d_weight * self.e_eq_slice[lambda - segment.d_start_abs];
        }
        if segment.has_b {
            weight += segment.b_weight * self.t_eq_slice[lambda - segment.b_start_abs];
        }
        if segment.has_a {
            weight += segment.a_weight * self.z_eq_slice[lambda - segment.a_start_abs];
        }
        weight
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

#[allow(clippy::too_many_arguments)]
fn validate_packed_scan_access<F, E>(
    d_rows: usize,
    d_physical_cols: usize,
    d_view: Option<&FlatRingMatrixView<'_, F>>,
    n_b: usize,
    t_cols: usize,
    b_view: &FlatRingMatrixView<'_, F>,
    n_a: usize,
    z_cols: usize,
    a_view: &FlatRingMatrixView<'_, F>,
    segments: &[GroupSetupSegment<E>],
) -> Result<(), AkitaError>
where
    F: FieldCore,
    E: FieldCore,
{
    for segment in segments {
        if segment.has_d && d_view.is_none() {
            return Err(AkitaError::InvalidSetup(
                "grouped packed D scan missing D view".into(),
            ));
        }
    }
    let d_required = checked_mul(d_rows, d_physical_cols, "grouped D setup footprint")?;
    if d_required > 0 {
        if let Some(d_view) = d_view {
            let probe = d_required - 1;
            d_view.elem(probe / d_physical_cols, probe % d_physical_cols)?;
        }
    }
    let b_required = checked_mul(n_b, t_cols, "grouped B setup footprint")?;
    if b_required > 0 {
        let probe = b_required - 1;
        b_view.elem(probe / t_cols, probe % t_cols)?;
    }
    let a_required = checked_mul(n_a, z_cols, "grouped A setup footprint")?;
    if a_required > 0 {
        let probe = a_required - 1;
        a_view.elem(probe / z_cols, probe % z_cols)?;
    }
    Ok(())
}

struct AlphaChunkScales<E> {
    scales: Vec<E>,
    shift: usize,
    mask: usize,
}

impl<E> AlphaChunkScales<E> {
    fn ratio(&self) -> usize {
        self.scales.len()
    }

    fn scales(&self) -> &[E] {
        &self.scales
    }

    fn shift(&self) -> usize {
        self.shift
    }

    fn mask(&self) -> usize {
        self.mask
    }
}

fn alpha_chunk_scales<E: FieldCore>(
    alpha_pows: &[E],
    base_pows: &[E],
) -> Option<AlphaChunkScales<E>> {
    let base_d = base_pows.len();
    if base_d == 0 || !alpha_pows.len().is_multiple_of(base_d) {
        return None;
    }
    let ratio = alpha_pows.len() / base_d;
    if ratio == 0 || !ratio.is_power_of_two() {
        return None;
    }
    let mut scales = Vec::with_capacity(ratio);
    for chunk in 0..ratio {
        let offset = chunk * base_d;
        let scale = alpha_pows[offset];
        for idx in 0..base_d {
            if alpha_pows[offset + idx] != scale * base_pows[idx] {
                return None;
            }
        }
        scales.push(scale);
    }
    Some(AlphaChunkScales {
        scales,
        shift: ratio.trailing_zeros() as usize,
        mask: ratio - 1,
    })
}

fn scaled_row_weights<E: FieldCore>(row_weights: &[E], scales: &[E]) -> Vec<E> {
    let mut scaled = Vec::with_capacity(row_weights.len() * scales.len());
    for &row_weight in row_weights {
        scaled.extend(scales.iter().map(|&scale| row_weight * scale));
    }
    scaled
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn group_bar_omega_segment_eval<E, const HAS_D: bool, const HAS_B: bool, const HAS_A: bool>(
    range: std::ops::Range<usize>,
    eq_lambda: &[E],
    d_start: usize,
    d_weight: E,
    e_eq: &[E],
    b_start: usize,
    b_weight: E,
    t_eq: &[E],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    E: FieldCore,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            let mut weight = E::zero();
            if HAS_D {
                weight += d_weight * e_eq[lambda - d_start];
            }
            if HAS_B {
                weight += b_weight * t_eq[lambda - b_start];
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            if !weight.is_zero() {
                acc += eq_lambda[lambda] * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_uniform_group_slice_inner_sum<
    F,
    E,
    const HAS_D: bool,
    const HAS_B: bool,
    const HAS_A: bool,
>(
    range: std::ops::Range<usize>,
    setup_view: &FlatRingMatrixView<'_, F>,
    alpha_pows: &[E],
    d_start: usize,
    d_weight: E,
    e_eq: &[E],
    b_start: usize,
    b_weight: E,
    t_eq: &[E],
    a_start: usize,
    a_weight: E,
    z_eq: &[E],
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            let mut weight = E::zero();
            if HAS_D {
                weight += d_weight * e_eq[lambda - d_start];
            }
            if HAS_B {
                weight += b_weight * t_eq[lambda - b_start];
            }
            if HAS_A {
                weight += a_weight * z_eq[lambda - a_start];
            }
            if !weight.is_zero() {
                let coeffs = setup_view.elem_in_band(0, lambda);
                acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows) * weight;
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packed_group_slice_inner_sum<F, E, const HAS_D: bool, const HAS_B: bool, const HAS_A: bool>(
    range: std::ops::Range<usize>,
    d_view: Option<&FlatRingMatrixView<'_, F>>,
    d_physical_cols: usize,
    b_view: &FlatRingMatrixView<'_, F>,
    t_cols: usize,
    a_view: &FlatRingMatrixView<'_, F>,
    z_cols: usize,
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
) -> E
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    cfg_fold_reduce!(
        range,
        E::zero,
        |mut acc, lambda| {
            if HAS_D {
                let eq_w = d_weight * e_eq[lambda - d_start];
                if !eq_w.is_zero() {
                    if let Some(d_view) = d_view {
                        let d_row = lambda / d_physical_cols;
                        let d_col = lambda % d_physical_cols;
                        let coeffs = d_view.elem_in_band(d_row, d_col);
                        acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_d) * eq_w;
                    }
                }
            }
            if HAS_B {
                let eq_w = b_weight * t_eq[lambda - b_start];
                if !eq_w.is_zero() {
                    let b_row = lambda / t_cols;
                    let b_col = lambda % t_cols;
                    let coeffs = b_view.elem_in_band(b_row, b_col);
                    acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_b) * eq_w;
                }
            }
            if HAS_A {
                let eq_w = a_weight * z_eq[lambda - a_start];
                if !eq_w.is_zero() {
                    let a_row = lambda / z_cols;
                    let a_col = lambda % z_cols;
                    let coeffs = a_view.elem_in_band(a_row, a_col);
                    acc += eval_flat_ring_at_pows_fast::<F, E>(coeffs, alpha_pows_a) * eq_w;
                }
            }
            acc
        },
        |lhs, rhs| lhs + rhs
    )
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

#[cfg(test)]
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
