use super::super::*;

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_divisible_packed_direct<F, const BASE_D: usize>(
        &self,
        setup_view: &RingMatrixView<'_, F, BASE_D>,
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
        let d_required = d_rows
            .checked_mul(d_physical_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
        let b_required = self
            .n_b
            .checked_mul(self.t_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
        let a_required = self
            .n_a
            .checked_mul(self.z_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
        let required = d_required
            .checked_mul(d_scales.scales.len())
            .ok_or_else(|| AkitaError::InvalidSetup("setup D base footprint overflow".into()))?
            .max(
                b_required
                    .checked_mul(b_scales.scales.len())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup B base footprint overflow".into())
                    })?,
            )
            .max(
                a_required
                    .checked_mul(a_scales.scales.len())
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup A base footprint overflow".into())
                    })?,
            );
        if required > 0 {
            let setup_flat = setup_view.as_slice();
            if required > setup_flat.len() {
                return Err(AkitaError::InvalidSetup(
                    "shared matrix is too small for selected verifier layout".into(),
                ));
            }
        }
        let setup_flat = setup_view.as_slice();
        let d_scaled_weights = scaled_row_weights(&self.d_weights, &d_scales.scales);
        let b_scaled_weights = scaled_row_weights(&self.b_weights, &b_scales.scales);
        let a_scaled_weights = scaled_row_weights(&self.a_weights, &a_scales.scales);

        Ok(cfg_fold_reduce!(
            0..required,
            E::zero,
            |mut acc, base_idx| {
                let weight = self.divisible_base_weight_at(
                    base_idx,
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
                    acc += eval_ring_at_pows_fast(&setup_flat[base_idx], base_pows) * weight;
                }
                acc
            },
            |lhs, rhs| lhs + rhs
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn divisible_base_weight_at(
        &self,
        base_idx: usize,
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

        let d_ratio = d_scales.scales.len();
        if !self.e_eq_slice.is_empty() {
            let d_idx = base_idx >> d_scales.shift;
            if d_idx < d_required {
                let d_col = d_idx % d_physical_cols;
                let e_end = self.e_col_offset + self.e_eq_slice.len();
                if d_col >= self.e_col_offset && d_col < e_end {
                    let d_row = d_idx / d_physical_cols;
                    let d_start_abs = d_row * d_physical_cols + self.e_col_offset;
                    let scaled_weight =
                        d_scaled_weights[d_row * d_ratio + (base_idx & d_scales.mask)];
                    weight += scaled_weight * self.e_eq_slice[d_idx - d_start_abs];
                }
            }
        }

        let b_ratio = b_scales.scales.len();
        let b_idx = base_idx >> b_scales.shift;
        if b_idx < b_required {
            if let Some(b_row) = b_idx.checked_div(self.t_cols) {
                let b_start_abs = b_row * self.t_cols;
                let scaled_weight = b_scaled_weights[b_row * b_ratio + (base_idx & b_scales.mask)];
                weight += scaled_weight * self.t_eq_slice[b_idx - b_start_abs];
            }
        }

        let a_ratio = a_scales.scales.len();
        let a_idx = base_idx >> a_scales.shift;
        if a_idx < a_required {
            if let Some(a_row) = a_idx.checked_div(self.z_cols) {
                let a_start_abs = a_row * self.z_cols;
                let scaled_weight = a_scaled_weights[a_row * a_ratio + (base_idx & a_scales.mask)];
                weight += scaled_weight * self.z_eq_slice[a_idx - a_start_abs];
            }
        }

        weight
    }

    pub(super) fn evaluate_uniform_packed_direct_typed<F, const D: usize>(
        &self,
        setup_view: &RingMatrixView<'_, F, D>,
        alpha_pows: &[E],
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let (required, segments) = self.packed_segments(d_rows, d_physical_cols)?;
        let setup_flat = setup_view.as_slice();
        if required > setup_flat.len() {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }

        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                dispatch_segment_roles!(segment, E::zero(), |HAS_D, HAS_B, HAS_A| {
                    packed_uniform_group_slice_inner_sum_typed::<F, E, D, HAS_D, HAS_B, HAS_A>(
                        segment.lo..segment.hi,
                        setup_flat,
                        alpha_pows,
                        segment,
                        &self.e_eq_slice,
                        &self.t_eq_slice,
                        &self.z_eq_slice,
                    )
                })
            })
            .collect();

        Ok(segment_sums.into_iter().sum())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_packed_direct_typed<
        F,
        const D_A: usize,
        const D_B: usize,
        const D_D: usize,
    >(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let d_view = if d_rows != 0 && !self.e_eq_slice.is_empty() {
            Some(
                setup
                    .shared_matrix
                    .ring_view::<D_D>(d_rows, d_physical_cols)?,
            )
        } else {
            None
        };
        let b_view = setup
            .shared_matrix
            .ring_view::<D_B>(self.n_b, self.t_cols)?;
        let a_view = setup
            .shared_matrix
            .ring_view::<D_A>(self.n_a, self.z_cols)?;
        let d_flat = d_view.as_ref().map(|view| view.as_slice());
        let b_flat = b_view.as_slice();
        let a_flat = a_view.as_slice();
        let d_len = d_flat.map_or(0, |slice| slice.len());

        let (_, segments) = self.packed_segments(d_rows, d_physical_cols)?;
        validate_typed_packed_scan_access(
            d_rows,
            d_physical_cols,
            d_flat.is_some(),
            d_len,
            self.n_b,
            self.t_cols,
            b_flat.len(),
            self.n_a,
            self.z_cols,
            a_flat.len(),
            segments,
        )?;

        let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
            .map(|idx| {
                let segment = &segments[idx];
                dispatch_segment_roles!(segment, E::zero(), |HAS_D, HAS_B, HAS_A| {
                    packed_group_slice_inner_sum_typed::<F, E, D_A, D_B, D_D, HAS_D, HAS_B, HAS_A>(
                        segment.lo..segment.hi,
                        d_flat,
                        d_physical_cols,
                        b_flat,
                        self.t_cols,
                        a_flat,
                        self.z_cols,
                        alpha_pows_a,
                        alpha_pows_b,
                        alpha_pows_d,
                        segment,
                        &self.e_eq_slice,
                        &self.t_eq_slice,
                        &self.z_eq_slice,
                    )
                })
            })
            .collect();

        Ok(segment_sums.into_iter().sum())
    }
}
