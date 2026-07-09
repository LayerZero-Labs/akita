use super::super::*;

enum BaseRingSegments<'a, E> {
    /// Identity projection: borrow the cached logical segment partition.
    Borrowed(&'a [GroupSetupSegment<E>]),
    /// Non-identity projection: use a base-ring segment partition.
    Projected(Vec<GroupSetupSegment<E>>),
}

impl<'a, E> BaseRingSegments<'a, E> {
    fn as_slice(&self) -> &[GroupSetupSegment<E>] {
        match self {
            Self::Borrowed(segments) => segments,
            Self::Projected(segments) => segments,
        }
    }
}

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_base_ring_direct<F, const BASE_D: usize>(
        &self,
        setup_view: &RingMatrixView<'_, F, BASE_D>,
        base_pows: &[E],
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let required = self.required_base_ring_rows_for_group(
            a_projection.ratio(),
            b_projection.ratio(),
            d_projection.ratio(),
            d_rows,
            d_physical_cols,
        )?;
        let setup_flat = setup_view.as_slice();
        if required > setup_flat.len() {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }

        if a_projection.is_identity() && b_projection.is_identity() && d_projection.is_identity() {
            let (_, segments) = self.packed_segments(d_rows, d_physical_cols)?;
            let segment_sums: Vec<E> = cfg_into_iter!(0..segments.len())
                .map(|idx| {
                    let segment = &segments[idx];
                    dispatch_segment_roles!(segment, E::zero(), |HAS_D, HAS_B, HAS_A| {
                        identity_base_ring_segment_inner_sum_typed::<
                            F,
                            E,
                            BASE_D,
                            HAS_D,
                            HAS_B,
                            HAS_A,
                        >(
                            segment.lo..segment.hi,
                            setup_flat,
                            base_pows,
                            segment,
                            &self.e_eq_slice,
                            &self.t_eq_slice,
                            &self.z_eq_slice,
                        )
                    })
                })
                .collect();
            return Ok(segment_sums.into_iter().sum());
        }

        let segments = self.base_ring_segments(
            a_projection,
            b_projection,
            d_projection,
            d_rows,
            d_physical_cols,
            required,
        )?;
        let d_weights = ProjectedRoleWeights::new(&self.d_weights, d_projection);
        let b_weights = ProjectedRoleWeights::new(&self.b_weights, b_projection);
        let a_weights = ProjectedRoleWeights::new(&self.a_weights, a_projection);

        dispatch_role_projections!(
            d_projection,
            b_projection,
            a_projection,
            |D_IDENTITY, B_IDENTITY, A_IDENTITY| {
                let segment_slice = segments.as_slice();
                let segment_sums: Vec<E> = cfg_into_iter!(0..segment_slice.len())
                    .map(|idx| {
                        let segment = &segment_slice[idx];
                        dispatch_segment_roles!(segment, E::zero(), |HAS_D, HAS_B, HAS_A| {
                            base_ring_segment_inner_sum_typed::<
                                F,
                                E,
                                BASE_D,
                                HAS_D,
                                HAS_B,
                                HAS_A,
                                D_IDENTITY,
                                B_IDENTITY,
                                A_IDENTITY,
                            >(
                                segment.lo..segment.hi,
                                setup_flat,
                                base_pows,
                                segment,
                                &self.e_eq_slice,
                                &self.t_eq_slice,
                                &self.z_eq_slice,
                                d_projection,
                                b_projection,
                                a_projection,
                                &d_weights,
                                &b_weights,
                                &a_weights,
                            )
                        })
                    })
                    .collect();
                Ok(segment_sums.into_iter().sum())
            }
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn base_ring_segments<'a>(
        &'a self,
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
        d_rows: usize,
        d_physical_cols: usize,
        required: usize,
    ) -> Result<BaseRingSegments<'a, E>, AkitaError> {
        let (_, segments) = self.packed_segments(d_rows, d_physical_cols)?;
        if a_projection.is_identity() && b_projection.is_identity() && d_projection.is_identity() {
            return Ok(BaseRingSegments::Borrowed(segments));
        }

        let projected = self.build_projected_base_ring_segments(
            a_projection,
            b_projection,
            d_projection,
            d_rows,
            d_physical_cols,
            required,
        )?;
        Ok(BaseRingSegments::Projected(projected))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_projected_base_ring_segments(
        &self,
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
        d_rows: usize,
        d_physical_cols: usize,
        required: usize,
    ) -> Result<Vec<GroupSetupSegment<E>>, AkitaError> {
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

        let mut endpoints = Vec::new();
        endpoints.push(0);
        endpoints.push(required);
        push_projected_d_boundaries(
            &mut endpoints,
            d_rows,
            d_physical_cols,
            self.e_col_offset,
            self.e_eq_slice.len(),
            d_projection.ratio(),
        )?;
        push_projected_role_boundaries(
            &mut endpoints,
            self.n_b,
            self.t_cols,
            b_projection.ratio(),
            "B",
        )?;
        push_projected_role_boundaries(
            &mut endpoints,
            self.n_a,
            self.z_cols,
            a_projection.ratio(),
            "A",
        )?;
        endpoints.sort_unstable();
        endpoints.dedup();

        let e_end = self
            .e_col_offset
            .checked_add(self.e_eq_slice.len())
            .ok_or_else(|| AkitaError::InvalidSetup("setup D active columns overflow".into()))?;
        let segments = (0..endpoints.len().saturating_sub(1))
            .filter_map(|idx| {
                let lo = endpoints[idx];
                let hi = endpoints[idx + 1];
                if lo == hi {
                    return None;
                }

                let d_idx = lo >> d_projection.shift;
                let has_d =
                    if d_physical_cols == 0 || self.e_eq_slice.is_empty() || d_idx >= d_required {
                        false
                    } else {
                        let d_col = d_idx % d_physical_cols;
                        d_col >= self.e_col_offset && d_col < e_end
                    };
                let d_row = if has_d { d_idx / d_physical_cols } else { 0 };
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

                let b_idx = lo >> b_projection.shift;
                let has_b = self.t_cols != 0 && b_idx < b_required;
                let b_row = if has_b { b_idx / self.t_cols } else { 0 };
                let b_start_abs = if has_b { b_row * self.t_cols } else { 0 };
                let b_weight = if has_b {
                    self.b_weights[b_row]
                } else {
                    E::zero()
                };

                let a_idx = lo >> a_projection.shift;
                let has_a = self.z_cols != 0 && a_idx < a_required;
                let a_row = if has_a { a_idx / self.z_cols } else { 0 };
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
                    d_row,
                    d_start_abs,
                    d_weight,
                    has_b,
                    b_row,
                    b_start_abs,
                    b_weight,
                    has_a,
                    a_row,
                    a_start_abs,
                    a_weight,
                })
            })
            .collect();
        Ok(segments)
    }

    fn required_base_ring_rows_for_group(
        &self,
        a_ratio: usize,
        b_ratio: usize,
        d_ratio: usize,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<usize, AkitaError> {
        let d_required = d_rows
            .checked_mul(d_physical_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?
            .checked_mul(d_ratio)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup D base-ring footprint overflow".into())
            })?;
        let b_required = self
            .n_b
            .checked_mul(self.t_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?
            .checked_mul(b_ratio)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup B base-ring footprint overflow".into())
            })?;
        let a_required = self
            .n_a
            .checked_mul(self.z_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?
            .checked_mul(a_ratio)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup A base-ring footprint overflow".into())
            })?;
        Ok(d_required.max(b_required).max(a_required))
    }
}

fn push_projected_role_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    ratio: usize,
    name: &'static str,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let mut boundary = 0usize;
    for _ in 0..rows {
        boundary = boundary
            .checked_add(stride)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("packed {name} boundary overflow")))?;
        endpoints.push(boundary.checked_mul(ratio).ok_or_else(|| {
            AkitaError::InvalidSetup(format!("packed {name} base-ring boundary overflow"))
        })?);
    }
    Ok(())
}

fn push_projected_d_boundaries(
    endpoints: &mut Vec<usize>,
    rows: usize,
    stride: usize,
    active_col_start: usize,
    active_cols: usize,
    ratio: usize,
) -> Result<(), AkitaError> {
    if rows == 0 || stride == 0 {
        return Ok(());
    }
    let active_col_end = active_col_start
        .checked_add(active_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D active columns overflow".into()))?;
    let mut row_start = 0usize;
    for _ in 0..rows {
        let row_end = row_start
            .checked_add(stride)
            .ok_or_else(|| AkitaError::InvalidSetup("packed D boundary overflow".into()))?;
        endpoints.push(row_end.checked_mul(ratio).ok_or_else(|| {
            AkitaError::InvalidSetup("packed D base-ring boundary overflow".into())
        })?);
        if active_cols != 0 {
            let active_start = row_start.checked_add(active_col_start).ok_or_else(|| {
                AkitaError::InvalidSetup("packed D active boundary overflow".into())
            })?;
            let active_end = row_start.checked_add(active_col_end).ok_or_else(|| {
                AkitaError::InvalidSetup("packed D active boundary overflow".into())
            })?;
            endpoints.push(active_start.checked_mul(ratio).ok_or_else(|| {
                AkitaError::InvalidSetup("packed D active base-ring boundary overflow".into())
            })?);
            endpoints.push(active_end.checked_mul(ratio).ok_or_else(|| {
                AkitaError::InvalidSetup("packed D active base-ring boundary overflow".into())
            })?);
        }
        row_start = row_end;
    }
    Ok(())
}
