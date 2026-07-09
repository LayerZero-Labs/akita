use super::*;

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    #[cfg(test)]
    pub(crate) fn refresh_segments(
        &mut self,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<(), AkitaError> {
        let (required, segments) = build_packed_segments(
            self.e_col_offset,
            self.e_eq_slice.len(),
            self.t_cols,
            self.z_cols,
            self.n_a,
            self.n_b,
            &self.a_weights,
            &self.b_weights,
            &self.d_weights,
            d_rows,
            d_physical_cols,
        )?;
        self.required = required;
        self.segments = segments;
        Ok(())
    }

    pub(super) fn packed_segments(
        &self,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<(usize, &[GroupSetupSegment<E>]), AkitaError> {
        debug_assert_eq!(self.d_weights.len(), d_rows);
        debug_assert_eq!(self.a_weights.len(), self.n_a);
        debug_assert_eq!(self.b_weights.len(), self.n_b);
        debug_assert_eq!(self.t_eq_slice.len(), self.t_cols);
        debug_assert_eq!(self.z_eq_slice.len(), self.z_cols);
        debug_assert!(self.e_col_offset.saturating_add(self.e_eq_slice.len()) <= d_physical_cols);
        debug_assert_eq!(
            self.required,
            setup_group_required(
                d_rows,
                d_physical_cols,
                self.n_b,
                self.t_cols,
                self.n_a,
                self.z_cols,
            )?
        );
        Ok((self.required, &self.segments))
    }
}

#[allow(clippy::too_many_arguments)]
fn setup_group_required(
    d_rows: usize,
    d_physical_cols: usize,
    n_b: usize,
    t_cols: usize,
    n_a: usize,
    z_cols: usize,
) -> Result<usize, AkitaError> {
    let d_required = d_rows
        .checked_mul(d_physical_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    let b_required = n_b
        .checked_mul(t_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
    let a_required = n_a
        .checked_mul(z_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
    Ok(d_required.max(b_required).max(a_required))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_packed_segments<E: FieldCore>(
    e_col_offset: usize,
    e_eq_len: usize,
    t_cols: usize,
    z_cols: usize,
    n_a: usize,
    n_b: usize,
    a_weights: &[E],
    b_weights: &[E],
    d_weights: &[E],
    d_rows: usize,
    d_physical_cols: usize,
) -> Result<(usize, Vec<GroupSetupSegment<E>>), AkitaError> {
    if d_weights.len() != d_rows {
        return Err(AkitaError::InvalidSize {
            expected: d_rows,
            actual: d_weights.len(),
        });
    }
    if a_weights.len() != n_a {
        return Err(AkitaError::InvalidSize {
            expected: n_a,
            actual: a_weights.len(),
        });
    }
    if b_weights.len() != n_b {
        return Err(AkitaError::InvalidSize {
            expected: n_b,
            actual: b_weights.len(),
        });
    }
    let e_end = e_col_offset
        .checked_add(e_eq_len)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    if e_end > d_physical_cols {
        return Err(AkitaError::InvalidSetup(
            "setup D weights exceed physical D width".into(),
        ));
    }

    let d_required = d_rows
        .checked_mul(d_physical_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    let b_required = n_b
        .checked_mul(t_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
    let a_required = n_a
        .checked_mul(z_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
    let required = d_required.max(b_required).max(a_required);

    let mut endpoints = Vec::new();
    endpoints.push(0);
    endpoints.push(required);
    push_group_d_boundaries(
        &mut endpoints,
        d_rows,
        d_physical_cols,
        e_col_offset,
        e_eq_len,
    )?;
    push_role_boundaries(&mut endpoints, n_b, t_cols, "B")?;
    push_role_boundaries(&mut endpoints, n_a, z_cols, "A")?;
    endpoints.sort_unstable();
    endpoints.dedup();

    let segments = (0..endpoints.len().saturating_sub(1))
        .filter_map(|idx| {
            let lo = endpoints[idx];
            let hi = endpoints[idx + 1];
            if lo == hi {
                return None;
            }

            let has_d = if d_physical_cols == 0 || e_eq_len == 0 || lo >= d_required {
                false
            } else {
                let d_col = lo % d_physical_cols;
                d_col >= e_col_offset && d_col < e_end
            };
            let d_row = if has_d { lo / d_physical_cols } else { 0 };
            let d_start_abs = if has_d {
                d_row * d_physical_cols + e_col_offset
            } else {
                0
            };
            let d_weight = if has_d { d_weights[d_row] } else { E::zero() };

            let has_b = t_cols != 0 && lo < b_required;
            let b_row = if has_b { lo / t_cols } else { 0 };
            let b_start_abs = if has_b { b_row * t_cols } else { 0 };
            let b_weight = if has_b { b_weights[b_row] } else { E::zero() };

            let has_a = z_cols != 0 && lo < a_required;
            let a_row = if has_a { lo / z_cols } else { 0 };
            let a_start_abs = if has_a { a_row * z_cols } else { 0 };
            let a_weight = if has_a { a_weights[a_row] } else { E::zero() };

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

pub(super) fn validate_group_chunk_layout(
    group: &SetupContributionGroupInputs,
    num_groups: usize,
) -> Result<(), AkitaError> {
    if group.chunks.is_empty()
        || group.blocks_per_chunk == 0
        || !group.blocks_per_chunk.is_power_of_two()
    {
        return Err(AkitaError::InvalidSetup(
            "malformed setup witness chunk layout".into(),
        ));
    }
    if group
        .chunks
        .len()
        .checked_mul(group.blocks_per_chunk)
        .ok_or_else(|| AkitaError::InvalidSetup("setup chunk block coverage overflow".into()))?
        != group.num_blocks
    {
        return Err(AkitaError::InvalidSetup(
            "setup witness chunk windows do not tile num_blocks".into(),
        ));
    }
    if group.chunks.len() > 1 && num_groups != 1 {
        // This is an intentional product-surface limit, not a verifier panic
        // guard: multi-chunk witness layouts are currently only enabled for
        // the singleton recursive suffix. Keep the rejection here so direct
        // callers cannot build an ambiguous multi-chunk, multi-group setup
        // plan before the schedule/proof boundary has learned that shape.
        return Err(AkitaError::InvalidSetup(
            "multi-chunk setup contribution requires exactly one commitment group".into(),
        ));
    }
    Ok(())
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
    let active_col_end = active_col_start
        .checked_add(active_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D active columns overflow".into()))?;
    let mut row_start = 0usize;
    for _ in 0..rows {
        let row_end = row_start
            .checked_add(stride)
            .ok_or_else(|| AkitaError::InvalidSetup("packed D boundary overflow".into()))?;
        endpoints.push(row_end);
        if active_cols != 0 {
            endpoints.push(row_start.checked_add(active_col_start).ok_or_else(|| {
                AkitaError::InvalidSetup("packed D active boundary overflow".into())
            })?);
            endpoints.push(row_start.checked_add(active_col_end).ok_or_else(|| {
                AkitaError::InvalidSetup("packed D active boundary overflow".into())
            })?);
        }
        row_start = row_end;
    }
    Ok(())
}
