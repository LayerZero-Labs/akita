use super::*;

/// Target scan-job size. At fp128/D64 this is 2 MiB of contiguous setup data,
/// large enough to amortize scheduling while exposing hundreds of root jobs.
pub(super) const SETUP_SCAN_JOB_RINGS: usize = 2048;

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    pub(crate) fn refresh_segments(
        &mut self,
        d_weights: &[E],
        d_rows: usize,
        d_physical_cols: usize,
        a_ratio: usize,
        b_ratio: usize,
        d_ratio: usize,
    ) -> Result<(), AkitaError> {
        let (required, segments) = build_packed_segments(
            self.d_col_range.start,
            self.e_eq_slice.len(),
            self.t_cols,
            self.z_cols,
            self.n_a,
            self.n_b,
            &self.a_row_weights,
            &self.b_weights,
            d_weights,
            d_rows,
            d_physical_cols,
            a_ratio,
            b_ratio,
            d_ratio,
        )?;
        self.required = required;
        self.segments = segments.into();
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_packed_segments<E: FieldCore>(
    d_col_start: usize,
    e_eq_len: usize,
    t_cols: usize,
    z_cols: usize,
    n_a: usize,
    n_b: usize,
    a_row_weights: &[E],
    b_weights: &[E],
    d_weights: &[E],
    d_rows: usize,
    d_physical_cols: usize,
    a_ratio: usize,
    b_ratio: usize,
    d_ratio: usize,
) -> Result<(usize, Vec<GroupSetupSegment<E>>), AkitaError> {
    if [a_ratio, b_ratio, d_ratio]
        .into_iter()
        .any(|ratio| !ratio.is_power_of_two())
    {
        return Err(AkitaError::InvalidSetup(
            "setup projection ratios must be powers of two".into(),
        ));
    }
    if d_weights.len() != d_rows {
        return Err(AkitaError::InvalidSize {
            expected: d_rows,
            actual: d_weights.len(),
        });
    }
    if a_row_weights.len() != n_a {
        return Err(AkitaError::InvalidSize {
            expected: n_a,
            actual: a_row_weights.len(),
        });
    }
    if b_weights.len() != n_b {
        return Err(AkitaError::InvalidSize {
            expected: n_b,
            actual: b_weights.len(),
        });
    }
    let e_end = d_col_start
        .checked_add(e_eq_len)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    if e_end > d_physical_cols {
        return Err(AkitaError::InvalidSetup(
            "setup D weights exceed physical D width".into(),
        ));
    }

    let d_required = d_rows
        .checked_mul(d_physical_cols)
        .and_then(|len| len.checked_mul(d_ratio))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    let b_required = n_b
        .checked_mul(t_cols)
        .and_then(|len| len.checked_mul(b_ratio))
        .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
    let a_required = n_a
        .checked_mul(z_cols)
        .and_then(|len| len.checked_mul(a_ratio))
        .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
    let required = d_required.max(b_required).max(a_required);

    let mut endpoints = Vec::new();
    endpoints.push(0);
    endpoints.push(required);
    push_group_d_boundaries(
        &mut endpoints,
        d_rows,
        d_physical_cols,
        d_col_start,
        e_eq_len,
        d_ratio,
    )?;
    push_projected_role_boundaries(&mut endpoints, n_b, t_cols, b_ratio, "B")?;
    push_projected_role_boundaries(&mut endpoints, n_a, z_cols, a_ratio, "A")?;
    endpoints.sort_unstable();
    endpoints.dedup();

    let segments = (0..endpoints.len().saturating_sub(1))
        .filter_map(|idx| {
            let lo = endpoints[idx];
            let hi = endpoints[idx + 1];
            if lo == hi {
                return None;
            }

            let d_idx = lo / d_ratio;
            let has_d = if d_physical_cols == 0 || e_eq_len == 0 || lo >= d_required {
                false
            } else {
                let d_col = d_idx % d_physical_cols;
                d_col >= d_col_start && d_col < e_end
            };
            let d_row = if has_d { d_idx / d_physical_cols } else { 0 };
            let d_start_abs = if has_d {
                d_row * d_physical_cols + d_col_start
            } else {
                0
            };
            let d_weight = if has_d { d_weights[d_row] } else { E::zero() };

            let b_idx = lo / b_ratio;
            let has_b = t_cols != 0 && lo < b_required;
            let b_row = if has_b { b_idx / t_cols } else { 0 };
            let b_start_abs = if has_b { b_row * t_cols } else { 0 };
            let b_weight = if has_b { b_weights[b_row] } else { E::zero() };

            let a_idx = lo / a_ratio;
            let has_a = z_cols != 0 && lo < a_required;
            let a_row = if has_a { a_idx / z_cols } else { 0 };
            let a_start_abs = if has_a { a_row * z_cols } else { 0 };
            let a_row_weight = if has_a {
                a_row_weights[a_row]
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
                a_row_weight,
            })
        })
        .collect::<Vec<_>>();
    let mut jobs = Vec::new();
    for segment in segments {
        let mut lo = segment.lo;
        while lo < segment.hi {
            let hi = lo.saturating_add(SETUP_SCAN_JOB_RINGS).min(segment.hi);
            let mut job = segment.clone();
            job.lo = lo;
            job.hi = hi;
            jobs.push(job);
            lo = hi;
        }
    }

    Ok((required, jobs))
}

#[inline(always)]
fn push_group_d_boundaries(
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
