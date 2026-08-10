use super::*;

struct PackedDLayout<'a, E> {
    active_col_start: usize,
    active_cols: usize,
    physical_cols: usize,
    row_weights: &'a [E],
    ratio: usize,
}

struct PackedFlatBLayout {
    weights_len: usize,
    ratio: usize,
}

struct PackedALayout<'a, E> {
    cols: usize,
    row_weights: &'a [E],
    ratio: usize,
}

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
        let weights = self.direct_scan_weights.as_ref().ok_or_else(|| {
            AkitaError::InvalidSetup("direct setup scan weights are missing".into())
        })?;
        if d_weights.len() != d_rows {
            return Err(AkitaError::InvalidSize {
                expected: d_rows,
                actual: d_weights.len(),
            });
        }
        let (required, segments) = build_packed_segments(
            PackedDLayout {
                active_col_start: self.d_col_range.start,
                active_cols: weights.e.len(),
                physical_cols: d_physical_cols,
                row_weights: d_weights,
                ratio: d_ratio,
            },
            PackedFlatBLayout {
                weights_len: weights.b_setup.len(),
                ratio: b_ratio,
            },
            PackedALayout {
                cols: self.z_cols,
                row_weights: &self.a_row_weights,
                ratio: a_ratio,
            },
        )?;
        self.required = required;
        self.segments = segments.into();
        Ok(())
    }
}

fn build_packed_segments<E: FieldCore>(
    d: PackedDLayout<'_, E>,
    b: PackedFlatBLayout,
    a: PackedALayout<'_, E>,
) -> Result<(usize, Vec<GroupSetupSegment<E>>), AkitaError> {
    if [a.ratio, b.ratio, d.ratio]
        .into_iter()
        .any(|ratio| !ratio.is_power_of_two())
    {
        return Err(AkitaError::InvalidSetup(
            "setup projection ratios must be powers of two".into(),
        ));
    }
    let e_end = d
        .active_col_start
        .checked_add(d.active_cols)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    if e_end > d.physical_cols {
        return Err(AkitaError::InvalidSetup(
            "setup D weights exceed physical D width".into(),
        ));
    }

    let d_required = d
        .row_weights
        .len()
        .checked_mul(d.physical_cols)
        .and_then(|len| len.checked_mul(d.ratio))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
    let b_required = b
        .weights_len
        .checked_mul(b.ratio)
        .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
    let a_required = a
        .row_weights
        .len()
        .checked_mul(a.cols)
        .and_then(|len| len.checked_mul(a.ratio))
        .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
    let required = d_required.max(b_required).max(a_required);

    let mut endpoints = Vec::new();
    endpoints.push(0);
    endpoints.push(required);
    push_group_d_boundaries(
        &mut endpoints,
        d.row_weights.len(),
        d.physical_cols,
        d.active_col_start,
        d.active_cols,
        d.ratio,
    )?;
    endpoints.push(b_required);
    push_projected_role_boundaries(&mut endpoints, a.row_weights.len(), a.cols, a.ratio, "A")?;
    endpoints.sort_unstable();
    endpoints.dedup();

    let segments = (0..endpoints.len().saturating_sub(1))
        .filter_map(|idx| {
            let lo = endpoints[idx];
            let hi = endpoints[idx + 1];
            if lo == hi {
                return None;
            }

            let d_idx = lo / d.ratio;
            let has_d = if d.physical_cols == 0 || d.active_cols == 0 || lo >= d_required {
                false
            } else {
                let d_col = d_idx % d.physical_cols;
                d_col >= d.active_col_start && d_col < e_end
            };
            let d_row = if has_d { d_idx / d.physical_cols } else { 0 };
            let d_start_abs = if has_d {
                d_row * d.physical_cols + d.active_col_start
            } else {
                0
            };
            let d_weight = if has_d {
                d.row_weights[d_row]
            } else {
                E::zero()
            };

            let has_b = b.weights_len != 0 && lo < b_required;
            let b_start_abs = 0;

            let a_idx = lo / a.ratio;
            let has_a = a.cols != 0 && lo < a_required;
            let a_row = if has_a { a_idx / a.cols } else { 0 };
            let a_start_abs = if has_a { a_row * a.cols } else { 0 };
            let a_row_weight = if has_a {
                a.row_weights[a_row]
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
