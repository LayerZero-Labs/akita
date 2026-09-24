use super::*;
use std::sync::Arc;

struct PackedDLayout<'a, E> {
    active_col_start: usize,
    active_cols: usize,
    physical_cols: usize,
    row_weights: &'a [E],
    ratio: usize,
}

struct PackedBLayout<'a, E> {
    segments: &'a [PhysicalBWeightSegment<E>],
    physical_footprint: usize,
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

/// Partition `group`'s projected setup footprint into scan jobs.
pub(super) fn scan_partition<E: Field>(
    group: &SetupContributionGroupPlan<E>,
    active_d_cols: usize,
    d_weights: &[E],
    d_rows: usize,
    d_physical_cols: usize,
) -> Result<GroupScanPartition<E>, AkitaError> {
    if d_weights.len() != d_rows {
        return Err(AkitaError::InvalidSize {
            expected: d_rows,
            actual: d_weights.len(),
        });
    }
    let (required, segments) = build_packed_segments(
        PackedDLayout {
            active_col_start: group.d_col_range.start,
            active_cols: active_d_cols,
            physical_cols: d_physical_cols,
            row_weights: d_weights,
            ratio: group.d_ratio,
        },
        PackedBLayout {
            segments: group.physical_b.weight_segments(),
            physical_footprint: group.physical_b.physical_footprint()?,
            ratio: group.b_ratio,
        },
        PackedALayout {
            cols: group.z_cols,
            row_weights: &group.a_row_weights,
            ratio: group.a_ratio,
        },
    )?;
    Ok(GroupScanPartition { required, segments })
}

fn build_packed_segments<E: Field>(
    d: PackedDLayout<'_, E>,
    b: PackedBLayout<'_, E>,
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
        .physical_footprint
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
    for segment in b.segments {
        endpoints.push(segment.physical_start.checked_mul(b.ratio).ok_or_else(|| {
            AkitaError::InvalidSetup("packed B segment boundary overflow".into())
        })?);
        let end = segment
            .physical_start
            .checked_add(segment.len)
            .and_then(|end| end.checked_mul(b.ratio))
            .ok_or_else(|| AkitaError::InvalidSetup("packed B segment extent overflow".into()))?;
        endpoints.push(end);
    }
    push_projected_role_boundaries(&mut endpoints, a.row_weights.len(), a.cols, a.ratio, "A")?;
    endpoints.sort_unstable();
    endpoints.dedup();

    let malformed = || AkitaError::InvalidSetup("packed setup segment is malformed".into());
    let mut segments = Vec::new();
    for window in endpoints.windows(2) {
        let &[lo, hi] = window else {
            return Err(malformed());
        };
        if lo == hi {
            continue;
        }

        let d_idx = lo.checked_div(d.ratio).ok_or_else(malformed)?;
        let has_d = if d.physical_cols == 0 || d.active_cols == 0 || lo >= d_required {
            false
        } else {
            let d_col = d_idx.checked_rem(d.physical_cols).ok_or_else(malformed)?;
            d_col >= d.active_col_start && d_col < e_end
        };
        let (d_start_abs, d_weight) = if has_d {
            let d_row = d_idx.checked_div(d.physical_cols).ok_or_else(malformed)?;
            let d_start_abs = d_row
                .checked_mul(d.physical_cols)
                .and_then(|start| start.checked_add(d.active_col_start))
                .ok_or_else(malformed)?;
            let d_weight = *d.row_weights.get(d_row).ok_or_else(malformed)?;
            (d_start_abs, d_weight)
        } else {
            (0, E::zero())
        };

        let b_idx = lo.checked_div(b.ratio).ok_or_else(malformed)?;
        let b_segment = b.segments.iter().find(|segment| {
            b_idx >= segment.physical_start
                && b_idx < segment.physical_start.saturating_add(segment.len)
        });
        let has_b = b_segment.is_some();
        let b_start_abs = b_segment.map_or(0, |segment| segment.physical_start);
        let b_terms = b_segment.map_or_else(|| Arc::from([]), |segment| Arc::clone(&segment.terms));

        let has_a = a.cols != 0 && lo < a_required;
        let (a_start_abs, a_row_weight) = if has_a {
            let a_row = lo
                .checked_div(a.ratio)
                .and_then(|a_idx| a_idx.checked_div(a.cols))
                .ok_or_else(malformed)?;
            let a_start_abs = a_row.checked_mul(a.cols).ok_or_else(malformed)?;
            let a_row_weight = *a.row_weights.get(a_row).ok_or_else(malformed)?;
            (a_start_abs, a_row_weight)
        } else {
            (0, E::zero())
        };

        if !has_d && !has_b && !has_a {
            continue;
        }

        segments.push(GroupSetupSegment {
            lo,
            hi,
            has_d,
            d_start_abs,
            d_weight,
            has_b,
            b_start_abs,
            b_terms,
            has_a,
            a_start_abs,
            a_row_weight,
        });
    }
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
