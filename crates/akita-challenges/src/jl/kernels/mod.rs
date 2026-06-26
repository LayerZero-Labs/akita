//! JL projection kernels: reference (checked `i64`) and fast (`i8` digits +
//! `i32` accumulation, optional SIMD).

mod reference;
mod scalar;

#[cfg(target_arch = "x86_64")]
mod simd_x86;

#[cfg(target_arch = "aarch64")]
mod simd_neon;

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

pub(crate) use reference::project_row as project_row_reference;

fn fast_panel_bytes(row_bytes: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        super::panel::panel_span(row_bytes, super::panel::JL_PANEL_UNIT_MAX).min(row_bytes.max(1))
    }
    #[cfg(not(feature = "parallel"))]
    {
        row_bytes.clamp(1, super::panel::JL_PANEL_UNIT_MAX)
    }
}

#[inline]
fn project_row_fast(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    project_row_fast_dispatch(row, digits, cols)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn project_row_fast_dispatch(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    // SAFETY: NEON is mandatory on aarch64; digit-bound contract holds.
    unsafe { simd_neon::project_row_neon(row, digits, cols) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn project_row_fast_dispatch(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    if simd_x86::avx512_available() {
        // SAFETY: feature detection + digit-bound contract.
        return unsafe { simd_x86::project_row_avx512(row, digits, cols) };
    }
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: feature detection + digit-bound contract.
        return unsafe { simd_x86::project_row_avx2(row, digits, cols) };
    }
    scalar::project_row(row, digits, cols)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
fn project_row_fast_dispatch(row: &[u8], digits: &[i8], cols: usize) -> i32 {
    scalar::project_row(row, digits, cols)
}

struct PanelProjection<'a> {
    row_bytes: usize,
    packed_rows: &'a [u8],
    digits: &'a [i8],
    cols: usize,
    project_row: fn(&[u8], &[i8], usize) -> i32,
}

#[inline]
fn accumulate_panel(coords: &mut [i32], p: usize, panel_bytes: usize, ctx: &PanelProjection<'_>) {
    let b0 = p * panel_bytes;
    if b0 >= ctx.row_bytes {
        return;
    }
    let b1 = (b0 + panel_bytes).min(ctx.row_bytes);
    let col0 = b0 * 8;
    let col1 = (b1 * 8).min(ctx.cols);
    let panel_cols = col1 - col0;
    let digit_chunk = &ctx.digits[col0..col1];

    for (r, coord) in coords.iter_mut().enumerate() {
        let row_start = r * ctx.row_bytes;
        let row_chunk = &ctx.packed_rows[row_start + b0..row_start + b1];
        *coord += (ctx.project_row)(row_chunk, digit_chunk, panel_cols);
    }
}

/// Project all rows with the fast kernel using the column-panel layout.
pub(crate) fn project_rows_fast(
    n_rows: usize,
    row_bytes: usize,
    packed_rows: &[u8],
    digits: &[i32],
    cols: usize,
    parallel: bool,
) -> Vec<i32> {
    project_rows_with_kernel(
        &ProjectionJob {
            n_rows,
            row_bytes,
            packed_rows,
            digits,
            cols,
        },
        parallel,
        project_row_fast,
        fast_panel_bytes,
    )
}

fn serial_panel_bytes(row_bytes: usize) -> usize {
    row_bytes.clamp(1, super::panel::JL_PANEL_UNIT_MAX)
}

/// Project all rows with the scalar row kernel, bypassing SIMD dispatch.
pub(crate) fn project_rows_scalar(
    n_rows: usize,
    row_bytes: usize,
    packed_rows: &[u8],
    digits: &[i32],
    cols: usize,
) -> Vec<i32> {
    project_rows_with_kernel(
        &ProjectionJob {
            n_rows,
            row_bytes,
            packed_rows,
            digits,
            cols,
        },
        false,
        scalar::project_row,
        serial_panel_bytes,
    )
}

struct ProjectionJob<'a> {
    n_rows: usize,
    row_bytes: usize,
    packed_rows: &'a [u8],
    digits: &'a [i32],
    cols: usize,
}

fn project_rows_with_kernel(
    job: &ProjectionJob<'_>,
    parallel: bool,
    project_row_kernel: fn(&[u8], &[i8], usize) -> i32,
    panel_bytes_for: fn(usize) -> usize,
) -> Vec<i32> {
    let digits_i8: Vec<i8> = job.digits.iter().map(|&d| d as i8).collect();
    let ctx = PanelProjection {
        row_bytes: job.row_bytes,
        packed_rows: job.packed_rows,
        digits: &digits_i8,
        cols: job.cols,
        project_row: project_row_kernel,
    };

    #[cfg(feature = "parallel")]
    if parallel {
        let panel_bytes = panel_bytes_for(job.row_bytes);
        let num_panels = job.row_bytes.div_ceil(panel_bytes);
        return (0..num_panels)
            .into_par_iter()
            .map(|p| {
                let mut partial = vec![0i32; job.n_rows];
                accumulate_panel(&mut partial, p, panel_bytes, &ctx);
                partial
            })
            .reduce(
                || vec![0i32; job.n_rows],
                |mut acc, partial| {
                    for (a, b) in acc.iter_mut().zip(partial.iter()) {
                        *a += *b;
                    }
                    acc
                },
            );
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    let panel_bytes = panel_bytes_for(job.row_bytes);
    let num_panels = job.row_bytes.div_ceil(panel_bytes);
    let mut coords = vec![0i32; job.n_rows];
    for p in 0..num_panels {
        accumulate_panel(&mut coords, p, panel_bytes, &ctx);
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jl::MAX_JL_DIGIT;

    #[test]
    fn simd_dispatch_matches_scalar_kernel() {
        let cols: usize = 1027;
        let row_bytes = cols.div_ceil(8);
        let row: Vec<u8> = (0..row_bytes)
            .map(|i: usize| i.wrapping_mul(37) as u8)
            .collect();
        let digits: Vec<i8> = (0..cols).map(|i| (i % 17) as i8 - 8).collect();
        assert!(digits.iter().all(|d| (*d as i32).abs() <= MAX_JL_DIGIT));

        let scalar = scalar::project_row(&row, &digits, cols);
        let dispatched = project_row_fast(&row, &digits, cols);
        assert_eq!(scalar, dispatched);
    }
}
