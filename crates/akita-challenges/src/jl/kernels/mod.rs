//! JL projection kernels: reference (checked `i64`) and fast (`i8` digits +
//! `i32` accumulation, optional SIMD).
//!
//! The fast path narrows the validated balanced digits to `i8` once, then
//! projects with a **column-panel** layout: the columns are partitioned into
//! byte-aligned panels whose `i8` digit chunk stays L1-resident and is reused
//! across all `n_rows` rows. The witness is therefore read once (not once per
//! row), and the parallel path fans out over panels (reducing per-panel partial
//! images) instead of over rows. See `project_columns` below.

mod reference;
mod scalar;
pub(crate) use scalar::SIGNS_FOR_BYTE;

#[cfg(target_arch = "x86_64")]
mod simd_x86;

#[cfg(target_arch = "aarch64")]
mod simd_neon;

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

pub(crate) use reference::project_row as project_row_reference;

/// Map a packed 2-bit pair to its ternary sign: `0b00 -> -1`, `0b11 -> +1`,
/// `0b01`/`0b10 -> 0`.
#[inline]
pub(crate) fn pair_to_sign(pair: u8) -> i8 {
    ((pair == 0b11) as i8) - ((pair == 0b00) as i8)
}

/// Target column count per parallel panel. Sized so a panel's `i8` digit chunk
/// (one byte per column) stays comfortably within L1 and is reused across every
/// row of the panel.
#[cfg(feature = "parallel")]
fn parallel_panel_bytes(row_bytes: usize) -> usize {
    super::panel::panel_span(row_bytes, super::panel::JL_PANEL_UNIT_MAX).min(row_bytes.max(1))
}
/// One row coordinate over one byte-aligned column panel, dispatched to the
/// fastest available kernel (NEON, AVX-512/AVX2 `madd`, or scalar).
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

/// Accumulate panel `p`'s contribution to every row coordinate into `coords`.
///
/// Panel `p` owns the byte range `[p*panel_bytes, (p+1)*panel_bytes)` of every
/// row (its `i8` digit chunk is reused across all rows), so each digit and each
/// matrix entry is read exactly once across the whole projection.
#[inline]
fn accumulate_panel(
    coords: &mut [i32],
    p: usize,
    panel_bytes: usize,
    row_bytes: usize,
    packed_rows: &[u8],
    digits: &[i8],
    cols: usize,
) {
    let b0 = p * panel_bytes;
    if b0 >= row_bytes {
        return;
    }
    let b1 = (b0 + panel_bytes).min(row_bytes);
    let col0 = b0 * 4;
    let col1 = (b1 * 4).min(cols);
    let panel_cols = col1 - col0;
    let digit_chunk = &digits[col0..col1];

    for (r, coord) in coords.iter_mut().enumerate() {
        let row_start = r * row_bytes;
        let row_chunk = &packed_rows[row_start + b0..row_start + b1];
        *coord += project_row_fast(row_chunk, digit_chunk, panel_cols);
    }
}

#[inline]
fn accumulate_panel_scalar(
    coords: &mut [i32],
    p: usize,
    panel_bytes: usize,
    row_bytes: usize,
    packed_rows: &[u8],
    digits: &[i8],
    cols: usize,
) {
    let b0 = p * panel_bytes;
    if b0 >= row_bytes {
        return;
    }
    let b1 = (b0 + panel_bytes).min(row_bytes);
    let col0 = b0 * 4;
    let col1 = (b1 * 4).min(cols);
    let panel_cols = col1 - col0;
    let digit_chunk = &digits[col0..col1];

    for (r, coord) in coords.iter_mut().enumerate() {
        let row_start = r * row_bytes;
        let row_chunk = &packed_rows[row_start + b0..row_start + b1];
        *coord += scalar::project_row(row_chunk, digit_chunk, panel_cols);
    }
}

/// Project all rows with the fast kernel using the column-panel layout.
///
/// Validated digits are narrowed to `i8` once (`|d| <= MAX_JL_DIGIT` fits `i8`).
/// The narrowing pass is `O(cols)`, amortized over `n_rows` projection passes.
pub(crate) fn project_rows_fast(
    n_rows: usize,
    row_bytes: usize,
    packed_rows: &[u8],
    digits: &[i32],
    cols: usize,
    parallel: bool,
) -> Vec<i32> {
    let digits_i8: Vec<i8> = digits.iter().map(|&d| d as i8).collect();

    #[cfg(feature = "parallel")]
    if parallel {
        let panel_bytes = parallel_panel_bytes(row_bytes);
        let num_panels = row_bytes.div_ceil(panel_bytes);
        return (0..num_panels)
            .into_par_iter()
            .map(|p| {
                let mut partial = vec![0i32; n_rows];
                accumulate_panel(
                    &mut partial,
                    p,
                    panel_bytes,
                    row_bytes,
                    packed_rows,
                    &digits_i8,
                    cols,
                );
                partial
            })
            .reduce(
                || vec![0i32; n_rows],
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

    // Serial path still blocks by column panel so the digit chunk stays
    // L1-resident and is reused across rows.
    let panel_bytes = row_bytes.clamp(1, super::panel::JL_PANEL_UNIT_MAX);
    let num_panels = row_bytes.div_ceil(panel_bytes);
    let mut coords = vec![0i32; n_rows];
    for p in 0..num_panels {
        accumulate_panel(
            &mut coords,
            p,
            panel_bytes,
            row_bytes,
            packed_rows,
            &digits_i8,
            cols,
        );
    }
    coords
}

/// Project all rows with the scalar row kernel, bypassing SIMD dispatch.
///
/// This is a benchmark/differential hook used to make scalar A/B evidence
/// explicit on hosts where [`project_rows_fast`] would otherwise dispatch to
/// NEON or AVX.
pub(crate) fn project_rows_scalar(
    n_rows: usize,
    row_bytes: usize,
    packed_rows: &[u8],
    digits: &[i32],
    cols: usize,
) -> Vec<i32> {
    let digits_i8: Vec<i8> = digits.iter().map(|&d| d as i8).collect();
    let panel_bytes = row_bytes.clamp(1, super::panel::JL_PANEL_UNIT_MAX);
    let num_panels = row_bytes.div_ceil(panel_bytes);
    let mut coords = vec![0i32; n_rows];
    for p in 0..num_panels {
        accumulate_panel_scalar(
            &mut coords,
            p,
            panel_bytes,
            row_bytes,
            packed_rows,
            &digits_i8,
            cols,
        );
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
        let row_bytes = (cols * 2).div_ceil(8);
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
