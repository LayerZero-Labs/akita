//! Shared column-panel span helpers for JL row sweeps.
//!
//! Panels partition a long column axis so each chunk stays L1-resident while
//! every matrix row reuses the same slice (projection kernels, MLE row weights,
//! LUT quad windows).

/// Minimum panel count so parallel schedulers have enough independent tasks.
pub(crate) const JL_MIN_PANELS: usize = 64;

/// Default maximum units (columns, bytes, or quad windows) per panel.
pub(crate) const JL_PANEL_UNIT_MAX: usize = 4096;

/// Minimum `n_rows * cols` before the `parallel` feature fans JL work out.
pub(crate) const JL_PARALLEL_ELEMS_THRESHOLD: usize = 1 << 16;

/// Whether JL projection / MLE should use rayon at this geometry.
#[inline]
pub(crate) fn parallel_jl_enabled(n_rows: usize, cols: usize) -> bool {
    cfg!(feature = "parallel") && n_rows.saturating_mul(cols) >= JL_PARALLEL_ELEMS_THRESHOLD
}

/// Units per panel: at least [`JL_MIN_PANELS`] tasks, each at most `max_per_panel`.
#[inline]
pub(crate) fn panel_span(total_units: usize, max_per_panel: usize) -> usize {
    total_units
        .div_ceil(JL_MIN_PANELS)
        .max(1)
        .min(max_per_panel)
}

/// Byte-aligned column span for row-weight / MLE panels.
#[inline]
pub(crate) fn byte_aligned_panel_cols(cols: usize) -> usize {
    let span = panel_span(cols, JL_PANEL_UNIT_MAX);
    span.div_ceil(4) * 4
}
