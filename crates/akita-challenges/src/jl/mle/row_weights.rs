//! Prover row-weight table construction (`g[i] = Σ_j eq(r_J,j) J[j,i]`).

#[cfg(feature = "parallel")]
use akita_field::parallel::*;

use akita_field::FieldCore;

use crate::jl::packed_byte::BIT_LANES_FOR_BYTE;
use crate::jl::panel::byte_aligned_panel_cols;
use crate::jl::JlProjectionMatrix;

use super::common::scatter_row_weight_range;

/// Fill `g` from batched row equality weights `e_j`.
pub(super) fn fill_row_weights<L: FieldCore>(
    g: &mut [L],
    panel_cols: usize,
    cols: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    row_eq_total: L,
) {
    #[cfg(feature = "parallel")]
    if parallel_jl_enabled(n_rows, cols) {
        g.par_chunks_mut(panel_cols)
            .enumerate()
            .for_each(|(p, g_panel)| {
                scatter_panel(
                    g_panel,
                    p * panel_cols,
                    cols,
                    n_rows,
                    matrix,
                    e_j,
                    row_eq_total,
                );
            });
        return;
    }
    for (p, g_panel) in g.chunks_mut(panel_cols).enumerate() {
        scatter_panel(
            g_panel,
            p * panel_cols,
            cols,
            n_rows,
            matrix,
            e_j,
            row_eq_total,
        );
    }
}

fn scatter_panel<L: FieldCore>(
    g_panel: &mut [L],
    col0: usize,
    cols: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    row_eq_total: L,
) {
    if col0 >= cols {
        return;
    }
    let n = g_panel.len().min(cols - col0);
    let g_active = &mut g_panel[..n];

    if (col0 & 0b111) == 0 {
        scatter_panel_byte_sums(g_active, col0, n_rows, matrix, e_j, row_eq_total);
        return;
    }

    for (j, &w) in e_j.iter().take(n_rows).enumerate() {
        scatter_row_weight_range(g_active, matrix.row_slice(j), col0, w);
    }
}

fn scatter_panel_byte_sums<L: FieldCore>(
    g_active: &mut [L],
    col0: usize,
    n_rows: usize,
    matrix: &JlProjectionMatrix,
    e_j: &[L],
    row_eq_total: L,
) {
    debug_assert_eq!(col0 & 0b111, 0);

    for byte_col in (0..g_active.len()).step_by(8) {
        let lanes = (g_active.len() - byte_col).min(8);
        let matrix_byte = (col0 + byte_col) >> 3;
        let mut ones = [L::zero(); 8];

        for (j, &w) in e_j.iter().take(n_rows).enumerate() {
            let byte = matrix.row_slice(j)[matrix_byte];
            let (set_lanes, set_count) = &BIT_LANES_FOR_BYTE[byte as usize];
            for &lane in set_lanes.iter().take(*set_count) {
                let lane = lane as usize;
                if lane < lanes {
                    ones[lane] += w;
                }
            }
        }

        // sign ∈ {-1,+1}: contrib = (+1)·sum_{bit=1} w + (-1)·sum_{bit=0} w
        //              = 2·sum_{bit=1} w - sum_j w
        for (lane, &sum_ones) in ones.iter().take(lanes).enumerate() {
            g_active[byte_col + lane] = sum_ones + sum_ones - row_eq_total;
        }
    }
}

/// Column panel width for row-weight construction.
pub(super) fn row_weight_panel_cols(cols: usize) -> usize {
    byte_aligned_panel_cols(cols)
}

/// Sum `eq(r_J, ·)` over live rows.
#[inline]
pub(super) fn sum_row_eq<L: FieldCore>(e_j: &[L], n_rows: usize) -> L {
    e_j.iter().take(n_rows).copied().sum()
}
