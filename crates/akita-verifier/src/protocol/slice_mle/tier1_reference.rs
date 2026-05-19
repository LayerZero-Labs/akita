//! Reference (brute-force) implementation of the tiered root M-row
//! contribution.
//!
//! `specs/tiered_commit.md` §3 specifies the tiered M-row layout as
//!
//! ```text
//! consistency (1) | public | D (n_d) | tier1 (f · n_b' · num_points)
//!   | F (n_F · num_points) | A (n_a)
//! ```
//!
//! Each tier-1 row encodes the single relation
//! `B' · t̂_i − G · û_i = 0`, single-counted across the
//! setup-matrix half and the structured `−G · û_i` half. Each F row
//! encodes `F · û_concat − u_final = 0`.
//!
//! This module computes only the **new** rows the tiered path
//! introduces (tier1 + F). The D and A halves are unchanged from the
//! legacy path and continue to be computed by
//! [`super::setup_contribution::compute_setup_contribution`]. This split
//! is convenient for testing: the reference implementation here is
//! brute-force and slow but obviously correct, and serves as the oracle
//! against which any future optimised tiered evaluator can be
//! validated.
//!
//! See `specs/tiered_commit.md` §10 for the optimised production loop
//! that will share the `B'` α-eval rectangle across all `f` chunks.
//!
//! Nothing currently calls into this module from the rest of the
//! verifier — that wiring lands in Phase 4c/4d when the prover's M-row
//! generation switches to the tiered layout. Until then, the
//! `#[allow(dead_code)]` annotation keeps clippy quiet without hiding
//! the public surface.

#![allow(dead_code)]

use akita_algebra::ring::eval_ring_at_pows;
use akita_field::{CanonicalField, ExtField, FieldCore};
use akita_types::layout::flat_matrix::RingMatrixView;

/// Inputs describing one opening point's tiered M-row contribution.
pub struct Tier1AndFInputs<'a, F: FieldCore, E: FieldCore, const D: usize> {
    /// `B'` view (the column-window restriction of B used by the tiered
    /// path). `b_prime_view.num_rows() == n_b'`,
    /// `b_prime_view.num_cols() == chunk_width`.
    pub b_prime_view: RingMatrixView<'a, F, D>,
    /// `F` view. `f_view.num_rows() == n_F`,
    /// `f_view.num_cols() == n_b' · split_factor · num_digits_outer`.
    pub f_view: RingMatrixView<'a, F, D>,
    /// Per-row eq-weights for **all** tier-1 rows of every point,
    /// flattened in `(point, chunk, b'_row)` major order. Length:
    /// `num_points · split_factor · n_b'`. These are slices of the
    /// caller's `eq_tau1[tier1_start..tier1_end]`.
    pub tier1_row_weights: &'a [E],
    /// Per-row eq-weights for **all** F rows of every point, flattened
    /// in `(point, F_row)` major order. Length: `num_points · n_F`.
    /// Slices of the caller's `eq_tau1[f_start..f_end]`.
    pub f_row_weights: &'a [E],
    /// `α^0, α^1, …, α^{D-1}`.
    pub alpha_pows: &'a [E],
    /// The verifier's outer challenge `r_col`. Length = log2(M column
    /// count rounded up to a power of two).
    pub full_vec_randomness: &'a [E],
    /// Outer gadget vector `G = (1, 2^{outer_log_basis}, 2^{2·b}, …)`,
    /// length `num_digits_outer`. Currently expressed in the base field
    /// `F`; the reference lifts to `E` via `MulBase`.
    pub outer_gadget: &'a [F],
    /// M-column offset of the `t̂` segment.
    pub offset_t: usize,
    /// M-column offset of the `uhat` segment (lies between `t̂` and the
    /// blinding/`z` segments per the tiered layout in §3).
    pub offset_uhat: usize,
    /// Splitting factor `f` (spec §2).
    pub split_factor: usize,
    /// Outer gadget depth `δ_outer` (spec §5).
    pub num_digits_outer: usize,
    /// Inner B-physical layout describing how a B-physical column index
    /// `c ∈ [0, outer_width)` decodes into `(digit, a_row, block,
    /// poly_idx, point_idx)`.
    pub b_physical: BPhysicalLayout,
    /// Number of opening-point commitments.
    pub num_points: usize,
}

/// B-physical layout for one polynomial bundle (matches
/// `get_eq_indices_for_b` in `setup_contribution.rs`).
#[derive(Clone, Copy, Debug)]
pub struct BPhysicalLayout {
    /// Inner SIS rank `n_a`.
    pub n_a: usize,
    /// `B = 2^r_vars` — number of committed blocks per polynomial.
    pub num_blocks: usize,
    /// Open-side gadget depth `δ_open`.
    pub depth_open: usize,
    /// `num_polys_per_point[g]` — the bundle size at each point. For the
    /// reference impl we only need the per-point bundle size; we accept
    /// a fixed-shape ragged layout via a flat slice + offsets.
    /// `total_polys_across_points = Σ_g num_polys_per_point[g]`.
    pub num_t_vectors: usize,
}

/// Brute-force reference. Iterates every (row, col) of every tier-1 and
/// F row of the materialised M, summing `eq_tau1[row] · eq_col[col] ·
/// M[row, col]` directly. `O(rows · M_cols)` runtime — for tests only.
///
/// Returns the tiered "tier1 + F" contribution to `M̃(r_row, r_col)`.
/// The caller adds the (unchanged) D, A, W-structured, T-structured,
/// Z-structured, r-tail, and (zk) blinding contributions separately.
///
/// `num_polys_per_point[g]` (length `num_points`) is the per-point
/// polynomial bundle size; tier-1 rows are emitted in
/// `(point, chunk, b'_row)` order.
///
/// # Panics
///
/// Panics if any of the input slices' lengths disagree with the
/// declared shape parameters: `f_view.num_cols()` must equal
/// `n_b' · split_factor · num_digits_outer`, `num_polys_per_point.len()`
/// must equal `num_points`, `outer_gadget.len()` must equal
/// `num_digits_outer`, `tier1_row_weights.len()` must equal
/// `num_points · split_factor · n_b'`, and `f_row_weights.len()` must
/// equal `num_points · n_F`. The shape parameters are deterministic
/// from `LevelParams`, so a panic here always indicates a caller bug.
pub fn compute_tier1_and_f_contribution_reference<F, E, const D: usize>(
    inputs: &Tier1AndFInputs<'_, F, E, D>,
    num_polys_per_point: &[usize],
) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let lp_split = inputs.split_factor;
    let n_b_prime = inputs.b_prime_view.num_rows();
    let chunk_width = inputs.b_prime_view.num_cols();
    let n_f = inputs.f_view.num_rows();
    let f_width = inputs.f_view.num_cols();
    assert_eq!(
        f_width,
        n_b_prime * lp_split * inputs.num_digits_outer,
        "F width must equal n_b' · split_factor · num_digits_outer"
    );
    assert_eq!(
        num_polys_per_point.len(),
        inputs.num_points,
        "num_polys_per_point length must match num_points"
    );
    assert_eq!(
        inputs.outer_gadget.len(),
        inputs.num_digits_outer,
        "outer gadget vector length must match num_digits_outer"
    );
    assert_eq!(
        inputs.tier1_row_weights.len(),
        inputs.num_points * lp_split * n_b_prime,
        "tier1_row_weights length must equal num_points · split · n_b'"
    );
    assert_eq!(
        inputs.f_row_weights.len(),
        inputs.num_points * n_f,
        "f_row_weights length must equal num_points · n_F"
    );

    // Materialise eq(r_col, *) over the whole M-column space the
    // reference touches. We use index-by-index `eq_eval_at_index`
    // instead of constructing the full `EqPolynomial::evals` table so
    // the test fixture's `full_vec_randomness` length need only be
    // long enough to address the highest M-column the tier1/F rows
    // touch.

    let mut total = E::zero();

    // Per-point flat t-vector base index in the M-layout `flat_t_vector`
    // dimension. Matches the prover's `t_vector_offsets` computation in
    // `compute_r_split_eq`.
    let mut t_vector_offsets = Vec::with_capacity(inputs.num_points);
    let mut acc = 0usize;
    for &k in num_polys_per_point {
        t_vector_offsets.push(acc);
        acc += k;
    }
    let num_t_vectors = inputs.b_physical.num_t_vectors;
    assert_eq!(
        num_t_vectors, acc,
        "b_physical.num_t_vectors must match Σ num_polys_per_point",
    );

    // ----- tier1 rows -----
    //
    // For each (point g, chunk i, b'-row r'):
    //   M-row index in the tiered layout is
    //     `tier1_start + g·split·n_b' + i·n_b' + r'`
    //   M-row cells:
    //     - For each c in [i·chunk_width, (i+1)·chunk_width) (B-physical
    //       col in this point's bundle):
    //         M[row, t_hat_M_col(g, c)] = α-eval(B'[r', c - i·chunk_width])
    //         (where t_hat_M_col follows the standard B-physical →
    //          M-layout bijection used by `get_eq_indices_for_b`).
    //     - For each digit d in 0..num_digits_outer:
    //         M[row, uhat_M_col(g, i, r', d)] = -outer_gadget[d]
    //
    // We iterate every cell and accumulate
    // `eq_tau1[row] · eq_col[col] · M[row, col]` directly.

    for (g, &bundle_size) in num_polys_per_point.iter().enumerate() {
        // B-physical row of the per-point bundle: width
        // `outer_width_per_point = bundle_size · n_a · num_blocks · depth_open`.
        let outer_width_per_point = bundle_size
            * inputs.b_physical.n_a
            * inputs.b_physical.num_blocks
            * inputs.b_physical.depth_open;
        assert_eq!(
            outer_width_per_point,
            lp_split * chunk_width,
            "per-point B-physical width must equal split · chunk_width",
        );

        for chunk_i in 0..lp_split {
            let chunk_start_col = chunk_i * chunk_width;
            for r_prime in 0..n_b_prime {
                let row_flat = g * (lp_split * n_b_prime) + chunk_i * n_b_prime + r_prime;
                let row_weight = inputs.tier1_row_weights[row_flat];

                // (a) B' · t̂_i half.
                for local_c in 0..chunk_width {
                    let b_physical_col = chunk_start_col + local_c;
                    let m_col = b_physical_to_m_col(
                        b_physical_col,
                        g,
                        &t_vector_offsets,
                        bundle_size,
                        &inputs.b_physical,
                        num_t_vectors,
                        inputs.offset_t,
                    );
                    let eq_col_at = akita_algebra::offset_eq::eq_eval_at_index(
                        inputs.full_vec_randomness,
                        m_col,
                    );
                    let alpha_eval = eval_ring_at_pows(
                        &inputs.b_prime_view.row(r_prime)[local_c],
                        inputs.alpha_pows,
                    );
                    total += row_weight * eq_col_at * alpha_eval;
                }

                // (b) -G · û_i half. uhat is stored as
                //     `dig → row → chunk → point` (spec §3 table).
                //     Flat index: g·(split·n_b'·δ_outer)
                //                 + chunk_i·(n_b'·δ_outer)
                //                 + r_prime·δ_outer
                //                 + d
                for d in 0..inputs.num_digits_outer {
                    let uhat_local = g * (lp_split * n_b_prime * inputs.num_digits_outer)
                        + chunk_i * (n_b_prime * inputs.num_digits_outer)
                        + r_prime * inputs.num_digits_outer
                        + d;
                    let m_col = inputs.offset_uhat + uhat_local;
                    let eq_col_at = akita_algebra::offset_eq::eq_eval_at_index(
                        inputs.full_vec_randomness,
                        m_col,
                    );
                    let gadget = inputs.outer_gadget[d];
                    // `-gadget[d]` weight; multiply by base then negate
                    // via subtraction.
                    let contribution = row_weight * eq_col_at;
                    let term = mul_ext_base::<F, E>(contribution, gadget);
                    total -= term;
                }
            }
        }
    }

    // ----- F rows -----
    //
    // For each (point g, F-row r):
    //   M-row index = f_start + g·n_F + r
    //   M-row cells: for each c in 0..f_width:
    //     M[row, uhat_concat_M_col(g, c)] = α-eval(F[r, c])
    //
    // uhat_concat for point g spans uhat cells `[g·f_width,
    // (g+1)·f_width)` under the same `dig → row → chunk → point`
    // layout.

    for g in 0..inputs.num_points {
        for r in 0..n_f {
            let row_flat = g * n_f + r;
            let row_weight = inputs.f_row_weights[row_flat];
            for c in 0..f_width {
                let uhat_local = g * f_width + c;
                let m_col = inputs.offset_uhat + uhat_local;
                let eq_col_at =
                    akita_algebra::offset_eq::eq_eval_at_index(inputs.full_vec_randomness, m_col);
                let alpha_eval = eval_ring_at_pows(&inputs.f_view.row(r)[c], inputs.alpha_pows);
                total += row_weight * eq_col_at * alpha_eval;
            }
        }
    }

    total
}

/// `eq` lift `weight * base` from `E × F` to `E` via repeated additions.
#[inline]
fn mul_ext_base<F, E>(weight: E, base: F) -> E
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    weight.mul_base(base)
}

/// Decode a B-physical column `c ∈ [0, outer_width_per_point)` of point
/// `g` to its M-layout `t̂` column index, mirroring
/// `get_eq_indices_for_b` in `setup_contribution.rs`.
#[inline]
fn b_physical_to_m_col(
    b_physical_col_in_point: usize,
    g: usize,
    t_vector_offsets: &[usize],
    bundle_size: usize,
    bp: &BPhysicalLayout,
    num_t_vectors: usize,
    offset_t: usize,
) -> usize {
    let depth_open = bp.depth_open;
    let n_a = bp.n_a;
    let num_blocks = bp.num_blocks;
    let stride_t = n_a * depth_open;
    let cols_per_poly_t = stride_t * num_blocks;

    let poly_idx = b_physical_col_in_point / cols_per_poly_t;
    assert!(poly_idx < bundle_size, "poly_idx within bundle");
    let inside_poly = b_physical_col_in_point % cols_per_poly_t;
    let digit_idx = inside_poly % depth_open;
    let a_row_idx = (inside_poly / depth_open) % n_a;
    let block_idx = inside_poly / stride_t;
    debug_assert!(block_idx < num_blocks);

    let flat_t_vector = t_vector_offsets[g] + poly_idx;
    let m_layout_high_idx =
        flat_t_vector + num_t_vectors * digit_idx + num_t_vectors * depth_open * a_row_idx;
    offset_t + block_idx + num_blocks * m_layout_high_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_field::Fp64;
    use akita_types::FlatMatrix;

    type F = Fp64<4294967197>;
    const D: usize = 4;

    fn f(x: u64) -> F {
        F::from_u64(x)
    }

    /// Smallest non-trivial tiered shape: f = 2, n_b' = 1, n_F = 1,
    /// δ_outer = 2, num_points = 1, bundle_size = 1, n_a = 1,
    /// num_blocks = 2, depth_open = 2. Outer-width = 4 ⇒ chunk_width = 2.
    /// F-width = 1·2·2 = 4.
    #[test]
    fn tier1_and_f_reference_matches_manual_sum() {
        let split = 2usize;
        let n_b_prime = 1usize;
        let n_f = 1usize;
        let depth_outer = 2usize;
        let num_points = 1usize;
        let bundle_size = 1usize;
        let n_a = 1usize;
        let num_blocks = 2usize;
        let depth_open = 2usize;
        let outer_width_per_point = bundle_size * n_a * num_blocks * depth_open;
        let chunk_width = outer_width_per_point / split;
        assert_eq!(chunk_width, 2);
        let f_width = n_b_prime * split * depth_outer;
        assert_eq!(f_width, 4);

        // B' has shape (n_b'=1, chunk_width=2). Small distinct entries.
        let b_prime_data: Vec<F> = (0..(n_b_prime * chunk_width * D))
            .map(|i| f(2 + i as u64))
            .collect();
        let b_prime_flat = FlatMatrix::<F>::from_flat_data(b_prime_data, D);
        let b_prime_view = b_prime_flat.ring_view::<D>(n_b_prime, chunk_width);

        let f_data: Vec<F> = (0..(n_f * f_width * D))
            .map(|i| f(50 + i as u64 * 3))
            .collect();
        let f_mat = FlatMatrix::<F>::from_flat_data(f_data, D);
        let f_view = f_mat.ring_view::<D>(n_f, f_width);

        // M-column layout (z_first = false):
        //   w_len = num_claims · num_blocks · depth_open = 1·2·2 = 4
        //   t_len = num_t_vectors · n_a · num_blocks · depth_open = 1·1·2·2 = 4
        //   uhat_len = num_points · n_b' · split · δ_outer = 1·1·2·2 = 4
        //   (no zk blinding, no z, no r-tail for the partial sum we test)
        let offset_t = 4usize;
        let offset_uhat = offset_t + 4;
        let total_uhat = num_points * n_b_prime * split * depth_outer;
        assert_eq!(total_uhat, 4);

        let total_used_cols = offset_uhat + total_uhat;
        let bits = total_used_cols.next_power_of_two().trailing_zeros() as usize;
        let full_vec_randomness: Vec<F> = (0..bits).map(|i| f(101 + i as u64)).collect();

        // Outer gadget G = (1, 2^outer_log_basis).
        let outer_log_basis = 2u32;
        let outer_gadget: Vec<F> = (0..depth_outer)
            .map(|d| f(1u64 << (outer_log_basis * d as u32)))
            .collect();

        let alpha = f(7);
        let alpha_pows = akita_algebra::ring::scalar_powers(alpha, D);

        // Row weights: arbitrary distinct values.
        let tier1_row_weights: Vec<F> = (0..(num_points * split * n_b_prime))
            .map(|i| f(11 + i as u64))
            .collect();
        let f_row_weights: Vec<F> = (0..(num_points * n_f)).map(|i| f(31 + i as u64)).collect();

        let inputs = Tier1AndFInputs::<F, F, D> {
            b_prime_view,
            f_view,
            tier1_row_weights: &tier1_row_weights,
            f_row_weights: &f_row_weights,
            alpha_pows: &alpha_pows,
            full_vec_randomness: &full_vec_randomness,
            outer_gadget: &outer_gadget,
            offset_t,
            offset_uhat,
            split_factor: split,
            num_digits_outer: depth_outer,
            b_physical: BPhysicalLayout {
                n_a,
                num_blocks,
                depth_open,
                num_t_vectors: 1,
            },
            num_points,
        };
        let num_polys_per_point = vec![bundle_size];

        let got =
            compute_tier1_and_f_contribution_reference::<F, F, D>(&inputs, &num_polys_per_point);

        // Manual brute-force: materialise eq(r, *) over the full column
        // range we touch, then enumerate every tier1 and F (row, col)
        // cell with hand-coded M-layout indexing and accumulate.
        let n_used = 1usize << bits;
        let eq_full: Vec<F> = (0..n_used)
            .map(|idx| akita_algebra::offset_eq::eq_eval_at_index(&full_vec_randomness, idx))
            .collect();

        let mut expected = F::zero();
        // tier1 rows: one point, two chunks, n_b' = 1 row each.
        for g in 0..num_points {
            for chunk_i in 0..split {
                for r_prime in 0..n_b_prime {
                    let row_flat = g * (split * n_b_prime) + chunk_i * n_b_prime + r_prime;
                    let w = tier1_row_weights[row_flat];

                    // B' · t̂_i half: enumerate B-physical chunk cols.
                    for local_c in 0..chunk_width {
                        let bp = chunk_i * chunk_width + local_c;
                        // Decode for (n_t_vectors = 1, bundle_size = 1):
                        let inside_poly = bp; // poly_idx = 0
                        let stride_t = n_a * depth_open;
                        let digit_idx = inside_poly % depth_open;
                        let a_row_idx = (inside_poly / depth_open) % n_a;
                        let block_idx = inside_poly / stride_t;
                        let flat_t_vector = 0; // g=0, poly=0
                                               // For this single-bundle / single-point fixture
                                               // num_t_vectors = 1; written explicitly so the
                                               // formula matches `get_eq_indices_for_b`.
                        let num_t_vectors_local = 1usize;
                        let high = flat_t_vector
                            + num_t_vectors_local * digit_idx
                            + num_t_vectors_local * depth_open * a_row_idx;
                        let m_col = offset_t + block_idx + num_blocks * high;
                        let alpha_eval = eval_ring_at_pows(
                            &inputs.b_prime_view.row(r_prime)[local_c],
                            &alpha_pows,
                        );
                        expected += w * eq_full[m_col] * alpha_eval;
                    }

                    // -G · û_i half: enumerate digits.
                    for (d, &gadget) in outer_gadget.iter().enumerate() {
                        let uhat_local = g * (split * n_b_prime * depth_outer)
                            + chunk_i * (n_b_prime * depth_outer)
                            + r_prime * depth_outer
                            + d;
                        let m_col = offset_uhat + uhat_local;
                        expected -= w * eq_full[m_col] * gadget;
                    }
                }
            }
        }
        // F rows: one point, n_F = 1 row, f_width = 4 cols.
        for g in 0..num_points {
            for r in 0..n_f {
                let row_flat = g * n_f + r;
                let w = f_row_weights[row_flat];
                for c in 0..f_width {
                    let uhat_local = g * f_width + c;
                    let m_col = offset_uhat + uhat_local;
                    let alpha_eval = eval_ring_at_pows(&inputs.f_view.row(r)[c], &alpha_pows);
                    expected += w * eq_full[m_col] * alpha_eval;
                }
            }
        }

        assert_eq!(
            got, expected,
            "tier1+F reference must match the manual brute-force sum",
        );
    }

    /// Avoid unused-import warnings when adding a non-test build later.
    #[test]
    fn cyclotomic_ring_import_is_used() {
        let _ = CyclotomicRing::<F, D>::zero();
    }
}
