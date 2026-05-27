//! Reference (brute-force) implementation of the tiered root M-row
//! generator.
//!
//! `specs/tiered_commit.md` §3 specifies the tiered M-row layout as
//!
//! ```text
//! consistency (1) | public | D (n_d) | tier1 (f · n_b' · num_points)
//!   | F (n_F · num_points) | A (n_a)
//! ```
//!
//! Each tier-1 row encodes the relation
//! `B' · t̂_i − G · û_i = 0`, single-counted across the setup-matrix
//! and structured halves. Each F row encodes
//! `F · û_concat − u_final = 0`.
//!
//! This module computes only the per-row **`r` quotient** values for
//! the new rows (tier-1 + F). The legacy D, A, consistency, and public
//! rows continue to be computed by
//! [`super::quadratic_equation::compute_r_split_eq`].
//!
//! The `r` quotient for one M-row witnesses the high-half (cyclic vs.
//! negacyclic) difference: if `cyclic_lhs` and `negacyclic_lhs` are the
//! two reductions of the row's polynomial product, then
//! `r = (cyclic_lhs − negacyclic_lhs) / (X^D + 1)`. The standard
//! identity `cyclic - negacyclic = 2 · high_half = (X^D + 1) · r`
//! makes this equivalent to per-coefficient `r[k] = (cyc[k] − neg[k])/2`,
//! which is exactly the legacy `quotient_from_cyclic_and_reduced` helper.
//! For a row whose negacyclic LHS is forced to zero by the relation,
//! `r[k] = cyc[k] / 2`. For an F row whose negacyclic LHS equals the
//! public `u_final`, `r[k] = (cyc[k] − u_final[k]) / 2`.
//!
//! Nothing in the prover calls this module yet — wiring lands in
//! Phase 4c production integration when `compute_r_split_eq` branches
//! on `lp.is_tiered_root()`. `#[allow(dead_code)]` documents that
//! until then. See `specs/tiered_commit.md` §10 for the optimised
//! production loop that shares the `B'` cyclic-product rectangle
//! across all `f` chunks.

#![allow(dead_code)]

use crate::kernels::crt_ntt::NttSlotCache;
use crate::kernels::linear::mat_vec_mul_ntt_single_i8_cyclic;
use akita_algebra::CyclotomicRing;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, HalvingField};

/// Inputs describing the prover-side state needed to emit the
/// tier-1 + F M-rows for *all* opening points in one call.
pub struct Tier1AndFRowsInputs<'a, F: FieldCore, const D: usize> {
    /// NTT cache of the **full** outer B matrix. `B'` is read as the
    /// leading `chunk_width` columns of each B row by passing a
    /// shorter input vector to `mat_vec_mul_ntt_single_i8_cyclic`.
    pub b_ntt_cache: &'a NttSlotCache<D>,
    /// Physical row stride of the B NTT cache (matches the prover
    /// setup's `expanded.seed.max_stride`).
    pub b_max_stride: usize,
    /// SIS rank of `B'`. The kernel will produce this many ring
    /// outputs per chunk.
    pub b_prime_n_rows: usize,
    /// `outer_width / split_factor` — the per-chunk B-physical width.
    pub chunk_width: usize,
    /// NTT cache of the tier-1 F matrix, derived deterministically from
    /// the setup seed via
    /// [`crate::kernels::matrix::derive_tier1_f_matrix_flat`].
    pub f_ntt_cache: &'a NttSlotCache<D>,
    /// Physical row stride of the F NTT cache.
    pub f_max_stride: usize,
    /// SIS rank of `F`.
    pub f_n_rows: usize,
    /// `n_b' · split_factor · num_digits_outer` — F's active column
    /// count.
    pub f_width: usize,
    /// Per-point B-physical `t̂` digit planes. `t_hat_digits[g]` has
    /// length `split_factor · chunk_width`.
    pub t_hat_digits_per_point: &'a [&'a [[i8; D]]],
    /// Per-point `uhat_concat` digit planes. `uhat_concat_digits[g]`
    /// has length `f_width = n_b' · split_factor · num_digits_outer`.
    pub uhat_concat_digits_per_point: &'a [&'a [[i8; D]]],
    /// Per-point public commitment `u_final`. `u_final_per_point[g]`
    /// has length `n_F`.
    pub u_final_per_point: &'a [&'a [CyclotomicRing<F, D>]],
    /// Splitting factor `f` (spec §2).
    pub split_factor: usize,
    /// Outer gadget depth `δ_outer`.
    pub num_digits_outer: usize,
    /// Outer gadget vector `G = (1, 2^{outer_log_basis}, …)`, length
    /// `num_digits_outer`.
    pub outer_gadget: &'a [F],
}

/// Brute-force reference: produces `r` quotients for every tier-1 +
/// F row across all opening points, in the order
///
/// `(tier1[g, chunk_i, b'_row], …, F[g, f_row], …)` per point `g`.
///
/// The returned `Vec` has length
/// `num_points · (split_factor · n_b' + n_F)`. The first
/// `split_factor · n_b'` entries per point are the tier-1 rows in
/// `(chunk_i, b'_row)` order; the next `n_F` are the F rows.
///
/// # Panics
///
/// Panics if the per-point slice lengths are inconsistent with the
/// declared shape parameters.
pub fn compute_tier1_and_f_rows_reference<F, const D: usize>(
    inputs: &Tier1AndFRowsInputs<'_, F, D>,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HalvingField + FromPrimitiveInt,
{
    let num_points = inputs.t_hat_digits_per_point.len();
    assert_eq!(
        num_points,
        inputs.uhat_concat_digits_per_point.len(),
        "t̂ and ûhat_concat must agree on num_points",
    );
    assert_eq!(
        num_points,
        inputs.u_final_per_point.len(),
        "u_final must agree on num_points",
    );
    assert_eq!(
        inputs.outer_gadget.len(),
        inputs.num_digits_outer,
        "outer gadget vector length must match num_digits_outer",
    );

    let n_b_prime = inputs.b_prime_n_rows;
    let split = inputs.split_factor;
    let depth_outer = inputs.num_digits_outer;
    let chunk_width = inputs.chunk_width;
    let f_width = inputs.f_width;
    let n_f = inputs.f_n_rows;
    assert_eq!(
        f_width,
        n_b_prime * split * depth_outer,
        "f_width must equal n_b' · split · num_digits_outer",
    );

    let total_rows_per_point = split * n_b_prime + n_f;
    let mut out = Vec::with_capacity(num_points * total_rows_per_point);

    for (g, t_hat_g) in inputs.t_hat_digits_per_point.iter().enumerate() {
        assert_eq!(
            t_hat_g.len(),
            split * chunk_width,
            "t̂ per-point digit count must equal split · chunk_width",
        );
        let uhat_g = inputs.uhat_concat_digits_per_point[g];
        assert_eq!(uhat_g.len(), f_width, "uhat_concat per-point length");
        let u_final_g = inputs.u_final_per_point[g];
        assert_eq!(u_final_g.len(), n_f, "u_final per-point length");

        // ----- tier-1 rows -----
        // Per-chunk cyclic mat-vec mul `B' · t̂_chunk_i`. The full B NTT
        // cache is reused; passing a short `chunk` slice automatically
        // restricts the kernel to B's leading `chunk_width` columns,
        // which is exactly B'.
        for chunk_i in 0..split {
            let chunk_start = chunk_i * chunk_width;
            let chunk = &t_hat_g[chunk_start..chunk_start + chunk_width];
            let b_prime_t_i_cyclic = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(
                inputs.b_ntt_cache,
                n_b_prime,
                inputs.b_max_stride,
                chunk,
            );
            for (r_prime, b_prime_t_i_cyclic_row) in b_prime_t_i_cyclic.iter().enumerate() {
                // G · ûhat_i[r'] = Σ_d gadget[d] · lift(ûhat_concat[g][chunk_i, r', d])
                let mut g_uhat = CyclotomicRing::<F, D>::zero();
                for (d, &gadget) in inputs.outer_gadget.iter().enumerate() {
                    let uhat_idx = chunk_i * (n_b_prime * depth_outer) + r_prime * depth_outer + d;
                    let digit_plane = &uhat_g[uhat_idx];
                    let lifted = lift_i8_plane_to_ring::<F, D>(digit_plane);
                    // Scalar multiplication: scale every coefficient by
                    // `gadget`. `CyclotomicRing<F, D>` doesn't impl
                    // `Mul<F>` directly, so apply the gadget in-place
                    // before accumulating.
                    let mut scaled_coeffs = *lifted.coefficients();
                    for c in scaled_coeffs.iter_mut() {
                        *c *= gadget;
                    }
                    g_uhat += CyclotomicRing::<F, D>::from_coefficients(scaled_coeffs);
                }
                let r = quotient_half_diff(b_prime_t_i_cyclic_row, &g_uhat);
                out.push(r);
            }
        }

        // ----- F rows -----
        let f_uhat_cyclic = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(
            inputs.f_ntt_cache,
            n_f,
            inputs.f_max_stride,
            uhat_g,
        );
        for r in 0..n_f {
            let q = quotient_half_diff(&f_uhat_cyclic[r], &u_final_g[r]);
            out.push(q);
        }
    }

    out
}

/// `(a - b) / 2` coefficient-wise — the legacy
/// `quotient_from_cyclic_and_reduced` formula, repeated here so this
/// reference module doesn't depend on a private prover helper.
#[inline]
fn quotient_half_diff<F: FieldCore + HalvingField, const D: usize>(
    a: &CyclotomicRing<F, D>,
    b: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let ac = a.coefficients();
    let bc = b.coefficients();
    CyclotomicRing::from_coefficients(std::array::from_fn(|k| (ac[k] - bc[k]).half()))
}

/// Lift a signed-i8 digit plane to a `CyclotomicRing<F, D>` by mapping
/// each i8 to its `F` representative.
#[inline]
fn lift_i8_plane_to_ring<F, const D: usize>(plane: &[i8; D]) -> CyclotomicRing<F, D>
where
    F: FieldCore + FromPrimitiveInt,
{
    CyclotomicRing::from_coefficients(std::array::from_fn(|k| F::from_i8(plane[k])))
}

// Tier-1 row reference tests removed during the origin/main merge: they
// targeted the pre-#105 `AkitaProverSetup`-based prover surface. Add
// back when the tier prover code is fully migrated onto the backend
// boundary.
