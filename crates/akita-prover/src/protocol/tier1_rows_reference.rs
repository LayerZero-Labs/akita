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

#[cfg(all(test, not(feature = "zk")))]
mod tests {
    use super::*;
    use crate::api::commit_tiered_with_params;
    use crate::backend::DensePoly;
    use crate::kernels::crt_ntt::build_ntt_slot;
    use crate::kernels::matrix::derive_tier1_f_matrix_flat;
    use crate::{AkitaPolyOps, AkitaProverSetup};
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp64;
    use akita_types::layout::sis_derivation::balanced_digit_delta_bound;
    use akita_types::{AjtaiKeyParams, LevelParams as LP, SisModulusFamily};

    type F = Fp64<4294967197>;
    const D: usize = 4;

    /// Q64 full-field balanced i8 depth for outer basis 2^6.
    const OUTER_LOG_BASIS: u32 = 6;
    const NUM_DIGITS_OUTER: usize = 6;

    #[allow(clippy::too_many_arguments)]
    fn tiered_level_params(
        n_a: usize,
        n_b_prime: usize,
        n_d: usize,
        n_f: usize,
        num_blocks: usize,
        block_len: usize,
        num_digits_commit: usize,
        num_digits_open: usize,
        split_factor: usize,
    ) -> LP {
        let log_basis: u32 = 2;
        let chunk_width = n_a * num_digits_open * num_blocks / split_factor;
        let f_width = n_b_prime * split_factor * NUM_DIGITS_OUTER;
        let inner_width = block_len * num_digits_commit;
        let d_matrix_width = num_digits_open * num_blocks;
        LP {
            ring_dimension: D,
            log_basis,
            a_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_a,
                inner_width,
                balanced_digit_delta_bound(log_basis),
                D,
            ),
            b_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_b_prime,
                chunk_width,
                balanced_digit_delta_bound(log_basis),
                D,
            ),
            d_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_d,
                d_matrix_width,
                balanced_digit_delta_bound(log_basis),
                D,
            ),
            num_blocks,
            block_len,
            m_vars: block_len.trailing_zeros() as usize,
            r_vars: num_blocks.trailing_zeros() as usize,
            stage1_config: SparseChallengeConfig::Uniform {
                weight: 2,
                nonzero_coeffs: vec![-1, 1],
            },
            num_digits_commit,
            num_digits_open,
            num_digits_fold: 1,
            split_factor,
            outer_log_basis: OUTER_LOG_BASIS,
            num_digits_outer: NUM_DIGITS_OUTER,
            f_key: AjtaiKeyParams::new_unchecked(
                SisModulusFamily::Q64,
                n_f,
                f_width,
                balanced_digit_delta_bound(OUTER_LOG_BASIS),
                D,
            ),
        }
    }

    /// Outer gadget vector matching the standard balanced-i8 basis.
    fn outer_gadget(depth: usize, log_basis: u32) -> Vec<F> {
        (0..depth)
            .map(|d| F::from_canonical_u128_reduced(1u128 << (log_basis as u128 * d as u128)))
            .collect()
    }

    /// End-to-end consistency: the reference impl's r-quotients
    /// satisfy `2·r[k] = cyclic_lhs[k] − negacyclic_lhs[k]` for every
    /// tier-1 and F row, where the negacyclic LHS is forced to zero by
    /// the witness (tier-1) or equals `u_final` (F).
    ///
    /// We verify this by running `commit_tiered_with_params` (which
    /// produces a witness that satisfies all tiered relations
    /// negacyclically), then independently computing the cyclic LHS
    /// inside the test, and asserting the reference impl returns the
    /// expected quotient.
    #[test]
    fn tier1_and_f_rows_reference_matches_quotient_definition() {
        let tier = tiered_level_params(1, 2, 1, 2, 2, 2, 1, 2, 2);
        let n_f = tier.f_key.row_len();
        let f_width = tier.f_key.col_len();
        let chunk_width = tier.b_key.col_len();
        let n_b_prime = tier.b_key.row_len();
        let split = tier.split_factor;
        let depth_outer = tier.num_digits_outer;

        let num_ring = tier.num_blocks * tier.block_len;
        let num_vars = (num_ring * D).trailing_zeros() as usize;
        let evals: Vec<F> = (0..(1usize << num_vars))
            .map(|idx| F::from_canonical_u128_reduced(((idx as u128) * 17 + 3) % 991))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).expect("dense");

        let max_stride = tier.full_outer_width().max(tier.inner_width());
        let max_rows = tier
            .a_key
            .row_len()
            .max(tier.b_key.row_len())
            .max(tier.d_key.row_len())
            .max(tier.f_key.row_len());
        let setup = AkitaProverSetup::<F, D>::generate_with_capacity(
            num_vars + 4,
            1,
            1,
            max_rows,
            max_stride,
        )
        .expect("setup");

        // Build F NTT cache from the same domain-separated derivation
        // used by `commit_with_params`.
        let f_flat = derive_tier1_f_matrix_flat::<F, D>(
            n_f * f_width,
            &setup.expanded.seed.public_matrix_seed,
        );
        let f_view = f_flat.ring_view::<D>(n_f, f_width).expect("test F view");
        let f_cache = build_ntt_slot(f_view).expect("f cache");

        // Run the tiered commit to get the witness.
        let (commitment, hint) = commit_tiered_with_params::<F, D, _>(
            std::slice::from_ref(&poly),
            &setup,
            &tier,
            &f_cache,
            f_width,
        )
        .expect("tiered commit");

        // Reconstruct t̂ digits for this single-point bundle via the
        // backend's inner-commit step.
        let inner = <DensePoly<F, D> as AkitaPolyOps<F, D>>::commit_inner_witness(
            &poly,
            &setup.expanded.shared_matrix,
            &setup.ntt_shared,
            tier.a_key.row_len(),
            tier.block_len,
            tier.num_digits_commit,
            tier.num_digits_open,
            tier.log_basis,
            setup.expanded.seed.max_stride,
        )
        .expect("inner");
        let t_hat_digits: Vec<[i8; D]> = inner.decomposed_inner_rows.flat_digits().to_vec();
        let uhat_concat = &hint.outer_digits()[0];
        let uhat_digits: Vec<[i8; D]> = uhat_concat.flat_digits().to_vec();

        let outer_gadget_vec = outer_gadget(depth_outer, OUTER_LOG_BASIS);

        let t_hat_refs: [&[[i8; D]]; 1] = [&t_hat_digits];
        let uhat_refs: [&[[i8; D]]; 1] = [&uhat_digits];
        let u_final_refs: [&[CyclotomicRing<F, D>]; 1] = [&commitment.u];

        let inputs = Tier1AndFRowsInputs::<F, D> {
            b_ntt_cache: &setup.ntt_shared,
            b_max_stride: setup.expanded.seed.max_stride,
            b_prime_n_rows: n_b_prime,
            chunk_width,
            f_ntt_cache: &f_cache,
            f_max_stride: f_width,
            f_n_rows: n_f,
            f_width,
            t_hat_digits_per_point: &t_hat_refs,
            uhat_concat_digits_per_point: &uhat_refs,
            u_final_per_point: &u_final_refs,
            split_factor: split,
            num_digits_outer: depth_outer,
            outer_gadget: &outer_gadget_vec,
        };

        let r_values = compute_tier1_and_f_rows_reference::<F, D>(&inputs);

        let total_per_point = split * n_b_prime + n_f;
        assert_eq!(r_values.len(), total_per_point);

        // Validate each r against the definition `2·r = cyclic − reduced`,
        // where `reduced` is the negacyclic LHS we expect from the
        // relation (0 for tier-1, u_final for F).
        for chunk_i in 0..split {
            let chunk_start = chunk_i * chunk_width;
            let chunk = &t_hat_digits[chunk_start..chunk_start + chunk_width];
            let b_prime_t_i_cyclic = mat_vec_mul_ntt_single_i8_cyclic::<F, D>(
                &setup.ntt_shared,
                n_b_prime,
                setup.expanded.seed.max_stride,
                chunk,
            );
            for r_prime in 0..n_b_prime {
                let row_idx = chunk_i * n_b_prime + r_prime;
                let r_got = &r_values[row_idx];

                // Independent G·uhat reconstruction (same scalar
                // multiplication trick as the production reference).
                let mut g_uhat = CyclotomicRing::<F, D>::zero();
                for (d, &gadget) in outer_gadget_vec.iter().enumerate() {
                    let uhat_idx = chunk_i * (n_b_prime * depth_outer) + r_prime * depth_outer + d;
                    let lifted = lift_i8_plane_to_ring::<F, D>(&uhat_digits[uhat_idx]);
                    let mut scaled_coeffs = *lifted.coefficients();
                    for c in scaled_coeffs.iter_mut() {
                        *c *= gadget;
                    }
                    g_uhat += CyclotomicRing::<F, D>::from_coefficients(scaled_coeffs);
                }

                // Expected r = (cyclic − G·uhat) / 2 coefficient-wise.
                let cyc = b_prime_t_i_cyclic[r_prime].coefficients();
                let red = g_uhat.coefficients();
                let expected =
                    CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|k| {
                        (cyc[k] - red[k]).half()
                    }));

                assert_eq!(
                    r_got, &expected,
                    "tier-1 row r value matches (cyclic − G·uhat)/2 for chunk={chunk_i}, r'={r_prime}",
                );

                // Also confirm the witness actually satisfies the
                // relation negacyclically (sanity check on the test
                // fixture, independent of the function under test).
                use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
                let b_prime_t_i_neg = mat_vec_mul_ntt_single_i8::<F, D>(
                    &setup.ntt_shared,
                    n_b_prime,
                    setup.expanded.seed.max_stride,
                    chunk,
                );
                assert_eq!(
                    b_prime_t_i_neg[r_prime], g_uhat,
                    "witness satisfies negacyclic tier-1 relation",
                );
            }
        }

        // F rows.
        let f_uhat_cyclic =
            mat_vec_mul_ntt_single_i8_cyclic::<F, D>(&f_cache, n_f, f_width, &uhat_digits);
        for r in 0..n_f {
            let row_idx = split * n_b_prime + r;
            let r_got = &r_values[row_idx];
            let cyc = f_uhat_cyclic[r].coefficients();
            let red = commitment.u[r].coefficients();
            let expected = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|k| {
                (cyc[k] - red[k]).half()
            }));
            assert_eq!(
                r_got, &expected,
                "F row r value matches (cyclic − u_final)/2 for r={r}",
            );

            // Sanity: F·uhat negacyclic equals u_final.
            use crate::kernels::linear::mat_vec_mul_ntt_single_i8;
            let f_uhat_neg =
                mat_vec_mul_ntt_single_i8::<F, D>(&f_cache, n_f, f_width, &uhat_digits);
            assert_eq!(
                f_uhat_neg[r], commitment.u[r],
                "F relation holds negacyclically"
            );
        }
    }
}
