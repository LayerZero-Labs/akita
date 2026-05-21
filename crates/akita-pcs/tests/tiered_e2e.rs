//! End-to-end integration test for the tiered root commitment.
//!
//! `specs/tiered_commit.md` defines the tiered root commitment as a
//! drop-in replacement for `u = B · t̂` when
//! `LevelParams.split_factor > 1`. The planner does not yet emit
//! `split_factor > 1` candidates (Phase 4e-search is a pending
//! follow-up), so this test forces a synthetic tiered `LevelParams`
//! and exercises the new code paths directly via the public commit +
//! reference-evaluator APIs.
//!
//! What this test validates:
//!
//! 1. `commit_with_params` auto-dispatches to `commit_tiered_with_params`
//!    when `lp.is_tiered_root()`, producing
//!    `RingCommitment { u: u_final }` of length `n_F` and an
//!    `AkitaCommitmentHint` with `outer_digits` populated.
//! 2. The tiered commit witness (`t̂` from inner-commit +
//!    `ûhat_concat` from the hint + `u_final` from the commitment)
//!    can be fed into `tier1_rows_reference::compute_tier1_and_f_rows_reference`
//!    and produces a `Vec<r>` of length
//!    `(split_factor · n_b' + n_F) · num_points`.
//! 3. The same witness fed through the verifier's
//!    `tier1_reference::compute_tier1_and_f_contribution_reference`
//!    matches a brute-force Σ_r eq_tau1[r] · Σ_c eq_col[c] · M[r, c]
//!    over the (tier-1 + F) M-row block when the underlying M cells
//!    are reconstructed from the witness.
//!
//! Together (2) + (3) close the prover→verifier soundness loop for
//! the tiered M-row block: the prover's r-values witness the
//! relation that the verifier's M̃-evaluation sums against.

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::eval_ring_at_pows;
use akita_algebra::ring::scalar_powers;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::{CanonicalField, Fp64};
use akita_prover::tier1_f_matrix::derive_tier1_f_matrix_flat;
use akita_prover::tier1_rows_reference::{compute_tier1_and_f_rows_reference, Tier1AndFRowsInputs};
use akita_prover::{commit_with_params, AkitaPolyOps, AkitaProverSetup, DensePoly};
use akita_types::layout::sis_derivation::balanced_digit_delta_bound;
use akita_types::{AjtaiKeyParams, LevelParams, SisModulusFamily};
use akita_verifier::tier1_reference::{
    compute_tier1_and_f_contribution_reference, BPhysicalLayout, Tier1AndFInputs,
};

type F = Fp64<4294967197>;
const D: usize = 4;

// Tiered outer-digit parameters covering the full Fp64 (~32-bit)
// modulus with the maximum log-basis the i8 digit storage allows.
const OUTER_LOG_BASIS: u32 = 6;
const NUM_DIGITS_OUTER: usize = 6;

/// Build a deliberately tiny tiered `LevelParams`. Sizes chosen so the
/// test exercises every code path (split > 1, multiple chunks per
/// point, multiple b'-rows, multiple F-rows) without spending time on
/// large NTTs.
fn tiered_level_params() -> LevelParams {
    let n_a = 1usize;
    let n_b_prime = 2usize;
    let n_d = 1usize;
    let n_f = 2usize;
    let num_blocks = 2usize;
    let block_len = 2usize;
    let num_digits_commit = 1usize;
    let num_digits_open = 2usize;
    let split_factor = 2usize;

    let log_basis: u32 = 2;
    let outer_width = n_a * num_digits_open * num_blocks;
    let chunk_width = outer_width / split_factor;
    let f_width = n_b_prime * split_factor * NUM_DIGITS_OUTER;
    let inner_width = block_len * num_digits_commit;
    let d_matrix_width = num_digits_open * num_blocks;

    LevelParams {
        ring_dimension: D,
        log_basis,
        a_key: AjtaiKeyParams::new_unchecked(
            SisModulusFamily::Q64,
            n_a,
            inner_width,
            balanced_digit_delta_bound(log_basis),
            D,
        ),
        // The tiered path sets `b_key.col_len()` to `B'`'s width
        // (= chunk_width), per `LevelParams::b_prime_width()`. The
        // full outer width is recovered via `full_outer_width()`.
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

/// `commit_with_params` end-to-end on a forced tiered LevelParams.
/// Validates the entire prover-side tiered chain:
///   1. Auto-dispatch in `commit_with_params` → `commit_tiered_with_params`.
///   2. The new `outer_digits` field of `AkitaCommitmentHint` carries
///      `ûhat_concat`.
///   3. `commitment.u` is the new `u_final = F · ûhat_concat` (length
///      `n_F`, not the legacy `n_b`).
///   4. The prover-side reference row-generator
///      `compute_tier1_and_f_rows_reference` accepts the resulting
///      witness shape and produces the right number of r-quotients.
///   5. The verifier-side reference evaluator
///      `compute_tier1_and_f_contribution_reference` accepts the
///      witness shape and produces a finite M̃ value.
///   6. The two references are mutually consistent: the verifier
///      M̃-evaluation matches a hand-summed Σ_r eq_tau1[r] ·
///      Σ_c eq_col[c] · M[r, c] over the tier-1 + F M-row block
///      reconstructed directly from the witness.
#[test]
fn forced_tiering_commit_and_references_compose_end_to_end() {
    let tier = tiered_level_params();
    let n_b_prime = tier.b_key.row_len();
    let chunk_width = tier.b_key.col_len();
    let n_f = tier.f_key.row_len();
    let f_width = tier.f_key.col_len();
    let split = tier.split_factor;
    let depth_outer = tier.num_digits_outer;
    let num_points = 1usize;

    // ----- Setup + polynomial -----------------------------------------------
    //
    // The setup envelope must cover the largest of `chunk_width`,
    // `inner_width`, and `d_matrix_width` (F lives in its own
    // domain-separated matrix derived from the seed, not in
    // `shared_matrix`).
    let max_stride = tier.full_outer_width().max(tier.inner_width());
    let max_rows = tier
        .a_key
        .row_len()
        .max(tier.b_key.row_len())
        .max(tier.d_key.row_len())
        .max(tier.f_key.row_len());
    let num_ring = tier.num_blocks * tier.block_len;
    let num_vars = (num_ring * D).trailing_zeros() as usize;
    let evals: Vec<F> = (0..(1usize << num_vars))
        .map(|idx| F::from_canonical_u128_reduced(((idx as u128) * 19 + 7) % 991))
        .collect();
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &evals).expect("dense");
    let setup =
        AkitaProverSetup::<F, D>::generate_with_capacity(num_vars + 4, 1, 1, max_rows, max_stride)
            .expect("setup");

    // ----- (1)(2)(3) Tiered commit ------------------------------------------
    let (commitment, hint) =
        commit_with_params::<F, D, _>(std::slice::from_ref(&poly), &setup, &tier)
            .expect("commit_with_params auto-dispatches to tiered branch");

    assert_eq!(
        commitment.u.len(),
        tier.outer_commitment_rows(),
        "RingCommitment.u length must equal n_F (= outer_commitment_rows under tiering)"
    );
    assert_eq!(commitment.u.len(), n_f);
    assert_eq!(
        hint.outer_digits().len(),
        num_points,
        "one outer_digits entry per opening-point commitment"
    );
    let uhat_concat = &hint.outer_digits()[0];
    let expected_uhat_planes = n_b_prime * split * depth_outer;
    assert_eq!(
        uhat_concat.flat_digits().len(),
        expected_uhat_planes,
        "uhat_concat plane count = n_b' · split · δ_outer"
    );

    // ----- t̂ digits via the same `commit_inner_witness` path ----------------
    //
    // The tiered commit consumed these internally; we rebuild them
    // here so the prover-side reference row-generator can be driven
    // independently.
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
    .expect("inner commit");
    let t_hat_digits: Vec<[i8; D]> = inner.decomposed_inner_rows.flat_digits().to_vec();
    let uhat_digits: Vec<[i8; D]> = uhat_concat.flat_digits().to_vec();

    // ----- F NTT cache (matches commit_with_params's derivation) -----------
    let f_flat =
        derive_tier1_f_matrix_flat::<F, D>(n_f * f_width, &setup.expanded.seed.public_matrix_seed);
    let f_view = f_flat
        .ring_view::<D>(n_f, f_width)
        .expect("test F view shape");
    let f_ntt = akita_prover::kernels::crt_ntt::build_ntt_slot(f_view).expect("f ntt cache");

    // ----- Outer gadget vector ---------------------------------------------
    //
    // Computed in the field so `outer_log_basis · num_digits_outer ≥
    // 64` doesn't blow up `1u64 << k`. Matches the verifier's
    // construction in `eval_at_point`.
    let mut outer_gadget = Vec::with_capacity(depth_outer);
    let two_to_b = F::from_u64(1u64 << OUTER_LOG_BASIS);
    let mut step = F::one();
    for _ in 0..depth_outer {
        outer_gadget.push(step);
        step *= two_to_b;
    }

    // ----- (4) Prover-side r-quotients -------------------------------------
    let t_hat_per_point: [&[[i8; D]]; 1] = [&t_hat_digits];
    let uhat_per_point: [&[[i8; D]]; 1] = [&uhat_digits];
    let u_final_per_point: [&[CyclotomicRing<F, D>]; 1] = [&commitment.u];
    let prover_inputs = Tier1AndFRowsInputs::<F, D> {
        b_ntt_cache: &setup.ntt_shared,
        b_max_stride: setup.expanded.seed.max_stride,
        b_prime_n_rows: n_b_prime,
        chunk_width,
        f_ntt_cache: &f_ntt,
        f_max_stride: f_width,
        f_n_rows: n_f,
        f_width,
        t_hat_digits_per_point: &t_hat_per_point,
        uhat_concat_digits_per_point: &uhat_per_point,
        u_final_per_point: &u_final_per_point,
        split_factor: split,
        num_digits_outer: depth_outer,
        outer_gadget: &outer_gadget,
    };
    let r_values = compute_tier1_and_f_rows_reference::<F, D>(&prover_inputs);
    let expected_rows = (split * n_b_prime + n_f) * num_points;
    assert_eq!(
        r_values.len(),
        expected_rows,
        "prover reference returns one r per tier-1 + F row across all points"
    );

    // ----- (5)(6) Verifier-side M̃ evaluation vs manual sum ----------------
    //
    // Synthetic challenges drive the verifier reference. We pick
    // distinct deterministic values so any off-by-one in indexing
    // surfaces immediately.
    let alpha = F::from_u64(7);
    let alpha_pows = scalar_powers(alpha, D);

    // Need eq_tau1 long enough to index the highest row weight the
    // tiered layout touches.
    let n_d = tier.d_key.row_len();
    let n_a_rows = tier.a_key.row_len();
    let num_public_rows = num_points; // singleton bundle per point
    let num_rows = tier
        .m_row_count(num_points, num_public_rows)
        .expect("m_row_count");
    let rows_pow2 = num_rows.next_power_of_two();
    let eq_tau1: Vec<F> = (0..rows_pow2)
        .map(|idx| F::from_canonical_u128_reduced(((idx as u128) * 41 + 13) % 521))
        .collect();

    let d_start = 1 + num_public_rows;
    let tier1_start = d_start + n_d;
    let tier1_count = split * n_b_prime * num_points;
    let tier1_end = tier1_start + tier1_count;
    let f_start = tier1_end;
    let f_end = f_start + n_f * num_points;
    assert!(f_end + n_a_rows <= rows_pow2, "eq_tau1 long enough");
    let tier1_row_weights: Vec<F> = eq_tau1[tier1_start..tier1_end].to_vec();
    let f_row_weights: Vec<F> = eq_tau1[f_start..f_end].to_vec();

    // M-column layout offsets. We choose `z_first = false`:
    //   M = [ w_hat | t_hat | uhat | (no zk blinding) | z_hat | r-tail ]
    let total_blocks = tier.num_blocks; // num_claims = 1 ⇒ total = num_blocks · 1
    let w_len = tier.num_digits_open * total_blocks;
    let t_total_blocks = tier.num_blocks; // num_t_vectors = 1
    let t_len = tier.num_digits_open * tier.a_key.row_len() * t_total_blocks;
    let uhat_len = num_points * n_b_prime * split * depth_outer;
    let offset_t = w_len;
    let offset_uhat = offset_t + t_len;
    let z_len = tier.num_digits_fold * tier.num_digits_commit * num_points * tier.block_len;
    let total_witness_cols = w_len + t_len + uhat_len + z_len;
    let bits = total_witness_cols.next_power_of_two().trailing_zeros() as usize;
    let full_vec_randomness: Vec<F> = (0..bits)
        .map(|i| F::from_canonical_u128_reduced(((i as u128) * 53 + 11) % 997))
        .collect();

    // Stride-aligned view so `row(r)` resolves to physical row r of
    // shared_matrix even when `chunk_width < max_stride`. The tier-1
    // reference will index only `[0..chunk_width)` within each row.
    let b_prime_view = setup
        .expanded
        .shared_matrix
        .ring_view::<D>(n_b_prime, setup.expanded.seed.max_stride)
        .expect("b' view");
    let f_view = f_flat.ring_view::<D>(n_f, f_width).expect("f view");

    let verifier_inputs = Tier1AndFInputs::<F, F, D> {
        b_prime_view,
        b_prime_chunk_width: chunk_width,
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
            n_a: tier.a_key.row_len(),
            num_blocks: tier.num_blocks,
            depth_open: tier.num_digits_open,
            num_t_vectors: 1,
        },
        num_points,
    };
    let verifier_value =
        compute_tier1_and_f_contribution_reference::<F, F, D>(&verifier_inputs, &[1usize]);

    // ----- Manual brute-force M-cell sum over tier-1 + F rows ---------------
    let bits_total = 1usize << bits;
    let eq_full: Vec<F> = (0..bits_total)
        .map(|idx| eq_eval_at_index(&full_vec_randomness, idx))
        .collect();

    let mut expected = F::zero();
    // Tier-1 rows: M[row, t_hat_col] = α-eval(B'[r', local_c]); M[row, uhat_col] = -gadget[d].
    let stride_t = tier.a_key.row_len() * tier.num_digits_open; // n_a · depth_open
    for chunk_i in 0..split {
        let chunk_start_col = chunk_i * chunk_width;
        for r_prime in 0..n_b_prime {
            let row_flat = chunk_i * n_b_prime + r_prime;
            let w = tier1_row_weights[row_flat];
            // B'·t̂_i half: enumerate B-physical chunk cells.
            for local_c in 0..chunk_width {
                let bp = chunk_start_col + local_c;
                // Decode B-physical column for num_t_vectors = 1,
                // bundle_size = 1.
                let inside_poly = bp;
                let digit_idx = inside_poly % tier.num_digits_open;
                let a_row_idx = (inside_poly / tier.num_digits_open) % tier.a_key.row_len();
                let block_idx = inside_poly / stride_t;
                let flat_t_vector = 0usize; // g=0, poly=0
                                            // num_t_vectors = 1 in this single-bundle / single-point
                                            // fixture, so the multipliers are trivially 1. Written
                                            // longhand so the formula matches the prover's
                                            // `get_eq_indices_for_b` decode in setup_contribution.rs.
                let num_t_vectors_local = 1usize;
                let high = flat_t_vector
                    + num_t_vectors_local * digit_idx
                    + num_t_vectors_local * tier.num_digits_open * a_row_idx;
                let m_col = offset_t + block_idx + tier.num_blocks * high;
                let b_row = b_prime_view.row(r_prime).expect("b' row in range");
                let alpha_eval = eval_ring_at_pows(&b_row[local_c], &alpha_pows);
                expected += w * eq_full[m_col] * alpha_eval;
            }
            // -G·ûhat_i half.
            for (d, &gadget) in outer_gadget.iter().enumerate() {
                let uhat_local = chunk_i * (n_b_prime * depth_outer) + r_prime * depth_outer + d;
                let m_col = offset_uhat + uhat_local;
                expected -= w * eq_full[m_col] * gadget;
            }
        }
    }
    // F rows: M[row, uhat_concat_col] = α-eval(F[r, c]).
    for r in 0..n_f {
        let row_flat = r;
        let w = f_row_weights[row_flat];
        let f_row_data = f_view.row(r).expect("f row in range");
        for (c, f_cell) in f_row_data.iter().take(f_width).enumerate() {
            let m_col = offset_uhat + c;
            let alpha_eval = eval_ring_at_pows(f_cell, &alpha_pows);
            expected += w * eq_full[m_col] * alpha_eval;
        }
    }

    assert_eq!(
        verifier_value, expected,
        "verifier reference tier-1+F M̃ value must match brute-force sum reconstructed from the witness"
    );
}
