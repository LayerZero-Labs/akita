//! Independent materialized `<S_{<=N}, omega_S>` reference oracle (test-only).
//!
//! This is the lane-02A correctness ground truth from
//! `specs/setup-layout-repack.md` ("Materialized Inner-Product Oracle"). It
//! recomputes the role-weight vector `omega_bar_S(lambda)` directly from the
//! D/B/A role pullbacks against the *full* `eq(r, .)` evaluation, sharing none
//! of the peeled-eq / segment machinery in `evaluator.rs`. The equivalence
//! tests therefore exercise the production weight math (column-order pullbacks,
//! peeled-eq factorisation, dense-vs-pow2 Z) rather than only the alpha
//! contraction, and later lanes (03A succinct evaluator, 03B product sumcheck)
//! get an oracle that is genuinely independent of the path under test.
//!
//! `alpha` lives on the weight side (`omega_S(lambda, y) = omega_bar_S(lambda)
//! * alpha^y`), never in committed setup, matching the spec.

use akita_algebra::offset_eq::eq_eval_at_index;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBase};
use akita_types::AkitaExpandedSetup;

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

/// Materialize `omega_bar_S(lambda)` from the explicit D/B/A role pullbacks.
///
/// Overlapping A/B/D raw setup coordinates accumulate by addition, as required
/// by the spec. The pullback index maps are the root digit-fast views
/// `j_M^D / j_M^B / x_Z` and the A/J adjoint `eta_Z`.
pub(crate) fn materialized_bar_omega<F, E>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    fold_gadget: &[F],
) -> Vec<E>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    let offset_w = prepared.witness_segment_layout.offset_w;
    let offset_t = prepared.witness_segment_layout.offset_t;
    let offset_z = prepared.witness_segment_layout.offset_z;

    let depth_open = prepared.depth_open;
    let depth_commit = prepared.depth_commit;
    let depth_fold = prepared.depth_fold;
    let num_blocks = prepared.num_blocks;
    let num_claims = prepared.num_claims;
    let num_points = prepared.num_points;
    let n_a = prepared.n_a;
    let n_b = prepared.n_b;
    let n_d_active = prepared.n_d_active();
    let block_len = prepared.block_len;
    let num_t_vectors = prepared.num_t_vectors;

    let n_cols_w = num_claims * num_blocks * depth_open; // W_D
    let stride_t = n_a * depth_open;
    let cols_per_poly_t = stride_t * num_blocks;
    let max_group = prepared
        .num_polys_per_commitment_group
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let n_cols_t = max_group * cols_per_poly_t; // W_B
    let z_range = block_len * depth_commit; // W_A
    let required = (n_d_active * n_cols_w)
        .max(n_b * n_cols_t)
        .max(n_a * z_range);

    let d_start = 1 + prepared.num_public_rows;
    let b_start = d_start + n_d_active;
    let a_start = b_start + n_b * num_points;

    let eq = |idx: usize| eq_eval_at_index(full_vec_randomness, idx);

    let mut bar_omega = vec![E::zero(); required];

    // D: omega_bar(iota_D) += eq(tau_1, D_row) * eq(r, offset_w + j_M^D)
    for d_row in 0..n_d_active {
        let row_weight = prepared.eq_tau1[d_start + d_row];
        for claim in 0..num_claims {
            for block in 0..num_blocks {
                for digit in 0..depth_open {
                    let phys = (claim * num_blocks + block) * depth_open + digit;
                    let lambda = d_row * n_cols_w + phys;
                    let m_idx = block + num_blocks * (claim + num_claims * digit);
                    bar_omega[lambda] += row_weight * eq(offset_w + m_idx);
                }
            }
        }
    }

    // B: per (point, poly) group, omega_bar(iota_B) += eq(tau_1, B_row@point)
    //    * eq(r, offset_t + j_M^B). `flat_t_vector` is the global t-vector slot.
    let mut flat_t_vector = 0usize;
    for (point, &group_poly_count) in prepared.num_polys_per_commitment_group.iter().enumerate() {
        for poly_idx in 0..group_poly_count {
            for b_row in 0..n_b {
                let row_weight = prepared.eq_tau1[b_start + point * n_b + b_row];
                for a_idx in 0..n_a {
                    for digit in 0..depth_open {
                        for block in 0..num_blocks {
                            let phys = block * stride_t + a_idx * depth_open + digit;
                            let local_col = poly_idx * cols_per_poly_t + phys;
                            let lambda = b_row * n_cols_t + local_col;
                            let m_idx = block
                                + num_blocks
                                    * (flat_t_vector
                                        + num_t_vectors * digit
                                        + num_t_vectors * depth_open * a_idx);
                            bar_omega[lambda] += row_weight * eq(offset_t + m_idx);
                        }
                    }
                }
            }
            flat_t_vector += 1;
        }
    }

    // A/J: omega_bar(iota_A) += eq(tau_1, A_row) * eta_Z, where the J term is on
    // the weight side: eta_Z(b_z, d_c) = -sum_p sum_{d_f} g_f[d_f]
    //   * eq(r, offset_z + x_Z(p, d_f, d_c, b_z)). One formula covers both the
    // pow2 and dense Z layouts of the production path.
    for a_row in 0..n_a {
        let row_weight = prepared.eq_tau1[a_start + a_row];
        for dc in 0..depth_commit {
            for block in 0..block_len {
                let local_col = block * depth_commit + dc;
                let lambda = a_row * z_range + local_col;
                let mut eta = E::zero();
                for point in 0..num_points {
                    for (df, &fold_weight) in fold_gadget.iter().enumerate().take(depth_fold) {
                        let m_idx = block
                            + block_len * (point + num_points * df + num_points * depth_fold * dc);
                        eta += eq(offset_z + m_idx).mul_base(fold_weight);
                    }
                }
                bar_omega[lambda] += row_weight * (-eta);
            }
        }
    }

    bar_omega
}

/// Contract the materialized weight vector against the packed setup prefix.
///
/// `sigma_S = sum_lambda sum_y omega_bar_S(lambda) * alpha^y * S(lambda, y)`.
pub(crate) fn materialized_setup_contribution<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    alpha_pows: &[E],
    fold_gadget: &[F],
    setup: &AkitaExpandedSetup<F>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + MulBase<F>,
{
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    let bar_omega = materialized_bar_omega::<F, E>(prepared, full_vec_randomness, fold_gadget);

    let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
    if bar_omega.len() > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected verifier layout".into(),
        ));
    }
    let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
    let setup_flat = setup_view.as_slice();

    let mut total = E::zero();
    for (lambda, &weight) in bar_omega.iter().enumerate() {
        if weight.is_zero() {
            continue;
        }
        for (y, &coeff) in setup_flat[lambda].coefficients().iter().enumerate() {
            total += (weight * alpha_pows[y]).mul_base(coeff);
        }
    }
    Ok(total)
}
