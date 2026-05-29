//! Materialized `<S, omega_S>` oracle for packed setup contribution.
//!
//! `bar_omega[lambda]` accumulates overlapping D/B/A role weights at each shared
//! setup ring slot. `omega_S(lambda, y) = bar_omega[lambda] * alpha^y`, so alpha
//! lives on the weight side rather than in committed setup.
//!
//! The direct verifier still evaluates the inner product incrementally in
//! `compute_setup_contribution`. This test-only module materializes the full
//! weight tensor as an equivalence oracle for later setup product-sumcheck work.

use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::CyclotomicRing;
use akita_field::{ExtField, FieldCore, MulBase};

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

/// Flat coefficient weights for `<S_{<=N}, omega_S>`.
pub(crate) struct MaterializedSetupOmega<E> {
    pub bar_omega: Vec<E>,
    pub omega_s: Vec<E>,
}

impl<E: FieldCore> MaterializedSetupOmega<E> {
    pub(super) fn coefficient_weight(&self, lambda: usize, y: usize, ring_dim: usize) -> E {
        self.omega_s[lambda * ring_dim + y]
    }

    pub(super) fn inner_product<F, const D: usize>(
        &self,
        setup_entries: &[CyclotomicRing<F, D>],
    ) -> E
    where
        F: FieldCore,
        E: ExtField<F> + MulBase<F>,
    {
        setup_entries
            .iter()
            .enumerate()
            .take(self.bar_omega.len())
            .map(|(lambda, ring)| {
                ring.coefficients()
                    .iter()
                    .enumerate()
                    .map(|(y, &coeff)| self.coefficient_weight(lambda, y, D).mul_base(coeff))
                    .sum::<E>()
            })
            .sum()
    }
}

/// Build the packed setup weight tensor used by the direct inner-product scan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_setup_omega<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    full_vec_randomness: &[E],
    alpha_pows: &[E],
    fold_gadget: &[F],
    offset_w: usize,
    offset_t: usize,
    offset_z: usize,
) -> MaterializedSetupOmega<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let n_d_active = prepared.n_d_active();
    let d_start = 1 + prepared.num_public_rows;
    let b_start = d_start + n_d_active;
    let a_start = b_start + prepared.n_b * prepared.num_points;

    let stride_t = prepared.n_a * prepared.depth_open;
    let cols_per_poly_t = stride_t * prepared.num_blocks;
    let b_per_claim_w = prepared.num_blocks * prepared.depth_open;
    let n_cols_w = prepared.num_claims * b_per_claim_w;
    let max_group_poly_count = prepared
        .num_polys_per_point
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let n_cols_t = max_group_poly_count * cols_per_poly_t;
    let z_range = prepared.inner_width;
    let required = (n_d_active * n_cols_w)
        .max(prepared.n_b * n_cols_t)
        .max(prepared.n_a * z_range);

    let mut group_offsets = Vec::with_capacity(prepared.num_polys_per_point.len());
    let mut next_offset = 0usize;
    for &group_poly_count in &prepared.num_polys_per_point {
        group_offsets.push(next_offset);
        next_offset += group_poly_count;
    }

    let mut bar_omega = vec![E::zero(); required];

    for row in 0..n_d_active {
        let row_weight = prepared.eq_tau1[d_start + row];
        for claim in 0..prepared.num_claims {
            for block in 0..prepared.num_blocks {
                for digit in 0..prepared.depth_open {
                    let col = (claim * prepared.num_blocks + block) * prepared.depth_open + digit;
                    let lambda = row * n_cols_w + col;
                    let m_idx = block + prepared.num_blocks * (claim + prepared.num_claims * digit);
                    bar_omega[lambda] +=
                        row_weight * eq_eval_at_index(full_vec_randomness, offset_w + m_idx);
                }
            }
        }
    }

    for (point_idx, &group_poly_count) in prepared.num_polys_per_point.iter().enumerate() {
        for poly_idx in 0..group_poly_count {
            let flat_t_vector = group_offsets[point_idx] + poly_idx;
            for row in 0..prepared.n_b {
                let row_weight = prepared.eq_tau1[b_start + point_idx * prepared.n_b + row];
                for a_idx in 0..prepared.n_a {
                    for digit in 0..prepared.depth_open {
                        for block in 0..prepared.num_blocks {
                            let col = poly_idx * cols_per_poly_t
                                + block * stride_t
                                + a_idx * prepared.depth_open
                                + digit;
                            let lambda = row * n_cols_t + col;
                            let m_idx = block
                                + prepared.num_blocks
                                    * (flat_t_vector
                                        + prepared.num_t_vectors * digit
                                        + prepared.num_t_vectors * prepared.depth_open * a_idx);
                            bar_omega[lambda] += row_weight
                                * eq_eval_at_index(full_vec_randomness, offset_t + m_idx);
                        }
                    }
                }
            }
        }
    }

    for row in 0..prepared.n_a {
        let row_weight = prepared.eq_tau1[a_start + row];
        for dc in 0..prepared.depth_commit {
            for (df, &fold_weight) in fold_gadget.iter().enumerate() {
                for point in 0..prepared.num_points {
                    for block in 0..prepared.block_len {
                        let col = block * prepared.depth_commit + dc;
                        let lambda = row * z_range + col;
                        let m_idx = block
                            + prepared.block_len
                                * (point
                                    + prepared.num_points * df
                                    + prepared.num_points * prepared.depth_fold * dc);
                        bar_omega[lambda] -= row_weight
                            * eq_eval_at_index(full_vec_randomness, offset_z + m_idx)
                                .mul_base(fold_weight);
                    }
                }
            }
        }
    }

    let omega_s = bar_omega
        .iter()
        .flat_map(|&weight| alpha_pows.iter().map(move |&alpha_pow| weight * alpha_pow))
        .collect();

    MaterializedSetupOmega { bar_omega, omega_s }
}
