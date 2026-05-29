//! Materialized `<S, omega_S>` oracle for packed setup contribution.
//!
//! `bar_omega[lambda]` accumulates overlapping D/B/A role weights at each shared
//! setup ring slot. `omega_S(lambda, y) = bar_omega[lambda] * alpha^y`, so alpha
//! lives on the weight side rather than in committed setup.
//!
//! The direct verifier still evaluates the inner product incrementally in
//! `compute_setup_contribution`. This module materializes the full weight tensor
//! for setup product-sumcheck work.

#![allow(dead_code)] // Production setup offloading consumes this path once wired.

use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, ExtField, FieldCore, MulBase};

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

/// Flat coefficient weights for `<S_{<=N}, omega_S>`.
pub(crate) struct MaterializedSetupOmega<E> {
    pub bar_omega: Vec<E>,
    pub omega_s: Vec<E>,
}

impl<E: FieldCore> MaterializedSetupOmega<E> {
    pub(super) fn coefficient_weight(
        &self,
        lambda: usize,
        y: usize,
        ring_dim: usize,
    ) -> Result<E, AkitaError> {
        let idx = checked_mul(lambda, ring_dim, "omega_S coefficient offset")?
            .checked_add(y)
            .ok_or_else(|| AkitaError::InvalidSetup("omega_S coefficient index overflow".into()))?;
        self.omega_s.get(idx).copied().ok_or_else(|| {
            AkitaError::InvalidSetup("omega_S coefficient index is out of bounds".into())
        })
    }

    pub(super) fn inner_product<F, const D: usize>(
        &self,
        setup_entries: &[CyclotomicRing<F, D>],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBase<F>,
    {
        if setup_entries.len() < self.bar_omega.len() {
            return Err(AkitaError::InvalidSize {
                expected: self.bar_omega.len(),
                actual: setup_entries.len(),
            });
        }
        let expected_omega_len = checked_mul(self.bar_omega.len(), D, "omega_S length")?;
        if self.omega_s.len() != expected_omega_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_omega_len,
                actual: self.omega_s.len(),
            });
        }

        let mut total = E::zero();
        for (lambda, ring) in setup_entries.iter().enumerate().take(self.bar_omega.len()) {
            for (y, &coeff) in ring.coefficients().iter().enumerate() {
                total += self.coefficient_weight(lambda, y, D)?.mul_base(coeff);
            }
        }
        Ok(total)
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
) -> Result<MaterializedSetupOmega<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    if alpha_pows.len() != D {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: alpha_pows.len(),
        });
    }
    if prepared.num_blocks == 0 || prepared.depth_open == 0 || prepared.depth_commit == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup omega layout has zero width".into(),
        ));
    }
    if fold_gadget.len() < prepared.depth_fold {
        return Err(AkitaError::InvalidSize {
            expected: prepared.depth_fold,
            actual: fold_gadget.len(),
        });
    }

    let n_d_active = prepared.n_d_active();
    let d_start = checked_add(1, prepared.num_public_rows, "D row start")?;
    let b_start = checked_add(d_start, n_d_active, "B row start")?;
    let b_rows = checked_mul(prepared.n_b, prepared.num_points, "B row count")?;
    let a_start = checked_add(b_start, b_rows, "A row start")?;
    let a_end = checked_add(a_start, prepared.n_a, "A row end")?;
    if a_end > prepared.rows || prepared.rows > prepared.eq_tau1.len() {
        return Err(AkitaError::InvalidSetup(
            "M-row weights are inconsistent with setup omega layout".into(),
        ));
    }

    let stride_t = checked_mul(prepared.n_a, prepared.depth_open, "T stride")?;
    let cols_per_poly_t = checked_mul(stride_t, prepared.num_blocks, "T polynomial width")?;
    let b_per_claim_w = checked_mul(prepared.num_blocks, prepared.depth_open, "W claim width")?;
    let n_cols_w = checked_mul(prepared.num_claims, b_per_claim_w, "W column width")?;
    let max_group_poly_count = prepared
        .num_polys_per_point
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let n_cols_t = checked_mul(max_group_poly_count, cols_per_poly_t, "T column width")?;
    let z_range = prepared.inner_width;
    let d_required = checked_mul(n_d_active, n_cols_w, "D setup footprint")?;
    let b_required = checked_mul(prepared.n_b, n_cols_t, "B setup footprint")?;
    let a_required = checked_mul(prepared.n_a, z_range, "A setup footprint")?;
    let required = d_required.max(b_required).max(a_required);

    let mut group_offsets = Vec::with_capacity(prepared.num_polys_per_point.len());
    let mut next_offset = 0usize;
    for &group_poly_count in &prepared.num_polys_per_point {
        group_offsets.push(next_offset);
        next_offset = checked_add(next_offset, group_poly_count, "T vector offset")?;
    }
    if next_offset != prepared.num_t_vectors {
        return Err(AkitaError::InvalidSetup(
            "T vector count is inconsistent with point polynomial counts".into(),
        ));
    }

    let mut bar_omega = vec![E::zero(); required];

    for row in 0..n_d_active {
        let row_weight = get_weight(&prepared.eq_tau1, checked_add(d_start, row, "D row")?)?;
        for claim in 0..prepared.num_claims {
            for block in 0..prepared.num_blocks {
                for digit in 0..prepared.depth_open {
                    let claim_block = checked_add(
                        checked_mul(claim, prepared.num_blocks, "D claim block")?,
                        block,
                        "D block column",
                    )?;
                    let col = checked_add(
                        checked_mul(claim_block, prepared.depth_open, "D column")?,
                        digit,
                        "D digit column",
                    )?;
                    let lambda =
                        checked_add(checked_mul(row, n_cols_w, "D lambda row")?, col, "D lambda")?;
                    let m_idx = checked_add(
                        block,
                        checked_mul(
                            prepared.num_blocks,
                            checked_add(
                                claim,
                                checked_mul(prepared.num_claims, digit, "D high index")?,
                                "D claim high index",
                            )?,
                            "D M index",
                        )?,
                        "D M index",
                    )?;
                    let eq_idx = checked_add(offset_w, m_idx, "D eq index")?;
                    *get_weight_mut(&mut bar_omega, lambda)? +=
                        row_weight * eq_eval_at_index(full_vec_randomness, eq_idx);
                }
            }
        }
    }

    for (point_idx, &group_poly_count) in prepared.num_polys_per_point.iter().enumerate() {
        for poly_idx in 0..group_poly_count {
            let flat_t_vector = checked_add(
                *group_offsets.get(point_idx).ok_or_else(|| {
                    AkitaError::InvalidSetup("T vector point offset is out of bounds".into())
                })?,
                poly_idx,
                "T vector index",
            )?;
            for row in 0..prepared.n_b {
                let b_row_offset = checked_mul(point_idx, prepared.n_b, "B point row offset")?;
                let row_weight = get_weight(
                    &prepared.eq_tau1,
                    checked_add(
                        b_start,
                        checked_add(b_row_offset, row, "B local row")?,
                        "B row",
                    )?,
                )?;
                for a_idx in 0..prepared.n_a {
                    for digit in 0..prepared.depth_open {
                        for block in 0..prepared.num_blocks {
                            let col = checked_add(
                                checked_add(
                                    checked_mul(poly_idx, cols_per_poly_t, "B poly column")?,
                                    checked_mul(block, stride_t, "B block column")?,
                                    "B column",
                                )?,
                                checked_add(
                                    checked_mul(a_idx, prepared.depth_open, "B A-row column")?,
                                    digit,
                                    "B digit column",
                                )?,
                                "B column",
                            )?;
                            let lambda = checked_add(
                                checked_mul(row, n_cols_t, "B lambda row")?,
                                col,
                                "B lambda",
                            )?;
                            let high_idx = checked_add(
                                checked_add(
                                    flat_t_vector,
                                    checked_mul(
                                        prepared.num_t_vectors,
                                        digit,
                                        "B digit high index",
                                    )?,
                                    "B high index",
                                )?,
                                checked_mul(
                                    checked_mul(
                                        prepared.num_t_vectors,
                                        prepared.depth_open,
                                        "B A-row high stride",
                                    )?,
                                    a_idx,
                                    "B A-row high index",
                                )?,
                                "B high index",
                            )?;
                            let m_idx = checked_add(
                                block,
                                checked_mul(prepared.num_blocks, high_idx, "B M index")?,
                                "B M index",
                            )?;
                            let eq_idx = checked_add(offset_t, m_idx, "B eq index")?;
                            *get_weight_mut(&mut bar_omega, lambda)? +=
                                row_weight * eq_eval_at_index(full_vec_randomness, eq_idx);
                        }
                    }
                }
            }
        }
    }

    for row in 0..prepared.n_a {
        let row_weight = get_weight(&prepared.eq_tau1, checked_add(a_start, row, "A row")?)?;
        for dc in 0..prepared.depth_commit {
            for (df, &fold_weight) in fold_gadget.iter().enumerate().take(prepared.depth_fold) {
                for point in 0..prepared.num_points {
                    for block in 0..prepared.block_len {
                        let col = checked_add(
                            checked_mul(block, prepared.depth_commit, "A column")?,
                            dc,
                            "A column",
                        )?;
                        let lambda = checked_add(
                            checked_mul(row, z_range, "A lambda row")?,
                            col,
                            "A lambda",
                        )?;
                        let z_high_idx = checked_add(
                            checked_add(
                                point,
                                checked_mul(prepared.num_points, df, "A fold high index")?,
                                "A point high index",
                            )?,
                            checked_mul(
                                checked_mul(
                                    prepared.num_points,
                                    prepared.depth_fold,
                                    "A commit high stride",
                                )?,
                                dc,
                                "A commit high index",
                            )?,
                            "A high index",
                        )?;
                        let m_idx = checked_add(
                            block,
                            checked_mul(prepared.block_len, z_high_idx, "A M index")?,
                            "A M index",
                        )?;
                        let eq_idx = checked_add(offset_z, m_idx, "A eq index")?;
                        *get_weight_mut(&mut bar_omega, lambda)? -= row_weight
                            * eq_eval_at_index(full_vec_randomness, eq_idx).mul_base(fold_weight);
                    }
                }
            }
        }
    }

    let omega_len = checked_mul(bar_omega.len(), D, "omega_S length")?;
    let mut omega_s = Vec::with_capacity(omega_len);
    for &weight in &bar_omega {
        for &alpha_pow in alpha_pows {
            omega_s.push(weight * alpha_pow);
        }
    }

    Ok(MaterializedSetupOmega { bar_omega, omega_s })
}

#[inline(always)]
fn checked_add(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}

#[inline(always)]
fn checked_mul(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, AkitaError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{name} overflow")))
}

#[inline(always)]
fn get_weight<E: Copy>(weights: &[E], idx: usize) -> Result<E, AkitaError> {
    weights.get(idx).copied().ok_or_else(|| {
        AkitaError::InvalidSetup("setup omega row weight index is out of bounds".into())
    })
}

#[inline(always)]
fn get_weight_mut<E>(weights: &mut [E], idx: usize) -> Result<&mut E, AkitaError> {
    weights.get_mut(idx).ok_or_else(|| {
        AkitaError::InvalidSetup("setup omega coordinate index is out of bounds".into())
    })
}
