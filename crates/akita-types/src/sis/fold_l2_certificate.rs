//! Folded-witness Euclidean (`L2`) certificate geometry: realization selection,
//! grouped-digit layout, and the structural no-wrap gate for grouped-carry proofs.
//!
//! The certificate proves an exact integer equality
//! `Σ z[i]² + Σ ell_h² = B_l2` on the committed fold-response digits `z_hat`.
//! This module holds the **public, value-independent** sizing that picks among:
//!
//! - **Field-fitting** — one degree-2 sumcheck when the squared norm and bound
//!   fit the base field (`N · digit_max² + 4·B_l2 < q`).
//! - **Grouped-carry** — digit grouping plus a carry chain when the field is too
//!   small but every convolution exponent passes the no-wrap gate.
//! - **Deterministic fallback** — no realized certificate; A-role pricing uses
//!   [`folded_witness_l2_bound_squared`].
//!
//! `q` is always the **base-field characteristic** (extension degree does not
//! widen the gate).

use akita_field::AkitaError;

use super::ajtai_key::{ceil_supported_collision, SisModulusFamily};
use super::decomposition_digits::balanced_positive_digit_max;

/// How a certifying fold level proves its Euclidean norm bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L2CertificateRealization {
    /// `Σ_x z_aug(x)² = B_l2` in one sumcheck; no carries.
    FieldFitting,
    /// Grouped limbs with `group_digits` original digits per limb and a carry chain.
    GroupedCarry {
        /// Original `z_hat` digits merged into one grouped limb (`g` in specs).
        group_digits: usize,
    },
    /// No realized certificate; price at [`folded_witness_l2_bound_squared`].
    DeterministicFallback,
}

/// Public layout of grouped fold digits for the grouped-carry realization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldNormGrouping {
    /// Original `z_hat` digits per grouped limb (`g`), except possibly the last.
    pub group_digits: usize,
    /// Grouped limb count `R = ceil(K / g)`.
    pub group_count: usize,
    /// Width of the last (possibly short) group.
    pub last_group_width: usize,
}

/// Per-exponent carry cell counts for `carry_hat` (base-2 recomposition).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CarryCellLayout {
    /// `delta_carry(e)` for each carry index `e` in `0..carry_count`.
    pub cells_per_carry: Vec<usize>,
}

impl FoldNormGrouping {
    /// Build the grouping for `num_digits_fold = K` and group size `g`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when `group_digits == 0` or
    /// `num_digits_fold == 0`.
    pub fn new(num_digits_fold: usize, group_digits: usize) -> Result<Self, AkitaError> {
        if num_digits_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "FoldNormGrouping: num_digits_fold must be positive".to_string(),
            ));
        }
        if group_digits == 0 {
            return Err(AkitaError::InvalidSetup(
                "FoldNormGrouping: group_digits must be positive".to_string(),
            ));
        }
        let group_count = num_digits_fold.div_ceil(group_digits);
        let consumed = group_digits.saturating_mul(group_count.saturating_sub(1));
        let last_group_width = num_digits_fold.saturating_sub(consumed).max(1);
        Ok(Self {
            group_digits,
            group_count,
            last_group_width,
        })
    }

    /// Width of grouped limb `j` (`g_j` in specs).
    #[inline]
    pub fn limb_width(&self, limb_index: usize) -> usize {
        if limb_index + 1 == self.group_count {
            self.last_group_width
        } else {
            self.group_digits
        }
    }
}

/// Conservative max absolute value of a grouped folded-witness limb:
/// `(b/2) · (b^{g_j} - 1) / (b - 1)` for `b = 2^log_basis`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on invalid `log_basis` or overflow.
pub fn grouped_fold_limb_bound(log_basis: u32, limb_width: usize) -> Result<u128, AkitaError> {
    if log_basis == 0 || log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(format!(
            "grouped_fold_limb_bound: invalid log_basis {log_basis}"
        )));
    }
    if limb_width == 0 {
        return Err(AkitaError::InvalidSetup(
            "grouped_fold_limb_bound: limb_width must be positive".to_string(),
        ));
    }
    let half_base = 1u128.checked_shl(log_basis - 1).ok_or_else(|| {
        AkitaError::InvalidSetup("grouped_fold_limb_bound: b/2 overflow".to_string())
    })?;
    let base = 1u128.checked_shl(log_basis).ok_or_else(|| {
        AkitaError::InvalidSetup("grouped_fold_limb_bound: b overflow".to_string())
    })?;
    let base_pow = base.checked_pow(limb_width as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("grouped_fold_limb_bound: b^g overflow".to_string())
    })?;
    let numerator = half_base.checked_mul(base_pow - 1).ok_or_else(|| {
        AkitaError::InvalidSetup("grouped_fold_limb_bound: numerator overflow".to_string())
    })?;
    Ok(numerator / (base - 1))
}

/// Deterministic worst-case squared Euclidean bound on the fold response:
/// `N · balanced_digit_max(lb, K)²`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on overflow.
pub fn folded_witness_l2_bound_squared(
    num_fold_coeffs: usize,
    log_basis: u32,
    num_digits_fold: usize,
) -> Result<u128, AkitaError> {
    if num_fold_coeffs == 0 {
        return Err(AkitaError::InvalidSetup(
            "folded_witness_l2_bound_squared: num_fold_coeffs must be positive".to_string(),
        ));
    }
    let digit_max = balanced_positive_digit_max(log_basis, num_digits_fold);
    let sq = digit_max.checked_mul(digit_max).ok_or_else(|| {
        AkitaError::InvalidSetup("folded_witness_l2_bound_squared: digit_max² overflow".to_string())
    })?;
    (num_fold_coeffs as u128).checked_mul(sq).ok_or_else(|| {
        AkitaError::InvalidSetup(
            "folded_witness_l2_bound_squared: N · digit_max² overflow".to_string(),
        )
    })
}

/// Round a realized squared norm up to the next audited L2 MSIS collision bucket.
pub fn l2_collision_bucket_for_z_squared(
    sis_family: SisModulusFamily,
    ring_dimension: u32,
    z_squared: u128,
) -> Option<u128> {
    ceil_supported_collision(sis_family, ring_dimension, z_squared)
}

/// Field-fitting realization: `N·m² + 4·B_l2 < q` for `m = digit_max(lb, K)`.
#[inline]
pub fn field_fitting_certificate_fits(
    num_fold_coeffs: usize,
    log_basis: u32,
    num_digits_fold: usize,
    b_l2: u128,
    field_characteristic: u128,
) -> Result<bool, AkitaError> {
    if field_characteristic == 0 {
        return Err(AkitaError::InvalidSetup(
            "field_fitting_certificate_fits: field_characteristic must be positive".to_string(),
        ));
    }
    let digit_max = balanced_positive_digit_max(log_basis, num_digits_fold);
    let witness_sq = (num_fold_coeffs as u128)
        .checked_mul(digit_max)
        .and_then(|n| n.checked_mul(digit_max))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("field_fitting_certificate_fits: N·m² overflow".to_string())
        })?;
    let slack = b_l2.checked_mul(4).ok_or_else(|| {
        AkitaError::InvalidSetup("field_fitting_certificate_fits: 4·B_l2 overflow".to_string())
    })?;
    let lhs = witness_sq.checked_add(slack).ok_or_else(|| {
        AkitaError::InvalidSetup("field_fitting_certificate_fits: sum overflow".to_string())
    })?;
    Ok(lhs < field_characteristic)
}

/// Structural bound `D_e` on unreduced convolution coefficient `e` (ordered pairs).
fn structural_convolution_bound_de(
    num_fold_coeffs: usize,
    log_basis: u32,
    grouping: &FoldNormGrouping,
    exponent: usize,
) -> Result<u128, AkitaError> {
    let mut sum = 0u128;
    for r in 0..grouping.group_count {
        for s in 0..grouping.group_count {
            if r + s != exponent {
                continue;
            }
            let a_r = grouped_fold_limb_bound(log_basis, grouping.limb_width(r))?;
            let a_s = grouped_fold_limb_bound(log_basis, grouping.limb_width(s))?;
            let witness_term = (num_fold_coeffs as u128)
                .checked_mul(a_r)
                .and_then(|t| t.checked_mul(a_s))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "structural_convolution_bound_de: overflow".to_string(),
                    )
                })?;
            let slack_term = 4u128
                .checked_mul(a_r)
                .and_then(|t| t.checked_mul(a_s))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "structural_convolution_bound_de: overflow".to_string(),
                    )
                })?;
            sum = sum
                .checked_add(witness_term)
                .and_then(|t| t.checked_add(slack_term))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "structural_convolution_bound_de: overflow".to_string(),
                    )
                })?;
        }
    }
    Ok(sum)
}

/// Tight carry magnitude budget `H_e` from the structural recurrence (ignoring `T_e`).
fn carry_magnitude_budget(
    log_basis: u32,
    grouping: &FoldNormGrouping,
    num_fold_coeffs: usize,
) -> Result<Vec<u128>, AkitaError> {
    if grouping.group_count == 0 {
        return Err(AkitaError::InvalidSetup(
            "carry_magnitude_budget: empty grouping".to_string(),
        ));
    }
    let base = 1u128.checked_shl(log_basis).ok_or_else(|| {
        AkitaError::InvalidSetup("carry_magnitude_budget: b overflow".to_string())
    })?;
    let max_exponent = grouping.group_count.saturating_mul(2).saturating_sub(2);
    let mut h = vec![0u128; max_exponent + 2];
    for e in 0..=max_exponent {
        let d_e = structural_convolution_bound_de(num_fold_coeffs, log_basis, grouping, e)?;
        h[e + 1] = d_e
            .checked_add(h[e])
            .and_then(|t| t.checked_add(base - 1))
            .map(|t| t.div_ceil(base))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("carry_magnitude_budget: recurrence overflow".to_string())
            })?;
    }
    Ok(h)
}

/// Realizable carry budget `H'_e = (b/2)·(2^δ - 1)` with minimal cell count `δ`.
pub fn realizable_carry_budget(log_basis: u32, tight_budget: u128) -> Result<u128, AkitaError> {
    if tight_budget == 0 {
        return Ok(0);
    }
    if log_basis == 0 || log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(format!(
            "realizable_carry_budget: invalid log_basis {log_basis}"
        )));
    }
    let half_base = 1u128.checked_shl(log_basis - 1).ok_or_else(|| {
        AkitaError::InvalidSetup("realizable_carry_budget: b/2 overflow".to_string())
    })?;
    let mut delta = 1usize;
    loop {
        let cells = 1u128
            .checked_shl(delta as u32)
            .and_then(|t| t.checked_sub(1))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("realizable_carry_budget: 2^δ overflow".to_string())
            })?;
        let h_prime = half_base.checked_mul(cells).ok_or_else(|| {
            AkitaError::InvalidSetup("realizable_carry_budget: H' overflow".to_string())
        })?;
        if h_prime >= tight_budget {
            return Ok(h_prime);
        }
        delta = delta.checked_add(1).ok_or_else(|| {
            AkitaError::InvalidSetup("realizable_carry_budget: delta overflow".to_string())
        })?;
        if delta > 128 {
            return Err(AkitaError::InvalidSetup(
                "realizable_carry_budget: no δ with H' >= H".to_string(),
            ));
        }
    }
}

/// Smallest `delta_carry(e)` with `H'_e >= H_e`.
pub fn carry_cell_count_for_budget(
    log_basis: u32,
    tight_budget: u128,
) -> Result<usize, AkitaError> {
    if tight_budget == 0 {
        return Ok(0);
    }
    if log_basis == 0 || log_basis >= 128 {
        return Err(AkitaError::InvalidSetup(format!(
            "carry_cell_count_for_budget: invalid log_basis {log_basis}"
        )));
    }
    let half_base = 1u128.checked_shl(log_basis - 1).ok_or_else(|| {
        AkitaError::InvalidSetup("carry_cell_count_for_budget: b/2 overflow".to_string())
    })?;
    for delta in 1..=128usize {
        let cells = 1u128
            .checked_shl(delta as u32)
            .and_then(|t| t.checked_sub(1))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("carry_cell_count_for_budget: 2^δ overflow".to_string())
            })?;
        let h_prime = half_base.checked_mul(cells).ok_or_else(|| {
            AkitaError::InvalidSetup("carry_cell_count_for_budget: H' overflow".to_string())
        })?;
        if h_prime >= tight_budget {
            return Ok(delta);
        }
    }
    Err(AkitaError::InvalidSetup(
        "carry_cell_count_for_budget: no δ with H' >= H".to_string(),
    ))
}

/// Carry cell layout for a grouped-carry realization.
pub fn carry_cell_layout(
    log_basis: u32,
    grouping: &FoldNormGrouping,
    num_fold_coeffs: usize,
) -> Result<CarryCellLayout, AkitaError> {
    let h = carry_magnitude_budget(log_basis, grouping, num_fold_coeffs)?;
    let cells: Result<Vec<_>, _> = h
        .iter()
        .take(h.len().saturating_sub(1))
        .map(|&budget| carry_cell_count_for_budget(log_basis, budget))
        .collect();
    Ok(CarryCellLayout {
        cells_per_carry: cells?,
    })
}

/// Grouped-carry no-wrap gate: `D_e + H'_e + (B-1) + B·H'_{e+1} < q` for every `e`.
pub fn grouped_carry_no_wrap_gate_holds(
    num_fold_coeffs: usize,
    log_basis: u32,
    grouping: &FoldNormGrouping,
    field_characteristic: u128,
) -> Result<bool, AkitaError> {
    if field_characteristic == 0 {
        return Err(AkitaError::InvalidSetup(
            "grouped_carry_no_wrap_gate_holds: field_characteristic must be positive".to_string(),
        ));
    }
    let base = 1u128.checked_shl(log_basis).ok_or_else(|| {
        AkitaError::InvalidSetup("grouped_carry_no_wrap_gate_holds: b overflow".to_string())
    })?;
    let grouped_base = base
        .checked_pow(grouping.group_digits as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("grouped_carry_no_wrap_gate_holds: B overflow".to_string())
        })?;
    let h = carry_magnitude_budget(log_basis, grouping, num_fold_coeffs)?;
    let max_exponent = grouping.group_count.saturating_mul(2).saturating_sub(2);
    for e in 0..=max_exponent {
        let d_e = structural_convolution_bound_de(num_fold_coeffs, log_basis, grouping, e)?;
        let h_e = realizable_carry_budget(log_basis, h[e])?;
        let h_next = realizable_carry_budget(log_basis, h[e + 1])?;
        let residual_bound = d_e
            .checked_add(h_e)
            .and_then(|t| t.checked_add(grouped_base - 1))
            .and_then(|t| t.checked_add(grouped_base.checked_mul(h_next)?))
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "grouped_carry_no_wrap_gate_holds: residual overflow".to_string(),
                )
            })?;
        if residual_bound >= field_characteristic {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Select the certificate realization from public fold geometry and field size.
///
/// `b_l2_for_field_fit` is the bound used in the field-fitting inequality (typically
/// the deterministic [`folded_witness_l2_bound_squared`] or the audited bucket).
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on invalid parameters or internal overflow.
pub fn select_l2_certificate_realization(
    num_fold_coeffs: usize,
    log_basis: u32,
    num_digits_fold: usize,
    b_l2_for_field_fit: u128,
    field_characteristic: u128,
) -> Result<L2CertificateRealization, AkitaError> {
    if num_digits_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "select_l2_certificate_realization: num_digits_fold must be positive".to_string(),
        ));
    }
    if field_fitting_certificate_fits(
        num_fold_coeffs,
        log_basis,
        num_digits_fold,
        b_l2_for_field_fit,
        field_characteristic,
    )? {
        return Ok(L2CertificateRealization::FieldFitting);
    }
    let max_group_digits = num_digits_fold.saturating_sub(1).max(1);
    for group_digits in (1..=max_group_digits).rev() {
        let grouping = FoldNormGrouping::new(num_digits_fold, group_digits)?;
        if grouped_carry_no_wrap_gate_holds(
            num_fold_coeffs,
            log_basis,
            &grouping,
            field_characteristic,
        )? {
            return Ok(L2CertificateRealization::GroupedCarry { group_digits });
        }
    }
    Ok(L2CertificateRealization::DeterministicFallback)
}

#[cfg(test)]
mod tests {
    use super::super::ajtai_key::SisModulusFamily;
    use super::*;

    const Q_FP32: u128 = (1u128 << 32) - 99;
    const Q_FP128: u128 = u128::MAX - (1u128 << 32) + 22_538;

    #[test]
    fn folded_witness_l2_bound_squared_matches_digit_max() {
        let n = 57_344usize;
        let bound = folded_witness_l2_bound_squared(n, 3, 5).unwrap();
        let m = balanced_positive_digit_max(3, 5);
        assert_eq!(bound, n as u128 * m * m);
    }

    #[test]
    fn fp128_large_root_uses_field_fitting() {
        let n = 67_108_864usize;
        let k = 9usize;
        let b_l2 = folded_witness_l2_bound_squared(n, 2, k).unwrap();
        assert!(field_fitting_certificate_fits(n, 2, k, b_l2, Q_FP128).unwrap());
        assert_eq!(
            select_l2_certificate_realization(n, 2, k, b_l2, Q_FP128).unwrap(),
            L2CertificateRealization::FieldFitting,
        );
    }

    #[test]
    fn fp32_large_root_grouped_carry_when_k_grows() {
        let n = 67_108_864usize;
        let b_l2 = folded_witness_l2_bound_squared(n, 2, 5).unwrap();
        assert!(!field_fitting_certificate_fits(n, 2, 5, b_l2, Q_FP32).unwrap());
        let realization = select_l2_certificate_realization(n, 2, 5, b_l2, Q_FP32).unwrap();
        assert!(matches!(
            realization,
            L2CertificateRealization::GroupedCarry { .. }
        ));
    }

    #[test]
    fn fp32_total_fallback_when_n_and_k_both_large() {
        let n = 67_108_864usize;
        let b_l2 = folded_witness_l2_bound_squared(n, 2, 9).unwrap();
        assert_eq!(
            select_l2_certificate_realization(n, 2, 9, b_l2, Q_FP32).unwrap(),
            L2CertificateRealization::DeterministicFallback,
        );
    }

    #[test]
    fn recursive_dense_level_certifies_on_fp32() {
        let n = 57_344usize;
        let b_l2 = folded_witness_l2_bound_squared(n, 3, 5).unwrap();
        let realization = select_l2_certificate_realization(n, 3, 5, b_l2, Q_FP32).unwrap();
        assert!(matches!(
            realization,
            L2CertificateRealization::GroupedCarry { .. } | L2CertificateRealization::FieldFitting
        ));
    }

    #[test]
    fn fold_norm_grouping_last_short_group() {
        let g = FoldNormGrouping::new(5, 2).unwrap();
        assert_eq!(g.group_count, 3);
        assert_eq!(g.limb_width(0), 2);
        assert_eq!(g.limb_width(2), 1);
    }

    #[test]
    fn l2_collision_bucket_rounds_up() {
        let bucket = l2_collision_bucket_for_z_squared(SisModulusFamily::Q32, 64, 1_000).unwrap();
        assert!(bucket >= 1_000);
    }
}
