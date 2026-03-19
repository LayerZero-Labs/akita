//! Native Hachi opening frontend lowered to Labrador constraints.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::commitment::utils::flat_matrix::FlatMatrix;
use crate::protocol::commitment::utils::linear::flatten_i8_blocks;
use crate::protocol::commitment::{CommitmentConfig, HachiCommitmentLayout, RingCommitment};
use crate::protocol::labrador::types::LabradorWitness;
use crate::protocol::labrador::{LabradorConstraint, LabradorConstraintTerm};
use crate::protocol::opening_point::RingOpeningPoint;
use crate::protocol::proof::HachiCommitmentHint;
use crate::{CanonicalField, FieldCore, FromSmallInt};

fn scalar_ring<F: FieldCore, const D: usize>(s: F) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(
        |idx| if idx == 0 { s } else { F::zero() },
    ))
}

fn digit_ring<F: FieldCore + FromSmallInt, const D: usize>(
    digits: &[i8; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(digits[idx] as i64)))
}

fn gadget_scalars<F: FieldCore + CanonicalField>(levels: usize, log_basis: u32) -> Vec<F> {
    let base = F::from_canonical_u128_reduced(1u128 << log_basis);
    let mut out = Vec::with_capacity(levels);
    let mut power = F::one();
    for _ in 0..levels {
        out.push(power);
        power = power * base;
    }
    out
}

/// Build the native Labrador witness `[current_w_digits, current_t_hat_digits]`.
pub(crate) fn build_hachi_opening_witness<F, const D: usize>(
    current_w: &[i8],
    current_hint: &HachiCommitmentHint<F, D>,
) -> Result<LabradorWitness<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
{
    let (w_digits, remainder) = current_w.as_chunks::<D>();
    if !remainder.is_empty() {
        return Err(HachiError::InvalidSize {
            expected: D,
            actual: current_w.len(),
        });
    }

    let row0: Vec<CyclotomicRing<F, D>> = w_digits.iter().map(digit_ring::<F, D>).collect();
    let t_hat_flat = flatten_i8_blocks(&current_hint.inner_opening_digits);
    let row1: Vec<CyclotomicRing<F, D>> = t_hat_flat.iter().map(digit_ring::<F, D>).collect();
    Ok(LabradorWitness::new_unchecked(vec![row0, row1]))
}

/// Build native Labrador constraints for the recursive opening claim.
///
/// Row 0 holds the carried witness digits `current_w`.
/// Row 1 holds the carried commitment hint digits `current_hint`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hachi_opening_constraints<F, const D: usize, Cfg>(
    a_mat: &FlatMatrix<F>,
    b_mat: &FlatMatrix<F>,
    opening_point: &RingOpeningPoint<F>,
    current_commitment: &RingCommitment<F, D>,
    y_ring: &CyclotomicRing<F, D>,
    layout: HachiCommitmentLayout,
    current_w_ring_len: usize,
) -> Result<Vec<LabradorConstraint<F, D>>, HachiError>
where
    F: FieldCore + CanonicalField,
    Cfg: CommitmentConfig,
{
    if layout.num_digits_commit != 1 {
        return Err(HachiError::InvalidInput(
            "native Hachi opening frontend requires num_digits_commit = 1".to_string(),
        ));
    }

    let depth_open = layout.num_digits_open;
    let num_blocks = opening_point.b.len();
    let block_len = layout.block_len;
    let t_block_width = Cfg::N_A * depth_open;
    let t_row_len = num_blocks * t_block_width;

    let a_view = a_mat.view::<D>();
    let b_view = b_mat.view::<D>();
    let neg_g_open: Vec<CyclotomicRing<F, D>> = gadget_scalars::<F>(depth_open, layout.log_basis)
        .into_iter()
        .map(|g| scalar_ring::<F, D>(-g))
        .collect();

    let mut constraints = Vec::with_capacity(num_blocks * Cfg::N_A + Cfg::N_B + 1);

    // Per-block Ajtai consistency:
    //   A * s_i = G_open * t_hat_i
    for block_idx in 0..num_blocks {
        let s_offset = block_idx * block_len;
        let live_len = current_w_ring_len.saturating_sub(s_offset).min(block_len);
        let t_block_offset = block_idx * t_block_width;
        for a_idx in 0..Cfg::N_A {
            let mut terms = Vec::with_capacity(if live_len > 0 { 2 } else { 1 });
            if live_len > 0 {
                let a_coeffs: Vec<CyclotomicRing<F, D>> =
                    a_view.row(a_idx).iter().take(live_len).copied().collect();
                terms.push(LabradorConstraintTerm::new(0, s_offset, a_coeffs));
            }
            let t_offset = t_block_offset + a_idx * depth_open;
            terms.push(LabradorConstraintTerm::new(1, t_offset, neg_g_open.clone()));
            constraints.push(LabradorConstraint::new(
                terms,
                CyclotomicRing::<F, D>::zero(),
            ));
        }
    }

    // Commitment consistency:
    //   B * t_hat_flat = current_commitment.u
    for (row_idx, &u_i) in current_commitment.u.iter().enumerate().take(Cfg::N_B) {
        let coeffs: Vec<CyclotomicRing<F, D>> = b_view
            .row(row_idx)
            .iter()
            .take(t_row_len)
            .copied()
            .collect();
        constraints.push(LabradorConstraint::new(
            vec![LabradorConstraintTerm::new(1, 0, coeffs)],
            u_i,
        ));
    }

    // Opening evaluation:
    //   sum_{block, inner} b_block * a_inner * s_{block, inner} = y_ring
    let phi_w: Vec<CyclotomicRing<F, D>> = (0..current_w_ring_len)
        .map(|idx| {
            let block_idx = idx / block_len;
            let inner_idx = idx % block_len;
            scalar_ring::<F, D>(opening_point.b[block_idx] * opening_point.a[inner_idx])
        })
        .collect();
    constraints.push(LabradorConstraint::new(
        vec![LabradorConstraintTerm::new(0, 0, phi_w)],
        *y_ring,
    ));

    Ok(constraints)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::commitment::{HachiCommitmentCore, RingCommitmentScheme};
    use crate::protocol::hachi_poly_ops::HachiPolyOps;
    use crate::protocol::opening_point::{ring_opening_point_from_field, BasisMode};
    use crate::protocol::ring_switch::{commit_w, WCommitmentConfig};
    use crate::test_utils::{TinyConfig, D, F};

    #[test]
    fn native_constraints_satisfied_by_witness() {
        const MAX_NUM_VARS: usize = 8;
        let (setup, _) = <HachiCommitmentCore as RingCommitmentScheme<
            F,
            D,
            WCommitmentConfig<D, TinyConfig>,
        >>::setup(MAX_NUM_VARS)
        .unwrap();
        let w_layout = WCommitmentConfig::<D, TinyConfig>::commitment_layout(MAX_NUM_VARS).unwrap();
        let ring_len = w_layout.num_blocks * w_layout.block_len;
        let current_w: Vec<i8> = (0..ring_len * D).map(|idx| ((idx % 5) as i8) - 2).collect();
        let (commitment, hint) =
            commit_w::<F, D, TinyConfig>(&current_w, &setup.ntt_A, &setup.ntt_B).unwrap();

        let alpha = D.trailing_zeros() as usize;
        let point: Vec<F> = (0..MAX_NUM_VARS)
            .map(|idx| F::from_u64((idx + 2) as u64))
            .collect();
        let mut padded_point = point.clone();
        padded_point.resize(w_layout.m_vars + w_layout.r_vars + alpha, F::zero());
        let ring_opening_point = ring_opening_point_from_field::<F>(
            &padded_point[alpha..],
            w_layout.r_vars,
            w_layout.m_vars,
            BasisMode::Lagrange,
        )
        .unwrap();
        let w_poly =
            crate::protocol::hachi_poly_ops::BalancedDigitPoly::<F, D>::from_i8_digits(&current_w)
                .unwrap();
        let (y_ring, _w_folded) = w_poly.evaluate_and_fold(
            &ring_opening_point.b,
            &ring_opening_point.a,
            w_layout.block_len,
        );

        let witness = build_hachi_opening_witness::<F, D>(&current_w, &hint).unwrap();
        let constraints =
            build_hachi_opening_constraints::<F, D, WCommitmentConfig<D, TinyConfig>>(
                &setup.expanded.A,
                &setup.expanded.B,
                &ring_opening_point,
                &commitment,
                &y_ring,
                w_layout,
                ring_len,
            )
            .unwrap();

        let rows = witness.rows();
        for (ci, constraint) in constraints.iter().enumerate() {
            let mut lhs = CyclotomicRing::<F, D>::zero();
            for term in &constraint.terms {
                for (j, coeff) in term.coefficients.iter().enumerate() {
                    lhs += *coeff * rows[term.row][term.offset + j];
                }
            }
            assert_eq!(
                lhs, constraint.target,
                "native constraint {ci} not satisfied"
            );
        }
    }
}
