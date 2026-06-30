//! Shared protocol relation helpers.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at, eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, MulBase};
use std::iter::repeat_n;

/// Build the RHS vector `y` matching the M row layout:
/// consistency | D (`v`) | COMMIT (`commitment_rows`) | B_inner (zeros) | A.
///
/// Public-output rows bind through the fused trace term, not `y`.
///
/// `commit_rows_per_group` is the sent-commitment row count per group
/// (`effective_commit_rows`: the `F` rows when tiered, the `B` rows otherwise);
/// `b_inner_rows_per_group` is the inner-consistency block size per group
/// (`0` for single-tier). The number of commitment bundles is inferred from
/// `commitment_rows.len() / commit_rows_per_group`.
///
/// # Errors
///
/// Returns an error if the supplied row slices do not match the expected row
/// counts for the level layout.
#[allow(clippy::too_many_arguments)]
pub fn generate_y<F, const D: usize>(
    consistency_row: CyclotomicRing<F, D>,
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    a_rows: &[CyclotomicRing<F, D>],
    n_d: usize,
    commit_rows_per_group: usize,
    b_inner_rows_per_group: usize,
    n_a: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore,
{
    if v.len() != n_d {
        return Err(AkitaError::InvalidSize {
            expected: n_d,
            actual: v.len(),
        });
    }
    if a_rows.len() != n_a {
        return Err(AkitaError::InvalidSize {
            expected: n_a,
            actual: a_rows.len(),
        });
    }
    if commit_rows_per_group == 0
        || commitment_rows.is_empty()
        || !commitment_rows.len().is_multiple_of(commit_rows_per_group)
    {
        return Err(AkitaError::InvalidSize {
            expected: commit_rows_per_group,
            actual: commitment_rows.len(),
        });
    }
    let num_commitments = commitment_rows.len() / commit_rows_per_group;
    let b_inner_total = b_inner_rows_per_group
        .checked_mul(num_commitments)
        .ok_or_else(|| AkitaError::InvalidSetup("generate_y B_inner overflow".to_string()))?;
    let mut out = Vec::with_capacity(1 + n_d + commitment_rows.len() + b_inner_total + n_a);
    out.push(consistency_row);
    out.extend_from_slice(v);
    out.extend_from_slice(commitment_rows);
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), b_inner_total));
    out.extend_from_slice(a_rows);
    Ok(out)
}

/// Compute the stage-2 relation claim from the public M-row data.
///
/// This evaluates `sum_i eq(tau1, i) * y_alpha[i]` where `y_alpha` follows
/// the M row layout. Public openings bind through the fused trace term, not M
/// rows.
///
/// # Errors
///
/// Returns an error if the equality table implied by `tau1` would overflow or
/// exceed the verifier sequence bound.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows")]
pub fn relation_claim_from_rows<F: FieldCore + CanonicalField, const D: usize>(
    tau1: &[F],
    alpha: F,
    y: &[CyclotomicRing<F, D>],
) -> Result<F, AkitaError> {
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let mut acc = F::zero();
    for (row_idx, r) in y.iter().enumerate() {
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
    }
    Ok(acc)
}

/// Compute the stage-2 relation claim with an extension-field evaluation point.
///
/// Ring rows remain over `F`; their coefficients are multiplied into `E`
/// with mixed base-field scaling while evaluating at `alpha`.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows_extension")]
pub fn relation_claim_from_rows_extension<F, E, const D: usize>(
    tau1: &[E],
    alpha: E,
    y: &[CyclotomicRing<F, D>],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = E::zero();
    for (row_idx, r) in y.iter().enumerate() {
        if row_idx >= eq_tau1.len() {
            break;
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(r, &alpha_pows);
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr};

    type F = Fp32<251>;
    type E = FpExt2<F, NegOneNr>;

    #[test]
    fn lifted_relation_claim_matches_base_for_constant_alpha() {
        const D: usize = 4;
        let tau1 = [
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let alpha = F::from_u64(13);
        let v = [CyclotomicRing::from_coefficients([
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ])];
        let u = [CyclotomicRing::from_coefficients([
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ])];
        let mut y = vec![CyclotomicRing::<F, D>::zero()];
        y.extend_from_slice(&v);
        y.extend_from_slice(&u);

        let base = relation_claim_from_rows::<F, D>(&tau1, alpha, &y).unwrap();
        let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
        let lifted =
            relation_claim_from_rows_extension::<F, E, D>(&lifted_tau1, E::lift_base(alpha), &y)
                .unwrap();

        assert_eq!(lifted, E::lift_base(base));
    }
}
