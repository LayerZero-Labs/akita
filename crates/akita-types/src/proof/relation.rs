//! Shared protocol relation helpers.

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at, eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, MulBase};

/// Compute the stage-2 relation claim from the public M-row data.
///
/// This evaluates `sum_i eq(tau1, i) * y_alpha[i]` where `y_alpha` follows
/// the M row layout: consistency zero row, public `y_rings`, D rows `v`, B
/// rows `u`, then A zero rows.
///
/// # Errors
///
/// Returns an error if the equality table implied by `tau1` would overflow or
/// exceed the verifier sequence bound.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows")]
pub fn relation_claim_from_rows<F: FieldCore + CanonicalField, const D: usize>(
    tau1: &[F],
    alpha: F,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
) -> Result<F, AkitaError> {
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let mut acc = F::zero();
    let mut row_idx = 1usize;

    for y_ring in y_rings {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at(y_ring, &alpha);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    for r in u {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    Ok(acc)
}

/// Tiered variant of [`relation_claim_from_rows_extension`] for the tiered
/// root M-row layout (`specs/tiered_commit.md` §3):
///
/// ```text
/// consistency (1) | public | D (n_d) | tier1 (split·n_b'·num_points)
///   | F (n_F·num_points) | A (n_a)
/// ```
///
/// Tier-1 rows have `y = 0` and so must be SKIPPED (advancing `row_idx`)
/// rather than contributing to the accumulator. `u_final_rows` lives at
/// the F-row positions, AFTER the tier-1 block.
///
/// The legacy [`relation_claim_from_rows_extension`] iterates `u` directly
/// after `v` and would place `u_final` at the tier-1 zero-row positions,
/// producing a relation claim that disagrees with the actual
/// `Σ_row eq_tau1[row] · y[row]` the sumcheck instance reconstructs from
/// the witness. See Phase 4f-sumcheck.
///
/// # Errors
///
/// Returns an error if the equality table implied by `tau1` would overflow
/// (matches the surrounding [`relation_claim_from_rows_extension`]
/// contract after the security-hardening pass).
#[tracing::instrument(skip_all, name = "relation_claim_from_rows_extension_tiered")]
pub fn relation_claim_from_rows_extension_tiered<F, E, const D: usize>(
    tau1: &[E],
    alpha: E,
    v: &[CyclotomicRing<F, D>],
    u_final_rows: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
    tier1_zero_rows: usize,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = E::zero();
    let mut row_idx = 1usize;

    for y_ring in y_rings {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(y_ring, &alpha_pows);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(r, &alpha_pows);
        row_idx += 1;
    }
    // Tier-1 rows have `y = 0`. Advance row_idx past them with no
    // accumulator contribution.
    row_idx = row_idx.saturating_add(tier1_zero_rows);
    for r in u_final_rows {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(r, &alpha_pows);
        row_idx += 1;
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
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
    y_rings: &[CyclotomicRing<F, D>],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = E::zero();
    let mut row_idx = 1usize;

    for y_ring in y_rings {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(y_ring, &alpha_pows);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(r, &alpha_pows);
        row_idx += 1;
    }
    for r in u {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows(r, &alpha_pows);
        row_idx += 1;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{Fp2, Fp32, LiftBase, NegOneNr};

    type F = Fp32<251>;
    type E = Fp2<F, NegOneNr>;

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
        let y = [CyclotomicRing::from_coefficients([
            F::from_u64(9),
            F::from_u64(10),
            F::from_u64(11),
            F::from_u64(12),
        ])];

        let base = relation_claim_from_rows::<F, D>(&tau1, alpha, &v, &u, &y).unwrap();
        let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
        let lifted = relation_claim_from_rows_extension::<F, E, D>(
            &lifted_tau1,
            E::lift_base(alpha),
            &v,
            &u,
            &y,
        )
        .unwrap();

        assert_eq!(lifted, E::lift_base(base));
    }
}
