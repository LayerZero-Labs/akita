//! Shared protocol relation helpers.

use crate::dispatch_for_field;
use crate::layout::relation::{RelationRowPlan, RelationRowRhs};
use crate::layout::RingRole;
use crate::proof::{RingVec, RingView};
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
#[cfg(test)]
use akita_algebra::ring::eval_ring_at;
use akita_algebra::ring::{eval_ring_at_pows_fast, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, MulBaseUnreduced};
use std::iter::repeat_n;

/// Build the RHS vector `y` matching the M row layout:
/// consistency (zero) | A (zeros) | B (`commitment_rows`) | D (`v`).
///
/// Public-output rows bind through the fused trace term, not `y`.
///
/// `commit_rows_per_group` is the B row count per commitment bundle
/// (`b_key.row_len()`). The number of commitment bundles is inferred from
/// `commitment_rows.len() / commit_rows_per_group`.
///
/// # Errors
///
/// Returns an error if the supplied row slices do not match the expected row
/// counts for the level layout.
#[cfg(test)]
fn generate_relation_rhs<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
    n_d: usize,
    commit_rows_per_group: usize,
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
    if commit_rows_per_group == 0
        || commitment_rows.is_empty()
        || !commitment_rows.len().is_multiple_of(commit_rows_per_group)
    {
        return Err(AkitaError::InvalidSize {
            expected: commit_rows_per_group,
            actual: commitment_rows.len(),
        });
    }
    let mut out = Vec::with_capacity(1 + n_a + commitment_rows.len() + n_d);
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    out.extend_from_slice(commitment_rows);
    out.extend_from_slice(v);
    Ok(out)
}

/// D-free assembly of `y` from per-role flat carriers (`v` at `d_d`, commitments at `d_b`).
///
/// Each segment is validated under its role dimension before concatenation.
/// The returned [`RingVec`] uses compact mode (`ring_dim = 0`); interpret segments
/// through each row family's scheduled native ring dimension when borrowing typed rows.
///
/// # Errors
///
/// Returns an error if segment lengths or role dimensions do not match `layout`.
pub fn assemble_relation_rhs<F: FieldCore>(
    layout: &RelationRowPlan,
    v: &RingVec<F>,
    commitment_rows: &RingVec<F>,
) -> Result<RingVec<F>, AkitaError> {
    let coeff_len = layout.rhs_coeff_len()?;
    let mut coeffs = Vec::with_capacity(coeff_len);
    let mut commit_offset = 0usize;
    let mut opening_offset = 0usize;
    for family in layout.families() {
        let family_coeffs = family
            .rows()
            .len()
            .checked_mul(family.native_ring_dim())
            .ok_or_else(|| AkitaError::InvalidSetup("relation RHS family overflow".into()))?;
        match family.rhs() {
            RelationRowRhs::Zero => coeffs.extend(repeat_n(F::zero(), family_coeffs)),
            RelationRowRhs::Commitment { .. } => {
                let end = commit_offset.checked_add(family_coeffs).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation commitment RHS offset overflow".into())
                })?;
                coeffs.extend_from_slice(
                    commitment_rows
                        .coeffs()
                        .get(commit_offset..end)
                        .ok_or(AkitaError::InvalidProof)?,
                );
                commit_offset = end;
            }
            RelationRowRhs::Opening => {
                let end = opening_offset.checked_add(family_coeffs).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation opening RHS offset overflow".into())
                })?;
                coeffs.extend_from_slice(
                    v.coeffs()
                        .get(opening_offset..end)
                        .ok_or(AkitaError::InvalidProof)?,
                );
                opening_offset = end;
            }
            RelationRowRhs::TerminalPayload { .. } => {
                return Err(AkitaError::InvalidSetup(
                    "base RHS assembly cannot supply compressed terminal payloads".into(),
                ));
            }
        }
    }
    if commit_offset != commitment_rows.coeff_len() || opening_offset != v.coeff_len() {
        return Err(AkitaError::InvalidSize {
            expected: commit_offset + opening_offset,
            actual: commitment_rows.coeff_len() + v.coeff_len(),
        });
    }
    Ok(RingVec::from_coeffs(coeffs))
}

fn accumulate_extension_rows<F, E, const D: usize>(
    eq_tau1: &[E],
    alpha: E,
    rows: &[CyclotomicRing<F, D>],
    row_idx: &mut usize,
    acc: &mut E,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBaseUnreduced<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    for r in rows {
        if *row_idx >= eq_tau1.len() {
            return Err(AkitaError::InvalidProof);
        }
        *acc += eq_tau1[*row_idx] * eval_ring_at_pows_fast(r, &alpha_pows);
        *row_idx += 1;
    }
    Ok(())
}

/// Compute the stage-2 relation claim from the public M-row data.
///
/// This evaluates `sum_i eq(tau1, i) * y_alpha[i]` where `y_alpha` follows
/// the M row layout: consistency zero row, A zero rows, B rows `u`, then D
/// rows `v`. Public openings bind through the fused trace term, not M rows.
///
/// # Errors
///
/// Returns an error if the equality table implied by `tau1` would overflow or
/// exceed the verifier sequence bound.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows")]
#[cfg(test)]
fn relation_claim_from_rows<F: FieldCore + CanonicalField, const D: usize>(
    tau1: &[F],
    alpha: F,
    n_a: usize,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
) -> Result<F, AkitaError> {
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let mut acc = F::zero();
    let mut row_idx = 1usize + n_a;

    for r in u {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at(r, &alpha);
        row_idx += 1;
    }
    Ok(acc)
}

/// Compute the stage-2 relation claim with an extension-field evaluation point.
///
/// Ring rows remain over `F`; their coefficients are multiplied into `E`
/// with mixed base-field scaling while evaluating at `alpha`.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows_extension")]
#[cfg(test)]
fn relation_claim_from_rows_extension<F, E, const D: usize>(
    tau1: &[E],
    alpha: E,
    n_a: usize,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBaseUnreduced<F>,
{
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = E::zero();
    let mut row_idx = 1usize + n_a;

    for r in u {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows_fast(r, &alpha_pows);
        row_idx += 1;
    }
    for r in v {
        if row_idx >= eq_tau1.len() {
            return Ok(acc);
        }
        acc += eq_tau1[row_idx] * eval_ring_at_pows_fast(r, &alpha_pows);
        row_idx += 1;
    }
    Ok(acc)
}

/// Per-role relation claim: `v` at `d_d`, commitment rows `u` at `d_b`.
///
/// Skips the consistency row at index 0 (always zero). Dispatches each segment
/// under its role dimension.
#[tracing::instrument(skip_all, name = "relation_claim_from_layout_extension")]
pub fn relation_claim_from_layout_extension<F, E>(
    layout: &RelationRowPlan,
    tau1: &[E],
    alpha: E,
    v: &RingVec<F>,
    u: &RingVec<F>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBaseUnreduced<F>,
{
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    if eq_tau1.len() < layout.trace_row() {
        return Err(AkitaError::InvalidProof);
    }
    let mut acc = E::zero();
    let mut commit_offset = 0usize;
    let mut opening_offset = 0usize;
    for family in layout.families() {
        let (carrier, offset, role) = match family.rhs() {
            RelationRowRhs::Commitment { .. } => (u, &mut commit_offset, RingRole::Outer),
            RelationRowRhs::Opening => (v, &mut opening_offset, RingRole::Opening),
            RelationRowRhs::Zero | RelationRowRhs::TerminalPayload { .. } => continue,
        };
        let coeff_len = family
            .rows()
            .len()
            .checked_mul(family.native_ring_dim())
            .ok_or_else(|| AkitaError::InvalidSetup("relation claim family overflow".into()))?;
        let end = offset
            .checked_add(coeff_len)
            .ok_or_else(|| AkitaError::InvalidSetup("relation claim offset overflow".into()))?;
        let coeffs = carrier
            .coeffs()
            .get(*offset..end)
            .ok_or(AkitaError::InvalidProof)?;
        let view = RingView::new(coeffs, family.native_ring_dim())?;
        match role {
            RingRole::Outer => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                F,
                family.native_ring_dim(),
                |D| {
                    let mut row_idx = family.rows().start();
                    accumulate_extension_rows::<F, E, D>(
                        &eq_tau1,
                        alpha,
                        view.as_ring_slice::<D>()?,
                        &mut row_idx,
                        &mut acc,
                    )
                }
            )?,
            RingRole::Opening => dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Opening),
                F,
                family.native_ring_dim(),
                |D| {
                    let mut row_idx = family.rows().start();
                    accumulate_extension_rows::<F, E, D>(
                        &eq_tau1,
                        alpha,
                        view.as_ring_slice::<D>()?,
                        &mut row_idx,
                        &mut acc,
                    )
                }
            )?,
            _ => return Err(AkitaError::InvalidSetup("invalid RHS ring role".into())),
        }
        *offset = end;
    }
    if commit_offset != u.coeff_len() || opening_offset != v.coeff_len() {
        return Err(AkitaError::InvalidSize {
            expected: commit_offset + opening_offset,
            actual: u.coeff_len() + v.coeff_len(),
        });
    }
    Ok(acc)
}

/// Row-index weight for the trailing EvaluationTrace row: `eq(row_index, last)`.
///
/// Fold paths combine this with `relation_claim_from_layout_extension` as
/// `relation_claim + weight * trace_eval_target` (and reuse `weight` for
/// Stage-2 `TraceClaim::trace_coeff`).
pub fn evaluation_trace_row_weight<E: FieldCore>(
    evaluation_trace_row: usize,
    tau1: &[E],
) -> Result<E, AkitaError> {
    let num_vars = tau1.len();
    if num_vars >= usize::BITS as usize {
        return Err(AkitaError::InvalidSize {
            expected: (usize::BITS as usize).saturating_sub(1),
            actual: num_vars,
        });
    }
    let domain_size = 1usize
        .checked_shl(num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("tau1 row-index domain overflow".to_string()))?;
    if evaluation_trace_row >= domain_size {
        return Err(AkitaError::InvalidSize {
            expected: domain_size,
            actual: evaluation_trace_row.saturating_add(1),
        });
    }
    Ok(eq_eval_at_index(tau1, evaluation_trace_row))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr};

    type F = Fp32<251>;
    type E = FpExt2<F, NegOneNr>;

    fn test_layout<const D: usize>(n_a: usize, n_b: usize, n_d: usize) -> crate::RelationLayout {
        let lp = crate::LevelParams::params_only(
            crate::SisModulusFamily::Q32,
            D,
            2,
            n_a,
            n_b,
            n_d,
            SparseChallengeConfig::pm1_only(D),
        )
        .with_decomp(1, 1, 1, 1, 0)
        .unwrap();
        let opening = crate::OpeningClaimsLayout::new(1, 1).unwrap();
        crate::RelationLayout::from_authenticated_statement(
            &lp,
            &opening,
            crate::RelationMatrixRowLayout::WithDBlock,
            lp.field_bits_for_cache(),
        )
        .unwrap()
    }

    #[test]
    fn lifted_relation_claim_matches_base_for_constant_alpha() {
        const D: usize = 4;
        const N_A: usize = 1;
        let tau1 = [
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
        ];
        let alpha = F::from_u64(17);
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

        let base = relation_claim_from_rows::<F, D>(&tau1, alpha, N_A, &v, &u).unwrap();
        let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
        let lifted = relation_claim_from_rows_extension::<F, E, D>(
            &lifted_tau1,
            E::lift_base(alpha),
            N_A,
            &v,
            &u,
        )
        .unwrap();

        assert_eq!(lifted, E::lift_base(base));
    }

    #[test]
    fn relation_claim_at_dims_matches_uniform_single_d() {
        const D: usize = 64;
        let tau1 = [
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let alpha = F::from_u64(13);
        let mut v_coeffs = [F::zero(); D];
        v_coeffs[..4].copy_from_slice(&[
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let mut u_coeffs = [F::zero(); D];
        u_coeffs[..4].copy_from_slice(&[
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ]);
        let v = [CyclotomicRing::from_coefficients(v_coeffs)];
        let u = [CyclotomicRing::from_coefficients(u_coeffs)];
        let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
        const N_A: usize = 1;
        let layout = test_layout::<D>(N_A, 1, 1);
        let at_dims = relation_claim_from_layout_extension::<F, E>(
            layout.row_plan(),
            &lifted_tau1,
            E::lift_base(alpha),
            &RingVec::from_ring_elems(&v),
            &RingVec::from_ring_elems(&u),
        )
        .unwrap();
        let monolithic = relation_claim_from_rows_extension::<F, E, D>(
            &lifted_tau1,
            E::lift_base(alpha),
            N_A,
            &v,
            &u,
        )
        .unwrap();
        assert_eq!(at_dims, monolithic);
    }

    #[test]
    fn relation_claim_from_layout_rejects_short_equality_domain() {
        const D: usize = 4;
        let layout = test_layout::<D>(1, 1, 1);
        let one = [CyclotomicRing::<F, D>::one()];
        let err = relation_claim_from_layout_extension::<F, E>(
            layout.row_plan(),
            &[],
            E::one(),
            &RingVec::from_ring_elems(&one),
            &RingVec::from_ring_elems(&one),
        )
        .unwrap_err();
        assert!(matches!(err, AkitaError::InvalidProof));
    }

    #[test]
    fn assemble_relation_rhs_matches_generate_rhs_for_uniform_dims() {
        const D: usize = 4;
        let v = [CyclotomicRing::from_coefficients([
            F::from_u64(1),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
        ])];
        let u = [CyclotomicRing::from_coefficients([
            F::from_u64(2),
            F::from_u64(0),
            F::from_u64(0),
            F::from_u64(0),
        ])];
        let layout = test_layout::<D>(2, 1, 1);
        let typed = generate_relation_rhs::<F, D>(&v, &u, 1, 1, 2).unwrap();
        let assembled = assemble_relation_rhs::<F>(
            layout.row_plan(),
            &RingVec::from_ring_elems(&v),
            &RingVec::from_ring_elems(&u),
        )
        .unwrap();
        assert_eq!(
            assembled.coeffs(),
            RingVec::from_ring_elems(&typed).coeffs()
        );
    }

    #[test]
    fn evaluation_trace_row_weight_uses_last_row() {
        // total_row_count = 4 → 2 row-index vars; eq table length 4.
        let tau1 = [F::from_u64(2), F::from_u64(3)];
        let weight = evaluation_trace_row_weight(3, &tau1).unwrap();
        assert_eq!(weight, eq_eval_at_index(&tau1, 3));
        assert_ne!(weight, eq_eval_at_index(&tau1, 0));
    }

    #[test]
    fn evaluation_trace_row_weight_rejects_out_of_domain_index() {
        let tau1 = [F::from_u64(2), F::from_u64(3)];
        assert!(evaluation_trace_row_weight(4, &tau1).is_err());
    }

    #[test]
    fn fused_relation_claim_matches_full_logical_row_evaluation() {
        const D: usize = 64;
        let tau1 = [
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(7),
            F::from_u64(11),
        ];
        let alpha = F::from_u64(13);
        let mut v_coeffs = [F::zero(); D];
        v_coeffs[..4].copy_from_slice(&[
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let mut u_coeffs = [F::zero(); D];
        u_coeffs[..4].copy_from_slice(&[
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ]);
        let v = [CyclotomicRing::from_coefficients(v_coeffs)];
        let u = [CyclotomicRing::from_coefficients(u_coeffs)];
        let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
        const N_A: usize = 1;
        let layout = test_layout::<D>(N_A, 1, 1);
        let evaluation_trace_row = layout.row_plan().trace_row();
        let trace_target = E::from_u64(19);
        let quotient_claim = relation_claim_from_layout_extension::<F, E>(
            layout.row_plan(),
            &lifted_tau1,
            E::lift_base(alpha),
            &RingVec::from_ring_elems(&v),
            &RingVec::from_ring_elems(&u),
        )
        .unwrap();
        let weight = evaluation_trace_row_weight(evaluation_trace_row, &lifted_tau1).unwrap();
        let fused = quotient_claim + weight * trace_target;

        let alpha_pows = scalar_powers(E::lift_base(alpha), D);
        let padded_domain = 1usize << lifted_tau1.len();
        let mut y_alpha = vec![E::zero(); padded_domain];
        let mut row_idx = 1usize + N_A;
        for ring in &u {
            y_alpha[row_idx] = eval_ring_at_pows_fast(ring, &alpha_pows);
            row_idx += 1;
        }
        for ring in &v {
            y_alpha[row_idx] = eval_ring_at_pows_fast(ring, &alpha_pows);
            row_idx += 1;
        }
        y_alpha[evaluation_trace_row] = trace_target;

        let mut independent = E::zero();
        for (row, value) in y_alpha.iter().enumerate() {
            independent += eq_eval_at_index(&lifted_tau1, row) * *value;
        }
        assert_eq!(fused, independent);
    }
}
