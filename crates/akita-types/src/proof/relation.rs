//! Shared protocol relation helpers.

use crate::dispatch_ring_dim_result;
use crate::proof::RingVec;
use crate::schedule_context::CommitmentRingDims;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::ring::{eval_ring_at, eval_ring_at_pows, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, MulBase};
use std::iter::repeat_n;

/// Row-count inputs for assembling the relation RHS vector `y`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationYLayout {
    pub n_d: usize,
    pub commit_rows_per_group: usize,
    pub b_inner_rows_per_group: usize,
    pub n_a: usize,
}

/// Logical M-row count encoded in assembled relation `y`.
///
/// Layout: consistency (1) | D (`n_d`) | COMMIT | B_inner | A (`n_a`).
#[must_use]
pub fn relation_y_row_count(layout: RelationYLayout, num_commitment_groups: usize) -> usize {
    let commit_rows = layout
        .commit_rows_per_group
        .saturating_mul(num_commitment_groups);
    let b_inner_total = layout
        .b_inner_rows_per_group
        .saturating_mul(num_commitment_groups);
    1 + layout.n_d + commit_rows + b_inner_total + layout.n_a
}

/// Expected flat coefficient length of assembled `y` under per-role dimensions.
///
/// # Errors
///
/// Returns an error if any segment length arithmetic overflows.
pub fn relation_y_coeff_len(
    dims: CommitmentRingDims,
    layout: RelationYLayout,
    num_commitment_groups: usize,
) -> Result<usize, AkitaError> {
    let commit_rows = layout
        .commit_rows_per_group
        .checked_mul(num_commitment_groups)
        .ok_or_else(|| AkitaError::InvalidSetup("relation y commit row count overflow".into()))?;
    let b_inner_total = layout
        .b_inner_rows_per_group
        .checked_mul(num_commitment_groups)
        .ok_or_else(|| AkitaError::InvalidSetup("relation y B_inner row count overflow".into()))?;
    let d_segment = layout
        .n_d
        .checked_mul(dims.d_d())
        .ok_or_else(|| AkitaError::InvalidSetup("relation y D segment overflow".into()))?;
    let commit_segment = commit_rows
        .checked_mul(dims.d_b())
        .ok_or_else(|| AkitaError::InvalidSetup("relation y COMMIT segment overflow".into()))?;
    let b_inner_segment = b_inner_total
        .checked_mul(dims.d_b())
        .ok_or_else(|| AkitaError::InvalidSetup("relation y B_inner segment overflow".into()))?;
    let a_segment = layout
        .n_a
        .checked_mul(dims.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("relation y A segment overflow".into()))?;
    dims.d_a()
        .checked_add(d_segment)
        .and_then(|len| len.checked_add(commit_segment))
        .and_then(|len| len.checked_add(b_inner_segment))
        .and_then(|len| len.checked_add(a_segment))
        .ok_or_else(|| AkitaError::InvalidSetup("relation y coefficient length overflow".into()))
}

/// Number of ring rows decodable at role dimension `d` (compact or tagged storage).
fn ring_row_count_at<F: FieldCore>(vec: &RingVec<F>, d: usize) -> Result<usize, AkitaError> {
    if vec.coeff_len() == 0 {
        return Ok(0);
    }
    if !vec.can_decode_vec(d) {
        return Err(AkitaError::InvalidSize {
            expected: d,
            actual: vec.coeff_len(),
        });
    }
    Ok(vec.coeff_len() / d)
}

/// Build the RHS vector `y` matching the M row layout:
/// consistency (zero) | D (`v`) | COMMIT (`commitment_rows`) | B_inner (zeros) | A (zeros).
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
pub fn generate_y<F, const D: usize>(
    v: &[CyclotomicRing<F, D>],
    commitment_rows: &[CyclotomicRing<F, D>],
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
    out.push(CyclotomicRing::<F, D>::zero());
    out.extend_from_slice(v);
    out.extend_from_slice(commitment_rows);
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), b_inner_total));
    out.extend(repeat_n(CyclotomicRing::<F, D>::zero(), n_a));
    Ok(out)
}

/// D-free assembly of `y` from per-role flat carriers (`v` at `d_d`, commitments at `d_b`).
///
/// Each segment is validated under its role dimension before concatenation.
/// The returned [`RingVec`] uses compact mode (`ring_dim = 0`); interpret segments
/// through [`CommitmentRingDims`] when borrowing typed rows.
///
/// # Errors
///
/// Returns an error if segment lengths or role dimensions do not match `layout`.
pub fn assemble_relation_y<F: FieldCore>(
    dims: CommitmentRingDims,
    layout: RelationYLayout,
    v: &RingVec<F>,
    commitment_rows: &RingVec<F>,
) -> Result<RingVec<F>, AkitaError> {
    let RelationYLayout {
        n_d,
        commit_rows_per_group,
        b_inner_rows_per_group,
        n_a,
    } = layout;
    let v_rows = ring_row_count_at(v, dims.d_d())?;
    if v_rows != n_d {
        return Err(AkitaError::InvalidSize {
            expected: n_d,
            actual: v_rows,
        });
    }
    let commit_rows = ring_row_count_at(commitment_rows, dims.d_b())?;
    if commit_rows_per_group == 0
        || commit_rows == 0
        || !commit_rows.is_multiple_of(commit_rows_per_group)
    {
        return Err(AkitaError::InvalidSize {
            expected: commit_rows_per_group,
            actual: commit_rows,
        });
    }
    let num_commitments = commit_rows / commit_rows_per_group;
    let b_inner_total = b_inner_rows_per_group
        .checked_mul(num_commitments)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("assemble_relation_y B_inner overflow".to_string())
        })?;
    let mut coeffs = Vec::with_capacity(
        dims.d_a()
            + n_d * dims.d_d()
            + commitment_rows.coeff_len()
            + b_inner_total * dims.d_b()
            + n_a * dims.d_a(),
    );
    coeffs.extend(repeat_n(F::zero(), dims.d_a()));
    coeffs.extend_from_slice(v.coeffs());
    coeffs.extend_from_slice(commitment_rows.coeffs());
    coeffs.extend(repeat_n(F::zero(), b_inner_total * dims.d_b()));
    coeffs.extend(repeat_n(F::zero(), n_a * dims.d_a()));
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
    E: FieldCore + MulBase<F>,
{
    let alpha_pows = scalar_powers(alpha, D);
    for r in rows {
        if *row_idx >= eq_tau1.len() {
            return Ok(());
        }
        *acc += eq_tau1[*row_idx] * eval_ring_at_pows(r, &alpha_pows);
        *row_idx += 1;
    }
    Ok(())
}

/// Compute the stage-2 relation claim from the public M-row data.
///
/// This evaluates `sum_i eq(tau1, i) * y_alpha[i]` where `y_alpha` follows
/// the M row layout: consistency zero row, D rows `v`, B rows `u`, then A zero
/// rows. Public openings bind through the fused trace term, not M rows.
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
) -> Result<F, AkitaError> {
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let mut acc = F::zero();
    let mut row_idx = 1usize;

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
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let alpha_pows = scalar_powers(alpha, D);
    let mut acc = E::zero();
    let mut row_idx = 1usize;

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

/// Per-role relation claim: `v` at `d_d`, commitment rows `u` at `d_b`.
///
/// Skips the consistency row at index 0 (always zero). Dispatches each segment
/// under its role dimension.
#[tracing::instrument(skip_all, name = "relation_claim_from_rows_extension_at_dims")]
pub fn relation_claim_from_rows_extension_at_dims<F, E>(
    dims: CommitmentRingDims,
    tau1: &[E],
    alpha: E,
    v: &RingVec<F>,
    u: &RingVec<F>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBase<F>,
{
    if !v.can_decode_vec(dims.d_d()) {
        return Err(AkitaError::InvalidSize {
            expected: dims.d_d(),
            actual: v.coeff_len(),
        });
    }
    if !u.can_decode_vec(dims.d_b()) {
        return Err(AkitaError::InvalidSize {
            expected: dims.d_b(),
            actual: u.coeff_len(),
        });
    }
    let eq_tau1 = EqPolynomial::evals(tau1)?;
    let mut acc = E::zero();
    let mut row_idx = 1usize;
    dispatch_ring_dim_result!(dims.d_d(), |D_D| {
        let v_typed = v.as_ring_slice::<D_D>()?;
        accumulate_extension_rows::<F, E, D_D>(&eq_tau1, alpha, v_typed, &mut row_idx, &mut acc)
    })?;
    dispatch_ring_dim_result!(dims.d_b(), |D_B| {
        let u_typed = u.as_ring_slice::<D_B>()?;
        accumulate_extension_rows::<F, E, D_B>(&eq_tau1, alpha, u_typed, &mut row_idx, &mut acc)
    })?;
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

        let base = relation_claim_from_rows::<F, D>(&tau1, alpha, &v, &u).unwrap();
        let lifted_tau1: Vec<E> = tau1.iter().copied().map(E::lift_base).collect();
        let lifted = relation_claim_from_rows_extension::<F, E, D>(
            &lifted_tau1,
            E::lift_base(alpha),
            &v,
            &u,
        )
        .unwrap();

        assert_eq!(lifted, E::lift_base(base));
    }

    #[test]
    fn relation_claim_at_dims_matches_uniform_single_d() {
        const D: usize = 32;
        let dims = CommitmentRingDims::uniform(D);
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
        let at_dims = relation_claim_from_rows_extension_at_dims::<F, E>(
            dims,
            &lifted_tau1,
            E::lift_base(alpha),
            &RingVec::from_ring_elems(&v),
            &RingVec::from_ring_elems(&u),
        )
        .unwrap();
        let monolithic = relation_claim_from_rows_extension::<F, E, D>(
            &lifted_tau1,
            E::lift_base(alpha),
            &v,
            &u,
        )
        .unwrap();
        assert_eq!(at_dims, monolithic);
    }

    #[test]
    fn assemble_relation_y_matches_generate_y_for_uniform_dims() {
        const D: usize = 4;
        let dims = CommitmentRingDims::uniform(D);
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
        let layout = RelationYLayout {
            n_d: 1,
            commit_rows_per_group: 1,
            b_inner_rows_per_group: 0,
            n_a: 2,
        };
        let typed = generate_y::<F, D>(&v, &u, layout.n_d, 1, 0, layout.n_a).unwrap();
        let assembled = assemble_relation_y::<F>(
            dims,
            layout,
            &RingVec::from_ring_elems(&v),
            &RingVec::from_ring_elems(&u),
        )
        .unwrap();
        assert_eq!(
            assembled.coeffs(),
            RingVec::from_ring_elems(&typed).coeffs()
        );
    }
}
