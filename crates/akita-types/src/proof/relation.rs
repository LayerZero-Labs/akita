//! Shared protocol relation helpers.

use crate::dispatch_for_field;
use crate::layout::CommitmentRingDims;
use crate::layout::{LevelParams, RelationMatrixRowLayout};
use crate::opening_claims::OpeningClaimsLayout;
use crate::proof::RingVec;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::{eval_ring_at, eval_ring_at_pows_fast, scalar_powers};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, MulBaseUnreduced};
use std::iter::repeat_n;

/// Per-group row-count inputs for assembling the relation rhs vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationGroupRows {
    pub n_a: usize,
    pub commit_rows: usize,
    pub b_inner_rows: usize,
}

/// Row-count inputs for assembling the relation rhs vector.
///
/// relation-matrix row order: `[final, precommitted_0, .., precommitted_{G-2}]`.
/// `groups.len() == 1` reproduces the historical scalar layout byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationRhsLayout {
    pub n_d: usize,
    pub groups: Vec<RelationGroupRows>,
}

impl RelationRhsLayout {
    #[must_use]
    pub fn uniform(
        n_d: usize,
        n_a: usize,
        commit_rows_per_group: usize,
        b_inner_rows_per_group: usize,
        num_groups: usize,
    ) -> Self {
        Self {
            n_d,
            groups: repeat_n(
                RelationGroupRows {
                    n_a,
                    commit_rows: commit_rows_per_group,
                    b_inner_rows: b_inner_rows_per_group,
                },
                num_groups,
            )
            .collect(),
        }
    }
}

/// Single source of truth for the relation rhs row layout at one level.
///
/// # Errors
///
/// Returns an error if the opening batch is malformed for multi-group root params.
pub fn relation_rhs_layout_for(
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    relation_matrix_row_layout: RelationMatrixRowLayout,
) -> Result<RelationRhsLayout, AkitaError> {
    opening_batch.check()?;
    let n_d = lp.n_d_active_for(relation_matrix_row_layout);
    if !lp.has_precommitted_groups() {
        return Ok(RelationRhsLayout::uniform(
            n_d,
            lp.a_key.row_len(),
            lp.b_key.row_len(),
            0,
            opening_batch.num_groups(),
        ));
    }
    lp.validate_root_opening_batch(opening_batch)?;
    let mut groups = Vec::with_capacity(lp.precommitted_groups.len() + 1);
    groups.push(RelationGroupRows {
        n_a: lp.a_key.row_len(),
        commit_rows: lp.b_key.row_len(),
        b_inner_rows: 0,
    });
    for group in &lp.precommitted_groups {
        groups.push(RelationGroupRows {
            n_a: group.a_key.row_len(),
            commit_rows: group.b_key.row_len(),
            b_inner_rows: 0,
        });
    }
    Ok(RelationRhsLayout { n_d, groups })
}

/// Logical relation-matrix row count encoded in assembled relation rhs.
///
/// Layout: consistency (1) | [A_g | B_g | B_inner_g]_g | D (`n_d`).
#[must_use]
pub fn relation_rhs_row_count(layout: &RelationRhsLayout) -> usize {
    let group_rows = layout.groups.iter().fold(0usize, |acc, group| {
        acc.saturating_add(group.n_a)
            .saturating_add(group.commit_rows)
            .saturating_add(group.b_inner_rows)
    });
    1usize.saturating_add(group_rows).saturating_add(layout.n_d)
}

/// Expected flat coefficient length of assembled `y` under per-role dimensions.
///
/// # Errors
///
/// Returns an error if any segment length arithmetic overflows.
pub fn relation_rhs_coeff_len(
    dims: CommitmentRingDims,
    layout: &RelationRhsLayout,
) -> Result<usize, AkitaError> {
    let mut a_rows = 0usize;
    let mut commit_rows = 0usize;
    let mut b_inner_total = 0usize;
    for group in &layout.groups {
        a_rows = a_rows
            .checked_add(group.n_a)
            .ok_or_else(|| AkitaError::InvalidSetup("relation y A row count overflow".into()))?;
        commit_rows = commit_rows.checked_add(group.commit_rows).ok_or_else(|| {
            AkitaError::InvalidSetup("relation y commit row count overflow".into())
        })?;
        b_inner_total = b_inner_total
            .checked_add(group.b_inner_rows)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation y B_inner row count overflow".into())
            })?;
    }
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
    let a_segment = a_rows
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
pub fn generate_relation_rhs<F, const D: usize>(
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
/// through [`CommitmentRingDims`] when borrowing typed rows.
///
/// # Errors
///
/// Returns an error if segment lengths or role dimensions do not match `layout`.
pub fn assemble_relation_rhs<F: FieldCore>(
    dims: CommitmentRingDims,
    layout: &RelationRhsLayout,
    v: &RingVec<F>,
    commitment_rows: &RingVec<F>,
) -> Result<RingVec<F>, AkitaError> {
    let v_rows = ring_row_count_at(v, dims.d_d())?;
    if v_rows != layout.n_d {
        return Err(AkitaError::InvalidSize {
            expected: layout.n_d,
            actual: v_rows,
        });
    }
    let expected_commit_rows = layout.groups.iter().try_fold(0usize, |acc, group| {
        acc.checked_add(group.commit_rows).ok_or_else(|| {
            AkitaError::InvalidSetup("assemble_relation_rhs commit rows overflow".into())
        })
    })?;
    let commit_rows = ring_row_count_at(commitment_rows, dims.d_b())?;
    if commit_rows != expected_commit_rows {
        return Err(AkitaError::InvalidSize {
            expected: expected_commit_rows,
            actual: commit_rows,
        });
    }
    let coeff_len = relation_rhs_coeff_len(dims, layout)?;
    let mut coeffs = Vec::with_capacity(coeff_len);
    coeffs.extend(repeat_n(F::zero(), dims.d_a()));
    let mut commit_offset = 0usize;
    for group in &layout.groups {
        coeffs.extend(repeat_n(F::zero(), group.n_a * dims.d_a()));
        let commit_coeff_len = group.commit_rows.checked_mul(dims.d_b()).ok_or_else(|| {
            AkitaError::InvalidSetup("assemble_relation_rhs B segment overflow".into())
        })?;
        let commit_end = commit_offset.checked_add(commit_coeff_len).ok_or_else(|| {
            AkitaError::InvalidSetup("assemble_relation_rhs B offset overflow".into())
        })?;
        let rows = commitment_rows
            .coeffs()
            .get(commit_offset..commit_end)
            .ok_or(AkitaError::InvalidProof)?;
        coeffs.extend_from_slice(rows);
        coeffs.extend(repeat_n(F::zero(), group.b_inner_rows * dims.d_b()));
        commit_offset = commit_end;
    }
    coeffs.extend_from_slice(v.coeffs());
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
            return Ok(());
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
pub fn relation_claim_from_rows<F: FieldCore + CanonicalField, const D: usize>(
    tau1: &[F],
    alpha: F,
    n_a: usize,
    v: &[CyclotomicRing<F, D>],
    u: &[CyclotomicRing<F, D>],
) -> Result<F, AkitaError> {
    let row_count = 1usize
        .checked_add(n_a)
        .and_then(|count| count.checked_add(u.len()))
        .and_then(|count| count.checked_add(v.len()))
        .ok_or_else(|| AkitaError::InvalidSetup("relation row count overflow".into()))?;
    let eq_tau1 = EqPolynomial::evals_prefix(tau1, row_count)?;
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
pub fn relation_claim_from_rows_extension<F, E, const D: usize>(
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
    let row_count = 1usize
        .checked_add(n_a)
        .and_then(|count| count.checked_add(u.len()))
        .and_then(|count| count.checked_add(v.len()))
        .ok_or_else(|| AkitaError::InvalidSetup("relation row count overflow".into()))?;
    let eq_tau1 = EqPolynomial::evals_prefix(tau1, row_count)?;
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
    dims: CommitmentRingDims,
    layout: &RelationRhsLayout,
    tau1: &[E],
    alpha: E,
    v: &RingVec<F>,
    u: &RingVec<F>,
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBaseUnreduced<F>,
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
    let expected_u_rows = layout.groups.iter().try_fold(0usize, |acc, group| {
        acc.checked_add(group.commit_rows)
            .ok_or_else(|| AkitaError::InvalidSetup("relation claim commit rows overflow".into()))
    })?;
    if u.coeff_len() / dims.d_b() != expected_u_rows {
        return Err(AkitaError::InvalidSize {
            expected: expected_u_rows,
            actual: u.coeff_len() / dims.d_b(),
        });
    }
    if v.coeff_len() / dims.d_d() != layout.n_d {
        return Err(AkitaError::InvalidSize {
            expected: layout.n_d,
            actual: v.coeff_len() / dims.d_d(),
        });
    }
    let row_count = 1usize
        .checked_add(layout.n_d)
        .and_then(|count| {
            layout.groups.iter().try_fold(count, |count, group| {
                count
                    .checked_add(group.n_a)
                    .and_then(|count| count.checked_add(group.commit_rows))
                    .and_then(|count| count.checked_add(group.b_inner_rows))
            })
        })
        .ok_or_else(|| AkitaError::InvalidSetup("relation row count overflow".into()))?;
    let eq_tau1 = EqPolynomial::evals_prefix(tau1, row_count)?;
    let mut acc = E::zero();
    let mut row_idx = 1usize;
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Outer),
        F,
        dims.d_b(),
        |D_B| {
            let u_typed = u.as_ring_slice::<D_B>()?;
            let mut commit_offset = 0usize;
            for group in &layout.groups {
                row_idx = row_idx.checked_add(group.n_a).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim row index overflow".into())
                })?;
                let commit_end = commit_offset
                    .checked_add(group.commit_rows)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation claim commit offset overflow".into())
                    })?;
                let rows = u_typed
                    .get(commit_offset..commit_end)
                    .ok_or(AkitaError::InvalidProof)?;
                accumulate_extension_rows::<F, E, D_B>(
                    &eq_tau1,
                    alpha,
                    rows,
                    &mut row_idx,
                    &mut acc,
                )?;
                row_idx = row_idx.checked_add(group.b_inner_rows).ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim row index overflow".into())
                })?;
                commit_offset = commit_end;
            }
            Ok::<(), AkitaError>(())
        }
    )?;
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        F,
        dims.d_d(),
        |D_D| {
            let v_typed = v.as_ring_slice::<D_D>()?;
            accumulate_extension_rows::<F, E, D_D>(&eq_tau1, alpha, v_typed, &mut row_idx, &mut acc)
        }
    )?;
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
    use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr};

    type F = Fp32<251>;
    type E = FpExt2<F, NegOneNr>;

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
        const N_A: usize = 1;
        let layout = RelationRhsLayout::uniform(1, N_A, 1, 0, 1);
        let at_dims = relation_claim_from_layout_extension::<F, E>(
            dims,
            &layout,
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
    fn assemble_relation_rhs_matches_generate_rhs_for_uniform_dims() {
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
        let layout = RelationRhsLayout::uniform(1, 2, 1, 0, 1);
        let typed =
            generate_relation_rhs::<F, D>(&v, &u, layout.n_d, 1, layout.groups[0].n_a).unwrap();
        let assembled = assemble_relation_rhs::<F>(
            dims,
            &layout,
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
    fn nested_role_dims_relation_rhs_coeff_len_matches_per_segment_widths() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        assert!(dims.nests());
        let layout = RelationRhsLayout::uniform(2, 4, 3, 1, 1);
        let coeff_len = relation_rhs_coeff_len(dims, &layout).expect("coeff len");
        let expected = 128 + 2 * 32 + 3 * 64 + 64 + 4 * 128;
        assert_eq!(coeff_len, expected);
        assert_eq!(relation_rhs_row_count(&layout), 1 + 2 + 3 + 1 + 4);
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
        const N_A: usize = 1;
        let layout = RelationRhsLayout::uniform(1, N_A, 1, 0, 1);
        let evaluation_trace_row = relation_rhs_row_count(&layout);
        let trace_target = E::from_u64(19);
        let quotient_claim = relation_claim_from_layout_extension::<F, E>(
            dims,
            &layout,
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
