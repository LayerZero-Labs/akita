//! Shared protocol relation helpers.

use crate::dispatch_for_field;
use crate::layout::{CommitmentRingDims, CommittedGroupParams};
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
    /// This group's A/B dimensions completed by the level-shared D dimension.
    pub role_dims: CommitmentRingDims,
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
    /// D dimension owned by the consuming level and shared by every group.
    pub opening_ring_dim: usize,
    pub n_d: usize,
    pub groups: Vec<RelationGroupRows>,
}

impl RelationRhsLayout {
    #[must_use]
    pub fn uniform(
        role_dims: CommitmentRingDims,
        n_d: usize,
        n_a: usize,
        commit_rows_per_group: usize,
        b_inner_rows_per_group: usize,
        num_groups: usize,
    ) -> Self {
        Self {
            opening_ring_dim: role_dims.d_d(),
            n_d,
            groups: repeat_n(
                RelationGroupRows {
                    role_dims,
                    n_a,
                    commit_rows: commit_rows_per_group,
                    b_inner_rows: b_inner_rows_per_group,
                },
                num_groups,
            )
            .collect(),
        }
    }

    fn validate(&self) -> Result<(), AkitaError> {
        if self.groups.is_empty() || self.opening_ring_dim == 0 {
            return Err(AkitaError::InvalidSetup(
                "relation rhs layout requires non-empty group and ring geometry".into(),
            ));
        }
        for group in &self.groups {
            group.role_dims.validate_role_projection()?;
            if group.role_dims.d_d() != self.opening_ring_dim {
                return Err(AkitaError::InvalidSetup(
                    "relation rhs groups disagree with the level-shared D dimension".into(),
                ));
            }
        }
        Ok(())
    }

    /// Ring dimension of every physical relation-quotient row, in canonical
    /// relation-matrix order.
    ///
    /// Each group contributes one native-A consistency row, its native A rows,
    /// and its native B rows. The trailing D rows use the level-shared opening
    /// dimension.
    pub fn row_ring_dims(&self) -> Result<Vec<usize>, AkitaError> {
        self.validate()?;
        let row_count = self.groups.iter().try_fold(0usize, |rows, group| {
            rows.checked_add(1)
                .and_then(|rows| rows.checked_add(group.n_a))
                .and_then(|rows| rows.checked_add(group.commit_rows))
                .and_then(|rows| rows.checked_add(group.b_inner_rows))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation quotient row count overflow".into())
                })
        })?;
        let row_count = row_count.checked_add(self.n_d).ok_or_else(|| {
            AkitaError::InvalidSetup("relation quotient row count overflow".into())
        })?;
        let mut dims = Vec::with_capacity(row_count);
        for group in &self.groups {
            dims.push(group.role_dims.d_a());
            dims.extend(repeat_n(group.role_dims.d_a(), group.n_a));
            let b_rows = group
                .commit_rows
                .checked_add(group.b_inner_rows)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation quotient B row count overflow".into())
                })?;
            dims.extend(repeat_n(group.role_dims.d_b(), b_rows));
        }
        dims.extend(repeat_n(self.opening_ring_dim, self.n_d));
        Ok(dims)
    }
}

/// Single source of truth for the relation rhs row layout at one level.
///
/// # Errors
///
/// Returns an error if the opening batch is malformed for multi-group root params.
pub fn relation_rhs_layout_for(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<RelationRhsLayout, AkitaError> {
    opening_batch.check()?;
    let n_d = lp.open_commit_matrix.output_rank();
    if !lp.has_precommitted_groups() {
        let role_dims = lp.role_dims();
        role_dims.validate_role_projection()?;
        return Ok(RelationRhsLayout::uniform(
            role_dims,
            n_d,
            lp.inner_commit_matrix.output_rank(),
            lp.outer_commit_matrix.output_rank(),
            0,
            opening_batch.num_groups(),
        ));
    }
    let final_group_index = lp.validate_opening_batch(opening_batch)?;
    let final_role_dims = lp.group_role_dims(opening_batch, final_group_index)?;
    let mut groups = Vec::with_capacity(lp.precommitted_group_count() + 1);
    groups.push(RelationGroupRows {
        role_dims: final_role_dims,
        n_a: lp.inner_commit_matrix.output_rank(),
        commit_rows: lp.outer_commit_matrix.output_rank(),
        b_inner_rows: 0,
    });
    for (group_index, group) in lp.precommitted_group_iter().enumerate() {
        groups.push(RelationGroupRows {
            role_dims: lp.group_role_dims(opening_batch, group_index)?,
            n_a: group.layout.inner_commit_matrix.output_rank(),
            commit_rows: group.layout.outer_commit_matrix.output_rank(),
            b_inner_rows: 0,
        });
    }
    let layout = RelationRhsLayout {
        opening_ring_dim: final_role_dims.d_d(),
        n_d,
        groups,
    };
    layout.validate()?;
    Ok(layout)
}

/// Logical relation-matrix row count encoded in assembled relation rhs.
///
/// Layout: [consistency_g | A_g | B_g | B_inner_g]_g | D (`n_d`).
#[must_use]
pub fn relation_rhs_row_count(layout: &RelationRhsLayout) -> usize {
    let group_rows = layout.groups.iter().fold(0usize, |acc, group| {
        acc.saturating_add(group.n_a)
            .saturating_add(group.commit_rows)
            .saturating_add(group.b_inner_rows)
    });
    layout
        .groups
        .len()
        .saturating_add(group_rows)
        .saturating_add(layout.n_d)
}

/// Expected flat coefficient length of assembled `y` under per-role dimensions.
///
/// # Errors
///
/// Returns an error if any segment length arithmetic overflows.
pub fn relation_rhs_coeff_len(layout: &RelationRhsLayout) -> Result<usize, AkitaError> {
    layout.validate()?;
    let mut group_segment = 0usize;
    for group in &layout.groups {
        let a_segment = group
            .n_a
            .checked_mul(group.role_dims.d_a())
            .ok_or_else(|| AkitaError::InvalidSetup("relation y A segment overflow".into()))?;
        let b_rows = group
            .commit_rows
            .checked_add(group.b_inner_rows)
            .ok_or_else(|| AkitaError::InvalidSetup("relation y B row count overflow".into()))?;
        let b_segment = b_rows
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| AkitaError::InvalidSetup("relation y B segment overflow".into()))?;
        group_segment = group_segment
            .checked_add(group.role_dims.d_a())
            .and_then(|len| len.checked_add(a_segment))
            .and_then(|len| len.checked_add(b_segment))
            .ok_or_else(|| AkitaError::InvalidSetup("relation y group segment overflow".into()))?;
    }
    let d_segment = layout
        .n_d
        .checked_mul(layout.opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("relation y D segment overflow".into()))?;
    d_segment
        .checked_add(group_segment)
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

/// Build the RHS vector `y` matching the scalar M row layout:
/// consistency (zero) | A (zeros) | B (`commitment_rows`) | D (`v`).
///
/// Public-output rows bind through the fused trace term, not `y`.
///
/// `commit_rows_per_group` is the B row count per commitment bundle
/// (`outer_commit_matrix.output_rank()`). The number of commitment bundles is inferred from
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
    layout: &RelationRhsLayout,
    v: &RingVec<F>,
    commitment_rows: &RingVec<F>,
) -> Result<RingVec<F>, AkitaError> {
    layout.validate()?;
    let v_rows = ring_row_count_at(v, layout.opening_ring_dim)?;
    if v_rows != layout.n_d {
        return Err(AkitaError::InvalidSize {
            expected: layout.n_d,
            actual: v_rows,
        });
    }
    let expected_commit_coeffs = layout.groups.iter().try_fold(0usize, |acc, group| {
        let group_coeffs = group
            .commit_rows
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("assemble_relation_rhs commit width overflow".into())
            })?;
        acc.checked_add(group_coeffs).ok_or_else(|| {
            AkitaError::InvalidSetup("assemble_relation_rhs commit length overflow".into())
        })
    })?;
    if commitment_rows.coeff_len() != expected_commit_coeffs {
        return Err(AkitaError::InvalidSize {
            expected: expected_commit_coeffs,
            actual: commitment_rows.coeff_len(),
        });
    }
    let coeff_len = relation_rhs_coeff_len(layout)?;
    let mut coeffs = Vec::with_capacity(coeff_len);
    let mut commit_offset = 0usize;
    for group in &layout.groups {
        coeffs.extend(repeat_n(F::zero(), group.role_dims.d_a()));
        let a_coeff_len = group
            .n_a
            .checked_mul(group.role_dims.d_a())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("assemble_relation_rhs A segment overflow".into())
            })?;
        coeffs.extend(repeat_n(F::zero(), a_coeff_len));
        let commit_coeff_len = group
            .commit_rows
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| {
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
        let b_inner_coeff_len = group
            .b_inner_rows
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("assemble_relation_rhs B_inner segment overflow".into())
            })?;
        coeffs.extend(repeat_n(F::zero(), b_inner_coeff_len));
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

fn accumulate_extension_flat_rows<F, E, const D: usize>(
    eq_tau1: &[E],
    alpha: E,
    coeffs: &[F],
    row_idx: &mut usize,
    acc: &mut E,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore + MulBaseUnreduced<F>,
{
    if !coeffs.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: coeffs.len(),
        });
    }
    let alpha_pows = scalar_powers(alpha, D);
    for row in coeffs.chunks_exact(D) {
        if *row_idx >= eq_tau1.len() {
            return Ok(());
        }
        let coefficients: [F; D] = row.try_into().map_err(|_| AkitaError::InvalidProof)?;
        let ring = CyclotomicRing::from_coefficients(coefficients);
        *acc += eq_tau1[*row_idx] * eval_ring_at_pows_fast(&ring, &alpha_pows);
        *row_idx += 1;
    }
    Ok(())
}

/// Compute the stage-2 relation claim from the public M-row data.
///
/// This evaluates `sum_i eq(tau1, i) * y_alpha[i]` where `y_alpha` follows
/// the M row layout: per-group consistency/A zero rows, B rows `u`, then D
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
/// Skips each group's native consistency and A rows (all zero in `y`) and
/// dispatches each public segment under its role dimension.
#[tracing::instrument(skip_all, name = "relation_claim_from_layout_extension")]
pub fn relation_claim_from_layout_extension<F, E>(
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
    layout.validate()?;
    if !v.can_decode_vec(layout.opening_ring_dim) {
        return Err(AkitaError::InvalidSize {
            expected: layout.opening_ring_dim,
            actual: v.coeff_len(),
        });
    }
    let expected_u_coeffs = layout.groups.iter().try_fold(0usize, |acc, group| {
        let group_coeffs = group
            .commit_rows
            .checked_mul(group.role_dims.d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation claim commit width overflow".into())
            })?;
        acc.checked_add(group_coeffs)
            .ok_or_else(|| AkitaError::InvalidSetup("relation claim commit length overflow".into()))
    })?;
    if u.coeff_len() != expected_u_coeffs {
        return Err(AkitaError::InvalidSize {
            expected: expected_u_coeffs,
            actual: u.coeff_len(),
        });
    }
    if v.coeff_len() / layout.opening_ring_dim != layout.n_d {
        return Err(AkitaError::InvalidSize {
            expected: layout.n_d,
            actual: v.coeff_len() / layout.opening_ring_dim,
        });
    }
    let row_count = layout
        .groups
        .len()
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
    let mut row_idx = 0usize;
    let uniform_outer_dim = layout.groups.first().and_then(|first| {
        layout
            .groups
            .iter()
            .all(|group| group.role_dims.d_b() == first.role_dims.d_b())
            .then_some(first.role_dims.d_b())
    });
    if let Some(outer_dim) = uniform_outer_dim {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Outer),
            F,
            outer_dim,
            |D_B| {
                let u_typed = u.as_ring_slice::<D_B>()?;
                let mut commit_offset = 0usize;
                for group in &layout.groups {
                    row_idx = row_idx
                        .checked_add(1)
                        .and_then(|row| row.checked_add(group.n_a))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("relation claim row index overflow".into())
                        })?;
                    let commit_end =
                        commit_offset
                            .checked_add(group.commit_rows)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "relation claim commit offset overflow".into(),
                                )
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
    } else {
        let mut commit_coeff_offset = 0usize;
        for group in &layout.groups {
            row_idx = row_idx
                .checked_add(1)
                .and_then(|row| row.checked_add(group.n_a))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim row index overflow".into())
                })?;
            let commit_coeff_len = group
                .commit_rows
                .checked_mul(group.role_dims.d_b())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim commit width overflow".into())
                })?;
            let commit_coeff_end = commit_coeff_offset
                .checked_add(commit_coeff_len)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("relation claim commit offset overflow".into())
                })?;
            let coeffs = u
                .coeffs()
                .get(commit_coeff_offset..commit_coeff_end)
                .ok_or(AkitaError::InvalidProof)?;
            dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Outer),
                F,
                group.role_dims.d_b(),
                |D_B| {
                    accumulate_extension_flat_rows::<F, E, D_B>(
                        &eq_tau1,
                        alpha,
                        coeffs,
                        &mut row_idx,
                        &mut acc,
                    )
                }
            )?;
            row_idx = row_idx.checked_add(group.b_inner_rows).ok_or_else(|| {
                AkitaError::InvalidSetup("relation claim row index overflow".into())
            })?;
            commit_coeff_offset = commit_coeff_end;
        }
    }
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Opening),
        F,
        layout.opening_ring_dim,
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
/// Stage-2 evaluation-trace row weight).
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
    use akita_field::{Fp32, FpExt2, LiftBase, NegOneNr, Prime128OffsetA7F7};

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
        let layout = RelationRhsLayout::uniform(dims, 1, N_A, 1, 0, 1);
        let at_dims = relation_claim_from_layout_extension::<F, E>(
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
        const D: usize = 64;
        let dims = CommitmentRingDims::uniform(D);
        let mut v_coeffs = [F::zero(); D];
        v_coeffs[0] = F::from_u64(1);
        let v = [CyclotomicRing::from_coefficients(v_coeffs)];
        let mut u_coeffs = [F::zero(); D];
        u_coeffs[0] = F::from_u64(2);
        let u = [CyclotomicRing::from_coefficients(u_coeffs)];
        let layout = RelationRhsLayout::uniform(dims, 1, 2, 1, 0, 1);
        let typed =
            generate_relation_rhs::<F, D>(&v, &u, layout.n_d, 1, layout.groups[0].n_a).unwrap();
        let assembled = assemble_relation_rhs::<F>(
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
    fn mixed_role_dims_relation_rhs_coeff_len_matches_per_segment_widths() {
        let dims = CommitmentRingDims {
            inner: 128,
            outer: 32,
            opening: 64,
        };
        let layout = RelationRhsLayout::uniform(dims, 2, 4, 3, 1, 1);
        let coeff_len = relation_rhs_coeff_len(&layout).expect("coeff len");
        let expected = 128 + 2 * 64 + 3 * 32 + 32 + 4 * 128;
        assert_eq!(coeff_len, expected);
        assert_eq!(relation_rhs_row_count(&layout), 1 + 2 + 3 + 1 + 4);
    }

    #[test]
    fn group_local_a_b_dims_share_d_in_rhs_and_claim() {
        type G = Prime128OffsetA7F7;
        let final_dims = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 32,
        };
        let precommitted_dims = CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 32,
        };
        let layout = RelationRhsLayout {
            opening_ring_dim: 32,
            n_d: 1,
            groups: vec![
                RelationGroupRows {
                    role_dims: final_dims,
                    n_a: 1,
                    commit_rows: 1,
                    b_inner_rows: 0,
                },
                RelationGroupRows {
                    role_dims: precommitted_dims,
                    n_a: 2,
                    commit_rows: 2,
                    b_inner_rows: 0,
                },
            ],
        };
        assert_eq!(
            relation_rhs_coeff_len(&layout).expect("mixed group rhs length"),
            128 + 128 + 64 + 64 + 2 * 64 + 2 * 32 + 32
        );
        assert_eq!(
            layout.row_ring_dims().expect("mixed quotient row dims"),
            vec![128, 128, 64, 64, 64, 64, 32, 32, 32]
        );

        let mut commitment_coeffs = vec![G::zero(); 64 + 2 * 32];
        commitment_coeffs[0] = G::from_u64(2);
        commitment_coeffs[64] = G::from_u64(3);
        commitment_coeffs[64 + 32] = G::from_u64(4);
        let commitment_rows = RingVec::from_coeffs(commitment_coeffs);
        let mut v_coeffs = vec![G::zero(); 32];
        v_coeffs[0] = G::from_u64(5);
        let v = RingVec::from_coeffs(v_coeffs);

        let rhs = assemble_relation_rhs(&layout, &v, &commitment_rows).expect("mixed group rhs");
        assert_eq!(
            rhs.coeff_len(),
            relation_rhs_coeff_len(&layout).expect("mixed group rhs length")
        );

        let tau1 = [
            G::from_u64(7),
            G::from_u64(11),
            G::from_u64(13),
            G::from_u64(19),
        ];
        let alpha = G::from_u64(17);
        let claim = relation_claim_from_layout_extension::<G, G>(
            &layout,
            &tau1,
            alpha,
            &v,
            &commitment_rows,
        )
        .expect("mixed group claim");
        let expected = eq_eval_at_index(&tau1, 2) * G::from_u64(2)
            + eq_eval_at_index(&tau1, 6) * G::from_u64(3)
            + eq_eval_at_index(&tau1, 7) * G::from_u64(4)
            + eq_eval_at_index(&tau1, 8) * G::from_u64(5);
        assert_eq!(claim, expected);
    }

    #[test]
    fn rows_allow_group_a_larger_than_final_group_a() {
        let layout = RelationRhsLayout {
            opening_ring_dim: 32,
            n_d: 1,
            groups: vec![
                RelationGroupRows {
                    role_dims: CommitmentRingDims {
                        inner: 64,
                        outer: 32,
                        opening: 32,
                    },
                    n_a: 1,
                    commit_rows: 1,
                    b_inner_rows: 0,
                },
                RelationGroupRows {
                    role_dims: CommitmentRingDims {
                        inner: 128,
                        outer: 32,
                        opening: 32,
                    },
                    n_a: 1,
                    commit_rows: 1,
                    b_inner_rows: 0,
                },
            ],
        };
        assert_eq!(
            layout.row_ring_dims().expect("native quotient row dims"),
            vec![64, 64, 32, 128, 128, 32, 32]
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
        let layout = RelationRhsLayout::uniform(dims, 1, N_A, 1, 0, 1);
        let evaluation_trace_row = relation_rhs_row_count(&layout);
        let trace_target = E::from_u64(19);
        let quotient_claim = relation_claim_from_layout_extension::<F, E>(
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
