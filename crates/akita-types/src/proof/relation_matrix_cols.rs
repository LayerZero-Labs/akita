//! Shared singleton and multi-group relation matrix column evaluation.
//!
//! [`compute_relation_weight_evals`] materializes the tau1-weighted relation
//! polynomial on the next witness's Boolean field-coefficient domain.
//! The verifier replays the same group-major geometry with its structured
//! `RelationMatrixEvaluator` path instead of rebuilding the dense vector.

use crate::layout::CommitmentRingDims;
use crate::proof::ring_relation::RingRelationInstance;
use crate::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    OpeningClaimsLayout, RelationMatrixRowLayout, SetupProjectionGeometry,
};
use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase, MulBaseUnreduced,
};

#[allow(clippy::too_many_arguments)]
fn write_role_weight<E: FieldCore, S>(
    sink: &mut S,
    opening_source_len: usize,
    opening_ring_dim: usize,
    witness_ring_dim: usize,
    witness_col: usize,
    role_subcol: usize,
    role_ring_dim: usize,
    alpha_pows: &[E],
    column_weight: E,
    columns: bool,
) -> Result<(), AkitaError>
where
    S: FnMut(usize, E) -> Result<(), AkitaError>,
{
    if alpha_pows.len() != role_ring_dim
        || witness_ring_dim == 0
        || opening_ring_dim == 0
        || !witness_ring_dim.is_multiple_of(role_ring_dim)
    {
        return Err(AkitaError::InvalidProof);
    }
    let role_ratio = witness_ring_dim / role_ring_dim;
    if role_subcol >= role_ratio {
        return Err(AkitaError::InvalidProof);
    }
    if columns {
        if role_subcol != 0 {
            return Err(AkitaError::InvalidProof);
        }
        let opening_col = crate::checked_opening_source_index(opening_source_len, witness_col)?;
        return sink(opening_col, column_weight);
    }
    let physical_base = witness_col
        .checked_mul(witness_ring_dim)
        .and_then(|base| base.checked_add(role_subcol * role_ring_dim))
        .ok_or_else(|| AkitaError::InvalidSetup("relation weight address overflow".into()))?;
    for (coefficient, &alpha_power) in alpha_pows.iter().enumerate() {
        write_coefficient_weight(
            sink,
            opening_source_len,
            opening_ring_dim,
            physical_base + coefficient,
            column_weight * alpha_power,
        )?;
    }
    Ok(())
}

fn write_coefficient_weight<E: FieldCore, S>(
    sink: &mut S,
    opening_source_len: usize,
    opening_ring_dim: usize,
    physical: usize,
    weight: E,
) -> Result<(), AkitaError>
where
    S: FnMut(usize, E) -> Result<(), AkitaError>,
{
    let opening_col =
        crate::checked_opening_source_index(opening_source_len, physical / opening_ring_dim)?;
    let opening_index = opening_col
        .checked_mul(opening_ring_dim)
        .and_then(|base| base.checked_add(physical % opening_ring_dim))
        .ok_or_else(|| AkitaError::InvalidSetup("relation weight address overflow".into()))?;
    sink(opening_index, weight)
}

fn relation_d_group_width(
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    group_index: usize,
) -> Result<usize, AkitaError> {
    let group_lp = lp.group_params(opening_batch, group_index)?;
    let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
    num_claims
        .checked_mul(group_lp.num_live_blocks())
        .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))
}

fn relation_total_d_columns(
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    let mut cursor = 0usize;
    let mut seen = vec![false; opening_batch.num_groups()];
    for group_id in opening_batch.root_group_order()? {
        let slot = seen
            .get_mut(group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
        let width = relation_d_group_width(lp, opening_batch, group_id)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        cursor = end;
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(cursor)
}

fn relation_d_column_start(
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    target_group_id: usize,
) -> Result<usize, AkitaError> {
    let mut cursor = 0usize;
    let mut start = None;
    let mut seen = vec![false; opening_batch.num_groups()];
    for group_id in opening_batch.root_group_order()? {
        let slot = seen
            .get_mut(group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
        let width = relation_d_group_width(lp, opening_batch, group_id)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        if group_id == target_group_id {
            start = Some(cursor);
        }
        cursor = end;
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    start.ok_or_else(|| AkitaError::InvalidSetup("setup D group is missing".into()))
}

/// Unified relation matrix column evaluation for singleton and multi-group root relations.
///
/// Singleton roots use the scalar/chunked witness layout. Multi-group roots use the
/// group-major layout and still reject multi-chunk witness emission.
///
/// # Errors
///
/// Returns an error if the batch shape, opening-point layout, challenge count,
/// chunking configuration, or expanded matrix dimensions are inconsistent.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "compute_relation_weight_evals")]
pub fn compute_relation_weight_evals<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    alpha_pows: &[E],
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    opening_source_len: usize,
    opening_ring_dim: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    Ok(compute_relation_weight_evals_inner(
        setup,
        instance,
        alpha,
        alpha_pows,
        role_dims,
        lp,
        tau1,
        gamma,
        relation_matrix_row_layout,
        opening_source_len,
        opening_ring_dim,
        None,
        false,
    )?
    .0)
}

/// Build the per-X-column relation weights `M(x)` for the uniform ring geometry.
///
/// This produces one scalar per canonical witness column over the Boolean X
/// domain (`opening_domain_len(opening_source_len)` entries), dropping the
/// per-coefficient `alpha` spread that the flattened builder bakes into the
/// full field domain. It is only valid when the role ring dimensions are
/// uniform (`d_a == d_b == d_d`), i.e. the separable `R(x, y) = M(x) * a(y)`
/// factorization holds; the caller must gate on that. The unused X-column
/// suffix is left zero-padded.
///
/// # Errors
///
/// Returns an error if the ring dimensions are not uniform or if any of the
/// shape/layout invariants checked by [`compute_relation_weight_evals`] fail.
#[allow(clippy::too_many_arguments)]
pub fn compute_relation_matrix_col_evals<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    alpha_pows: &[E],
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    opening_source_len: usize,
    opening_ring_dim: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    Ok(compute_relation_weight_evals_inner(
        setup,
        instance,
        alpha,
        alpha_pows,
        role_dims,
        lp,
        tau1,
        gamma,
        relation_matrix_row_layout,
        opening_source_len,
        opening_ring_dim,
        None,
        true,
    )?
    .0)
}

/// Evaluate the relation-weight MLE directly on the flattened opening domain.
///
/// This visits the same canonical nonzero weights as
/// [`compute_relation_weight_evals`] without allocating or scanning the padded
/// Boolean suffix.
#[allow(clippy::too_many_arguments)]
pub fn eval_relation_weight_at_point<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    alpha_pows: &[E],
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    opening_source_len: usize,
    opening_ring_dim: usize,
    point: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    Ok(compute_relation_weight_evals_inner(
        setup,
        instance,
        alpha,
        alpha_pows,
        role_dims,
        lp,
        tau1,
        gamma,
        relation_matrix_row_layout,
        opening_source_len,
        opening_ring_dim,
        Some(point),
        false,
    )?
    .1)
}

#[allow(clippy::too_many_arguments)]
fn compute_relation_weight_evals_inner<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    alpha_pows: &[E],
    role_dims: CommitmentRingDims,
    lp: &LevelParams,
    tau1: &[E],
    gamma: &[E],
    relation_matrix_row_layout: RelationMatrixRowLayout,
    opening_source_len: usize,
    opening_ring_dim: usize,
    point: Option<&[E]>,
    columns: bool,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    let opening_batch = instance.opening_batch();
    lp.witness_chunk.validate()?;
    lp.validate_opening_batch(opening_batch)?;
    if gamma.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    if columns && (point.is_some() || d_a != d_b || d_a != d_d) {
        return Err(AkitaError::InvalidSetup(
            "uniform column relation requires uniform ring dims and no point".into(),
        ));
    }
    if alpha_pows.len() != d_a {
        return Err(AkitaError::InvalidSize {
            expected: d_a,
            actual: alpha_pows.len(),
        });
    }
    let alpha_pows_a = alpha_pows;
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let rows =
        lp.relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)?;
    let eq_tau1 = SplitEqEvals::new(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    let n_d_active = lp.n_d_active_for(relation_matrix_row_layout);
    let levels = r_decomp_levels::<F>(lp.log_basis);
    let witness_layout = instance.segment_layout(lp, None)?;
    let expected_r_len = rows.checked_mul(levels).ok_or_else(|| {
        AkitaError::InvalidSetup("relation quotient witness width overflow".to_string())
    })?;
    if witness_layout.r_range().len() != expected_r_len {
        return Err(AkitaError::InvalidSetup(
            "relation matrix dimensions disagree with witness layout".to_string(),
        ));
    }
    let (b_ratio, d_ratio) = SetupProjectionGeometry::witness_subcolumn_ratios(role_dims)?;
    let d_physical_cols = relation_total_d_columns(lp, opening_batch)?;
    let e_total = d_physical_cols
        .checked_mul(d_ratio)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))?;
    let physical_field_len = witness_layout
        .total_len()
        .checked_mul(d_a)
        .ok_or_else(|| AkitaError::InvalidSetup("relation weight length overflow".into()))?;
    let expected_field_len = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;
    if physical_field_len > expected_field_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_field_len,
            actual: physical_field_len,
        });
    }
    let opening_field_len = crate::opening_domain_len(opening_source_len)?
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("relation weight length overflow".into()))?;
    if let Some(point) = point {
        let expected_bits = opening_field_len.trailing_zeros() as usize;
        if !opening_field_len.is_power_of_two() || point.len() != expected_bits {
            return Err(AkitaError::InvalidSize {
                expected: expected_bits,
                actual: point.len(),
            });
        }
    }
    let mut out = if point.is_some() {
        Vec::new()
    } else if columns {
        vec![E::zero(); crate::opening_domain_len(opening_source_len)?]
    } else {
        vec![E::zero(); opening_field_len]
    };
    let mut evaluation = E::zero();
    let mut sink = |index: usize, weight: E| -> Result<(), AkitaError> {
        if let Some(point) = point {
            evaluation += akita_algebra::offset_eq::eq_eval_at_index(point, index) * weight;
        } else {
            *out.get_mut(index).ok_or(AkitaError::InvalidProof)? = weight;
        }
        Ok(())
    };

    let d_view = setup
        .shared_matrix
        .ring_view_dyn(lp.d_key.row_len(), e_total, d_d)?;
    let d_rows: Vec<&[F]> = (0..lp.d_key.row_len())
        .map(|r| d_view.row_flat(r))
        .collect::<Result<_, _>>()?;
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let consistency_weight = eq_tau1.eval_at(0)?;

    for group_index in 0..opening_batch.num_groups() {
        let e_setup_offset = relation_d_column_start(lp, opening_batch, group_index)?;
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_id = group_index;
        let units = witness_layout.units_for_group(group_id)?;
        let k_g = group_layout.num_polynomials();
        let opening_point = instance.group_opening_point(group_index)?;
        let ring_multiplier_point = instance.group_ring_multiplier_point(group_index)?;
        let challenges = &instance.group_challenges()[group_index];
        if opening_point.position_weights.len() != group_lp.num_positions_per_block()
            || opening_point.live_block_weights.len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval opening-point layout mismatch".to_string(),
            ));
        }
        if ring_multiplier_point.position_len() != group_lp.num_positions_per_block()
            || ring_multiplier_point.fold_len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval multiplier layout mismatch".to_string(),
            ));
        }
        let total_blocks = k_g
            .checked_mul(group_lp.num_live_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.logical_len() != total_blocks {
            return Err(AkitaError::InvalidProof);
        }
        let depth_witness = group_lp.num_digits_witness();
        let depth_commit = group_lp.num_digits_commit();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
        let log_basis_witness = group_lp.log_basis_witness();
        let log_basis_commit = group_lp.log_basis_commit();
        let log_basis_open = group_lp.log_basis_open();
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.b_rows_len();
        let inner_width = group_lp.a_col_len();
        // Hoist per-group geometry into `Copy` locals so the parallel closures
        // below capture scalars instead of the `!Sync` `&dyn LevelParamsLike`.
        let num_live_blocks_g = group_lp.num_live_blocks();
        let num_positions_per_block_g = group_lp.num_positions_per_block();
        let semantic_t_vector_width = n_a
            .checked_mul(depth_commit)
            .and_then(|len| len.checked_mul(num_live_blocks_g))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let t_vector_width = semantic_t_vector_width
            .checked_mul(b_ratio)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let b_width = k_g
            .checked_mul(t_vector_width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".to_string()))?;
        let setup_a_view = setup.shared_matrix.ring_view_dyn(n_a, inner_width, d_a)?;
        let b_view = setup.shared_matrix.ring_view_dyn(n_b, b_width, d_b)?;
        let setup_a_rows: Vec<&[F]> = (0..n_a)
            .map(|r| setup_a_view.row_flat(r))
            .collect::<Result<_, _>>()?;
        let b_rows: Vec<&[F]> = (0..n_b)
            .map(|r| b_view.row_flat(r))
            .collect::<Result<_, _>>()?;
        let a_range = lp.a_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
        let b_range =
            lp.commitment_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
        if a_range.end > eq_tau1.len() || b_range.end > eq_tau1.len() {
            return Err(AkitaError::InvalidProof);
        }
        let g_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let t_commit_gadget: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis_commit)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let witness_gadget: Vec<E> = gadget_row_scalars::<F>(depth_witness, log_basis_witness)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();

        for claim in 0..k_g {
            for global_block in 0..num_live_blocks_g {
                let unit = witness_layout.unit_for_block(group_id, global_block)?;
                let challenge_index = claim
                    .checked_mul(num_live_blocks_g)
                    .and_then(|base| base.checked_add(global_block))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation challenge index overflow".into())
                    })?;
                let challenge_alpha =
                    challenges.eval_logical_at_pows::<F, E>(challenge_index, alpha_pows)?;
                for (digit, &opening_gadget) in g_open.iter().enumerate() {
                    let witness_col = unit.e_index(k_g, depth_open, claim, global_block, digit)?;
                    for role_subcol in 0..d_ratio {
                        let logical_block = claim * num_live_blocks_g + global_block;
                        let d_phys_col = logical_block
                            .checked_mul(d_ratio)
                            .and_then(|base| base.checked_add(role_subcol))
                            .and_then(|base| base.checked_mul(depth_open))
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|local| {
                                e_setup_offset
                                    .checked_mul(d_ratio)
                                    .and_then(|offset| offset.checked_add(local))
                            })
                            .ok_or(AkitaError::InvalidProof)?;
                        let consistency_acc = consistency_weight * challenge_alpha * opening_gadget;
                        let mut setup_acc = E::zero();
                        for (di, d_row) in d_rows.iter().take(n_d_active).enumerate() {
                            let eq_i = eq_tau1.eval_at(d_start + di)?;
                            if !eq_i.is_zero() {
                                setup_acc += eq_i
                                    * eval_flat_ring_at_pows_fast(
                                        &d_row[d_phys_col * d_d..(d_phys_col + 1) * d_d],
                                        &alpha_pows_d,
                                    );
                            }
                        }
                        if columns {
                            let opening_col = crate::checked_opening_source_index(
                                opening_source_len,
                                witness_col,
                            )?;
                            sink(opening_col, consistency_acc + setup_acc)?;
                        } else {
                            let physical_base = witness_col * d_a + role_subcol * d_d;
                            for coefficient in 0..d_d {
                                write_coefficient_weight(
                                    &mut sink,
                                    opening_source_len,
                                    opening_ring_dim,
                                    physical_base + coefficient,
                                    consistency_acc * alpha_pows_a[role_subcol * d_d + coefficient]
                                        + setup_acc * alpha_pows_d[coefficient],
                                )?;
                            }
                        }
                    }
                }
                for a_idx in 0..n_a {
                    let a_row_weight = eq_tau1.eval_at(a_range.start + a_idx)?;
                    for (digit, &opening_gadget) in t_commit_gadget.iter().enumerate() {
                        let block_claim = num_live_blocks_g
                            .checked_mul(claim)
                            .and_then(|base| base.checked_add(global_block))
                            .ok_or(AkitaError::InvalidProof)?;
                        let row_block_claim = n_a
                            .checked_mul(block_claim)
                            .and_then(|base| base.checked_add(a_idx))
                            .ok_or(AkitaError::InvalidProof)?;
                        let semantic_col = depth_commit
                            .checked_mul(row_block_claim)
                            .and_then(|base| base.checked_add(digit))
                            .ok_or(AkitaError::InvalidProof)?;
                        let witness_col = unit.t_index(
                            k_g,
                            n_a,
                            depth_commit,
                            claim,
                            global_block,
                            a_idx,
                            digit,
                        )?;
                        for role_subcol in 0..b_ratio {
                            let local_col = semantic_col
                                .checked_mul(b_ratio)
                                .and_then(|base| base.checked_add(role_subcol))
                                .ok_or(AkitaError::InvalidProof)?;
                            let a_acc = a_row_weight * challenge_alpha * opening_gadget;
                            let mut b_acc = E::zero();
                            for (row_idx, b_row) in b_rows.iter().take(n_b).enumerate() {
                                let eq_i = eq_tau1.eval_at(b_range.start + row_idx)?;
                                if !eq_i.is_zero() {
                                    b_acc += eq_i
                                        * eval_flat_ring_at_pows_fast(
                                            &b_row[local_col * d_b..(local_col + 1) * d_b],
                                            &alpha_pows_b,
                                        );
                                }
                            }
                            if columns {
                                let opening_col = crate::checked_opening_source_index(
                                    opening_source_len,
                                    witness_col,
                                )?;
                                sink(opening_col, a_acc + b_acc)?;
                            } else {
                                let physical_base = witness_col * d_a + role_subcol * d_b;
                                for coefficient in 0..d_b {
                                    write_coefficient_weight(
                                        &mut sink,
                                        opening_source_len,
                                        opening_ring_dim,
                                        physical_base + coefficient,
                                        a_acc * alpha_pows_a[role_subcol * d_b + coefficient]
                                            + b_acc * alpha_pows_b[coefficient],
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
        }

        // For z_hat[blk, dc, df], the column value is:
        //
        // -G_fold[df] * (
        //     tau_consistency * a_alpha[blk] * G_commit[dc]
        //     + sum_r tau_A[r] * A_alpha[r, blk, dc]
        //   ).
        //
        // The first term is the opening row. The second term is the A-row setup
        // contribution. A is already digit-domain, so the A-row setup term does
        // not multiply by G_commit.
        let z_base = cfg_into_iter!(0..inner_width)
            .map(|k| {
                let block_idx = k / depth_witness;
                let digit_idx = k % depth_witness;
                let opening_a_eval =
                    ring_multiplier_point.eval_position_at_dyn::<E>(block_idx, alpha_pows_a)?;
                let mut acc = consistency_weight * opening_a_eval * witness_gadget[digit_idx];
                for (a_idx, a_row) in setup_a_rows.iter().take(n_a).enumerate() {
                    let eq_i = eq_tau1.eval_at(a_range.start + a_idx)?;
                    if !eq_i.is_zero() {
                        acc += eq_i
                            * eval_flat_ring_at_pows_fast(
                                &a_row[k * d_a..(k + 1) * d_a],
                                alpha_pows_a,
                            );
                    }
                }
                Ok(acc)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        for unit in units {
            for position in 0..num_positions_per_block_g {
                for commit_digit in 0..depth_witness {
                    for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                        let phys_k = position * depth_witness + commit_digit;
                        let witness_col = unit.z_index(
                            num_positions_per_block_g,
                            depth_witness,
                            depth_fold,
                            position,
                            commit_digit,
                            fold_digit,
                        )?;
                        write_role_weight(
                            &mut sink,
                            opening_source_len,
                            opening_ring_dim,
                            d_a,
                            witness_col,
                            0,
                            d_a,
                            alpha_pows_a,
                            -(z_base[phys_k] * fold),
                            columns,
                        )?;
                    }
                }
            }
        }
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.log_basis)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for row in 0..rows {
        let eq_weight = eq_tau1.eval_at(row)?;
        let is_b_row = (0..opening_batch.num_groups()).try_fold(false, |found, group| {
            Ok::<_, AkitaError>(
                found
                    || lp
                        .commitment_row_range(opening_batch, group, relation_matrix_row_layout)?
                        .contains(&row),
            )
        })?;
        let (row_dim, row_alpha_pows): (usize, &[E]) = if row >= d_start {
            (d_d, alpha_pows_d.as_slice())
        } else if is_b_row {
            (d_b, alpha_pows_b.as_slice())
        } else {
            (d_a, alpha_pows_a)
        };
        let row_denom = row_alpha_pows[row_dim - 1] * alpha + E::one();
        for (digit, gadget) in r_gadget.iter().enumerate() {
            let witness_col = witness_layout.r_index(levels, row, digit)?;
            write_role_weight(
                &mut sink,
                opening_source_len,
                opening_ring_dim,
                d_a,
                witness_col,
                0,
                row_dim,
                row_alpha_pows,
                -(eq_weight * row_denom * *gadget),
                columns,
            )?;
        }
    }
    Ok((out, evaluation))
}
