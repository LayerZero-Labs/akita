use std::sync::Arc;

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, FieldCore};

use crate::{
    LevelParams, OpeningBatchWitnessLayout, OpeningBlockLayout, RingRelationInstance,
    SemanticGroupId, SetupContributionGroupInputs, SetupContributionPlan,
    SetupContributionPlanInputs, SetupContributionStatic,
};

/// Setup-contribution planning artifact shared by direct replay and recursive
/// stage-3 setup-product proving.
#[derive(Clone)]
pub struct SetupContributionArtifact<E: FieldCore> {
    /// Canonical semantic witness layout used by the setup weight formulas.
    pub chunk_layout: OpeningBatchWitnessLayout,
    /// Challenge-free setup-contribution inputs with expanded tau1 row weights.
    pub inputs: SetupContributionPlanInputs<E>,
    /// Per-commitment-group setup weight descriptors.
    pub groups: Vec<SetupContributionGroupInputs>,
    /// Challenge-free packed segment cache.
    pub static_plan: SetupContributionStatic<E>,
}

/// Build the canonical setup-contribution artifact for one ring-relation level.
pub fn prepare_setup_contribution_artifact<F, E>(
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    tau1: &[E],
    witness_ring_len: Option<usize>,
    opening_layout_override: Option<OpeningBlockLayout>,
) -> Result<SetupContributionArtifact<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    let opening_batch = relation.opening_batch();
    let relation_matrix_row_layout = relation.relation_matrix_row_layout();
    let chunk_layout = relation.segment_layout(lp, witness_ring_len)?;
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold(num_polys, lp.field_bits_for_cache())?;
    let rows =
        lp.relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)?;
    let eq_tau1: Arc<[E]> = EqPolynomial::evals_prefix(tau1, rows)?.into();

    let (inputs, groups, d_physical_cols, d_row_start, d_rows) = if lp.has_precommitted_groups() {
        lp.reject_multi_group_multi_chunk("prepare_setup_contribution_artifact")?;
        lp.validate_root_opening_batch(opening_batch)?;
        let order = opening_batch.root_group_order()?;
        if chunk_layout.ownership_units.len() != order.len() {
            return Err(AkitaError::InvalidSetup(
                "multi-group witness layout does not match root group order".to_string(),
            ));
        }
        let mut groups = Vec::with_capacity(order.len());
        let mut e_col_offset = 0usize;
        for &group_index in &order {
            let group_lp = lp.root_group_params(opening_batch, group_index)?;
            let group_layout = opening_batch.group_layout(group_index)?;
            let num_claims = group_layout.num_polynomials();
            let live_fold_count = group_lp.live_fold_count();
            let depth_open = group_lp.num_digits_open();
            let depth_commit = group_lp.num_digits_commit();
            let n_a = group_lp.a_rows_len();
            let n_b = group_lp.b_rows_len();
            let t_cols_per_vector = n_a
                .checked_mul(depth_open)
                .and_then(|n| n.checked_mul(live_fold_count))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group B vector width overflow".into())
                })?;
            let a_range =
                lp.root_a_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
            let b_range = lp.root_commitment_row_range(
                opening_batch,
                group_index,
                relation_matrix_row_layout,
            )?;
            if a_range.len() != n_a || b_range.len() != n_b {
                return Err(AkitaError::InvalidSetup(
                    "multi-group row ranges do not match group matrix heights".to_string(),
                ));
            }
            let e_len = num_claims
                .checked_mul(live_fold_count)
                .and_then(|n| n.checked_mul(depth_open))
                .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".into()))?;
            groups.push(SetupContributionGroupInputs {
                group_id: SemanticGroupId(group_index),
                e_col_offset,
                num_claims,
                live_fold_count,
                fold_position_count: group_lp.fold_position_count(),
                depth_open,
                depth_commit,
                depth_fold: lp.num_digits_fold_for_params(
                    group_lp,
                    num_claims,
                    lp.field_bits_for_cache(),
                )?,
                log_basis: group_lp.log_basis(),
                n_a,
                n_b,
                t_cols_per_vector,
                a_row_start: a_range.start,
                b_row_start: b_range.start,
                layout: Arc::new(chunk_layout.clone()),
                opening_layout: opening_layout_override.unwrap_or(OpeningBlockLayout::new(
                    live_fold_count,
                    group_lp.fold_position_count(),
                )?),
            });
            e_col_offset = e_col_offset
                .checked_add(e_len)
                .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".into()))?;
        }
        let inputs = SetupContributionPlanInputs {
            relation_matrix_row_layout,
            rows,
            n_a: lp.a_key.row_len(),
            n_b: lp.b_key.row_len(),
            n_d: lp.d_key.row_len(),
            num_groups: opening_batch.num_groups(),
            num_polys_per_group: opening_batch.group_sizes(),
            num_t_vectors: opening_batch.num_total_polynomials(),
            num_claims: opening_batch.num_total_polynomials(),
            live_fold_count: lp.live_fold_count,
            fold_position_count: lp.fold_position_count,
            depth_open: lp.num_digits_open,
            depth_commit: lp.num_digits_commit,
            depth_fold,
            inner_width: lp.a_key.col_len(),
            eq_tau1: eq_tau1.clone(),
        };
        let d_rows = lp.n_d_active_for(relation_matrix_row_layout);
        let d_row_start = rows.checked_sub(d_rows).ok_or(AkitaError::InvalidProof)?;
        (inputs, groups, e_col_offset, d_row_start, d_rows)
    } else {
        let mut inputs = SetupContributionPlanInputs::from_level_params(
            lp,
            &[num_polys],
            relation_matrix_row_layout,
            depth_fold,
        )?;
        inputs.eq_tau1 = eq_tau1;
        let single = SetupContributionGroupInputs::single_group_layout(
            &inputs,
            &chunk_layout,
            opening_layout_override.unwrap_or(OpeningBlockLayout::new(
                lp.live_fold_count,
                lp.fold_position_count,
            )?),
            lp.log_basis,
        )?;
        (
            inputs,
            vec![single.group],
            single.d_physical_cols,
            single.d_row_start,
            single.d_rows,
        )
    };

    if inputs.rows != rows {
        return Err(AkitaError::InvalidSetup(
            "setup contribution row count mismatch".to_string(),
        ));
    }
    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        d_row_start,
        d_rows,
        d_physical_cols,
    )?;
    Ok(SetupContributionArtifact {
        chunk_layout,
        inputs,
        groups,
        static_plan,
    })
}
