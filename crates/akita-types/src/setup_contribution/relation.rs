use std::sync::Arc;

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, FieldCore};

use crate::{
    LevelParams, RingRelationInstance, SetupContributionGroupInputs, SetupContributionLayout,
    SetupContributionPlan, SetupContributionPlanInputs, SetupContributionStatic,
};

/// Setup-contribution planning artifact shared by direct replay and recursive
/// stage-3 setup-product proving.
#[derive(Clone)]
pub struct SetupContributionArtifact<E: FieldCore> {
    /// Canonical semantic witness layout used by the setup weight formulas.
    pub layout: SetupContributionLayout,
    /// Challenge-free setup-contribution inputs with expanded tau1 row weights.
    pub inputs: SetupContributionPlanInputs<E>,
    /// Challenge-free packed segment cache.
    pub static_plan: SetupContributionStatic<E>,
}

/// Build the canonical setup-contribution artifact for one ring-relation level.
pub fn prepare_setup_contribution_artifact<F, E>(
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    tau1: &[E],
    witness_ring_len: Option<usize>,
    opening_layout_override: Option<usize>,
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

    let (inputs, groups) = if lp.has_precommitted_groups() {
        lp.validate_opening_batch(opening_batch)?;
        let order = opening_batch.root_group_order()?;
        if order.iter().any(|&group_index| {
            chunk_layout.num_chunks_for_group(group_index) != lp.witness_chunk.num_chunks
        }) {
            return Err(AkitaError::InvalidSetup(
                "multi-group witness layout does not match root group order".to_string(),
            ));
        }
        let mut groups = Vec::with_capacity(order.len());
        for &group_index in &order {
            let group_lp = lp.group_params(opening_batch, group_index)?;
            let group_layout = opening_batch.group_layout(group_index)?;
            let num_claims = group_layout.num_polynomials();
            let num_blocks = group_lp.num_blocks();
            let depth_open = group_lp.num_digits_open();
            let depth_commit = group_lp.num_digits_commit();
            let n_a = group_lp.a_rows_len();
            let n_b = group_lp.b_rows_len();
            let t_cols_per_vector = n_a
                .checked_mul(depth_open)
                .and_then(|n| n.checked_mul(num_blocks))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("multi-group B vector width overflow".into())
                })?;
            let a_range = lp.a_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
            let b_range =
                lp.commitment_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
            if a_range.len() != n_a || b_range.len() != n_b {
                return Err(AkitaError::InvalidSetup(
                    "multi-group row ranges do not match group matrix heights".to_string(),
                ));
            }
            groups.push(SetupContributionGroupInputs {
                group_id: group_index,
                num_claims,
                num_blocks,
                block_len: group_lp.block_len(),
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
            });
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
            num_blocks: lp.num_blocks,
            block_len: lp.block_len,
            depth_open: lp.num_digits_open,
            depth_commit: lp.num_digits_commit,
            depth_fold,
            inner_width: lp.a_key.col_len(),
            eq_tau1: eq_tau1.clone(),
        };
        (inputs, groups)
    } else {
        let mut inputs = SetupContributionPlanInputs::from_level_params(
            lp,
            &[num_polys],
            relation_matrix_row_layout,
            depth_fold,
        )?;
        inputs.eq_tau1 = eq_tau1;
        let group = SetupContributionGroupInputs::from_single_group(&inputs, lp.log_basis)?;
        (inputs, vec![group])
    };

    if inputs.rows != rows {
        return Err(AkitaError::InvalidSetup(
            "setup contribution row count mismatch".to_string(),
        ));
    }
    let opening_source_len = opening_layout_override.unwrap_or(chunk_layout.total_len());
    let layout = SetupContributionLayout::new(Arc::new(chunk_layout), opening_source_len, groups)?;
    let static_plan = SetupContributionPlan::prepare_static(&inputs, &layout)?;
    Ok(SetupContributionArtifact {
        layout,
        inputs,
        static_plan,
    })
}
