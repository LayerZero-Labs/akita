use std::sync::Arc;

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, FieldCore};

use crate::{
    LevelParams, RelationGroupId, RelationRowId, RingRelationInstance,
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionPlanInputs,
    SetupContributionStatic, WitnessLayout,
};

/// Setup-contribution planning artifact shared by direct replay and recursive
/// stage-3 setup-product proving.
#[derive(Clone)]
pub struct SetupContributionArtifact<E: FieldCore> {
    /// Resolved witness chunk layout used by the setup weight formulas.
    pub chunk_layout: WitnessLayout,
    /// Challenge-free setup-contribution inputs with expanded tau1 row weights.
    pub inputs: SetupContributionPlanInputs<E>,
    /// Per-commitment-group setup weight descriptors.
    pub groups: Vec<SetupContributionGroupInputs>,
    /// Challenge-free packed segment cache.
    pub static_plan: SetupContributionStatic<E>,
}

/// Build the canonical setup-contribution artifact for one ring-relation level.
///
/// The artifact preserves the grouped root geometry when `lp` carries
/// precommitted groups, so recursive stage-3 proves
/// `setup(i) * sum_g weight_g(i)` over the same packed setup prefix that direct
/// verifier replay scans.
pub fn prepare_setup_contribution_artifact<F, E>(
    relation: &RingRelationInstance<F>,
    lp: &LevelParams,
    tau1: &[E],
    witness_ring_len: Option<usize>,
) -> Result<SetupContributionArtifact<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FieldCore,
{
    let opening_batch = relation.opening_batch();
    let relation_layout = relation.relation_layout();
    let row_plan = relation_layout.row_plan();
    let chunk_layout = relation_layout.witness_layout(witness_ring_len)?.clone();
    let num_polys = opening_batch.num_total_polynomials();
    let depth_fold = lp.num_digits_fold(num_polys, lp.field_bits_for_cache())?;
    let rows = row_plan.trace_row();
    let eq_tau1: Arc<[E]> = EqPolynomial::evals(tau1)?.into();
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }

    let d_provider = if row_plan
        .families()
        .iter()
        .any(|family| matches!(family.id(), RelationRowId::D))
    {
        Some(relation_layout.family_provider(RelationRowId::D)?)
    } else {
        None
    };
    let n_d_active = d_provider
        .as_ref()
        .map_or(0, |provider| provider.rows().len());
    let d_row_start = d_provider
        .as_ref()
        .map_or(rows, |provider| provider.rows().start());

    let (inputs, groups, d_physical_cols) = if lp.has_precommitted_groups() {
        prepare_multi_group_setup_artifact_inputs(
            lp,
            relation_layout,
            opening_batch,
            &chunk_layout,
            rows,
            depth_fold,
            eq_tau1.clone(),
        )?
    } else {
        prepare_single_group_setup_artifact_inputs(
            lp,
            relation_layout,
            &chunk_layout,
            rows,
            depth_fold,
            num_polys,
            eq_tau1.clone(),
        )?
    };

    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        d_row_start,
        n_d_active,
        d_physical_cols,
    )?;

    Ok(SetupContributionArtifact {
        chunk_layout,
        inputs,
        groups,
        static_plan,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_single_group_setup_artifact_inputs<E: FieldCore>(
    lp: &LevelParams,
    relation_layout: &crate::RelationLayout,
    chunk_layout: &WitnessLayout,
    rows: usize,
    depth_fold: usize,
    num_polys: usize,
    eq_tau1: Arc<[E]>,
) -> Result<
    (
        SetupContributionPlanInputs<E>,
        Vec<SetupContributionGroupInputs>,
        usize,
    ),
    AkitaError,
> {
    let mut inputs = SetupContributionPlanInputs::from_level_params(
        lp,
        relation_layout,
        &[num_polys],
        depth_fold,
    )?;
    inputs.eq_tau1 = eq_tau1;
    if inputs.rows != rows {
        return Err(AkitaError::InvalidSetup(
            "setup contribution row count mismatch".to_string(),
        ));
    }

    let single_group =
        SetupContributionGroupInputs::single_group_layout(&inputs, chunk_layout, lp.log_basis)?;
    let d_physical_cols = single_group.d_physical_cols;
    Ok((inputs, vec![single_group.group], d_physical_cols))
}

#[allow(clippy::too_many_arguments)]
fn prepare_multi_group_setup_artifact_inputs<E: FieldCore>(
    lp: &LevelParams,
    relation_layout: &crate::RelationLayout,
    opening_batch: &crate::OpeningClaimsLayout,
    chunk_layout: &WitnessLayout,
    rows: usize,
    depth_fold: usize,
    eq_tau1: Arc<[E]>,
) -> Result<
    (
        SetupContributionPlanInputs<E>,
        Vec<SetupContributionGroupInputs>,
        usize,
    ),
    AkitaError,
> {
    lp.reject_multi_group_multi_chunk("prepare_setup_contribution_artifact")?;
    lp.validate_root_opening_batch(opening_batch)?;

    let order = opening_batch.root_group_order()?;
    if chunk_layout.chunks.len() != order.len() || chunk_layout.chunk_lengths.len() != order.len() {
        return Err(AkitaError::InvalidSetup(
            "multi-group witness layout does not match root group order".to_string(),
        ));
    }

    let mut d_physical_cols = 0usize;
    let mut groups = Vec::with_capacity(order.len());
    for (order_pos, &group_index) in order.iter().enumerate() {
        let group_lp = lp.root_group_params(opening_batch, group_index)?;
        let group_layout = opening_batch.group_layout(group_index)?;
        let k_g = group_layout.num_polynomials();
        let num_blocks = group_lp.num_blocks();
        let block_len = group_lp.block_len();
        let depth_open = group_lp.num_digits_open();
        let depth_commit = group_lp.num_digits_commit();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.b_rows_len();
        let inner_width = group_lp.a_col_len();
        let expected_inner_width = block_len.checked_mul(depth_commit).ok_or_else(|| {
            AkitaError::InvalidSetup("multi-group inner width overflow".to_string())
        })?;
        if inner_width < expected_inner_width {
            return Err(AkitaError::InvalidSetup(
                "multi-group A-key column width is too small".to_string(),
            ));
        }

        let t_cols_per_vector = n_a
            .checked_mul(depth_open)
            .and_then(|len| len.checked_mul(num_blocks))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let semantic_group = if group_index == opening_batch.root_final_group_index()? {
            RelationGroupId::Current
        } else {
            RelationGroupId::Precommitted { index: group_index }
        };
        let a_range = relation_layout
            .family_provider(RelationRowId::A {
                group: semantic_group,
            })?
            .rows()
            .range();
        let b_range = relation_layout
            .family_provider(RelationRowId::B {
                group: semantic_group,
            })?
            .rows()
            .range();
        if a_range.len() != n_a || b_range.len() != n_b {
            return Err(AkitaError::InvalidSetup(
                "multi-group row ranges do not match group matrix heights".to_string(),
            ));
        }
        let e_col_offset = d_physical_cols;
        let arithmetic_e_len = k_g
            .checked_mul(num_blocks)
            .and_then(|n| n.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".to_string()))?;
        let e_len = chunk_layout.chunk_lengths[order_pos].e_len;
        if e_len != arithmetic_e_len {
            return Err(AkitaError::InvalidSetup(
                "multi-group evaluator E width disagrees with semantic witness layout".into(),
            ));
        }
        d_physical_cols = d_physical_cols
            .checked_add(e_len)
            .ok_or_else(|| AkitaError::InvalidSetup("multi-group e width overflow".to_string()))?;

        let chunks = chunk_layout
            .chunks
            .get(order_pos..order_pos + 1)
            .ok_or(AkitaError::InvalidProof)?
            .to_vec();
        groups.push(SetupContributionGroupInputs {
            e_col_offset,
            num_claims: k_g,
            num_blocks,
            block_len,
            depth_open,
            depth_commit,
            depth_fold,
            log_basis: group_lp.log_basis(),
            n_a,
            n_b,
            t_cols_per_vector,
            a_row_start: a_range.start,
            b_row_start: b_range.start,
            blocks_per_chunk: num_blocks,
            chunks,
        });
    }

    let current_a_provider = relation_layout.family_provider(RelationRowId::A {
        group: RelationGroupId::Current,
    })?;
    let current_b_provider = relation_layout.family_provider(RelationRowId::B {
        group: RelationGroupId::Current,
    })?;
    let d_provider = if relation_layout
        .row_plan()
        .families()
        .iter()
        .any(|family| matches!(family.id(), RelationRowId::D))
    {
        Some(relation_layout.family_provider(RelationRowId::D)?)
    } else {
        None
    };
    let current_a = current_a_provider.family();
    let current_b = current_b_provider.family();
    let d_family = d_provider.as_ref().map(|provider| provider.family());
    if current_a.rows().len() != lp.a_key.row_len()
        || current_b.rows().len() != lp.b_key.row_len()
        || current_a.native_ring_dim() != lp.a_key.sis_table_key().ring_dimension as usize
        || current_b.native_ring_dim() != lp.b_key.sis_table_key().ring_dimension as usize
        || d_family.is_some_and(|family| {
            family.rows().len() != lp.d_key.row_len()
                || family.native_ring_dim() != lp.d_key.sis_table_key().ring_dimension as usize
        })
    {
        return Err(AkitaError::InvalidSetup(
            "multi-group relation plan disagrees with current setup geometry".into(),
        ));
    }
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: if d_family.is_some() {
            crate::RelationMatrixRowLayout::WithDBlock
        } else {
            crate::RelationMatrixRowLayout::WithoutDBlock
        },
        rows,
        n_a: current_a.rows().len(),
        n_b: current_b.rows().len(),
        n_d: d_family.map_or(0, |family| family.rows().len()),
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
        eq_tau1,
    };

    Ok((inputs, groups, d_physical_cols))
}
