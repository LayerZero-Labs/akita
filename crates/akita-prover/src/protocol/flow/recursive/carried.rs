use crate::protocol::flow::root_fold::evaluate_recursive_witness_at_multiplier_point;
use crate::protocol::flow::{ProveLevelOutput, RecursiveCarriedOpening, RecursiveCarriedSource};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};
use akita_types::{
    prepare_recursive_opening_point_ext, recover_ring_subfield_inner_product, BasisMode,
    BlockOrder, CarriedOpeningKind, CarriedOpeningProof, CarriedOpeningSourceProof, LevelParams,
    RingSubfieldEncoding,
};

fn evaluate_carried_source_opening<F, L, const D: usize>(
    source: &RecursiveCarriedSource<F>,
    point: &[L],
    source_params: &LevelParams,
) -> Result<L, AkitaError>
where
    F: FieldCore + CanonicalField,
    L: ExtField<F> + RingSubfieldEncoding<F>,
{
    if source_params.ring_dimension != D {
        return Err(AkitaError::InvalidSetup(
            "carried source propagation across ring dimensions is not implemented".to_string(),
        ));
    }
    let alpha = source_params.ring_dimension.trailing_zeros() as usize;
    let prepared_point = prepare_recursive_opening_point_ext::<F, L, D>(
        point,
        BasisMode::Lagrange,
        source_params,
        alpha,
        BlockOrder::ColumnMajor,
    )?;
    let source_view = source.w.view::<F, D>()?;
    let (y_ring, _) = evaluate_recursive_witness_at_multiplier_point(
        &source_view,
        &prepared_point.ring_multiplier_point,
        source_params.block_len,
        source_params.num_blocks,
    )?;
    recover_ring_subfield_inner_product::<F, L, D>(&y_ring, &prepared_point.inner_reduction)
}

pub(super) fn propagate_extra_carried_sources<F, L, const D: usize>(
    out: &mut ProveLevelOutput<F, L>,
    extra_carried_sources: &[RecursiveCarriedSource<F>],
    source_params: &LevelParams,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    L: ExtField<F> + RingSubfieldEncoding<F>,
{
    if extra_carried_sources.is_empty() {
        return Ok(());
    }
    let point = out
        .next_state
        .carried_openings
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .opening_point
        .clone();
    let point_domain_len = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("carried point domain overflow".to_string()))?;
    let mut common_padded_len = out
        .next_state
        .w
        .len()
        .max(point_domain_len)
        .next_power_of_two();
    for source in extra_carried_sources {
        let natural_len = source.logical_w.as_ref().unwrap_or(&source.w).len();
        common_padded_len = common_padded_len.max(natural_len.next_power_of_two());
    }
    if let Some(witness_claim) = out.next_state.carried_openings.first_mut() {
        witness_claim.padded_len = common_padded_len;
    }
    out.next_state.extra_carried_sources = extra_carried_sources.to_vec();
    for (source_idx, source) in extra_carried_sources.iter().enumerate() {
        let proof_source_idx = source_idx + 1;
        let source_logical = source.logical_w.as_ref().unwrap_or(&source.w);
        let opening = evaluate_carried_source_opening::<F, L, D>(source, &point, source_params)?;
        let natural_len = source_logical.len();
        out.level_proof
            .stage2
            .extra_carried_sources
            .push(CarriedOpeningSourceProof {
                commitment: source.commitment.clone(),
            });
        out.level_proof
            .stage2
            .extra_carried_openings
            .push(CarriedOpeningProof {
                source_idx: proof_source_idx,
                point: point.clone(),
                value: opening,
                basis: BasisMode::Lagrange,
                natural_len,
                padded_len: common_padded_len,
                kind: CarriedOpeningKind::SetupPrefix,
            });
        out.next_state
            .carried_openings
            .push(RecursiveCarriedOpening {
                source_idx: proof_source_idx,
                opening_point: point.clone(),
                opening,
                basis: BasisMode::Lagrange,
                natural_len,
                padded_len: common_padded_len,
                kind: CarriedOpeningKind::SetupPrefix,
            });
    }
    Ok(())
}
