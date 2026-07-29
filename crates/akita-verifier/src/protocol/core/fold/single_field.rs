//! Single-field fold verifier prefix (`EXT_DEGREE == 1`): no EOR replay.

use super::super::*;
use akita_types::{dispatch_for_field, Commitment, TerminalCommittedGroupParams};

pub(in crate::protocol::core) struct SingleFieldFoldPrefix<F: FieldCore, E: FieldCore> {
    pub prepared_points: Vec<PreparedOpeningPoint<F, E>>,
    pub row_coefficients: Vec<E>,
    pub trace_eval_target: E,
    pub trace_claim_coefficients: Vec<E>,
}

fn absorb_prepared_opening_points<F, E, T>(
    prepared_points: &[PreparedOpeningPoint<F, E>],
    transcript: &mut T,
) where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + AkitaSerialize,
    T: Transcript<F>,
{
    for prepared in prepared_points {
        for coordinate in &prepared.padded_point {
            append_ext_field::<F, E, T>(transcript, ABSORB_EVALUATION_CLAIMS, coordinate);
        }
    }
}

/// Scalar-root single-field prefix: `prepare_opening_point` only, no EOR.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_single_field_scalar_root_prefix<F, E, T>(
    shared_opening_point: &[E],
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    d_a: usize,
    transcript: &mut T,
) -> Result<SingleFieldFoldPrefix<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let prepared_point = dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
        prepare_opening_point::<F, E, D>(
            shared_opening_point,
            basis,
            root_lp.num_positions_per_block,
            root_lp.num_live_blocks,
            d_a.trailing_zeros() as usize,
        )
    })?;
    absorb_prepared_opening_points(&[prepared_point.clone()], transcript);
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    let row_coefficients = sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?;
    let trace_eval_target = opening_batch.batched_eval_target(&row_coefficients, openings)?;
    Ok(SingleFieldFoldPrefix {
        prepared_points: vec![prepared_point],
        row_coefficients: row_coefficients.clone(),
        trace_eval_target,
        trace_claim_coefficients: row_coefficients,
    })
}

/// Multi-group root single-field prefix: per-group `prepare_opening_point`, no EOR.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_single_field_multi_group_root_prefix<F, E, T>(
    claims: &OpeningClaims<'_, E, &Commitment<F>>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    root_lp: &CommittedGroupParams,
    transcript: &mut T,
) -> Result<SingleFieldFoldPrefix<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
    T: Transcript<F>,
{
    let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
    for group_index in 0..opening_batch.num_groups() {
        let group_dims = root_lp.group_role_dims(opening_batch, group_index)?;
        let group_alpha_bits = group_dims.d_a().trailing_zeros() as usize;
        let group_lp = root_lp.group_params(opening_batch, group_index)?;
        let group_point = claims.group_point(group_index)?;
        let prepared = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_dims.d_a(),
            |D| {
                prepare_opening_point::<F, E, D>(
                    &group_point,
                    basis,
                    group_lp.num_positions_per_block(),
                    group_lp.num_live_blocks(),
                    group_alpha_bits,
                )
            }
        )?;
        prepared_points.push(prepared);
    }
    absorb_prepared_opening_points(&prepared_points, transcript);
    append_claim_values_to_transcript::<F, E, T>(openings, transcript);
    let row_coefficients = sample_public_row_coefficients::<F, E, T>(opening_batch, transcript)?;
    let trace_eval_target = opening_batch.batched_eval_target(&row_coefficients, openings)?;
    Ok(SingleFieldFoldPrefix {
        prepared_points,
        row_coefficients: row_coefficients.clone(),
        trace_eval_target,
        trace_claim_coefficients: row_coefficients,
    })
}

/// Recursive suffix single-field prefix: per-group opening points, no EOR.
pub(in crate::protocol::core) fn verify_single_field_suffix_prefix<F, E>(
    block_claims: &OpeningClaims<'_, E>,
    openings: &[E],
    opening_batch: &OpeningClaimsLayout,
    lp: &CommittedGroupParams,
    role_d_a: usize,
    alpha_bits: usize,
) -> Result<SingleFieldFoldPrefix<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    let prepared_points = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        role_d_a,
        |D| {
            let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
            for group_index in 0..opening_batch.num_groups() {
                let group_lp = lp.group_params(opening_batch, group_index)?;
                let target_len = alpha_bits
                    .checked_add(group_lp.position_index_bits())
                    .and_then(|n| n.checked_add(group_lp.block_index_bits()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("group opening point length overflow".to_string())
                    })?;
                let point_vars = block_claims.group_point_vars(group_index)?;
                if point_vars.num_vars() != target_len {
                    return Err(AkitaError::InvalidInput(format!(
                        "suffix group point width mismatch: group={group_index}, \
                         groups={}, setup_prefix={}, target_len={target_len}, actual_len={}",
                        opening_batch.num_groups(),
                        lp.setup_prefix.is_some(),
                        point_vars.num_vars()
                    )));
                }
                let group_protocol_point = block_claims.group_point(group_index)?;
                prepared_points.push(prepare_opening_point::<F, E, D>(
                    &group_protocol_point,
                    BasisMode::Lagrange,
                    group_lp.num_positions_per_block(),
                    group_lp.num_live_blocks(),
                    alpha_bits,
                )?);
            }
            Ok(prepared_points)
        }
    )?;
    let row_coefficients = vec![E::one(); opening_batch.num_total_polynomials()];
    let trace_eval_target = opening_batch.batched_eval_target(&row_coefficients, openings)?;
    Ok(SingleFieldFoldPrefix {
        prepared_points,
        row_coefficients: row_coefficients.clone(),
        trace_eval_target,
        trace_claim_coefficients: row_coefficients,
    })
}

/// Terminal suffix single-field prefix: one `prepare_opening_point`, no EOR.
#[allow(clippy::too_many_arguments)]
pub(in crate::protocol::core) fn verify_single_field_terminal_suffix_prefix<F, E>(
    protocol_point: &[E],
    opening: E,
    _opening_batch: &OpeningClaimsLayout,
    basis: BasisMode,
    params: &TerminalCommittedGroupParams,
    alpha_bits: usize,
) -> Result<SingleFieldFoldPrefix<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + ExtField<F> + FrobeniusExtField<F> + FromPrimitiveInt + AkitaSerialize,
{
    let prepared_point = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        params.d_a(),
        |D| prepare_opening_point::<F, E, D>(
            protocol_point,
            basis,
            params.num_positions_per_block,
            params.num_live_blocks,
            alpha_bits,
        )
    )?;
    let row_coefficients = vec![E::one()];
    let trace_eval_target = opening;
    Ok(SingleFieldFoldPrefix {
        prepared_points: vec![prepared_point],
        row_coefficients: row_coefficients.clone(),
        trace_eval_target,
        trace_claim_coefficients: row_coefficients,
    })
}
