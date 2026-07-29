//! Single-field fold verifier prefix (`EXT_DEGREE == 1`): no EOR replay.

// Explicit imports only: the compiler enforces that the single-field path has
// no extension-opening-reduction symbols in scope.
use akita_field::{
    AkitaError, CanonicalField, ExtField, FieldCore, FrobeniusExtField, FromPrimitiveInt,
};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels::ABSORB_EVALUATION_CLAIMS;
use akita_transcript::{append_ext_field, Transcript};
use akita_types::{
    append_claim_values_to_transcript, dispatch_for_field, prepare_opening_point,
    sample_public_row_coefficients, BasisMode, Commitment, CommittedGroupParams, FpExtEncoding,
    OpeningClaims, OpeningClaimsLayout, PreparedOpeningPoint,
};

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
    let prepared_point =
        dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D| {
            prepare_opening_point::<F, E, D>(
                shared_opening_point,
                basis,
                root_lp.num_positions_per_block,
                root_lp.num_live_blocks,
                d_a.trailing_zeros() as usize,
            )
        })?;
    absorb_prepared_opening_points(std::slice::from_ref(&prepared_point), transcript);
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
        let target_len = group_alpha_bits
            .checked_add(group_lp.position_index_bits())
            .and_then(|n| n.checked_add(group_lp.block_index_bits()))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("group opening point length overflow".to_string())
            })?;
        let point_vars = claims.group_point_vars(group_index)?;
        if point_vars.num_vars() != target_len {
            return Err(AkitaError::InvalidProof);
        }
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
