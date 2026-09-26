//! Single-field fold verifier prefix (`EXT_DEGREE == 1`): no EOR replay.

// Explicit imports only: the compiler enforces that the single-field path has
// no extension-opening-reduction symbols in scope.
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    dispatch_for_field, prepare_opening_point, BasisMode, CommittedGroupParams, FpExtEncoding,
    OpeningClaims, OpeningClaimsLayout, PreparedOpeningPoint,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

/// Recursive-suffix single-field preparation: per-group `prepare_opening_point`
/// over the suffix opening groups, no EOR.
pub(in crate::protocol::core) fn prepare_single_field_suffix_groups<F, E>(
    block_claims: &OpeningClaims<'_, E>,
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<Vec<PreparedOpeningPoint<F, E>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize,
{
    let mut prepared_points = Vec::with_capacity(opening_batch.num_groups());
    let final_group_index = opening_batch.root_final_group_index()?;
    for group_index in 0..opening_batch.num_groups() {
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims(opening_batch, group_index)?;
        let group_alpha_bits = group_dims.d_a().trailing_zeros() as usize;
        let prepared = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            group_dims.d_a(),
            |D| {
                let target_len = group_alpha_bits
                    .checked_add(group_lp.position_index_bits())
                    .and_then(|n| n.checked_add(group_lp.block_index_bits()))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("group opening point length overflow".to_string())
                    })?;
                let group_protocol_point = block_claims.group_point(group_index)?;
                let point_width_is_valid = if group_index == final_group_index {
                    group_protocol_point.len() <= target_len
                } else {
                    group_protocol_point.len() == target_len
                };
                if !point_width_is_valid {
                    return Err(AkitaError::InvalidInput(format!(
                        "suffix group point width mismatch: group={group_index}, \
                         groups={}, setup_prefix={}, target_len={target_len}, actual_len={}",
                        opening_batch.num_groups(),
                        lp.setup_prefix().is_some(),
                        group_protocol_point.len()
                    )));
                }
                prepare_opening_point::<F, E, D>(
                    group_protocol_point,
                    BasisMode::Lagrange,
                    group_lp.num_positions_per_block(),
                    group_lp.num_live_blocks(),
                    group_alpha_bits,
                )
            }
        )?;
        prepared_points.push(prepared);
    }
    Ok(prepared_points)
}
