use akita_field::AkitaError;
use akita_types::{active_setup_field_len, CommittedGroupParams, OpeningClaimsLayout};

use crate::schedule_params::level_setup_field_elements;

pub(super) fn level_candidates(
    opening_layout: &OpeningClaimsLayout,
    candidates: Vec<(CommittedGroupParams, usize)>,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    let mut frontier: Vec<([usize; 5], CommittedGroupParams, usize)> = Vec::new();
    for (params, next_witness_len) in candidates {
        let outer_payload_coeffs = params.outer_payload_geometry()?.transmitted_coefficients();
        let coords = [
            akita_types::padded_setup_prefix_len(active_setup_field_len(&params, opening_layout)?),
            level_setup_field_elements(&params)?,
            outer_payload_coeffs,
            params
                .outer_commit_matrix
                .output_rank()
                .checked_mul(params.role_dims().d_b())
                .ok_or_else(|| AkitaError::InvalidSetup("B output dimension overflow".into()))?,
            params
                .open_commit_matrix
                .output_rank()
                .checked_mul(params.role_dims().d_d())
                .ok_or_else(|| AkitaError::InvalidSetup("D output dimension overflow".into()))?,
        ];
        let descriptor = params.canonical_descriptor_bytes();
        if frontier
            .iter()
            .any(|(best, best_params, best_next_witness_len)| {
                best_params.payload_mode == params.payload_mode
                    && *best_next_witness_len == next_witness_len
                    && best.iter().zip(coords).all(|(lhs, rhs)| *lhs <= rhs)
                    && (best != &coords || best_params.canonical_descriptor_bytes() <= descriptor)
            })
        {
            continue;
        }
        frontier.retain(|(other, other_params, other_next_witness_len)| {
            other_params.payload_mode != params.payload_mode
                || *other_next_witness_len != next_witness_len
                || !coords.iter().zip(*other).all(|(lhs, rhs)| *lhs <= rhs)
                || (other == &coords && other_params.canonical_descriptor_bytes() < descriptor)
        });
        frontier.push((coords, params, next_witness_len));
    }
    Ok(frontier
        .into_iter()
        .map(|(_, params, next_witness_len)| (params, next_witness_len))
        .collect())
}
