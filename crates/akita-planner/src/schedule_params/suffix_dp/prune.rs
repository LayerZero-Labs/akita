use akita_field::AkitaError;
use akita_types::{active_setup_field_len, CommittedGroupParams, OpeningClaimsLayout};

use crate::schedule_params::level_setup_field_elements;
use crate::schedule_params::pareto;

type LevelFrontierEntry = ([usize; 5], Vec<u8>, CommittedGroupParams, usize, usize);

pub(super) fn level_candidates(
    opening_layout: &OpeningClaimsLayout,
    candidates: Vec<(CommittedGroupParams, usize, usize)>,
) -> Result<Vec<(CommittedGroupParams, usize, usize)>, AkitaError> {
    let mut frontier: Vec<LevelFrontierEntry> = Vec::new();
    for (params, next_witness_len, eor_bytes) in candidates {
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
        pareto::insert(
            &mut frontier,
            (coords, descriptor, params, next_witness_len, eor_bytes),
            |(best, best_descriptor, best_params, best_next_witness_len, _),
             (candidate, candidate_descriptor, candidate_params, candidate_next_witness_len, _)| {
                best_params.payload_mode == candidate_params.payload_mode
                    && best_params.role_dims() == candidate_params.role_dims()
                    && best_next_witness_len == candidate_next_witness_len
                    && pareto::canonical_dominates(
                        best,
                        best_descriptor,
                        candidate,
                        candidate_descriptor,
                    )
            },
        );
    }
    Ok(frontier
        .into_iter()
        .map(|(_, _, params, next_witness_len, eor_bytes)| (params, next_witness_len, eor_bytes))
        .collect())
}
