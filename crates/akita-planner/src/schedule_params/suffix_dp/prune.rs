use akita_error::AkitaError;
use akita_types::{active_setup_field_len, OpeningClaimsLayout};

use crate::schedule_params::{level_setup_field_elements, pareto};

type LevelFrontierEntry = ([usize; 6], Vec<u8>, super::PlannedFoldCandidate);

pub(super) fn level_candidates(
    opening_layout: &OpeningClaimsLayout,
    candidates: Vec<super::PlannedFoldCandidate>,
) -> Result<Vec<super::PlannedFoldCandidate>, AkitaError> {
    let mut frontier: Vec<LevelFrontierEntry> = Vec::new();
    for candidate in candidates {
        let params = &candidate.params;
        let outer_payload_coeffs = params.outer_payload_geometry()?.transmitted_coefficients();
        let coords = [
            akita_types::padded_setup_prefix_len(active_setup_field_len(params, opening_layout)?),
            level_setup_field_elements(params)?,
            outer_payload_coeffs,
            params
                .outer()
                .matrix
                .output_rank()
                .checked_mul(params.role_dims().d_b())
                .ok_or_else(|| AkitaError::InvalidSetup("B output dimension overflow".into()))?,
            params
                .open()
                .matrix
                .output_rank()
                .checked_mul(params.role_dims().d_d())
                .ok_or_else(|| AkitaError::InvalidSetup("D output dimension overflow".into()))?,
            candidate.opening_reduction_bytes,
        ];
        let descriptor = params.canonical_descriptor_bytes();
        pareto::insert(
            &mut frontier,
            (coords, descriptor, candidate),
            |(best, best_descriptor, best_candidate),
             (candidate, candidate_descriptor, candidate_entry)| {
                best_candidate.params.payload_mode == candidate_entry.params.payload_mode
                    && best_candidate.params.ring_relation_mode
                        == candidate_entry.params.ring_relation_mode
                    && best_candidate.params.role_dims() == candidate_entry.params.role_dims()
                    && matches!(
                        best_candidate.params.opening_method(),
                        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
                    ) == matches!(
                        candidate_entry.params.opening_method(),
                        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
                    )
                    && std::mem::discriminant(
                        &best_candidate.params.inner().matrix.security_route(),
                    ) == std::mem::discriminant(
                        &candidate_entry.params.inner().matrix.security_route(),
                    )
                    && best_candidate.next_witness_len == candidate_entry.next_witness_len
                    && best_candidate.next_source_moment == candidate_entry.next_source_moment
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
        .map(|(_, _, candidate)| candidate)
        .collect())
}
