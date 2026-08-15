use akita_field::AkitaError;
use akita_types::{active_setup_field_len, CommittedGroupParams, OpeningClaimsLayout};

use crate::schedule_params::level_setup_field_elements;
use crate::schedule_params::pareto;

type LevelCandidate = (
    CommittedGroupParams,
    usize,
    usize,
    Option<crate::response_model::SourceMomentEstimate>,
);

type LevelFrontierEntry = (
    [usize; 6],
    Vec<u8>,
    CommittedGroupParams,
    usize,
    usize,
    Option<crate::response_model::SourceMomentEstimate>,
);

pub(super) fn level_candidates(
    opening_layout: &OpeningClaimsLayout,
    candidates: Vec<LevelCandidate>,
) -> Result<Vec<LevelCandidate>, AkitaError> {
    let mut frontier: Vec<LevelFrontierEntry> = Vec::new();
    for (params, next_witness_len, eor_bytes, next_source_moment) in candidates {
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
            eor_bytes,
        ];
        let descriptor = params.canonical_descriptor_bytes();
        pareto::insert(
            &mut frontier,
            (
                coords,
                descriptor,
                params,
                next_witness_len,
                eor_bytes,
                next_source_moment,
            ),
            |(best, best_descriptor, best_params, best_next_witness_len, _, best_source_moment),
             (
                candidate,
                candidate_descriptor,
                candidate_params,
                candidate_next_witness_len,
                _,
                candidate_source_moment,
            )| {
                best_params.payload_mode == candidate_params.payload_mode
                    && best_params.role_dims() == candidate_params.role_dims()
                    && matches!(
                        best_params.opening_method,
                        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
                    ) == matches!(
                        candidate_params.opening_method,
                        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
                    )
                    // This PR emits one L2 split and norm-proof shape per DP
                    // state. Keep Linf and L2 frontiers separate because these
                    // coordinates do not price the L2 norm payload.
                    && std::mem::discriminant(&best_params.inner_commit_matrix.security_route())
                        == std::mem::discriminant(
                            &candidate_params.inner_commit_matrix.security_route(),
                        )
                    && best_next_witness_len == candidate_next_witness_len
                    && best_source_moment == candidate_source_moment
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
        .map(
            |(_, _, params, next_witness_len, eor_bytes, next_source_moment)| {
                (params, next_witness_len, eor_bytes, next_source_moment)
            },
        )
        .collect())
}
