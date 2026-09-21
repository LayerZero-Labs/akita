use super::super::PlannedFoldCandidate;
use super::level_candidates;

#[path = "prune_bench.rs"]
mod bench;

// Preserve the original flat algorithm as an independent comparison oracle.
use akita_error::AkitaError;
use akita_types::{active_setup_field_len, OpeningClaimsLayout};

use crate::schedule_params::{level_setup_field_elements, pareto};

type LevelFrontierEntry = ([usize; 6], Vec<u8>, PlannedFoldCandidate);

fn flat_reference(
    opening_layout: &OpeningClaimsLayout,
    candidates: Vec<PlannedFoldCandidate>,
) -> Result<Vec<PlannedFoldCandidate>, AkitaError> {
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

fn generated_candidates() -> Vec<PlannedFoldCandidate> {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::{CommitmentPayloadMode, CommitmentRingDims};

    use crate::schedule_params::{
        derive_fold_candidates, FoldCandidatePolicy, PlannerOpeningCandidate,
        RecursiveCandidateRequest, RecursiveFoldWork, RelationSearchDomain, RelationTraversalOrder,
        SplitBoundPolicy,
    };

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let mut candidates = Vec::new();
    for dimension in [64, 128] {
        let dimensions = CommitmentRingDims::uniform(dimension);
        let trace = PlannerOpeningCandidate::evaluation_trace(
            Recursive::ring_challenge_config(dimension).expect("challenge config"),
        );
        let mut openings = vec![(3, trace, RelationSearchDomain::QuotientAndReduced)];
        openings.extend(
            PlannerOpeningCandidate::coefficient_packing_domain(
                1,
                policy.claim_ext_degree,
                dimensions,
            )
            .expect("packing domain")
            .into_iter()
            .map(|opening| (1, opening, RelationSearchDomain::QuotientOnly)),
        );
        for payload_mode in [
            CommitmentPayloadMode::Compressed,
            CommitmentPayloadMode::Raw,
        ] {
            for &(fold_level, opening, relation_domain) in &openings {
                let request = RecursiveCandidateRequest {
                    policy: &policy,
                    payload_mode,
                    opening,
                    dimensions,
                    current_witness_len: 948_672,
                    source: crate::InnerBasisSource::BalancedDigits { log_basis: 4 },
                    log_basis_inner: 4,
                    log_basis_open: 4,
                    fold_level,
                    source_moment: crate::response_model::SourceMomentEstimate::new(1_000_000),
                    relation_traversal_order: RelationTraversalOrder::Canonical,
                    guide: None,
                };
                candidates.extend(
                    derive_fold_candidates(
                        request,
                        RecursiveFoldWork::direct(relation_domain),
                        FoldCandidatePolicy::Frontier(SplitBoundPolicy::Enabled),
                    )
                    .expect("generated fold candidates")
                    .into_iter()
                    .map(|(params, next_witness_len)| PlannedFoldCandidate {
                        params,
                        next_witness_len,
                        opening_reduction_bytes: 0,
                        next_source_moment: None,
                    }),
                );
            }
        }
    }
    assert!(!candidates.is_empty());
    candidates
}

fn clone_candidate(candidate: &PlannedFoldCandidate) -> PlannedFoldCandidate {
    PlannedFoldCandidate {
        params: candidate.params.clone(),
        next_witness_len: candidate.next_witness_len,
        opening_reduction_bytes: candidate.opening_reduction_bytes,
        next_source_moment: candidate.next_source_moment,
    }
}

fn assert_matches_flat(candidates: Vec<PlannedFoldCandidate>) {
    let layout = crate::schedule_params::suffix_opening_layout(948_672, None).unwrap();
    let expected =
        flat_reference(&layout, candidates.iter().map(clone_candidate).collect()).unwrap();
    let actual = level_candidates(&layout, candidates).unwrap();
    assert_eq!(actual.len(), expected.len());
    // Compare in order, including data not present in the canonical descriptor.
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.params, expected.params);
        assert_eq!(actual.next_witness_len, expected.next_witness_len);
        assert_eq!(
            actual.opening_reduction_bytes,
            expected.opening_reduction_bytes
        );
        assert_eq!(actual.next_source_moment, expected.next_source_moment);
    }
}

#[test]
fn lazy_frontier_matches_eager_reference_for_generated_candidates() {
    let mut candidates = generated_candidates();
    assert!(candidates.iter().any(|candidate| matches!(
        candidate.params.inner().matrix.security_route(),
        akita_types::InnerCommitSecurityRoute::L2 { .. }
    )));
    assert!(candidates.iter().any(|candidate| matches!(
        candidate.params.inner().matrix.security_route(),
        akita_types::InnerCommitSecurityRoute::Linf(_)
    )));
    assert!(candidates.iter().any(|candidate| matches!(
        candidate.params.opening_method(),
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    )));
    assert_matches_flat(candidates.iter().map(clone_candidate).collect());
    candidates.reverse();
    assert_matches_flat(candidates);
}

#[test]
fn lazy_frontier_preserves_interleaved_survivor_order_and_ties() {
    let candidates = generated_candidates();
    let seed = &candidates[0];
    let mut interleaved = Vec::new();
    for cost in [9, 4, 4, 7, 0] {
        for witness_offset in [0, 1, 2] {
            for moment in [None, crate::response_model::SourceMomentEstimate::new(1234)] {
                let mut candidate = clone_candidate(seed);
                candidate.opening_reduction_bytes = cost;
                candidate.next_witness_len += witness_offset;
                candidate.next_source_moment = moment;
                interleaved.push(candidate);
            }
        }
    }
    assert_matches_flat(interleaved);
    assert_matches_flat(Vec::new());
    assert_matches_flat(vec![clone_candidate(seed)]);
}

#[test]
fn distinct_descriptors_break_exact_coordinate_ties_in_both_orders() {
    let candidates = generated_candidates();
    let seed = candidates
        .iter()
        .find(|candidate| {
            matches!(
                candidate.params.opening_method(),
                akita_types::OpeningMethod::EvaluationTrace
            )
        })
        .unwrap();
    let mut canonical = clone_candidate(seed);
    canonical.params.source_encoding =
        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable;
    let mut tensor = clone_candidate(seed);
    tensor.params.source_encoding =
        akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
            extension_degree: 2,
        };
    // These distinct physical source encodings are both valid at this ring
    // dimension. They change the descriptor but none of the local cost axes.
    let layout = crate::schedule_params::suffix_opening_layout(948_672, None).unwrap();
    let coordinates = |candidate: &PlannedFoldCandidate| {
        let params = &candidate.params;
        params.source_encoding.validate(params.d_a()).unwrap();
        [
            akita_types::padded_setup_prefix_len(active_setup_field_len(params, &layout).unwrap()),
            level_setup_field_elements(params).unwrap(),
            params
                .outer_payload_geometry()
                .unwrap()
                .transmitted_coefficients(),
            params.outer().matrix.output_rank() * params.role_dims().d_b(),
            params.open().matrix.output_rank() * params.role_dims().d_d(),
            candidate.opening_reduction_bytes,
        ]
    };
    assert_eq!(coordinates(&canonical), coordinates(&tensor));
    let mut restored_params = tensor.params.clone();
    restored_params.source_encoding = canonical.params.source_encoding;
    assert_eq!(canonical.params, restored_params);
    assert_eq!(canonical.next_witness_len, tensor.next_witness_len);
    assert_eq!(canonical.next_source_moment, tensor.next_source_moment);
    let canonical_descriptor = canonical.params.canonical_descriptor_bytes();
    let tensor_descriptor = tensor.params.canonical_descriptor_bytes();
    assert_ne!(canonical_descriptor, tensor_descriptor);
    let expected_descriptor = canonical_descriptor.min(tensor_descriptor);
    for order in [[&canonical, &tensor], [&tensor, &canonical]] {
        assert_matches_flat(
            order
                .iter()
                .map(|candidate| clone_candidate(candidate))
                .collect(),
        );
        let result = level_candidates(
            &layout,
            order
                .iter()
                .map(|candidate| clone_candidate(candidate))
                .collect(),
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].params.canonical_descriptor_bytes(),
            expected_descriptor
        );
    }
}
