//! Manual local-pruning timing; candidate construction is outside every timer.

use std::{hint::black_box, time::Duration, time::Instant};

use akita_config::{
    policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_types::{
    CommitmentPayloadPhase, CommitmentRingDims, OpeningClaimsLayout, PolynomialGroupLayout,
    RingRelationPhase,
};

use super::{clone_candidate, flat_reference, level_candidates, PlannedFoldCandidate};
use crate::schedule_params::{
    suffix_dp::{candidates::CandidateDomain, source::attach_source_moments},
    RelationModeFilter, RelationTraversalOrder, SetupPrefixSearchCache, SuffixCtx, SuffixState,
    SuffixTopology,
};

fn compare_batch(label: &str, layout: &OpeningClaimsLayout, candidates: &[PlannedFoldCandidate]) {
    let expected =
        flat_reference(layout, candidates.iter().map(clone_candidate).collect()).unwrap();
    let actual =
        level_candidates(layout, candidates.iter().map(clone_candidate).collect()).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.params, expected.params);
        assert_eq!(actual.next_witness_len, expected.next_witness_len);
        assert_eq!(actual.next_source_moment, expected.next_source_moment);
        assert_eq!(
            actual.opening_reduction_bytes,
            expected.opening_reduction_bytes
        );
    }

    let mut elapsed = [Duration::ZERO; 2];
    const ROUNDS: usize = 256;
    for round in 0..ROUNDS {
        // Alternate order to avoid assigning every warm run to one algorithm.
        for algorithm in [round % 2, (round + 1) % 2] {
            let input = candidates.iter().map(clone_candidate).collect();
            let start = Instant::now();
            let result = if algorithm == 0 {
                flat_reference(black_box(layout), black_box(input))
            } else {
                level_candidates(black_box(layout), black_box(input))
            }
            .unwrap();
            elapsed[algorithm] += start.elapsed();
            black_box(result);
        }
    }
    eprintln!(
        "{label} candidates={} retained={} flat_ns={} lazy_ns={}",
        candidates.len(),
        actual.len(),
        elapsed[0].as_nanos() / ROUNDS as u128,
        elapsed[1].as_nanos() / ROUNDS as u128,
    );
}

#[test]
#[ignore = "manual release microbenchmark; does not run suffix DP"]
fn local_pruning_per_opening_basis_microbench() {
    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let policy = policy_of::<Recursive>();
    let challenge_config = |dimension| Recursive::ring_challenge_config(dimension);
    let ctx = SuffixCtx {
        policy: &policy,
        diagnostics: None,
        ring_challenge_config: &challenge_config,
        key: PolynomialGroupLayout::singleton(20),
        setup_field_budget: None,
        root_lookup_key: None,
        root_main_constraint: None,
        adaptation_guide: None,
        root_honest_fold_policy: None,
        precommitted_source_contracts: &[],
        level_zero_is_root: false,
        relation_traversal_order: RelationTraversalOrder::Canonical,
        relation_mode_filter: RelationModeFilter::All,
    };
    for witness_len in [4_096, 65_536, 948_672] {
        for level in [1, 3] {
            let state = SuffixState {
                level,
                current_witness_len: witness_len,
                current_lb: 4,
                source_moment: crate::response_model::SourceMomentEstimate::new(1_000_000),
                dimension_ceiling: CommitmentRingDims::uniform(64),
                topology: SuffixTopology::Direct {
                    payload_phase: CommitmentPayloadPhase::CompressedPrefix,
                    relation_phase: RingRelationPhase::QuotientPrefix,
                },
            };
            let domain = CandidateDomain::prepare(&ctx, state).unwrap();
            let mut setup_prefixes = SetupPrefixSearchCache::default();
            for open_basis in domain.opening_basis_range.clone() {
                let generated = domain
                    .generate_for_opening_basis(&ctx, state, open_basis, &mut setup_prefixes)
                    .unwrap();
                let candidates = attach_source_moments(
                    &ctx,
                    state,
                    false,
                    &domain.opening_layout,
                    generated.folds,
                )
                .unwrap();
                if candidates.is_empty() {
                    continue;
                }
                let label = format!("witness={witness_len} level={level} open_basis={open_basis}");
                compare_batch(&label, &domain.opening_layout, &candidates);
                // Explicit synthetic prefixes complement complete real batches
                // without claiming to reproduce the workload's size distribution.
                for count in [1, 4, 8, 16] {
                    if count < candidates.len() {
                        compare_batch(
                            &format!("{label} synthetic_prefix={count}"),
                            &domain.opening_layout,
                            &candidates[..count],
                        );
                    }
                }
            }
        }
    }
}
