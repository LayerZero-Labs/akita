use super::*;

#[test]
fn seed_recursive_split_candidates_falls_back_to_exhaustive_for_small_domains() {
    assert_eq!(
        seed_recursive_split_candidates(64, 5, 1, 22, 1),
        vec![4, 3, 2, 1]
    );
}

#[test]
fn seed_recursive_split_candidates_includes_endpoints_and_unique_window() {
    let candidates = seed_recursive_split_candidates(8192, 13, 1, 22, 1);
    assert!(candidates.contains(&1));
    assert!(candidates.contains(&12));
    assert!(
        candidates.windows(2).all(|pair| pair[0] > pair[1]),
        "candidates must be unique and descending: {candidates:?}"
    );
}

#[test]
fn recursive_split_lower_bound_prices_score_floor() {
    assert_eq!(
        recursive_split_lower_bound(RecursiveSplitLowerBoundInput {
            num_ring_elems: 100,
            ring_dimension: 64,
            reduced_vars: 7,
            r: 3,
            delta_commit: 1,
            delta_open: 4,
            num_chunks: 2,
            requested_fold_shape: TensorChallengeShape::Flat,
        }),
        Some(5646)
    );
}

#[test]
fn recursive_candidate_order_preserves_exhaustive_tie_break() {
    let score = (100, 90, 5, 0);
    assert!(
        recursive_candidate_order_key(score, 9) < recursive_candidate_order_key(score, 8),
        "the old descending exhaustive scan retained the larger split on a tie"
    );
    assert!(
        recursive_candidate_order_key((99, 98, 1, 0), 1) < recursive_candidate_order_key(score, 9),
        "the exact layout score must remain the primary objective"
    );
}
