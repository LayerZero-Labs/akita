/// Insert one candidate into a Pareto frontier.
///
/// `dominates(left, right)` owns the objective's exact equivalence class,
/// coordinate ordering, and canonical tie-break. The skip/retain dual is kept
/// here so callers cannot hand-maintain inconsistent logical negations.
pub(super) fn insert<T>(
    frontier: &mut Vec<T>,
    candidate: T,
    dominates: impl Fn(&T, &T) -> bool,
) -> bool {
    if frontier.iter().any(|other| dominates(other, &candidate)) {
        return false;
    }
    frontier.retain(|other| !dominates(&candidate, other));
    frontier.push(candidate);
    true
}

/// Componentwise minimization with a canonical ordered tie-key.
pub(super) fn canonical_dominates<const N: usize, K: Ord + ?Sized>(
    left_coords: &[usize; N],
    left_tie_key: &K,
    right_coords: &[usize; N],
    right_tie_key: &K,
) -> bool {
    left_coords
        .iter()
        .zip(right_coords)
        .all(|(left, right)| left <= right)
        && (left_coords != right_coords || left_tie_key <= right_tie_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_tie_keeps_only_smallest_descriptor() {
        let mut frontier = Vec::new();
        let dominates = |left: &([usize; 2], Vec<u8>), right: &([usize; 2], Vec<u8>)| {
            canonical_dominates(&left.0, &left.1, &right.0, &right.1)
        };
        assert!(insert(&mut frontier, ([2, 3], vec![2]), dominates));
        assert!(insert(&mut frontier, ([2, 3], vec![1]), dominates));
        assert!(!insert(&mut frontier, ([2, 3], vec![3]), dominates));
        assert_eq!(frontier, vec![([2, 3], vec![1])]);
    }

    #[test]
    fn incomparable_coordinates_are_retained() {
        let mut frontier = Vec::new();
        let dominates = |left: &([usize; 2], Vec<u8>), right: &([usize; 2], Vec<u8>)| {
            canonical_dominates(&left.0, &left.1, &right.0, &right.1)
        };
        assert!(insert(&mut frontier, ([1, 4], vec![1]), dominates));
        assert!(insert(&mut frontier, ([4, 1], vec![2]), dominates));
        assert_eq!(frontier.len(), 2);
    }

    #[test]
    fn composite_tie_key_uses_score_before_descriptor() {
        let low_score = (4usize, vec![9u8]);
        let low_descriptor = (5usize, vec![1u8]);
        assert!(canonical_dominates(
            &[2, 3],
            &low_score,
            &[2, 3],
            &low_descriptor,
        ));
        assert!(!canonical_dominates(
            &[2, 3],
            &low_descriptor,
            &[2, 3],
            &low_score,
        ));
    }
}
