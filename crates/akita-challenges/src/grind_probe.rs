//! Fisher–Yates permutation for fold grind probe order (ZK witness hiding).

use crate::sampler::XofCursor;

/// Fisher–Yates shuffle of `0..cap` (equivalent to uniform sampling without
/// replacement over the full probe sequence).
pub fn grind_probe_permutation(seed: &[u8], cap: u32) -> Vec<u32> {
    let cap_usize = cap as usize;
    let mut order: Vec<u32> = (0..cap).collect();
    if cap_usize <= 1 {
        return order;
    }
    let mut cursor = XofCursor::from_seed(seed);
    for i in 0..cap_usize {
        let j = i + cursor.next_usize_mod(cap_usize - i);
        order.swap(i, j);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grind_probe_permutation_is_bijection() {
        let cap = 64u32;
        let order = grind_probe_permutation(b"seed-a", cap);
        assert_eq!(order.len(), cap as usize);
        let mut seen = vec![false; cap as usize];
        for nonce in order {
            let idx = nonce as usize;
            assert!(!seen[idx], "duplicate nonce {nonce}");
            seen[idx] = true;
        }
    }

    #[test]
    fn grind_probe_permutation_depends_on_seed() {
        let cap = 32u32;
        let left = grind_probe_permutation(b"left", cap);
        let right = grind_probe_permutation(b"right", cap);
        assert_ne!(left, right);
    }
}
