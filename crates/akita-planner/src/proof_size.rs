pub const FIELD_BITS: u32 = 128;

// Optimized (header-stripped) sizing.

/// Serialized byte width for a field element with `field_bits` bits.
pub const fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

/// Ring vector bytes without length prefix.
pub fn ring_vec_bytes(ring_len: usize, ring_dim: u32, field_bits: u32) -> usize {
    ring_len * ring_dim as usize * field_bytes(field_bits)
}

/// Sumcheck proof bytes (header-stripped): `rounds * degree * 16`.
pub fn sumcheck_bytes(rounds: usize, degree: usize, field_bits: u32) -> usize {
    rounds * degree * field_bytes(field_bits)
}

/// Packed digit bytes without length/tag prefix.
pub fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    (num_elems * bits_per_elem as usize).div_ceil(8)
}

/// Stage 1 bytes with eq-compression + fully 4-ary GKR tree.
///
/// lb=2 (d=2): 1 stage, eq-compressed degree 2 -> 2 elems/round
/// lb=3 (d=4): 1 stage, eq-compressed degree 4 -> 4 elems/round
/// lb>=4: floor((lb-1)/2) degree-4 stages + (lb-1)%2 degree-2 stage at root
///        + inter-stage claims
pub fn stage1_bytes_optimized(n_rounds: usize, lb: u32, field_bits: u32) -> usize {
    let elem_bytes = field_bytes(field_bits);
    if lb <= 3 {
        let d = ((1u32 << lb) >> 1) as usize;
        return n_rounds * d * elem_bytes;
    }
    let num_levels = (lb - 1) as usize;
    let num_4ary = num_levels / 2;
    let has_binary_top = num_levels % 2;

    let deg4_cost = n_rounds * 4 * elem_bytes;
    let deg2_cost = n_rounds * 2 * elem_bytes;
    let stage_cost = num_4ary * deg4_cost + has_binary_top * deg2_cost;

    let total_stages = num_4ary + has_binary_top;
    let inter_claims = if total_stages <= 1 {
        0
    } else if has_binary_top != 0 {
        let mut claims: usize = 2;
        let mut nodes: usize = 2;
        for _ in 0..num_4ary.saturating_sub(1) {
            claims += 4 * nodes;
            nodes *= 4;
        }
        claims * elem_bytes
    } else {
        let mut claims: usize = 0;
        let mut nodes: usize = 1;
        for _ in 0..num_4ary.saturating_sub(1) {
            claims += 4 * nodes;
            nodes *= 4;
        }
        claims * elem_bytes
    };

    stage_cost + inter_claims
}

/// Total sumcheck rounds (col_bits + ring_bits).
pub fn sumcheck_rounds(level_d: u32, next_w_len: usize) -> usize {
    let ring_bits = level_d.trailing_zeros() as usize;
    let num_ring = next_w_len / level_d as usize;
    let col_bits = num_ring.next_power_of_two().trailing_zeros() as usize;
    col_bits + ring_bits
}

/// Single field element size in bytes.
pub const fn elem_bytes() -> usize {
    field_bytes(FIELD_BITS)
}

// Baseline sizing (with serialization headers).

/// Ring vector bytes with 8-byte length prefix.
pub fn baseline_ring_vec_bytes(ring_len: usize, ring_dim: u32, field_bits: u32) -> usize {
    8 + ring_len * ring_dim as usize * field_bytes(field_bits)
}

/// Sumcheck bytes with nested headers: outer 8 + rounds * (8 + degree * 16).
pub fn baseline_sumcheck_bytes(rounds: usize, degree: usize, field_bits: u32) -> usize {
    8 + rounds * (8 + degree * field_bytes(field_bits))
}

/// Packed digits bytes with 8-byte len + 1-byte tag prefix.
pub fn baseline_packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    8 + 1 + (num_elems * bits_per_elem as usize).div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_vec() {
        assert_eq!(ring_vec_bytes(1, 64, FIELD_BITS), 1024);
        assert_eq!(baseline_ring_vec_bytes(1, 64, FIELD_BITS), 1032);
        assert_eq!(ring_vec_bytes(1, 64, 32), 256);
        assert_eq!(baseline_ring_vec_bytes(1, 64, 32), 264);
    }

    #[test]
    fn stage1_lb2() {
        assert_eq!(stage1_bytes_optimized(17, 2, FIELD_BITS), 17 * 2 * 16);
        assert_eq!(stage1_bytes_optimized(17, 2, 32), 17 * 2 * 4);
    }

    #[test]
    fn stage1_lb3() {
        assert_eq!(stage1_bytes_optimized(17, 3, FIELD_BITS), 17 * 4 * 16);
    }

    #[test]
    fn stage1_lb4() {
        let cost = stage1_bytes_optimized(17, 4, FIELD_BITS);
        let expected_stages = (17 * 4 * 16) + (17 * 2 * 16);
        let expected_claims = 2 * 16;
        assert_eq!(cost, expected_stages + expected_claims);
    }

    #[test]
    fn rounds() {
        assert_eq!(sumcheck_rounds(64, 64 * 1024), 10 + 6);
    }
}
