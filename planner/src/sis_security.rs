/// Max secure SIS width (in ring elements) at 128-bit security for
/// `(D, collision_inf, rank)`, verified with lattice-estimator
/// (BDGL16 + lgsa, q = 2^128 - 275).
///
/// Indexed as `[rank-1]` for ranks 1..4.
/// collision_inf values: root onehot A-role = 2, balanced base-2^lb digits = 2^lb - 1.
fn sis_max_widths(d: u32, collision_inf: u32) -> Option<[usize; 4]> {
    match (d, collision_inf) {
        // D=16
        (16, 2) => Some([158, 10_450, 260_593, 200_000]),
        (16, 3) => Some([158, 10_450, 260_593, 200_000]),
        (16, 7) => Some([31, 1_919, 47_864, 200_000]),
        (16, 15) => Some([21, 418, 10_423, 155_015]),
        (16, 31) => Some([18, 97, 2_440, 36_294]),
        (16, 63) => Some([15, 38, 590, 8_787]),
        (16, 127) => Some([14, 30, 145, 2_162]),
        // D=32
        (32, 2) => Some([11_757, 4_359_823, 5_000_000, 5_000_000]),
        (32, 3) => Some([5_225, 1_937_699, 5_000_000, 5_000_000]),
        (32, 7) => Some([959, 355_903, 5_000_000, 5_000_000]),
        (32, 15) => Some([209, 77_507, 7_357_796, 5_000_000]),
        (32, 31) => Some([48, 18_147, 1_722_689, 5_000_000]),
        (32, 63) => Some([19, 4_393, 417_108, 5_000_000]),
        (32, 127) => Some([15, 1_081, 102_641, 4_824_061]),
        // D=64
        (64, 2) => Some([2_179_911, 20_000_000, 20_000_000, 20_000_000]),
        (64, 3) => Some([968_849, 20_000_000, 20_000_000, 20_000_000]),
        (64, 7) => Some([177_951, 20_000_000, 20_000_000, 20_000_000]),
        (64, 15) => Some([38_753, 20_000_000, 20_000_000, 20_000_000]),
        (64, 31) => Some([9_073, 20_000_000, 20_000_000, 20_000_000]),
        (64, 63) => Some([2_196, 9_801_875, 20_000_000, 20_000_000]),
        (64, 127) => Some([540, 2_412_030, 20_000_000, 20_000_000]),
        _ => None,
    }
}

/// Smallest Module-SIS rank (1..4) whose width cap covers `width`.
///
/// Returns `None` if no rank up to 4 is sufficient.
pub fn min_rank_for_secure_width(d: u32, collision_inf: u32, width: usize) -> Option<u32> {
    let widths = sis_max_widths(d, collision_inf)?;
    for (i, &max_w) in widths.iter().enumerate() {
        if width <= max_w {
            return Some((i + 1) as u32);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_lookup() {
        assert_eq!(min_rank_for_secure_width(32, 7, 500), Some(1));
        assert_eq!(min_rank_for_secure_width(32, 7, 959), Some(1));
        assert_eq!(min_rank_for_secure_width(32, 7, 960), Some(2));
        assert_eq!(min_rank_for_secure_width(16, 7, 32), Some(2));
        assert_eq!(min_rank_for_secure_width(16, 7, 31), Some(1));
    }

    #[test]
    fn missing_entry() {
        assert_eq!(min_rank_for_secure_width(8, 7, 10), None);
    }
}
