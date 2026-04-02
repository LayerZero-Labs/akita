/// Maximum Module-SIS rank considered by the planner.
///
/// The SIS width table below was generated for ranks 1..=MAX_RANK only.
/// If this constant is changed, the table must be re-verified with the
/// lattice estimator.
pub const MAX_RANK: u32 = 4;
const _: () = assert!(MAX_RANK == 4, "SIS width table only covers ranks 1..=4");

/// Max secure SIS width (in ring elements) at 128-bit security, indexed by
/// `[rank - 1]` for ranks `1..=MAX_RANK`.
///
/// Verified with lattice-estimator (BDGL16 + lgsa, q = 2^128 - 275).
///
/// Parameters:
/// - `d`: ring dimension of ZqX/(X^D + 1). One of {32, 64}.
/// - `collision_inf`: worst-case L-infinity norm of the difference between
///   two valid witness vectors that collide under the SIS commitment.
///   For the B/D roles this is the balanced-digit bound `2^lb - 1`.
///   For the A role, the planner uses a challenge-aware proxy:
///   the raw digit collision is scaled by the maximum absolute coefficient
///   in the stage-1 challenge family and rounded up to the next supported
///   SIS bucket.
fn sis_max_widths(d: u32, collision_inf: u32) -> Option<[usize; MAX_RANK as usize]> {
    match (d, collision_inf) {
        // D=32
        (32, 2) => Some([11_757, 4_359_823, 5_000_000, 5_000_000]),
        (32, 3) => Some([5_225, 1_937_699, 5_000_000, 5_000_000]),
        (32, 7) => Some([959, 355_903, 5_000_000, 5_000_000]),
        (32, 15) => Some([209, 77_507, 7_357_796, 5_000_000]),
        (32, 31) => Some([48, 18_147, 1_722_689, 5_000_000]),
        (32, 63) => Some([19, 4_393, 417_108, 5_000_000]),
        (32, 127) => Some([15, 1_081, 102_641, 4_824_061]),
        (32, 255) => Some([13, 268, 25_459, 1_196_574]),
        (32, 511) => Some([11, 66, 6_339, 297_974]),
        (32, 1023) => Some([10, 27, 1_581, 74_347]),
        (32, 2047) => Some([9, 23, 395, 18_568]),
        // D=64
        (64, 2) => Some([2_179_911, 20_000_000, 20_000_000, 20_000_000]),
        (64, 3) => Some([968_849, 20_000_000, 20_000_000, 20_000_000]),
        (64, 7) => Some([177_951, 20_000_000, 20_000_000, 20_000_000]),
        (64, 15) => Some([38_753, 20_000_000, 20_000_000, 20_000_000]),
        (64, 31) => Some([9_073, 20_000_000, 20_000_000, 20_000_000]),
        (64, 63) => Some([2_196, 9_801_875, 20_000_000, 20_000_000]),
        (64, 127) => Some([540, 2_412_030, 20_000_000, 20_000_000]),
        (64, 255) => Some([134, 598_287, 20_000_000, 20_000_000]),
        (64, 511) => Some([33, 148_987, 20_000_000, 20_000_000]),
        _ => None,
    }
}

/// Returns the smallest Module-SIS rank in `1..=MAX_RANK` that provides
/// 128-bit security for an SIS instance with `width` ring-element columns
/// at ring dimension `d` and collision bound `collision_inf`.
///
/// Returns `None` if the `(d, collision_inf)` pair is not in the table, or
/// if no rank up to `MAX_RANK` can accommodate the requested width.
pub fn min_rank_for_secure_width(d: u32, collision_inf: u32, width: usize) -> Option<u32> {
    let widths = sis_max_widths(d, collision_inf)?;
    for (i, &max_w) in widths.iter().enumerate() {
        if width <= max_w {
            return Some((i + 1) as u32);
        }
    }
    None
}

/// Round a requested collision bound up to the next supported SIS bucket.
pub fn ceil_supported_collision(d: u32, collision_inf: u32) -> Option<u32> {
    const D32: &[u32] = &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047];
    const D64: &[u32] = &[2, 3, 7, 15, 31, 63, 127, 255, 511];
    let buckets = match d {
        32 => D32,
        64 => D64,
        _ => return None,
    };
    buckets
        .iter()
        .copied()
        .find(|&bucket| collision_inf <= bucket)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_lookup() {
        assert_eq!(min_rank_for_secure_width(32, 7, 500), Some(1));
        assert_eq!(min_rank_for_secure_width(32, 7, 959), Some(1));
        assert_eq!(min_rank_for_secure_width(32, 7, 960), Some(2));
    }

    #[test]
    fn exceeds_max_rank() {
        assert_eq!(min_rank_for_secure_width(32, 127, 5_000_000), None);
    }

    #[test]
    fn missing_entry() {
        assert_eq!(min_rank_for_secure_width(8, 7, 10), None);
    }

    #[test]
    fn ceil_collision_bucket() {
        assert_eq!(ceil_supported_collision(32, 248), Some(255));
        assert_eq!(ceil_supported_collision(64, 62), Some(63));
    }
}
