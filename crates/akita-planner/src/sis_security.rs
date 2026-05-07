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
/// Regenerate with:
///
/// ```bash
/// sage -python scripts/gen_sis_table.py --search-cap 10000000000
/// ```
///
/// Requires a checkout of <https://github.com/malb/lattice-estimator> at
/// `../lattice-estimator` (sibling of the repo root) or pointed to by
/// `LATTICE_ESTIMATOR_PATH`.
///
/// Parameters:
/// - `d`: ring dimension of ZqX/(X^D + 1). One of {32, 64, 128}.
/// - `collision_inf`: worst-case L-infinity norm of the difference between
///   two valid witness vectors that collide under the SIS commitment.
///   For the B/D roles this is the balanced-digit bound `2^lb - 1`.
///   For the A role, the planner uses a challenge-aware proxy:
///   the raw digit collision is scaled by the maximum absolute coefficient
///   in the stage-1 challenge family and rounded up to the next supported
///   SIS bucket.
///
/// Entries that equal the search cap (10^10 for D=32/64, 5*10^10 for D=128)
/// should be read as "at least this large", not as tight cutoffs.
fn sis_max_widths(d: u32, collision_inf: u32) -> Option<[u64; MAX_RANK as usize]> {
    match (d, collision_inf) {
        // D=32  (search cap: 10^10)
        (32, 2) => Some([11_757, 4_359_823, 413_876_042, 10_000_000_000]),
        (32, 3) => Some([5_225, 1_937_699, 183_944_907, 8_645_254_247]),
        (32, 7) => Some([959, 355_903, 33_785_799, 1_587_903_841]),
        (32, 15) => Some([209, 77_507, 7_357_796, 345_810_169]),
        (32, 31) => Some([48, 18_147, 1_722_689, 80_964_920]),
        (32, 63) => Some([19, 4_393, 417_108, 19_603_751]),
        (32, 127) => Some([15, 1_081, 102_641, 4_824_061]),
        (32, 255) => Some([13, 268, 25_459, 1_196_574]),
        (32, 511) => Some([11, 66, 6_339, 297_974]),
        (32, 1023) => Some([10, 27, 1_581, 74_347]),
        (32, 2047) => Some([9, 23, 395, 18_568]),
        // D=64  (search cap: 10^10)
        (64, 2) => Some([2_179_911, 9_725_911_028, 10_000_000_000, 10_000_000_000]),
        (64, 3) => Some([968_849, 4_322_627_123, 10_000_000_000, 10_000_000_000]),
        (64, 7) => Some([177_951, 793_951_920, 10_000_000_000, 10_000_000_000]),
        (64, 15) => Some([38_753, 172_905_084, 10_000_000_000, 10_000_000_000]),
        (64, 31) => Some([9_073, 40_482_460, 10_000_000_000, 10_000_000_000]),
        (64, 63) => Some([2_196, 9_801_875, 6_215_651_928, 10_000_000_000]),
        (64, 127) => Some([540, 2_412_030, 1_529_538_254, 10_000_000_000]),
        (64, 255) => Some([134, 598_287, 379_391_349, 10_000_000_000]),
        (64, 511) => Some([33, 148_987, 94_476_976, 10_000_000_000]),
        (64, 1023) => Some([13, 37_173, 23_573_090, 5_520_444_163]),
        (64, 2047) => Some([11, 9_284, 5_887_515, 1_378_762_947]),
        // D=128  (search cap: 5*10^10)
        (128, 2) => Some([
            4_862_955_514,
            50_000_000_000,
            50_000_000_000,
            50_000_000_000,
        ]),
        (128, 3) => Some([
            2_161_313_561,
            50_000_000_000,
            50_000_000_000,
            50_000_000_000,
        ]),
        (128, 7) => Some([396_975_960, 50_000_000_000, 50_000_000_000, 50_000_000_000]),
        (128, 15) => Some([86_452_542, 50_000_000_000, 50_000_000_000, 50_000_000_000]),
        (128, 31) => Some([20_241_230, 50_000_000_000, 50_000_000_000, 50_000_000_000]),
        (128, 63) => Some([4_900_937, 50_000_000_000, 50_000_000_000, 50_000_000_000]),
        (128, 255) => Some([299_143, 44_423_720_955, 50_000_000_000, 50_000_000_000]),
        (128, 511) => Some([74_493, 11_062_505_333, 50_000_000_000, 50_000_000_000]),
        (128, 1023) => Some([18_586, 2_760_222_081, 50_000_000_000, 50_000_000_000]),
        (128, 2047) => Some([4_642, 689_381_473, 50_000_000_000, 50_000_000_000]),
        (128, 4095) => Some([1_159, 172_261_205, 50_000_000_000, 50_000_000_000]),
        (128, 8191) => Some([289, 43_054_786, 50_000_000_000, 50_000_000_000]),
        _ => None,
    }
}

/// Expose the raw SIS width array for a given `(d, collision_inf)` pair.
pub fn sis_max_widths_public(d: u32, collision_inf: u32) -> Option<[u64; MAX_RANK as usize]> {
    sis_max_widths(d, collision_inf)
}

/// Returns the smallest Module-SIS rank in `1..=MAX_RANK` that provides
/// 128-bit security for an SIS instance with `width` ring-element columns
/// at ring dimension `d` and collision bound `collision_inf`.
///
/// Returns `None` if the `(d, collision_inf)` pair is not in the table, or
/// if no rank up to `MAX_RANK` can accommodate the requested width.
pub fn min_rank_for_secure_width(d: u32, collision_inf: u32, width: u64) -> Option<u32> {
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
    const D64: &[u32] = &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047];
    const D128: &[u32] = &[2, 3, 7, 15, 31, 63, 255, 511, 1023, 2047, 4095, 8191];
    let buckets = match d {
        32 => D32,
        64 => D64,
        128 => D128,
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
        assert_eq!(ceil_supported_collision(128, 62), Some(63));
        assert_eq!(ceil_supported_collision(128, 248), Some(255));
        assert_eq!(ceil_supported_collision(128, 7_812), Some(8191));
    }

    #[test]
    fn d128_rank_lookup() {
        assert_eq!(min_rank_for_secure_width(128, 2, 4_862_955_514), Some(1));
        assert_eq!(min_rank_for_secure_width(128, 2, 4_862_955_515), Some(2));
        assert_eq!(min_rank_for_secure_width(128, 63, 4_900_937), Some(1));
        assert_eq!(min_rank_for_secure_width(128, 63, 4_900_938), Some(2));
        assert_eq!(min_rank_for_secure_width(128, 31, 20_241_230), Some(1));
        assert_eq!(min_rank_for_secure_width(128, 31, 20_241_231), Some(2));
    }

    #[test]
    fn rank1_widths_are_monotone_in_collision_bound() {
        let d32 = [2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
            .into_iter()
            .map(|collision| sis_max_widths(32, collision).unwrap()[0])
            .collect::<Vec<_>>();
        assert!(d32.windows(2).all(|pair| pair[0] >= pair[1]));

        let d64 = [2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]
            .into_iter()
            .map(|collision| sis_max_widths(64, collision).unwrap()[0])
            .collect::<Vec<_>>();
        assert!(d64.windows(2).all(|pair| pair[0] >= pair[1]));

        let d128 = [2, 3, 7, 15, 31, 63, 255, 511, 1023, 2047, 4095, 8191]
            .into_iter()
            .map(|collision| sis_max_widths(128, collision).unwrap()[0])
            .collect::<Vec<_>>();
        assert!(d128.windows(2).all(|pair| pair[0] >= pair[1]));
    }
}
