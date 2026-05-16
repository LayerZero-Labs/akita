//! Planner-facing accessors for generated Module-SIS security floors.
//!
//! The canonical table lives in `akita-types` so runtime config validation and
//! offline planner search consume the same generated data.

pub use akita_types::SisModulusFamily;

/// Maximum Module-SIS rank covered by the generated floor table.
pub const MAX_RANK: u32 = akita_types::generated::sis_floor::MAX_RANK as u32;

/// Expose the raw SIS width array for a given `(family, d, collision_inf)` pair.
pub fn sis_max_widths_public(
    family: SisModulusFamily,
    d: u32,
    collision_inf: u32,
) -> Option<[u64; akita_types::generated::sis_floor::MAX_RANK]> {
    akita_types::generated::sis_floor::sis_max_widths(family, d, collision_inf)
}

/// Returns the smallest generated Module-SIS rank that supports `width`.
pub fn min_rank_for_secure_width(
    family: SisModulusFamily,
    d: u32,
    collision_inf: u32,
    width: u64,
) -> Option<u32> {
    akita_types::generated::sis_floor::min_rank_for_secure_width(family, d, collision_inf, width)
        .and_then(|rank| u32::try_from(rank).ok())
}

/// Round a requested collision bound up to the next supported SIS bucket.
pub fn ceil_supported_collision(
    family: SisModulusFamily,
    d: u32,
    collision_inf: u32,
) -> Option<u32> {
    akita_types::generated::sis_floor::ceil_supported_collision(family, d, collision_inf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q128_rank_lookup_matches_legacy_floor() {
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q128, 32, 7, 500),
            Some(1)
        );
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q128, 32, 7, 959),
            Some(1)
        );
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q128, 32, 7, 960),
            Some(2)
        );
    }

    #[test]
    fn small_fields_do_not_reuse_q128_floor() {
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q32, 32, 7, 140),
            Some(3)
        );
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q32, 32, 7, 141),
            Some(4)
        );
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q64, 32, 7, 959),
            Some(2)
        );
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q128, 32, 7, 959),
            Some(1)
        );
    }

    #[test]
    fn larger_small_field_ring_dimensions_are_available() {
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q32, 128, 7, 960),
            Some(2)
        );
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q32, 512, 7, 960),
            Some(1)
        );
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q64, 256, 7, 960),
            Some(1)
        );
    }

    #[test]
    fn exceeds_max_rank() {
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q32, 32, 127, 16),
            None
        );
    }

    #[test]
    fn missing_entry() {
        assert_eq!(
            min_rank_for_secure_width(SisModulusFamily::Q128, 512, 7, 10),
            None
        );
    }

    #[test]
    fn ceil_collision_bucket() {
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q32, 512, 248),
            Some(255)
        );
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q64, 256, 62),
            Some(63)
        );
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q128, 128, 62),
            Some(63)
        );
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q128, 32, 248),
            Some(255)
        );
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q128, 64, 126),
            Some(127)
        );
        assert_eq!(
            ceil_supported_collision(SisModulusFamily::Q128, 128, 64),
            Some(127)
        );
    }

    #[test]
    fn rank1_widths_are_monotone_in_collision_bound() {
        for (family, d, buckets) in [
            (
                SisModulusFamily::Q32,
                512,
                &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047][..],
            ),
            (
                SisModulusFamily::Q64,
                256,
                &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047][..],
            ),
            (
                SisModulusFamily::Q128,
                128,
                &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047][..],
            ),
        ] {
            let widths = buckets
                .iter()
                .map(|&collision| sis_max_widths_public(family, d, collision).unwrap()[0])
                .collect::<Vec<_>>();
            assert!(widths.windows(2).all(|pair| pair[0] >= pair[1]));
        }
    }
}
