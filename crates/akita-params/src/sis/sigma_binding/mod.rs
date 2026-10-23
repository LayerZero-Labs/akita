//! Module-SIS pricing of the commitments the Whiteout Σ-protocol opens, at
//! its collision radius (`specs/whiteout-akita.md` §5.5.2).
//!
//! The Σ-protocol binds each commitment matrix it opens at the
//! coefficient-ℓ∞ radius `4Γ_Δ B` of Thm `wo:thm:knowledge`, owned by
//! jolt-zk ([`sigma_binding_radius`]). That radius is far above Akita's
//! honest fold bounds, so these matrices are priced by a separate generated
//! table (`akita-sis-estimator/examples/sigma_binding_width_table.rs`, the
//! same ADPS16 quantum ℓ∞ model as Akita's table) with its own digest; the
//! clear table, and every clear schedule, is unaffected.
//!
//! A commitment to range-checked digits, which the extractor needs bound only
//! at the digit-difference bound, is priced by the planner instead
//! (`akita_schedules::digit_commitment_min_rank`).

mod generated;

use super::ajtai_key::SisModulusProfileId;

/// The coefficient-ℓ∞ radius at which a matrix the Σ-protocol opens must be
/// Module-SIS hard: jolt-zk's `binding_collision_radius`.
#[must_use]
pub fn sigma_binding_radius() -> u128 {
    jolt_zk::whiteout::binding_collision_radius()
}

/// The smallest module rank at which a `d`-dimensional matrix of `width`
/// ring columns is Module-SIS hard at [`sigma_binding_radius`] over
/// `profile`'s modulus, or `None` if no rank up to the table's maximum is,
/// or the table does not cover the profile, the dimension, or the current
/// radius (a stale table, regenerated with the generator).
#[must_use]
pub fn sigma_binding_min_rank(
    profile: SisModulusProfileId,
    d: usize,
    width: usize,
) -> Option<usize> {
    if width == 0 || generated::SIGMA_BINDING_RADIUS != sigma_binding_radius() {
        return None;
    }
    let width = u64::try_from(width).ok()?;
    generated::sigma_binding_max_widths(profile.row_owner(), u32::try_from(d).ok()?)?
        .iter()
        .position(|&max_width| width <= max_width)
        .map(|index| index + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha3::{Digest, Sha3_256};

    /// The cells are the checked-in certified CSV's, generated at the
    /// radius jolt-zk owns today; a new radius or digit count needs the
    /// generator rerun.
    #[test]
    fn cells_match_the_certified_csv_and_the_current_radius() {
        let csv =
            include_bytes!("../../../../akita-sis-estimator/data/sigma_binding_infinity_width.csv");
        let digest: [u8; 32] = Sha3_256::digest(csv).into();
        assert_eq!(digest, generated::SIGMA_BINDING_TABLE_DIGEST);
        assert_eq!(generated::SIGMA_BINDING_RADIUS, sigma_binding_radius());
    }

    #[test]
    fn the_binding_rank_grows_with_width_and_covers_no_empty_matrix() {
        let profile = SisModulusProfileId::Q128OffsetA7F7;
        assert_eq!(sigma_binding_min_rank(profile, 64, 0), None);
        assert_eq!(sigma_binding_min_rank(profile, 32, 1), None);
        for d in [64, 128, 256, 512] {
            let ranks = [1, 10, 1_000, 10_000]
                .map(|width| sigma_binding_min_rank(profile, d, width).unwrap());
            assert!(
                ranks.windows(2).all(|pair| pair[0] <= pair[1]),
                "{d}: {ranks:?}"
            );
        }
    }
}
