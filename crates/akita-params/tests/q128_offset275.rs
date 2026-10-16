//! `SisModulusProfileId::Q128Offset275`: identity, wire tags, and row sharing.
//!
//! The profile owns no generated SIS rows. Every lookup resolves
//! `SisModulusProfileId::row_owner` first and reads the `Q128OffsetA7F7`
//! rows. `specs/sis-quantum128-scalar-n-table.md` states why.

#![allow(missing_docs)]

use akita_params::descriptor_bytes::sis_modulus_profile_tag;
use akita_params::sis::compression::compression_sis_cell;
use akita_params::sis::{
    ceil_supported_linf_bound, inner_coeff_linf_bounds, min_secure_l2_rank, min_secure_rank,
    sis_l2_table_key_for_collision_sq, sis_role_cell, sis_role_cells, SisL2TableDigest,
};
use akita_params::{
    compression_ring_dimensions, SisModulusProfileId, SisRoleCell, SisTableDigest, SisTableKey,
    DEFAULT_SIS_SECURITY_POLICY,
};

const OWNER: SisModulusProfileId = SisModulusProfileId::Q128OffsetA7F7;
const ADDED: SisModulusProfileId = SisModulusProfileId::Q128Offset275;

#[test]
fn adding_the_profile_moves_no_existing_identity_or_tag() {
    // (profile, modulus, catalog tag, descriptor tag, name)
    let existing = [
        (
            SisModulusProfileId::Q32Offset99,
            (1u128 << 32) - 99,
            1,
            0,
            "Q32Offset99",
        ),
        (
            SisModulusProfileId::Q64Offset59,
            (1u128 << 64) - 59,
            2,
            1,
            "Q64Offset59",
        ),
        (
            SisModulusProfileId::Q128OffsetA7F7,
            u128::MAX - ((1u128 << 32) - 22_537) + 1,
            3,
            2,
            "Q128OffsetA7F7",
        ),
    ];
    for (profile, modulus, tag, descriptor_tag, name) in existing {
        assert_eq!(profile.modulus(), modulus);
        assert_eq!(profile.tag(), tag);
        assert_eq!(SisModulusProfileId::from_tag(tag), Some(profile));
        assert_eq!(sis_modulus_profile_tag(profile), descriptor_tag);
        assert_eq!(profile.name(), name);
        assert_eq!(profile.row_owner(), profile);
    }
    assert_eq!(SisModulusProfileId::default(), OWNER);

    assert_eq!(ADDED.modulus(), u128::MAX - 274);
    assert_eq!(ADDED.field_bits(), 128);
    assert_eq!(ADDED.name(), "Q128Offset275");
    assert!(ADDED.matches_modulus(u128::MAX - 274));
    assert!(!ADDED.matches_modulus(OWNER.modulus()));
    assert!(!OWNER.matches_modulus(ADDED.modulus()));
    // The added profile sorts after every existing one.
    assert!(existing.iter().all(|&(profile, ..)| profile < ADDED));
    // Catalog tag 5 and descriptor tag 4: one apart, like every other
    // profile. Catalog tag 4 stays unassigned.
    assert_eq!(ADDED.tag(), 5);
    assert_eq!(SisModulusProfileId::from_tag(5), Some(ADDED));
    assert_eq!(SisModulusProfileId::from_tag(4), None);
    assert_eq!(sis_modulus_profile_tag(ADDED), 4);
    // It reads the rows of the other 128-bit profile, which owns its own.
    assert_eq!(ADDED.row_owner(), OWNER);
    assert_eq!(ADDED.row_owner().row_owner(), OWNER);
}

fn table_key(cell: &SisRoleCell, profile: SisModulusProfileId) -> SisTableKey {
    SisTableKey {
        policy: DEFAULT_SIS_SECURITY_POLICY,
        table_digest: SisTableDigest::CURRENT,
        modulus_profile: profile,
        role: cell.role,
        ring_dimension: cell.ring_dimension,
        coeff_linf_bound: cell.coeff_linf_bound,
    }
}

/// Largest width that `min_secure_rank` admits at `rank` or below, found by
/// bisection. `min_secure_rank` is the first rank whose cutoff covers the
/// width, so these maxima over every rank determine it completely.
fn max_width_up_to_rank(key: SisTableKey, rank: usize) -> Option<u64> {
    let admits = |width| min_secure_rank(key, width).is_some_and(|needed| needed <= rank);
    if !admits(0) {
        return None;
    }
    let (mut admitted, mut rejected) = (0u64, u64::MAX);
    if admits(rejected) {
        return Some(rejected);
    }
    while rejected - admitted > 1 {
        let middle = admitted + (rejected - admitted) / 2;
        if admits(middle) {
            admitted = middle;
        } else {
            rejected = middle;
        }
    }
    Some(admitted)
}

#[test]
fn the_profile_reads_exactly_the_q128_infinity_rows() {
    // The added profile contributes no cell, so the generated table and its
    // digest have the same inputs as before.
    let cells = sis_role_cells();
    assert!(cells.iter().all(|cell| cell.modulus_profile != ADDED));

    // Cells of every profile: q32 and q64 cells that q128 does not cover are
    // the negative cases.
    let (mut shared, mut uncovered) = (0usize, 0usize);
    for cell in &cells {
        let (role, dimension, bound) = (cell.role, cell.ring_dimension, cell.coeff_linf_bound);
        let owner_cell = sis_role_cell(role, OWNER, dimension, bound);
        let added_cell = sis_role_cell(role, ADDED, dimension, bound);
        assert_eq!(
            added_cell.map(|cell| (cell.max_module_rank, cell.required_max_width)),
            owner_cell.map(|cell| (cell.max_module_rank, cell.required_max_width)),
            "role={role:?}, D={dimension}, bound={bound}",
        );
        // A covered cell keeps the exact profile it was asked for.
        assert!(added_cell.is_none_or(|cell| cell.modulus_profile == ADDED));

        for raw_bound in [bound - 1, bound, bound + 1] {
            let ceil = |profile| {
                ceil_supported_linf_bound(
                    DEFAULT_SIS_SECURITY_POLICY,
                    SisTableDigest::CURRENT,
                    profile,
                    role,
                    dimension,
                    raw_bound,
                )
            };
            assert_eq!(ceil(ADDED), ceil(OWNER));
        }

        let Some(owner_cell) = owner_cell else {
            assert_eq!(min_secure_rank(table_key(cell, ADDED), 1), None);
            uncovered += 1;
            continue;
        };
        for rank in 1..=usize::try_from(owner_cell.max_module_rank).unwrap() {
            assert_eq!(
                max_width_up_to_rank(table_key(cell, ADDED), rank),
                max_width_up_to_rank(table_key(cell, OWNER), rank),
                "role={role:?}, D={dimension}, bound={bound}, rank={rank}",
            );
        }
        shared += 1;
    }
    assert!(shared > 0);
    assert!(uncovered > 0);

    for dimension in [32, 64, 128, 256, 512, 1024] {
        assert_eq!(
            inner_coeff_linf_bounds(ADDED, dimension),
            inner_coeff_linf_bounds(OWNER, dimension)
        );
    }
}

#[test]
fn the_profile_reads_exactly_the_q128_compression_and_euclidean_rows() {
    assert_eq!(
        compression_ring_dimensions(ADDED),
        compression_ring_dimensions(OWNER)
    );
    for dimension in [8, 16, 32, 64] {
        assert_eq!(
            compression_sis_cell(ADDED, dimension, 1).map(|cell| cell.sis_max_width),
            compression_sis_cell(OWNER, dimension, 1).map(|cell| cell.sis_max_width),
        );
    }
    assert_eq!(
        compression_sis_cell(ADDED, 16, 1).map(|cell| cell.modulus_profile),
        Some(ADDED)
    );

    let mut shared = 0usize;
    for dimension in [32, 64, 128, 256, 512, 1024] {
        for log_bucket in [1, 40, 84] {
            let key = |profile| {
                sis_l2_table_key_for_collision_sq(
                    DEFAULT_SIS_SECURITY_POLICY,
                    SisL2TableDigest::CURRENT,
                    profile,
                    dimension,
                    1u128 << log_bucket,
                )
            };
            let (added_key, owner_key) = (key(ADDED), key(OWNER));
            assert_eq!(added_key.is_some(), owner_key.is_some());
            let (Some(added_key), Some(owner_key)) = (added_key, owner_key) else {
                continue;
            };
            assert_eq!(added_key.modulus_profile, ADDED);
            shared += 1;
            for width in [1, 1 << 10, 1 << 20, 1 << 30, 1 << 40] {
                assert_eq!(
                    min_secure_l2_rank(added_key, width),
                    min_secure_l2_rank(owner_key, width)
                );
            }
        }
    }
    assert!(shared > 0);
}
