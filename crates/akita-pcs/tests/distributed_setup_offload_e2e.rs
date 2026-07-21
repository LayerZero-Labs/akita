//! End-to-end coverage for the mixed distributed (multi-chunk) + recursive
//! setup-offload profile.
//!
//! This test uses `RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>` (the
//! production `W8R2` preset, `fp128_d64_onehot_recursive_multi_chunk_w8r2`
//! family): two precommitted singleton groups at `nv=16` and a two-polynomial
//! final group at `nv=32`. That schedule combines the `W8R2` chunked witness
//! layout (8 chunks on the two leading fold levels) with recursive setup
//! offloading (Stage-3 setup-product sum-check and a carried setup-prefix
//! opening), so a successful proof exercises the mix: chunked folds that also
//! run the offloaded setup-contribution path.

#![allow(missing_docs)]

mod common;

use akita_types::SetupContributionMode;
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"distributed_setup_offload_e2e/w8r2";

/// Assert the exact shipped `W8R2` profile shape, not just "some mixed fold".
///
/// The generated table is exact for the `(32, 2) + two (16, 1)` profiling key, so
/// the test pins every distinguishing fact. This catches `W4R2` vs `W8R2`, a
/// level-0/level-1 mode swap, only one mixed leading fold, and a missing/extra
/// setup-prefix handoff — none of which a bare "any chunked recursive fold" check
/// would detect. `D64OneHot` (single-chunk) also fails this on levels 0/1.
fn assert_w8r2_profile_shape(schedule: &Schedule) {
    assert!(
        schedule.folds.len() >= 3,
        "W8R2 profile must have at least three fold levels, got {}",
        schedule.folds.len()
    );

    // Levels 0 and 1: both chunked W8R2 (8 chunks over 2 leading levels) AND both
    // running the Stage-3 setup-product sum-check (`Recursive`).
    for level in [0usize, 1usize] {
        let params = &schedule.folds[level].params;
        assert_eq!(
            params.witness_chunk.num_chunks, 8,
            "level {level} must be chunked with num_chunks == 8 (W8R2)"
        );
        assert_eq!(
            params.witness_chunk.num_activated_levels, 2,
            "level {level} must carry num_activated_levels == 2 (W8R2)"
        );
        assert_eq!(
            params.setup_contribution_mode,
            SetupContributionMode::Recursive,
            "level {level} must run the recursive setup-offload path"
        );
    }

    // Level 0 produces the first setup prefix (no incoming prefix); level 1
    // consumes it and produces its own.
    assert!(
        schedule.folds[0].params.setup_prefix.is_none(),
        "root fold (level 0) must not carry an incoming setup prefix"
    );
    assert!(
        schedule.folds[1].params.setup_prefix.is_some(),
        "level 1 must consume the level-0 setup prefix"
    );

    // Level 2 is the single-chunk `Direct` fold that consumes the level-1 prefix.
    let level2 = &schedule.folds[2].params;
    assert_eq!(
        level2.witness_chunk.num_chunks, 1,
        "level 2 must be single-chunk (chunking activates only levels 0 and 1)"
    );
    assert_eq!(
        level2.setup_contribution_mode,
        SetupContributionMode::Direct,
        "level 2 must be Direct (no Stage-3 sum-check after the activated window)"
    );
    assert!(
        level2.setup_prefix.is_some(),
        "level 2 must consume the level-1 setup prefix"
    );
}

#[test]
fn mix_multi_chunk_recursive_profile_proves_and_verifies() {
    recursive_multi_group_round_trip::<fp128::D64OneHotMultiChunk>(
        TRANSCRIPT_DOMAIN,
        assert_w8r2_profile_shape,
    );
}
