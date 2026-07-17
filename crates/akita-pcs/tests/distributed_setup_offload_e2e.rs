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

/// The mix-defining invariant: at least one fold is BOTH chunked (`W8R2`) and
/// runs the recursive setup-offload path. `D64OneHot` (single-chunk) would fail
/// this, which is exactly what distinguishes this profile from the plain
/// recursive one in `recursive_setup_e2e`.
fn assert_chunked_recursive_fold(schedule: &Schedule) {
    assert!(
        schedule.folds.iter().any(|fold| {
            fold.params.witness_chunk.num_chunks > 1
                && fold.params.setup_contribution_mode == SetupContributionMode::Recursive
        }),
        "mix profile must contain a fold that is BOTH chunked and recursive"
    );
}

#[test]
fn mix_multi_chunk_recursive_profile_proves_and_verifies() {
    recursive_multi_group_round_trip::<fp128::D64OneHotMultiChunk>(
        TRANSCRIPT_DOMAIN,
        assert_chunked_recursive_fold,
    );
}
