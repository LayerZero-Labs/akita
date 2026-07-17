//! End-to-end coverage for the generated recursive setup-offload profile.
//!
//! This test intentionally uses the profile emitted in
//! `fp128_d64_onehot_recursive`: two precommitted singleton groups at `nv=16`
//! and a two-polynomial final group at `nv=32`. That generated schedule carries
//! setup-prefix metadata, so a successful recursive proof exercises the
//! offloaded setup-contribution path rather than the inline direct setup scan.

#![allow(missing_docs)]

mod common;

use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_e2e/generated_onehot";

#[test]
fn generated_recursive_onehot_profile_proves_with_setup_offload() {
    // Single-chunk base: the shared round-trip already asserts the setup-prefix
    // metadata and stage-3 setup sumcheck, so no profile-specific schedule check
    // is needed here.
    recursive_multi_group_round_trip::<fp128::D64OneHot>(TRANSCRIPT_DOMAIN, |_schedule| {});
}
