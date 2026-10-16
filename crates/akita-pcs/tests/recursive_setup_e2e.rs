//! End-to-end coverage for the generated recursive setup-offload profile.
//!
//! This test intentionally uses the profile emitted in
//! `fp128_onehot`: two precommitted singleton groups at `nv=16`
//! and a two-polynomial final group at `nv=32`. That generated schedule carries
//! setup-prefix metadata, so a successful recursive proof exercises the
//! offloaded setup-contribution path rather than the inline direct setup scan.
//!
//! A second test proves both fp128 recursive one-hot families on one setup
//! built from the union of their requirements. At the round-trip bounds the two
//! families require distinct setup-prefix slot sets, so each proof imports its
//! own slots from a registry that also carries the other family's slots.
//!
//! The fixtures are production-sized and must be run explicitly in an optimized
//! profile:
//!
//! `cargo test --release -p akita-pcs --test recursive_setup_e2e --features profile-ci -- --ignored`

#![cfg(feature = "profile-ci")]
#![allow(missing_docs)]

mod common;

use akita_config::SetupRequirements;
use akita_cpu_backend::CpuBackend;
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"recursive_setup_e2e/generated_onehot";

#[test]
#[ignore = "production-sized profile E2E; run explicitly with --release"]
fn generated_recursive_onehot_profile_proves_with_setup_offload() {
    // Single-chunk base: the shared round-trip already asserts the setup-prefix
    // metadata and stage-3 setup sumcheck, so no profile-specific schedule check
    // is needed here.
    recursive_multi_group_round_trip::<fp128::OneHot>(
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
        TRANSCRIPT_DOMAIN,
        |_schedule| {},
    );
}

#[test]
#[ignore = "production-sized profile E2E; run explicitly with --release"]
fn combined_recursive_setup_proves_both_onehot_families() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let onehot = SetupRequirements::from_catalog::<fp128::OneHot>(
            load_workspace_scheme::<fp128::OneHot>(
                akita_params::ChunkedWitnessCfg::default_non_chunked(),
            )
            .expect("recursive one-hot catalog")
            .schedules(),
            RECURSIVE_ROUND_TRIP_NV,
            RECURSIVE_ROUND_TRIP_POLYS,
        )
        .expect("recursive one-hot requirements");
        let multichunk = SetupRequirements::from_catalog::<fp128::OneHotMultiChunk>(
            load_workspace_scheme::<fp128::OneHotMultiChunk>(
                akita_params::MultiChunkProfileId::W8R2.cfg(),
            )
            .expect("recursive multi-chunk catalog")
            .schedules(),
            RECURSIVE_ROUND_TRIP_NV,
            RECURSIVE_ROUND_TRIP_POLYS,
        )
        .expect("recursive multi-chunk requirements");
        assert_ne!(
            onehot.prefix_slot_ids(),
            multichunk.prefix_slot_ids(),
            "the families must require distinct setup-prefix slot sets"
        );

        let setup = akita_pcs::new_prover_setup::<F>(
            &onehot.union(multichunk).expect("combined requirements"),
        )
        .expect("combined setup");
        let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
        recursive_multi_group_round_trip_on::<fp128::OneHot>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
            &setup,
            &stack,
            b"recursive_setup_e2e/combined/onehot",
            |_schedule| {},
        );
        recursive_multi_group_round_trip_on::<fp128::OneHotMultiChunk>(
            akita_params::MultiChunkProfileId::W8R2.cfg(),
            &setup,
            &stack,
            b"recursive_setup_e2e/combined/multi_chunk",
            |_schedule| {},
        );
    });
}
