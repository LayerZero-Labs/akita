//! End-to-end coverage for the mixed distributed (multi-chunk) + recursive
//! setup-offload profile.
//!
//! This test uses `RecursiveCommitmentConfig<fp128::OneHotMultiChunk>` (the
//! production `W8R2` preset, `fp128_d64_onehot_recursive_multi_chunk_w8r2`
//! family): two precommitted singleton groups at `nv=16` and a two-polynomial
//! final group at `nv=32`. That schedule combines the `W8R2` chunked witness
//! layout (8 chunks on the two leading fold levels) with recursive setup
//! offloading (Stage-3 setup-product sum-check and a carried setup-prefix
//! opening), so a successful proof exercises the mix: chunked folds that also
//! run the offloaded setup-contribution path.
//!
//! The fixture is production-sized and must be run explicitly in an optimized
//! profile:
//!
//! `cargo test --release -p akita-pcs --test distributed_setup_offload_e2e --features profile-ci -- --ignored`

#![cfg(feature = "profile-ci")]
#![allow(missing_docs)]

mod common;

use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_types::{
    setup_matrix_capacity_for_schedule, verifier_setup_matrix_capacity_for_schedule,
    AkitaScheduleLookupKey, CommitmentRingDims, FoldSchedule, PolynomialGroupLayout,
    SetupContributionMode,
};
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"distributed_setup_offload_e2e/w8r2";

type W8R2Cfg = RecursiveCommitmentConfig<fp128::OneHotMultiChunk>;
fn w8r2_profiling_key() -> AkitaScheduleLookupKey {
    let pre_group = PolynomialGroupLayout::new(16, 1);
    let precommitted = akita_config::resolve_prior_group_profile::<W8R2Cfg>(&pre_group)
        .expect("precommit profile");
    AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        prior_group_profiles: vec![precommitted, precommitted],
    }
}

#[test]
fn w8r2_verifier_setup_stops_after_the_offloaded_chain() {
    let key = w8r2_profiling_key();
    let root_layout = key.opening_layout().expect("root layout");
    let schedule = W8R2Cfg::select_schedule_for_key(&key).expect("W8R2 schedule");
    assert_w8r2_profile_shape(&schedule);
    let prover = setup_matrix_capacity_for_schedule(&schedule).expect("prover capacity");
    let verifier = verifier_setup_matrix_capacity_for_schedule(&schedule, &root_layout)
        .expect("verifier capacity");
    let setup_for_two = W8R2Cfg::setup_matrix_capacity(32, 2).expect("setup capacity for K=2");
    let setup_for_four = W8R2Cfg::setup_matrix_capacity(32, 4).expect("setup capacity for K=4");
    let incoming_prefixes = schedule
        .recursive_folds
        .iter()
        .map(|fold| {
            fold.params
                .incoming_setup_prefix
                .as_ref()
                .map(|slot| slot.natural_len)
        })
        .collect::<Vec<_>>();
    eprintln!(
        "W8R2 setup capacities: provisioned_k2={}, provisioned_k4={}, exact_prover={}, exact_verifier={}, incoming_prefixes={:?}",
        setup_for_two.num_field_elements,
        setup_for_four.num_field_elements,
        prover.num_field_elements,
        verifier.num_field_elements,
        incoming_prefixes
    );
    assert_eq!(&incoming_prefixes[..2], &[Some(11_316_224), None]);
    assert!(incoming_prefixes[2..].iter().all(Option::is_none));
    assert_eq!(prover.num_field_elements, 11_316_224);
    assert_eq!(verifier.num_field_elements, 8_388_608);
    assert_eq!(setup_for_two.num_field_elements, 8_388_608);
    assert_eq!(setup_for_four.num_field_elements, 11_316_224);
}

/// Assert the exact shipped `W8R2` profile shape, not just "some mixed fold".
///
/// The generated table is exact for the `(32, 2) + two (16, 1)` profiling key, so
/// the test pins every distinguishing fact. This catches `W4R2` vs `W8R2`, a
/// level-0/level-1 mode swap, only one mixed leading fold, and a missing/extra
/// setup-prefix handoff — none of which a bare "any chunked recursive fold" check
/// would detect. `OneHot` (single-chunk) also fails this on levels 0/1.
fn assert_w8r2_profile_shape(schedule: &FoldSchedule) {
    assert!(
        schedule.recursive_folds.len() >= 2,
        "W8R2 profile must have at least three fold levels, got {}",
        1 + schedule.recursive_folds.len()
    );
    assert_eq!(
        schedule.root.params.final_group.commitment.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 64,
        },
        "level 0 must retain the shipped adaptive A/B/D role dimensions"
    );
    assert_eq!(
        schedule.recursive_folds[0].params.witness.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 64,
        },
        "level 1 must retain the shipped adaptive A/B/D role dimensions"
    );
    assert_eq!(
        schedule.recursive_folds[1].params.witness.role_dims(),
        CommitmentRingDims::uniform(64),
        "level 2 must retain the shipped uniform D64 suffix dimensions"
    );

    // Levels 0 and 1 both use the W8R2 witness partition: 8 chunks over the two
    // leading levels.
    for (level, params) in [
        &schedule.root.params.final_group.commitment,
        &schedule.recursive_folds[0].params.witness,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            params.witness_chunk.num_chunks, 8,
            "level {level} must be chunked with num_chunks == 8 (W8R2)"
        );
        assert_eq!(
            params.witness_chunk.num_activated_levels, 2,
            "level {level} must carry num_activated_levels == 2 (W8R2)"
        );
    }

    assert_eq!(
        schedule.recursive_folds[0]
            .params
            .predecessor_setup_contribution_mode(),
        SetupContributionMode::Recursive,
        "level 0 must run the planner-selected recursive setup-offload path"
    );

    // Level 0 produces the selected setup prefix and level 1 consumes it.
    assert!(
        schedule.recursive_folds[0]
            .params
            .incoming_setup_prefix
            .is_some(),
        "level 1 must consume the level-0 setup prefix"
    );

    // Level 2 is the single-chunk direct fold after the selected offload edge.
    let level2 = &schedule.recursive_folds[1].params;
    assert_eq!(
        level2.witness.witness_chunk.num_chunks, 1,
        "level 2 must be single-chunk (chunking activates only levels 0 and 1)"
    );
    let level2_mode = schedule
        .recursive_folds
        .get(2)
        .map_or(SetupContributionMode::Direct, |consumer| {
            consumer.params.predecessor_setup_contribution_mode()
        });
    assert_eq!(
        level2_mode,
        SetupContributionMode::Direct,
        "level 2 must be Direct (no Stage-3 sum-check after the activated window)"
    );
    assert!(
        level2.incoming_setup_prefix.is_none(),
        "level 2 must not carry an unselected setup prefix"
    );
}

#[test]
#[ignore = "production-sized profile E2E; run explicitly with --release"]
fn mix_multi_chunk_recursive_profile_proves_and_verifies() {
    recursive_multi_group_round_trip::<fp128::OneHotMultiChunk>(
        TRANSCRIPT_DOMAIN,
        assert_w8r2_profile_shape,
    );
}
