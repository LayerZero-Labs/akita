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
use akita_prover::{NttExecutionRequirements, NttOperationCluster};
use akita_types::{
    centered_quotient_requires_i16_tail, setup_matrix_capacity_for_schedule,
    verifier_setup_matrix_capacity_for_schedule, AkitaScheduleLookupKey, FoldSchedule,
    InnerCommitMatrixParams, NttCacheKey, NttTransformDomain, OpeningMethod, PolynomialGroupLayout,
    SetupContributionMode, SubringCoefficientPackingGeometry,
};
use common::*;

const TRANSCRIPT_DOMAIN: &[u8] = b"distributed_setup_offload_e2e/w8r2";

type W8R2Cfg = RecursiveCommitmentConfig<fp128::OneHotMultiChunk>;
fn w8r2_profiling_key(
    base_catalog: &akita_config::TrustedScheduleCatalog,
) -> AkitaScheduleLookupKey {
    let pre_group = PolynomialGroupLayout::new(16, 1);
    let precommitted = base_catalog
        .resolve_key(&AkitaScheduleLookupKey::single(pre_group))
        .expect("independent row")
        .profiles()
        .final_group;
    AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![precommitted, precommitted],
    }
}

#[test]
fn w8r2_verifier_setup_stops_after_the_offloaded_chain() {
    let base_catalog =
        akita_config::test_support::workspace_schedule_catalog::<fp128::OneHotMultiChunk>()
            .expect("base W8R2 catalog");
    let catalog = akita_config::test_support::workspace_schedule_catalog::<W8R2Cfg>()
        .expect("recursive W8R2 catalog");
    let key = w8r2_profiling_key(&base_catalog);
    let root_layout = key.opening_layout().expect("root layout");
    let schedule = catalog.resolve_key(&key).expect("W8R2 schedule");
    assert_w8r2_profile_shape(schedule.schedule());
    let prover = setup_matrix_capacity_for_schedule(schedule.schedule()).expect("prover capacity");
    let verifier = verifier_setup_matrix_capacity_for_schedule(schedule.schedule(), &root_layout)
        .expect("verifier capacity");
    let setup_for_two = akita_config::SetupRequirements::from_catalog::<W8R2Cfg>(&catalog, 32, 2)
        .map(|requirements| requirements.matrix_capacity)
        .expect("setup capacity for K=2");
    let setup_for_four = akita_config::SetupRequirements::from_catalog::<W8R2Cfg>(&catalog, 32, 4)
        .map(|requirements| requirements.matrix_capacity)
        .expect("setup capacity for K=4");
    let incoming_prefixes = schedule
        .schedule()
        .recursive_folds
        .iter()
        .map(|fold| {
            fold.params
                .setup_prefix()
                .as_ref()
                .and_then(|slot| slot.setup_natural_len)
        })
        .collect::<Vec<_>>();
    assert_eq!(setup_for_two.num_field_elements, 88_064);
    assert_eq!(setup_for_four.num_field_elements, 8_388_608);
    assert_eq!(prover.num_field_elements, 8_388_608);
    assert_eq!(verifier.num_field_elements, 4_194_304);
    assert_eq!(
        incoming_prefixes,
        [Some(8_388_608), None, None, None, None, None]
    );
    // Exactly one fold carries a setup prefix, and it carries the length the
    // committed catalog states. These values deliberately pin the shipped row:
    // a planner change must update the fixture and explain the new setup shape.
    assert_eq!(
        incoming_prefixes.len(),
        schedule.schedule().recursive_folds.len()
    );
    assert!(incoming_prefixes[0].is_some());
    assert!(incoming_prefixes[1..].iter().all(Option::is_none));
    assert!(
        verifier.num_field_elements <= prover.num_field_elements,
        "verifier setup must remain a prefix of the prover setup"
    );

    // `K=2` cannot reach the four-polynomial grouped root, and this family
    // ships no row without precommitted groups. The only shape it can still serve is
    // an independent commitment of the frozen precommit descriptor, performed
    // under the base config. Provisioning must cover exactly that, so derive
    // the expectation from the same primitive commit-time admission uses.
    let frozen_precommit = key.precommitteds[0];
    let precommit_footprint = akita_types::commit_only_setup_field_elements(
        &frozen_precommit.inner.matrix,
        &frozen_precommit.outer.matrix,
        frozen_precommit.outer_slice_count,
    )
    .expect("frozen precommit footprint");
    assert_eq!(setup_for_two.num_field_elements, precommit_footprint);
    assert_eq!(setup_for_four.num_field_elements, prover.num_field_elements);
}

#[test]
fn w8r2_ntt_requirements_match_distributed_a_tail_decisions() {
    let base_catalog =
        akita_config::test_support::workspace_schedule_catalog::<fp128::OneHotMultiChunk>()
            .expect("base W8R2 catalog");
    let catalog = akita_config::test_support::workspace_schedule_catalog::<W8R2Cfg>()
        .expect("recursive W8R2 catalog");
    let key = w8r2_profiling_key(&base_catalog);
    let schedule = catalog
        .resolve_key(&key)
        .expect("W8R2 schedule")
        .schedule()
        .clone();
    let first_recursive = &schedule.recursive_folds[0].params;
    assert_eq!(first_recursive.witness_chunk.num_chunks, 8);
    let prefix = first_recursive
        .setup_prefix()
        .expect("W8R2 first recursive fold must consume a setup prefix");
    let witness_a = &first_recursive.inner().matrix;
    let prefix_a = &prefix.profile.inner.matrix;
    assert_eq!(
        (
            witness_a.ring_dimension(),
            witness_a.output_rank(),
            witness_a.input_width(),
        ),
        (128, 3, 2_048),
    );
    assert_eq!(
        (
            prefix_a.ring_dimension(),
            prefix_a.output_rank(),
            prefix_a.input_width(),
        ),
        (128, 4, 2_048),
    );

    let witness_tail =
        NttCacheKey::from_matrix_shape(128, 3, 2_048, NttTransformDomain::I16TailBothTransforms)
            .expect("valid W8R2 witness tail key");
    let prefix_tail =
        NttCacheKey::from_matrix_shape(128, 4, 2_048, NttTransformDomain::I16TailBothTransforms)
            .expect("valid W8R2 prefix tail key");
    let requirements =
        NttExecutionRequirements::from_prove_schedule(&schedule).expect("NTT requirements");
    let has_tail = |expected| {
        requirements.entries().iter().any(|entry| {
            entry.fold_level == 1
                && entry.cluster == NttOperationCluster::RingSwitch
                && entry.key == expected
        })
    };
    let expects_tail =
        |matrix: &InnerCommitMatrixParams, log_basis: u32, num_digits_fold: usize| {
            let (negative, positive) =
                akita_types::sis::balanced_digit_representable_bounds(log_basis, num_digits_fold);
            let rhs_abs_bound = u64::try_from(
                negative
                    .max(positive)
                    .checked_mul(first_recursive.witness_chunk.num_chunks as u128)
                    .expect("W8 aggregated fold bound"),
            )
            .expect("W8 aggregated fold bound fits u64");
            centered_quotient_requires_i16_tail(
                matrix.sis_modulus_profile(),
                matrix.ring_dimension(),
                rhs_abs_bound,
            )
            .expect("W8 tail decision")
        };
    assert_eq!(
        has_tail(witness_tail),
        expects_tail(
            witness_a,
            first_recursive.open().digits.log_basis,
            first_recursive.num_digits_fold(),
        ),
        "recursive-witness prewarm must match its aggregated W8 bound"
    );
    assert_eq!(
        has_tail(prefix_tail),
        expects_tail(
            prefix_a,
            prefix.opening.log_basis_open,
            prefix.opening.num_digits_fold,
        ),
        "incoming-prefix prewarm must inherit the consuming W8 chunk count"
    );
}

/// Assert the exact shipped `W8R2` profile shape, not just "some mixed fold".
///
/// The shipped artifact is exact for the `(32, 2) + two (16, 1)` profiling key, so
/// the test pins every distinguishing fact. This catches `W4R2` vs `W8R2`, a
/// level-0/level-1 mode swap, only one mixed leading fold, and a missing/extra
/// setup-prefix handoff — none of which a bare "any chunked recursive fold" check
/// would detect. `OneHot` (single-chunk) also fails this on levels 0/1.
fn assert_w8r2_profile_shape(schedule: &FoldSchedule) {
    assert!(
        schedule.recursive_folds.len() >= 2,
        "W8R2 profile must have at least three fold levels"
    );
    for (level, (params, expected_d_a, expected_packing_factor)) in [
        (&schedule.root.params, 256, 4),
        (&schedule.recursive_folds[0].params, 128, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } = params.opening_method()
        else {
            panic!("level {level} must use coefficient packing");
        };
        let geometry = SubringCoefficientPackingGeometry::try_new(
            W8R2Cfg::EXT_DEGREE,
            params.d_a(),
            challenge_subring_dimension,
        )
        .expect("valid coefficient-packing geometry");
        assert_eq!(
            params.d_a(),
            expected_d_a,
            "level {level} must preserve its exact A-ring dimension"
        );
        assert_eq!(
            challenge_subring_dimension, 64,
            "level {level} must use the 64-coefficient challenge subring"
        );
        assert_eq!(geometry.packing_factor(), expected_packing_factor);
        assert!(
            geometry.packing_factor() > 1,
            "level {level} must use reduced-width coefficient packing"
        );
    }
    assert_eq!(
        schedule.root.params.precommitted_groups().len(),
        2,
        "W8R2 must carry the two frozen singleton groups"
    );
    for (group_index, group) in schedule
        .root
        .params
        .precommitted_groups()
        .iter()
        .enumerate()
    {
        assert_eq!(group.profile.inner.matrix.ring_dimension(), 512);
        let expected_subring_dimension = 256;
        assert_eq!(
            group.opening.opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: expected_subring_dimension,
            },
            "precommitted group {group_index} must preserve its exact packing domain"
        );
        let geometry = SubringCoefficientPackingGeometry::try_new(
            W8R2Cfg::EXT_DEGREE,
            group.profile.inner.matrix.ring_dimension(),
            expected_subring_dimension,
        )
        .expect("valid precommitted packing geometry");
        assert_eq!(geometry.packing_factor(), 2);
    }
    assert_eq!(
        schedule.recursive_folds[1].params.opening_method(),
        OpeningMethod::EvaluationTrace,
        "the level-2 fold must consume the packing-produced flat witness through EvaluationTrace"
    );

    // Levels 0 and 1 both use the W8R2 witness partition: 8 chunks over the two
    // leading levels.
    for (level, params) in [&schedule.root.params, &schedule.recursive_folds[0].params]
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
        schedule.recursive_folds[0].predecessor_setup_contribution_mode(),
        SetupContributionMode::Recursive,
        "level 0 must run the planner-selected recursive setup-offload path"
    );

    // Level 0 produces the selected setup prefix and level 1 consumes it.
    assert!(
        schedule.recursive_folds[0].params.setup_prefix().is_some(),
        "level 1 must consume the level-0 setup prefix"
    );
    assert!(matches!(
        schedule.recursive_folds[0]
            .params
            .setup_prefix()
            .as_ref()
            .expect("level-1 setup prefix")
            .opening
            .opening_method,
        OpeningMethod::SubringCoefficientPacking { .. }
    ));

    // Level 2 is the single-chunk direct fold after the selected offload edge.
    let level2 = &schedule.recursive_folds[1].params;
    assert_eq!(
        level2.witness_chunk.num_chunks, 1,
        "level 2 must be single-chunk (chunking activates only levels 0 and 1)"
    );
    let level2_mode = schedule
        .recursive_folds
        .get(2)
        .map_or(SetupContributionMode::Direct, |consumer| {
            consumer.predecessor_setup_contribution_mode()
        });
    assert_eq!(
        level2_mode,
        SetupContributionMode::Direct,
        "level 2 must be Direct (no Stage-3 sum-check after the activated window)"
    );
    assert!(
        level2.setup_prefix().is_none(),
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
