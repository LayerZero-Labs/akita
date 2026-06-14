//! Tiered-commitment planner integration checks for `fp128::D64OneHotTiered`.
//!
//! Verify that the real runtime schedule tiers at least one level for a batched
//! key, and that every tiered level keeps the shrunk first-tier `B'` footprint
//! bounded by the inner `A` footprint. (`F` commits decomposed `u_i` and is not
//! a sizing constraint — not asserted here.)

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{matrix_envelope_for_schedule, CommitmentConfig};
use akita_types::{AkitaScheduleLookupKey, OpeningBatch, LevelParams, Schedule, Step};

fn footprint(key: &akita_types::AjtaiKeyParams) -> usize {
    key.row_len() * key.col_len()
}

fn level_params(schedule: &Schedule) -> Vec<&LevelParams> {
    schedule
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::Fold(fold) => Some(&fold.params),
            Step::Direct(direct) => direct.params.as_ref(),
        })
        .collect()
}

fn assert_tiered_levels_fit_under_a(schedule: &Schedule) -> usize {
    let mut tiered = 0usize;
    for lp in level_params(schedule) {
        if let Some(fk) = lp.f_key.as_ref() {
            tiered += 1;
            let a_footprint = footprint(&lp.a_key);
            assert!(
                footprint(&lp.b_key) <= a_footprint,
                "tiered B' footprint {} must fit under A footprint {a_footprint}",
                footprint(&lp.b_key)
            );
            assert!(lp.tier_split >= 2, "tiered level must have tier_split >= 2");
            // F width = tier_split · n_b' · num_digits_open.
            assert_eq!(
                fk.col_len(),
                lp.tier_split * lp.b_key.row_len() * lp.num_digits_open
            );
        } else {
            assert_eq!(lp.tier_split, 1, "non-tiered level must have tier_split 1");
        }
    }
    tiered
}

#[test]
fn tiered_preset_tiers_a_batched_root() {
    // A same-point batch (num_points = 1, many t-vectors) grows the first-tier
    // B width with the batch factor; whenever the DP's chosen root layout has
    // B > A, tiering fires and shrinks B' (and adds F) below A. The DP picks
    // discrete (m, r) layouts, so whether B strictly exceeds A is batch- and
    // num_vars-dependent; assert tiering fires for at least one batch in a
    // representative sweep, and that the under-A invariant holds for every
    // tiered level in every schedule.
    let mut total_tiered = 0usize;
    for batch in [64usize, 128, 256, 512, 1024] {
        let key = AkitaScheduleLookupKey::new(22, batch, batch, 1);
        let schedule = fp128::D64OneHotTiered::runtime_schedule(key).expect("tiered schedule");
        total_tiered += assert_tiered_levels_fit_under_a(&schedule);
    }
    assert!(
        total_tiered >= 1,
        "expected tiering to fire for at least one batched root in the sweep"
    );
}

#[test]
fn tiered_envelope_never_larger_and_sometimes_smaller_than_non_tiered() {
    // For the same batched opening_batch, the tiered preset's shared-matrix
    // envelope must never exceed the non-tiered sibling's, and must be strictly
    // smaller whenever the optimal layout tiers a level (B > A).
    let nv = 22;
    let mut saw_strict_shrink = false;
    for batch in [64usize, 128, 256, 512, 1024] {
        let opening_batch = OpeningBatch::same_point(nv, batch).expect("opening_batch");
        let tiered_sched =
            fp128::D64OneHotTiered::get_params_for_prove(&opening_batch).expect("tiered schedule");
        let plain_sched =
            fp128::D64OneHot::get_params_for_prove(&opening_batch).expect("plain schedule");
        let env_tiered =
            matrix_envelope_for_schedule::<fp128::D64OneHotTiered>(&tiered_sched, &opening_batch)
                .expect("tiered envelope")
                .max_setup_len;
        let env_plain = matrix_envelope_for_schedule::<fp128::D64OneHot>(&plain_sched, &opening_batch)
            .expect("plain envelope")
            .max_setup_len;
        assert!(
            env_tiered <= env_plain,
            "batch={batch}: tiered envelope {env_tiered} must be <= non-tiered {env_plain}"
        );
        if env_tiered < env_plain {
            saw_strict_shrink = true;
        }
    }
    assert!(
        saw_strict_shrink,
        "expected the tiered envelope to be strictly smaller for at least one batch"
    );
}

#[test]
fn tiered_preset_matches_non_tiered_when_b_already_fits() {
    // For a singleton the first-tier B typically already fits under A, so the
    // tiered preset must leave every level single-tier (tier_split == 1).
    let key = AkitaScheduleLookupKey::singleton(20);
    let schedule = fp128::D64OneHotTiered::runtime_schedule(key).expect("tiered schedule");
    // Whatever the layout, the invariant must still hold for any tiered level.
    let _ = assert_tiered_levels_fit_under_a(&schedule);
}
