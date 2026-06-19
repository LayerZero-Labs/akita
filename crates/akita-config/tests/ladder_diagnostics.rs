//! Ladder DP soundness checks and fp128 dense experiments.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::find_schedule;
use akita_planner::ladder_byte_model::{
    find_ladder_schedule, fold_ring_dimensions, is_mixed_ring_dimension_ladder,
    schedule_fold_bytes, schedule_terminal_bytes,
};
use akita_types::AkitaScheduleLookupKey;

fn stage1_full(d: usize) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
    fp128::D64Full::ring_challenge_config(d)
}

fn fold_shape_full(
    inputs: akita_types::AkitaScheduleInputs,
) -> akita_challenges::TensorChallengeShape {
    fp128::D64Full::fold_challenge_shape_at_level(inputs)
}

#[test]
fn ladder_with_single_d_matches_find_schedule_dense() {
    let key = AkitaScheduleLookupKey::singleton(28);
    let policy = policy_of::<fp128::D64Full>();
    let fixed = find_schedule(key, &policy, stage1_full, fold_shape_full).expect("fixed");
    let ladder = find_ladder_schedule(key, &policy, &[64], stage1_full, fold_shape_full).expect("ladder");
    assert_eq!(
        ladder.total_bytes, fixed.total_bytes,
        "single-D ladder must reproduce find_schedule"
    );
    assert_eq!(fold_ring_dimensions(&ladder), fold_ring_dimensions(&fixed));
}

#[test]
fn ladder_is_superset_of_each_fixed_d_dense() {
    let policy = policy_of::<fp128::D64Full>();
    let allowed = [128usize, 64, 32];
    for nv in 18..=34 {
        let key = AkitaScheduleLookupKey::singleton(nv);
        let ladder = find_ladder_schedule(key, &policy, &allowed, stage1_full, fold_shape_full)
            .unwrap_or_else(|e| panic!("nv={nv}: {e}"));
        for &d in &allowed {
            let preset_policy = match d {
                32 => policy_of::<fp128::D32Full>(),
                64 => policy_of::<fp128::D64Full>(),
                128 => policy_of::<fp128::D128Full>(),
                _ => panic!("unsupported d"),
            };
            let fixed = find_schedule(key, &preset_policy, stage1_full, fold_shape_full)
                .unwrap_or_else(|e| panic!("nv={nv} d={d}: {e}"));
            assert!(
                ladder.total_bytes <= fixed.total_bytes,
                "nv={nv}: ladder {} must not exceed fixed D{d} {}",
                ladder.total_bytes,
                fixed.total_bytes
            );
        }
    }
}

#[test]
fn fp128_dense_fixed_d_ranking_at_nv28() {
    let key = AkitaScheduleLookupKey::singleton(28);
    let mut rows = Vec::new();
    for (d, policy) in [
        (32, policy_of::<fp128::D32Full>()),
        (64, policy_of::<fp128::D64Full>()),
        (128, policy_of::<fp128::D128Full>()),
    ] {
        let sched = find_schedule(key, &policy, stage1_full, fold_shape_full).expect("schedule");
        rows.push((d, sched.total_bytes, schedule_fold_bytes(&sched), schedule_terminal_bytes(&sched)));
    }
    rows.sort_by_key(|(_, total, _, _)| *total);
    eprintln!("fp128 dense nv=28 fixed-D ranking: {rows:?}");
    assert_eq!(rows[0].0, 64, "D64 should be the uniform-D winner at nv=28");
}

#[test]
fn fp128_dense_ladder_never_mixed_in_sweep() {
    let policy = policy_of::<fp128::D64Full>();
    let allowed = [128usize, 64, 32];
    let mut saw_mixed = false;
    for nv in 18..=34 {
        let key = AkitaScheduleLookupKey::singleton(nv);
        let ladder = find_ladder_schedule(key, &policy, &allowed, stage1_full, fold_shape_full)
            .expect("ladder");
        let dims = fold_ring_dimensions(&ladder);
        if is_mixed_ring_dimension_ladder(&dims) {
            saw_mixed = true;
            eprintln!("nv={nv}: mixed ladder dims {dims:?} bytes={}", ladder.total_bytes);
        }
    }
    assert!(
        !saw_mixed,
        "no mixed-D schedule won in nv=18..34 dense sweep (expected)"
    );
}

#[test]
fn fp128_onehot_ladder_matches_best_fixed_in_sweep() {
    let policy = policy_of::<fp128::D64OneHot>();
    let allowed = [128usize, 64, 32];
    for nv in [24, 28, 32] {
        let key = AkitaScheduleLookupKey::singleton(nv);
        let ladder = find_ladder_schedule(
            key,
            &policy,
            &allowed,
            |d| fp128::D64OneHot::ring_challenge_config(d),
            fp128::D64OneHot::fold_challenge_shape_at_level,
        )
        .expect("ladder");
        let best = fp128::best_onehot_schedule(key)
            .expect("best")
            .expect("some");
        assert_eq!(
            ladder.total_bytes,
            best.schedule.total_bytes,
            "nv={nv}: ladder should match best fixed onehot"
        );
    }
}
