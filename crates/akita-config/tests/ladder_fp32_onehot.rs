//! fp32 onehot ladder experiment.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp32;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::find_schedule;
use akita_planner::ladder_byte_model::{
    find_ladder_schedule, fold_ring_dimensions, is_mixed_ring_dimension_ladder,
};
use akita_types::AkitaScheduleLookupKey;

fn stage1(d: usize) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
    fp32::D128OneHot::ring_challenge_config(d)
}

fn fold_shape(
    inputs: akita_types::AkitaScheduleInputs,
) -> akita_challenges::TensorChallengeShape {
    fp32::D128OneHot::fold_challenge_shape_at_level(inputs)
}

#[test]
fn fp32_onehot_ladder_single_d128_matches_find_schedule_at_nv28() {
    let key = AkitaScheduleLookupKey::singleton(28);
    let policy = policy_of::<fp32::D128OneHot>();
    let fixed = find_schedule(key, &policy, stage1, fold_shape).expect("fixed");
    let ladder =
        find_ladder_schedule(key, &policy, &[128], stage1, fold_shape).expect("ladder");
    assert_eq!(
        ladder.total_bytes, fixed.total_bytes,
        "nv=28: ladder D128-only must match find_schedule"
    );
}

#[test]
fn fp32_onehot_ladder_single_d128_matches_find_schedule_at_nv30() {
    let key = AkitaScheduleLookupKey::singleton(30);
    let policy = policy_of::<fp32::D128OneHot>();
    let fixed = find_schedule(key, &policy, stage1, fold_shape).expect("fixed");
    let ladder =
        find_ladder_schedule(key, &policy, &[128], stage1, fold_shape).expect("ladder");
    assert_eq!(
        ladder.total_bytes,
        fixed.total_bytes,
        "nv=30: ladder D128-only must match find_schedule (fixed={}, ladder={})",
        fixed.total_bytes,
        ladder.total_bytes
    );
}

#[test]
fn fp32_onehot_mixed_d_sweep_reports_wins() {
    let policy = policy_of::<fp32::D128OneHot>();
    let allowed = [256usize, 128, 64];
    let mut wins = Vec::new();
    let mut mixed = Vec::new();
    for nv in 16..=30 {
        let key = AkitaScheduleLookupKey::singleton(nv);
        let ladder = find_ladder_schedule(key, &policy, &allowed, stage1, fold_shape)
            .expect("ladder");
        let dims = fold_ring_dimensions(&ladder);
        if is_mixed_ring_dimension_ladder(&dims) {
            mixed.push((nv, ladder.total_bytes, dims.clone()));
        }
        let mut min_fixed = usize::MAX;
        for &d in &allowed {
            let preset = match d {
                64 => policy_of::<fp32::D64OneHot>(),
                128 => policy_of::<fp32::D128OneHot>(),
                256 => policy_of::<fp32::D256OneHot>(),
                _ => continue,
            };
            if let Ok(fixed) = find_schedule(key, &preset, stage1, fold_shape) {
                min_fixed = min_fixed.min(fixed.total_bytes);
            }
        }
        if ladder.total_bytes < min_fixed {
            wins.push((nv, min_fixed, ladder.total_bytes, dims));
        }
    }
    eprintln!("fp32 mixed-D ladders: {mixed:?}");
    eprintln!("fp32 ladder wins vs min fixed: {wins:?}");
    assert!(
        wins.is_empty(),
        "expected no ladder wins in nv=16..30; got {wins:?}"
    );
}
