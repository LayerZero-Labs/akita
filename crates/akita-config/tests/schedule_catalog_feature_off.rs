//! Feature-off guard: without schedule features, presets use DP-only resolution.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::find_schedule;
use akita_types::AkitaScheduleLookupKey;

#[test]
fn schedule_catalog_none_without_feature_uses_dp() {
    if cfg!(feature = "schedules-fp128-d64-onehot") {
        return;
    }

    assert!(
        fp128::D64OneHot::schedule_catalog().is_none(),
        "schedule feature disabled: schedule_catalog must be None"
    );

    let key = AkitaScheduleLookupKey::new_from_opening_batch(
        &akita_types::OpeningBatch::new(28, 1).expect("opening batch"),
    )
    .expect("lookup key");

    let dp = find_schedule(
        key,
        &policy_of::<fp128::D64OneHot>(),
        fp128::D64OneHot::ring_challenge_config,
        fp128::D64OneHot::fold_challenge_shape_at_level,
    )
    .expect("dp schedule");

    let runtime = fp128::D64OneHot::runtime_schedule(key).expect("runtime schedule");
    assert_eq!(runtime.total_bytes, dp.total_bytes);
    assert_eq!(runtime.steps.len(), dp.steps.len());
}
