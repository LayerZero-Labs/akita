//! Debug nv=30 fp32 ladder vs find_schedule divergence.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp32;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::find_schedule;
use akita_planner::ladder_byte_model::find_ladder_schedule;
use akita_types::{AkitaScheduleLookupKey, Step};

#[test]
fn debug_fp32_nv30_schedule_diff() {
    let key = AkitaScheduleLookupKey::singleton(30);
    let policy = policy_of::<fp32::D128OneHot>();
    let fixed = find_schedule(
        key,
        &policy,
        |d| fp32::D128OneHot::ring_challenge_config(d),
        fp32::D128OneHot::fold_challenge_shape_at_level,
    )
    .expect("fixed");
    let ladder = find_ladder_schedule(
        key,
        &policy,
        &[128],
        |d| fp32::D128OneHot::ring_challenge_config(d),
        fp32::D128OneHot::fold_challenge_shape_at_level,
    )
    .expect("ladder");

    eprintln!("fixed total={} steps={}", fixed.total_bytes, fixed.steps.len());
    eprintln!("ladder total={} steps={}", ladder.total_bytes, ladder.steps.len());

    for (i, (a, b)) in fixed.steps.iter().zip(ladder.steps.iter()).enumerate() {
        match (a, b) {
            (Step::Fold(fa), Step::Fold(fb)) => {
                eprintln!(
                    "fold {i}: fixed lb={} bytes={} next_w={} | ladder lb={} bytes={} next_w={}",
                    fa.params.log_basis,
                    fa.level_bytes,
                    fa.next_w_len,
                    fb.params.log_basis,
                    fb.level_bytes,
                    fb.next_w_len
                );
            }
            (Step::Direct(da), Step::Direct(db)) => {
                eprintln!(
                    "direct {i}: fixed bytes={} | ladder bytes={}",
                    da.direct_bytes, db.direct_bytes
                );
            }
            _ => eprintln!("step {i}: kind mismatch {a:?} vs {b:?}"),
        }
    }
    assert_eq!(fixed.total_bytes, ladder.total_bytes);
}
