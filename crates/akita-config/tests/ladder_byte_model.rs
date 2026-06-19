//! Phase-0 ladder byte model: compare mixed-D schedules vs fixed-D presets.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::ladder_byte_model::{
    find_ladder_schedule, fold_ring_dimensions, FP128_ONEHOT_LADDER_DIMS,
};
use akita_planner::PlannerPolicy;
use akita_types::AkitaScheduleLookupKey;

/// Side-by-side byte totals: best fixed-D fp128 onehot preset vs mixed-D ladder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderVsFixedReport {
    pub num_vars: usize,
    pub fixed_bytes: usize,
    pub fixed_preset: &'static str,
    pub fixed_ring_dims: Vec<usize>,
    pub ladder_bytes: usize,
    pub ladder_ring_dims: Vec<usize>,
}

impl LadderVsFixedReport {
    pub fn byte_delta(&self) -> i64 {
        self.ladder_bytes as i64 - self.fixed_bytes as i64
    }
}

fn fp128_onehot_policy() -> PlannerPolicy {
    policy_of::<fp128::D64OneHot>()
}

fn ladder_stage1(d: usize) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
    fp128::D64OneHot::ring_challenge_config(d)
}

fn ladder_fold_shape(
    inputs: akita_types::AkitaScheduleInputs,
) -> akita_challenges::TensorChallengeShape {
    fp128::D64OneHot::fold_challenge_shape_at_level(inputs)
}

pub fn compare_fp128_onehot_ladder(
    key: AkitaScheduleLookupKey,
) -> Result<LadderVsFixedReport, akita_field::AkitaError> {
    let fixed = fp128::best_onehot_schedule(key)?
        .ok_or_else(|| akita_field::AkitaError::InvalidSetup("no fixed-D schedule".into()))?;
    let policy = fp128_onehot_policy();
    let ladder = find_ladder_schedule(
        key,
        &policy,
        FP128_ONEHOT_LADDER_DIMS,
        ladder_stage1,
        ladder_fold_shape,
    )?;
    Ok(LadderVsFixedReport {
        num_vars: key.num_vars,
        fixed_bytes: fixed.schedule.total_bytes,
        fixed_preset: fixed.preset.name(),
        fixed_ring_dims: fold_ring_dimensions(&fixed.schedule),
        ladder_bytes: ladder.total_bytes,
        ladder_ring_dims: fold_ring_dimensions(&ladder),
    })
}

#[test]
fn ladder_byte_model_fp128_onehot_vs_best_fixed_d() {
    let rows: Vec<LadderVsFixedReport> = [24, 26, 28, 30, 32]
        .into_iter()
        .map(|nv| {
            compare_fp128_onehot_ladder(AkitaScheduleLookupKey::singleton(nv))
                .unwrap_or_else(|e| panic!("nv={nv}: {e}"))
        })
        .collect();

    let mut summary = String::from("fp128 onehot ladder vs fixed-D:\n");
    for row in &rows {
        summary.push_str(&format!(
            "  nv={}: fixed {} ({} B, dims {:?}) | ladder ({} B, dims {:?}) | delta={:+}\n",
            row.num_vars,
            row.fixed_preset,
            row.fixed_bytes,
            row.fixed_ring_dims,
            row.ladder_bytes,
            row.ladder_ring_dims,
            row.byte_delta(),
        ));
    }
    eprintln!("{summary}");

    let nv32 = rows.iter().find(|r| r.num_vars == 32).expect("nv=32 row");
    assert!(nv32.ladder_bytes > 0);
    assert!(!nv32.ladder_ring_dims.is_empty());
}
