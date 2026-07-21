//! Planner guard: shipped `fp128::D64OneHot` schedules must stay within the
//! configured proof-optimized basis search window.

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

/// Sparse singleton keys covering small, production, stress, and table-max nv.
const BASIS_ENVELOPE_NUM_VARS: &[usize] = &[10, 16, 28, 30, 64, 120];

#[test]
fn d64_onehot_schedule_stays_within_basis_envelope() {
    type Cfg = fp128::D64OneHot;

    for &nv in BASIS_ENVELOPE_NUM_VARS {
        let schedule = match Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        )) {
            Ok(schedule) => schedule,
            Err(_) => continue,
        };
        let within_window = schedule.folds.iter().all(|fold| {
            fold.params.log_basis_inner <= 6
                && fold.params.log_basis_outer <= 6
                && fold.params.log_basis_open <= 6
        }) && schedule.terminal.witness_shape.layout.log_basis_open <= 6;
        assert!(
            within_window,
            "adaptive onehot schedule selected log_basis > 6 at nv={nv}: {schedule:?}"
        );
    }
}
