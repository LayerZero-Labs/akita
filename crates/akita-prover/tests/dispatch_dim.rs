//! Runtime ring-dimension dispatch against real typed schedules.

#![cfg(feature = "schedules-default")]
#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::CommitmentConfig;
use akita_types::{
    validate_schedule_ring_dims, AkitaScheduleLookupKey, FoldSchedule, PolynomialGroupLayout,
};

fn schedule<Cfg: CommitmentConfig>(num_vars: usize) -> FoldSchedule {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(num_vars),
    ))
    .expect("runtime schedule")
}

fn assert_schedule_geometry(schedule: &FoldSchedule, expected_d: usize) {
    let params = std::iter::once(&schedule.root.params.final_group.commitment).chain(
        schedule
            .recursive_folds
            .iter()
            .map(|step| &step.params.witness),
    );
    for params in params {
        assert_eq!(params.d_a(), expected_d);
        assert_eq!(
            params.flat_field_len().expect("flat length"),
            params.n_ring_elems().expect("ring elements") * expected_d
        );
    }
    assert_eq!(schedule.terminal.params.witness.d_a(), expected_d);
}

#[test]
fn accepts_real_fp64_d128_schedule() {
    let schedule = schedule::<fp64::D128Dense>(20);
    validate_schedule_ring_dims(&schedule).expect("D128 schedule");
    assert_schedule_geometry(&schedule, 128);
}

#[test]
fn accepts_real_fp128_d64_schedule() {
    let schedule = schedule::<fp128::D64Dense>(16);
    validate_schedule_ring_dims(&schedule).expect("D64 schedule");
    assert_schedule_geometry(&schedule, 64);
}
