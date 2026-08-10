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

fn assert_schedule_geometry(schedule: &FoldSchedule, allowed_dims: &[usize]) {
    let params = std::iter::once(&schedule.root.params.final_group.commitment).chain(
        schedule
            .recursive_folds
            .iter()
            .map(|step| &step.params.witness),
    );
    for params in params {
        let dims = params.role_dims();
        assert!(allowed_dims.contains(&dims.d_a()));
        assert!(allowed_dims.contains(&dims.d_b()));
        assert!(allowed_dims.contains(&dims.d_d()));
        assert_eq!(
            params.flat_field_len().expect("flat length"),
            params.n_ring_elems().expect("ring elements") * params.d_a()
        );
    }
    assert!(allowed_dims.contains(&schedule.terminal.params.witness.d_a()));
}

#[test]
fn accepts_real_fp64_adaptive_schedule() {
    let schedule = schedule::<fp64::Dense>(20);
    validate_schedule_ring_dims(&schedule).expect("adaptive fp64 schedule");
    assert_schedule_geometry(&schedule, &[64, 128, 256]);
}

#[test]
fn accepts_real_fp128_adaptive_schedule() {
    let schedule = schedule::<fp128::Dense>(16);
    validate_schedule_ring_dims(&schedule).expect("adaptive schedule");
    assert!(
        schedule
            .recursive_folds
            .iter()
            .any(|fold| fold.params.witness.d_a()
                != schedule.root.params.final_group.commitment.d_a())
    );
}
