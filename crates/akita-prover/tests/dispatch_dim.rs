//! Runtime ring-dimension dispatch against real typed schedules.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::CommitmentConfig;
use akita_types::{
    validate_schedule_ring_dims, AkitaScheduleLookupKey, AkitaSetupSeed, FoldSchedule,
    PolynomialGroupLayout,
};

fn schedule<Cfg: CommitmentConfig>(num_vars: usize) -> FoldSchedule {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(num_vars),
    ))
    .expect("runtime schedule")
}

fn seed(gen_ring_dim: usize) -> AkitaSetupSeed {
    AkitaSetupSeed {
        max_num_vars: 20,
        max_num_batched_polys: 1,
        gen_ring_dim,
        max_setup_len: 1 << 20,
        public_matrix_seed: [0; 32],
    }
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
fn accepts_real_fp64_d64_schedule() {
    let schedule = schedule::<fp64::D64Dense>(20);
    validate_schedule_ring_dims(&schedule, &seed(64)).expect("D64 schedule");
    assert_schedule_geometry(&schedule, 64);
}

#[test]
fn accepts_real_fp128_d128_schedule() {
    let schedule = schedule::<fp128::D128Dense>(18);
    validate_schedule_ring_dims(&schedule, &seed(128)).expect("D128 schedule");
    assert_schedule_geometry(&schedule, 128);
}

#[test]
fn rejects_schedule_larger_than_setup_ring_dimension() {
    let schedule = schedule::<fp128::D128Dense>(16);
    assert!(validate_schedule_ring_dims(&schedule, &seed(64)).is_err());
}
