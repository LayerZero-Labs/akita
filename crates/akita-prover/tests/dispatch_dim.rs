//! Runtime ring-dimension dispatch against real typed schedules.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::CommitmentConfig;
use akita_types::{
    validate_schedule_ring_dims, AkitaScheduleLookupKey, FoldSchedule, PolynomialGroupLayout,
};

fn schedule<Cfg: CommitmentConfig>(num_vars: usize) -> FoldSchedule {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule catalog");
    let group = match akita_config::honest_fold_policy_of::<Cfg>() {
        akita_types::sis::HonestFoldPolicySpec::BalancedSignedDigit(_) => {
            PolynomialGroupLayout::singleton(num_vars)
        }
        akita_types::sis::HonestFoldPolicySpec::UnitOneHot(_) => {
            PolynomialGroupLayout::new(num_vars, 1)
        }
    };
    catalog
        .resolve_key(&AkitaScheduleLookupKey::single(group))
        .expect("runtime schedule")
        .schedule()
        .clone()
}

fn assert_schedule_geometry(schedule: &FoldSchedule, allowed_dims: &[usize]) {
    let params = std::iter::once(&schedule.root.params)
        .chain(schedule.recursive_folds.iter().map(|step| &step.params));
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
    assert!(allowed_dims.contains(&schedule.terminal.d_a()));
}

#[test]
fn accepts_real_fp64_adaptive_schedule() {
    let schedule = schedule::<fp64::Dense>(20);
    validate_schedule_ring_dims(&schedule).expect("adaptive fp64 schedule");
    assert_schedule_geometry(&schedule, &[64, 128, 256, 512, 1024, 2048]);
    assert_eq!(schedule.root.params.d_a(), 1024);
}

#[test]
fn accepts_real_fp32_adaptive_schedule() {
    let schedule = schedule::<fp32::Dense>(20);
    validate_schedule_ring_dims(&schedule).expect("adaptive fp32 schedule");
    assert_schedule_geometry(&schedule, &[64, 128, 256, 512, 1024, 2048]);
    assert_eq!(schedule.root.params.d_a(), 2048);
}

#[test]
fn accepts_real_fp128_adaptive_schedule() {
    let schedule = schedule::<fp128::Dense>(16);
    validate_schedule_ring_dims(&schedule).expect("adaptive schedule");
    assert_schedule_geometry(&schedule, &[64, 128, 256, 512]);
    assert_eq!(schedule.root.params.d_a(), 512);
}
