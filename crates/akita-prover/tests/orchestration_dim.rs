//! Schedule-authority and role-dispatch orchestration gates.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::test_support::ring_plan_test_seed;
use akita_config::{effective_batched_schedule, CommitmentConfig};
use akita_types::{
    validate_role_dispatch, validate_schedule_ring_dims, AkitaScheduleLookupKey,
    OpeningClaimsLayout, PolynomialGroupLayout, RingRole,
};

#[test]
fn batched_selection_preserves_typed_schedule_topology() {
    type Cfg = fp64::D64Full;
    let nv = 14;
    let expected = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(nv),
    ))
    .expect("runtime schedule");
    let batch = OpeningClaimsLayout::new(nv, 1).expect("opening batch");
    let point = vec![<Cfg as CommitmentConfig>::ExtField::zero(); nv];
    let actual = effective_batched_schedule::<Cfg>(&batch, &point).expect("effective schedule");
    assert_eq!(actual.recursive_folds.len(), expected.recursive_folds.len());
    assert_eq!(
        actual.terminal.input_witness_len,
        expected.terminal.input_witness_len
    );
}

#[test]
fn role_dispatch_rejects_wrong_inner_dimension() {
    let schedule = fp128::D128Full::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(16),
    ))
    .expect("runtime schedule");
    let dims = schedule.root.params.final_group.commitment.role_dims();
    assert!(validate_role_dispatch::<64>(dims, RingRole::Inner).is_err());
}

#[test]
fn real_presets_validate_against_setup_ring_dimension() {
    for schedule in [
        fp64::D64Full::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(14),
        ))
        .expect("fp64 schedule"),
        fp128::D64Full::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(13),
        ))
        .expect("fp128 schedule"),
    ] {
        validate_schedule_ring_dims(&schedule, &ring_plan_test_seed(64))
            .expect("D64 schedule envelope");
    }
}
