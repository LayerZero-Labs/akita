//! Schedule-authority and role-dispatch orchestration gates.

#![cfg(feature = "schedules-default")]
#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_config::{effective_batched_schedule, CommitmentConfig};
use akita_types::{
    validate_role_dispatch, validate_schedule_ring_dims, AkitaScheduleLookupKey,
    CommittedGroupBatchProfile, CommittedGroupProfile, OpeningClaimsLayout, PolynomialGroupLayout,
    RingRole,
};

#[test]
fn batched_selection_preserves_typed_schedule_topology() {
    type Cfg = fp64::Dense;
    let nv = 14;
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(nv));
    let expected = Cfg::runtime_schedule(key.clone()).expect("runtime schedule");
    let batch = OpeningClaimsLayout::new(nv, 1).expect("opening batch");
    let final_group_point = vec![<Cfg as CommitmentConfig>::ExtField::zero(); nv];
    let profiles = CommittedGroupBatchProfile {
        final_group: CommittedGroupProfile::from_params(
            key.final_group,
            &expected.root.params.final_group.commitment,
        ),
        precommitteds: Vec::new(),
    };
    let selected = Cfg::select_schedule_for_profiles(&profiles).expect("selected schedule");
    let actual = effective_batched_schedule::<Cfg>(selected, &batch, &final_group_point)
        .expect("effective schedule");
    assert_eq!(
        actual.schedule().recursive_folds.len(),
        expected.recursive_folds.len()
    );
    assert_eq!(
        actual.schedule().terminal.input_witness_len,
        expected.terminal.input_witness_len
    );
}

#[test]
fn role_dispatch_rejects_wrong_inner_dimension() {
    let schedule = fp128::Dense::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(16),
    ))
    .expect("runtime schedule");
    let dims = schedule.root.params.final_group.commitment.role_dims();
    assert!(validate_role_dispatch::<128>(dims, RingRole::Inner).is_err());
}

#[test]
fn real_presets_validate_against_setup_ring_dimension() {
    let fp64_schedule = fp64::Dense::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(14),
    ))
    .expect("fp64 schedule");
    validate_schedule_ring_dims(&fp64_schedule).expect("adaptive fp64 schedule envelope");

    let fp128_schedule = fp128::Dense::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(14),
    ))
    .expect("fp128 schedule");
    validate_schedule_ring_dims(&fp128_schedule).expect("adaptive schedule envelope");
}
