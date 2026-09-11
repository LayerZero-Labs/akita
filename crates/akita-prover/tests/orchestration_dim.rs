//! Schedule-authority and role-dispatch orchestration gates.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp64};
use akita_types::{
    validate_role_dispatch, validate_schedule_ring_dims, AkitaScheduleLookupKey,
    CommittedGroupBatchProfile, GroupCommitPhaseParams, OpeningClaimsLayout, PolynomialGroupLayout,
    RingRole,
};

#[test]
fn batched_selection_preserves_typed_schedule_topology() {
    type Cfg = fp64::Dense;
    let catalog = akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule catalog");
    let nv = 14;
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(nv));
    let expected = catalog.resolve_key(&key).expect("runtime schedule");
    let batch = OpeningClaimsLayout::new(nv, 1).expect("opening batch");
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &expected.schedule().root.params,
        )
        .expect("valid profile"),
        precommitteds: Vec::new(),
    };
    let selected = catalog
        .resolve_profiles(&profiles)
        .expect("selected schedule");
    selected
        .validate_opening_layout(&batch)
        .expect("matching layout");
    assert!(selected
        .validate_opening_layout(&OpeningClaimsLayout::new(nv + 1, 1).unwrap())
        .is_err());
    let actual = selected;
    assert_eq!(
        actual.schedule().recursive_folds.len(),
        expected.schedule().recursive_folds.len()
    );
    assert_eq!(
        actual.schedule().terminal.input_witness_len,
        expected.schedule().terminal.input_witness_len
    );
}

#[test]
fn role_dispatch_rejects_wrong_inner_dimension() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<fp128::Dense>()
        .expect("workspace schedule catalog");
    let schedule = catalog
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(16),
        ))
        .expect("runtime schedule");
    let dims = schedule.schedule().root.params.role_dims();
    assert!(validate_role_dispatch::<128>(dims, RingRole::Inner).is_err());
}

#[test]
fn real_presets_validate_against_setup_ring_dimension() {
    let fp64_catalog = akita_config::test_support::workspace_schedule_catalog::<fp64::Dense>()
        .expect("fp64 workspace schedule catalog");
    let fp128_catalog = akita_config::test_support::workspace_schedule_catalog::<fp128::Dense>()
        .expect("fp128 workspace schedule catalog");
    let fp64_schedule = fp64_catalog
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(14),
        ))
        .expect("fp64 schedule");
    validate_schedule_ring_dims(fp64_schedule.schedule()).expect("adaptive fp64 schedule envelope");

    let fp128_schedule = fp128_catalog
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(14),
        ))
        .expect("fp128 schedule");
    validate_schedule_ring_dims(fp128_schedule.schedule()).expect("adaptive schedule envelope");
}
