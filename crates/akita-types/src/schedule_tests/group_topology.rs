use super::*;

#[test]
fn schedule_rejects_root_and_own_prefix_topologies_without_unwinding() {
    let mut root_prefix = recursive_schedule(64, 64, false);
    let mut prefix = *root_prefix.root.params.own_group();
    prefix.setup_natural_len = Some(64);
    root_prefix
        .root
        .params
        .set_setup_prefix(Some(prefix))
        .unwrap();
    let validation = std::panic::catch_unwind(|| root_prefix.validate_structure());
    assert!(matches!(validation, Ok(Err(AkitaError::InvalidSetup(_)))));

    let mut own_prefix = recursive_schedule(64, 64, false);
    own_prefix.root.params.own_group_mut().setup_natural_len = Some(64);
    let validation = std::panic::catch_unwind(|| own_prefix.validate_structure());
    assert!(matches!(validation, Ok(Err(AkitaError::InvalidSetup(_)))));
}

#[test]
fn schedule_rejects_mutated_own_profile_without_unwinding() {
    let mut wrong_version = recursive_schedule(64, 64, false);
    wrong_version.root.params.own_group_mut().profile.version = GroupCommitPhaseParams::VERSION + 1;
    let validation = std::panic::catch_unwind(|| wrong_version.validate_structure());
    assert!(matches!(validation, Ok(Err(AkitaError::InvalidSetup(_)))));

    let mut wrong_layout = recursive_schedule(64, 64, false);
    let own_layout = wrong_layout.root.params.own_group().profile.group;
    wrong_layout.root.params.own_group_mut().profile.group =
        PolynomialGroupLayout::new(own_layout.num_vars() + 1, own_layout.num_polynomials());
    let validation = std::panic::catch_unwind(|| wrong_layout.validate_structure());
    assert!(matches!(validation, Ok(Err(AkitaError::InvalidSetup(_)))));
}
