use super::*;

fn set_valid_setup_prefix(params: &mut CommittedGroupParams, natural_len: usize) {
    provision_setup_prefix_capacity(params, natural_len);
    let commitment_params = crate::setup_prefix_precommitted_params(params, natural_len)
        .expect("setup-prefix commitment params");
    let prefix = crate::scheduled_setup_prefix(natural_len, commitment_params);
    params
        .set_setup_prefix(Some(prefix))
        .expect("valid setup-prefix topology");
}

#[test]
fn rejects_reduced_evaluation_at_root_and_level_one() {
    let mut root_reduced = recursive_schedule(64, 64, false);
    root_reduced.root.params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
    let validation = root_reduced.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("level 0")
        ),
        "unexpected root validation result: {validation:?}"
    );

    let mut level_one_reduced = recursive_schedule(64, 64, false);
    level_one_reduced.recursive_folds[0]
        .params
        .ring_relation_mode = RingRelationMode::ReducedEvaluation;
    let validation = level_one_reduced.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("level 1")
        ),
        "unexpected level-one validation result: {validation:?}"
    );
}

#[test]
fn accepts_reduced_evaluation_suffix_with_independent_payload_phase() {
    for payload_mode in [
        CommitmentPayloadMode::Compressed,
        CommitmentPayloadMode::Raw,
    ] {
        let mut schedule = recursive_schedule(64, 64, false);
        append_recursive_fold(&mut schedule);
        schedule.recursive_folds[1].params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
        schedule.recursive_folds[1].params.payload_mode = payload_mode;

        schedule
            .validate_structure()
            .expect("level-two reduced-evaluation suffix is eligible");
    }
}

#[test]
fn rejects_quotient_lift_after_reduced_evaluation_cutover() {
    let mut schedule = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut schedule);
    append_recursive_fold(&mut schedule);
    schedule.recursive_folds[1].params.ring_relation_mode = RingRelationMode::ReducedEvaluation;

    let validation = schedule.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("ring relation mode disagrees with the reduced-evaluation suffix policy")
        ),
        "unexpected suffix validation result: {validation:?}"
    );
}

#[test]
fn rejects_reduced_evaluation_without_trace_opening() {
    let mut schedule = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut schedule);
    let reduced = &mut schedule.recursive_folds[1].params;
    reduced.ring_relation_mode = RingRelationMode::ReducedEvaluation;
    reduced.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };

    let validation = schedule.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("ring relation mode disagrees with the reduced-evaluation suffix policy")
        ),
        "unexpected opening validation result: {validation:?}"
    );
}

#[test]
fn rejects_setup_prefix_anywhere_in_reduced_evaluation_suffix() {
    let mut at_cutover = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut at_cutover);
    let reduced = &mut at_cutover.recursive_folds[1].params;
    reduced.ring_relation_mode = RingRelationMode::ReducedEvaluation;
    set_valid_setup_prefix(reduced, 64);
    let validation = at_cutover.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("ring relation mode disagrees with the reduced-evaluation suffix policy")
        ),
        "unexpected cutover-prefix validation result: {validation:?}"
    );

    let mut later = recursive_schedule(64, 64, false);
    append_recursive_fold(&mut later);
    append_recursive_fold(&mut later);
    later.recursive_folds[1].params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
    later.recursive_folds[2].params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
    set_valid_setup_prefix(&mut later.recursive_folds[2].params, 64);
    let validation = later.validate_structure();
    assert!(
        matches!(
            validation,
            Err(AkitaError::InvalidSetup(ref message)) if message.contains("ring relation mode disagrees with the reduced-evaluation suffix policy")
        ),
        "unexpected suffix-prefix validation result: {validation:?}"
    );
}
