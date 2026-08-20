use super::*;

#[test]
fn borrowed_schedule_descriptor_matches_materialized_schedule() {
    let mut schedule = recursive_schedule(64, 64, true);
    let precommitted = precommitted_group_params(
        &schedule.root.params.final_group.commitment,
        PolynomialGroupLayout::singleton(6),
    );
    schedule
        .root
        .params
        .final_group
        .commitment
        .precommitted_groups
        .push(precommitted);
    schedule
        .root
        .params
        .precommitted_groups
        .push(RootPrecommittedGroupParams {
            descriptor: precommitted.profile,
            commitment: precommitted,
        });
    append_recursive_fold(&mut schedule);

    let mut steps = Vec::with_capacity(schedule.recursive_folds.len() + 1);
    steps.push(FoldScheduleDescriptorStep {
        params: &schedule.root.params.final_group.commitment,
        payload_mode: schedule.root.params.final_group.commitment.payload_mode,
        input_witness_len: schedule.root.input_witness_len,
        output_witness_len: schedule.root.output_witness_len,
    });
    steps.extend(
        schedule
            .recursive_folds
            .iter()
            .map(|fold| FoldScheduleDescriptorStep {
                params: &fold.params.witness,
                payload_mode: fold.params.witness.payload_mode,
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            }),
    );
    let terminal = TerminalFoldDescriptor {
        witness: &schedule.terminal.params.witness,
        sparse_challenge_config: &schedule.terminal.params.sparse_challenge_config,
        response_shape: &schedule.terminal.params.response_shape,
        input_witness_len: schedule.terminal.input_witness_len,
    };
    let mut borrowed_descriptor = Vec::new();
    FoldSchedule::append_descriptor_bytes_from_steps(
        &mut borrowed_descriptor,
        steps.into_iter(),
        terminal,
    )
    .unwrap();
    assert_eq!(borrowed_descriptor, schedule.canonical_descriptor_bytes());
}
