use super::*;

#[test]
fn borrowed_schedule_descriptor_matches_materialized_schedule() {
    let mut schedule = recursive_schedule(64, 64, true);
    let precommitted =
        precommitted_group_params(&schedule.root.params, PolynomialGroupLayout::singleton(6));
    schedule
        .root
        .params
        .insert_precommitted_group(precommitted)
        .unwrap();
    schedule
        .root
        .params
        .insert_precommitted_group(precommitted)
        .unwrap();
    append_recursive_fold(&mut schedule);

    let mut steps = Vec::with_capacity(schedule.recursive_folds.len() + 1);
    steps.push(FoldScheduleDescriptorStep {
        params: &schedule.root.params,
        payload_mode: schedule.root.params.payload_mode,
        input_witness_len: schedule.root.input_witness_len,
        output_witness_len: schedule.root.output_witness_len,
    });
    steps.extend(
        schedule
            .recursive_folds
            .iter()
            .map(|fold| FoldScheduleDescriptorStep {
                params: &fold.params,
                payload_mode: fold.params.payload_mode,
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            }),
    );
    let terminal = &schedule.terminal;
    let mut borrowed_descriptor = Vec::new();
    FoldSchedule::append_descriptor_bytes_from_steps(
        &mut borrowed_descriptor,
        steps.into_iter(),
        terminal,
    )
    .unwrap();
    assert_eq!(borrowed_descriptor, schedule.canonical_descriptor_bytes());
}
