use super::*;

#[test]
fn covers_prefix_compressed_raw_terminal_and_l2_routes() {
    let mut schedule = recursive_schedule(64, 64, true);
    append_recursive_fold(&mut schedule);
    schedule.recursive_folds[1].params.payload_mode = CommitmentPayloadMode::Raw;

    let terminal_matrix = schedule.terminal.inner.matrix;
    let table_key = crate::sis::sis_l2_table_key_for_collision_sq(
        crate::DEFAULT_SIS_SECURITY_POLICY,
        crate::SisL2TableDigest::CURRENT,
        terminal_matrix.sis_modulus_profile(),
        u32::try_from(terminal_matrix.ring_dimension()).expect("test ring dimension"),
        1u128 << 50,
    )
    .expect("generated L2 key");
    schedule.terminal.inner.matrix = crate::InnerCommitMatrixParams::try_new_l2_with_min_rank(
        table_key,
        terminal_matrix.input_width(),
        1u128 << 30,
        crate::PhysicalL2NormProofShape::Direct {
            physical_response_len: terminal_matrix.input_width() * terminal_matrix.ring_dimension(),
        },
    )
    .expect("audited terminal L2 matrix");

    let occurrences = schedule
        .sis_occurrences()
        .expect("valid mixed-topology schedule");
    assert_eq!(occurrences.len(), 22);
    assert_eq!(role_count(&occurrences, ScheduleSisRole::Inner), 5);
    assert_eq!(role_count(&occurrences, ScheduleSisRole::Outer), 4);
    assert_eq!(role_count(&occurrences, ScheduleSisRole::Open), 3);
    assert_eq!(role_count(&occurrences, ScheduleSisRole::Compression), 10);
    assert!(occurrences.iter().any(|occurrence| {
        occurrence.location == "recursive fold 0 setup prefix A"
            && occurrence.role == ScheduleSisRole::Inner
    }));
    assert!(occurrences.iter().any(|occurrence| {
        occurrence.location == "terminal fold A"
            && occurrence.bound == ScheduleSisBound::L2Squared(1u128 << 50)
    }));
    assert!(!occurrences.iter().any(|occurrence| {
        occurrence.location.starts_with("recursive fold 1")
            && occurrence.role == ScheduleSisRole::Compression
    }));

    let canonical_setup = crate::setup_matrix_field_elements_for_schedule(&schedule)
        .expect("canonical occurrence setup envelope");
    let mut level_accumulation = 1;
    crate::accumulate_matrix_field_elements_for_level(
        &schedule.root.params,
        &mut level_accumulation,
    )
    .expect("root setup envelope");
    for fold in &schedule.recursive_folds {
        crate::accumulate_matrix_field_elements_for_level(&fold.params, &mut level_accumulation)
            .expect("recursive setup envelope");
    }
    crate::accumulate_terminal_matrix_field_elements(&schedule.terminal, &mut level_accumulation)
        .expect("terminal setup envelope");
    assert_eq!(canonical_setup, level_accumulation);
}

fn role_count(occurrences: &[ScheduleSisOccurrence], role: ScheduleSisRole) -> usize {
    occurrences
        .iter()
        .filter(|occurrence| occurrence.role == role)
        .count()
}

#[test]
fn rejects_a_structurally_invalid_schedule() {
    let mut schedule = recursive_schedule(64, 64, false);
    schedule.root.input_witness_len = 0;

    assert!(schedule.validate_structure().is_err());
    assert!(schedule.sis_occurrences().is_err());
}

#[test]
fn covers_precommitted_groups_and_preserves_their_identity_binding() {
    let mut schedule = recursive_schedule(64, 64, false);
    let group_layout = PolynomialGroupLayout::singleton(8);
    let mut group_params = schedule.root.params.clone();
    group_params.own_group_mut().opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(group_params.d_a())
            .expect("precommitted test group production challenge");
    let a_bound = execution_admission::exact_test_a_bound(&group_params);
    let inner = group_params.inner().matrix;
    group_params.own_group_mut().profile.inner.matrix =
        crate::sis::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner
                .sis_table_key()
                .expect("L infinity matrix")
                .table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            inner.input_width(),
            a_bound,
            inner.ring_dimension(),
        );
    let outer = group_params.outer().matrix;
    group_params.own_group_mut().profile.outer.matrix =
        crate::sis::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width(),
            3,
            outer.ring_dimension(),
        );
    let precommitted = preceding_group_params(&group_params, group_layout);
    let extra_d_width = precommitted
        .d_segment_width(1, schedule.root.params.role_dims().d_d())
        .expect("root precommitted D width");
    let open = schedule.root.params.open().matrix;
    let widened_open = crate::sis::OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width() + extra_d_width,
        open.coeff_linf_bound(),
        open.ring_dimension(),
    );
    schedule.root.params.open_matrix = widened_open;
    schedule
        .root
        .params
        .set_precommitted_groups(vec![precommitted])
        .unwrap();
    schedule
        .validate_structure()
        .expect("valid root precommitted schedule");

    let occurrences = schedule
        .sis_occurrences()
        .expect("valid precommitted occurrence topology");
    assert!(occurrences
        .iter()
        .any(|occurrence| occurrence.location == "root fold precommitted group 0 A"));
    assert!(occurrences.iter().any(|occurrence| {
        occurrence.location == "root fold precommitted group 0 B compression map 1"
    }));

    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::from_params_unchecked_for_test(
            PolynomialGroupLayout::singleton(8),
            &schedule.root.params,
        ),
        precommitteds: vec![schedule.root.params.precommitted_groups()[0].profile],
    };
    let digest = crate::schedule_row_digest(&profiles, &schedule).expect("row digest");
    let mut changed = schedule;
    changed
        .root
        .params
        .preceding_group_mut_for_test(0)
        .unwrap()
        .opening
        .opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    assert_ne!(
        digest,
        crate::schedule_row_digest(&profiles, &changed)
            .expect("changed root precommitted opening-method digest")
    );
    assert!(changed.validate_structure().is_ok());
    assert!(changed.validate_nonterminal_opening_execution(1).is_err());
}
