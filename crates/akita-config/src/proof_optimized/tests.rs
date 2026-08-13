use super::*;

#[cfg(feature = "schedules-default")]
use crate::proof_optimized::{fp128, fp32, fp64};
#[cfg(feature = "schedules-default")]
use crate::CommitmentConfig;
#[cfg(feature = "schedules-default")]
use akita_schedules::fp32_onehot_table;
#[cfg(feature = "schedules-default")]
use akita_schedules::{schedule_from_entry, GeneratedScheduleTable};
#[cfg(feature = "schedules-default")]
use akita_types::{ntt_cache_requires_i16_tail, AkitaScheduleLookupKey, PolynomialGroupLayout};

#[cfg(feature = "schedules-default")]
#[test]
fn setup_levels_are_exactly_root_and_recursive_folds() {
    let schedule = fp128::Dense::select_schedule_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(30),
    ))
    .expect("generated fp128 schedule")
    .into_schedule();
    let setup_levels = setup_level_params_from_schedule(&schedule);
    assert_eq!(setup_levels.len(), 1 + schedule.recursive_folds.len());
    assert_eq!(
        setup_levels[0].role_dims(),
        schedule.root.params.final_group.commitment.role_dims()
    );
}

#[cfg(feature = "schedules-default")]
#[test]
fn generated_schedule_has_explicit_terminal_inner_only_topology() {
    let schedule = fp128::OneHot::select_schedule_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(32),
    ))
    .expect("generated one-hot schedule")
    .into_schedule();
    schedule.validate_structure().expect("typed topology");
    assert!(schedule.terminal.params.witness.inner_width() > 0);
    assert_eq!(
        schedule.terminal.input_witness_len,
        schedule
            .recursive_folds
            .last()
            .map_or(schedule.root.output_witness_len, |step| step
                .output_witness_len)
    );
}

#[cfg(feature = "schedules-default")]
#[test]
fn d64_selective_l2_binds_the_certified_operator_norm_family() {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(40));
    let schedule = fp128::OneHot::select_schedule_for_key(&key)
        .expect("generated one-hot schedule")
        .into_schedule();
    let (step, table_key, response_cap) = schedule
        .recursive_folds
        .iter()
        .find_map(
            |step| match step.params.witness.inner_commit_matrix.security_route() {
                akita_types::InnerCommitSecurityRoute::Linf(_) => None,
                akita_types::InnerCommitSecurityRoute::L2 {
                    table_key,
                    response_l2_sq_cap,
                    ..
                } => Some((step, table_key, response_l2_sq_cap)),
            },
        )
        .expect("shipped fp128 row must retain one L2 route");
    assert_eq!(
        step.params.witness.fold_challenge_config,
        akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    );
    assert_eq!(step.params.witness.log_basis_open, 4);
    assert_eq!(step.params.witness.num_digits_fold, 3);
    assert_eq!(
        step.params.witness.inner_commit_matrix.input_width()
            * step.params.witness.inner_commit_matrix.ring_dimension(),
        131_072,
    );
    assert_eq!(step.params.witness.inner_commit_matrix.output_rank(), 4);
    assert_eq!(response_cap, 783_496_643);
    let expected_collision = akita_types::sis::role_a_collision_l2_sq_for_response_bound(
        u128::from(akita_challenges::OperatorNormRejection::D64_SELECTIVE_L2.threshold),
        response_cap,
    )
    .expect("collision bound");
    assert_eq!(
        table_key.collision_l2_sq,
        expected_collision.next_power_of_two()
    );

    let catalog = fp128::OneHot::schedule_catalog().expect("fp128 catalog");
    let entry = akita_schedules::generated::table_entry(catalog, &key).expect("catalog row");
    let proof_bytes = akita_schedules::estimate_proof_bytes(
        entry,
        &key,
        &crate::policy_of::<fp128::OneHot>(),
        fp128::OneHot::ring_challenge_config,
    )
    .expect("proof estimate");
    let mut no_l2_policy = crate::policy_of::<fp128::OneHot>();
    no_l2_policy.selective_l2_response_model =
        akita_schedules::SelectiveL2ResponseModelId::Disabled;
    let no_l2_bytes = akita_planner::find_schedule(
        &key,
        fp128::OneHot::root_honest_fold_policy(),
        &[],
        &no_l2_policy,
        fp128::OneHot::ring_challenge_config,
    )
    .expect("Linf-only schedule")
    .estimate
    .estimated_proof_payload_bytes()
    .expect("Linf-only proof estimate");
    assert!(proof_bytes < no_l2_bytes);
}

#[cfg(feature = "schedules-default")]
#[test]
fn fp64_response_model_selects_globally_winning_l2_suffix() {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(28));
    let schedule = fp64::OneHot::select_schedule_for_key(&key)
        .expect("generated fp64 schedule")
        .into_schedule();
    assert!(schedule.recursive_folds.iter().any(|step| matches!(
        step.params.witness.inner_commit_matrix.security_route(),
        akita_types::InnerCommitSecurityRoute::L2 { .. }
    )));
    let terminal = &schedule.terminal.params;
    assert_eq!(
        terminal.sparse_challenge_config,
        akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    );
    assert_eq!(terminal.witness.response_l2_sq_cap(), Some(2_618_810_696));
    assert_eq!(terminal.witness.inner_commit_matrix.output_rank(), 7);

    let catalog = fp64::OneHot::schedule_catalog().expect("fp64 catalog");
    let entry = akita_schedules::generated::table_entry(catalog, &key).expect("catalog row");
    let proof_bytes = akita_schedules::estimate_proof_bytes(
        entry,
        &key,
        &crate::policy_of::<fp64::OneHot>(),
        fp64::OneHot::ring_challenge_config,
    )
    .expect("proof estimate");
    let mut linf_policy = crate::policy_of::<fp64::OneHot>();
    linf_policy.selective_l2_response_model = akita_schedules::SelectiveL2ResponseModelId::Disabled;
    let linf_schedule = akita_planner::find_schedule(
        &key,
        fp64::OneHot::root_honest_fold_policy(),
        &[],
        &linf_policy,
        fp64::OneHot::ring_challenge_config,
    )
    .expect("fp64 Linf schedule");
    let linf_bytes = linf_schedule
        .estimate
        .estimated_proof_payload_bytes()
        .expect("fp64 proof estimate");
    assert!(proof_bytes < linf_bytes);
}

#[cfg(feature = "schedules-default")]
#[test]
fn terminal_l2_uses_its_catalog_fold_geometry() {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(28));
    let catalog = fp64::OneHot::schedule_catalog().expect("fp64 one-hot catalog");
    let entry = akita_schedules::generated::table_entry(catalog, &key).expect("catalog row");
    assert_eq!(
        (
            entry.terminal.fold_log_basis,
            entry.terminal.fold_digit_count
        ),
        (6, 2)
    );

    let schedule = fp64::OneHot::select_schedule_for_key(&key)
        .expect("generated one-hot schedule")
        .into_schedule();
    assert_eq!(
        (
            schedule.terminal.params.witness.fold_log_basis,
            schedule.terminal.params.witness.fold_digit_count,
        ),
        (6, 2)
    );
    assert!(matches!(
        schedule
            .terminal
            .params
            .witness
            .inner_commit_matrix
            .security_route(),
        akita_types::InnerCommitSecurityRoute::L2 { .. }
    ));
}

#[cfg(feature = "all-schedules")]
#[test]
fn every_generated_profile_opts_in_and_selected_l2_coverage_remains_broad() {
    fn assert_typed_model<Cfg: CommitmentConfig>() {
        let policy = crate::policy_of::<Cfg>();
        assert!(
            matches!(
                policy.selective_l2_response_model,
                akita_schedules::SelectiveL2ResponseModelId::TypedProtocolMomentsV1
            ),
            "{} must use the typed L2 response model",
            std::any::type_name::<Cfg>()
        );
    }

    fn assert_selected_l2<Cfg: CommitmentConfig>() {
        assert_typed_model::<Cfg>();
        let catalog = Cfg::schedule_catalog().expect("generated catalog");
        let has_l2 = catalog.entries.iter().any(|entry| {
            let key = entry.to_runtime_lookup_key();
            let schedule = Cfg::select_schedule_for_key(&key)
                .expect("generated schedule must expand")
                .into_schedule();
            schedule.recursive_folds.iter().any(|step| {
                matches!(
                    step.params.witness.inner_commit_matrix.security_route(),
                    akita_types::InnerCommitSecurityRoute::L2 { .. }
                )
            }) || matches!(
                schedule
                    .terminal
                    .params
                    .witness
                    .inner_commit_matrix
                    .security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            )
        });
        assert!(
            has_l2,
            "{} must ship at least one selected L2 route",
            std::any::type_name::<Cfg>()
        );
    }

    assert_selected_l2::<fp32::Dense>();
    assert_selected_l2::<fp32::OneHot>();
    assert_selected_l2::<fp64::Dense>();
    assert_selected_l2::<fp64::OneHot>();
    assert_selected_l2::<fp128::Dense>();
    assert_selected_l2::<fp128::OneHot>();
    // Selective L2 is not a catalog admission requirement. The current dense
    // W8R2 winner opts into typed modeling but no eligible L2 candidate lowers
    // its A rank, so retaining its Linf-only suffix is the correct outcome.
    assert_typed_model::<fp128::DenseMultiChunk>();
    assert_selected_l2::<fp128::OneHotMultiChunk>();
    assert_selected_l2::<fp128::OneHotMultiChunkW2R2>();
    assert_selected_l2::<fp128::OneHotMultiChunkW4R2>();
    assert_selected_l2::<crate::RecursiveCommitmentConfig<fp128::OneHot>>();
    assert_selected_l2::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>();
}

#[cfg(feature = "schedules-default")]
#[test]
fn setup_capacity_includes_terminal_inner_matrix() {
    let schedule = fp128::Dense::select_schedule_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(28),
    ))
    .expect("generated fp128 schedule")
    .into_schedule();
    let envelope = setup_matrix_capacity_for_schedule(&schedule).expect("setup capacity");
    let terminal = &schedule.terminal.params.witness;
    let terminal_a = terminal
        .inner_commit_matrix
        .output_rank()
        .checked_mul(terminal.inner_width())
        .and_then(|width| width.checked_mul(terminal.inner_commit_matrix.ring_dimension()))
        .expect("terminal setup capacity");
    assert!(envelope.num_field_elements >= terminal_a);
}

#[cfg(feature = "schedules-default")]
fn assert_every_table_terminal_uses_i16_tail<Cfg: CommitmentConfig>(
    table: GeneratedScheduleTable,
) -> (usize, usize) {
    let policy = crate::policy_of::<Cfg>();
    let mut min_width = usize::MAX;
    let mut max_width = 0usize;
    for entry in table.entries {
        if !entry.root.precommitted_groups.is_empty() {
            continue;
        }
        let key = entry.root.final_group.layout;
        let schedule = schedule_from_entry(
            entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            Cfg::ring_challenge_config,
        )
        .expect("shipped entry should materialize");
        let terminal = &schedule.terminal.params.witness;
        let width = terminal.inner_width();
        min_width = min_width.min(width);
        max_width = max_width.max(width);
        let requires_i16_tail = match terminal.d_a() {
            64 => ntt_cache_requires_i16_tail::<Cfg::Field, 64>(width, 1 << 15),
            128 => ntt_cache_requires_i16_tail::<Cfg::Field, 128>(width, 1 << 15),
            dimension => panic!("unsupported generated q32 terminal dimension D{dimension}"),
        };
        assert!(
            requires_i16_tail.expect("generated terminal i16 accumulation should fit"),
            "generated q32 terminal unexpectedly fits the base CRT profile for {} key={key:?}, D={}, width={width}",
            std::any::type_name::<Cfg>(),
            terminal.d_a(),
        );
    }
    assert_ne!(min_width, usize::MAX, "generated table should not be empty");
    (min_width, max_width)
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_q32_terminals_require_the_i16_tail() {
    assert_eq!(
        assert_every_table_terminal_uses_i16_tail::<fp32::OneHot>(fp32_onehot_table()),
        (128, 128),
    );
}

#[cfg(feature = "schedules-default")]
#[test]
fn fp128_adaptive_onehot_catalog_freezes_root_fold_digits() {
    let table = fp128::OneHot::schedule_catalog().expect("fp128 one-hot catalog");
    let first = table
        .entries
        .first()
        .expect("nonempty adaptive one-hot catalog");
    let schedule = fp128::OneHot::select_schedule_for_key(&first.to_runtime_lookup_key())
        .expect("resolve adaptive one-hot row")
        .into_schedule();
    let root = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root.num_digits_fold,
        first.root.final_group.num_digits_fold as usize
    );
}

#[cfg(feature = "schedules-default")]
#[test]
fn setup_envelope_scan_includes_multi_polynomial_precommitted_groups() {
    let layouts = setup_capacity_scan_layouts::<fp128::OneHot>(14, 3).expect("setup scan layouts");

    assert!(layouts.iter().any(|layout| {
        layout.groups()
            == [
                PolynomialGroupLayout::new(14, 2),
                PolynomialGroupLayout::new(14, 1),
            ]
    }));
}

#[cfg(feature = "schedules-default")]
#[test]
fn setup_capacity_includes_standalone_precommit_recipes() {
    let profile =
        fp128::Dense::profile_without_precommitted_groups(PolynomialGroupLayout::new(16, 1))
            .expect("independent profile");
    let capacity = fp128::Dense::setup_matrix_capacity(16, 1).expect("dense setup capacity");
    let a_fields = profile.inner_commit_matrix.output_rank()
        * profile.inner_commit_matrix.input_width()
        * profile.inner_commit_matrix.ring_dimension();
    let b_fields = profile.outer_commit_matrix.output_rank()
        * profile.outer_commit_matrix.input_width()
        * profile.outer_commit_matrix.ring_dimension();

    assert!(capacity.num_field_elements >= a_fields);
    assert!(capacity.num_field_elements >= b_fields);
}
