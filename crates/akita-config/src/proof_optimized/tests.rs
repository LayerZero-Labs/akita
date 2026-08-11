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
    let schedule = fp128::Dense::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(30),
    ))
    .expect("generated fp128 schedule");
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
    let schedule = fp128::OneHot::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(32),
    ))
    .expect("generated one-hot schedule");
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
    let schedule =
        fp128::OneHot::runtime_schedule(key.clone()).expect("generated one-hot schedule");
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
        65_536,
    );
    assert_eq!(step.params.witness.inner_commit_matrix.output_rank(), 3);
    assert_eq!(response_cap, 262_954_353);
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
    eprintln!(
        "D64 selective L2: rank={}, collision_l2_sq={}, response_cap={}, proof_bytes={}, linf_only_bytes={}",
        step.params.witness.inner_commit_matrix.output_rank(),
        table_key.collision_l2_sq,
        response_cap,
        proof_bytes,
        no_l2_bytes,
    );
    assert!(proof_bytes < no_l2_bytes);
}

#[cfg(feature = "schedules-default")]
#[test]
fn fp64_response_model_selects_globally_winning_l2_suffix() {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(28));
    let schedule = fp64::OneHot::runtime_schedule(key.clone()).expect("generated fp64 schedule");
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
    assert_eq!(
        schedule.recursive_folds.len(),
        linf_schedule.schedule.recursive_folds.len(),
        "the modeled L2 suffix should improve bytes without adding a fold"
    );
    eprintln!(
        "fp64 direct terminal L2: rank={}, cap={:?}, proof_bytes={}, linf_only_bytes={}",
        terminal.witness.inner_commit_matrix.output_rank(),
        terminal.witness.response_l2_sq_cap(),
        proof_bytes,
        linf_bytes,
    );
    assert!(proof_bytes < linf_bytes);
}

#[cfg(feature = "schedules-default")]
#[test]
fn terminal_l2_preserves_its_own_fold_geometry() {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(28));
    let catalog = fp128::Dense::schedule_catalog().expect("fp128 dense catalog");
    let entry = akita_schedules::generated::table_entry(catalog, &key).expect("catalog row");
    assert_eq!(
        (
            entry.terminal.fold_log_basis,
            entry.terminal.fold_digit_count
        ),
        (6, 2)
    );

    let schedule = fp128::Dense::runtime_schedule(key).expect("generated dense schedule");
    let predecessor = &schedule
        .recursive_folds
        .last()
        .expect("recursive predecessor")
        .params
        .witness;
    assert_eq!(
        (predecessor.log_basis_open, predecessor.num_digits_fold),
        (5, 2)
    );
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

#[cfg(feature = "schedules-default")]
#[test]
fn response_model_reduces_planned_payload_in_every_field_profile() {
    fn compare<Cfg: CommitmentConfig>(num_vars: usize) -> (usize, usize) {
        let policy = crate::policy_of::<Cfg>();
        assert!(
            matches!(
                policy.selective_l2_response_model,
                akita_schedules::SelectiveL2ResponseModelId::TypedProtocolMomentsV1
            ),
            "{} must use the typed L2 response model",
            std::any::type_name::<Cfg>()
        );
        assert!(
            Cfg::SELECTIVE_L2_FOLD_CAPS.is_empty(),
            "{} must not depend on empirical production caps",
            std::any::type_name::<Cfg>()
        );
        let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
        let catalog = Cfg::schedule_catalog().expect("generated catalog");
        let entry =
            akita_schedules::generated::table_entry(catalog, &key).expect("generated schedule row");
        let schedule = Cfg::runtime_schedule(key.clone()).expect("generated runtime schedule");
        let recursive_l2 = schedule.recursive_folds.iter().any(|step| {
            matches!(
                step.params.witness.inner_commit_matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            )
        });
        let terminal_l2 = matches!(
            schedule
                .terminal
                .params
                .witness
                .inner_commit_matrix
                .security_route(),
            akita_types::InnerCommitSecurityRoute::L2 { .. }
        );
        assert!(
            recursive_l2 || terminal_l2,
            "{} must ship at least one selected L2 route",
            std::any::type_name::<Cfg>()
        );
        let modeled = akita_schedules::estimate_proof_bytes(
            entry,
            &key,
            &crate::policy_of::<Cfg>(),
            Cfg::ring_challenge_config,
        )
        .expect("modeled proof estimate");
        let mut linf_policy = crate::policy_of::<Cfg>();
        linf_policy.selective_l2_response_model =
            akita_schedules::SelectiveL2ResponseModelId::Disabled;
        let linf = akita_planner::find_schedule(
            &key,
            Cfg::root_honest_fold_policy(),
            &[],
            &linf_policy,
            Cfg::ring_challenge_config,
        )
        .expect("L-infinity schedule")
        .estimate
        .estimated_proof_payload_bytes()
        .expect("L-infinity proof estimate");
        assert!(modeled < linf);
        (modeled, linf)
    }

    let fp32_onehot = compare::<fp32::OneHot>(30);
    let fp32_dense = compare::<fp32::Dense>(26);
    let fp64_onehot = compare::<fp64::OneHot>(30);
    let fp64_dense = compare::<fp64::Dense>(26);
    let fp128_onehot = compare::<fp128::OneHot>(36);
    let fp128_dense = compare::<fp128::Dense>(28);
    eprintln!(
        "planned response-model/Linf bytes: fp32 onehot={fp32_onehot:?}, fp32 dense={fp32_dense:?}, fp64 onehot={fp64_onehot:?}, fp64 dense={fp64_dense:?}, fp128 onehot={fp128_onehot:?}, fp128 dense={fp128_dense:?}"
    );

    #[cfg(feature = "all-schedules")]
    {
        let fp128_dense_w8r2 = compare::<fp128::DenseMultiChunk>(16);
        eprintln!("planned response-model/Linf bytes: fp128 dense W8R2={fp128_dense_w8r2:?}");
    }
}

#[cfg(feature = "all-schedules")]
#[test]
fn every_generated_profile_opts_in_and_ships_an_l2_route() {
    fn assert_profile<Cfg: CommitmentConfig>() {
        let policy = crate::policy_of::<Cfg>();
        assert!(
            matches!(
                policy.selective_l2_response_model,
                akita_schedules::SelectiveL2ResponseModelId::TypedProtocolMomentsV1
            ),
            "{} must use the typed L2 response model",
            std::any::type_name::<Cfg>()
        );
        assert!(
            Cfg::SELECTIVE_L2_FOLD_CAPS.is_empty(),
            "{} must not depend on empirical production caps",
            std::any::type_name::<Cfg>()
        );
        let catalog = Cfg::schedule_catalog().expect("generated catalog");
        let has_l2 = catalog.entries.iter().any(|entry| {
            let schedule = Cfg::runtime_schedule(entry.to_runtime_lookup_key())
                .expect("generated schedule must expand");
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

    assert_profile::<fp32::Dense>();
    assert_profile::<fp32::OneHot>();
    assert_profile::<fp64::Dense>();
    assert_profile::<fp64::OneHot>();
    assert_profile::<fp128::Dense>();
    assert_profile::<fp128::OneHot>();
    assert_profile::<fp128::DenseMultiChunk>();
    assert_profile::<fp128::OneHotMultiChunk>();
    assert_profile::<fp128::OneHotMultiChunkW2R2>();
    assert_profile::<fp128::OneHotMultiChunkW4R2>();
    assert_profile::<crate::RecursiveCommitmentConfig<fp128::OneHot>>();
    assert_profile::<crate::RecursiveCommitmentConfig<fp128::OneHotMultiChunk>>();
}

#[cfg(feature = "schedules-default")]
#[test]
fn setup_capacity_includes_terminal_inner_matrix() {
    let schedule = fp128::Dense::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(28),
    ))
    .expect("generated fp128 schedule");
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
fn assert_every_table_terminal_uses_i16_tail<Cfg: CommitmentConfig, const D: usize>(
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
        assert_eq!(terminal.d_a(), D);
        let width = terminal.inner_width();
        min_width = min_width.min(width);
        max_width = max_width.max(width);
        assert!(
            ntt_cache_requires_i16_tail::<Cfg::Field, D>(width, 1 << 15)
                .expect("generated terminal i16 accumulation should fit"),
            "generated q32 terminal unexpectedly fits the base CRT profile for {} key={key:?}, D={D}, width={width}",
            std::any::type_name::<Cfg>(),
        );
    }
    assert_ne!(min_width, usize::MAX, "generated table should not be empty");
    (min_width, max_width)
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_q32_terminals_require_the_i16_tail() {
    assert_eq!(
        assert_every_table_terminal_uses_i16_tail::<fp32::OneHot, 128>(fp32_onehot_table()),
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
    let schedule = fp128::OneHot::runtime_schedule(first.to_runtime_lookup_key())
        .expect("resolve adaptive one-hot row");
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
        crate::committed_group_profile::<fp128::Dense>(&PolynomialGroupLayout::new(16, 1))
            .expect("dense precommit profile");
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
