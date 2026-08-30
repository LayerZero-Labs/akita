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
use akita_types::{
    ntt_cache_requires_exactness_tail, AkitaScheduleLookupKey, PolynomialGroupLayout,
};

#[cfg(feature = "schedules-default")]
#[test]
fn setup_levels_are_exactly_root_and_recursive_folds() {
    let schedule = fp128::Dense::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(30),
    ))
    .expect("generated fp128 schedule")
    .into_schedule();
    let setup_levels = setup_level_params_from_schedule(&schedule);
    assert_eq!(setup_levels.len(), 1 + schedule.recursive_folds.len());
    assert_eq!(
        setup_levels[0].role_dims(),
        schedule.root.params.role_dims()
    );
}

#[cfg(feature = "schedules-default")]
#[test]
fn generated_schedule_has_explicit_terminal_inner_only_topology() {
    let schedule = fp128::OneHot::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::new(32, 1),
    ))
    .expect("generated one-hot schedule")
    .into_schedule();
    schedule.validate_structure().expect("typed topology");
    assert!(schedule.terminal.inner_width() > 0);
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
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(40, 1));
    let schedule = fp128::OneHot::resolve_catalog_row_for_key(&key)
        .expect("generated one-hot schedule")
        .into_schedule();
    let (step, table_key, response_cap) = schedule
        .recursive_folds
        .iter()
        .find_map(|step| match step.params.inner().matrix.security_route() {
            akita_types::InnerCommitSecurityRoute::Linf(_) => None,
            akita_types::InnerCommitSecurityRoute::L2 {
                table_key,
                response_l2_sq_cap,
                ..
            } => Some((step, table_key, response_l2_sq_cap)),
        })
        .expect("shipped fp128 row must retain one L2 route");
    assert_eq!(
        step.params.fold_challenge_config(),
        akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    );
    assert_eq!(step.params.open().digits.log_basis, 4);
    assert_eq!(step.params.num_digits_fold(), 3);
    assert_eq!(
        step.params.inner().matrix.input_width() * step.params.inner().matrix.ring_dimension(),
        65_536,
    );
    assert_eq!(
        step.params.inner().matrix.output_rank(),
        akita_types::sis::min_secure_l2_rank(
            table_key,
            step.params.inner().matrix.input_width() as u64,
        )
        .expect("shipped L2 geometry must have an audited rank")
    );
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
        crate::honest_fold_policy_of::<fp128::OneHot>(),
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
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(28, 1));
    let schedule = fp64::OneHot::resolve_catalog_row_for_key(&key)
        .expect("generated fp64 schedule")
        .into_schedule();
    assert!(schedule.recursive_folds.iter().any(|step| matches!(
        step.params.inner().matrix.security_route(),
        akita_types::InnerCommitSecurityRoute::L2 { .. }
    )));
    let terminal = &schedule.terminal;
    assert_eq!(
        terminal.fold_challenge_config,
        akita_challenges::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
    );
    assert_eq!(terminal.response_l2_sq_cap(), Some(655_224_517));
    assert_eq!(terminal.inner.matrix.output_rank(), 6);

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
        crate::honest_fold_policy_of::<fp64::OneHot>(),
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
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(28, 1));
    let catalog = fp64::OneHot::schedule_catalog().expect("fp64 one-hot catalog");
    let entry = akita_schedules::generated::table_entry(catalog, &key).expect("catalog row");

    let schedule = fp64::OneHot::resolve_catalog_row_for_key(&key)
        .expect("generated one-hot schedule")
        .into_schedule();
    assert_eq!(
        (
            schedule.terminal.fold.log_basis,
            schedule.terminal.fold.num_digits,
        ),
        (
            entry.terminal.fold_log_basis,
            entry.terminal.fold_digit_count as usize,
        ),
        "expanded terminal must preserve its generated selective-L2 fold geometry"
    );
    assert!(matches!(
        schedule.terminal.inner.matrix.security_route(),
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
            let schedule = Cfg::resolve_catalog_row_for_key(&key)
                .expect("generated schedule must expand")
                .into_schedule();
            schedule.recursive_folds.iter().any(|step| {
                matches!(
                    step.params.inner().matrix.security_route(),
                    akita_types::InnerCommitSecurityRoute::L2 { .. }
                )
            }) || matches!(
                schedule.terminal.inner.matrix.security_route(),
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
    let schedule = fp128::Dense::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::singleton(28),
    ))
    .expect("generated fp128 schedule")
    .into_schedule();
    let envelope = setup_matrix_capacity_for_schedule(&schedule).expect("setup capacity");
    let terminal = &schedule.terminal;
    let terminal_a = terminal
        .inner
        .matrix
        .output_rank()
        .checked_mul(terminal.inner_width())
        .and_then(|width| width.checked_mul(terminal.inner.matrix.ring_dimension()))
        .expect("terminal setup capacity");
    assert!(envelope.num_field_elements >= terminal_a);
}

#[cfg(feature = "schedules-default")]
struct TerminalExactCacheCoverage {
    eligible: usize,
    base: usize,
    tail: usize,
}

#[cfg(feature = "schedules-default")]
fn validate_table_terminal_exact_cache_plans<Cfg: CommitmentConfig>(
    table: GeneratedScheduleTable,
) -> TerminalExactCacheCoverage {
    let policy = crate::policy_of::<Cfg>();
    let mut coverage = TerminalExactCacheCoverage {
        eligible: 0,
        base: 0,
        tail: 0,
    };
    for entry in table.entries {
        if !entry.root.precommitted_groups.is_empty() {
            continue;
        }
        let key = entry.final_group;
        let schedule = schedule_from_entry(
            entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            Cfg::ring_challenge_config,
        )
        .expect("shipped entry should materialize");
        let terminal = &schedule.terminal;
        let width = terminal.inner_width();
        let requires_i16_tail = akita_types::dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            <Cfg as CommitmentConfig>::Field,
            terminal.d_a(),
            |D| {
                ntt_cache_requires_exactness_tail::<<Cfg as CommitmentConfig>::Field, D>(
                    width,
                    1 << 15,
                )
            }
        )
        .expect("generated terminal i16 accumulation should fit");
        coverage.eligible += 1;
        if requires_i16_tail {
            coverage.tail += 1;
        } else {
            coverage.base += 1;
        }
    }
    coverage
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_q32_terminals_have_valid_exact_cache_plans() {
    let coverage = validate_table_terminal_exact_cache_plans::<fp32::OneHot>(fp32_onehot_table());
    assert!(
        coverage.eligible > 0,
        "generated q32 table must contain at least one eligible terminal"
    );
    // Exact backend capacity and the generated terminal width may legitimately
    // select the base route. Every row must still receive one complete plan.
    assert_eq!(coverage.eligible, coverage.base + coverage.tail);
    assert!(
        coverage.tail > 0,
        "generated q32 table must exercise the exact i16-tail route on this target"
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
    let schedule = fp128::OneHot::resolve_catalog_row_for_key(&first.to_runtime_lookup_key())
        .expect("resolve adaptive one-hot row")
        .into_schedule();
    let root = &schedule.root.params;
    assert_eq!(
        root.num_digits_fold(),
        first.root.core.group.num_digits_fold as usize
    );
}

/// The layout scan enumerates single-group shapes only.
///
/// `proof_optimized_schedule_key` is the only route from an
/// `OpeningClaimsLayout` to a catalog row and rejects layouts with more than one
/// group, so a multi-group layout in this list could never be priced. Grouped
/// rows reach the envelope through the catalog scan instead, which
/// `grouped_catalog_rows_are_priced_without_a_multi_group_layout_scan` covers.
#[cfg(feature = "schedules-default")]
#[test]
fn setup_envelope_scan_enumerates_only_single_group_layouts() {
    let layouts = setup_capacity_scan_layouts(14, 3).expect("setup scan layouts");

    assert!(!layouts.is_empty());
    assert!(layouts.iter().all(|layout| layout.groups().len() == 1));
    assert!(layouts
        .iter()
        .any(|layout| layout.groups() == [PolynomialGroupLayout::new(14, 3)]));
    for layout in &layouts {
        assert!(
            crate::proof_optimized::proof_optimized_schedule_key(layout).is_ok(),
            "every scanned layout must be resolvable to a catalog key"
        );
    }
}

/// A grouped catalog row still raises the setup envelope.
///
/// This is the coverage the deleted multi-group layout enumeration appeared to
/// provide: the envelope for a request that admits a two-group root must be at
/// least the grouped row's own matrix footprint.
#[cfg(feature = "schedules-default")]
#[test]
fn grouped_catalog_rows_are_priced_without_a_multi_group_layout_scan() {
    let pre = fp128::OneHot::profile_without_precommitted_groups(PolynomialGroupLayout::new(14, 1))
        .expect("independent one-hot profile");
    let key = akita_types::AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(16, 1),
        precommitteds: vec![pre],
    };
    let grouped = fp128::OneHot::resolve_catalog_row_for_key(&key).expect("grouped catalog row");
    let grouped_fields = setup_matrix_capacity_for_schedule(grouped.schedule())
        .expect("grouped setup capacity")
        .num_field_elements;

    let capacity = fp128::OneHot::setup_matrix_capacity(16, 2).expect("one-hot setup capacity");
    assert!(capacity.num_field_elements >= grouped_fields);
}

#[cfg(feature = "schedules-default")]
#[test]
fn setup_capacity_includes_standalone_precommit_recipes() {
    let profile =
        fp128::Dense::profile_without_precommitted_groups(PolynomialGroupLayout::new(16, 1))
            .expect("independent profile");
    let capacity = fp128::Dense::setup_matrix_capacity(16, 1).expect("dense setup capacity");
    let a_fields = profile.inner.matrix.output_rank()
        * profile.inner.matrix.input_width()
        * profile.inner.matrix.ring_dimension();
    let b_fields = profile.outer.matrix.output_rank()
        * profile.outer.matrix.input_width()
        * profile.outer.matrix.ring_dimension();

    assert!(capacity.num_field_elements >= a_fields);
    assert!(capacity.num_field_elements >= b_fields);
}
