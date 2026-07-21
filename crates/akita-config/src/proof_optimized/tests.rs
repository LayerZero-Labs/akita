use super::*;
use akita_challenges::SparseChallengeConfig;
#[cfg(feature = "schedules-default")]
use akita_field::{CanonicalField, One};
#[cfg(feature = "schedules-default")]
use akita_planner::generated::GeneratedScheduleTable;
#[cfg(feature = "schedules-default")]
use akita_planner::schedule_from_entry;
#[cfg(feature = "schedules-default")]
use akita_schedules::{
    fp128_d128_full_table, fp128_d128_onehot_table, fp128_d64_full_table, fp128_d64_onehot_table,
};
#[cfg(feature = "schedules-default")]
use akita_schedules::{
    fp32_d128_onehot_table, fp32_d256_onehot_table, fp64_d128_onehot_table, fp64_d128_table,
    fp64_d256_onehot_table,
};
#[cfg(feature = "schedules-default")]
use akita_types::SisModulusProfileId;

#[cfg(feature = "schedules-default")]
const MAX_I8_LOG_BASIS: u32 = 6;
#[cfg(feature = "schedules-default")]
const RAW_I8_RHS_MAX_ABS: u64 = 128;
#[test]
fn setup_level_params_from_schedule_excludes_terminal_direct() {
    // Terminal-direct steps ship the cleartext witness without
    // committing, so they have no `LevelParams` of their own and
    // must not contribute to the FS-bound `setup_levels`. Only
    // the preceding Fold steps (which do commit) appear.
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{
        FoldStep, LevelParamsLike, Schedule, SisModulusProfileId, TerminalWitnessPlan,
    };

    let sparse = SparseChallengeConfig::pm1_only(1);
    let fold_lp =
        LevelParams::params_only(SisModulusProfileId::Q128OffsetA7F7, 64, 3, 1, 1, 1, sparse)
            .with_decomp(1, 4, 1, 1, 1)
            .expect("laid-out fold params");
    let witness_shape = akita_types::SegmentTypedWitnessShape::from_groups(
        &fold_lp,
        128,
        [(&fold_lp as &dyn LevelParamsLike, 1, 1, 1)],
    )
    .expect("terminal witness shape");

    let schedule = Schedule {
        folds: vec![FoldStep {
            params: fold_lp.clone(),
            current_w_len: 1 << 8,
            next_w_len: 1 << 4,
            level_bytes: 0,
        }],
        terminal: TerminalWitnessPlan {
            current_w_len: 1 << 4,
            witness_shape,
            terminal_bytes: 0,
        },
        total_bytes: 0,
    };

    let setup_levels = setup_level_params_from_schedule(&schedule);
    assert_eq!(
        setup_levels,
        vec![fold_lp],
        "terminal direct steps must not feed setup levels"
    );
}

#[test]
fn multi_group_multi_chunk_schedule_resolves_at_effective_schedule_boundary() {
    type Cfg = fp128::D64OneHotMultiChunkW2R2;
    let opening_batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(32, 1),
    ])
    .expect("multi-group opening batch");
    let point = vec![fp128::Field::zero(); opening_batch.max_num_vars()];

    let schedule = crate::effective_batched_schedule::<Cfg>(&opening_batch, &point)
        .expect("canonical group-by-chunk layout must resolve");
    schedule
        .validate_structure()
        .expect("resolved grouped chunk schedule must validate");
}

#[test]
fn setup_matrix_envelope_covers_multi_group_batch_schedules() {
    let opening_batch =
        OpeningClaimsLayout::new(30, 4).expect("multi-group same-point opening_batch");
    let multi_group_same_point = setup_matrix_envelope_for_shape::<fp128::D128Full>(&opening_batch)
        .unwrap()
        .expect("multi-group same-point shape should resolve to a setup envelope");

    let setup_envelope = proof_optimized_max_setup_matrix_size::<fp128::D128Full>(30, 4)
        .expect("setup envelope should cover generated multi-group batch schedules");
    assert!(setup_envelope.max_setup_len >= multi_group_same_point.max_setup_len);
}

#[test]
fn recursive_setup_envelope_covers_selected_exact_group_batch_keys() {
    type Cfg = crate::RecursiveCommitmentConfig<fp128::D64OneHot>;

    let candidates =
        crate::generated_families::recursive_group_batch_candidates_for_capacity::<Cfg>(32, 4)
            .expect("selected recursive setup keys");
    assert!(
        !candidates.is_empty(),
        "recursive setup profile must select exact multi-group keys"
    );

    let setup_envelope =
        proof_optimized_max_setup_matrix_size::<Cfg>(32, 4).expect("recursive setup envelope");
    for key in candidates {
        let schedule = Cfg::runtime_schedule(key.clone()).expect("selected exact-key schedule");
        let required =
            setup_matrix_envelope_for_schedule(&schedule).expect("selected exact-key envelope");
        assert!(
            setup_envelope.max_setup_len >= required.max_setup_len,
            "capacity envelope {} must cover selected-key requirement {} for {key:?}",
            setup_envelope.max_setup_len,
            required.max_setup_len,
        );
    }
}

fn expected_runtime_root_setup_len(lp: &LevelParams, opening_batch: &OpeningClaimsLayout) -> usize {
    if lp.has_precommitted_groups() {
        return expected_multi_group_runtime_root_setup_len(lp, opening_batch);
    }

    let (a_len, b_len, d_width) = expected_group_setup_footprint(
        lp.a_key.row_len(),
        lp.a_key.col_len(),
        lp.b_key.row_len(),
        opening_batch.num_total_polynomials(),
        lp.num_live_blocks,
        lp.num_digits_open,
    );
    expected_root_setup_len(lp.d_key.row_len(), d_width, a_len, b_len)
}

fn expected_group_setup_footprint(
    a_rows: usize,
    a_width: usize,
    b_rows: usize,
    num_polys: usize,
    num_live_blocks: usize,
    num_digits_open: usize,
) -> (usize, usize, usize) {
    let a_len = a_rows * a_width;
    let d_width = num_polys * num_live_blocks * num_digits_open;
    let t_vector_width = a_rows * num_digits_open * num_live_blocks;
    let b_len = b_rows * num_polys * t_vector_width;
    (a_len, b_len, d_width)
}

fn expected_root_setup_len(
    d_rows: usize,
    d_width: usize,
    max_a_len: usize,
    max_b_len: usize,
) -> usize {
    (d_rows * d_width).max(max_a_len).max(max_b_len)
}

fn expected_multi_group_runtime_root_setup_len(
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
) -> usize {
    let final_group_index = lp
        .validate_opening_batch(opening_batch)
        .expect("valid grouped root");
    let final_group = opening_batch
        .group_layout(final_group_index)
        .expect("final group");
    let (mut max_a_len, mut max_b_len, mut d_width) = expected_group_setup_footprint(
        lp.a_key.row_len(),
        lp.a_key.col_len(),
        lp.b_key.row_len(),
        final_group.num_polynomials(),
        lp.num_live_blocks,
        lp.num_digits_open,
    );

    for group in &lp.precommitted_groups {
        let (a_len, b_len, group_d_width) = expected_group_setup_footprint(
            group.a_key.row_len(),
            group.a_key.col_len(),
            group.b_key.row_len(),
            group.layout.group.num_polynomials(),
            group.layout.num_live_blocks,
            group.num_digits_open,
        );
        max_a_len = max_a_len.max(a_len);
        max_b_len = max_b_len.max(b_len);
        d_width += group_d_width;
    }

    expected_root_setup_len(lp.d_key.row_len(), d_width, max_a_len, max_b_len)
}

#[test]
fn setup_matrix_envelope_covers_batched_runtime_root_widths() {
    type Cfg = fp128::D128Full;
    let opening_batch = OpeningClaimsLayout::new(30, 4).expect("batched same-point opening_batch");
    let schedule = Cfg::get_params_for_prove(&opening_batch).expect("runtime schedule");
    let root_params = schedule
        .root_fold()
        .expect("batched root schedule should carry a root fold")
        .params
        .clone();
    let required = expected_runtime_root_setup_len(&root_params, &opening_batch);

    let runtime_envelope = setup_matrix_envelope_for_shape::<Cfg>(&opening_batch)
        .expect("runtime setup envelope")
        .expect("batched root schedule should be supported");
    assert!(runtime_envelope.max_setup_len >= required);

    let setup_envelope = proof_optimized_max_setup_matrix_size::<Cfg>(30, 4)
        .expect("setup envelope should cover generated batched root widths");
    assert!(setup_envelope.max_setup_len >= required);
}

#[test]
fn runtime_setup_guard_rejects_undersized_matrix() {
    type Cfg = fp128::D128Full;
    let opening_batch = OpeningClaimsLayout::new(30, 4).expect("supported opening batch");
    let required = setup_matrix_envelope_for_shape::<Cfg>(&opening_batch)
        .expect("runtime setup envelope")
        .expect("runtime schedule should be supported")
        .max_setup_len;

    super::ensure_required_setup_len(required, required, Cfg::D)
        .expect("exact setup capacity should fit");
    let err = super::ensure_required_setup_len(required, required - 1, Cfg::D)
        .expect_err("undersized setup must be rejected");
    assert!(err.to_string().contains("setup provides"));
}

#[test]
fn setup_matrix_scan_uses_one_shared_opening_point() {
    let opening_batch = OpeningClaimsLayout::new(30, 4).expect("valid opening batch");
    assert_eq!(opening_batch.num_total_polynomials(), 4);
}

#[test]
fn proof_optimized_setup_includes_arbitrary_precommit_group_sizes() {
    type Cfg = fp128::D64OneHot;

    let layout = OpeningClaimsLayout::from_root_groups(
        &[
            PolynomialGroupLayout::new(16, 1),
            PolynomialGroupLayout::new(16, 1),
        ],
        PolynomialGroupLayout::new(32, 1),
    )
    .expect("max precommitted group layout");
    let runtime = setup_matrix_envelope_for_shape::<Cfg>(&layout)
        .expect("runtime setup envelope")
        .expect("max precommitted group should be schedulable");
    let setup_envelope =
        super::proof_optimized_max_setup_matrix_size::<Cfg>(32, 3).expect("setup envelope");

    assert!(
        setup_envelope.max_setup_len >= runtime.max_setup_len,
        "setup envelope must cover multi-group roots at the precommitted num_vars ceiling"
    );
}

#[test]
fn setup_envelope_dominates_bounded_precommit_shape_grid() {
    type Cfg = fp128::D64OneHot;

    const MIN_NUM_VARS: usize = 16;
    const MAX_NUM_VARS: usize = 32;
    let setup_envelope = super::proof_optimized_max_setup_matrix_size::<Cfg>(MAX_NUM_VARS, 1)
        .expect("bounded setup envelope");
    let mut schedulable_shapes = 0usize;

    for main_num_vars in MIN_NUM_VARS..=MAX_NUM_VARS {
        let main_group = PolynomialGroupLayout::new(main_num_vars, 1);
        for first_num_vars in MIN_NUM_VARS..=MAX_NUM_VARS {
            for second_num_vars in 0..=MAX_NUM_VARS {
                if second_num_vars != 0 && second_num_vars < MIN_NUM_VARS {
                    continue;
                }
                let mut precommitted = vec![PolynomialGroupLayout::new(first_num_vars, 1)];
                if second_num_vars != 0 {
                    precommitted.push(PolynomialGroupLayout::new(second_num_vars, 1));
                }
                let layout = OpeningClaimsLayout::from_root_groups(&precommitted, main_group)
                    .expect("bounded precommit layout");
                let Some(required) = setup_matrix_envelope_for_shape::<Cfg>(&layout)
                    .expect("bounded precommit shape must either schedule or be infeasible")
                else {
                    continue;
                };
                schedulable_shapes += 1;
                assert!(
                    setup_envelope.max_setup_len >= required.max_setup_len,
                    "setup envelope {} does not cover requirement {} for main_num_vars={main_num_vars}, precommitted={precommitted:?}",
                    setup_envelope.max_setup_len,
                    required.max_setup_len,
                );
            }
        }
    }

    assert!(
        schedulable_shapes > 0,
        "bounded shape grid must exercise at least one schedulable precommit layout"
    );
}

#[test]
fn grouped_root_runtime_setup_uses_per_group_roles_and_summed_d_width() {
    type Cfg = fp128::D64OneHot;

    let layout = OpeningClaimsLayout::from_root_groups(
        &[
            PolynomialGroupLayout::new(16, 1),
            PolynomialGroupLayout::new(16, 1),
        ],
        PolynomialGroupLayout::new(32, 1),
    )
    .expect("grouped root layout");
    let key = super::proof_optimized_schedule_key::<Cfg>(&layout).expect("grouped root key");
    let schedule = Cfg::runtime_schedule(key).expect("grouped root schedule");
    let root_params = schedule
        .root_fold()
        .expect("grouped root should carry a root fold")
        .params
        .clone();

    let expected = expected_runtime_root_setup_len(&root_params, &layout);
    let actual = super::root_runtime_matrix_len_for_opening_batch(&root_params, &layout)
        .expect("grouped root runtime setup len");
    assert_eq!(
        actual, expected,
        "grouped-root setup footprint must use per-group A/B widths and shared D width"
    );

    let final_group = layout.root_final_group_layout().expect("final group");
    let precommitted_d_width: usize = root_params
        .precommitted_groups
        .iter()
        .map(|group| group.d_segment_width().expect("precommitted D width"))
        .sum();
    let expected_d_width =
        final_group.num_polynomials() * root_params.num_live_blocks * root_params.num_digits_open
            + precommitted_d_width;
    assert_eq!(
        root_params.d_key.col_len(),
        expected_d_width,
        "multi-group root D columns are final plus all precommitted segments"
    );
}

#[test]
fn recursive_setup_envelope_counts_setup_prefix_d_segment() {
    use akita_types::{
        padded_setup_prefix_len, setup_prefix_precommitted_params, setup_prefix_slot_id,
        AjtaiKeyParams, DecompositionParams, FoldStep, LevelParamsLike, SegmentTypedWitnessShape,
        SetupContributionMode, TerminalWitnessPlan, SETUP_OFFLOAD_D_SETUP,
    };

    fn scalar_level_params() -> LevelParams {
        let full_field_digits = akita_types::sis::compute_num_digits_full_field(128, 2);
        LevelParams::params_only(
            akita_types::SisModulusProfileId::Q128OffsetA7F7,
            SETUP_OFFLOAD_D_SETUP,
            2,
            2,
            3,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(2, 3, full_field_digits, 2, 2)
        .expect("scalar params")
    }

    fn terminal_direct_step(params: &LevelParams) -> TerminalWitnessPlan {
        let witness_shape = SegmentTypedWitnessShape::from_groups(
            params,
            128,
            [(params as &dyn LevelParamsLike, 1, 1, 1)],
        )
        .expect("segment-typed witness shape");
        TerminalWitnessPlan {
            current_w_len: witness_shape.layout.logical_num_elems,
            witness_shape,
            terminal_bytes: 0,
        }
    }

    fn add_setup_prefix_d_width(params: &mut LevelParams) {
        let prefix_d_width = params
            .setup_prefix
            .as_ref()
            .expect("setup prefix")
            .commitment_params
            .d_segment_width()
            .expect("setup-prefix D width");
        let d_width = params
            .d_key
            .col_len()
            .checked_add(prefix_d_width)
            .expect("D width");
        params.d_key = AjtaiKeyParams::new_unchecked(
            params.d_key.security_policy(),
            params.d_key.sis_table_key().table_digest,
            params.d_key.sis_modulus_profile(),
            params.d_key.sis_table_key().role,
            params.d_key.row_len(),
            d_width,
            params.d_key.coeff_linf_bound(),
            params.ring_dimension,
        );
    }

    fn recursive_schedule(_layout: &OpeningClaimsLayout) -> Schedule {
        let mut root = scalar_level_params();
        root.setup_contribution_mode = SetupContributionMode::Recursive;

        let natural_len = 129usize;
        let n_prefix = padded_setup_prefix_len(natural_len);
        let mut successor = scalar_level_params();
        successor.setup_prefix = Some(setup_prefix_slot_id(
            SETUP_OFFLOAD_D_SETUP,
            natural_len,
            setup_prefix_precommitted_params(&root, n_prefix).expect("prefix params"),
        ));
        add_setup_prefix_d_width(&mut successor);

        let terminal_params = scalar_level_params();
        let direct = terminal_direct_step(&terminal_params);
        let terminal_current_w_len = 64;
        Schedule {
            folds: vec![
                FoldStep {
                    params: root,
                    current_w_len: 256,
                    next_w_len: 128,
                    level_bytes: 0,
                },
                FoldStep {
                    params: successor,
                    current_w_len: 128,
                    next_w_len: terminal_current_w_len,
                    level_bytes: 0,
                },
                FoldStep {
                    params: terminal_params,
                    current_w_len: terminal_current_w_len,
                    next_w_len: direct.current_w_len,
                    level_bytes: 0,
                },
            ],
            terminal: direct,
            total_bytes: 0,
        }
    }

    #[derive(Clone)]
    struct SyntheticRecursiveCfg;
    impl CommitmentConfig for SyntheticRecursiveCfg {
        type Field = akita_field::Prime128OffsetA7F7;
        type ExtField = akita_field::Prime128OffsetA7F7;
        const D: usize = SETUP_OFFLOAD_D_SETUP;

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 128,
                log_open_bound: None,
            }
        }

        fn ring_challenge_config(_d: usize) -> Result<SparseChallengeConfig, AkitaError> {
            Ok(SparseChallengeConfig::pm1_only(3))
        }

        fn sis_modulus_profile() -> akita_types::SisModulusProfileId {
            akita_types::SisModulusProfileId::Q128OffsetA7F7
        }

        fn max_setup_matrix_size(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
        ) -> Result<SetupMatrixEnvelope, AkitaError> {
            Ok(SetupMatrixEnvelope { max_setup_len: 1 })
        }

        fn basis_range() -> (u32, u32) {
            (3, 3)
        }

        fn recursive_setup_planning() -> bool {
            true
        }

        fn get_params_for_prove(layout: &OpeningClaimsLayout) -> Result<Schedule, AkitaError> {
            Ok(recursive_schedule(layout))
        }
    }

    let layout = OpeningClaimsLayout::new(5, 1).expect("recursive opening layout");
    let schedule =
        SyntheticRecursiveCfg::get_params_for_prove(&layout).expect("synthetic recursive schedule");
    schedule
        .validate_structure()
        .expect("synthetic recursive schedule is valid");
    let envelope = setup_matrix_envelope_for_shape::<SyntheticRecursiveCfg>(&layout)
        .expect("setup envelope")
        .expect("recursive schedule should be supported");

    let mut saw_setup_prefix = false;
    for fold in schedule.fold_steps() {
        let Some(slot) = &fold.params.setup_prefix else {
            continue;
        };
        saw_setup_prefix = true;
        let prefix_d_width = slot
            .commitment_params
            .d_segment_width()
            .expect("setup-prefix D segment width");
        assert!(
            fold.params.d_matrix_width() >= prefix_d_width,
            "consuming fold D width must include setup-prefix e_hat columns"
        );
        let fold_d_len = fold.params.d_key.row_len() * fold.params.d_matrix_width();
        assert!(
            envelope.max_setup_len >= fold_d_len,
            "matrix envelope must cover the consuming fold's shared D matrix"
        );
        let mut slot_envelope = akita_types::SetupMatrixEnvelope { max_setup_len: 1 };
        crate::matrix_envelope::inflate_envelope_for_setup_prefix_slot(&mut slot_envelope, slot)
            .expect("setup-prefix slot envelope");
        assert!(
            envelope.max_setup_len >= slot_envelope.max_setup_len,
            "matrix envelope must cover setup-prefix storage and A/B matrices"
        );
    }

    assert!(
        saw_setup_prefix,
        "fixture must exercise a setup-prefix fold"
    );
}

#[test]
fn presets_select_expected_sis_modulus_profile() {
    assert_eq!(
        <fp128::D64Full as CommitmentConfig>::sis_modulus_profile(),
        akita_types::SisModulusProfileId::Q128OffsetA7F7
    );
    assert_eq!(
        <fp32::D64Full as CommitmentConfig>::sis_modulus_profile(),
        akita_types::SisModulusProfileId::Q32Offset99
    );
    assert_eq!(
        <fp64::D64Full as CommitmentConfig>::sis_modulus_profile(),
        akita_types::SisModulusProfileId::Q64Offset59
    );
}

// ----- migrated from former `schedule_policy::tests` -------------------

fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(num_vars: usize) {
    assert_plan_matches_runtime_w_sizes_for_key::<Cfg>(PolynomialGroupLayout::singleton(num_vars));
}

fn assert_plan_matches_runtime_w_sizes_for_key<Cfg: CommitmentConfig>(key: PolynomialGroupLayout) {
    let schedule =
        Cfg::runtime_schedule(AkitaScheduleLookupKey::single(key)).expect("planner should succeed");
    let num_fold_levels = schedule.num_fold_levels();
    for (idx, fold) in schedule.fold_steps().enumerate() {
        let is_terminal_fold = idx + 1 == num_fold_levels;
        let runtime_next_w_len = if is_terminal_fold {
            // Terminal W is quotient-free. Its canonical shape is the Direct
            // successor materialized by the planner, not an M-matrix row count.
            schedule.terminal.witness_shape.logical_num_elems()
        } else {
            // Root-level batched witnesses fan out over the key's polynomial
            // count; recursive levels collapse back to singleton-by-construction.
            let (num_polynomials, num_public_rows) = if idx == 0 {
                (key.num_polynomials(), 1)
            } else {
                (1, 1)
            };
            akita_types::intermediate_w_ring_element_count_with_counts::<Cfg::Field>(
                &fold.params,
                num_polynomials,
                num_public_rows,
            )
            .expect("valid planned witness")
                * fold.params.ring_dimension
        };
        assert_eq!(
            runtime_next_w_len, fold.next_w_len,
            "planner/runtime next_w_len mismatch at level {idx} for key={key:?}",
        );
    }
}

#[cfg(feature = "schedules-default")]
fn assert_every_table_entry_materializes<Cfg: CommitmentConfig>(table: GeneratedScheduleTable) {
    let policy = crate::policy_of::<Cfg>();
    for entry in table.entries {
        if !entry.precommitteds.is_empty() {
            continue;
        }
        let key = PolynomialGroupLayout::new(
            entry.final_group.num_vars(),
            entry.final_group.num_polynomials(),
        );
        schedule_from_entry(
            entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )
        .expect("shipped entry should materialize");
    }
}

#[cfg(feature = "schedules-default")]
fn crt_product_for_small_field_cfg<Cfg: CommitmentConfig>() -> (&'static str, u128) {
    match Cfg::sis_modulus_profile() {
        SisModulusProfileId::Q32Offset99 => ("Q32/2xi32", 1_152_837_945_367_908_353),
        SisModulusProfileId::Q64Offset59 => ("Q64/3xi32", 1_237_793_655_097_897_487_951_597_569),
        family => panic!("small-field capacity test does not cover {family:?}"),
    }
}

#[cfg(feature = "schedules-default")]
fn small_field_single_term_safe_width<Cfg: CommitmentConfig>(
    ring_dimension: usize,
    rhs_abs_bound: u64,
) -> Option<usize> {
    if rhs_abs_bound == 0 || ring_dimension == 0 {
        return None;
    }
    let (_profile_id, crt_product) = crt_product_for_small_field_cfg::<Cfg>();
    let modulus = (-Cfg::Field::one()).to_canonical_u128() + 1;
    let setup_abs_bound = modulus / 2;
    let denom = 2u128
        .checked_mul(ring_dimension as u128)?
        .checked_mul(setup_abs_bound)?
        .checked_mul(u128::from(rhs_abs_bound))?;
    if denom == 0 || crt_product <= denom {
        return None;
    }
    let width = (crt_product - 1) / denom;
    usize::try_from(width).ok()
}

#[cfg(feature = "schedules-default")]
fn assert_level_has_crt_i8_capacity<Cfg: CommitmentConfig>(
    key: PolynomialGroupLayout,
    level: &LevelParams,
) {
    assert!(
        [
            level.log_basis_inner,
            level.log_basis_outer,
            level.log_basis_open
        ]
        .into_iter()
        .all(|basis| (1..=MAX_I8_LOG_BASIS).contains(&basis)),
        "generated schedule uses semantic basis outside the i8 kernel contract for {} key={key:?}",
        std::any::type_name::<Cfg>()
    );
    let balanced_digit_bound = 1u64 << (level.log_basis_open - 1);
    let (profile_id, _product) = crt_product_for_small_field_cfg::<Cfg>();
    for (role, rhs_abs_bound) in [
        ("schedule balanced digit", balanced_digit_bound),
        ("max balanced i8 digit", 1u64 << (MAX_I8_LOG_BASIS - 1)),
        ("raw signed-i8", RAW_I8_RHS_MAX_ABS),
    ] {
        let safe_width =
            small_field_single_term_safe_width::<Cfg>(level.ring_dimension, rhs_abs_bound);
        assert!(
            matches!(safe_width, Some(width) if width > 0),
            "{profile_id} has no single-term CRT capacity for {role} at D={} rhs_abs_bound={} in {} key={key:?}",
            level.ring_dimension,
            rhs_abs_bound,
            std::any::type_name::<Cfg>()
        );
    }
}

#[cfg(feature = "schedules-default")]
fn assert_every_table_entry_has_crt_i8_capacity<Cfg: CommitmentConfig>(
    table: GeneratedScheduleTable,
) {
    let policy = crate::policy_of::<Cfg>();
    for entry in table.entries {
        if !entry.precommitteds.is_empty() {
            continue;
        }
        let key = PolynomialGroupLayout::new(
            entry.final_group.num_vars(),
            entry.final_group.num_polynomials(),
        );
        let schedule = schedule_from_entry(
            entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )
        .expect("shipped entry should materialize");
        let levels = setup_level_params_from_schedule(&schedule);
        for level in &levels {
            assert_level_has_crt_i8_capacity::<Cfg>(key, level);
        }
    }
}

#[cfg(feature = "schedules-default")]
fn assert_generated_batched_roots_are_scaled<Cfg: CommitmentConfig>(table: GeneratedScheduleTable) {
    let policy = crate::policy_of::<Cfg>();
    let mut checked_folded_entry = false;
    for entry in table
        .entries
        .iter()
        .filter(|entry| entry.precommitteds.is_empty() && entry.final_group.num_polynomials() > 1)
    {
        let key = PolynomialGroupLayout::new(
            entry.final_group.num_vars(),
            entry.final_group.num_polynomials(),
        );
        let generated = schedule_from_entry(
            entry,
            &AkitaScheduleLookupKey::single(key),
            &policy,
            Cfg::ring_challenge_config,
            Cfg::fold_challenge_shape_at_level,
        )
        .expect("shipped entry should materialize");
        let Some(root) = generated.fold_steps().next() else {
            continue;
        };
        checked_folded_entry = true;
        let root_lp = &root.params;
        let singleton_outer_width =
            root_lp.a_key.row_len() * root_lp.num_digits_open * root_lp.num_live_blocks;
        let singleton_d_width = root_lp.num_digits_open * root_lp.num_live_blocks;
        assert_eq!(
            root_lp.outer_width(),
            singleton_outer_width * entry.final_group.num_polynomials(),
            "generated batched root B width should be claim-scaled for key={key:?}"
        );
        assert_eq!(
            root_lp.d_matrix_width(),
            singleton_d_width * entry.final_group.num_polynomials(),
            "generated batched root D width should be claim-scaled for key={key:?}"
        );
    }
    assert!(
        checked_folded_entry,
        "generated table should include at least one folded batched entry"
    );
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_fp128_schedule_tables_match_cfg_schedule() {
    assert_every_table_entry_materializes::<fp128::D128Full>(fp128_d128_full_table());
    assert_every_table_entry_materializes::<fp128::D128OneHot>(fp128_d128_onehot_table());
    assert_every_table_entry_materializes::<fp128::D64Full>(fp128_d64_full_table());
    assert_every_table_entry_materializes::<fp128::D64OneHot>(fp128_d64_onehot_table());
}

#[cfg(feature = "schedules-fp128-d64-onehot-recursive")]
#[test]
fn recursive_d64_onehot_empty_key_delegates_to_scalar_catalog() {
    type Cfg = crate::RecursiveCommitmentConfig<fp128::D64OneHot>;
    let schedule = <Cfg as CommitmentConfig>::runtime_schedule(AkitaScheduleLookupKey::single(
        PolynomialGroupLayout::new(46, 1),
    ))
    .expect("empty-precommit recursive config should delegate to scalar catalog");

    for fold in schedule.fold_steps() {
        assert_eq!(
            fold.params.setup_contribution_mode,
            akita_types::SetupContributionMode::Direct,
            "empty-precommit scalar keys must not materialize recursive setup contribution"
        );
        assert!(
            fold.params.setup_prefix.is_none(),
            "empty-precommit scalar keys must not carry setup_prefix groups"
        );
    }
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_small_field_schedule_tables_match_cfg_schedule() {
    assert_every_table_entry_materializes::<fp32::D128OneHot>(fp32_d128_onehot_table());
    assert_every_table_entry_materializes::<fp32::D256OneHot>(fp32_d256_onehot_table());
    assert_every_table_entry_materializes::<fp64::D128Full>(fp64_d128_table());
    assert_every_table_entry_materializes::<fp64::D128OneHot>(fp64_d128_onehot_table());
    assert_every_table_entry_materializes::<fp64::D256OneHot>(fp64_d256_onehot_table());
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_small_field_schedule_tables_have_crt_i8_capacity() {
    assert_every_table_entry_has_crt_i8_capacity::<fp32::D128OneHot>(fp32_d128_onehot_table());
    assert_every_table_entry_has_crt_i8_capacity::<fp32::D256OneHot>(fp32_d256_onehot_table());
    assert_every_table_entry_has_crt_i8_capacity::<fp64::D128Full>(fp64_d128_table());
    assert_every_table_entry_has_crt_i8_capacity::<fp64::D128OneHot>(fp64_d128_onehot_table());
    assert_every_table_entry_has_crt_i8_capacity::<fp64::D256OneHot>(fp64_d256_onehot_table());
}

#[test]
#[cfg(feature = "schedules-default")]
fn generated_batched_roots_restore_scaled_widths() {
    assert_generated_batched_roots_are_scaled::<fp128::D128Full>(fp128_d128_full_table());
    assert_generated_batched_roots_are_scaled::<fp128::D128OneHot>(fp128_d128_onehot_table());
    assert_generated_batched_roots_are_scaled::<fp128::D64OneHot>(fp128_d64_onehot_table());
}

#[test]
fn adaptive_bounded_plan_matches_runtime_next_w_len() {
    for num_vars in [14, 20, 30] {
        assert_plan_matches_runtime_w_sizes::<fp128::D64Full>(num_vars);
    }
}

#[test]
fn adaptive_onehot_plan_matches_runtime_next_w_len() {
    for num_vars in [15, 30, 44] {
        assert_plan_matches_runtime_w_sizes::<fp128::D64OneHot>(num_vars);
    }
}

#[test]
#[cfg(feature = "schedules-default")]
fn batched_root_plan_matches_runtime_next_w_len() {
    let table = fp128_d64_onehot_table();
    let entry = table
        .entries
        .iter()
        .find(|entry| entry.precommitteds.is_empty() && entry.final_group.num_polynomials() > 1)
        .expect("generated table should contain a non-singleton batched-root row");
    let key = PolynomialGroupLayout::new(
        entry.final_group.num_vars(),
        entry.final_group.num_polynomials(),
    );

    assert_plan_matches_runtime_w_sizes_for_key::<fp128::D64OneHot>(key);
}

#[test]
fn batched_onehot_4x30_plan_keeps_terminal_witness_bounded() {
    let key = PolynomialGroupLayout::new(30, 4);
    let schedule = <fp128::D64OneHot as CommitmentConfig>::runtime_schedule(
        AkitaScheduleLookupKey::single(key),
    )
    .expect("config schedule should succeed");

    assert_plan_matches_runtime_w_sizes_for_key::<fp128::D64OneHot>(key);
    assert!(
        schedule.num_fold_levels() > 2,
        "4x30 onehot schedule should keep a recursive suffix after the root fold"
    );

    let shape = &schedule.terminal.witness_shape;
    // Bound reflects the committed-fold A-role SIS pricing: honest pricing
    // lifts the per-level rank, widening the terminal witness, but the
    // byte-aware schedule still keeps folding rather than dumping a huge
    // cleartext root.
    assert!(
        shape.layout.logical_num_elems <= 375_104,
        "expected byte-aware batched schedule to keep folding, got final_w with {} elems",
        shape.layout.logical_num_elems
    );
}

#[test]
fn power_of_two_positions_cover_exact_source() {
    for num_vars in [14, 20, 30] {
        let schedule = fp128::D64Full::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(num_vars),
        ))
        .expect("planner should succeed");
        for (level_idx, fold) in schedule.fold_steps().enumerate() {
            let lp = &fold.params;
            let pow2_block = 1usize << lp.position_index_bits();
            assert!(
                lp.num_positions_per_block <= pow2_block,
                "num_positions_per_block {} should be <= 2^position_index_bits {} at level {level_idx} (num_vars={num_vars})",
                lp.num_positions_per_block,
                pow2_block,
            );
            if level_idx > 0 {
                let num_ring = fold.current_w_len / lp.ring_dimension;
                let expected_position_count =
                    num_ring.div_ceil(lp.num_live_blocks).next_power_of_two();
                assert_eq!(
                    lp.num_positions_per_block, expected_position_count,
                    "recursive level {level_idx} should use the least power-of-two num_positions_per_block covering ceil({num_ring} / {})",
                    lp.num_live_blocks
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ring-challenge soundness guards
// ---------------------------------------------------------------------------
//
// Every proof-optimized preset folds against a short ring challenge whose
// support sets the Fiat-Shamir soundness of the fold. These tests pin the
// shared dimension-keyed policy to its designed >=128-bit families and assert
// no preset can silently regress to a low-support family (the historical
// `Uniform { weight: 8, [-1, 1] }`, which has only ~31 bits at small D).

/// `log2` of the binomial coefficient `C(n, k)`, summed over logs so the large
/// `(D, weight)` pairs used by these families never overflow.
fn log2_binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    (1..=k)
        .map(|i| ((n - k + i) as f64 / i as f64).log2())
        .sum::<f64>()
}

/// Bits of Fiat-Shamir support in a ring-challenge family at ring degree `d`.
fn challenge_support_bits(cfg: &SparseChallengeConfig, d: usize) -> f64 {
    let w = cfg.weight();
    log2_binomial(d, w) + log2_binomial(w, cfg.count_pm1) + w as f64
}

#[test]
fn proof_optimized_ring_challenge_policy_pins_secure_families() {
    // (d, expected family, (l1, linf)). Each family must clear 128 bits of
    // Fiat-Shamir support; the (l1, linf) pin guards the folded-witness norm
    // the schedules are generated against.
    let cases = [
        (
            64,
            SparseChallengeConfig {
                count_pm1: akita_challenges::D64_PRODUCTION_PM1_COUNT,
                count_pm2: akita_challenges::D64_PRODUCTION_PM2_COUNT,
            },
            (51, 2),
        ),
        (128, SparseChallengeConfig::pm1_only(31), (31, 1)),
        (256, SparseChallengeConfig::pm1_only(23), (23, 1)),
        (512, SparseChallengeConfig::pm1_only(19), (19, 1)),
        (1024, SparseChallengeConfig::pm1_only(16), (16, 1)),
        (2048, SparseChallengeConfig::pm1_only(14), (14, 1)),
    ];
    for (d, expected, (l1, linf)) in cases {
        let got = proof_optimized_ring_challenge_config(d).unwrap();
        assert_eq!(got, expected, "ring-challenge family changed at d={d}");
        assert_eq!(
            (got.l1_norm(), got.infinity_norm()),
            (l1, linf),
            "ring-challenge norms changed at d={d}"
        );
        let bits = challenge_support_bits(&got, d);
        assert!(
            bits >= 128.0,
            "ring-challenge family {got:?} at d={d} has only {bits:.1} bits of support (<128)"
        );
    }

    let d64 = proof_optimized_ring_challenge_config(64).unwrap();
    assert_eq!(d64.l1_norm(), 51);
    assert_eq!(d64.challenge_l2_sq_max(), 71);

    // d_a=32 has no sparse fold challenge; non-power-of-two degrees are rejected.
    assert!(proof_optimized_ring_challenge_config(32).is_err());
    assert!(proof_optimized_ring_challenge_config(16).is_err());
    assert!(proof_optimized_ring_challenge_config(48).is_err());
}

#[test]
#[cfg(feature = "schedules-fp128-d64-onehot")]
fn d64_shipped_catalog_ring_challenge_digest_matches_runtime_policy() {
    use akita_planner::ring_challenge_config_digest;

    use crate::proof_optimized::fp128;

    let expected = ring_challenge_config_digest(&[64], fp128::D64OneHot::ring_challenge_config)
        .expect("d=64 ring challenge digest");
    let catalog = fp128::D64OneHot::schedule_catalog().expect("D64 one-hot catalog");
    assert_eq!(
        catalog.identity.ring_challenge_config_digest, expected,
        "shipped fp128_d64_onehot digest must track proof_optimized_ring_challenge_config"
    );
}

/// Assert one preset delegates its ring challenge to the shared policy.
/// Support for 128-bit-and-larger fields in each shared family is proven once in
/// `proof_optimized_ring_challenge_policy_pins_secure_families`, so this only
/// has to catch a preset that bypasses the shared helper with a weaker family.
fn assert_preset_uses_shared_ring_challenge<Cfg: CommitmentConfig>() {
    let name = std::any::type_name::<Cfg>();
    let got = Cfg::ring_challenge_config(Cfg::D)
        .unwrap_or_else(|err| panic!("{name} ring_challenge_config(D) failed: {err}"));
    let want = proof_optimized_ring_challenge_config(Cfg::D).unwrap();
    assert_eq!(
        got, want,
        "{name} bypassed the shared ring-challenge policy"
    );
}

#[test]
fn all_proof_optimized_presets_use_shared_ring_challenge() {
    assert_preset_uses_shared_ring_challenge::<fp32::D64Full>();
    assert_preset_uses_shared_ring_challenge::<fp32::D64OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp32::D128Full>();
    assert_preset_uses_shared_ring_challenge::<fp32::D128OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp32::D256Full>();
    assert_preset_uses_shared_ring_challenge::<fp32::D256OneHot>();

    assert_preset_uses_shared_ring_challenge::<fp64::D64Full>();
    assert_preset_uses_shared_ring_challenge::<fp64::D64OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp64::D128Full>();
    assert_preset_uses_shared_ring_challenge::<fp64::D128OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp64::D256Full>();
    assert_preset_uses_shared_ring_challenge::<fp64::D256OneHot>();

    assert_preset_uses_shared_ring_challenge::<fp128::D64Full>();
    assert_preset_uses_shared_ring_challenge::<fp128::D64OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp128::D64OneHotK16>();
    assert_preset_uses_shared_ring_challenge::<fp128::D128Full>();
    assert_preset_uses_shared_ring_challenge::<fp128::D128OneHot>();

    // Hand-written (non-macro) preset: guards that the bespoke impl still
    // routes through the shared policy.
    assert_preset_uses_shared_ring_challenge::<crate::tensor_verifier::fp128::D64OneHotTensor>();
}

#[test]
fn d64_onehot_k16_uses_the_canonical_chunk_policy_without_a_catalog() {
    assert_eq!(fp128::D64OneHotK16::onehot_chunk_size(), 16);
    assert!(fp128::D64OneHotK16::schedule_catalog().is_none());
}

#[test]
fn tensor_onehot_preset_keeps_d64_onehot_chunk_size() {
    assert_eq!(
        crate::tensor_verifier::fp128::D64OneHotTensor::onehot_chunk_size(),
        fp128::D64OneHot::onehot_chunk_size(),
        "tensor verifier preset must preserve the D64 one-hot witness sparsity envelope"
    );
}

#[test]
fn setup_capacity_scan_rejects_hostile_metadata_before_planning() {
    let vars_err =
        proof_optimized_max_setup_matrix_size::<fp128::D64OneHot>(usize::BITS as usize, 1)
            .expect_err("unrepresentable polynomial variable count must fail closed");
    assert!(matches!(vars_err, AkitaError::InvalidSetup(_)));

    let oversized_batch = MAX_VERIFIER_SETUP_SCHEDULE_SCANS / 32 + 1;
    let work_err = proof_optimized_max_setup_matrix_size::<fp128::D64OneHot>(32, oversized_batch)
        .expect_err("oversized setup schedule scan must fail closed");
    assert!(matches!(work_err, AkitaError::InvalidSetup(_)));

    let grouped_work_err = proof_optimized_max_setup_matrix_size::<fp128::D64OneHot>(32, 512)
        .expect_err("grouped layouts must count toward the setup scan bound");
    assert!(matches!(grouped_work_err, AkitaError::InvalidSetup(_)));
}
