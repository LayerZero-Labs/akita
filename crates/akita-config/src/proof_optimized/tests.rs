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
use akita_types::SisModulusFamily;

#[cfg(feature = "schedules-default")]
const MAX_I8_LOG_BASIS: u32 = 6;
#[cfg(feature = "schedules-default")]
const RAW_I8_RHS_MAX_ABS: u64 = 128;
#[test]
fn setup_level_params_from_runtime_schedule_excludes_terminal_direct() {
    // Terminal-direct steps ship the cleartext witness without
    // committing, so they have no `LevelParams` of their own and
    // must not contribute to the FS-bound `setup_levels`. Only
    // the preceding Fold steps (which do commit) appear.
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{CleartextWitnessShape, DirectStep, FoldStep, SisModulusFamily, Step};

    let sparse = SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![-1, 1],
    };
    let fold_lp = LevelParams::params_only(SisModulusFamily::Q128, 64, 3, 1, 1, 1, sparse);

    let steps = vec![
        Step::Fold(FoldStep {
            params: fold_lp.clone(),
            current_w_len: 1 << 8,
            next_w_len: 1 << 4,
            level_bytes: 0,
        }),
        Step::Direct(DirectStep {
            current_w_len: 1 << 4,
            witness_shape: CleartextWitnessShape::FieldElements(16),
            direct_bytes: 0,
            params: None,
        }),
    ];

    let setup_levels = setup_level_params_from_runtime_schedule(&steps);
    assert_eq!(
        setup_levels,
        vec![fold_lp],
        "terminal Direct.params is None and must not feed setup_levels; see DirectStep::params"
    );
}

#[test]
fn uncommittable_root_direct_schedule_yields_empty_setup_levels_and_loud_get_params_error() {
    // Documents the deliberate asymmetry between
    // `setup_level_params_from_runtime_schedule` (silently skips
    // root-direct schedules with `params: None`) and
    // `Cfg::get_params_for_batched_commitment` (rejects the same
    // schedule with a documented `InvalidSetup` message). The
    // contract is described on `DirectStep::params` and the
    // materializer comment that branches on it; this test locks
    // it in so neither side drifts.
    use akita_types::{CleartextWitnessShape, DirectStep, Schedule, Step};

    let uncommittable = Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len: 1 << 10,
            witness_shape: CleartextWitnessShape::FieldElements(1 << 10),
            direct_bytes: 0,
            params: None,
        })],
        total_bytes: 0,
    };

    let bound = setup_level_params_from_runtime_schedule(&uncommittable.steps);
    assert!(
        bound.is_empty(),
        "uncommittable root-direct schedule must produce no setup levels; \
         see DirectStep::params"
    );

    // `get_params_for_batched_commitment` reads the root commit off the
    // runtime schedule's first step: a root-direct `params: None` is the
    // uncommittable edge and must be rejected loudly (rather than silently
    // dropped, as the setup-levels reader above does). Drive the real trait
    // method through a config whose `runtime_schedule` yields exactly this
    // uncommittable schedule, and assert the documented `InvalidSetup`.
    #[derive(Clone)]
    struct UncommittableRootDirectCfg;
    impl CommitmentConfig for UncommittableRootDirectCfg {
        type Field = akita_field::Fp32<251>;
        type ExtField = akita_field::Fp32<251>;
        const D: usize = 8;
        fn decomposition() -> akita_types::DecompositionParams {
            akita_types::DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(8),
            }
        }
        fn ring_challenge_config(
            _d: usize,
        ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
            Ok(akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            })
        }
        fn sis_modulus_family() -> akita_types::SisModulusFamily {
            akita_types::SisModulusFamily::Q32
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
        // Inject the uncommittable root-direct schedule so the default
        // `get_params_for_batched_commitment` hits its rejection branch.
        fn runtime_schedule(_key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
            Ok(Schedule {
                steps: vec![Step::Direct(DirectStep {
                    current_w_len: 1 << 10,
                    witness_shape: CleartextWitnessShape::FieldElements(1 << 10),
                    direct_bytes: 0,
                    params: None,
                })],
                total_bytes: 0,
            })
        }
    }

    let opening_batch = OpeningClaimsLayout::new(10, 1).expect("singleton opening batch");
    let err = UncommittableRootDirectCfg::get_params_for_batched_commitment(&opening_batch)
        .expect_err("uncommittable root-direct must reject get_params_for_batched_commitment");
    assert!(
        err.to_string()
            .contains("root-direct schedule is missing commit params"),
        "unexpected error: {err}"
    );
}

#[test]
fn setup_matrix_envelope_does_not_add_conservative_layout() {
    use akita_types::{CleartextWitnessShape, DecompositionParams, DirectStep, Schedule, Step};

    #[derive(Clone)]
    struct GroupLayoutRejectCfg;

    impl CommitmentConfig for GroupLayoutRejectCfg {
        type Field = akita_field::Fp32<251>;
        type ExtField = akita_field::Fp32<251>;
        const D: usize = 8;

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: 3,
                log_commit_bound: 1,
                log_open_bound: Some(8),
            }
        }

        fn ring_challenge_config(
            _d: usize,
        ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
            Ok(akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            })
        }

        fn sis_modulus_family() -> akita_types::SisModulusFamily {
            akita_types::SisModulusFamily::Q32
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

        fn runtime_schedule(_key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
            Ok(Schedule {
                steps: vec![Step::Direct(DirectStep {
                    current_w_len: 1 << 8,
                    witness_shape: CleartextWitnessShape::FieldElements(1 << 8),
                    direct_bytes: 0,
                    params: None,
                })],
                total_bytes: 0,
            })
        }
    }

    let opening_batch = OpeningClaimsLayout::new(8, 1).expect("singleton opening batch");
    let envelope = setup_matrix_envelope_for_shape::<GroupLayoutRejectCfg>(&opening_batch)
        .expect("runtime setup envelope should use runtime schedule only")
        .expect("runtime schedule is supported");
    assert_eq!(envelope.max_setup_len, 1);
}

#[test]
fn fallback_root_direct_schedule_binds_real_opening_batch_commit_params() {
    // Locks in the fix for the descriptor-binding bug at
    // `akita_prover::protocol::core` and
    // `akita_verifier::protocol::core`: when the planner-selected
    // folded root cannot handle the opening shape, both sides build
    // a fallback root-direct schedule. That schedule's `params` are
    // hashed into the per-proof effective-schedule digest
    // (`PlanSection::from_schedule` -> `digest_effective_schedule`),
    // while the root-direct verification closure recomputes commitments
    // using `Cfg::get_params_for_batched_commitment(real_opening_batch)`. If
    // the fallback used a synthetic `same_point(num_vars, 1)`
    // singleton opening batch (the pre-fix behavior), the descriptor
    // would bind singleton-sized params while verification ran
    // against batched ones.
    use akita_types::{digest_effective_schedule, root_direct_schedule};
    type Cfg = fp128::D128Full;
    let real_opening_batch =
        OpeningClaimsLayout::new(30, 4).expect("batched same-point opening batch");
    let real_params =
        Cfg::get_params_for_batched_commitment(&real_opening_batch).expect("batched commit params");
    let singleton_opening_batch = OpeningClaimsLayout::new(30, 1).expect("singleton opening batch");
    let singleton_params = Cfg::get_params_for_batched_commitment(&singleton_opening_batch)
        .expect("singleton commit params");

    // Sanity: a non-singleton opening batch should resolve to a
    // different commit layout, otherwise the regression couldn't
    // manifest with this fixture.
    assert_ne!(
        real_params, singleton_params,
        "test fixture: pick an opening batch where batched and singleton params differ"
    );

    let real_schedule =
        root_direct_schedule(real_opening_batch.max_num_vars(), real_params.clone())
            .expect("fallback root-direct schedule");
    let bound_levels = setup_level_params_from_runtime_schedule(&real_schedule.steps);
    assert_eq!(
        bound_levels,
        vec![real_params],
        "fallback schedule must carry the real opening-batch params the verifier recomputes"
    );

    // The descriptor binds those params through the schedule digest: a
    // singleton-params fallback at the same `num_vars` must produce a
    // different preamble than the real batched-params fallback.
    let singleton_schedule =
        root_direct_schedule(real_opening_batch.max_num_vars(), singleton_params)
            .expect("singleton fallback root-direct schedule");
    assert_ne!(
        digest_effective_schedule(&real_schedule),
        digest_effective_schedule(&singleton_schedule),
        "schedule digest must distinguish batched vs singleton root-direct commit params"
    );
}

#[test]
fn setup_matrix_envelope_covers_grouped_batch_schedules() {
    let opening_batch = OpeningClaimsLayout::new(30, 4).expect("grouped same-point opening_batch");
    let grouped_same_point = setup_matrix_envelope_for_shape::<fp128::D128Full>(&opening_batch)
        .unwrap()
        .expect("grouped same-point shape should resolve to a setup envelope");

    let setup_envelope = proof_optimized_max_setup_matrix_size::<fp128::D128Full>(30, 4)
        .expect("setup envelope should cover generated grouped batch schedules");
    assert!(setup_envelope.max_setup_len >= grouped_same_point.max_setup_len);
}

fn expected_runtime_root_setup_len(lp: &LevelParams, opening_batch: &OpeningClaimsLayout) -> usize {
    let max_group_poly_count = opening_batch.num_total_polynomials();
    let d_width = lp.num_blocks * opening_batch.num_total_polynomials() * lp.num_digits_open;
    let t_cols_per_vector = lp.a_key.row_len() * lp.num_digits_open * lp.num_blocks;
    let b_width = max_group_poly_count * t_cols_per_vector;
    (lp.d_key.row_len() * d_width).max(lp.b_key.row_len() * b_width)
}

#[test]
fn setup_matrix_envelope_covers_batched_runtime_root_widths() {
    type Cfg = fp128::D128Full;
    let opening_batch = OpeningClaimsLayout::new(30, 4).expect("batched same-point opening_batch");
    let schedule = Cfg::get_params_for_prove(&opening_batch).expect("runtime schedule");
    let root_params = root_commit_params_from_schedule(&schedule)
        .unwrap()
        .expect("batched root schedule should carry commit params");
    let required = expected_runtime_root_setup_len(&root_params, &opening_batch);

    let runtime_envelope = matrix_envelope_for_schedule::<Cfg>(&schedule, &opening_batch).unwrap();
    assert!(runtime_envelope.max_setup_len >= required);

    let setup_envelope = proof_optimized_max_setup_matrix_size::<Cfg>(30, 4)
        .expect("setup envelope should cover generated batched root widths");
    assert!(setup_envelope.max_setup_len >= required);
}

#[test]
fn setup_matrix_envelope_covers_single_point_batch_root_widths() {
    use akita_types::root_direct_schedule;

    type Cfg = fp128::D128Full;
    let opening_batch = OpeningClaimsLayout::new(30, 4).expect("supported batched opening_batch");
    let root_params = Cfg::get_params_for_batched_commitment(&opening_batch)
        .expect("supported batched commit params");
    let schedule = root_direct_schedule(opening_batch.max_num_vars(), root_params.clone())
        .expect("synthetic direct schedule");
    let required = expected_runtime_root_setup_len(&root_params, &opening_batch);

    let runtime_envelope = matrix_envelope_for_schedule::<Cfg>(&schedule, &opening_batch).unwrap();
    assert!(runtime_envelope.max_setup_len >= required);
}

#[test]
fn setup_matrix_scan_uses_one_shared_opening_point() {
    let opening_batch =
        worst_case_grouped_opening_batch_for_shape(30, 4).expect("valid opening batch");
    assert_eq!(opening_batch.num_total_polynomials(), 4);
}

#[test]
fn setup_envelope_poly_counts_match_shipped_table_keys() {
    assert_eq!(super::setup_envelope_poly_counts(1), vec![1]);
    assert_eq!(super::setup_envelope_poly_counts(2), vec![1, 2]);
    assert_eq!(super::setup_envelope_poly_counts(4), vec![1, 4]);
}

#[test]
fn setup_envelope_endpoint_poly_scan_matches_exhaustive_scan() {
    fn exhaustive_envelope<Cfg: CommitmentConfig>(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> SetupMatrixEnvelope {
        let mut max_setup_len = 1usize;
        let poly_counts = super::setup_envelope_poly_counts(max_num_batched_polys);
        for num_vars in 1..=max_num_vars {
            for &num_polys in &poly_counts {
                let opening_batch =
                    worst_case_grouped_opening_batch_for_shape(num_vars, num_polys).unwrap();
                if let Ok(Some(envelope)) = setup_matrix_envelope_for_shape::<Cfg>(&opening_batch) {
                    max_setup_len = max_setup_len.max(envelope.max_setup_len);
                }
            }
        }
        let mut envelope = SetupMatrixEnvelope { max_setup_len };
        if Cfg::decomposition().log_commit_bound == 1 && !Cfg::TIERED_COMMITMENT {
            crate::conservative_commitment::inflate_setup_envelope_for_conservative_commitments::<
                Cfg,
            >(max_num_vars, max_num_batched_polys, &mut envelope)
            .expect("conservative setup envelope inflation");
            super::inflate_setup_envelope_for_precommitted_grouped_roots::<Cfg>(
                max_num_vars,
                max_num_batched_polys,
                &mut envelope,
            )
            .expect("precommitted grouped-root setup envelope inflation");
        }
        envelope
    }

    const MAX_NV: usize = 30;
    let exhaustive = exhaustive_envelope::<fp128::D64OneHot>(MAX_NV, 4);
    let endpoint = super::proof_optimized_max_setup_matrix_size::<fp128::D64OneHot>(MAX_NV, 4)
        .expect("endpoint poly scan");
    assert_eq!(
        exhaustive.max_setup_len, endpoint.max_setup_len,
        "D64OneHot nv<={MAX_NV}: endpoint scan must match exhaustive poly scan"
    );
}

#[test]
#[cfg(feature = "schedules-fp128-d64-onehot")]
fn proof_optimized_setup_includes_precommitted_grouped_root_catalog_entries() {
    type Cfg = fp128::D64OneHot;

    let catalog = Cfg::schedule_catalog().expect("D64 one-hot catalog");
    let entry = catalog
        .entries
        .iter()
        .find(|entry| {
            entry.final_group.num_vars() == 16
                && entry.final_group.num_polynomials() == 1
                && entry.precommitteds.len() == 2
                && entry
                    .precommitteds
                    .iter()
                    .all(|group| group.group.num_vars() == 8 && group.group.num_polynomials() == 1)
        })
        .expect("generated two-precommit grouped-root entry");
    let key = super::runtime_key_from_generated_entry(entry);
    let schedule = Cfg::runtime_schedule(key).expect("precommitted grouped-root schedule");
    let grouped_d_len =
        super::grouped_root_d_setup_len(&schedule).expect("precommitted grouped-root D footprint");

    let setup_envelope = super::proof_optimized_max_setup_matrix_size::<Cfg>(16, 1)
        .expect("setup envelope should include precommitted grouped-root catalog entries");

    assert!(
        setup_envelope.max_setup_len >= grouped_d_len,
        "setup envelope must cover generated precommitted grouped-root D footprints"
    );
}

#[test]
fn proof_optimized_setup_includes_conservative_precommit_inflation() {
    type Cfg = fp128::D64OneHot;

    let runtime_only = setup_matrix_envelope_for_shape::<Cfg>(
        &worst_case_grouped_opening_batch_for_shape(30, 4).expect("valid opening batch"),
    )
    .expect("runtime envelope")
    .expect("supported runtime schedule");

    let proof_optimized = super::proof_optimized_max_setup_matrix_size::<Cfg>(30, 4)
        .expect("proof-optimized setup envelope");

    assert!(
        proof_optimized.max_setup_len > runtime_only.max_setup_len,
        "proof-optimized setup sizing must include conservative precommit B-rank capacity"
    );
}

#[test]
#[ignore = "full envelope exhaustive cross-check is slow; run before envelope logic changes"]
fn setup_envelope_endpoint_poly_scan_full_manual() {
    fn exhaustive_envelope<Cfg: CommitmentConfig>(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> SetupMatrixEnvelope {
        let mut max_setup_len = 1usize;
        let poly_counts = super::setup_envelope_poly_counts(max_num_batched_polys);
        for num_vars in 1..=max_num_vars {
            for &num_polys in &poly_counts {
                let opening_batch =
                    worst_case_grouped_opening_batch_for_shape(num_vars, num_polys).unwrap();
                if let Ok(Some(envelope)) = setup_matrix_envelope_for_shape::<Cfg>(&opening_batch) {
                    max_setup_len = max_setup_len.max(envelope.max_setup_len);
                }
            }
        }
        let mut envelope = SetupMatrixEnvelope { max_setup_len };
        if Cfg::decomposition().log_commit_bound == 1 && !Cfg::TIERED_COMMITMENT {
            crate::conservative_commitment::inflate_setup_envelope_for_conservative_commitments::<
                Cfg,
            >(max_num_vars, max_num_batched_polys, &mut envelope)
            .expect("conservative setup envelope inflation");
            super::inflate_setup_envelope_for_precommitted_grouped_roots::<Cfg>(
                max_num_vars,
                max_num_batched_polys,
                &mut envelope,
            )
            .expect("precommitted grouped-root setup envelope inflation");
        }
        envelope
    }

    for max_nv in [16usize, 24, 30] {
        let exhaustive = exhaustive_envelope::<fp128::D64OneHot>(max_nv, 4);
        let endpoint = super::proof_optimized_max_setup_matrix_size::<fp128::D64OneHot>(max_nv, 4)
            .expect("endpoint poly scan");
        assert_eq!(
            exhaustive.max_setup_len, endpoint.max_setup_len,
            "D64OneHot nv<={max_nv}: endpoint scan must match exhaustive poly scan"
        );

        let exhaustive_full = exhaustive_envelope::<fp128::D64Full>(max_nv, 4);
        let endpoint_full =
            super::proof_optimized_max_setup_matrix_size::<fp128::D64Full>(max_nv, 4)
                .expect("endpoint poly scan");
        assert_eq!(
            exhaustive_full.max_setup_len, endpoint_full.max_setup_len,
            "D64Full nv<={max_nv}: endpoint scan must match exhaustive poly scan"
        );
    }
}

#[test]
fn presets_select_expected_sis_modulus_family() {
    assert_eq!(
        <fp128::D32Full as CommitmentConfig>::sis_modulus_family(),
        akita_types::SisModulusFamily::Q128
    );
    assert_eq!(
        <fp32::D64Full as CommitmentConfig>::sis_modulus_family(),
        akita_types::SisModulusFamily::Q32
    );
    assert_eq!(
        <fp64::D64Full as CommitmentConfig>::sis_modulus_family(),
        akita_types::SisModulusFamily::Q64
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
        // The last fold in a fold-then-direct schedule is the terminal
        // recursive fold and ships its W in cleartext under
        // MRowLayout::Terminal (drops the D-block from the per-row `r`
        // quotients), so its `next_w_len` is smaller than what the
        // intermediate-layout helper would report.
        let is_terminal_fold = idx + 1 == num_fold_levels;
        let layout = if is_terminal_fold {
            akita_types::MRowLayout::WithoutDBlock
        } else {
            akita_types::MRowLayout::WithDBlock
        };
        // Root-level batched witnesses fan out over the key's polynomial
        // count; recursive levels collapse back to singleton-by-construction.
        let (num_polynomials, num_public_rows) = if idx == 0 {
            (key.num_polynomials(), 1)
        } else {
            (1, 1)
        };
        let runtime_next_w_len = akita_types::w_ring_element_count_with_counts_for_layout::<
            Cfg::Field,
        >(&fold.params, num_polynomials, num_public_rows, layout)
        .expect("valid planned witness")
            * fold.params.ring_dimension;
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
    match Cfg::sis_modulus_family() {
        SisModulusFamily::Q32 => ("Q32/2xi32", 1_152_837_945_367_908_353),
        SisModulusFamily::Q64 => ("Q64/3xi32", 1_237_793_655_097_897_487_951_597_569),
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
        (1..=MAX_I8_LOG_BASIS).contains(&level.log_basis),
        "generated schedule uses log_basis={} outside the i8 kernel contract for {} key={key:?}",
        level.log_basis,
        std::any::type_name::<Cfg>()
    );
    let balanced_digit_bound = 1u64 << (level.log_basis - 1);
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
        let levels = setup_level_params_from_runtime_schedule(&schedule.steps);
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
            root_lp.a_key.row_len() * root_lp.num_digits_open * root_lp.num_blocks;
        let singleton_d_width = root_lp.num_digits_open * root_lp.num_blocks;
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

    let akita_types::CleartextWitnessShape::SegmentTyped(ref shape) =
        *akita_types::schedule_terminal_direct_witness_shape(&schedule)
            .expect("4x30 onehot schedule should end in a direct step")
    else {
        panic!("4x30 onehot schedule should end in segment-typed witness");
    };
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
fn tight_block_len_is_no_larger_than_pow2() {
    for num_vars in [14, 20, 30] {
        let schedule = fp128::D64Full::runtime_schedule(AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(num_vars),
        ))
        .expect("planner should succeed");
        for (level_idx, fold) in schedule.fold_steps().enumerate() {
            let lp = &fold.params;
            let pow2_block = 1usize << lp.m_vars;
            assert!(
                lp.block_len <= pow2_block,
                "block_len {} should be <= 2^m_vars {} at level {level_idx} (num_vars={num_vars})",
                lp.block_len,
                pow2_block,
            );
            if level_idx > 0 {
                let num_ring = fold.current_w_len / lp.ring_dimension;
                let expected_tight = num_ring.div_ceil(lp.num_blocks);
                assert_eq!(
                    lp.block_len, expected_tight,
                    "recursive level {level_idx} should use tight block_len = ceil({num_ring} / {})",
                    lp.num_blocks
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
// `Uniform { weight: 8, [-1, 1] }`, which has only ~31 bits at D=32).

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
/// `BoundedL1Norm` is constructed so its space exceeds `2^128` (proven in
/// `akita_challenges::sampler::bounded_l1_support`), so it is reported as
/// infinite here rather than re-running the bounded-L1 DP.
fn challenge_support_bits(cfg: &SparseChallengeConfig, d: usize) -> f64 {
    match cfg {
        SparseChallengeConfig::Uniform {
            weight,
            nonzero_coeffs,
        } => log2_binomial(d, *weight) + (*weight as f64) * (nonzero_coeffs.len() as f64).log2(),
        SparseChallengeConfig::ExactShell {
            count_mag1,
            count_mag2,
            ..
        } => {
            let support = count_mag1 + count_mag2;
            log2_binomial(d, support) + log2_binomial(support, *count_mag1) + (support as f64)
        }
        SparseChallengeConfig::BoundedL1Norm => f64::INFINITY,
    }
}

#[test]
fn proof_optimized_ring_challenge_policy_pins_secure_families() {
    // (d, expected family, (l1, linf)). Each family must clear 128 bits of
    // Fiat-Shamir support; the (l1, linf) pin guards the folded-witness norm
    // the schedules are generated against.
    let cases = [
        (
            32usize,
            SparseChallengeConfig::BoundedL1Norm,
            (121usize, 8u32),
        ),
        (
            64,
            SparseChallengeConfig::ExactShell {
                count_mag1: akita_challenges::D64_PRODUCTION_EXACT_SHELL_MAG1,
                count_mag2: akita_challenges::D64_PRODUCTION_EXACT_SHELL_MAG2,
            },
            (51, 2),
        ),
        (
            128,
            SparseChallengeConfig::Uniform {
                weight: 31,
                nonzero_coeffs: vec![-1, 1],
            },
            (31, 1),
        ),
        (
            256,
            SparseChallengeConfig::Uniform {
                weight: 23,
                nonzero_coeffs: vec![-1, 1],
            },
            (23, 1),
        ),
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

    // `BoundedL1Norm` is only valid at `D = 32`; confirm the policy wires it to
    // the one degree its sampler accepts.
    proof_optimized_ring_challenge_config(32)
        .unwrap()
        .validate::<32>()
        .unwrap();

    // Ring degrees no preset uses must be rejected, not silently defaulted.
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

    assert_preset_uses_shared_ring_challenge::<fp64::D32Full>();
    assert_preset_uses_shared_ring_challenge::<fp64::D32OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp64::D64Full>();
    assert_preset_uses_shared_ring_challenge::<fp64::D64OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp64::D128Full>();
    assert_preset_uses_shared_ring_challenge::<fp64::D128OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp64::D256Full>();
    assert_preset_uses_shared_ring_challenge::<fp64::D256OneHot>();

    assert_preset_uses_shared_ring_challenge::<fp128::D32Full>();
    assert_preset_uses_shared_ring_challenge::<fp128::D32OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp128::D64Full>();
    assert_preset_uses_shared_ring_challenge::<fp128::D64OneHot>();
    assert_preset_uses_shared_ring_challenge::<fp128::D128Full>();
    assert_preset_uses_shared_ring_challenge::<fp128::D128OneHot>();

    // Hand-written (non-macro) preset: guards that the bespoke impl still
    // routes through the shared policy.
    assert_preset_uses_shared_ring_challenge::<crate::tensor_verifier::fp128::D64OneHotTensor>();
}

#[test]
fn tensor_onehot_preset_keeps_d64_onehot_chunk_size() {
    assert_eq!(
        crate::tensor_verifier::fp128::D64OneHotTensor::onehot_chunk_size(),
        fp128::D64OneHot::onehot_chunk_size(),
        "tensor verifier preset must preserve the D64 one-hot witness sparsity envelope"
    );
}
