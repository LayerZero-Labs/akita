use super::*;
#[cfg(not(feature = "zk"))]
use akita_types::generated::{
    fp128_d32_full_table, fp128_d32_onehot_table, fp128_d64_full_table, fp128_d64_onehot_table,
    fp16_d32_full_table, fp16_d32_onehot_table, fp16_d64_full_table, fp16_d64_onehot_table,
    fp32_d32_onehot_table, fp32_d32_table, fp32_d64_onehot_table, fp32_d64_table,
    fp64_d32_onehot_table, fp64_d32_table, fp64_d64_onehot_table, fp64_d64_table,
    GeneratedScheduleTable,
};
use akita_types::layout::digit_math::optimal_m_r_split;
use akita_types::level_layout_from_params;
use akita_types::planned_w_ring_element_count;
use akita_types::AkitaScheduleInputs;
use akita_types::DecompositionParams;

#[test]
fn setup_level_params_from_runtime_schedule_excludes_terminal_direct() {
    // Terminal-direct steps ship the cleartext witness without
    // committing, so they have no `LevelParams` of their own and
    // must not contribute to the FS-bound `setup_levels`. Only
    // the preceding Fold steps (which do commit) appear.
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{DirectStep, DirectWitnessShape, FoldStep, SisModulusFamily, Step};

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
            witness_shape: DirectWitnessShape::PackedDigits((16, 3)),
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
    use akita_types::{DirectStep, DirectWitnessShape, Schedule, Step};

    let uncommittable = Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len: 1 << 10,
            witness_shape: DirectWitnessShape::FieldElements(1 << 10),
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

    // The trait default `get_params_for_batched_commitment` resolves
    // the compact entry and expands its root-commit step. Construct a
    // tiny stub Cfg that resolves to a root-direct entry whose terminal
    // `Direct` carries `commit: None` (the uncommittable edge), so we
    // exercise the loud-rejection branch in `root_commit_params`.
    use akita_types::generated::{
        GeneratedDirectStep, GeneratedScheduleKey, GeneratedScheduleTableEntry, GeneratedStep,
    };
    static UNCOMMITTABLE_STEPS: [GeneratedStep; 1] =
        [GeneratedStep::Direct(GeneratedDirectStep { commit: None })];
    static UNCOMMITTABLE_ENTRY: GeneratedScheduleTableEntry = GeneratedScheduleTableEntry {
        key: GeneratedScheduleKey {
            num_vars: 10,
            num_commitment_groups: 1,
            num_t_vectors: 1,
            num_w_vectors: 1,
            num_z_vectors: 1,
        },
        steps: &UNCOMMITTABLE_STEPS,
    };
    #[derive(Clone)]
    struct UncommittableRootDirectCfg;
    impl CommitmentConfig for UncommittableRootDirectCfg {
        type Field = akita_field::Fp32<251>;
        type ClaimField = akita_field::Fp32<251>;
        type ChallengeField = akita_field::Fp32<251>;
        const D: usize = 8;
        fn decomposition() -> akita_types::DecompositionParams {
            akita_types::DecompositionParams {
                log_basis: 3,
                log_commit_bound: 8,
                log_open_bound: Some(8),
            }
        }
        fn stage1_challenge_config(
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
        fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
            None
        }
        fn schedule_plan(
            _key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
            Ok(None)
        }
        fn max_setup_matrix_size(
            _max_num_vars: usize,
            _max_num_batched_polys: usize,
            _max_num_points: usize,
        ) -> Result<SetupMatrixEnvelope, AkitaError> {
            Ok(SetupMatrixEnvelope {
                max_setup_len: 1,
                #[cfg(feature = "zk")]
                max_zk_b_len: 1,
                #[cfg(feature = "zk")]
                max_zk_d_len: 1,
            })
        }
        fn basis_range() -> (u32, u32) {
            (3, 3)
        }
        fn resolve_schedule(
            _key: AkitaScheduleLookupKey,
        ) -> Result<Option<&'static GeneratedScheduleTableEntry>, AkitaError> {
            Ok(Some(&UNCOMMITTABLE_ENTRY))
        }
    }

    let incidence = ClaimIncidenceSummary::same_point(10, 1).expect("singleton");
    let err = UncommittableRootDirectCfg::get_params_for_batched_commitment(&incidence)
        .expect_err("uncommittable root-direct must reject get_params_for_batched_commitment");
    assert!(
        err.to_string()
            .contains("root-direct schedule is missing commit params"),
        "unexpected error: {err}"
    );
}

#[test]
#[cfg(not(feature = "zk"))]
fn fallback_root_direct_schedule_binds_real_incidence_commit_params() {
    // Locks in the fix for the descriptor-binding bug at
    // `akita_prover::protocol::flow` and
    // `akita_verifier::protocol::batched`: when the planner-selected
    // folded root cannot handle the opening shape, both sides build
    // a fallback root-direct schedule. That schedule's
    // `params` get hashed into
    // `SetupSection::level_params_digest` via
    // `setup_level_params_from_runtime_schedule`, while the
    // root-direct verification closure recomputes commitments using
    // `Cfg::get_params_for_batched_commitment(real_incidence)`. If
    // the fallback used a synthetic `same_point(num_vars, 1)`
    // singleton incidence (the pre-fix behavior), the descriptor
    // would bind singleton-sized params while verification ran
    // against batched ones.
    use akita_types::root_direct_schedule;
    type Cfg = fp128::D32Full;
    let real_incidence =
        ClaimIncidenceSummary::same_point(30, 4).expect("batched same-point incidence");
    let real_params =
        Cfg::get_params_for_batched_commitment(&real_incidence).expect("batched commit params");
    let singleton_incidence =
        ClaimIncidenceSummary::same_point(30, 1).expect("singleton incidence");
    let singleton_params = Cfg::get_params_for_batched_commitment(&singleton_incidence)
        .expect("singleton commit params");

    // Sanity: a non-singleton incidence should resolve to a
    // different commit layout, otherwise the regression couldn't
    // manifest with this fixture.
    assert_ne!(
        real_params, singleton_params,
        "test fixture: pick an incidence where batched and singleton params differ"
    );

    let schedule = root_direct_schedule(real_incidence.num_vars(), real_params.clone())
        .expect("fallback root-direct schedule");
    let bound_levels = setup_level_params_from_runtime_schedule(&schedule.steps);
    assert_eq!(
        bound_levels,
        vec![real_params],
        "fallback schedule must bind the real-incidence params the verifier recomputes"
    );
}

#[test]
fn setup_matrix_envelope_covers_grouped_batch_schedules() {
    let incidence = ClaimIncidenceSummary::same_point(30, 4).expect("grouped same-point incidence");
    let grouped_same_point = setup_matrix_envelope_for_shape::<fp128::D32Full>(&incidence)
        .unwrap()
        .expect("D32 full table must contain the grouped same-point schedule");

    let setup_envelope = proof_optimized_max_setup_matrix_size::<fp128::D32Full>(30, 4, 1)
        .expect("setup envelope should cover generated grouped batch schedules");
    assert!(setup_envelope.max_setup_len >= grouped_same_point.max_setup_len);
}

fn expected_runtime_root_setup_len(lp: &LevelParams, incidence: &ClaimIncidenceSummary) -> usize {
    let max_group_poly_count = incidence
        .num_polys_per_point()
        .iter()
        .copied()
        .max()
        .expect("nonempty incidence");
    let d_width = lp.num_blocks * incidence.num_claims() * lp.num_digits_open;
    let t_cols_per_vector = lp.a_key.row_len() * lp.num_digits_open * lp.num_blocks;
    let b_width = max_group_poly_count * t_cols_per_vector;
    (lp.d_key.row_len() * d_width).max(lp.b_key.row_len() * b_width)
}

#[test]
fn setup_matrix_envelope_covers_batched_runtime_root_widths() {
    type Cfg = fp128::D32Full;
    let incidence = ClaimIncidenceSummary::same_point(30, 4).expect("batched same-point incidence");
    let schedule = Cfg::get_params_for_prove(&incidence).expect("runtime schedule");
    let root_params = root_commit_params_from_schedule(&schedule)
        .unwrap()
        .expect("folded or direct root params");
    let required = expected_runtime_root_setup_len(&root_params, &incidence);

    let runtime_envelope = matrix_envelope_for_schedule::<Cfg>(&schedule, &incidence).unwrap();
    assert!(runtime_envelope.max_setup_len >= required);

    let setup_envelope = proof_optimized_max_setup_matrix_size::<Cfg>(30, 4, 1)
        .expect("setup envelope should cover generated batched root widths");
    assert!(setup_envelope.max_setup_len >= required);
}

#[test]
fn setup_matrix_envelope_covers_skewed_multipoint_root_widths() {
    use akita_types::root_direct_schedule;

    type Cfg = fp128::D32Full;
    let incidence =
        ClaimIncidenceSummary::from_point_polys(30, vec![3, 1]).expect("skewed incidence");
    let commit_incidence =
        ClaimIncidenceSummary::same_point(30, 4).expect("supported batched incidence");
    let root_params = Cfg::get_params_for_batched_commitment(&commit_incidence)
        .expect("supported batched commit params");
    let schedule = root_direct_schedule(incidence.num_vars(), root_params.clone())
        .expect("synthetic direct schedule");
    let required = expected_runtime_root_setup_len(&root_params, &incidence);

    let runtime_envelope = matrix_envelope_for_schedule::<Cfg>(&schedule, &incidence).unwrap();
    assert!(runtime_envelope.max_setup_len >= required);
}

#[test]
fn setup_matrix_scan_uses_worst_case_grouping_for_aggregate_shape() {
    let incidence =
        worst_case_grouped_incidence_for_shape(30, 4, 2).expect("valid aggregate incidence");
    assert_eq!(incidence.num_polys_per_point(), &[3, 1]);
}

#[test]
#[cfg(feature = "zk")]
fn setup_matrix_envelope_excludes_zk_blinding_tail_columns() {
    use akita_challenges::SparseChallengeConfig;
    use akita_types::SisModulusFamily;

    type Cfg = fp128::D32Full;
    let sparse = SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![-1, 1],
    };
    let lp = LevelParams::params_only(SisModulusFamily::Q128, Cfg::D, 5, 2, 3, 5, sparse)
        .with_decomp(4, 3, 2, 6, 3)
        .unwrap();

    let mut got = 1usize;
    accumulate_matrix_envelope_for_level::<Cfg>(&lp, &mut got).unwrap();

    let expected = (lp.a_key.row_len() * lp.inner_width())
        .max(lp.b_key.row_len() * lp.outer_width())
        .max(lp.d_key.row_len() * lp.d_matrix_width());
    assert_eq!(got, expected);

    let b_tail = akita_types::zk::blinding_column_count::<<Cfg as CommitmentConfig>::Field>(
        lp.b_key.row_len(),
        lp.ring_dimension,
        lp.log_basis,
    );
    let d_tail = akita_types::zk::blinding_column_count::<<Cfg as CommitmentConfig>::Field>(
        lp.d_key.row_len(),
        lp.ring_dimension,
        lp.log_basis,
    );
    let old_tail_inflated = (lp.a_key.row_len() * lp.inner_width())
        .max(lp.b_key.row_len() * (lp.outer_width() + b_tail))
        .max(lp.d_key.row_len() * (lp.d_matrix_width() + d_tail));
    assert!(
        old_tail_inflated > expected,
        "test fixture must catch accidental ZK tail columns in setup envelope"
    );
}

#[test]
#[cfg(feature = "zk")]
fn setup_matrix_envelope_covers_zk_hiding_blinding_columns() {
    type Cfg = fp32::D32Full;
    let incidence = ClaimIncidenceSummary::same_point(26, 1).expect("singleton incidence");
    let schedule = Cfg::get_params_for_prove(&incidence).expect("runtime schedule");
    let root_params = root_commit_params_from_schedule(&schedule)
        .unwrap()
        .expect("folded or direct root params");
    let hiding_len = zk_hiding_witness_len::<Cfg>(&schedule, &incidence).unwrap();
    let num_ring = hiding_len.div_ceil(Cfg::D).max(1).next_power_of_two();
    let hiding_params = root_params
        .with_decomp(
            num_ring.trailing_zeros() as usize,
            0,
            root_params.num_digits_commit,
            root_params.num_digits_open,
            num_ring,
        )
        .unwrap();
    let blinding_cols =
        akita_types::zk::blinding_digit_plane_count::<<Cfg as CommitmentConfig>::Field>(
            hiding_params.b_key.row_len(),
            hiding_params.ring_dimension,
            hiding_params.log_basis,
        );
    let required = hiding_params.b_key.row_len() * blinding_cols;

    let runtime_envelope = matrix_envelope_for_schedule::<Cfg>(&schedule, &incidence).unwrap();
    assert!(runtime_envelope.max_zk_b_len >= required);

    let setup_envelope = proof_optimized_max_setup_matrix_size::<Cfg>(26, 1, 1).unwrap();
    assert!(setup_envelope.max_zk_b_len >= required);
}

#[test]
#[cfg(not(feature = "zk"))]
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
    assert_eq!(
        <fp16::D64Full as CommitmentConfig>::sis_modulus_family(),
        akita_types::SisModulusFamily::Q16
    );
}

#[test]
#[cfg(not(feature = "zk"))]
fn fp16_generated_schedule_tables_are_wired() {
    let onehot_key = AkitaScheduleLookupKey::singleton(32);
    let onehot_plan = <fp16::D32OneHot as crate::CommitmentConfig>::schedule_plan(onehot_key)
        .unwrap()
        .expect("fp16 D32 onehot nv32 schedule should be generated");
    assert!(!onehot_plan.steps.is_empty());

    let dense_key = AkitaScheduleLookupKey::singleton(27);
    let dense_plan = <fp16::D32Full as crate::CommitmentConfig>::schedule_plan(dense_key)
        .unwrap()
        .expect("fp16 D32 full nv27 schedule should be generated");
    assert!(!dense_plan.steps.is_empty());
}

#[test]
#[cfg(not(feature = "zk"))]
fn fp32_d32_generated_schedule_tables_are_wired() {
    let onehot_key = AkitaScheduleLookupKey::singleton(32);
    let onehot_plan = <fp32::D32OneHot as crate::CommitmentConfig>::schedule_plan(onehot_key)
        .unwrap()
        .expect("fp32 D32 onehot nv32 schedule should be generated");
    assert!(!onehot_plan.steps.is_empty());

    let dense_key = AkitaScheduleLookupKey::singleton(26);
    let dense_plan = <fp32::D32Full as crate::CommitmentConfig>::schedule_plan(dense_key)
        .unwrap()
        .expect("fp32 D32 full nv26 schedule should be generated");
    assert!(!dense_plan.steps.is_empty());
}

// ----- migrated from former `schedule_policy::tests` -------------------

#[cfg(not(feature = "zk"))]
fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(num_vars: usize) {
    assert_plan_matches_runtime_w_sizes_for_key::<Cfg>(AkitaScheduleLookupKey::singleton(num_vars));
}

#[cfg(not(feature = "zk"))]
fn assert_plan_matches_runtime_w_sizes_for_key<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) {
    let plan = Cfg::schedule_plan(key)
        .expect("planner should succeed")
        .expect("config should provide a planner");
    let num_fold_levels = plan.num_fold_levels();
    for (idx, level) in plan.fold_levels().enumerate() {
        // The last fold in a fold-then-direct schedule is the terminal
        // recursive fold and ships its W in cleartext under
        // MRowLayout::Terminal (drops the D-block from the per-row `r`
        // quotients), so its `next_w_len` is smaller than what the
        // intermediate-layout helper would report.
        let is_terminal_fold = idx + 1 == num_fold_levels;
        let layout = if is_terminal_fold {
            akita_types::MRowLayout::Terminal
        } else {
            akita_types::MRowLayout::Intermediate
        };
        // Root-level batched witnesses fan out over the key's vector
        // counts; recursive levels collapse back to singleton-by-construction.
        let (num_points, num_t_vectors, num_w_vectors, num_public_rows) = if idx == 0 {
            (
                key.num_points,
                key.num_t_vectors,
                key.num_w_vectors,
                key.num_z_vectors,
            )
        } else {
            (1, 1, 1, 1)
        };
        let runtime_next_w_len =
            akita_types::w_ring_element_count_with_counts_for_layout::<Cfg::Field>(
                &level.lp,
                num_points,
                num_t_vectors,
                num_w_vectors,
                num_public_rows,
                layout,
            )
            .expect("valid planned witness")
                * level.lp.ring_dimension;
        assert_eq!(
            runtime_next_w_len, level.next_inputs.current_w_len,
            "planner/runtime next_w_len mismatch at level {} for key={key:?}",
            level.inputs.level
        );
    }
}

#[cfg(not(feature = "zk"))]
fn assert_every_table_entry_materializes<Cfg: CommitmentConfig>(table: GeneratedScheduleTable) {
    for entry in table.entries {
        let key = AkitaScheduleLookupKey::new_with_points(
            entry.key.num_vars,
            entry.key.num_commitment_groups,
            entry.key.num_t_vectors,
            entry.key.num_w_vectors,
            entry.key.num_z_vectors,
        );
        Cfg::schedule_plan(key)
            .expect("config schedule should succeed")
            .expect("config should provide a generated schedule");
    }
}

#[cfg(not(feature = "zk"))]
fn assert_generated_batched_roots_are_scaled<Cfg: CommitmentConfig>(table: GeneratedScheduleTable) {
    let mut checked_folded_entry = false;
    for entry in table
        .entries
        .iter()
        .filter(|entry| entry.key.num_t_vectors > 1)
    {
        let key = AkitaScheduleLookupKey::new_with_points(
            entry.key.num_vars,
            entry.key.num_commitment_groups,
            entry.key.num_t_vectors,
            entry.key.num_w_vectors,
            entry.key.num_z_vectors,
        );
        let generated = Cfg::schedule_plan(key)
            .expect("config schedule should succeed")
            .expect("config should provide a generated schedule");
        let Some(root) = generated.fold_levels().next() else {
            continue;
        };
        checked_folded_entry = true;
        let singleton_outer_width =
            root.lp.a_key.row_len() * root.lp.num_digits_open * root.lp.num_blocks;
        let singleton_d_width = root.lp.num_digits_open * root.lp.num_blocks;
        assert_eq!(
            root.lp.outer_width(),
            singleton_outer_width * entry.key.num_t_vectors,
            "generated batched root B width should be claim-scaled for key={key:?}"
        );
        assert_eq!(
            root.lp.d_matrix_width(),
            singleton_d_width * entry.key.num_t_vectors,
            "generated batched root D width should be claim-scaled for key={key:?}"
        );
    }
    assert!(
        checked_folded_entry,
        "generated table should include at least one folded batched entry"
    );
}

#[cfg(not(feature = "zk"))]
fn assert_exact_root_fold_matches_runtime_root_plan<Cfg: CommitmentConfig, const D: usize>(
    num_vars: usize,
) {
    let key = AkitaScheduleLookupKey::singleton(num_vars);
    let plan = Cfg::schedule_plan(key)
        .expect("config schedule should succeed")
        .expect("config should provide an exact schedule");
    let planned_root = akita_types::exact_planned_level_execution(
        &plan,
        AkitaScheduleInputs {
            num_vars,
            level: 0,
            current_w_len: 1usize.checked_shl(num_vars as u32).unwrap_or(0),
        },
        plan.fold_levels()
            .next()
            .expect("exact schedule should begin with a fold")
            .lp
            .log_basis,
        Cfg::stage1_challenge_config,
    )
    .expect("exact plan should resolve the root fold")
    .expect("exact plan should contain a matching root fold");
    let incidence = ClaimIncidenceSummary::same_point(num_vars, 1).expect("singleton incidence");
    let runtime_root =
        Cfg::get_params_for_prove(&incidence).expect("runtime root plan should succeed");
    let Some(akita_types::Step::Fold(runtime_root_step)) = runtime_root.steps.first() else {
        panic!("runtime root schedule should start with a fold");
    };
    assert_eq!(
        planned_root.level.inputs.current_w_len,
        runtime_root_step.current_w_len,
        "planned/runtime root current_w_len mismatch for {} at num_vars={num_vars}",
        std::any::type_name::<Cfg>()
    );
    assert_eq!(
        planned_root.level.lp,
        runtime_root_step.params,
        "planned/runtime root lp mismatch for {} at num_vars={num_vars}",
        std::any::type_name::<Cfg>()
    );
    assert_eq!(
        planned_root.level.next_inputs.current_w_len,
        runtime_root_step.next_w_len,
        "planned/runtime next_w_len mismatch for {} at num_vars={num_vars}",
        std::any::type_name::<Cfg>()
    );
}

/// Parity guard: the new compact-entry expansion path
/// (`resolve_schedule` + `expand_fold_level` / `root_commit_params`)
/// reproduces the old `akita-derive` materializer's per-level
/// `LevelParams` and root commit exactly. Locks the refactor before the
/// materializer is deleted and consumers are switched over.
#[cfg(not(feature = "zk"))]
fn assert_new_expand_matches_old_plan<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) {
    use akita_types::generated::GeneratedStep;

    let plan = Cfg::schedule_plan(key)
        .expect("old materializer should succeed")
        .expect("config should provide a plan");
    let entry = Cfg::resolve_schedule(key)
        .expect("resolve_schedule should succeed")
        .expect("config should resolve an entry");

    let fold_steps: Vec<_> = entry
        .steps
        .iter()
        .filter_map(|step| match step {
            GeneratedStep::Fold(fold) => Some(*fold),
            GeneratedStep::Direct(_) => None,
        })
        .collect();
    let old_levels: Vec<_> = plan.fold_levels().collect();
    assert_eq!(
        fold_steps.len(),
        old_levels.len(),
        "fold-level count mismatch for {} key={key:?}",
        std::any::type_name::<Cfg>()
    );

    let field_bits = Cfg::decomposition().field_bits();
    let batched = crate::root_batched_dims(key, field_bits);
    for (idx, level) in old_levels.iter().enumerate() {
        let batched_root = if idx == 0 { batched } else { None };
        let new_lp = Cfg::expand_fold_level(
            &fold_steps[idx],
            key.num_vars,
            idx,
            level.inputs.current_w_len,
            batched_root,
        )
        .expect("new expansion should succeed");
        assert_eq!(
            new_lp,
            level.lp,
            "new-vs-old level lp mismatch for {} key={key:?} level={idx}",
            std::any::type_name::<Cfg>()
        );
    }

    // Root-direct entries have no fold levels; compare the new root
    // commit against the old materializer's commit params. The B/D
    // matrices are vestigial for a root-direct commit (the witness ships
    // in the clear, no relation fold runs), and the old materializer
    // sized the B width against the per-`r` intermediate A-rank rather
    // than the committed A-rank — a latent inconsistency. The new
    // self-consistent expansion fixes that, so we compare the functional
    // fields (ranks, A column width, block split, digit counts) and
    // tolerate the corrected B/D column widths.
    if old_levels.is_empty() && key.num_t_vectors == 1 && key.num_w_vectors == 1 {
        let incidence =
            ClaimIncidenceSummary::same_point(key.num_vars, 1).expect("singleton incidence");
        if let Ok(old_commit) = Cfg::get_params_for_batched_commitment(&incidence) {
            let new_commit = Cfg::root_commit_params(key).expect("new root commit");
            assert_eq!(
                new_commit.a_key,
                old_commit.a_key,
                "root-direct A-key mismatch for {} key={key:?}",
                std::any::type_name::<Cfg>()
            );
            assert_eq!(
                (
                    new_commit.ring_dimension,
                    new_commit.log_basis,
                    new_commit.b_key.row_len(),
                    new_commit.d_key.row_len(),
                    new_commit.num_blocks,
                    new_commit.block_len,
                    new_commit.m_vars,
                    new_commit.r_vars,
                    new_commit.num_digits_commit,
                    new_commit.num_digits_open,
                ),
                (
                    old_commit.ring_dimension,
                    old_commit.log_basis,
                    old_commit.b_key.row_len(),
                    old_commit.d_key.row_len(),
                    old_commit.num_blocks,
                    old_commit.block_len,
                    old_commit.m_vars,
                    old_commit.r_vars,
                    old_commit.num_digits_commit,
                    old_commit.num_digits_open,
                ),
                "root-direct commit functional fields mismatch for {} key={key:?}",
                std::any::type_name::<Cfg>()
            );
        }
    }
}

#[test]
#[cfg(not(feature = "zk"))]
fn new_expand_matches_old_plan_across_fp128_presets() {
    for num_vars in 6..=30 {
        let key = AkitaScheduleLookupKey::singleton(num_vars);
        assert_new_expand_matches_old_plan::<fp128::D32Full>(key);
        assert_new_expand_matches_old_plan::<fp128::D32OneHot>(key);
        assert_new_expand_matches_old_plan::<fp128::D64Full>(key);
        assert_new_expand_matches_old_plan::<fp128::D64OneHot>(key);
    }
}

#[test]
#[cfg(not(feature = "zk"))]
fn new_expand_matches_old_plan_for_batched_root() {
    let key = AkitaScheduleLookupKey::new(30, 4, 4, 1);
    assert_new_expand_matches_old_plan::<fp128::D64OneHot>(key);
}

#[test]
#[cfg(not(feature = "zk"))]
fn generated_fp128_schedule_tables_match_cfg_schedule() {
    assert_every_table_entry_materializes::<fp128::D32Full>(fp128_d32_full_table());
    assert_every_table_entry_materializes::<fp128::D32OneHot>(fp128_d32_onehot_table());
    assert_every_table_entry_materializes::<fp128::D64Full>(fp128_d64_full_table());
    assert_every_table_entry_materializes::<fp128::D64OneHot>(fp128_d64_onehot_table());
}

#[test]
#[cfg(not(feature = "zk"))]
fn generated_small_field_schedule_tables_match_cfg_schedule() {
    assert_every_table_entry_materializes::<fp16::D32Full>(fp16_d32_full_table());
    assert_every_table_entry_materializes::<fp16::D32OneHot>(fp16_d32_onehot_table());
    assert_every_table_entry_materializes::<fp16::D64Full>(fp16_d64_full_table());
    assert_every_table_entry_materializes::<fp16::D64OneHot>(fp16_d64_onehot_table());
    assert_every_table_entry_materializes::<fp32::D32Full>(fp32_d32_table());
    assert_every_table_entry_materializes::<fp32::D32OneHot>(fp32_d32_onehot_table());
    assert_every_table_entry_materializes::<fp32::D64Full>(fp32_d64_table());
    assert_every_table_entry_materializes::<fp32::D64OneHot>(fp32_d64_onehot_table());
    assert_every_table_entry_materializes::<fp64::D32Full>(fp64_d32_table());
    assert_every_table_entry_materializes::<fp64::D32OneHot>(fp64_d32_onehot_table());
    assert_every_table_entry_materializes::<fp64::D64Full>(fp64_d64_table());
    assert_every_table_entry_materializes::<fp64::D64OneHot>(fp64_d64_onehot_table());
}

#[test]
#[cfg(not(feature = "zk"))]
fn generated_batched_roots_restore_scaled_widths() {
    assert_generated_batched_roots_are_scaled::<fp128::D32Full>(fp128_d32_full_table());
    assert_generated_batched_roots_are_scaled::<fp128::D32OneHot>(fp128_d32_onehot_table());
    assert_generated_batched_roots_are_scaled::<fp128::D64Full>(fp128_d64_full_table());
    assert_generated_batched_roots_are_scaled::<fp128::D64OneHot>(fp128_d64_onehot_table());
    assert_generated_batched_roots_are_scaled::<fp16::D32Full>(fp16_d32_full_table());
    assert_generated_batched_roots_are_scaled::<fp16::D32OneHot>(fp16_d32_onehot_table());
    assert_generated_batched_roots_are_scaled::<fp16::D64Full>(fp16_d64_full_table());
    assert_generated_batched_roots_are_scaled::<fp16::D64OneHot>(fp16_d64_onehot_table());
}

#[test]
#[cfg(not(feature = "zk"))]
fn generated_d32_full_root_fold_matches_runtime_root_plan() {
    assert_exact_root_fold_matches_runtime_root_plan::<fp128::D32Full, 32>(26);
}

#[test]
#[cfg(not(feature = "zk"))]
fn generated_d64_full_table_materializes_valid_plans() {
    let table = fp128_d64_full_table();
    for entry in table.entries {
        let key = AkitaScheduleLookupKey::new(
            entry.key.num_vars,
            entry.key.num_t_vectors,
            entry.key.num_w_vectors,
            entry.key.num_z_vectors,
        );
        <fp128::D64Full as CommitmentConfig>::schedule_plan(key)
            .expect("config schedule should succeed")
            .expect("entry should exist in generated table");
    }
}

#[test]
#[cfg(not(feature = "zk"))]
fn generated_table_rejects_sis_family_mismatch() {
    type Cfg = fp128::D64Full;
    let table = fp128_d64_full_table();
    let mismatched = GeneratedScheduleTable {
        sis_family: akita_types::SisModulusFamily::Q32,
        entries: table.entries,
    };
    let entry = mismatched
        .entries
        .iter()
        .find(|entry| entry.key.num_t_vectors == 1)
        .expect("fp128 table should contain singleton rows");
    let key = AkitaScheduleLookupKey::new_with_points(
        entry.key.num_vars,
        entry.key.num_commitment_groups,
        entry.key.num_t_vectors,
        entry.key.num_w_vectors,
        entry.key.num_z_vectors,
    );
    // Drive the planner materializer directly with the mismatched table:
    // `Cfg::schedule_plan` would use the unmodified `Cfg::schedule_table()`,
    // so we bypass it to test the SIS-family mismatch rejection path.
    let err = akita_types::schedule_plan_from_table::<<Cfg as CommitmentConfig>::Field, _>(
        key,
        mismatched,
        akita_types::PlanPolicy {
            sis_family: Cfg::sis_modulus_family(),
            ring_dimension: Cfg::D,
            root_decomp: Cfg::decomposition(),
            challenge_field_bits: Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
            recursive_public_rows: 1,
            extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
            stage1_challenge_config: Cfg::stage1_challenge_config,
            ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
            fold_challenge_shape: Cfg::fold_challenge_shape_at_level,
        },
    )
    .expect_err("mismatched SIS family must be rejected");
    assert!(
        err.to_string().contains("SIS family mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
#[cfg(not(feature = "zk"))]
fn adaptive_bounded_plan_matches_runtime_next_w_len() {
    for num_vars in [14, 20, 30] {
        assert_plan_matches_runtime_w_sizes::<fp128::D64Full>(num_vars);
    }
}

#[test]
#[cfg(not(feature = "zk"))]
fn adaptive_onehot_plan_matches_runtime_next_w_len() {
    for num_vars in [15, 30, 44] {
        assert_plan_matches_runtime_w_sizes::<fp128::D64OneHot>(num_vars);
    }
}

#[test]
#[cfg(not(feature = "zk"))]
fn batched_root_plan_matches_runtime_next_w_len() {
    let table = fp128_d64_onehot_table();
    let entry = table
        .entries
        .iter()
        .find(|entry| {
            entry.key.num_commitment_groups > 1
                || entry.key.num_t_vectors > 1
                || entry.key.num_w_vectors > 1
                || entry.key.num_z_vectors > 1
        })
        .expect("generated table should contain a non-singleton batched-root row");
    let key = AkitaScheduleLookupKey::new_with_points(
        entry.key.num_vars,
        entry.key.num_commitment_groups,
        entry.key.num_t_vectors,
        entry.key.num_w_vectors,
        entry.key.num_z_vectors,
    );

    assert_plan_matches_runtime_w_sizes_for_key::<fp128::D64OneHot>(key);
}

#[test]
#[cfg(not(feature = "zk"))]
fn batched_onehot_4x30_plan_keeps_terminal_witness_bounded() {
    let key = AkitaScheduleLookupKey::new(30, 4, 4, 1);
    let plan = <fp128::D64OneHot as CommitmentConfig>::schedule_plan(key)
        .expect("config schedule should succeed")
        .expect("fp128 D64 onehot 4x30 schedule should be generated");

    assert_plan_matches_runtime_w_sizes_for_key::<fp128::D64OneHot>(key);
    assert!(
        plan.num_fold_levels() > 2,
        "4x30 onehot schedule should keep a recursive suffix after the root fold"
    );

    let akita_types::DirectWitnessShape::PackedDigits((num_elems, _bits)) =
        plan.direct_step().witness_shape
    else {
        panic!("4x30 onehot schedule should end in packed digits");
    };
    assert!(
        num_elems <= 245_888,
        "expected byte-aware batched schedule to keep folding, got final_w with {num_elems} elems"
    );
}

#[test]
fn recursive_onehot_split_matches_open_digit_witness_count() {
    type Cfg = fp128::D64OneHot;

    // Use the root decomposition basis directly: this test exercises the
    // tight (m, r) split optimizer at a recursive state that is not part of
    // the canonical schedule, so we don't rely on `log_basis_at_level`.
    let log_basis = Cfg::decomposition().log_basis;
    let inputs = AkitaScheduleInputs {
        num_vars: 30,
        level: 1,
        current_w_len: 25_974_272,
    };
    let params =
        crate::proof_optimized::level_params_with_log_basis::<Cfg>(inputs, log_basis).unwrap();
    let root = Cfg::decomposition();
    let decomp = DecompositionParams {
        log_basis: params.log_basis,
        log_commit_bound: params.log_basis,
        log_open_bound: Some(root.log_open_bound.unwrap_or(root.log_commit_bound)),
    };
    let num_ring = inputs.current_w_len / params.ring_dimension;
    let lp_12_7 = level_layout_from_params(12, 7, &params, decomp, num_ring).unwrap();
    let lp_11_8 = level_layout_from_params(11, 8, &params, decomp, num_ring).unwrap();
    let w_12_7 = planned_w_ring_element_count::<<Cfg as CommitmentConfig>::Field>(
        Cfg::decomposition().field_bits(),
        &lp_12_7,
    )
    .unwrap();
    let w_11_8 = planned_w_ring_element_count::<<Cfg as CommitmentConfig>::Field>(
        Cfg::decomposition().field_bits(),
        &lp_11_8,
    )
    .unwrap();
    let reduced_vars = (inputs.current_w_len / params.ring_dimension)
        .next_power_of_two()
        .trailing_zeros() as usize;

    assert!(w_12_7 < w_11_8);
    let (m, r, _n_a) = optimal_m_r_split(
        params.a_key.sis_family(),
        params.ring_dimension as u32,
        params.a_key.collision_inf(),
        params.challenge_l1_mass(),
        decomp.log_commit_bound,
        decomp.log_basis,
        reduced_vars,
        num_ring,
        decomp.field_bits(),
    );
    assert_eq!((m, r), (12, 7));
}

#[test]
#[cfg(not(feature = "zk"))]
fn tight_block_len_is_no_larger_than_pow2() {
    for num_vars in [14, 20, 30] {
        let plan = fp128::D64Full::schedule_plan(AkitaScheduleLookupKey::singleton(num_vars))
            .expect("planner should succeed")
            .expect("config should provide a planner");
        for level in plan.fold_levels() {
            let pow2_block = 1usize << level.lp.m_vars;
            assert!(
                level.lp.block_len <= pow2_block,
                "block_len {} should be <= 2^m_vars {} at level {} (num_vars={})",
                level.lp.block_len,
                pow2_block,
                level.inputs.level,
                num_vars
            );
            if level.inputs.level > 0 {
                let num_ring = level.inputs.current_w_len / level.lp.ring_dimension;
                let expected_tight = num_ring.div_ceil(level.lp.num_blocks);
                assert_eq!(
                    level.lp.block_len, expected_tight,
                    "recursive level {} should use tight block_len = ceil({num_ring} / {})",
                    level.inputs.level, level.lp.num_blocks
                );
            }
        }
    }
}
