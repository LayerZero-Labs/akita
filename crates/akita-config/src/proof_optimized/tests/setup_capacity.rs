use super::*;

#[test]
fn setup_level_params_from_runtime_schedule_excludes_terminal_direct() {
    // Terminal-direct steps ship the cleartext witness without
    // committing, so they have no `LevelParams` of their own and
    // must not contribute to the FS-bound `setup_levels`. Only
    // the preceding Fold steps (which do commit) appear.
    use akita_challenges::SparseChallengeConfig;
    use akita_types::{CleartextWitnessShape, DirectStep, FoldStep, SisModulusFamily, Step};

    let sparse = SparseChallengeConfig::pm1_only(1);
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
            Ok(akita_challenges::SparseChallengeConfig::pm1_only(1))
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
            Ok(akita_challenges::SparseChallengeConfig::pm1_only(1))
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
        .expect("runtime setup envelope should use runtime schedule only");
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

    let real_schedule = root_direct_schedule(
        real_opening_batch
            .root_direct_witness_len()
            .expect("witness len"),
        real_params.clone(),
    )
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
    let singleton_schedule = root_direct_schedule(
        real_opening_batch
            .root_direct_witness_len()
            .expect("witness len"),
        singleton_params,
    )
    .expect("singleton fallback root-direct schedule");
    assert_ne!(
        digest_effective_schedule(&real_schedule),
        digest_effective_schedule(&singleton_schedule),
        "schedule digest must distinguish batched vs singleton root-direct commit params"
    );
}

#[test]
fn multi_group_multi_chunk_schedule_rejects_at_effective_schedule_boundary() {
    type Cfg = fp128::D64OneHotMultiChunkW2R2;
    let opening_batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(8, 1),
        PolynomialGroupLayout::new(16, 1),
    ])
    .expect("multi-group opening batch");
    let point = vec![fp128::Field::zero(); opening_batch.max_num_vars()];

    let err = crate::effective_batched_schedule::<Cfg>(&opening_batch, &point)
        .expect_err("multi-group multi-chunk schedule must reject");

    assert!(
        err.to_string().contains("multi-chunk witness layout"),
        "unexpected error: {err}"
    );
}

#[test]
fn setup_matrix_envelope_covers_multi_group_batch_schedules() {
    let opening_batch =
        OpeningClaimsLayout::new(30, 4).expect("multi-group same-point opening_batch");
    let multi_group_same_point = setup_matrix_envelope_for_shape::<fp128::D128Full>(&opening_batch)
        .expect("multi-group same-point shape should resolve to a setup envelope");

    let setup_envelope = proof_optimized_max_setup_matrix_size::<fp128::D128Full>(30, 4)
        .expect("setup envelope should cover generated multi-group batch schedules");
    assert!(setup_envelope.max_setup_len >= multi_group_same_point.max_setup_len);
}

#[test]
fn dense_setup_capacity_does_not_require_conservative_onehot_plans() {
    type Cfg = fp128::D128Full;
    let layout = OpeningClaimsLayout::new(16, 4).expect("dense layout");
    let runtime = setup_matrix_envelope_for_shape::<Cfg>(&layout).expect("runtime envelope");
    let envelope = proof_optimized_max_setup_matrix_size::<Cfg>(16, 4)
        .expect("dense setup capacity certificate");
    assert!(envelope.max_setup_len >= runtime.max_setup_len);
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
        lp.num_blocks,
        lp.num_digits_open,
    );
    expected_root_setup_len(lp.d_key.row_len(), d_width, a_len, b_len)
}

fn expected_group_setup_footprint(
    a_rows: usize,
    a_width: usize,
    b_rows: usize,
    num_polys: usize,
    num_blocks: usize,
    num_digits_open: usize,
) -> (usize, usize, usize) {
    let a_len = a_rows * a_width;
    let d_width = num_polys * num_blocks * num_digits_open;
    let t_cols_per_vector = a_rows * num_digits_open * num_blocks;
    let b_len = b_rows * num_polys * t_cols_per_vector;
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
        lp.num_blocks,
        lp.num_digits_open,
    );

    for group in &lp.precommitted_groups {
        let (a_len, b_len, group_d_width) = expected_group_setup_footprint(
            group.a_key.row_len(),
            group.a_key.col_len(),
            group.b_key.row_len(),
            group.layout.group.num_polynomials(),
            group.num_blocks,
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
    let schedule = root_direct_schedule(
        opening_batch
            .root_direct_witness_len()
            .expect("witness len"),
        root_params.clone(),
    )
    .expect("synthetic direct schedule");
    let required = expected_runtime_root_setup_len(&root_params, &opening_batch);

    let runtime_envelope = matrix_envelope_for_schedule::<Cfg>(&schedule, &opening_batch).unwrap();
    assert!(runtime_envelope.max_setup_len >= required);
}

#[test]
fn setup_matrix_scan_uses_one_shared_opening_point() {
    let opening_batch =
        worst_case_multi_group_opening_batch_for_shape(30, 4).expect("valid opening batch");
    assert_eq!(opening_batch.num_total_polynomials(), 4);
}

fn runtime_layout_setup_requirement<Cfg: CommitmentConfig>(layout: &OpeningClaimsLayout) -> usize {
    let schedule = Cfg::get_params_for_prove(layout).expect("runtime schedule");
    let root_params = root_commit_params_from_schedule(&schedule)
        .expect("root params lookup")
        .expect("runtime schedule should carry commit params");
    let required = expected_runtime_root_setup_len(&root_params, layout);
    let runtime_envelope =
        matrix_envelope_for_schedule::<Cfg>(&schedule, layout).expect("runtime setup envelope");
    assert!(
        runtime_envelope.max_setup_len >= required,
        "runtime envelope must cover grouped-root setup footprint"
    );
    required
}

fn assert_all_small_runtime_layouts_fit<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) {
    let setup = proof_optimized_max_setup_matrix_size::<Cfg>(max_num_vars, max_num_batched_polys)
        .expect("small setup certificate");
    let mut accepted = 0usize;
    let mut accepted_grouped = 0usize;

    fn visit_precommitteds<Cfg: CommitmentConfig>(
        max_num_vars: usize,
        remaining: usize,
        final_group: PolynomialGroupLayout,
        precommitteds: &mut Vec<PolynomialGroupLayout>,
        setup: SetupMatrixEnvelope,
        accepted: &mut usize,
        accepted_grouped: &mut usize,
    ) {
        let layout = OpeningClaimsLayout::from_root_groups(precommitteds, final_group)
            .expect("enumerated layout");
        if let Ok(schedule) = Cfg::get_params_for_prove(&layout) {
            let runtime = matrix_envelope_for_schedule::<Cfg>(&schedule, &layout)
                .expect("accepted runtime envelope");
            assert!(
                setup.max_setup_len >= runtime.max_setup_len,
                "certificate {} misses runtime {} for {:?}",
                setup.max_setup_len,
                runtime.max_setup_len,
                layout
            );
            *accepted += 1;
            *accepted_grouped += usize::from(!precommitteds.is_empty());
        }
        for num_polynomials in 1..=remaining {
            for num_vars in 1..=max_num_vars {
                precommitteds.push(PolynomialGroupLayout::new(num_vars, num_polynomials));
                visit_precommitteds::<Cfg>(
                    max_num_vars,
                    remaining - num_polynomials,
                    final_group,
                    precommitteds,
                    setup,
                    accepted,
                    accepted_grouped,
                );
                precommitteds.pop();
            }
        }
    }

    for final_num_polynomials in 1..=max_num_batched_polys {
        for final_num_vars in 1..=max_num_vars {
            visit_precommitteds::<Cfg>(
                max_num_vars,
                max_num_batched_polys - final_num_polynomials,
                PolynomialGroupLayout::new(final_num_vars, final_num_polynomials),
                &mut Vec::new(),
                setup,
                &mut accepted,
                &mut accepted_grouped,
            );
        }
    }
    assert!(accepted > 0);
    // Tiny finals often cannot host Fold→Fold after the terminal-fold cutover,
    // so grouped acceptance is optional at this enumeration size.
    let _ = accepted_grouped;
}

#[test]
fn setup_certificate_covers_schedulable_grouped_onehot_layout() {
    type Cfg = fp128::D64OneHot;
    let layout = OpeningClaimsLayout::from_root_groups(
        &[PolynomialGroupLayout::new(16, 1)],
        PolynomialGroupLayout::new(16, 1),
    )
    .expect("schedulable grouped layout");
    let required = runtime_layout_setup_requirement::<Cfg>(&layout);
    let setup = proof_optimized_max_setup_matrix_size::<Cfg>(16, 2).expect("setup certificate");
    assert!(setup.max_setup_len >= required);
}

#[test]
fn setup_certificate_covers_every_small_flat_onehot_runtime_layout() {
    assert_all_small_runtime_layouts_fit::<fp128::D64OneHot>(7, 3);
}

#[test]
fn setup_certificate_covers_every_small_tensor_onehot_runtime_layout() {
    assert_all_small_runtime_layouts_fit::<crate::tensor_verifier::fp128::D64OneHotTensor>(7, 3);
}

#[test]
fn multi_chunk_onehot_base_setup_covers_conservative_adapter() {
    type Base = fp128::D64OneHotMultiChunkW2R2;
    type Conservative = crate::ConservativeCommitmentConfig<Base>;
    let layout = OpeningClaimsLayout::new(16, 4).expect("multi-chunk conservative layout");
    let params = Conservative::get_params_for_batched_commitment(&layout)
        .expect("multi-chunk conservative params");
    let required = expected_runtime_root_setup_len(&params, &layout);
    let envelope = Base::max_setup_matrix_size(16, 4).expect("base setup envelope");
    assert!(envelope.max_setup_len >= required);
}

#[derive(Clone)]
struct CanonicalGroupedRootShapeCfg;

impl CommitmentConfig for CanonicalGroupedRootShapeCfg {
    type Field = <fp128::D64OneHot as CommitmentConfig>::Field;
    type ExtField = <fp128::D64OneHot as CommitmentConfig>::ExtField;
    const D: usize = fp128::D64OneHot::D;

    fn decomposition() -> akita_types::DecompositionParams {
        fp128::D64OneHot::decomposition()
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        fp128::D64OneHot::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(
        inputs: akita_types::AkitaScheduleInputs,
    ) -> akita_challenges::TensorChallengeShape {
        if inputs.current_w_len == 1usize << inputs.num_vars {
            akita_challenges::TensorChallengeShape::Flat
        } else {
            akita_challenges::TensorChallengeShape::Tensor
        }
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        fp128::D64OneHot::sis_modulus_family()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        proof_optimized_max_setup_matrix_size::<Self>(max_num_vars, max_num_batched_polys)
    }

    fn basis_range() -> (u32, u32) {
        fp128::D64OneHot::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        fp128::D64OneHot::onehot_chunk_size()
    }
}

#[test]
fn grouped_direct_and_fold_share_canonical_root_shape_input() {
    type Cfg = CanonicalGroupedRootShapeCfg;
    // Grouped roots require a nonterminal fold successor; final nv must leave
    // room for Fold→Fold after the terminal-fold cutover.
    let layout = OpeningClaimsLayout::from_root_groups(
        &[PolynomialGroupLayout::new(8, 1)],
        PolynomialGroupLayout::new(16, 1),
    )
    .expect("grouped layout");
    let schedule = Cfg::get_params_for_prove(&layout).expect("grouped schedule");
    let root = root_commit_params_from_schedule(&schedule)
        .expect("root params")
        .expect("committed root");
    assert_eq!(
        root.fold_challenge_shape,
        akita_challenges::TensorChallengeShape::Flat
    );
    assert!(root
        .precommitted_groups
        .iter()
        .all(|group| group.layout.group.num_vars() == 8));

    let runtime =
        matrix_envelope_for_schedule::<Cfg>(&schedule, &layout).expect("runtime envelope");
    let setup = Cfg::max_setup_matrix_size(16, 2).expect("setup envelope");
    assert!(setup.max_setup_len >= runtime.max_setup_len);
}

#[test]
fn setup_capacity_cap_32_1_does_not_size_impossible_three_group_layout() {
    type Cfg = fp128::D64OneHot;

    let impossible = OpeningClaimsLayout::from_root_groups(
        &[
            PolynomialGroupLayout::new(1, 1),
            PolynomialGroupLayout::new(1, 1),
        ],
        PolynomialGroupLayout::new(1, 1),
    )
    .expect("three-group singleton layout");
    assert!(
        !super::layout_within_setup_capacity(&impossible, 32, 1),
        "total polynomial capacity 1 excludes three singleton groups"
    );

    let setup_envelope =
        proof_optimized_max_setup_matrix_size::<Cfg>(32, 1).expect("singleton setup envelope");
    let scalar_requirement =
        runtime_layout_setup_requirement::<Cfg>(&OpeningClaimsLayout::new(32, 1).expect("scalar"));
    assert!(
        setup_envelope.max_setup_len >= scalar_requirement,
        "setup envelope must cover the largest accepted singleton layout"
    );

    if let Ok(schedule) = Cfg::get_params_for_prove(&impossible) {
        let inflated = matrix_envelope_for_schedule::<Cfg>(&schedule, &impossible)
            .expect("impossible envelope");
        assert!(
            setup_envelope.max_setup_len < inflated.max_setup_len,
            "cap(32,1) must not inflate from the out-of-capacity three-group layout"
        );
    }
}

#[test]
fn setup_capacity_cap_16_4_covers_final_nv16_k1_plus_pre_nv14_k3() {
    type Cfg = fp128::D64OneHot;

    // final nv=1 cannot fold; use a foldable final inside the same poly budget.
    let layout = OpeningClaimsLayout::from_root_groups(
        &[PolynomialGroupLayout::new(14, 3)],
        PolynomialGroupLayout::new(16, 1),
    )
    .expect("unequal multi-group layout");
    assert!(super::layout_within_setup_capacity(&layout, 16, 4));

    let required = runtime_layout_setup_requirement::<Cfg>(&layout);
    let setup_envelope =
        proof_optimized_max_setup_matrix_size::<Cfg>(16, 4).expect("setup envelope");
    assert!(
        setup_envelope.max_setup_len >= required,
        "cap(16,4) must cover final(nv16,K1)+pre(nv14,K3)"
    );
}

#[test]
fn setup_capacity_cap_16_17_covers_final_nv16_k1_plus_multiple_singleton_pre_groups() {
    type Cfg = fp128::D64OneHot;

    // Sixteen precommitted singletons remain inside the poly budget, but after
    // the terminal-fold cutover that arity is not necessarily schedulable.
    let sixteen = vec![PolynomialGroupLayout::new(16, 1); 16];
    let sixteen_layout =
        OpeningClaimsLayout::from_root_groups(&sixteen, PolynomialGroupLayout::new(16, 1))
            .expect("sixteen-precommit layout");
    assert!(super::layout_within_setup_capacity(&sixteen_layout, 16, 17));

    // Cover a high-arity but still Fold→Fold-schedulable representative.
    let precommitteds = vec![PolynomialGroupLayout::new(16, 1); 2];
    let layout =
        OpeningClaimsLayout::from_root_groups(&precommitteds, PolynomialGroupLayout::new(16, 1))
            .expect("two-precommit layout");
    assert!(super::layout_within_setup_capacity(&layout, 16, 17));

    let required = runtime_layout_setup_requirement::<Cfg>(&layout);
    let setup_envelope =
        proof_optimized_max_setup_matrix_size::<Cfg>(16, 17).expect("setup envelope");
    assert!(
        setup_envelope.max_setup_len >= required,
        "cap(16,17) must cover final(nv16,K1)+multiple singleton precommitted groups"
    );
}

#[test]
fn setup_capacity_conservative_d64_onehot_cap_30_4_covers_nv29_k4() {
    type Cfg = fp128::D64OneHot;

    let layout = OpeningClaimsLayout::new(29, 4).expect("conservative batched layout");
    assert!(super::layout_within_setup_capacity(&layout, 30, 4));

    let conservative_params = crate::conservative_commitment::conservative_commit_params::<Cfg>(
        &PolynomialGroupLayout::new(29, 4),
    )
    .expect("conservative commit params");
    let conservative_schedule = akita_types::root_direct_schedule(
        layout.root_direct_witness_len().expect("witness len"),
        conservative_params,
    )
    .expect("conservative root-direct schedule");
    let required = expected_runtime_root_setup_len(
        &root_commit_params_from_schedule(&conservative_schedule)
            .expect("root params lookup")
            .expect("conservative schedule should carry params"),
        &layout,
    );
    let conservative_envelope =
        matrix_envelope_for_schedule::<Cfg>(&conservative_schedule, &layout)
            .expect("conservative runtime envelope");
    assert!(conservative_envelope.max_setup_len >= required);

    let setup_envelope =
        proof_optimized_max_setup_matrix_size::<Cfg>(30, 4).expect("setup envelope");
    assert!(
        setup_envelope.max_setup_len >= required,
        "conservative D64OneHot cap(30,4) must cover widened conservative B ranks"
    );
}

#[test]
#[cfg(feature = "schedules-fp128-d64-onehot")]
fn proof_optimized_setup_includes_precommitted_multi_group_root_catalog_entries() {
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
        .expect("generated two-precommit multi-group-root entry");
    let key = super::runtime_key_from_generated_entry(entry);
    let schedule =
        Cfg::runtime_schedule(key.clone()).expect("precommitted multi-group-root schedule");
    let layout = key.opening_layout().expect("multi-group layout");
    let entry_envelope =
        super::matrix_envelope_for_schedule::<Cfg>(&schedule, &layout).expect("entry envelope");

    let setup_envelope = super::proof_optimized_max_setup_matrix_size::<Cfg>(16, 3)
        .expect("setup envelope should include precommitted multi-group-root catalog entries");

    assert!(
        setup_envelope.max_setup_len >= entry_envelope.max_setup_len,
        "setup envelope must cover generated precommitted multi-group-root catalog footprints"
    );
}

#[test]
fn proof_optimized_setup_includes_arbitrary_precommit_group_sizes() {
    type Cfg = fp128::D64OneHot;

    let layout = OpeningClaimsLayout::from_root_groups(
        &[PolynomialGroupLayout::new(8, 1)],
        PolynomialGroupLayout::new(16, 1),
    )
    .expect("larger precommitted group layout");
    let runtime = setup_matrix_envelope_for_shape::<Cfg>(&layout)
        .expect("larger precommitted group should be schedulable");
    let setup_envelope =
        super::proof_optimized_max_setup_matrix_size::<Cfg>(16, 2).expect("setup envelope");

    assert!(
        setup_envelope.max_setup_len >= runtime.max_setup_len,
        "setup envelope must cover precommitted groups with independent sizes from the final group"
    );
}

#[test]
fn grouped_root_runtime_setup_uses_per_group_roles_and_summed_d_width() {
    type Cfg = fp128::D64OneHot;

    let layout = OpeningClaimsLayout::from_root_groups(
        &[PolynomialGroupLayout::new(24, 1)],
        PolynomialGroupLayout::new(20, 1),
    )
    .expect("grouped root layout");
    let key = crate::opening_schedule_key::<Cfg>(&layout).expect("grouped root key");
    let schedule = Cfg::runtime_schedule(key).expect("grouped root schedule");
    let root_params = root_commit_params_from_schedule(&schedule)
        .expect("root params lookup")
        .expect("grouped root should carry params");

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
        final_group.num_polynomials() * root_params.num_blocks * root_params.num_digits_open
            + precommitted_d_width;
    assert_eq!(
        root_params.d_key.col_len(),
        expected_d_width,
        "multi-group root D columns are final plus all precommitted segments"
    );
}

