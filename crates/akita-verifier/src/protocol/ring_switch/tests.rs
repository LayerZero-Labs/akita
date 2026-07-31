use super::*;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::scalar_powers;
use akita_challenges::{SparseChallenge, SparseChallengeConfig, TensorChallenges};
use akita_field::{Fp32, Prime128OffsetA7F7};
use akita_types::{
    gadget_row_scalars, AkitaSetupSeed, FlatMatrix, OpenCommitMatrixParams, OpeningClaimsLayout,
    OuterCommitMatrixParams, SetupContributionGroupInputs, SetupContributionPlan,
    SisModulusProfileId,
};

type F = Fp32<251>;
const D: usize = 64;

fn fold_challenge_config() -> SparseChallengeConfig {
    SparseChallengeConfig::pm1_only(1)
}

#[test]
fn ring_switch_prepare_rejects_invalid_log_basis() {
    let err = validate_log_basis(0).expect_err("invalid log_basis should be rejected");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn ring_switch_prepare_rejects_zero_num_live_blocks() {
    let lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    );
    let opening_batch = OpeningClaimsLayout::new(0, 1).expect("opening batch");
    let valid_lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(1, 1, 1, 1, 1)
    .unwrap();
    let witness_layout = WitnessLayout::new(&valid_lp, &opening_batch, 1, 4, 1).unwrap();
    let setup_groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims: 1,
        depth_fold: 1,
        a_row_start: 1,
        b_row_start: 2,
    }];
    let relation_address_geometry =
        RelationAddressGeometry::new(CommitmentRingDims::uniform(D), D, 3).unwrap();
    let err = match SetupContributionPlan::prepare::<F>(
        &lp,
        &opening_batch,
        vec![F::one(); 4].into(),
        &witness_layout,
        &setup_groups,
        &[F::one(), F::one()],
        None,
        relation_address_geometry,
        F::one(),
    ) {
        Ok(_) => panic!("zero num_live_blocks should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn mixed_relation_accepts_exact_deferred_setup_claim_and_caches_its_plan() {
    type MixedF = Prime128OffsetA7F7;
    const D_INNER: usize = 64;
    const D_PROJECTED: usize = 32;
    let mut lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D_INNER,
        2,
        1,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(4, 8, 1, 1, 1)
    .unwrap();
    let outer = &lp.outer_commit_matrix;
    lp.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * (D_INNER / D_PROJECTED),
        outer.coeff_linf_bound(),
        D_PROJECTED,
    );
    let opening = &lp.open_commit_matrix;
    lp.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        opening.security_policy(),
        opening.sis_table_key().table_digest,
        opening.sis_modulus_profile(),
        opening.output_rank(),
        opening.input_width() * (D_INNER / D_PROJECTED),
        opening.coeff_linf_bound(),
        D_PROJECTED,
    );

    let opening_batch = OpeningClaimsLayout::new(0, 1).unwrap();
    let rows = lp
        .relation_matrix_row_count(opening_batch.num_groups())
        .unwrap();
    let quotient_depth = r_decomp_levels::<MixedF>(lp.log_basis_open);
    let witness_layout = WitnessLayout::new(&lp, &opening_batch, 1, rows, quotient_depth).unwrap();
    let opening_source_len = witness_layout.total_len() * D_INNER / D_PROJECTED;
    let role_dims = lp.role_dims();
    let relation_address_geometry =
        RelationAddressGeometry::new(role_dims, D_PROJECTED, opening_source_len).unwrap();
    let eq_tau1: Arc<[MixedF]> = (0..rows.next_power_of_two())
        .map(|index| MixedF::from_u64(11 + index as u64))
        .collect::<Vec<_>>()
        .into();
    let depth_fold = lp.num_digits_fold();
    let evaluator = RelationMatrixEvaluator {
        role_dims,
        relation_address_geometry,
        groups: vec![RelationMatrixGroupEvaluator {
            c_alphas: PreparedChallengeEvals::Flat(
                (0..lp.num_live_blocks)
                    .map(|index| MixedF::from_u64(31 + index as u64))
                    .collect(),
            ),
            opening_a_evals: (0..lp.num_positions_per_block)
                .map(|index| MixedF::from_u64(41 + index as u64))
                .collect(),
            group_id: 0,
            num_claims: 1,
            num_live_blocks: lp.num_live_blocks,
            depth_witness: lp.num_digits_inner,
            depth_open: lp.num_digits_open,
            depth_commit: lp.num_digits_outer,
            depth_fold,
            log_basis_inner: lp.log_basis_inner,
            log_basis_outer: lp.log_basis_outer,
            log_basis_open: lp.log_basis_open,
            n_a: lp.inner_commit_matrix.output_rank(),
            a_row_start: 1,
            b_row_start: 1 + lp.inner_commit_matrix.output_rank(),
        }],
        log_basis: lp.log_basis_open,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp,
            opening_batch,
            witness_layout: Arc::new(witness_layout),
            opening_source_len,
            opening_ring_dim: D_PROJECTED,
        }),
        setup_plan_cache: Default::default(),
    };
    let setup_ring_elements = 1 << 14;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 16,
            max_num_batched_polys: 1,
            num_field_elements: setup_ring_elements,
            public_matrix_id: [7; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_ring_elements * D_INNER)
                .map(|index| MixedF::from_u64(101 + index as u64))
                .collect(),
        ),
    );
    let point = (0..relation_address_geometry.relation_point_variable_count())
        .map(|index| MixedF::from_u64(211 + index as u64))
        .collect::<Vec<_>>();
    let address_point =
        &point[relation_address_geometry.common_relation_witness_variable_count()..];
    let alpha = MixedF::from_u64(7);
    let fold_gadget = evaluator
        .setup_contribution_fold_gadget::<MixedF>()
        .unwrap()
        .unwrap();
    let direct_plan = evaluator
        .setup_contribution_plan::<MixedF>(address_point, Some(&fold_gadget), alpha)
        .unwrap();
    let setup_claim = direct_plan
        .evaluate_direct::<MixedF>(
            &setup,
            &scalar_powers(alpha, D_INNER),
            &scalar_powers(alpha, D_PROJECTED),
            &scalar_powers(alpha, D_PROJECTED),
        )
        .unwrap();

    let direct = super::mixed_relation::evaluate_lane_factored_relation_at_point::<MixedF, MixedF>(
        &evaluator, &point, &setup, alpha, None,
    )
    .unwrap();
    let deferred =
        super::mixed_relation::evaluate_lane_factored_relation_at_point::<MixedF, MixedF>(
            &evaluator,
            &point,
            &setup,
            alpha,
            Some(setup_claim),
        )
        .unwrap();
    assert_eq!(deferred, direct);

    let claim_delta = MixedF::from_u64(17);
    let changed =
        super::mixed_relation::evaluate_lane_factored_relation_at_point::<MixedF, MixedF>(
            &evaluator,
            &point,
            &setup,
            alpha,
            Some(setup_claim + claim_delta),
        )
        .unwrap();
    let coefficient_point =
        &point[..relation_address_geometry.common_relation_witness_variable_count()];
    let common_alpha = akita_sumcheck::multilinear_eval(
        &scalar_powers(
            alpha,
            relation_address_geometry.common_relation_witness_coeff_count(),
        ),
        coefficient_point,
    )
    .unwrap();
    assert_eq!(changed, direct + common_alpha * claim_delta);

    let cached = evaluator
        .take_cached_setup_contribution_plan(address_point)
        .unwrap()
        .expect("mixed deferred evaluation must cache its Stage-3 plan");
    let (e_slice, t_slice, z_slice) = cached
        .group_column_eq_slices(0)
        .expect("cached plan must retain checked group spans");
    assert!(
        e_slice.is_empty() && t_slice.is_empty() && z_slice.is_empty(),
        "deferred mixed evaluation should cache the compact plan"
    );
}

#[test]
fn tensor_et_intervals_match_dense_oracle_across_residual_shards() {
    let lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        D,
        2,
        2,
        1,
        1,
        fold_challenge_config(),
    )
    .with_decomp(4, 25, 1, 1, 3)
    .unwrap();
    let opening_batch = OpeningClaimsLayout::new(0, 1).unwrap();
    let depth_fold = lp.num_digits_fold();
    let witness_layout = WitnessLayout::new(&lp, &opening_batch, 2, 4, 2).unwrap();
    let units = witness_layout.units_for_group(0).unwrap();
    assert_eq!(
        units
            .iter()
            .map(|unit| unit.num_live_blocks())
            .collect::<Vec<_>>(),
        vec![4, 3]
    );

    let sparse = |position: u32, sign: i8| SparseChallenge {
        positions: vec![position],
        coeffs: vec![sign],
    };
    let tensor = TensorChallenges {
        fold_high: vec![sparse(0, 1), sparse(1, -1)],
        fold_low: vec![sparse(4, 1), sparse(5, 1), sparse(6, -1), sparse(7, 1)],
        num_live_blocks_per_claim: 7,
        fold_low_len: 4,
        num_claims: 1,
    };
    let alpha_pows = scalar_powers(F::from_u64(5), D);
    let group = RelationMatrixGroupEvaluator {
        c_alphas: PreparedChallengeEvals::Tensor {
            challenges: tensor.clone(),
            alpha_pows: alpha_pows.clone(),
        },
        opening_a_evals: Vec::new(),
        group_id: 0,
        num_claims: 1,
        num_live_blocks: 7,
        depth_witness: 1,
        depth_open: 3,
        depth_commit: 1,
        depth_fold,
        log_basis_inner: 2,
        log_basis_outer: 2,
        log_basis_open: 2,
        n_a: 2,
        a_row_start: 1,
        b_row_start: 3,
    };
    let opening_source_len = witness_layout.total_len();
    let bits = opening_source_len.next_power_of_two().trailing_zeros() as usize;
    let x_challenges = (0..bits)
        .map(|index| F::from_u64(17 + index as u64))
        .collect::<Vec<_>>();
    let consistency_weight = F::from_u64(29);
    let a_row_weights = [F::from_u64(31), F::from_u64(37)];
    let gadget = [F::from_u64(1), F::from_u64(4), F::from_u64(16)];

    let setup_groups = vec![SetupContributionGroupInputs {
        group_id: group.group_id,
        num_claims: group.num_claims,
        depth_fold: group.depth_fold,
        a_row_start: group.a_row_start,
        b_row_start: group.b_row_start,
    }];
    let rows = lp
        .relation_matrix_row_count(opening_batch.num_groups())
        .unwrap();
    let mut eq_tau1 = vec![F::from_u64(41); rows];
    eq_tau1[0] = consistency_weight;
    eq_tau1[group.a_row_start..group.a_row_start + group.n_a].copy_from_slice(&a_row_weights);
    let fold_gadget = gadget_row_scalars::<F>(group.depth_fold, group.log_basis_open);
    let relation_address_geometry =
        RelationAddressGeometry::new(CommitmentRingDims::uniform(D), D, opening_source_len)
            .unwrap();
    let setup_plan = SetupContributionPlan::prepare::<F>(
        &lp,
        &opening_batch,
        eq_tau1.into(),
        &witness_layout,
        &setup_groups,
        &x_challenges,
        Some(&fold_gadget),
        relation_address_geometry,
        F::from_u64(43),
    )
    .unwrap();
    let (e_eq_slice, t_eq_slice, _) = setup_plan.group_column_eq_slices(0).unwrap();
    let g_open_ext = gadget_row_scalars::<F>(group.depth_open, group.log_basis_open);
    let g_t_commit_ext = gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);

    let got = evaluate_group_et_from_eq_slices::<F, F>(
        &group,
        consistency_weight,
        &a_row_weights,
        &g_open_ext,
        &g_t_commit_ext,
        e_eq_slice,
        t_eq_slice,
    )
    .unwrap();

    let mut expected = (F::zero(), F::zero());
    for &unit in &units {
        for claim in 0..group.num_claims {
            for global_block in unit.global_block_range() {
                let logical = claim * group.num_live_blocks + global_block;
                let challenge = tensor
                    .eval_logical_at_pows::<F, F>(logical, &alpha_pows)
                    .unwrap();
                for (digit, &digit_weight) in gadget.iter().enumerate() {
                    let e_index = unit
                        .e_index(
                            group.num_claims,
                            group.depth_open,
                            claim,
                            global_block,
                            digit,
                        )
                        .unwrap();
                    expected.0 += eq_eval_at_index(&x_challenges, e_index)
                        * consistency_weight
                        * challenge
                        * digit_weight;
                }
                for (digit, &digit_weight) in gadget[..group.depth_commit].iter().enumerate() {
                    for (a_row, &row_weight) in a_row_weights.iter().enumerate() {
                        let t_index = unit
                            .t_index(
                                group.num_claims,
                                group.n_a,
                                group.depth_commit,
                                claim,
                                global_block,
                                a_row,
                                digit,
                            )
                            .unwrap();
                        expected.1 += eq_eval_at_index(&x_challenges, t_index)
                            * row_weight
                            * challenge
                            * digit_weight;
                    }
                }
            }
        }
    }
    assert_eq!(got, expected);
}
