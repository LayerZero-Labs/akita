use super::*;
use akita_algebra::ring::scalar_powers;
use akita_challenges::SparseChallengeConfig;
use akita_types::{
    r_decomp_levels, AkitaSetupDescriptor, CommitmentRingDims, FlatMatrix, OpenCommitMatrixParams,
    OpeningClaimsLayout, OuterCommitMatrixParams, PreparedRelationAddress,
    SetupContributionGroupInputs, SetupContributionPlan, SisModulusProfileId,
};
use jolt_field::One;
use jolt_field::{Fp32, Prime128OffsetA7F7};

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
    let witness_layout = WitnessLayout::new(&valid_lp, &opening_batch, 1, 1).unwrap();
    let setup_groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims: 1,
        depth_fold: 1,
        a_row_start: 1,
        b_row_start: 2,
    }];
    let relation_address_geometry = RelationAddressGeometry::new(
        CommitmentRingDims::uniform(D),
        D,
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let relation_address = vec![F::one(); relation_address_geometry.relation_lane_variable_count()];
    let err = match SetupContributionPlan::prepare::<F>(
        &lp,
        &opening_batch,
        vec![F::one(); 4].into(),
        &witness_layout,
        &setup_groups,
        PreparedRelationAddress::new(&relation_address).unwrap(),
        None,
        relation_address_geometry,
    ) {
        Ok(_) => panic!("zero num_live_blocks should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn prepared_relation_accepts_exact_deferred_setup_claim_and_caches_its_plan() {
    type MixedF = Prime128OffsetA7F7;
    const D_INNER: usize = 128;
    const D_PROJECTED: usize = 64;
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
    let witness_layout = WitnessLayout::new(&lp, &opening_batch, 1, quotient_depth).unwrap();
    let role_dims = lp.role_dims();
    let relation_address_geometry =
        RelationAddressGeometry::new(role_dims, D_PROJECTED, witness_layout.live_coeff_len())
            .unwrap();
    let eq_tau1: Arc<[MixedF]> = (0..rows.next_power_of_two())
        .map(|index| MixedF::from_u64(11 + index as u64))
        .collect::<Vec<_>>()
        .into();
    let depth_fold = lp.num_digits_fold();
    let evaluator = RelationMatrixEvaluator {
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
            depth_fold,
            a_row_start: 1,
            b_row_start: 1 + lp.inner_commit_matrix.output_rank(),
        }],
        log_basis: lp.log_basis_open,
        eq_tau1,
        flat_context: Some(FlatRelationContext {
            level_params: lp,
            opening_batch,
            witness_layout: Arc::new(witness_layout),
        }),
        setup_plan_cache: Default::default(),
    };
    let setup_ring_elements = 1 << 14;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 16,
            max_num_batched_polys: 1,
            num_field_elements: setup_ring_elements,
            setup_seed: [7; 32].into(),
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
    let address_point = &point[relation_address_geometry.relation_coefficient_variable_count()..];
    let alpha = MixedF::from_u64(7);
    let fold_gadget = evaluator
        .setup_contribution_fold_gadget::<MixedF>()
        .unwrap()
        .unwrap();
    let mut direct_plan = evaluator
        .setup_contribution_plan::<MixedF>(
            PreparedRelationAddress::new(address_point).unwrap(),
            Some(&fold_gadget),
        )
        .unwrap();
    direct_plan.materialize_direct_scan(alpha).unwrap();
    assert!(direct_plan
        .materialize_direct_scan(MixedF::from_u64(11))
        .is_err());
    let setup_claim = direct_plan
        .evaluate_direct::<MixedF>(
            &setup,
            &scalar_powers(alpha, D_INNER),
            &scalar_powers(alpha, D_PROJECTED),
            &scalar_powers(alpha, D_PROJECTED),
        )
        .unwrap();

    let direct = super::relation_evaluation::evaluate_relation_at_point::<MixedF, MixedF>(
        &evaluator, &point, &setup, alpha, None,
    )
    .unwrap();
    let deferred = super::relation_evaluation::evaluate_relation_at_point::<MixedF, MixedF>(
        &evaluator,
        &point,
        &setup,
        alpha,
        Some(setup_claim),
    )
    .unwrap();
    assert_eq!(deferred, direct);

    let claim_delta = MixedF::from_u64(17);
    let changed = super::relation_evaluation::evaluate_relation_at_point::<MixedF, MixedF>(
        &evaluator,
        &point,
        &setup,
        alpha,
        Some(setup_claim + claim_delta),
    )
    .unwrap();
    let coefficient_point =
        &point[..relation_address_geometry.relation_coefficient_variable_count()];
    let common_alpha = akita_sumcheck::multilinear_eval(
        &scalar_powers(
            alpha,
            relation_address_geometry.relation_coefficient_block_len(),
        ),
        coefficient_point,
    )
    .unwrap();
    assert_eq!(changed, direct + common_alpha * claim_delta);

    let cached = evaluator
        .take_cached_setup_contribution_plan(address_point)
        .unwrap()
        .expect("mixed deferred evaluation must cache its Stage-3 plan");
    assert!(
        cached.group_column_eq_slices(0).is_none(),
        "deferred relation evaluation should cache spans without prepared columns"
    );
}
