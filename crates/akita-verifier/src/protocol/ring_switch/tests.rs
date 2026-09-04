use super::*;
use akita_algebra::ring::scalar_powers;
use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_types::{
    relation_rhs_coeff_len, AkitaSetupDescriptor, CommitmentRingDims, FlatMatrix,
    OpenCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, PreparedRelationAddress,
    RingOpeningPoint, RingRelationGroupOpening, RingVec, SetupContributionGroupInputs,
    SetupContributionPlan, SisModulusProfileId,
};
use jolt_field::{Fp32, One, Prime128OffsetA7F7, Zero};

type F = Fp32<251>;
const D: usize = 64;
type MixedF = Prime128OffsetA7F7;
const D_INNER: usize = 128;
const D_PROJECTED: usize = 64;

fn fold_challenge_config() -> SparseChallengeConfig {
    SparseChallengeConfig::pm1_only(1)
}

struct MixedRelationFixture {
    evaluator: RelationMatrixEvaluator<MixedF>,
    setup: AkitaExpandedSetup<MixedF>,
    point: Vec<MixedF>,
    alpha: MixedF,
}

struct MixedReplayFixture {
    lp: CommittedGroupParams,
    relation: RingRelationInstance<MixedF>,
    setup: AkitaExpandedSetup<MixedF>,
    row_coefficients: Vec<MixedF>,
    tau1: Vec<MixedF>,
    point: Vec<MixedF>,
    alpha: MixedF,
    opening_source_len: usize,
}

impl MixedReplayFixture {
    fn prepare_evaluator(&self) -> Result<RelationMatrixEvaluator<MixedF>, AkitaError> {
        prepare_relation_matrix_evaluator::<MixedF, MixedF>(
            &RingSwitchReplay {
                setup: &self.setup,
                relation: &self.relation,
                row_coefficients: &self.row_coefficients,
                lp: &self.lp,
                opening_source_len: self.opening_source_len,
                opening_ring_dim: D_PROJECTED,
            },
            self.alpha,
            &self.tau1,
            None,
        )
    }
}

fn mixed_replay_fixture(mode: akita_types::RingRelationMode) -> MixedReplayFixture {
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
    lp.ring_relation_mode = mode;
    let outer = &lp.outer().matrix;
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * (D_INNER / D_PROJECTED),
        outer.coeff_linf_bound(),
        D_PROJECTED,
    );
    let opening = &lp.open().matrix;
    lp.open_matrix = OpenCommitMatrixParams::new_unchecked(
        opening.security_policy(),
        opening.sis_table_key().table_digest,
        opening.sis_modulus_profile(),
        opening.output_rank(),
        opening.input_width() * (D_INNER / D_PROJECTED),
        opening.coeff_linf_bound(),
        D_PROJECTED,
    );

    let opening_batch = OpeningClaimsLayout::new(0, 1).unwrap();
    let relation_geometry =
        akita_types::RelationWitnessGeometry::for_evaluation_trace_execution(&lp, &opening_batch)
            .unwrap();
    let challenges = Challenges::from_sparse(
        (0..lp.blocks().live_blocks)
            .map(|index| SparseChallenge {
                positions: vec![(index % D_INNER) as u32].into(),
                coeffs: vec![1].into(),
            })
            .collect(),
        lp.blocks().live_blocks,
        1,
    )
    .unwrap();
    let gamma = MixedF::from_u64(31);
    let mut gamma_ring = vec![MixedF::zero(); D_INNER];
    gamma_ring[0] = gamma;
    let relation = RingRelationInstance::new(
        vec![RingRelationGroupOpening::evaluation_trace(
            challenges,
            RingMultiplierOpeningPoint::from_base(&RingOpeningPoint {
                position_weights: (0..lp.blocks().positions_per_block)
                    .map(|index| MixedF::from_u64(41 + index as u64))
                    .collect(),
                live_block_weights: vec![MixedF::zero(); lp.blocks().live_blocks],
            }),
        )],
        1,
        opening_batch,
        vec![gamma],
        RingVec::from_coeffs_with_ring_dim(gamma_ring, D_INNER).unwrap(),
        RingVec::from_coeffs(vec![
            MixedF::zero();
            relation_rhs_coeff_len(relation_geometry.rhs_layout())
                .unwrap()
        ]),
        RingVec::from_coeffs(Vec::new()),
        lp.role_dims(),
    )
    .unwrap();
    let witness_layout = relation.segment_layout(&lp, None).unwrap();
    let opening_source_len = witness_layout.live_coeff_len().div_ceil(D_PROJECTED);
    let placeholder_setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 1,
            num_field_elements: 1,
            setup_seed: [7; 32].into(),
        },
        FlatMatrix::from_flat_data(vec![MixedF::one()]),
    );
    let rows = lp
        .relation_matrix_row_count(relation.opening_batch().num_groups())
        .unwrap();
    let tau1 = (0..rows.next_power_of_two().trailing_zeros() as usize)
        .map(|index| MixedF::from_u64(11 + index as u64))
        .collect::<Vec<_>>();
    let alpha = MixedF::from_u64(7);
    let row_coefficients = vec![MixedF::from_u64(31)];
    let replay = RingSwitchReplay {
        setup: &placeholder_setup,
        relation: &relation,
        row_coefficients: &row_coefficients,
        lp: &lp,
        opening_source_len,
        opening_ring_dim: D_PROJECTED,
    };
    let evaluator =
        prepare_relation_matrix_evaluator::<MixedF, MixedF>(&replay, alpha, &tau1, None).unwrap();
    let point = (0..evaluator
        .relation_address_geometry
        .relation_point_variable_count())
        .map(|index| MixedF::from_u64(211 + index as u64))
        .collect::<Vec<_>>();
    let mut prepared = super::relation_evaluation::PreparedDirectRelation::prepare::<MixedF>(
        &evaluator, &point, alpha,
    )
    .unwrap();
    prepared.materialize_setup().unwrap();
    let setup_field_len = prepared.setup_field_len();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 1,
            num_field_elements: setup_field_len,
            setup_seed: [7; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_field_len)
                .map(|index| MixedF::from_u64(101 + index as u64))
                .collect(),
        ),
    );
    MixedReplayFixture {
        lp,
        relation,
        setup,
        row_coefficients,
        tau1,
        point,
        alpha,
        opening_source_len,
    }
}

fn mixed_relation_fixture(mode: akita_types::RingRelationMode) -> MixedRelationFixture {
    let fixture = mixed_replay_fixture(mode);
    let evaluator = fixture.prepare_evaluator().unwrap();
    MixedRelationFixture {
        evaluator,
        setup: fixture.setup,
        point: fixture.point,
        alpha: fixture.alpha,
    }
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
    let relation_geometry = akita_types::RelationWitnessGeometry::for_evaluation_trace_execution(
        &valid_lp,
        &opening_batch,
    )
    .unwrap();
    let witness_layout = WitnessLayout::new(
        &valid_lp,
        &opening_batch,
        &relation_geometry,
        1,
        akita_types::RelationQuotientPlan::quotient_lift(1).unwrap(),
    )
    .unwrap();
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
        1,
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
    let MixedRelationFixture {
        evaluator,
        setup,
        point,
        alpha,
    } = mixed_relation_fixture(akita_types::RingRelationMode::QuotientLift);
    let relation_address_geometry = evaluator.relation_address_geometry;
    let address_point = &point[relation_address_geometry.relation_coefficient_variable_count()..];
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
    direct_plan
        .materialize_direct_scan(akita_types::PreparedCoefficientFunctional::lifted_power(
            alpha,
        ))
        .unwrap();
    assert!(direct_plan
        .materialize_direct_scan(akita_types::PreparedCoefficientFunctional::lifted_power(
            MixedF::from_u64(11),
        ))
        .is_err());
    let setup_claim = direct_plan.evaluate_direct::<MixedF>(&setup).unwrap();

    let direct = super::relation_evaluation::evaluate_relation_at_point::<MixedF, MixedF>(
        &evaluator, &point, &setup, alpha,
    )
    .unwrap();
    let deferred = super::relation_evaluation::evaluate_quotient_relation_with_deferred_setup::<
        MixedF,
        MixedF,
    >(&evaluator, &point, &setup, alpha, setup_claim)
    .unwrap();
    assert_eq!(deferred, direct);

    let claim_delta = MixedF::from_u64(17);
    let changed = super::relation_evaluation::evaluate_quotient_relation_with_deferred_setup::<
        MixedF,
        MixedF,
    >(&evaluator, &point, &setup, alpha, setup_claim + claim_delta)
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

#[test]
fn reduced_relation_dispatch_is_complete_and_rejects_deferred_or_mismatched_state() {
    let MixedRelationFixture {
        evaluator,
        setup,
        point,
        alpha,
    } = mixed_relation_fixture(akita_types::RingRelationMode::ReducedEvaluation);
    let geometry = evaluator.relation_address_geometry;
    let coefficient_bits = geometry.relation_coefficient_variable_count();
    let (coefficient_point, address_point) = point.split_at(coefficient_bits);
    let fold_gadget = evaluator
        .setup_contribution_fold_gadget::<MixedF>()
        .unwrap()
        .unwrap();
    let mut plan = evaluator
        .setup_contribution_plan::<MixedF>(
            PreparedRelationAddress::new(address_point).unwrap(),
            Some(&fold_gadget),
        )
        .unwrap();
    plan.materialize_direct_scan(
        akita_types::PreparedCoefficientFunctional::reduced_evaluation(
            alpha,
            coefficient_point,
            geometry,
        )
        .unwrap(),
    )
    .unwrap();
    let PreparedRelationGroups::ReducedEvaluation(groups) = &evaluator.groups else {
        panic!("fixture must prepare reduced relation groups");
    };
    let structured = groups
        .iter()
        .try_fold(MixedF::zero(), |sum, group| {
            Ok::<_, AkitaError>(
                sum + plan.evaluate_reduced_structured_group::<MixedF>(
                    group.group_id,
                    &group.multipliers.challenges,
                    &group.multipliers.opening,
                )?,
            )
        })
        .unwrap();
    let expected = structured + plan.evaluate_direct::<MixedF>(&setup).unwrap();
    let got = super::relation_evaluation::evaluate_relation_at_point::<MixedF, MixedF>(
        &evaluator, &point, &setup, alpha,
    )
    .unwrap();
    assert_eq!(got, expected);

    let common_alpha = akita_sumcheck::multilinear_eval(
        &scalar_powers(alpha, geometry.relation_coefficient_block_len()),
        coefficient_point,
    )
    .unwrap();
    assert_ne!(common_alpha, MixedF::one());
    assert_ne!(got, common_alpha * expected);
    assert!(
        super::relation_evaluation::evaluate_quotient_relation_with_deferred_setup::<
            MixedF,
            MixedF,
        >(
            &evaluator,
            &point,
            &setup,
            alpha,
            MixedF::one(),
        )
        .is_err()
    );
    assert!(evaluator
        .take_cached_setup_contribution_plan(address_point)
        .unwrap()
        .is_none());

    let mut mismatched = evaluator.clone();
    mismatched.flat_context.level_params.ring_relation_mode =
        akita_types::RingRelationMode::QuotientLift;
    assert!(
        super::relation_evaluation::evaluate_relation_at_point::<MixedF, MixedF>(
            &mismatched,
            &point,
            &setup,
            alpha,
        )
        .is_err()
    );

    for malformed in [&point[..point.len() - 1], &[MixedF::one(); 128][..]] {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::relation_evaluation::evaluate_relation_at_point::<MixedF, MixedF>(
                &evaluator, malformed, &setup, alpha,
            )
        }));
        assert!(outcome.is_ok(), "malformed verifier input must not panic");
        assert!(outcome.unwrap().is_err());
    }
}

#[test]
fn malformed_relation_property_matrix_rejects_without_panicking() {
    const MODE: u8 = 1 << 0;
    const DIMENSION: u8 = 1 << 1;
    const ROW: u8 = 1 << 2;
    const POINT: u8 = 1 << 3;
    const SETUP: u8 = 1 << 4;

    for mode in [
        RingRelationMode::QuotientLift,
        RingRelationMode::ReducedEvaluation,
    ] {
        for mutation_mask in 1..(1 << 5) {
            let mut fixture = mixed_replay_fixture(mode);
            if mutation_mask & MODE != 0 {
                fixture.lp.ring_relation_mode = match mode {
                    RingRelationMode::QuotientLift => RingRelationMode::ReducedEvaluation,
                    RingRelationMode::ReducedEvaluation => RingRelationMode::QuotientLift,
                };
            }
            if mutation_mask & DIMENSION != 0 {
                let outer = fixture.lp.outer().matrix;
                fixture.lp.own_group_mut().profile.outer.matrix =
                    OuterCommitMatrixParams::new_unchecked(
                        outer.security_policy(),
                        outer.sis_table_key().table_digest,
                        outer.sis_modulus_profile(),
                        outer.output_rank(),
                        outer.input_width(),
                        outer.coeff_linf_bound(),
                        D_INNER,
                    );
            }
            if mutation_mask & ROW != 0 {
                fixture.row_coefficients.pop();
            }
            if mutation_mask & POINT != 0 {
                fixture.point.clear();
            }
            if mutation_mask & SETUP != 0 {
                fixture.setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 0,
                        max_num_batched_polys: 1,
                        num_field_elements: 1,
                        setup_seed: [9; 32].into(),
                    },
                    FlatMatrix::from_flat_data(vec![MixedF::one()]),
                );
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let evaluator = fixture.prepare_evaluator()?;
                evaluator
                    .eval_flat_at_point::<MixedF>(&fixture.point, &fixture.setup, fixture.alpha)
                    .map(|_| ())
            }));
            assert!(
                matches!(outcome, Ok(Err(_))),
                "{mode:?} mutation mask {mutation_mask:#07b} must reject without panicking"
            );
        }
    }
}
