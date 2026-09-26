use super::*;

use akita_algebra::poly::multilinear_eval;
use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_types::{
    coefficient_packing_relation_events, prepare_coefficient_packing_batch_semantics,
    r_decomp_levels, relation_rhs_coeff_len, validate_coefficient_packing_batch_groups,
    AkitaExpandedSetup, AkitaSetupDescriptor, BasisMode, CoefficientPackingBatchSemanticInputs,
    CoefficientPackingBatchSemantics, CoefficientPackingChallenges, CoefficientPackingStage2Source,
    CommitmentPayloadMode, DigitRangePlan, FlatMatrix, OpenCommitMatrixParams, OpeningClaimsLayout,
    OpeningFamily, OpeningMethod, PreparedSubringCoefficientPackingPoint, RelationAddressGeometry,
    RelationRangeImagePlan, RelationWeightEvent, RelationWitnessGeometry, RingRelationGroupOpening,
    RingRelationInstance, RingVec, SisModulusProfileId, SubringCoefficientPackingGeometry,
    WitnessLayout,
};
use jolt_field::{Ext2, One, Prime64Offset59, Ring, Zero};

type F = Prime64Offset59;
type E = Ext2<F>;

struct Fixture {
    params: akita_types::CommittedGroupParams,
    opening_batch: OpeningClaimsLayout,
    relation_plan: RelationRangeImagePlan,
    relation: RingRelationInstance<F>,
    prepared_point: PreparedSubringCoefficientPackingPoint<E>,
    claim_coefficients: Vec<E>,
    tau1: Vec<E>,
    relation_events: Vec<RelationWeightEvent<E>>,
    batch: CoefficientPackingBatchSemantics<E>,
}

fn fixture() -> Fixture {
    fixture_for_basis(BasisMode::Lagrange)
}

fn fixture_for_basis(basis: BasisMode) -> Fixture {
    let s = 64;
    let d_a = 256;
    let d_d = 128;
    let config = SparseChallengeConfig::production_for_ring_dim(s).unwrap();
    let mut params = akita_types::CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        d_a,
        2,
        2,
        2,
        2,
        config,
    )
    .with_decomp(4, 6, 2, 2, 2)
    .unwrap();
    params.payload_mode = CommitmentPayloadMode::Raw;
    params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: s,
    };
    let opening = params.open().matrix;
    params.open_matrix = OpenCommitMatrixParams::new_unchecked(
        opening.security_policy(),
        opening.sis_table_key().table_digest,
        opening.sis_modulus_profile(),
        opening.output_rank(),
        opening.input_width(),
        opening.coeff_linf_bound(),
        d_d,
    );
    let opening_batch = OpeningClaimsLayout::new(11, 2).unwrap();
    let relation_geometry = RelationWitnessGeometry::for_level(&params, &opening_batch, 2).unwrap();
    let witness_layout = WitnessLayout::new(
        &params,
        &opening_batch,
        &relation_geometry,
        1,
        akita_types::RelationQuotientPlan::quotient_lift(r_decomp_levels::<F>(
            params.open().digits.log_basis,
        ))
        .unwrap(),
    )
    .unwrap();
    let relation_address_geometry = RelationAddressGeometry::for_relation(
        &relation_geometry,
        d_d,
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let relation_plan = RelationRangeImagePlan::new(
        relation_geometry.clone(),
        relation_address_geometry,
        DigitRangePlan::new(4).unwrap(),
        witness_layout,
        &opening_batch,
    )
    .unwrap();
    let geometry = SubringCoefficientPackingGeometry::try_new(2, d_a, s).unwrap();
    let prepared_point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        basis,
        6,
        4,
        11,
        &(0..11)
            .map(|index| E::from_u64(2 + index as u64))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let challenge_count = 2 * prepared_point.num_live_blocks();
    let challenges = Challenges::from_sparse(
        (0..challenge_count)
            .map(|challenge| SparseChallenge {
                positions: (0..config.weight())
                    .map(|term| ((term + challenge) % s) as u32)
                    .collect(),
                coeffs: (0..config.count_pm1)
                    .map(|term| if term.is_multiple_of(2) { 1 } else { -1 })
                    .chain((0..config.count_pm2).map(|_| 2))
                    .collect(),
            })
            .collect(),
        prepared_point.num_live_blocks(),
        2,
    )
    .unwrap();
    let relation = RingRelationInstance::new(
        vec![RingRelationGroupOpening::coefficient_packing(
            CoefficientPackingChallenges::new(geometry, challenges).unwrap(),
        )],
        2,
        opening_batch.clone(),
        vec![F::from_u64(3), F::from_u64(5)],
        RingVec::from_coeffs_with_ring_dim(
            [F::from_u64(3), F::from_u64(5)]
                .into_iter()
                .flat_map(|coefficient| {
                    let mut ring = vec![F::zero(); d_a];
                    ring[0] = coefficient;
                    ring
                })
                .collect(),
            d_a,
        )
        .unwrap(),
        RingVec::from_coeffs(vec![
            F::zero();
            relation_rhs_coeff_len(relation_geometry.rhs_layout())
                .unwrap()
        ]),
        params.role_dims(),
    )
    .unwrap();
    let claim_coefficients = vec![E::from_u64(7), E::from_u64(11)];
    let tau1 = (0..relation_plan.relation_row_index_num_vars().unwrap())
        .map(|index| E::from_u64(13 + index as u64))
        .collect::<Vec<_>>();
    let prepared_points = [(0, &prepared_point)];
    let inputs = || CoefficientPackingBatchSemanticInputs {
        level_params: &params,
        opening_batch: &opening_batch,
        relation_plan: &relation_plan,
        relation: &relation,
        prepared_points: &prepared_points,
        alpha: E::from_u64(17),
        tau1: &tau1,
        claim_coefficients: &claim_coefficients,
    };
    let relation_events = validate_coefficient_packing_batch_groups(&inputs(), |group| {
        coefficient_packing_relation_events(&group)
    })
    .unwrap()
    .concat();
    let batch = prepare_coefficient_packing_batch_semantics(inputs()).unwrap();
    Fixture {
        params,
        opening_batch,
        relation_plan,
        relation,
        prepared_point,
        claim_coefficients,
        tau1,
        relation_events,
        batch,
    }
}

fn materialize_shared(semantics: &CoefficientPackingGroupSemantics<E>) -> Vec<E> {
    let terms = semantics.stage2_terms();
    let mut dense = vec![E::zero(); terms.physical_field_len()];
    for term in terms.terms() {
        let source = match term.source() {
            CoefficientPackingStage2Source::DirectOpening => terms.direct_opening_source(),
            CoefficientPackingStage2Source::PackingZ => terms.packing_z_source(),
        };
        for segment in &terms.segments()[term.segments()] {
            for (physical, source_index) in segment
                .physical_coefficients()
                .zip(segment.source_coefficients())
            {
                dense[physical] += term.factor() * source[source_index];
            }
        }
    }
    dense
}

#[test]
fn prover_adapter_preserves_shared_stage2_semantics() {
    let fixture = fixture();
    let semantics = &fixture.batch.groups()[0];
    let authenticated_opening = E::from_u64(19);
    let prepared =
        prepare_coefficient_packing_linear_terms(semantics.clone(), authenticated_opening).unwrap();
    assert_eq!(prepared.group_index, 0);
    assert_eq!(prepared.geometry, semantics.geometry());
    assert_eq!(prepared.linear_terms.source_count(), 2);
    assert_eq!(
        prepared.linear_terms.materialize_dense(),
        materialize_shared(semantics)
    );
    assert_eq!(
        prepared.weighted_scalar_opening_claim,
        semantics.stage2_terms().scalar_claim_weight() * authenticated_opening
    );
}

#[test]
fn prover_adapter_folds_to_shared_stage2_point_evaluation() {
    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        let fixture = fixture_for_basis(basis);
        let semantics = &fixture.batch.groups()[0];
        let mut prepared = prepare_coefficient_packing_linear_terms(semantics.clone(), E::zero())
            .unwrap()
            .linear_terms;
        let padded_len = semantics
            .stage2_terms()
            .physical_field_len()
            .next_power_of_two();
        let point = (0..padded_len.trailing_zeros())
            .map(|index| E::from_u64(101 + u64::from(index)))
            .collect::<Vec<_>>();
        let coefficient_bits = semantics
            .stage2_terms()
            .relation_coefficient_block_len()
            .trailing_zeros() as usize;
        for &challenge in &point[..coefficient_bits] {
            prepared.fold_coefficients(challenge);
        }
        let lane_point = &point[coefficient_bits..];
        let mut lanes = vec![E::zero(); 1 << lane_point.len()];
        let live_lanes = prepared.materialize_dense().len();
        prepared.drain_into_lane_weights(&mut lanes[..live_lanes], E::one());
        assert_eq!(
            multilinear_eval(&lanes, lane_point).unwrap(),
            semantics.stage2_terms().evaluate_at_point(&point).unwrap()
        );
    }
}

#[test]
fn duplicate_packing_group_support_is_rejected() {
    let first = fixture_for_basis(BasisMode::Lagrange);
    let second = fixture_for_basis(BasisMode::Monomial);
    let mut combined = prepare_coefficient_packing_linear_terms(
        first.batch.into_groups().into_iter().next().unwrap(),
        E::zero(),
    )
    .unwrap()
    .linear_terms;
    let duplicate = prepare_coefficient_packing_linear_terms(
        second.batch.into_groups().into_iter().next().unwrap(),
        E::zero(),
    )
    .unwrap()
    .linear_terms;
    let source_count = combined.source_count();
    let materialized = combined.materialize_dense();
    assert!(matches!(
        combined.merge(duplicate),
        Err(AkitaError::InvalidSetup(_))
    ));
    assert_eq!(combined.source_count(), source_count);
    assert_eq!(combined.materialize_dense(), materialized);
}

#[test]
fn method_aware_relation_builder_uses_shared_packing_events_once() {
    use crate::opaque::relation_weights::{
        build_relation_lane_weights, RelationLaneWeightInputs, RelationSetupSource,
    };
    use akita_algebra::ring::scalar_powers;

    let fixture = fixture();
    let domain = fixture.relation_plan.digit_witness_domain();
    let opening_ring_dim = fixture.params.role_dims().d_d();
    let alpha = E::from_u64(17);
    // Packing weights must use the batch that also supplies the Stage 2 terms,
    // even if the separately supplied relation coefficients differ.
    let unrelated_coefficients = vec![E::from_u64(23); fixture.claim_coefficients.len()];
    let deferred = build_relation_lane_weights(RelationLaneWeightInputs {
        setup: RelationSetupSource::DeferredClaim,
        instance: &fixture.relation,
        alpha,
        level_params: &fixture.params,
        relation_row_point: &fixture.tau1,
        claim_coefficients: &unrelated_coefficients,
        opening_source_len: domain.domain_len() / opening_ring_dim,
        opening_ring_dim,
        relation_plan: &fixture.relation_plan,
        packing_semantics: Some(&fixture.batch),
        opening_points: OpeningFamily::SubringCoefficientPacking(&[(0, &fixture.prepared_point)]),
    })
    .unwrap();

    // Without setup terms, the shared packing events are the only
    // contributions to their lanes, added once.
    let block_len = fixture
        .relation_plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    let alpha_powers = scalar_powers(
        alpha,
        fixture
            .relation_events
            .iter()
            .map(|event| event.alpha_exponent_start() + event.physical_coefficients().len())
            .max()
            .unwrap(),
    );
    let mut shared_lanes = vec![None; deferred.lanes().len()];
    for event in &fixture.relation_events {
        let coefficients = event.physical_coefficients();
        for (offset, lane) in
            (coefficients.start / block_len..coefficients.end / block_len).enumerate()
        {
            *shared_lanes[lane].get_or_insert(E::zero()) +=
                event.scalar() * alpha_powers[event.alpha_exponent_start() + offset * block_len];
        }
    }
    assert!(shared_lanes.iter().any(Option::is_some));
    for (lane, expected) in shared_lanes.iter().enumerate() {
        if let Some(expected) = expected {
            assert_eq!(deferred.lanes()[lane], *expected);
        }
    }

    let units = fixture
        .relation_plan
        .witness_layout()
        .units_for_group(0)
        .unwrap()
        .collect::<Vec<_>>();
    for unit in &units {
        let z_range = unit.z_range();
        assert!(z_range.start.is_multiple_of(block_len) && z_range.end.is_multiple_of(block_len));
        assert!(
            deferred.lanes()[z_range.start / block_len..z_range.end / block_len]
                .iter()
                .all(Zero::is_zero)
        );
    }

    let setup_field_len = 1usize << 18;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_field_len,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_field_len)
                .map(|index| F::from_u64(1 + index as u64))
                .collect(),
        ),
    );
    let direct = build_relation_lane_weights(RelationLaneWeightInputs {
        setup: RelationSetupSource::Matrix(&setup),
        instance: &fixture.relation,
        alpha,
        level_params: &fixture.params,
        relation_row_point: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
        opening_source_len: domain.domain_len() / opening_ring_dim,
        opening_ring_dim,
        relation_plan: &fixture.relation_plan,
        packing_semantics: Some(&fixture.batch),
        opening_points: OpeningFamily::SubringCoefficientPacking(&[(0, &fixture.prepared_point)]),
    })
    .unwrap();
    // Every packed D column adds one setup term across its opening-ring lanes.
    let setup_e_lanes = units
        .iter()
        .flat_map(|unit| unit.e_range().start / block_len..unit.e_range().end / block_len)
        .filter(|&lane| direct.lanes()[lane] != deferred.lanes()[lane])
        .count();
    let expected_d_columns = fixture.opening_batch.num_total_polynomials()
        * fixture.prepared_point.num_live_blocks()
        * fixture.params.open().digits.num_digits
        * (fixture.prepared_point.geometry().partial_base_field_width() / opening_ring_dim);
    assert_eq!(
        setup_e_lanes,
        expected_d_columns * (opening_ring_dim / block_len)
    );

    assert!(build_relation_lane_weights(RelationLaneWeightInputs {
        setup: RelationSetupSource::DeferredClaim,
        instance: &fixture.relation,
        alpha: E::from_u64(17),
        level_params: &fixture.params,
        relation_row_point: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
        opening_source_len: domain.domain_len() / opening_ring_dim,
        opening_ring_dim,
        relation_plan: &fixture.relation_plan,
        packing_semantics: None,
        opening_points: OpeningFamily::EvaluationTrace(()),
    })
    .is_err());
    assert!(
        prepare_coefficient_packing_batch_semantics(CoefficientPackingBatchSemanticInputs {
            level_params: &fixture.params,
            opening_batch: &fixture.opening_batch,
            relation_plan: &fixture.relation_plan,
            relation: &fixture.relation,
            prepared_points: &[(0, &fixture.prepared_point), (0, &fixture.prepared_point),],
            alpha: E::from_u64(17),
            tau1: &fixture.tau1,
            claim_coefficients: &fixture.claim_coefficients,
        })
        .is_err()
    );
}

#[test]
fn recursive_packing_phases_share_one_relation_authority() {
    use crate::opaque::RecursiveWitnessFlat;

    use crate::opaque::PreparedRelationGroupPublic;
    use crate::opaque::{fold_coefficient_packing_group, materialize_coefficient_packing_d_input};
    use crate::opaque::{
        CpuBackend, RootOpeningSource, SubringCoefficientPackingBatchKernel,
        SubringCoefficientPackingPlan,
    };
    use crate::protocol::validate_prepared_relation_groups;
    use akita_types::{coefficient_packing_scalar_opening, RingRelationGroupOpeningView};

    let fixture = fixture();
    let point = &fixture.prepared_point;
    let sources = (0..2)
        .map(|claim| {
            RecursiveWitnessFlat::from_i8_digits(
                (0..point.num_live_positions() * point.geometry().a_ring_dimension())
                    .map(|index| ((claim * 7 + index) % 5) as i8 - 2)
                    .collect(),
            )
            .align_for_commitment_ring_dim(point.geometry().a_ring_dimension())
            .unwrap()
        })
        .collect::<Vec<_>>();
    let source_refs = sources.iter().collect::<Vec<_>>();
    let batch =
        <RecursiveWitnessFlat as RootOpeningSource<F, 256>>::opening_batch(&source_refs).unwrap();
    let partials = CpuBackend::<F, E>::for_arithmetic_tests()
        .coefficient_packing_partials_batch(None, batch, SubringCoefficientPackingPlan { point })
        .unwrap();
    let d_input = materialize_coefficient_packing_d_input::<F, 128>(
        &fixture.params,
        &fixture.opening_batch,
        fixture.relation_plan.relation_witness_geometry(),
        0,
        &partials,
    )
    .unwrap();
    assert_eq!(
        d_input.block_count(),
        2 * point.num_live_blocks() * (point.geometry().partial_base_field_width() / 128),
    );
    assert!(d_input
        .block_sizes()
        .iter()
        .all(|&planes| planes == fixture.params.open().digits.num_digits));

    let RingRelationGroupOpeningView::SubringCoefficientPacking {
        geometry,
        canonical_subring_challenges,
        ..
    } = fixture.relation.group_opening_view(0).unwrap()
    else {
        panic!("packing fixture changed method");
    };
    let product =
        fold_coefficient_packing_group(geometry, &partials, canonical_subring_challenges).unwrap();
    assert_eq!(product.geometry(), geometry);
    assert_eq!(
        product.reduced_base_field_coordinates().len(),
        geometry.partial_base_field_width(),
    );
    assert_eq!(
        product.quotient_high_half_base_field_coordinates().len(),
        product.reduced_base_field_coordinates().len(),
    );

    let scalar_openings = partials
        .iter()
        .map(|partial| {
            coefficient_packing_scalar_opening::<F, E>(
                geometry,
                point.num_live_blocks(),
                std::slice::from_ref(partial),
                &[E::one()],
                point.live_block_weights(),
                point.tail_weights(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let relation_groups = vec![PreparedRelationGroupPublic::new(
        akita_types::OpeningFamily::SubringCoefficientPacking(point.clone()),
        scalar_openings.clone(),
    )];
    validate_prepared_relation_groups(
        &relation_groups,
        &fixture.params,
        &fixture.opening_batch,
        &fixture.relation,
    )
    .unwrap();

    let semantics = &fixture.batch.groups()[0];
    let authenticated_opening = scalar_openings
        .iter()
        .zip(&fixture.claim_coefficients)
        .fold(E::zero(), |sum, (&opening, &coefficient)| {
            sum + opening * coefficient
        });
    let prepared =
        prepare_coefficient_packing_linear_terms(semantics.clone(), authenticated_opening).unwrap();
    assert_eq!(
        prepared.linear_terms.materialize_dense(),
        materialize_shared(semantics),
    );
    assert_eq!(
        prepared.weighted_scalar_opening_claim,
        semantics.stage2_terms().scalar_claim_weight() * authenticated_opening,
    );
}
