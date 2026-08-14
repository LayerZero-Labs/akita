use super::*;

use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use akita_field::{Ext2, Prime64Offset59};
use akita_types::{
    prepare_coefficient_packing_batch_semantics, r_decomp_levels, relation_rhs_coeff_len,
    AkitaExpandedSetup, AkitaSetupDescriptor, CoefficientPackingBatchSemanticInputs,
    CoefficientPackingBatchSemantics, CommitmentPayloadMode, DigitRangePlan, FlatMatrix,
    OpenCommitMatrixParams, OpeningClaimsLayout, OpeningMethod,
    PreparedSubringCoefficientPackingPoint, RelationAddressGeometry, RelationRangeImagePlan,
    RelationWitnessGeometry, RingRelationGroupOpening, RingRelationInstance, RingVec,
    SisModulusProfileId, SubringCoefficientPackingGeometry, WitnessLayout,
};

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
    batch: CoefficientPackingBatchSemantics<F, E>,
}

fn fixture() -> Fixture {
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
    params.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: s,
    };
    let opening = params.open_commit_matrix;
    params.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
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
        r_decomp_levels::<F>(params.log_basis_open),
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
        vec![RingRelationGroupOpening::subring_coefficient_packing(geometry, challenges).unwrap()],
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
        RingVec::from_coeffs(Vec::new()),
        params.role_dims(),
    )
    .unwrap();
    let claim_coefficients = vec![E::from_u64(7), E::from_u64(11)];
    let tau1 = (0..relation_plan.relation_row_index_num_vars().unwrap())
        .map(|index| E::from_u64(13 + index as u64))
        .collect::<Vec<_>>();
    let batch =
        prepare_coefficient_packing_batch_semantics(CoefficientPackingBatchSemanticInputs {
            level_params: &params,
            opening_batch: &opening_batch,
            relation_plan: &relation_plan,
            relation: &relation,
            prepared_points: &[(0, &prepared_point)],
            alpha: E::from_u64(17),
            tau1: &tau1,
            claim_coefficients: &claim_coefficients,
        })
        .unwrap();
    Fixture {
        params,
        opening_batch,
        relation_plan,
        relation,
        prepared_point,
        claim_coefficients,
        tau1,
        batch,
    }
}

fn rebuild_batch(
    fixture: &Fixture,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    claim_coefficients: &[E],
) -> CoefficientPackingBatchSemantics<F, E> {
    prepare_coefficient_packing_batch_semantics(CoefficientPackingBatchSemanticInputs {
        level_params: &fixture.params,
        opening_batch: &fixture.opening_batch,
        relation_plan: &fixture.relation_plan,
        relation,
        prepared_points: &[(0, &fixture.prepared_point)],
        alpha: E::from_u64(17),
        tau1,
        claim_coefficients,
    })
    .unwrap()
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
        prepare_coefficient_packing_linear_terms(semantics, authenticated_opening).unwrap();
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
fn method_aware_relation_builder_uses_shared_packing_events_once() {
    use crate::protocol::ring_switch::{
        build_relation_weight_events, RelationSetupSource, RelationWeightEventInputs,
    };

    let fixture = fixture();
    let domain = fixture.relation_plan.digit_witness_domain();
    let opening_ring_dim = fixture.params.role_dims().d_d();
    let events = build_relation_weight_events(RelationWeightEventInputs {
        setup: RelationSetupSource::DeferredClaim,
        instance: &fixture.relation,
        alpha: E::from_u64(17),
        level_params: &fixture.params,
        relation_row_point: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
        opening_source_len: domain.domain_len() / opening_ring_dim,
        opening_ring_dim,
        coefficient_packing_batch: Some(&fixture.batch),
    })
    .unwrap();

    let shared = fixture.batch.groups()[0].relation_events();
    let shared_ranges = shared
        .events()
        .iter()
        .map(|event| event.physical_coefficients())
        .collect::<Vec<_>>();
    let emitted_on_shared_ranges = events
        .events()
        .iter()
        .filter(|event| shared_ranges.contains(&event.physical_coefficients()))
        .map(|event| {
            (
                event.physical_coefficients(),
                event.alpha_exponent_start(),
                event.scalar(),
            )
        })
        .collect::<Vec<_>>();
    let expected = shared
        .events()
        .iter()
        .map(|event| {
            (
                event.physical_coefficients(),
                event.alpha_exponent_start(),
                event.scalar(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(emitted_on_shared_ranges, expected);

    for unit in fixture
        .relation_plan
        .witness_layout()
        .units_for_group(0)
        .unwrap()
    {
        assert!(events.events().iter().all(|event| {
            let range = event.physical_coefficients();
            range.end <= unit.z_range().start || range.start >= unit.z_range().end
        }));
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
    let direct_events = build_relation_weight_events(RelationWeightEventInputs {
        setup: RelationSetupSource::Matrix(&setup),
        instance: &fixture.relation,
        alpha: E::from_u64(17),
        level_params: &fixture.params,
        relation_row_point: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
        opening_source_len: domain.domain_len() / opening_ring_dim,
        opening_ring_dim,
        coefficient_packing_batch: Some(&fixture.batch),
    })
    .unwrap();
    let e_ranges = fixture
        .relation_plan
        .witness_layout()
        .units_for_group(0)
        .unwrap()
        .map(|unit| unit.e_range())
        .collect::<Vec<_>>();
    let setup_e_events = direct_events
        .events()
        .iter()
        .filter(|event| {
            event.contribution()
                == crate::protocol::ring_switch::RelationWeightContribution::SetupMatrix
                && e_ranges.iter().any(|range| {
                    let event_range = event.physical_coefficients();
                    event_range.start >= range.start && event_range.end <= range.end
                })
        })
        .count();
    let expected_d_columns = fixture.opening_batch.num_total_polynomials()
        * fixture.prepared_point.num_live_blocks()
        * fixture.params.num_digits_open
        * (fixture.prepared_point.geometry().partial_base_field_width() / opening_ring_dim);
    assert_eq!(setup_e_events, expected_d_columns);

    assert!(build_relation_weight_events(RelationWeightEventInputs {
        setup: RelationSetupSource::DeferredClaim,
        instance: &fixture.relation,
        alpha: E::from_u64(17),
        level_params: &fixture.params,
        relation_row_point: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
        opening_source_len: domain.domain_len() / opening_ring_dim,
        opening_ring_dim,
        coefficient_packing_batch: None,
    })
    .is_err());
    assert!(build_relation_weight_events(RelationWeightEventInputs {
        setup: RelationSetupSource::DeferredClaim,
        instance: &fixture.relation,
        alpha: E::from_u64(18),
        level_params: &fixture.params,
        relation_row_point: &fixture.tau1,
        claim_coefficients: &fixture.claim_coefficients,
        opening_source_len: domain.domain_len() / opening_ring_dim,
        opening_ring_dim,
        coefficient_packing_batch: Some(&fixture.batch),
    })
    .is_err());

    let mut wrong_tau1 = fixture.tau1.clone();
    wrong_tau1[0] += E::one();
    let wrong_tau_batch = rebuild_batch(
        &fixture,
        &fixture.relation,
        &wrong_tau1,
        &fixture.claim_coefficients,
    );
    let mut wrong_claims = fixture.claim_coefficients.clone();
    wrong_claims[0] += E::one();
    let wrong_claim_batch =
        rebuild_batch(&fixture, &fixture.relation, &fixture.tau1, &wrong_claims);
    let config = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    let changed_challenges = Challenges::from_sparse(
        (0..2 * fixture.prepared_point.num_live_blocks())
            .map(|challenge| SparseChallenge {
                positions: (0..config.weight())
                    .map(|term| ((term + challenge + 7) % 64) as u32)
                    .collect(),
                coeffs: (0..config.count_pm1)
                    .map(|term| if term.is_multiple_of(2) { 1 } else { -1 })
                    .chain((0..config.count_pm2).map(|_| 2))
                    .collect(),
            })
            .collect(),
        fixture.prepared_point.num_live_blocks(),
        2,
    )
    .unwrap();
    let changed_relation = RingRelationInstance::new(
        vec![RingRelationGroupOpening::subring_coefficient_packing(
            fixture.prepared_point.geometry(),
            changed_challenges,
        )
        .unwrap()],
        fixture.relation.extension_degree(),
        fixture.opening_batch.clone(),
        fixture.relation.gamma().to_vec(),
        fixture.relation.row_coefficient_rings().clone(),
        fixture.relation.rhs().clone(),
        fixture.relation.v().clone(),
        fixture.relation.role_dims(),
    )
    .unwrap();
    let wrong_relation_batch = rebuild_batch(
        &fixture,
        &changed_relation,
        &fixture.tau1,
        &fixture.claim_coefficients,
    );
    for wrong_batch in [&wrong_tau_batch, &wrong_claim_batch, &wrong_relation_batch] {
        assert!(build_relation_weight_events(RelationWeightEventInputs {
            setup: RelationSetupSource::DeferredClaim,
            instance: &fixture.relation,
            alpha: E::from_u64(17),
            level_params: &fixture.params,
            relation_row_point: &fixture.tau1,
            claim_coefficients: &fixture.claim_coefficients,
            opening_source_len: domain.domain_len() / opening_ring_dim,
            opening_ring_dim,
            coefficient_packing_batch: Some(wrong_batch),
        })
        .is_err());
    }
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
