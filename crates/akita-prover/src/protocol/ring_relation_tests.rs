use super::*;
use akita_challenges::{SparseChallenge, SparseChallengeConfig};
use akita_field::{Ext2, ExtField, Prime64Offset59};
use akita_types::{
    relation_rhs_coeff_len, BasisMode, CommitmentPayloadMode, OpenCommitMatrixParams,
    OpeningClaimsLayout, OpeningMethod, PreparedSubringCoefficientPackingPoint,
    RelationWitnessGeometry, SisModulusProfileId, SubringCoefficientPackingGeometry,
};

type F = Prime64Offset59;
type E = Ext2<F>;

fn fixture() -> (
    CommittedGroupParams,
    OpeningClaimsLayout,
    RingRelationInstance<F>,
    PreparedSubringCoefficientPackingPoint<E>,
) {
    let s = 64;
    let d_a = 256;
    let config = SparseChallengeConfig::production_for_ring_dim(s).unwrap();
    let mut params = CommittedGroupParams::params_only(
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
        128,
    );
    let opening_batch = OpeningClaimsLayout::new(11, 2).unwrap();
    let relation_geometry =
        RelationWitnessGeometry::for_level(&params, &opening_batch, <E as ExtField<F>>::EXT_DEGREE)
            .unwrap();
    let geometry = SubringCoefficientPackingGeometry::try_new(2, d_a, s).unwrap();
    let public_point = (0..11)
        .map(|index| E::from_u64(2 + index as u64))
        .collect::<Vec<_>>();
    let prepared_point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        6,
        4,
        11,
        &public_point,
    )
    .unwrap();
    let challenges = Challenges::from_sparse(
        (0..4)
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
        2,
        2,
    )
    .unwrap();
    let relation = RingRelationInstance::new(
        vec![RingRelationGroupOpening::coefficient_packing(
            akita_types::CoefficientPackingChallenges::new(geometry, challenges).unwrap(),
        )],
        <E as ExtField<F>>::EXT_DEGREE,
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
    (params, opening_batch, relation, prepared_point)
}

#[test]
fn prepared_relation_group_rejects_stale_shape_and_claims() {
    let (params, opening_batch, relation, point) = fixture();
    let valid = vec![PreparedRelationGroup {
        kind: OpeningFamily::SubringCoefficientPacking(point),
        scalar_openings: vec![E::from_u64(7), E::from_u64(11)],
    }];
    validate_prepared_relation_groups(&valid, &params, &opening_batch, &relation).unwrap();

    let public_point = (0..11)
        .map(|index| E::from_u64(23 + index as u64))
        .collect::<Vec<_>>();
    for stale_point in [
        PreparedSubringCoefficientPackingPoint::new(
            valid[0].coefficient_packing_point().unwrap().geometry(),
            BasisMode::Lagrange,
            7,
            4,
            11,
            &public_point,
        )
        .unwrap(),
        PreparedSubringCoefficientPackingPoint::new(
            valid[0].coefficient_packing_point().unwrap().geometry(),
            BasisMode::Lagrange,
            6,
            8,
            11,
            &public_point,
        )
        .unwrap(),
    ] {
        let stale = vec![PreparedRelationGroup {
            kind: OpeningFamily::SubringCoefficientPacking(stale_point),
            scalar_openings: vec![E::from_u64(7), E::from_u64(11)],
        }];
        assert!(
            validate_prepared_relation_groups(&stale, &params, &opening_batch, &relation,).is_err()
        );
    }

    let wrong_geometry = SubringCoefficientPackingGeometry::try_new(2, 128, 64).unwrap();
    let point = (0..11)
        .map(|index| E::from_u64(13 + index as u64))
        .collect::<Vec<_>>();
    let wrong_point = PreparedSubringCoefficientPackingPoint::new(
        wrong_geometry,
        BasisMode::Lagrange,
        16,
        4,
        11,
        &point,
    )
    .unwrap();
    let stale = vec![PreparedRelationGroup {
        kind: OpeningFamily::SubringCoefficientPacking(wrong_point),
        scalar_openings: vec![E::from_u64(7), E::from_u64(11)],
    }];
    assert!(validate_prepared_relation_groups(&stale, &params, &opening_batch, &relation).is_err());

    let missing_claim = vec![PreparedRelationGroup {
        kind: OpeningFamily::SubringCoefficientPacking(
            valid[0].coefficient_packing_point().unwrap().clone(),
        ),
        scalar_openings: vec![E::from_u64(7)],
    }];
    assert!(
        validate_prepared_relation_groups(&missing_claim, &params, &opening_batch, &relation,)
            .is_err()
    );
}
