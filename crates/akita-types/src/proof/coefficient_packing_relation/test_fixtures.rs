//! Coefficient-packing relation fixtures shared by the akita-types tests and
//! the verifier's compact-factor tests.

use akita_challenges::{Challenges, SparseChallenge, SparseChallengeConfig};
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Field, Prime64Offset59, Ring, Zero};

use crate::{
    r_decomp_levels, relation_rhs_coeff_len, BasisMode, ChunkedWitnessCfg,
    CoefficientPackingChallenges, CommitmentPayloadMode, CommittedGroupParams, DigitRangePlan,
    FpExtEncoding, GroupCommitPhaseParams, GroupOpenPhaseParams, GroupOpeningPlan,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OpeningClaimsLayout, OpeningMethod,
    OuterCommitMatrixParams, PolynomialGroupLayout, PreparedSubringCoefficientPackingPoint,
    RelationAddressGeometry, RelationRangeImagePlan, RelationWitnessGeometry,
    RingRelationGroupOpening, RingRelationInstance, RingVec, SisModulusProfileId,
    SubringCoefficientPackingGeometry, WitnessLayout,
};

/// One coefficient-packing group with its relation, prepared point, claims
/// and row challenges.
pub struct CoefficientPackingFixture<Base: Field, Extension: Field> {
    pub params: CommittedGroupParams,
    pub opening_batch: OpeningClaimsLayout,
    pub relation_plan: RelationRangeImagePlan,
    pub relation: RingRelationInstance<Base>,
    pub prepared_point: PreparedSubringCoefficientPackingPoint<Extension>,
    pub claim_coefficients: Vec<Extension>,
    pub tau1: Vec<Extension>,
}

/// Build a single-group packing fixture.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn coefficient_packing_fixture<Base, Extension>(
    profile: SisModulusProfileId,
    d_a: usize,
    d_d: usize,
    s: usize,
    live_positions: usize,
    positions_per_block: usize,
    num_vars: usize,
    num_claims: usize,
    num_chunks: usize,
) -> CoefficientPackingFixture<Base, Extension>
where
    Base: Field + CanonicalEncoding + Ring,
    Extension: ExtField<Base> + FpExtEncoding<Base> + Ring + ExtField<Base>,
{
    let config = SparseChallengeConfig::production_for_ring_dim(s).unwrap();
    let mut params = CommittedGroupParams::params_only(profile, d_a, 2, 2, 2, 2, config)
        .with_decomp(positions_per_block, live_positions, 2, 2, 2)
        .unwrap();
    params.payload_mode = CommitmentPayloadMode::Raw;
    params.witness_chunk = ChunkedWitnessCfg {
        num_chunks,
        num_activated_levels: usize::from(num_chunks > 1),
    };
    params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: s,
    };
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        outer.coeff_linf_bound(),
        64,
    );
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
    let opening_batch =
        OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::new(num_vars, num_claims)])
            .unwrap();
    let extension_degree = <Extension as ExtField<Base>>::DEGREE;
    let relation_geometry =
        RelationWitnessGeometry::for_level(&params, &opening_batch, extension_degree).unwrap();
    let witness_layout = WitnessLayout::new(
        &params,
        &opening_batch,
        &relation_geometry,
        params.witness_chunk.num_chunks,
        crate::RelationQuotientPlan::quotient_lift(r_decomp_levels::<Base>(
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
    let geometry = SubringCoefficientPackingGeometry::try_new(extension_degree, d_a, s).unwrap();
    let point_values = (0..num_vars)
        .map(|index| Extension::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let prepared_point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        live_positions,
        positions_per_block,
        num_vars,
        &point_values,
    )
    .unwrap();
    let challenge_count = num_claims * prepared_point.num_live_blocks();
    let sparse = (0..challenge_count)
        .map(|challenge| SparseChallenge {
            positions: (0..config.weight())
                .map(|term| ((term + challenge) % s) as u32)
                .collect(),
            coeffs: (0..config.count_pm1)
                .map(|term| if term.is_multiple_of(2) { 1 } else { -1 })
                .chain((0..config.count_pm2).map(|_| 2))
                .collect(),
        })
        .collect();
    let challenges =
        Challenges::from_sparse(sparse, prepared_point.num_live_blocks(), num_claims).unwrap();
    let group_opening = RingRelationGroupOpening::coefficient_packing(
        CoefficientPackingChallenges::new(geometry, challenges).unwrap(),
    );
    let gamma = (0..num_claims)
        .map(|claim| Base::from_u64((claim + 3) as u64))
        .collect::<Vec<_>>();
    let mut row_coefficients = vec![Base::zero(); num_claims * d_a];
    for (claim, &coefficient) in gamma.iter().enumerate() {
        row_coefficients[claim * d_a] = coefficient;
    }
    let rhs = RingVec::from_coeffs(vec![
        Base::zero();
        relation_rhs_coeff_len(relation_geometry.rhs_layout())
            .unwrap()
    ]);
    let relation = RingRelationInstance::new(
        vec![group_opening],
        extension_degree,
        opening_batch.clone(),
        gamma,
        RingVec::from_coeffs_with_ring_dim(row_coefficients, d_a).unwrap(),
        rhs,
        params.role_dims(),
    )
    .unwrap();
    let claim_coefficients = (0..num_claims)
        .map(|claim| Extension::from_u64((claim + 5) as u64))
        .collect();
    let tau1 = (0..relation_plan.relation_row_index_num_vars().unwrap())
        .map(|index| Extension::from_u64((index + 7) as u64))
        .collect();
    CoefficientPackingFixture {
        params,
        opening_batch,
        relation_plan,
        relation,
        prepared_point,
        claim_coefficients,
        tau1,
    }
}

/// Two packing groups: a precommitted group with one claim (group 0) and the
/// final group with two claims (group 1). The relation orders them `[1, 0]`.
pub struct CoefficientPackingMultigroupFixture {
    pub params: CommittedGroupParams,
    pub opening_batch: OpeningClaimsLayout,
    pub relation_plan: RelationRangeImagePlan,
    pub relation: RingRelationInstance<Prime64Offset59>,
    /// Prepared points keyed by group index, in relation group order.
    pub points: Vec<(
        usize,
        PreparedSubringCoefficientPackingPoint<Ext2<Prime64Offset59>>,
    )>,
    pub claim_coefficients: Vec<Ext2<Prime64Offset59>>,
    pub tau1: Vec<Ext2<Prime64Offset59>>,
}

/// Build the two-group packing fixture.
#[must_use]
pub fn coefficient_packing_multigroup_fixture() -> CoefficientPackingMultigroupFixture {
    type F = Prime64Offset59;
    type E = Ext2<F>;

    let base = coefficient_packing_fixture::<F, E>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        64,
        6,
        4,
        11,
        2,
        1,
    );
    let mut params = base.params.clone();
    let mut frozen = params.with_decomp(4, 8, 2, 2, 2).unwrap();
    frozen.set_precommitted_groups(Vec::new()).unwrap();
    frozen.own_group_mut().profile.inner.digits.log_basis = 9;
    let a_bound = *crate::sis::inner_coeff_linf_bounds(
        frozen.inner().matrix.sis_modulus_profile(),
        u32::try_from(frozen.d_a()).expect("test ring dimension"),
    )
    .first()
    .expect("exact frozen A bounds");
    let inner = frozen.inner().matrix;
    frozen.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        a_bound,
        inner.ring_dimension(),
    );
    let outer = frozen.outer().matrix;
    frozen.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        3,
        outer.ring_dimension(),
    );
    let pre_layout = PolynomialGroupLayout::new(11, 1);
    params
        .set_precommitted_groups(vec![GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: GroupCommitPhaseParams::from_params_unchecked_for_test(pre_layout, &frozen),
            opening: GroupOpeningPlan {
                opening_method: OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension: 64,
                },
                fold_challenge_config: params.fold_challenge_config(),
                log_basis_open: params.open().digits.log_basis,
                num_digits_open: params.open().digits.num_digits,
                num_digits_fold: params.num_digits_fold(),
            },
        }])
        .unwrap();
    let final_layout = PolynomialGroupLayout::new(11, 2);
    let opening_batch = OpeningClaimsLayout::from_root_groups(&[pre_layout], final_layout).unwrap();
    let relation_geometry = RelationWitnessGeometry::for_level(&params, &opening_batch, 2).unwrap();
    let witness_layout = WitnessLayout::new(
        &params,
        &opening_batch,
        &relation_geometry,
        params.witness_chunk.num_chunks,
        crate::RelationQuotientPlan::quotient_lift(r_decomp_levels::<F>(
            params.open().digits.log_basis,
        ))
        .unwrap(),
    )
    .unwrap();
    let relation_address = RelationAddressGeometry::for_relation(
        &relation_geometry,
        params.role_dims().d_d(),
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let relation_plan = RelationRangeImagePlan::new(
        relation_geometry.clone(),
        relation_address,
        DigitRangePlan::new(4).unwrap(),
        witness_layout,
        &opening_batch,
    )
    .unwrap();

    let config = params.fold_challenge_config();
    let make_challenges = |claims: usize| {
        Challenges::from_sparse(
            (0..claims * params.blocks().live_blocks)
                .map(|challenge| SparseChallenge {
                    positions: (0..config.weight())
                        .map(|term| ((term + challenge) % 64) as u32)
                        .collect(),
                    coeffs: (0..config.count_pm1)
                        .map(|_| 1)
                        .chain((0..config.count_pm2).map(|_| 2))
                        .collect(),
                })
                .collect(),
            params.blocks().live_blocks,
            claims,
        )
        .unwrap()
    };
    let geometry = SubringCoefficientPackingGeometry::try_new(2, 256, 64).unwrap();
    let openings = vec![
        RingRelationGroupOpening::coefficient_packing(
            CoefficientPackingChallenges::new(geometry, make_challenges(1)).unwrap(),
        ),
        RingRelationGroupOpening::coefficient_packing(
            CoefficientPackingChallenges::new(geometry, make_challenges(2)).unwrap(),
        ),
    ];
    let total_claims = opening_batch.num_total_polynomials();
    let gamma = (0..total_claims)
        .map(|claim| F::from_u64((claim + 2) as u64))
        .collect::<Vec<_>>();
    let mut row_coefficients = vec![F::zero(); total_claims * 256];
    for (claim, &coefficient) in gamma.iter().enumerate() {
        row_coefficients[claim * 256] = coefficient;
    }
    let relation = RingRelationInstance::new(
        openings,
        2,
        opening_batch.clone(),
        gamma,
        RingVec::from_coeffs_with_ring_dim(row_coefficients, 256).unwrap(),
        RingVec::from_coeffs(vec![
            F::zero();
            relation_rhs_coeff_len(relation_geometry.rhs_layout())
                .unwrap()
        ]),
        params.role_dims(),
    )
    .unwrap();
    let claim_coefficients = (0..total_claims)
        .map(|claim| E::from_u64((claim + 11) as u64))
        .collect::<Vec<_>>();
    let tau1 = vec![E::from_u64(7); relation_plan.relation_row_index_num_vars().unwrap()];
    let mut points = Vec::new();
    for group in relation_plan.groups() {
        let group_index = group.group_index();
        let group_layout = opening_batch.group_layout(group_index).unwrap();
        let group_params = params
            .group_params_geometry(&opening_batch, group_index)
            .unwrap();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            group_params.num_live_ring_elements_per_claim(),
            group_params.num_positions_per_block(),
            group_layout.num_vars(),
            &vec![E::from_u64(5 + group_index as u64); group_layout.num_vars()],
        )
        .unwrap();
        points.push((group_index, point));
    }
    CoefficientPackingMultigroupFixture {
        params,
        opening_batch,
        relation_plan,
        relation,
        points,
        claim_coefficients,
        tau1,
    }
}
