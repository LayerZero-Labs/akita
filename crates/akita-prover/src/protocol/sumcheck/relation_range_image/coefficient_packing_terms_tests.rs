use super::*;

use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_field::{
    Ext2, ExtField, FieldCore, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
};
use akita_types::{
    AkitaScheduleLookupKey, CommitmentPayloadMode, CommittedGroupParams, DigitRangePlan,
    OpenCommitMatrixParams, PolynomialGroupLayout, RelationAddressGeometry, RelationRangeImagePlan,
    RelationWitnessGeometry, SisModulusProfileId, WitnessLayout,
};

type F = Prime64Offset59;
type E = Ext2<F>;

#[allow(clippy::too_many_arguments)]
fn packing_plan(
    profile: SisModulusProfileId,
    d_a: usize,
    d_d: usize,
    challenge_subring_dimension: usize,
    extension_degree: usize,
    num_live_positions: usize,
    num_positions_per_block: usize,
    num_vars: usize,
) -> (
    CommittedGroupParams,
    OpeningClaimsLayout,
    RelationRangeImagePlan,
) {
    let mut params = CommittedGroupParams::params_only(
        profile,
        d_a,
        2,
        2,
        2,
        2,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(
            challenge_subring_dimension,
        )
        .unwrap(),
    )
    .with_decomp(num_positions_per_block, num_live_positions, 2, 2, 2)
    .unwrap();
    params.payload_mode = CommitmentPayloadMode::Raw;
    params.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension,
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
    let opening_batch = OpeningClaimsLayout::new(num_vars, 2).unwrap();
    let relation_geometry =
        RelationWitnessGeometry::for_level(&params, &opening_batch, extension_degree).unwrap();
    let witness_layout =
        WitnessLayout::new(&params, &opening_batch, &relation_geometry, 2, 2).unwrap();
    let relation_address_geometry = RelationAddressGeometry::for_relation(
        &relation_geometry,
        d_d,
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let relation_plan = RelationRangeImagePlan::new(
        relation_geometry,
        relation_address_geometry,
        DigitRangePlan::new(4).unwrap(),
        witness_layout,
        &opening_batch,
    )
    .unwrap();
    (params, opening_batch, relation_plan)
}

fn packing_fixture() -> (
    CommittedGroupParams,
    OpeningClaimsLayout,
    RelationRangeImagePlan,
    PreparedSubringCoefficientPackingPoint<E>,
    Vec<SubringCoefficientPackingPartials<F>>,
) {
    let (params, opening_batch, relation_plan) =
        packing_plan(SisModulusProfileId::Q64Offset59, 256, 128, 64, 2, 6, 4, 11);
    let geometry = SubringCoefficientPackingGeometry::try_new(2, 256, 64).unwrap();
    let point_values = (0..11)
        .map(|index| E::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let point =
        PreparedSubringCoefficientPackingPoint::new(geometry, 6, 4, 11, &point_values).unwrap();
    let partials = (0..2)
        .map(|claim| {
            let coordinates = (0..2 * geometry.partial_base_field_width())
                .map(|index| F::from_u64(((claim + index) % 2 + 1) as u64))
                .collect();
            SubringCoefficientPackingPartials::new(geometry, 2, coordinates).unwrap()
        })
        .collect();
    (params, opening_batch, relation_plan, point, partials)
}

fn independent_scalar_opening<Base, Extension>(
    point: &PreparedSubringCoefficientPackingPoint<Extension>,
    partials: &[SubringCoefficientPackingPartials<Base>],
    claim_coefficients: &[Extension],
) -> Extension
where
    Base: FieldCore,
    Extension: ExtField<Base>,
{
    let geometry = point.geometry();
    let mut opening = Extension::zero();
    for (claim, partial) in partials.iter().enumerate() {
        for block in 0..point.num_live_blocks() {
            let block_start = block * geometry.partial_base_field_width();
            for coefficient in 0..geometry.challenge_subring_dimension() {
                let coordinates = (0..geometry.extension_degree())
                    .map(|plane| {
                        partial.coordinates()[block_start
                            + plane * geometry.challenge_subring_dimension()
                            + coefficient]
                    })
                    .collect::<Vec<_>>();
                let packed = Extension::from_base_slice(&coordinates);
                opening += claim_coefficients[claim]
                    * point.live_block_weights()[block]
                    * point.tail_weights()[coefficient]
                    * packed;
            }
        }
    }
    opening
}

fn relation_row_point<Extension: FieldCore + FromPrimitiveInt>(
    relation_plan: &RelationRangeImagePlan,
) -> Vec<Extension> {
    (0..relation_plan.relation_row_index_num_vars().unwrap())
        .map(|index| Extension::from_u64(19 + index as u64))
        .collect()
}

fn assert_geometry_case<Base, Extension>(
    profile: SisModulusProfileId,
    d_a: usize,
    d_d: usize,
    challenge_subring_dimension: usize,
    num_live_positions: usize,
    num_positions_per_block: usize,
    num_vars: usize,
) where
    Base: FieldCore + CanonicalField + FromPrimitiveInt,
    Extension: ExtField<Base> + FromPrimitiveInt + LiftBase<Base>,
{
    let (params, opening_batch, relation_plan) = packing_plan(
        profile,
        d_a,
        d_d,
        challenge_subring_dimension,
        <Extension as ExtField<Base>>::EXT_DEGREE,
        num_live_positions,
        num_positions_per_block,
        num_vars,
    );
    let geometry = SubringCoefficientPackingGeometry::try_new(
        <Extension as ExtField<Base>>::EXT_DEGREE,
        d_a,
        challenge_subring_dimension,
    )
    .unwrap();
    let point_values = (0..num_vars)
        .map(|index| Extension::from_u64(31 + index as u64))
        .collect::<Vec<_>>();
    let point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        num_live_positions,
        num_positions_per_block,
        num_vars,
        &point_values,
    )
    .unwrap();
    let partials = (0..2)
        .map(|claim| {
            let coordinates = (0..point.num_live_blocks() * geometry.partial_base_field_width())
                .map(|index| Base::from_u64(1 + ((claim + index) % 3) as u64))
                .collect();
            SubringCoefficientPackingPartials::new(geometry, point.num_live_blocks(), coordinates)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let claim_coefficients = [Extension::from_u64(3), Extension::from_u64(5)];
    let claimed_scalar_opening = independent_scalar_opening(&point, &partials, &claim_coefficients);
    let tau1 = relation_row_point(&relation_plan);
    let prepared = prepare_coefficient_packing_linear_terms(CoefficientPackingLinearTermInputs {
        level_params: &params,
        opening_batch: &opening_batch,
        relation_plan: &relation_plan,
        group_index: 0,
        prepared_point: &point,
        partials_by_claim: &partials,
        claim_coefficients: &claim_coefficients,
        claimed_scalar_opening,
        alpha: Extension::from_u64(7),
        tau1: &tau1,
    })
    .unwrap();
    assert_eq!(prepared.geometry, geometry);
    assert_eq!(prepared.linear_terms.source_count(), 2);
    assert_eq!(
        relation_plan
            .relation_witness_geometry()
            .group_opening_method(0)
            .unwrap(),
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        }
    );
    assert_eq!(
        geometry.partial_base_field_width() / d_d,
        prepared.geometry.partial_base_field_width() / d_d
    );
}

#[test]
fn structured_terms_cover_four_planes_and_exact_overlap_without_width_dispatch() {
    assert_geometry_case::<Prime32Offset99, FpExt4<Prime32Offset99>>(
        SisModulusProfileId::Q32Offset99,
        256,
        128,
        64,
        6,
        4,
        11,
    );
    assert_geometry_case::<Prime64Offset59, Prime64Offset59>(
        SisModulusProfileId::Q64Offset59,
        256,
        128,
        256,
        2,
        4,
        9,
    );
}

#[test]
fn structured_terms_merge_in_authenticated_root_group_order() {
    type RootF = Prime128OffsetA7F7;
    type RootE = RootF;
    let precommitted_layout = PolynomialGroupLayout::unit_one_hot(16, 1, 256);
    let final_layout = PolynomialGroupLayout::unit_one_hot(32, 2, 256);
    let precommitted = fp128::OneHot::profile_without_precommitted_groups(precommitted_layout)
        .expect("independent precommitted profile");
    let mut params = fp128::OneHot::select_schedule_for_key(&AkitaScheduleLookupKey {
        final_group: final_layout,
        precommitteds: vec![precommitted, precommitted],
    })
    .expect("generated grouped schedule")
    .into_schedule()
    .root
    .params
    .final_group
    .commitment;
    params.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    params.fold_challenge_config =
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    for precommitted in &mut params.precommitted_groups {
        precommitted.opening.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
        precommitted.opening.fold_challenge_config =
            akita_challenges::SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    }
    let opening_batch =
        OpeningClaimsLayout::from_root_groups(&[precommitted_layout; 2], final_layout).unwrap();
    let relation_geometry = RelationWitnessGeometry::for_level(
        &params,
        &opening_batch,
        <RootE as ExtField<RootF>>::EXT_DEGREE,
    )
    .unwrap();
    let witness_layout = WitnessLayout::new(
        &params,
        &opening_batch,
        &relation_geometry,
        params.witness_chunk.num_chunks,
        2,
    )
    .unwrap();
    let d_d = params.role_dims().d_d();
    let relation_address_geometry = RelationAddressGeometry::for_relation(
        &relation_geometry,
        d_d,
        witness_layout.live_coeff_len(),
    )
    .unwrap();
    let relation_plan = RelationRangeImagePlan::new(
        relation_geometry,
        relation_address_geometry,
        DigitRangePlan::new(1usize << params.log_basis_open).unwrap(),
        witness_layout,
        &opening_batch,
    )
    .unwrap();
    let group_order = opening_batch.root_group_order().unwrap();
    assert_eq!(group_order, vec![2, 0, 1]);

    let global_claim_coefficients = (0..opening_batch.num_total_polynomials())
        .map(|claim| RootE::from_u64(3 + claim as u64))
        .collect::<Vec<_>>();
    let tau1 = relation_row_point(&relation_plan);
    let mut by_group = Vec::new();
    for &group_index in &group_order {
        let group_params = params
            .group_params_geometry(&opening_batch, group_index)
            .unwrap();
        let group_layout = opening_batch.group_layout(group_index).unwrap();
        let geometry = SubringCoefficientPackingGeometry::try_new(
            <RootE as ExtField<RootF>>::EXT_DEGREE,
            group_params.inner_commit_matrix_params().ring_dimension(),
            64,
        )
        .unwrap();
        let point_values = (0..group_layout.num_vars())
            .map(|index| RootE::from_u64(41 + 17 * group_index as u64 + index as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            group_params.num_live_ring_elements_per_claim(),
            group_params.num_positions_per_block(),
            group_layout.num_vars(),
            &point_values,
        )
        .unwrap();
        let partials = (0..group_layout.num_polynomials())
            .map(|claim| {
                let coordinates = (0..point.num_live_blocks()
                    * geometry.partial_base_field_width())
                    .map(|index| RootF::from_u64(1 + ((group_index + claim + index) % 3) as u64))
                    .collect();
                SubringCoefficientPackingPartials::new(
                    geometry,
                    point.num_live_blocks(),
                    coordinates,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let claim_range = opening_batch.root_group_claim_range(group_index).unwrap();
        let claim_coefficients = &global_claim_coefficients[claim_range];
        let claimed_scalar_opening =
            independent_scalar_opening(&point, &partials, claim_coefficients);
        let prepared =
            prepare_coefficient_packing_linear_terms(CoefficientPackingLinearTermInputs {
                level_params: &params,
                opening_batch: &opening_batch,
                relation_plan: &relation_plan,
                group_index,
                prepared_point: &point,
                partials_by_claim: &partials,
                claim_coefficients: &global_claim_coefficients,
                claimed_scalar_opening,
                alpha: RootE::from_u64(11),
                tau1: &tau1,
            })
            .unwrap();
        assert_eq!(prepared.group_index, group_index);
        by_group.push(prepared.linear_terms);
    }
    let expected = by_group
        .iter()
        .map(PreparedProverLinearTerms::materialize_dense)
        .reduce(|mut sum, group| {
            for (value, contribution) in sum.iter_mut().zip(group) {
                *value += contribution;
            }
            sum
        })
        .unwrap();
    let mut merged = by_group.remove(0);
    for group in by_group {
        merged.merge(group).unwrap();
    }
    assert_eq!(merged.materialize_dense(), expected);
}

#[test]
fn structured_terms_match_independent_dense_weights_and_scalar_opening() {
    let (params, opening_batch, relation_plan, point, partials) = packing_fixture();
    let witness_layout = relation_plan.witness_layout();
    let claim_coefficients = [E::from_u64(3), E::from_u64(5)];
    let alpha = E::from_u64(7);
    let tau1 = relation_row_point(&relation_plan);
    let consistency_row_weight =
        relation_row_weight(relation_plan.consistency_row_index(0).unwrap(), &tau1).unwrap();
    let scalar_opening_row_weight =
        relation_row_weight(relation_plan.scalar_opening_row_index().unwrap(), &tau1).unwrap();
    let claimed_scalar_opening = independent_scalar_opening(&point, &partials, &claim_coefficients);
    let prepared = prepare_coefficient_packing_linear_terms(CoefficientPackingLinearTermInputs {
        level_params: &params,
        opening_batch: &opening_batch,
        relation_plan: &relation_plan,
        group_index: 0,
        prepared_point: &point,
        partials_by_claim: &partials,
        claim_coefficients: &claim_coefficients,
        claimed_scalar_opening,
        alpha,
        tau1: &tau1,
    })
    .unwrap();
    assert_eq!(prepared.group_index, 0);
    assert_eq!(prepared.geometry, point.geometry());
    assert_eq!(prepared.linear_terms.source_count(), 2);

    let geometry = point.geometry();
    let d_a = geometry.a_ring_dimension();
    let d_d = params.role_dims().d_d();
    let depth_open = params.num_digits_open;
    let depth_witness = params.num_digits_inner;
    let depth_fold = params.num_digits_fold();
    let opening_gadget = gadget_row_scalars::<F>(depth_open, params.log_basis_open)
        .into_iter()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    let witness_gadget = gadget_row_scalars::<F>(depth_witness, params.log_basis_inner)
        .into_iter()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, params.log_basis_open)
        .into_iter()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    let basis = extension_basis::<F, E>(2).unwrap();
    let mut expected = vec![E::zero(); witness_layout.live_coeff_len()];
    for (claim, &claim_coefficient) in claim_coefficients.iter().enumerate() {
        for unit in witness_layout.units_for_group(0).unwrap() {
            for block in unit.global_block_range() {
                for (digit, &digit_weight) in opening_gadget.iter().enumerate() {
                    for (plane, &basis_element) in basis.iter().enumerate() {
                        for coefficient in 0..64 {
                            let index = unit
                                .e_coefficient_index(
                                    d_d,
                                    2,
                                    depth_open,
                                    claim,
                                    block,
                                    0,
                                    digit,
                                    plane * 64 + coefficient,
                                )
                                .unwrap();
                            expected[index] += scalar_opening_row_weight
                                * claim_coefficient
                                * point.live_block_weights()[block]
                                * digit_weight
                                * basis_element
                                * point.tail_weights()[coefficient];
                        }
                    }
                }
            }
        }
    }
    let alpha_powers = akita_algebra::ring::scalar_powers(alpha, 64);
    for unit in witness_layout.units_for_group(0).unwrap() {
        for position in 0..4 {
            for (witness_digit, &witness_weight) in witness_gadget.iter().enumerate() {
                for (fold_digit, &fold_weight) in fold_gadget.iter().enumerate() {
                    let factor = -(consistency_row_weight
                        * point.position_weights()[position]
                        * witness_weight
                        * fold_weight);
                    for (subring_index, &alpha_power) in alpha_powers.iter().enumerate() {
                        for low_index in 0..4 {
                            let coefficient = 4 * subring_index + low_index;
                            let index = unit
                                .z_coefficient_index(
                                    d_a,
                                    4,
                                    depth_witness,
                                    depth_fold,
                                    position,
                                    witness_digit,
                                    fold_digit,
                                    coefficient,
                                )
                                .unwrap();
                            expected[index] +=
                                factor * point.packing_weights()[low_index] * alpha_power;
                        }
                    }
                }
            }
        }
    }
    let dense_weights = prepared.linear_terms.materialize_dense();
    assert_eq!(dense_weights, expected);

    let opening_from_coordinates =
        independent_scalar_opening(&point, &partials, &claim_coefficients);
    assert_eq!(prepared.scalar_opening, opening_from_coordinates);
    assert_eq!(
        prepared.weighted_scalar_opening_claim,
        scalar_opening_row_weight * opening_from_coordinates
    );

    let mut digit_witness = vec![0i8; witness_layout.live_coeff_len()];
    for (claim, partial) in partials.iter().enumerate() {
        for block in 0..2 {
            let unit = witness_layout.unit_for_block(0, block).unwrap();
            let block_start = block * geometry.partial_base_field_width();
            for physical in 0..geometry.partial_base_field_width() {
                let index = unit
                    .e_coefficient_index(
                        d_d,
                        2,
                        depth_open,
                        claim,
                        block,
                        physical / d_d,
                        0,
                        physical % d_d,
                    )
                    .unwrap();
                digit_witness[index] = if partial.coordinates()[block_start + physical] == F::one()
                {
                    1
                } else {
                    2
                };
            }
        }
    }
    let direct_linear_claim = digit_witness
        .iter()
        .zip(&dense_weights)
        .fold(E::zero(), |sum, (&digit, &weight)| {
            sum + weight * E::from_i64(i64::from(digit))
        });
    assert_eq!(direct_linear_claim, prepared.weighted_scalar_opening_claim);

    let padded_len = witness_layout.live_coeff_len().next_power_of_two();
    let stage2_point = (0..padded_len.trailing_zeros() as usize)
        .map(|index| E::from_u64((index + 17) as u64))
        .collect::<Vec<_>>();
    let mut padded_dense_weights = dense_weights;
    padded_dense_weights.resize(padded_len, E::zero());
    let dense_evaluation =
        akita_algebra::poly::multilinear_eval(&padded_dense_weights, &stage2_point)
            .expect("dense structured-weight evaluation");
    let mut folded = prepared.linear_terms;
    for &challenge in &stage2_point[..6] {
        folded.fold_coefficients(challenge);
    }
    for &challenge in &stage2_point[6..] {
        folded.fold_lanes(challenge);
    }
    assert_eq!(folded.final_value().unwrap(), dense_evaluation);
}

#[test]
fn structured_terms_reject_authority_and_shape_aliases() {
    let (params, opening_batch, relation_plan, point, partials) = packing_fixture();
    let claim_coefficients = [E::one(), E::one()];
    let claimed_scalar_opening = independent_scalar_opening(&point, &partials, &claim_coefficients);
    let tau1 = relation_row_point(&relation_plan);
    let make_inputs = |prepared_point| CoefficientPackingLinearTermInputs {
        level_params: &params,
        opening_batch: &opening_batch,
        relation_plan: &relation_plan,
        group_index: 0,
        prepared_point,
        partials_by_claim: &partials,
        claim_coefficients: &claim_coefficients,
        claimed_scalar_opening,
        alpha: E::from_u64(7),
        tau1: &tau1,
    };
    let wrong_geometry = SubringCoefficientPackingGeometry::try_new(2, 128, 64).unwrap();
    let wrong_point_values = vec![E::one(); 10];
    let wrong_point =
        PreparedSubringCoefficientPackingPoint::new(wrong_geometry, 8, 4, 10, &wrong_point_values)
            .unwrap();
    assert!(prepare_coefficient_packing_linear_terms(make_inputs(&wrong_point)).is_err());

    let short_claims = &partials[..1];
    let malformed = CoefficientPackingLinearTermInputs {
        partials_by_claim: short_claims,
        ..make_inputs(&point)
    };
    assert!(prepare_coefficient_packing_linear_terms(malformed).is_err());

    let short_coefficients = &claim_coefficients[..1];
    let malformed = CoefficientPackingLinearTermInputs {
        claim_coefficients: short_coefficients,
        ..make_inputs(&point)
    };
    assert!(prepare_coefficient_packing_linear_terms(malformed).is_err());

    let short_tau1 = &tau1[..tau1.len() - 1];
    let malformed = CoefficientPackingLinearTermInputs {
        tau1: short_tau1,
        ..make_inputs(&point)
    };
    assert!(prepare_coefficient_packing_linear_terms(malformed).is_err());

    let wrong_claim = CoefficientPackingLinearTermInputs {
        claimed_scalar_opening: claimed_scalar_opening + E::one(),
        ..make_inputs(&point)
    };
    assert!(prepare_coefficient_packing_linear_terms(wrong_claim).is_err());
}
