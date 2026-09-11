use super::*;
use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::scalar_powers;
use akita_challenges::SparseChallengeConfig;
use akita_types::{
    build_reduced_compression_relation_weights, gadget_row_scalars, AkitaSetupDescriptor,
    CommitmentPayloadMode, FlatMatrix, GroupCommitPhaseParams, GroupOpenPhaseParams,
    GroupOpeningPlan, InnerCommitMatrixParams, NegativeBinarySupport, OpenCommitMatrixParams,
    OpeningClaimsLayout, OuterCommitMatrixParams, PolynomialGroupLayout, RelationQuotientPlan,
    RelationRowFamily, RelationWitnessGeometry, RingRelationMode, SisModulusProfileId,
    WitnessLayout,
};
use jolt_field::{Ext2, ExtField, One, Prime64Offset59, Zero};

type F = Prime64Offset59;
type E = Ext2<F>;

fn certify_test_matrices(params: &mut akita_types::CommittedGroupParams) {
    let inner_bound = *akita_types::sis::inner_coeff_linf_bounds(
        params.inner().matrix.sis_modulus_profile(),
        u32::try_from(params.d_a()).unwrap(),
    )
    .first()
    .unwrap();
    let inner = params.inner().matrix;
    params.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().unwrap().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner_bound,
        inner.ring_dimension(),
    );
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        3,
        outer.ring_dimension(),
    );
}

fn retarget_outer_and_open(params: &mut akita_types::CommittedGroupParams, dimension: usize) {
    let d_a = params.d_a();
    let outer = params.outer().matrix;
    params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * (d_a / dimension),
        outer.coeff_linf_bound(),
        dimension,
    );
    let opening = params.open().matrix;
    params.open_matrix = OpenCommitMatrixParams::new_unchecked(
        opening.security_policy(),
        opening.sis_table_key().table_digest,
        opening.sis_modulus_profile(),
        opening.output_rank(),
        opening.input_width() * (d_a / dimension),
        opening.coeff_linf_bound(),
        dimension,
    );
}

#[allow(clippy::too_many_arguments)]
fn add_literal_recomposition(
    dense: &mut [E],
    span_start: usize,
    digit_dimension: usize,
    source_coefficients: usize,
    source_dimension: usize,
    source_row_start: usize,
    source_row_count: usize,
    row_weights: &[E],
    alpha: E,
    mutate_row: bool,
) {
    assert_eq!(source_coefficients, source_row_count * source_dimension);
    let alpha_powers = scalar_powers(alpha, source_dimension);
    for (bit, gadget) in gadget_row_scalars::<F>(F::MODULUS_BITS as usize, 1)
        .into_iter()
        .enumerate()
    {
        for source_row in 0..source_row_count {
            let row = if mutate_row {
                (source_row_start + source_row + 1) % row_weights.len()
            } else {
                source_row_start + source_row
            };
            let scalar = -(row_weights[row] * E::lift_base(gadget));
            for (coefficient, &alpha_power) in alpha_powers.iter().enumerate() {
                let physical = span_start
                    + bit * source_coefficients
                    + source_row * source_dimension
                    + coefficient;
                dense[physical] += scalar * alpha_power;
            }
        }
    }
    assert!(source_dimension.is_multiple_of(digit_dimension));
}

fn compression_span<'a>(
    layout: &'a WitnessLayout,
    family: &RelationRowFamily,
) -> (
    Option<usize>,
    usize,
    &'a akita_types::CompressionWitnessSpan,
) {
    match *family {
        RelationRowFamily::CompressionF {
            group_index,
            map_index,
            ..
        } => (
            Some(group_index),
            map_index,
            layout.compression_layers()[map_index]
                .f_spans()
                .iter()
                .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
                .unwrap(),
        ),
        RelationRowFamily::CompressionH { map_index, .. } => (
            None,
            map_index,
            layout.compression_layers()[map_index].h_span(),
        ),
        _ => panic!("not a compression row"),
    }
}

#[allow(clippy::too_many_arguments)]
fn literal_reduced_compression_table(
    params: &akita_types::CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    layout: &WitnessLayout,
    setup: &AkitaExpandedSetup<F>,
    tau1: &[E],
    alpha: E,
    physical_field_len: usize,
    mutate_successor_row: bool,
) -> Vec<E> {
    let relation_geometry =
        RelationWitnessGeometry::for_level(params, opening_batch, <E as ExtField<F>>::DEGREE)
            .unwrap();
    let relation_layout = relation_geometry.rhs_layout();
    let families = relation_layout.row_families().unwrap();
    let row_weights = EqPolynomial::evals_prefix(tau1, families.len()).unwrap();
    let mut dense = vec![E::zero(); physical_field_len];

    for relation_group_index in 0..relation_layout.groups.len() {
        let (group_index, plan) = relation_layout
            .group_compression_plan(relation_group_index)
            .unwrap();
        let source_rows = params
            .commitment_row_range(opening_batch, group_index)
            .unwrap();
        let span = layout.compression_layers()[0]
            .f_spans()
            .iter()
            .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
            .unwrap();
        add_literal_recomposition(
            &mut dense,
            span.range().start,
            span.map().ring_dimension(),
            plan.source_coefficients(),
            relation_layout.groups[relation_group_index].role_dims.d_b(),
            source_rows.start,
            source_rows.len(),
            &row_weights,
            alpha,
            false,
        );
    }
    let opening_row = families
        .iter()
        .position(|family| matches!(family, RelationRowFamily::Opening { .. }))
        .unwrap();
    let opening_plan = relation_layout.opening_compression_plan().unwrap();
    let opening_span = layout.compression_layers()[0].h_span();
    add_literal_recomposition(
        &mut dense,
        opening_span.range().start,
        opening_span.map().ring_dimension(),
        opening_plan.source_coefficients(),
        relation_layout.d_ring_dimension,
        opening_row,
        relation_layout.n_d,
        &row_weights,
        alpha,
        false,
    );

    let setup_coefficients = setup.shared_matrix().as_field_slice();
    let mut mutated_successor = false;
    for (row, family) in families.iter().enumerate() {
        if !matches!(
            family,
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        let (group_index, map_index, span) = compression_span(layout, family);
        let map = span.map();
        let powers = scalar_powers(alpha, map.ring_dimension());
        for column in 0..map.input_width() {
            for witness_coefficient in 0..map.ring_dimension() {
                let residue = (0..map.ring_dimension()).fold(E::zero(), |sum, map_coefficient| {
                    let exponent = map_coefficient + witness_coefficient;
                    let product = E::lift_base(
                        setup_coefficients[column * map.ring_dimension() + map_coefficient],
                    ) * powers[exponent % map.ring_dimension()];
                    if exponent < map.ring_dimension() {
                        sum + product
                    } else {
                        sum - product
                    }
                });
                dense[span.range().start + column * map.ring_dimension() + witness_coefficient] +=
                    row_weights[row] * residue;
            }
        }
        if map_index + 1 < akita_types::COMPRESSION_MAP_COUNT {
            let successor_layer = &layout.compression_layers()[map_index + 1];
            let successor = match group_index {
                Some(group_index) => successor_layer
                    .f_spans()
                    .iter()
                    .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
                    .unwrap(),
                None => successor_layer.h_span(),
            };
            let mutate_this_row = mutate_successor_row && !mutated_successor;
            add_literal_recomposition(
                &mut dense,
                successor.range().start,
                successor.map().ring_dimension(),
                map.output_coefficients(),
                map.ring_dimension(),
                row,
                1,
                &row_weights,
                alpha,
                mutate_this_row,
            );
            mutated_successor |= mutate_this_row;
        }
    }
    dense
}

fn compressed_reduced_fixture() -> (
    akita_types::CommittedGroupParams,
    OpeningClaimsLayout,
    WitnessLayout,
    AkitaExpandedSetup<F>,
    Vec<E>,
    E,
    usize,
) {
    let root_challenge = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
    let mut params = akita_types::CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        128,
        2,
        2,
        2,
        2,
        root_challenge,
    )
    .with_decomp(4, 8, 1, 1, 1)
    .unwrap();
    retarget_outer_and_open(&mut params, 64);
    params.payload_mode = CommitmentPayloadMode::Compressed;
    params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
    params.witness_chunk = akita_types::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 1,
    };
    params.own_group_mut().profile.group = PolynomialGroupLayout::new(10, 1);

    let frozen_challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
    let mut frozen = akita_types::CommittedGroupParams::params_only(
        SisModulusProfileId::Q64Offset59,
        64,
        2,
        2,
        2,
        2,
        frozen_challenge,
    )
    .with_decomp(4, 8, 1, 1, 1)
    .unwrap();
    certify_test_matrices(&mut frozen);
    let frozen_layout = PolynomialGroupLayout::new(9, 1);
    params
        .set_precommitted_groups(vec![GroupOpenPhaseParams {
            setup_natural_len: None,
            profile: GroupCommitPhaseParams::try_from_params(frozen_layout, &frozen).unwrap(),
            opening: GroupOpeningPlan::evaluation_trace(
                frozen.fold_challenge_config(),
                frozen.open().digits.log_basis,
                frozen.open().digits.num_digits,
                frozen.num_digits_fold(),
            ),
        }])
        .unwrap();
    let opening_batch =
        OpeningClaimsLayout::from_root_groups(&[frozen_layout], PolynomialGroupLayout::new(10, 1))
            .unwrap();
    let relation_geometry =
        RelationWitnessGeometry::for_level(&params, &opening_batch, <E as ExtField<F>>::DEGREE)
            .unwrap();
    let layout = WitnessLayout::new(
        &params,
        &opening_batch,
        &relation_geometry,
        2,
        RelationQuotientPlan::ReducedEvaluation,
    )
    .unwrap();
    let physical_field_len = layout.live_coeff_len().next_power_of_two();
    let largest_map = layout
        .compression_layers()
        .iter()
        .flat_map(|layer| {
            layer
                .f_spans()
                .iter()
                .map(|(_, span)| span)
                .chain(std::iter::once(layer.h_span()))
        })
        .map(|span| span.map().input_width() * span.map().ring_dimension())
        .max()
        .unwrap();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: largest_map,
            setup_seed: [23; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..largest_map)
                .map(|index| F::from_u64(1_001 + index as u64))
                .collect(),
        ),
    );
    let tau1 = (0..params.relation_row_index_num_vars(&opening_batch).unwrap())
        .map(|index| {
            E::from_base_slice(&[
                F::from_u64(31 + index as u64),
                F::from_u64(131 + index as u64),
            ])
        })
        .collect();
    let alpha = E::from_base_slice(&[F::from_u64(7), F::from_u64(17)]);
    (
        params,
        opening_batch,
        layout,
        setup,
        tau1,
        alpha,
        physical_field_len,
    )
}

#[test]
fn compressed_reduced_stage2_matches_literal_full_equation_and_rejects_mutations() {
    let (params, opening_batch, layout, setup, tau1, alpha, physical_field_len) =
        compressed_reduced_fixture();
    assert_eq!(
        layout.compression_layers().len(),
        akita_types::COMPRESSION_MAP_COUNT
    );
    assert_eq!(params.precommitted_groups().len() + 1, 2);
    assert_eq!(params.witness_chunk.num_chunks, 2);
    let program = build_reduced_compression_relation_weights::<F, E>(
        alpha,
        &params,
        &opening_batch,
        <E as ExtField<F>>::DEGREE,
        &tau1,
        &layout,
        64,
        physical_field_len,
    )
    .unwrap();
    let prepared = program;
    let support = NegativeBinarySupport::new(&layout, physical_field_len).unwrap();
    let point = (0..physical_field_len.trailing_zeros() as usize)
        .map(|index| {
            E::from_base_slice(&[
                F::from_u64(211 + index as u64),
                F::from_u64(311 + index as u64),
            ])
        })
        .collect::<Vec<_>>();
    let stage1_point = point
        .iter()
        .enumerate()
        .map(|(index, value)| *value + E::from_u64(401 + index as u64))
        .collect::<Vec<_>>();
    let witness_evaluation = E::from_base_slice(&[F::from_u64(13), F::from_u64(29)]);
    let binary_batching = E::from_base_slice(&[F::from_u64(19), F::from_u64(37)]);

    let dense = literal_reduced_compression_table(
        &params,
        &opening_batch,
        &layout,
        &setup,
        &tau1,
        alpha,
        physical_field_len,
        false,
    );
    let relation_weight = multilinear_eval(&dense, &point).unwrap();
    assert_eq!(
        prepared.evaluate_at_point(&setup, &point).unwrap(),
        relation_weight
    );
    let binary_weight = layout
        .negative_binary_support_intervals()
        .into_iter()
        .flatten()
        .fold(E::zero(), |sum, index| {
            sum + eq_eval_at_index(&stage1_point, index) * eq_eval_at_index(&point, index)
        });
    let expected = witness_evaluation * relation_weight
        + binary_batching * binary_weight * witness_evaluation * (witness_evaluation + E::one());
    let compression = Stage2CompressionOracle::ReducedEvaluation {
        weights: &prepared,
        support: &support,
        binary_batching,
    };
    assert_eq!(
        evaluate_compression_oracle(
            &compression,
            &setup,
            &stage1_point,
            &point,
            witness_evaluation,
        )
        .unwrap(),
        expected
    );

    let wrong_successor = literal_reduced_compression_table(
        &params,
        &opening_batch,
        &layout,
        &setup,
        &tau1,
        alpha,
        physical_field_len,
        true,
    );
    assert_ne!(
        multilinear_eval(&wrong_successor, &point).unwrap(),
        relation_weight
    );

    let mut reordered_tau1 = tau1.clone();
    reordered_tau1.swap(0, 1);
    let reordered = build_reduced_compression_relation_weights::<F, E>(
        alpha,
        &params,
        &opening_batch,
        <E as ExtField<F>>::DEGREE,
        &reordered_tau1,
        &layout,
        64,
        physical_field_len,
    )
    .unwrap();
    assert_ne!(
        reordered.evaluate_at_point(&setup, &point).unwrap(),
        relation_weight
    );

    let mut changed_setup_coefficients = setup.shared_matrix().as_field_slice().to_vec();
    changed_setup_coefficients[0] += F::one();
    let changed_setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: changed_setup_coefficients.len(),
            setup_seed: [23; 32].into(),
        },
        FlatMatrix::from_flat_data(changed_setup_coefficients),
    );
    assert_ne!(
        prepared.evaluate_at_point(&changed_setup, &point).unwrap(),
        relation_weight
    );

    let digit_index = layout.compression_layers()[0].f_spans()[0].1.range().start;
    let tampered_witness_evaluation = witness_evaluation + eq_eval_at_index(&point, digit_index);
    assert_ne!(
        evaluate_compression_oracle(
            &compression,
            &setup,
            &stage1_point,
            &point,
            tampered_witness_evaluation,
        )
        .unwrap(),
        expected
    );

    let mut wrong_mode = params.clone();
    wrong_mode.ring_relation_mode = RingRelationMode::QuotientLift;
    assert!(build_reduced_compression_relation_weights::<F, E>(
        alpha,
        &wrong_mode,
        &opening_batch,
        <E as ExtField<F>>::DEGREE,
        &tau1,
        &layout,
        64,
        physical_field_len,
    )
    .is_err());
    assert_eq!(
        evaluate_compression_oracle(
            &Stage2CompressionOracle::Raw,
            &setup,
            &stage1_point,
            &point,
            witness_evaluation,
        )
        .unwrap(),
        E::zero()
    );
}
