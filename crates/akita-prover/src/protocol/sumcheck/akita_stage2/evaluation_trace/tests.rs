use super::*;

use akita_algebra::CyclotomicRing;
use akita_config::{proof_optimized::fp128, CommitmentConfig};
use akita_field::{Ext2, ExtField, FromPrimitiveInt};
use akita_types::{
    basis_weights_prefix, r_decomp_levels, relation_rhs_layout_for, relation_rhs_row_count,
    ring_opening_point_from_field, BasisMode, DigitRangePlan, EvaluationTraceInputs,
    FlatBooleanDomain, FpExtEncoding, OpeningClaimsLayout, PreparedOpeningPoint,
    RelationMatrixRowLayout, RelationRangeImagePlan, RingMultiplierOpeningPoint, WitnessLayout,
};

type Cfg = fp128::D128Full;
type F = fp128::Field;
const D: usize = Cfg::D;
const NUM_VARIABLES: usize = 16;

fn fold_prepared_trace_at_point<E: FieldCore>(
    mut trace: PreparedProverEvaluationTrace<E>,
    live_len: usize,
    coeff_count: usize,
    point: &[E],
) -> E {
    let coefficient_bits = coeff_count.trailing_zeros() as usize;
    let mut live_lanes = live_len / coeff_count;
    for &challenge in &point[..coefficient_bits] {
        trace.fold_coefficients(challenge);
    }
    for &challenge in &point[coefficient_bits..] {
        trace.fold_lanes(challenge);
        live_lanes = live_lanes.div_ceil(2);
    }
    assert_eq!(live_lanes, 1);
    trace.get(0, 0, 1)
}

fn materialize_semantic_trace_oracle<E: FieldCore>(
    weights: &EvaluationTraceWeights<E>,
    output_scale: E,
) -> Vec<E> {
    let mut table = vec![E::zero(); weights.physical_field_len];
    for term in &weights.terms {
        let block_weights = basis_weights_prefix(
            &term.block_opening_point,
            term.basis,
            term.group_block_count,
        )
        .unwrap();
        let block_stride = term.opening_digit_weights.len() * term.source_ring_dimension;
        for segment in &term.segments {
            for local_block in 0..segment.block_count {
                let global_block = segment.global_block_start + local_block;
                let block_start = segment.physical_coefficient_start + local_block * block_stride;
                for (digit, &digit_weight) in term.opening_digit_weights.iter().enumerate() {
                    let digit_start = block_start + digit * term.source_ring_dimension;
                    let factor = output_scale
                        * term.coefficient
                        * block_weights[global_block]
                        * digit_weight;
                    for (inner_coordinate, &inner_weight) in term.inner_trace.iter().enumerate() {
                        table[digit_start + inner_coordinate] += factor * inner_weight;
                    }
                }
            }
        }
    }
    table
}

fn assert_prepared_opening_support_matches_semantic_trace<E>(basis: BasisMode)
where
    E: FpExtEncoding<F> + ExtField<F> + FromPrimitiveInt,
{
    let opening_batch = OpeningClaimsLayout::new(NUM_VARIABLES, 2).unwrap();
    let level_params = Cfg::get_params_for_batched_commitment(&opening_batch).unwrap();
    let rhs_layout = relation_rhs_layout_for(
        &level_params,
        &opening_batch,
        RelationMatrixRowLayout::WithDBlock,
    )
    .unwrap();
    let witness_layout = WitnessLayout::new(
        &level_params,
        &opening_batch,
        2,
        relation_rhs_row_count(&rhs_layout),
        r_decomp_levels::<F>(level_params.log_basis_open),
    )
    .unwrap();
    let live_len = witness_layout.total_len() * D;
    let digit_witness_domain = FlatBooleanDomain::new(
        live_len,
        live_len.next_power_of_two().trailing_zeros() as usize,
    )
    .unwrap();
    let plan = RelationRangeImagePlan::new(
        digit_witness_domain,
        DigitRangePlan::new(1usize << level_params.log_basis_open).unwrap(),
        witness_layout,
        &opening_batch,
        level_params.role_dims(),
        D,
    )
    .unwrap();
    let group_params = level_params.group_params(&opening_batch, 0).unwrap();
    let alpha_variables = D.trailing_zeros() as usize;
    let base_outer_point = vec![F::zero(); NUM_VARIABLES - alpha_variables];
    let ring_opening_point = ring_opening_point_from_field(
        &base_outer_point,
        group_params.num_positions_per_block(),
        group_params.num_live_blocks(),
        basis,
    )
    .unwrap();
    let prepared_points = vec![PreparedOpeningPoint::from_parts(
        (0..NUM_VARIABLES)
            .map(|index| E::from_u64(17 + 2 * index as u64))
            .collect(),
        ring_opening_point.clone(),
        RingMultiplierOpeningPoint::from_base(&ring_opening_point),
        CyclotomicRing::<F, D>::one(),
    )];
    let claim_coefficients = vec![E::from_u64(41), E::from_u64(43)];
    let semantic_trace = build_evaluation_trace_weights::<F, E, D>(EvaluationTraceInputs {
        digit_witness_domain: plan.digit_witness_domain(),
        witness_layout: plan.witness_layout(),
        role_dims: plan.role_dims(),
        level_params: &level_params,
        opening_batch: &opening_batch,
        prepared_points: &prepared_points,
        claim_coefficients: &claim_coefficients,
        basis,
    })
    .unwrap();
    let output_scale = E::from_u64(47);
    let point = (0..digit_witness_domain.num_vars())
        .map(|index| E::from_u64(53 + 2 * index as u64))
        .collect::<Vec<_>>();
    let expected_table = materialize_semantic_trace_oracle(&semantic_trace, output_scale);

    for coeff_count in [D, D / 2, D / 4] {
        let prepared =
            PreparedProverEvaluationTrace::new(&semantic_trace, coeff_count, output_scale).unwrap();
        assert_eq!(prepared.materialize_dense(), expected_table,);
        let folded = fold_prepared_trace_at_point(prepared, live_len, coeff_count, &point);
        assert_eq!(
            folded,
            output_scale * semantic_trace.evaluate_at_point(&point).unwrap()
        );
    }
    for malformed_common_count in [0, 3, D * 2] {
        assert!(PreparedProverEvaluationTrace::new(
            &semantic_trace,
            malformed_common_count,
            output_scale,
        )
        .is_err());
    }
}

#[test]
fn prepared_opening_support_matches_semantic_trace_across_bases_and_extension() {
    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        assert_prepared_opening_support_matches_semantic_trace::<F>(basis);
        assert_prepared_opening_support_matches_semantic_trace::<Ext2<F>>(basis);
    }
}
