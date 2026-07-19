use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::ExactPrefixTable;
use super::range_class_tables::{
    product_coefficients, FoldedProductPairTable, OrderedProductPairCoefficients, ProductNodeTable,
};
use super::round_accumulation::accumulate_equality_weighted_round;
use super::MAX_TREE_STAGE_Q_DEGREE;
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, EqFactoredUniPoly};
use akita_types::DigitRangePlan;

enum ProductValues<E: FieldCore, const LANES: usize> {
    Compact {
        source: CompactDigitSource,
        nodes: ProductNodeTable<E, LANES>,
        pair_coefficients: OrderedProductPairCoefficients<E>,
    },
    Folded(ExactPrefixTable<[E; LANES]>),
}

fn accumulate_round<E: FieldCore + HasUnreducedOps, const LANES: usize>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    padding: [E; LANES],
    pair_at: impl Fn(usize) -> ([E; LANES], [E; LANES]) + Sync,
    arity: usize,
    parent_weights: &[E],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let padding_coefficients = product_coefficients(padding, padding, arity, parent_weights);
    accumulate_equality_weighted_round(
        first,
        second,
        explicit_pair_count,
        |pair_index| {
            let (left, right) = pair_at(pair_index);
            product_coefficients(left, right, arity, parent_weights)
        },
        padding_coefficients,
    )
}

/// One eq-factored product substage over compact classes followed by folded field lanes.
pub(super) struct StreamingProductStage<E: FieldCore, const LANES: usize> {
    values: ProductValues<E, LANES>,
    parent_weights: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    arity: usize,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: FieldCore + FromPrimitiveInt, const LANES: usize> StreamingProductStage<E, LANES> {
    pub(super) fn new(
        source: CompactDigitSource,
        plan: DigitRangePlan,
        leaf_polynomials: &[Vec<E>],
        stage_index: usize,
        parent_weights: Vec<E>,
        equality_point: &[E],
        input_claim: E,
    ) -> Result<Self, AkitaError> {
        let arity = plan
            .product_stage_arities()
            .get(stage_index)
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        let expected_lanes = arity.checked_mul(parent_weights.len()).ok_or_else(|| {
            AkitaError::InvalidInput("range-product lane count overflow".to_string())
        })?;
        if LANES != expected_lanes {
            return Err(AkitaError::InvalidSize {
                expected: expected_lanes,
                actual: LANES,
            });
        }
        let nodes = {
            let _span = tracing::info_span!(
                "digit_range_build_node_table",
                stage_index,
                arity,
                lane_count = LANES,
            )
            .entered();
            ProductNodeTable::new(plan, leaf_polynomials, stage_index)?
        };
        let pair_coefficients = {
            let _span = tracing::info_span!(
                "digit_range_build_pair_coefficients",
                stage_index,
                arity,
                lane_count = LANES,
                class_count = plan.basis() / 2,
            )
            .entered();
            OrderedProductPairCoefficients::new(&nodes, plan.basis() / 2, arity, &parent_weights)
        };
        Ok(Self {
            values: ProductValues::Compact {
                source,
                nodes,
                pair_coefficients,
            },
            parent_weights,
            split_eq: GruenSplitEq::new(equality_point)?,
            input_claim,
            arity,
            num_rounds: equality_point.len(),
            rounds_completed: 0,
        })
    }

    pub(super) fn final_child_claims(&self) -> Vec<E> {
        let ProductValues::Folded(table) = &self.values else {
            panic!("product stage remained compact after its final round")
        };
        table
            .final_value()
            .expect("product stage was not fully folded")
            .to_vec()
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasOptimizedFold + HasUnreducedOps, const LANES: usize>
    EqFactoredSumcheckInstanceProver<E> for StreamingProductStage<E, LANES>
{
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.arity
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn current_linear_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<E> {
        let (first, second) = self.split_eq.remaining_eq_tables();
        let coefficients = match &self.values {
            ProductValues::Compact {
                source,
                pair_coefficients,
                ..
            } => {
                let _span = tracing::info_span!(
                    "digit_range_initial_round",
                    round = self.rounds_completed,
                    explicit_rows = source.live_len(),
                    kernel_strategy = "ordered-pair-coefficients",
                )
                .entered();
                accumulate_equality_weighted_round(
                    first,
                    second,
                    source.pair_count(),
                    |pair_index| {
                        pair_coefficients
                            .coefficients_by_pair_index(source.ordered_pair_index(pair_index))
                    },
                    pair_coefficients.coefficients_by_pair_index(0),
                )
            }
            ProductValues::Folded(table) => {
                let _span = tracing::info_span!(
                    "digit_range_later_round",
                    round = self.rounds_completed,
                    explicit_rows = table.explicit_len(),
                    domain_len = table.domain_len(),
                )
                .entered();
                accumulate_round(
                    first,
                    second,
                    table.explicit_len().div_ceil(2),
                    table.default_value(),
                    |pair_index| {
                        (
                            table.value_or_default(2 * pair_index),
                            table.value_or_default(2 * pair_index + 1),
                        )
                    },
                    self.arity,
                    &self.parent_weights,
                )
            }
        };
        EqFactoredUniPoly::from_q_coeffs(coefficients[..=self.arity].to_vec())
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: E) {
        self.split_eq.bind(challenge);
        let folded_from_compact = match &self.values {
            ProductValues::Compact { source, nodes, .. } => {
                let _span = tracing::info_span!(
                    "digit_range_materialize_folded_lanes",
                    round = self.rounds_completed,
                    explicit_rows = source.live_len(),
                    lane_count = LANES,
                )
                .entered();
                let folded_pairs = {
                    let _span = tracing::info_span!(
                        "digit_range_build_folded_pair_table",
                        round = self.rounds_completed,
                        class_count = source.class_count(),
                        lane_count = LANES,
                    )
                    .entered();
                    FoldedProductPairTable::new(nodes, source.class_count(), challenge)
                };
                let explicit = cfg_into_iter!(0..source.pair_count())
                    .map(|pair_index| {
                        folded_pairs.row_by_pair_index(source.ordered_pair_index(pair_index))
                    })
                    .collect();
                Some(
                    ExactPrefixTable::new(
                        source.domain_len() / 2,
                        explicit,
                        folded_pairs.row_by_pair_index(0),
                    )
                    .expect("compact source and Boolean domain were validated"),
                )
            }
            ProductValues::Folded(_) => None,
        };
        if let Some(table) = folded_from_compact {
            self.values = ProductValues::Folded(table);
        } else if let ProductValues::Folded(table) = &mut self.values {
            let _span = tracing::info_span!(
                "digit_range_fold_lanes",
                round = self.rounds_completed,
                explicit_rows = table.explicit_len(),
                domain_len = table.domain_len(),
                lane_count = LANES,
            )
            .entered();
            let fold_context = E::precompute_fold(challenge);
            table
                .fold_in_place(|left, right| {
                    std::array::from_fn(|lane| E::fold_one(&fold_context, left[lane], right[lane]))
                })
                .expect("validated exact-prefix product state can fold");
        }
        self.rounds_completed += 1;
    }
}
