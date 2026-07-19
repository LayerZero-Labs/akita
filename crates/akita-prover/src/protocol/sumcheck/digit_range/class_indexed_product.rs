//! Class-indexed product-subcheck prover and specialized coefficient kernels.

use super::class_indexed_state::ClassIndexedTableState;
use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::ExactPrefixTable;
use super::range_class_tables::{
    product_coefficients, FoldedProductPairTable, OrderedProductPairCoefficients, ProductNodeTable,
    SecondRoundProductQuartetCoefficients,
};
use super::round_accumulation::accumulate_equality_weighted_round;
use super::{MAX_QUARTET_TABLE_CLASS_COUNT, MAX_TREE_STAGE_Q_DEGREE};
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, EqFactoredUniPoly};
use akita_types::DigitRangePlan;

struct CompactProductState<E: FieldCore, const LANES: usize> {
    source: CompactDigitSource,
    nodes: ProductNodeTable<E, LANES>,
    pair_coefficients: OrderedProductPairCoefficients<E>,
}

struct FirstChallengeFoldedProductState<E: FieldCore, const LANES: usize> {
    source: CompactDigitSource,
    folded_pairs: FoldedProductPairTable<E, LANES>,
    cached_second_round_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
}

type ProductTableState<E, const LANES: usize> = ClassIndexedTableState<
    CompactProductState<E, LANES>,
    FirstChallengeFoldedProductState<E, LANES>,
    [E; LANES],
>;

fn accumulate_round<E: FieldCore + HasUnreducedOps, const LANES: usize>(
    equality_prefix_weights: &[E],
    equality_suffix_weights: &[E],
    explicit_pair_count: usize,
    padding: [E; LANES],
    pair_at: impl Fn(usize) -> ([E; LANES], [E; LANES]) + Sync,
    arity: usize,
    parent_weights: &[E],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let padding_coefficients = product_coefficients(padding, padding, arity, parent_weights);
    accumulate_equality_weighted_round(
        equality_prefix_weights,
        equality_suffix_weights,
        explicit_pair_count,
        |pair_index| {
            let (left, right) = pair_at(pair_index);
            product_coefficients(left, right, arity, parent_weights)
        },
        padding_coefficients,
    )
}

/// One eq-factored product substage that keeps compact classes through its first two rounds.
pub(super) struct ClassIndexedProductSubcheckProver<E: FieldCore, const LANES: usize> {
    product_table: ProductTableState<E, LANES>,
    parent_weights: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    arity: usize,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: FieldCore + FromPrimitiveInt, const LANES: usize>
    ClassIndexedProductSubcheckProver<E, LANES>
{
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
            product_table: ProductTableState::Compact(CompactProductState {
                source,
                nodes,
                pair_coefficients,
            }),
            parent_weights,
            split_eq: GruenSplitEq::new(equality_point)?,
            input_claim,
            arity,
            num_rounds: equality_point.len(),
            rounds_completed: 0,
        })
    }

    pub(super) fn final_child_claims(&self) -> Vec<E> {
        self.product_table
            .final_value()
            .expect("product stage was not fully folded")
            .to_vec()
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasOptimizedFold + HasUnreducedOps, const LANES: usize>
    EqFactoredSumcheckInstanceProver<E> for ClassIndexedProductSubcheckProver<E, LANES>
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

    fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<E> {
        debug_assert_eq!(round, self.rounds_completed);
        let (equality_prefix_weights, equality_suffix_weights) =
            self.split_eq.remaining_eq_tables();
        let coefficients = match &self.product_table {
            ProductTableState::Compact(CompactProductState {
                source,
                pair_coefficients,
                ..
            }) => {
                let _span = tracing::info_span!(
                    "digit_range_product_initial_round",
                    round = self.rounds_completed,
                    live_digits = source.live_len(),
                    explicit_pairs = source.pair_count(),
                    kernel_strategy = "ordered-pair-coefficients",
                )
                .entered();
                accumulate_equality_weighted_round(
                    equality_prefix_weights,
                    equality_suffix_weights,
                    source.pair_count(),
                    |pair_index| {
                        pair_coefficients
                            .coefficients_by_pair_index(source.ordered_pair_index(pair_index))
                    },
                    pair_coefficients.coefficients_by_pair_index(0),
                )
            }
            ProductTableState::FirstChallengeFolded(FirstChallengeFoldedProductState {
                cached_second_round_coefficients,
                ..
            }) => {
                let _span = tracing::info_span!(
                    "digit_range_product_initial_round",
                    round = self.rounds_completed,
                    kernel_strategy = "cached-second-round",
                )
                .entered();
                *cached_second_round_coefficients
            }
            ProductTableState::Materialized(table) => {
                let _span = tracing::info_span!(
                    "digit_range_product_materialized_round",
                    round = self.rounds_completed,
                    materialized_rows = table.explicit_len(),
                    domain_len = table.domain_len(),
                )
                .entered();
                accumulate_round(
                    equality_prefix_weights,
                    equality_suffix_weights,
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

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        debug_assert_eq!(round, self.rounds_completed);
        self.split_eq.bind(challenge);
        if self.rounds_completed == 0 && self.num_rounds >= 2 {
            let deferred = match &self.product_table {
                ProductTableState::Compact(CompactProductState { source, nodes, .. }) => {
                    let _span = tracing::info_span!(
                        "digit_range_prepare_deferred_second_round",
                        live_digits = source.live_len(),
                        lane_count = LANES,
                        kernel_strategy = if source.class_count() == MAX_QUARTET_TABLE_CLASS_COUNT {
                            "quartet-coefficient-table"
                        } else {
                            "factorized-pair-rescan"
                        },
                    )
                    .entered();
                    let folded_pairs = FoldedProductPairTable::new(nodes, challenge);
                    let (equality_prefix_weights, equality_suffix_weights) =
                        self.split_eq.remaining_eq_tables();
                    let coefficients = if source.class_count() == MAX_QUARTET_TABLE_CLASS_COUNT {
                        let _span = tracing::info_span!(
                            "digit_range_build_second_round_quartet_table",
                            class_count = source.class_count(),
                            lane_count = LANES,
                        )
                        .entered();
                        let quartets = SecondRoundProductQuartetCoefficients::new(
                            &folded_pairs,
                            self.arity,
                            &self.parent_weights,
                        );
                        accumulate_equality_weighted_round(
                            equality_prefix_weights,
                            equality_suffix_weights,
                            source.quartet_count(),
                            |quartet_index| {
                                let (left_pair, right_pair) =
                                    source.ordered_pair_indices_for_quartet(quartet_index);
                                quartets.coefficients_by_pair_indices(left_pair, right_pair)
                            },
                            quartets.coefficients_by_pair_indices(0, 0),
                        )
                    } else {
                        accumulate_round(
                            equality_prefix_weights,
                            equality_suffix_weights,
                            source.quartet_count(),
                            folded_pairs.row_by_pair_index(0),
                            |quartet_index| {
                                let (left_pair, right_pair) =
                                    source.ordered_pair_indices_for_quartet(quartet_index);
                                (
                                    folded_pairs.row_by_pair_index(left_pair),
                                    folded_pairs.row_by_pair_index(right_pair),
                                )
                            },
                            self.arity,
                            &self.parent_weights,
                        )
                    };
                    Some(ProductTableState::FirstChallengeFolded(
                        FirstChallengeFoldedProductState {
                            source: source.clone(),
                            folded_pairs,
                            cached_second_round_coefficients: coefficients,
                        },
                    ))
                }
                ProductTableState::FirstChallengeFolded(_) | ProductTableState::Materialized(_) => {
                    None
                }
            };
            if let Some(deferred) = deferred {
                self.product_table = deferred;
                self.rounds_completed += 1;
                return;
            }
        }

        if self.rounds_completed == 1 {
            let folded_after_two_rounds = match &self.product_table {
                ProductTableState::FirstChallengeFolded(FirstChallengeFoldedProductState {
                    source,
                    folded_pairs,
                    ..
                }) => {
                    let _span = tracing::info_span!(
                        "digit_range_materialize_after_two_rounds",
                        live_digits = source.live_len(),
                        explicit_quartets = source.quartet_count(),
                        lane_count = LANES,
                        kernel_strategy = "factorized-pair-rescan",
                    )
                    .entered();
                    let fold_context = E::precompute_fold(challenge);
                    let explicit = cfg_into_iter!(0..source.quartet_count())
                        .map(|quartet_index| {
                            let (left_pair, right_pair) =
                                source.ordered_pair_indices_for_quartet(quartet_index);
                            let left = folded_pairs.row_by_pair_index(left_pair);
                            let right = folded_pairs.row_by_pair_index(right_pair);
                            std::array::from_fn(|lane| {
                                E::fold_one(&fold_context, left[lane], right[lane])
                            })
                        })
                        .collect();
                    let padding_pair = folded_pairs.row_by_pair_index(0);
                    let padding = std::array::from_fn(|lane| {
                        E::fold_one(&fold_context, padding_pair[lane], padding_pair[lane])
                    });
                    Some(
                        ExactPrefixTable::new(source.domain_len() / 4, explicit, padding)
                            .expect("compact source and Boolean domain were validated"),
                    )
                }
                ProductTableState::Compact(_) | ProductTableState::Materialized(_) => None,
            };
            if let Some(table) = folded_after_two_rounds {
                self.product_table = ProductTableState::Materialized(table);
                self.rounds_completed += 1;
                return;
            }
        }

        let folded_from_compact = match &self.product_table {
            ProductTableState::Compact(CompactProductState { source, nodes, .. }) => {
                let _span = tracing::info_span!(
                    "digit_range_materialize_folded_lanes",
                    round = self.rounds_completed,
                    live_digits = source.live_len(),
                    explicit_pairs = source.pair_count(),
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
                    FoldedProductPairTable::new(nodes, challenge)
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
            ProductTableState::FirstChallengeFolded(_) | ProductTableState::Materialized(_) => None,
        };
        if let Some(table) = folded_from_compact {
            self.product_table = ProductTableState::Materialized(table);
        } else if let ProductTableState::Materialized(table) = &mut self.product_table {
            let _span = tracing::info_span!(
                "digit_range_fold_lanes",
                round = self.rounds_completed,
                materialized_rows = table.explicit_len(),
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
