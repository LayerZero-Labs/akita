//! Class-indexed product-subcheck prover and specialized coefficient kernels.

use super::class_indexed_state::ClassIndexedTableState;
use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::ExactPrefixTable;
#[cfg(test)]
use super::range_class_tables::product_coefficients;
use super::range_class_tables::{
    FoldedProductPairTable, OrderedProductPairCoefficients, ProductNodeTable,
    SecondRoundProductQuartetCoefficients,
};
use super::round_accumulation::accumulate_equality_weighted_values;
use super::{MAX_QUARTET_TABLE_CLASS_COUNT, MAX_TREE_STAGE_Q_DEGREE};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_params::DigitRangePlan;
use akita_sumcheck::EqFactoredSumcheckInstanceProver;
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, Ring};
use jolt_field::{Fold, Unreduced};
use jolt_poly::OmittedConstantPoly;

struct CompactProductState<E: Field, const LANES: usize> {
    source: CompactDigitSource,
    nodes: ProductNodeTable<E, LANES>,
    pair_coefficients: OrderedProductPairCoefficients<E>,
}

struct FirstChallengeFoldedProductState<E: Field, const LANES: usize> {
    source: CompactDigitSource,
    folded_pairs: FoldedProductPairTable<E, LANES>,
    cached_second_round_coefficients: Option<[E; MAX_TREE_STAGE_Q_DEGREE + 1]>,
}

type ProductTableState<E, const LANES: usize> = ClassIndexedTableState<
    CompactProductState<E, LANES>,
    FirstChallengeFoldedProductState<E, LANES>,
    [E; LANES],
>;

#[derive(Clone, Copy)]
pub(super) enum ProductArity {
    Two,
    Four,
}

impl ProductArity {
    pub(super) fn new(arity: usize) -> Option<Self> {
        match arity {
            2 => Some(Self::Two),
            4 => Some(Self::Four),
            _ => None,
        }
    }

    pub(super) fn degree(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Four => 4,
        }
    }
}

#[derive(Clone, Copy)]
struct ParentWeights<'a, E> {
    unweighted_parent_count: usize,
    weighted: &'a [E],
}

impl<'a, E: Field> ParentWeights<'a, E> {
    fn new(weights: &'a [E]) -> Self {
        let unweighted_parent_count = weights
            .iter()
            .position(|&weight| weight != E::one())
            .unwrap_or(weights.len());
        Self {
            unweighted_parent_count,
            weighted: &weights[unweighted_parent_count..],
        }
    }
}

#[inline(always)]
fn accumulate_parent_values<E: Field, const VALUES: usize>(
    parent_weights: ParentWeights<'_, E>,
    parent_values: impl Fn(usize) -> [E; VALUES],
) -> [E; VALUES] {
    let mut values = [E::zero(); VALUES];
    for parent_index in 0..parent_weights.unweighted_parent_count {
        for (value, parent_value) in values.iter_mut().zip(parent_values(parent_index)) {
            *value += parent_value;
        }
    }
    for (weighted_index, &weight) in parent_weights.weighted.iter().enumerate() {
        let parent_index = parent_weights.unweighted_parent_count + weighted_index;
        for (value, parent_value) in values.iter_mut().zip(parent_values(parent_index)) {
            *value += weight * parent_value;
        }
    }
    values
}

/// Batched product of one pair at `X = 1` and infinity, for arity-two parents.
#[cfg(not(debug_assertions))]
#[inline(always)]
fn arity_two_product_values<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    parent_weights: ParentWeights<'_, E>,
) -> [E; 2] {
    accumulate_parent_values(parent_weights, |parent_index| {
        let lane = 2 * parent_index;
        [
            right[lane] * right[lane + 1],
            (right[lane] - left[lane]) * (right[lane + 1] - left[lane + 1]),
        ]
    })
}

/// The arity-two values with `X = 0` prepended.
#[inline(always)]
fn arity_two_product_values_with_zero<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    parent_weights: ParentWeights<'_, E>,
) -> [E; 3] {
    accumulate_parent_values(parent_weights, |parent_index| {
        let lane = 2 * parent_index;
        [
            left[lane] * left[lane + 1],
            right[lane] * right[lane + 1],
            (right[lane] - left[lane]) * (right[lane + 1] - left[lane + 1]),
        ]
    })
}

/// Values of a quadratic half-product at `0`, `1`, infinity, `-1`, and `2`.
#[inline(always)]
fn half_product_values<E: Field>(left: [E; 2], right: [E; 2]) -> [E; 5] {
    let at_zero = left[0] * left[1];
    let at_one = right[0] * right[1];
    let leading = (right[0] - left[0]) * (right[1] - left[1]);
    let zero_plus_leading = at_zero + leading;
    let one_plus_leading = at_one + leading;
    [
        at_zero,
        at_one,
        leading,
        zero_plus_leading + zero_plus_leading - at_one,
        one_plus_leading + one_plus_leading - at_zero,
    ]
}

/// Batched product of one pair at `X = 1`, infinity, `X = -1`, and `X = 2`,
/// for arity-four parents.
///
/// Each parent splits into two quadratic half-products. The two halves cost
/// six multiplications and the four combined values cost four more.
#[inline(always)]
#[cfg(not(debug_assertions))]
fn arity_four_product_values<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    parent_weights: ParentWeights<'_, E>,
) -> [E; 4] {
    accumulate_parent_values(parent_weights, |parent_index| {
        let lane = 4 * parent_index;
        let head =
            half_product_values([left[lane], left[lane + 1]], [right[lane], right[lane + 1]]);
        let tail = half_product_values(
            [left[lane + 2], left[lane + 3]],
            [right[lane + 2], right[lane + 3]],
        );
        [
            head[1] * tail[1],
            head[2] * tail[2],
            head[3] * tail[3],
            head[4] * tail[4],
        ]
    })
}

/// The arity-four values with `X = 0` prepended.
#[inline(always)]
fn arity_four_product_values_with_zero<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    parent_weights: ParentWeights<'_, E>,
) -> [E; 5] {
    accumulate_parent_values(parent_weights, |parent_index| {
        let lane = 4 * parent_index;
        let head =
            half_product_values([left[lane], left[lane + 1]], [right[lane], right[lane + 1]]);
        let tail = half_product_values(
            [left[lane + 2], left[lane + 3]],
            [right[lane + 2], right[lane + 3]],
        );
        std::array::from_fn(|index| head[index] * tail[index])
    })
}

/// Constants of the round interpolation, fixed at construction.
#[derive(Clone, Copy)]
struct RoundInterpolation<E> {
    inverse_two: E,
    inverse_three: E,
}

impl<E: Field + Ring> RoundInterpolation<E> {
    fn new() -> Option<Self> {
        Some(Self {
            inverse_two: E::from_u64(2).inverse()?,
            inverse_three: E::from_u64(3).inverse()?,
        })
    }
}

/// Round coefficients of a materialized product round.
///
/// The round sums the pair products at `X = 1` and infinity, plus `X = -1`
/// and `X = 2` for arity four. It normally recovers `q(0)` from the running
/// claim `(1 - tau) q(0) + tau q(1)`. Debug builds also sum `q(0)` directly
/// and check it; when `tau = 1`, all builds use the direct sum.
#[allow(clippy::too_many_arguments)]
fn accumulate_round<E: Field + Ring + Unreduced, const LANES: usize>(
    equality_prefix_weights: &[E],
    equality_suffix_weights: &[E],
    explicit_pair_count: usize,
    padding: [E; LANES],
    pair_at: impl Fn(usize) -> ([E; LANES], [E; LANES]) + Sync,
    arity: ProductArity,
    parent_weights: &[E],
    claim: E,
    tau: E,
    interpolation: RoundInterpolation<E>,
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let inverse_one_minus_tau = (E::one() - tau).inverse();
    let parent_weights = ParentWeights::new(parent_weights);
    let arity_two_coefficients = |constant: E, at_one: E, leading: E| {
        [
            constant,
            at_one - constant - leading,
            leading,
            E::zero(),
            E::zero(),
        ]
    };
    let arity_four_coefficients =
        |constant: E, at_one: E, leading: E, at_minus_one: E, at_two: E| {
            let known = constant + leading;
            // `odd_plus` is q1 + q2 + q3 and `odd_minus` is -q1 + q2 - q3.
            let odd_plus = at_one - known;
            let odd_minus = at_minus_one - known;
            let quadratic = (odd_plus + odd_minus) * interpolation.inverse_two;
            let linear_plus_cubic = (odd_plus - odd_minus) * interpolation.inverse_two;
            // q(2) = q0 + 2 q1 + 4 q2 + 8 q3 + 16 q4.
            let linear_plus_four_cubic =
                (at_two - constant - E::from_u64(4) * quadratic - E::from_u64(16) * leading)
                    * interpolation.inverse_two;
            let cubic = (linear_plus_four_cubic - linear_plus_cubic) * interpolation.inverse_three;
            [
                constant,
                linear_plus_cubic - cubic,
                quadratic,
                cubic,
                leading,
            ]
        };
    #[cfg(not(debug_assertions))]
    let arity_two_values = || {
        accumulate_equality_weighted_values(
            equality_prefix_weights,
            equality_suffix_weights,
            explicit_pair_count,
            |pair_index| {
                let (left, right) = pair_at(pair_index);
                arity_two_product_values(left, right, parent_weights)
            },
            arity_two_product_values(padding, padding, parent_weights),
        )
    };
    let arity_two_values_with_zero = || {
        accumulate_equality_weighted_values(
            equality_prefix_weights,
            equality_suffix_weights,
            explicit_pair_count,
            |pair_index| {
                let (left, right) = pair_at(pair_index);
                arity_two_product_values_with_zero(left, right, parent_weights)
            },
            arity_two_product_values_with_zero(padding, padding, parent_weights),
        )
    };
    #[cfg(not(debug_assertions))]
    let arity_four_values = || {
        accumulate_equality_weighted_values(
            equality_prefix_weights,
            equality_suffix_weights,
            explicit_pair_count,
            |pair_index| {
                let (left, right) = pair_at(pair_index);
                arity_four_product_values(left, right, parent_weights)
            },
            arity_four_product_values(padding, padding, parent_weights),
        )
    };
    let arity_four_values_with_zero = || {
        accumulate_equality_weighted_values(
            equality_prefix_weights,
            equality_suffix_weights,
            explicit_pair_count,
            |pair_index| {
                let (left, right) = pair_at(pair_index);
                arity_four_product_values_with_zero(left, right, parent_weights)
            },
            arity_four_product_values_with_zero(padding, padding, parent_weights),
        )
    };
    match arity {
        ProductArity::Two => match inverse_one_minus_tau {
            None => {
                let [at_zero, at_one, leading] = arity_two_values_with_zero();
                arity_two_coefficients(at_zero, at_one, leading)
            }
            Some(inverse) => {
                #[cfg(debug_assertions)]
                let [at_zero, at_one, leading] = arity_two_values_with_zero();
                #[cfg(not(debug_assertions))]
                let [at_one, leading] = arity_two_values();
                let constant = (claim - tau * at_one) * inverse;
                #[cfg(debug_assertions)]
                debug_assert_eq!(constant, at_zero);
                arity_two_coefficients(constant, at_one, leading)
            }
        },
        ProductArity::Four => match inverse_one_minus_tau {
            None => {
                let [at_zero, at_one, leading, at_minus_one, at_two] =
                    arity_four_values_with_zero();
                arity_four_coefficients(at_zero, at_one, leading, at_minus_one, at_two)
            }
            Some(inverse) => {
                #[cfg(debug_assertions)]
                let [at_zero, at_one, leading, at_minus_one, at_two] =
                    arity_four_values_with_zero();
                #[cfg(not(debug_assertions))]
                let [at_one, leading, at_minus_one, at_two] = arity_four_values();
                let constant = (claim - tau * at_one) * inverse;
                #[cfg(debug_assertions)]
                debug_assert_eq!(constant, at_zero);
                arity_four_coefficients(constant, at_one, leading, at_minus_one, at_two)
            }
        },
    }
}

/// One eq-factored product substage that keeps compact classes through its first two rounds.
pub(super) struct ClassIndexedProductSubcheckProver<E: Field, const LANES: usize> {
    product_table: ProductTableState<E, LANES>,
    parent_weights: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    interpolation: RoundInterpolation<E>,
    arity: ProductArity,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: Field + Ring, const LANES: usize> ClassIndexedProductSubcheckProver<E, LANES> {
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
            .ok_or_else(|| {
                AkitaError::Internal("digit-range product stage has no planned arity".into())
            })?;
        let arity = ProductArity::new(arity).ok_or_else(|| {
            AkitaError::Internal("digit-range product stage arity is unsupported".into())
        })?;
        let expected_lanes = arity
            .degree()
            .checked_mul(parent_weights.len())
            .ok_or_else(|| AkitaError::Internal("range-product lane count overflow".to_string()))?;
        if LANES != expected_lanes {
            return Err(AkitaError::Internal(format!(
                "range-product implementation lane count: expected {expected_lanes}, actual {LANES}"
            )));
        }
        let nodes = {
            let _span = tracing::info_span!(
                "digit_range_build_node_table",
                stage_index,
                arity = arity.degree(),
                lane_count = LANES,
            )
            .entered();
            ProductNodeTable::new(plan, leaf_polynomials, stage_index)?
        };
        let pair_coefficients = {
            let _span = tracing::info_span!(
                "digit_range_build_pair_coefficients",
                stage_index,
                arity = arity.degree(),
                lane_count = LANES,
                class_count = plan.basis() / 2,
            )
            .entered();
            OrderedProductPairCoefficients::new(&nodes, arity, &parent_weights)
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
            interpolation: RoundInterpolation::new().ok_or_else(|| {
                AkitaError::Internal(
                    "digit-range interpolation constants are not invertible in the field".into(),
                )
            })?,
            arity,
            num_rounds: equality_point.len(),
            rounds_completed: 0,
        })
    }

    pub(super) fn final_child_claims(&self) -> Result<Vec<E>, AkitaError> {
        self.product_table
            .final_value()
            .map(|claims| claims.to_vec())
            .ok_or_else(|| AkitaError::Internal("product stage was not fully folded".into()))
    }
}

impl<E: Field + Ring + Fold + Unreduced, const LANES: usize> EqFactoredSumcheckInstanceProver<E>
    for ClassIndexedProductSubcheckProver<E, LANES>
{
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.arity.degree()
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn current_tau(&self) -> E {
        self.split_eq.current_tau()
    }

    fn compute_round_eq_factored(&mut self, round: usize, claim: E) -> OmittedConstantPoly<E> {
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
                accumulate_equality_weighted_values(
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
                source,
                folded_pairs,
                cached_second_round_coefficients,
            }) => {
                if let Some(coefficients) = cached_second_round_coefficients {
                    let _span = tracing::info_span!(
                        "digit_range_product_initial_round",
                        round = self.rounds_completed,
                        kernel_strategy = "cached-second-round",
                    )
                    .entered();
                    *coefficients
                } else {
                    let _span = tracing::info_span!(
                        "digit_range_product_initial_round",
                        round = self.rounds_completed,
                        kernel_strategy = "factorized-pair-rescan",
                    )
                    .entered();
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
                        claim,
                        self.split_eq.current_tau(),
                        self.interpolation,
                    )
                }
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
                    claim,
                    self.split_eq.current_tau(),
                    self.interpolation,
                )
            }
        };
        OmittedConstantPoly::from_q_coefficients(coefficients[..=self.arity.degree()].to_vec())
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
                        let (equality_prefix_weights, equality_suffix_weights) =
                            self.split_eq.remaining_eq_tables();
                        Some(accumulate_equality_weighted_values(
                            equality_prefix_weights,
                            equality_suffix_weights,
                            source.quartet_count(),
                            |quartet_index| {
                                let (left_pair, right_pair) =
                                    source.ordered_pair_indices_for_quartet(quartet_index);
                                quartets.coefficients_by_pair_indices(left_pair, right_pair)
                            },
                            quartets.coefficients_by_pair_indices(0, 0),
                        ))
                    } else {
                        None
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
                    let fold_context = E::precompute(challenge);
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
            let fold_context = E::precompute(challenge);
            table
                .fold_in_place(|left, right| {
                    std::array::from_fn(|lane| E::fold_one(&fold_context, left[lane], right[lane]))
                })
                .expect("validated exact-prefix product state can fold");
        }
        self.rounds_completed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Ext2, Prime128Offset275, Prime64Offset59};

    #[test]
    fn final_child_claims_reject_every_unfinished_storage_phase() {
        use crate::sources::packed_digits::PackedSignedDigits;
        use akita_params::FlatBooleanDomain;
        use jolt_field::Zero;
        type F = Prime128Offset275;
        let plan = DigitRangePlan::new(16).unwrap();
        let source = CompactDigitSource::new(
            PackedSignedDigits::from_i8_digits_auto(vec![1; 16]),
            FlatBooleanDomain::new(16, 4).unwrap(),
            plan,
        )
        .unwrap();
        let mut prover = ClassIndexedProductSubcheckProver::<F, 2>::new(
            source,
            plan,
            &plan.leaf_coeffs::<F>(),
            0,
            vec![F::from_u64(1)],
            &[F::from_u64(7); 4],
            F::zero(),
        )
        .unwrap();
        for round in 0..4 {
            assert!(matches!(
                prover.final_child_claims(), Err(AkitaError::Internal(message))
                    if message == "product stage was not fully folded"
            ));
            prover.ingest_challenge(round, F::from_u64(11));
        }
        assert_eq!(prover.final_child_claims().unwrap().len(), 2);
        assert!(ProductArity::new(3).is_none());
    }

    fn check_interpolated_rounds<E: Field + Ring + Unreduced, const LANES: usize>(
        arity: ProductArity,
        parent_weights: &[E],
    ) {
        let first = [2, 3, 5, 7].map(E::from_u64);
        let second = [11, 13].map(E::from_u64);
        let lanes = |seed: usize| -> [E; LANES] {
            std::array::from_fn(|lane| E::from_u64((seed * 31 + lane * 7 + 3) as u64))
        };
        let pairs = (0..first.len() * second.len())
            .map(|pair_index| (lanes(2 * pair_index), lanes(2 * pair_index + 1)))
            .collect::<Vec<_>>();
        let padding = lanes(1000);
        let interpolation = RoundInterpolation::new().expect("test fields support interpolation");
        let tau = E::from_u64(17);
        for explicit_pair_count in 0..=pairs.len() {
            let round = |claim: E, tau: E| {
                accumulate_round(
                    &first,
                    &second,
                    explicit_pair_count,
                    padding,
                    |pair_index| pairs[pair_index],
                    arity,
                    parent_weights,
                    claim,
                    tau,
                    interpolation,
                )
            };
            let expected = accumulate_equality_weighted_values(
                &first,
                &second,
                explicit_pair_count,
                |pair_index| {
                    let (left, right) = pairs[pair_index];
                    product_coefficients(left, right, arity, parent_weights)
                },
                product_coefficients(padding, padding, arity, parent_weights),
            );
            assert_eq!(round(E::zero(), E::one()), expected);
            let claim = expected[0] + tau * expected[1..].iter().copied().sum::<E>();
            assert_eq!(round(claim, tau), expected);
        }
    }

    fn check_all_arities<E: Field + Ring + Unreduced>() {
        check_interpolated_rounds::<E, 2>(ProductArity::Two, &[E::one()]);
        check_interpolated_rounds::<E, 4>(ProductArity::Four, &[E::one()]);
        check_interpolated_rounds::<E, 8>(ProductArity::Four, &[E::one(), E::from_u64(19)]);
    }

    #[test]
    fn interpolated_rounds_match_full_coefficients_for_canonical_sums() {
        check_all_arities::<Prime128Offset275>();
    }

    #[test]
    fn interpolated_rounds_match_full_coefficients_for_delayed_sums() {
        check_all_arities::<Ext2<Prime64Offset59>>();
    }
}
