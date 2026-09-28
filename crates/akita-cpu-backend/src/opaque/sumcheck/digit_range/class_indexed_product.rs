//! Class-indexed product-subcheck prover and specialized coefficient kernels.

use super::class_indexed_state::ClassIndexedTableState;
use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::ExactPrefixTable;
use super::range_class_tables::{
    product_coefficients, FoldedProductPairTable, OrderedProductPairCoefficients, ProductNodeTable,
    SecondRoundProductQuartetCoefficients,
};
use super::round_accumulation::{
    accumulate_equality_weighted_round, accumulate_equality_weighted_values,
};
use super::{MAX_QUARTET_TABLE_CLASS_COUNT, MAX_TREE_STAGE_Q_DEGREE};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_sumcheck::EqFactoredSumcheckInstanceProver;
use akita_types::DigitRangePlan;
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
    cached_second_round_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
}

type ProductTableState<E, const LANES: usize> = ClassIndexedTableState<
    CompactProductState<E, LANES>,
    FirstChallengeFoldedProductState<E, LANES>,
    [E; LANES],
>;

/// Batched product of one pair at `X = 1` and the leading coefficient, for
/// arity-two parents.
#[inline(always)]
fn arity_two_product_values<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    parent_weights: &[E],
) -> [E; 2] {
    let mut values = [E::zero(); 2];
    for (parent_index, &weight) in parent_weights.iter().enumerate() {
        let lane = 2 * parent_index;
        let mut at_one = right[lane] * right[lane + 1];
        let mut leading = (right[lane] - left[lane]) * (right[lane + 1] - left[lane + 1]);
        if weight != E::one() {
            at_one *= weight;
            leading *= weight;
        }
        values[0] += at_one;
        values[1] += leading;
    }
    values
}

/// Batched product of one pair at `X = 1`, the leading coefficient, `X = -1`
/// and `X = 2`, for arity-four parents.
///
/// Each parent splits into two quadratic half-products. A half-product
/// `h(X) = (l0 + X s0)(l1 + X s1)` costs three multiplications for `h(0)`,
/// `h(1)` and its leading coefficient `h_inf`, and then `h(-1) = 2 (h(0) +
/// h_inf) - h(1)` and `h(2) = 2 (h(1) + h_inf) - h(0)` are additions. The four
/// product values cost one multiplication each, ten in all per parent.
#[inline(always)]
fn arity_four_product_values<E: Field, const LANES: usize>(
    left: [E; LANES],
    right: [E; LANES],
    parent_weights: &[E],
) -> [E; 4] {
    #[inline(always)]
    fn half_product<E: Field>(left: [E; 2], right: [E; 2], weight: E) -> [E; 4] {
        let mut at_zero = left[0] * left[1];
        let mut at_one = right[0] * right[1];
        let mut leading = (right[0] - left[0]) * (right[1] - left[1]);
        if weight != E::one() {
            at_zero *= weight;
            at_one *= weight;
            leading *= weight;
        }
        let zero_plus_leading = at_zero + leading;
        let one_plus_leading = at_one + leading;
        [
            at_one,
            leading,
            zero_plus_leading + zero_plus_leading - at_one,
            one_plus_leading + one_plus_leading - at_zero,
        ]
    }
    let mut values = [E::zero(); 4];
    for (parent_index, &weight) in parent_weights.iter().enumerate() {
        let lane = 4 * parent_index;
        let head = half_product(
            [left[lane], left[lane + 1]],
            [right[lane], right[lane + 1]],
            weight,
        );
        let tail = half_product(
            [left[lane + 2], left[lane + 3]],
            [right[lane + 2], right[lane + 3]],
            E::one(),
        );
        for ((value, head), tail) in values.iter_mut().zip(head).zip(tail) {
            *value += head * tail;
        }
    }
    values
}

/// Constants of the round interpolation, fixed at construction.
#[derive(Clone, Copy)]
struct RoundInterpolation<E> {
    inverse_two: E,
    inverse_three: E,
}

impl<E: Field + Ring> RoundInterpolation<E> {
    fn new() -> Self {
        Self {
            inverse_two: E::from_u64(2)
                .inverse()
                .expect("the extension field has odd characteristic"),
            inverse_three: E::from_u64(3)
                .inverse()
                .expect("the extension field has characteristic above three"),
        }
    }
}

/// Round coefficients of a materialized product round.
///
/// The round sums the pair products at `X = 1` and at infinity, plus `X = -1`
/// and `X = 2` for arity four, and recovers `q(0)` from the running claim
/// `(1 - tau) q(0) + tau q(1)`. When `tau = 1` the claim does not determine
/// `q(0)`, and the round falls back to full coefficient accumulation.
#[allow(clippy::too_many_arguments)]
fn accumulate_round<E: Field + Ring + Unreduced, const LANES: usize>(
    equality_prefix_weights: &[E],
    equality_suffix_weights: &[E],
    explicit_pair_count: usize,
    padding: [E; LANES],
    pair_at: impl Fn(usize) -> ([E; LANES], [E; LANES]) + Sync,
    arity: usize,
    parent_weights: &[E],
    claim: E,
    tau: E,
    interpolation: RoundInterpolation<E>,
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let Some(inverse_one_minus_tau) = (E::one() - tau).inverse() else {
        let padding_coefficients = product_coefficients(padding, padding, arity, parent_weights);
        return accumulate_equality_weighted_round(
            equality_prefix_weights,
            equality_suffix_weights,
            explicit_pair_count,
            |pair_index| {
                let (left, right) = pair_at(pair_index);
                product_coefficients(left, right, arity, parent_weights)
            },
            padding_coefficients,
        );
    };
    let constant_from = |at_one: E| (claim - tau * at_one) * inverse_one_minus_tau;
    match arity {
        2 => {
            let [at_one, leading] = accumulate_equality_weighted_values(
                equality_prefix_weights,
                equality_suffix_weights,
                explicit_pair_count,
                |pair_index| {
                    let (left, right) = pair_at(pair_index);
                    arity_two_product_values(left, right, parent_weights)
                },
                arity_two_product_values(padding, padding, parent_weights),
            );
            let constant = constant_from(at_one);
            [
                constant,
                at_one - constant - leading,
                leading,
                E::zero(),
                E::zero(),
            ]
        }
        4 => {
            let [at_one, leading, at_minus_one, at_two] = accumulate_equality_weighted_values(
                equality_prefix_weights,
                equality_suffix_weights,
                explicit_pair_count,
                |pair_index| {
                    let (left, right) = pair_at(pair_index);
                    arity_four_product_values(left, right, parent_weights)
                },
                arity_four_product_values(padding, padding, parent_weights),
            );
            let constant = constant_from(at_one);
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
        }
        _ => unreachable!("validated range-product arity"),
    }
}

/// One eq-factored product substage that keeps compact classes through its first two rounds.
pub(super) struct ClassIndexedProductSubcheckProver<E: Field, const LANES: usize> {
    product_table: ProductTableState<E, LANES>,
    parent_weights: Vec<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    /// Running inner claim `(1 - tau) q(0) + tau q(1)` of the current round.
    claim: E,
    /// Inner polynomial of the last computed round, which advances `claim`.
    last_round_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
    interpolation: RoundInterpolation<E>,
    arity: usize,
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
            claim: input_claim,
            last_round_coefficients: [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
            interpolation: RoundInterpolation::new(),
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

impl<E: Field + Ring + Fold + Unreduced, const LANES: usize> EqFactoredSumcheckInstanceProver<E>
    for ClassIndexedProductSubcheckProver<E, LANES>
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

    fn current_tau(&self) -> E {
        self.split_eq.current_tau()
    }

    fn compute_round_eq_factored(&mut self, round: usize) -> OmittedConstantPoly<E> {
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
                    self.claim,
                    self.split_eq.current_tau(),
                    self.interpolation,
                )
            }
        };
        self.last_round_coefficients = coefficients;
        OmittedConstantPoly::from_q_coefficients(coefficients[..=self.arity].to_vec())
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        debug_assert_eq!(round, self.rounds_completed);
        self.claim = self
            .last_round_coefficients
            .iter()
            .rev()
            .fold(E::zero(), |acc, &coefficient| acc * challenge + coefficient);
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
                            self.claim,
                            self.split_eq.current_tau(),
                            self.interpolation,
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

    fn check_interpolated_rounds<E: Field + Ring + Unreduced, const LANES: usize>(
        arity: usize,
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
        let interpolation = RoundInterpolation::new();
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
            // `tau = 1` leaves `q(0)` undetermined and takes the full
            // coefficient path, which does not read the claim.
            let expected = round(E::zero(), E::one());
            let claim = expected[0] + tau * expected[1..].iter().copied().sum::<E>();
            assert_eq!(round(claim, tau), expected);
        }
    }

    fn check_all_arities<E: Field + Ring + Unreduced>() {
        check_interpolated_rounds::<E, 2>(2, &[E::one()]);
        check_interpolated_rounds::<E, 4>(4, &[E::one()]);
        check_interpolated_rounds::<E, 8>(4, &[E::one(), E::from_u64(19)]);
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
