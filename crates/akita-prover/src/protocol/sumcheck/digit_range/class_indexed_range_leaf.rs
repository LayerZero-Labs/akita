//! Class-indexed range-polynomial leaf prover.

use super::class_indexed_state::ClassIndexedTableState;
use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::ExactPrefixTable;
use super::range_class_tables::{
    FoldedRangeImagePairTable, OrderedRangePairCoefficients, SecondRoundRangeQuartetCoefficients,
};
use super::round_accumulation::accumulate_equality_weighted_round;
use super::{
    compose_small_poly_with_affine, MAX_QUARTET_TABLE_CLASS_COUNT, MAX_TREE_STAGE_Q_DEGREE,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, EqFactoredUniPoly};

struct CompactRangeLeafState<E: FieldCore> {
    source: CompactDigitSource,
    pair_coefficients: OrderedRangePairCoefficients<E>,
}

struct FirstChallengeFoldedRangeLeafState<E: FieldCore> {
    source: CompactDigitSource,
    folded_pairs: FoldedRangeImagePairTable<E>,
    cached_second_round_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
}

type RangeImageTableState<E> =
    ClassIndexedTableState<CompactRangeLeafState<E>, FirstChallengeFoldedRangeLeafState<E>, E>;

fn accumulate_round<E: FieldCore + HasUnreducedOps>(
    equality_prefix_weights: &[E],
    equality_suffix_weights: &[E],
    explicit_pair_count: usize,
    padding_range_image: E,
    pair_at: impl Fn(usize) -> (E, E) + Sync,
    polynomial_coefficients: &[E],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let padding_coefficients =
        compose_small_poly_with_affine(polynomial_coefficients, padding_range_image, E::zero());
    accumulate_equality_weighted_round(
        equality_prefix_weights,
        equality_suffix_weights,
        explicit_pair_count,
        |pair_index| {
            let (left, right) = pair_at(pair_index);
            compose_small_poly_with_affine(polynomial_coefficients, left, right - left)
        },
        padding_coefficients,
    )
}

/// Final equality-factored quartic over the virtual range-image table.
pub(super) struct ClassIndexedRangeLeafProver<E: FieldCore> {
    range_image: RangeImageTableState<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    polynomial_coefficients: Vec<E>,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: FieldCore + FromPrimitiveInt> ClassIndexedRangeLeafProver<E> {
    pub(super) fn new(
        source: CompactDigitSource,
        equality_point: &[E],
        input_claim: E,
        polynomial_coefficients: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if polynomial_coefficients.len() > MAX_TREE_STAGE_Q_DEGREE + 1 {
            return Err(AkitaError::InvalidSize {
                expected: MAX_TREE_STAGE_Q_DEGREE + 1,
                actual: polynomial_coefficients.len(),
            });
        }
        let pair_coefficients = {
            let _span = tracing::info_span!(
                "digit_range_build_pair_coefficients",
                arity = polynomial_coefficients.len().saturating_sub(1),
                lane_count = 1,
                class_count = source.class_count(),
            )
            .entered();
            OrderedRangePairCoefficients::new(source.class_count(), &polynomial_coefficients)
        };
        Ok(Self {
            range_image: RangeImageTableState::Compact(CompactRangeLeafState {
                source,
                pair_coefficients,
            }),
            split_eq: GruenSplitEq::new(equality_point)?,
            input_claim,
            polynomial_coefficients,
            num_rounds: equality_point.len(),
            rounds_completed: 0,
        })
    }

    pub(super) fn final_range_image_eval(&self) -> E {
        self.range_image
            .final_value()
            .expect("range-image leaf was not fully folded")
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasOptimizedFold + HasUnreducedOps>
    EqFactoredSumcheckInstanceProver<E> for ClassIndexedRangeLeafProver<E>
{
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.polynomial_coefficients.len().saturating_sub(1)
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
        let coefficients = match &self.range_image {
            RangeImageTableState::Compact(CompactRangeLeafState {
                source,
                pair_coefficients,
            }) => {
                let _span = tracing::info_span!(
                    "digit_range_leaf_initial_round",
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
            RangeImageTableState::FirstChallengeFolded(FirstChallengeFoldedRangeLeafState {
                cached_second_round_coefficients,
                ..
            }) => {
                let _span = tracing::info_span!(
                    "digit_range_leaf_initial_round",
                    round = self.rounds_completed,
                    kernel_strategy = "cached-second-round",
                )
                .entered();
                *cached_second_round_coefficients
            }
            RangeImageTableState::Materialized(table) => {
                let _span = tracing::info_span!(
                    "digit_range_leaf_materialized_round",
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
                    &self.polynomial_coefficients,
                )
            }
        };
        EqFactoredUniPoly::from_q_coeffs(coefficients[..=self.degree_bound()].to_vec())
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        debug_assert_eq!(round, self.rounds_completed);
        self.split_eq.bind(challenge);
        if self.rounds_completed == 0 && self.num_rounds >= 2 {
            let deferred = match &self.range_image {
                RangeImageTableState::Compact(CompactRangeLeafState { source, .. })
                    if source.class_count() == MAX_QUARTET_TABLE_CLASS_COUNT =>
                {
                    let _span = tracing::info_span!(
                        "digit_range_prepare_deferred_second_round",
                        live_digits = source.live_len(),
                        lane_count = 1,
                        kernel_strategy = "quartet-coefficient-table",
                    )
                    .entered();
                    let folded_pairs =
                        FoldedRangeImagePairTable::new(source.class_count(), challenge);
                    let (equality_prefix_weights, equality_suffix_weights) =
                        self.split_eq.remaining_eq_tables();
                    let _span = tracing::info_span!(
                        "digit_range_build_second_round_quartet_table",
                        class_count = source.class_count(),
                        lane_count = 1,
                    )
                    .entered();
                    let quartets = SecondRoundRangeQuartetCoefficients::new(
                        &folded_pairs,
                        &self.polynomial_coefficients,
                    );
                    let coefficients = accumulate_equality_weighted_round(
                        equality_prefix_weights,
                        equality_suffix_weights,
                        source.quartet_count(),
                        |quartet_index| {
                            let (left_pair, right_pair) =
                                source.ordered_pair_indices_for_quartet(quartet_index);
                            quartets.coefficients_by_pair_indices(left_pair, right_pair)
                        },
                        quartets.coefficients_by_pair_indices(0, 0),
                    );
                    Some(RangeImageTableState::FirstChallengeFolded(
                        FirstChallengeFoldedRangeLeafState {
                            source: source.clone(),
                            folded_pairs,
                            cached_second_round_coefficients: coefficients,
                        },
                    ))
                }
                RangeImageTableState::Compact(_)
                | RangeImageTableState::FirstChallengeFolded(_)
                | RangeImageTableState::Materialized(_) => None,
            };
            if let Some(deferred) = deferred {
                self.range_image = deferred;
                self.rounds_completed += 1;
                return;
            }
        }

        if self.rounds_completed == 1 {
            let folded_after_two_rounds = match &self.range_image {
                RangeImageTableState::FirstChallengeFolded(
                    FirstChallengeFoldedRangeLeafState {
                        source,
                        folded_pairs,
                        ..
                    },
                ) => {
                    let _span = tracing::info_span!(
                        "digit_range_materialize_after_two_rounds",
                        live_digits = source.live_len(),
                        explicit_quartets = source.quartet_count(),
                        lane_count = 1,
                        kernel_strategy = "factorized-pair-rescan",
                    )
                    .entered();
                    let fold_context = E::precompute_fold(challenge);
                    let explicit = cfg_into_iter!(0..source.quartet_count())
                        .map(|quartet_index| {
                            let (left_pair, right_pair) =
                                source.ordered_pair_indices_for_quartet(quartet_index);
                            let left = folded_pairs.value_by_pair_index(left_pair);
                            let right = folded_pairs.value_by_pair_index(right_pair);
                            E::fold_one(&fold_context, left, right)
                        })
                        .collect();
                    let padding_pair = folded_pairs.value_by_pair_index(0);
                    let padding = E::fold_one(&fold_context, padding_pair, padding_pair);
                    Some(
                        ExactPrefixTable::new(source.domain_len() / 4, explicit, padding)
                            .expect("compact source and Boolean domain were validated"),
                    )
                }
                RangeImageTableState::Compact(_) | RangeImageTableState::Materialized(_) => None,
            };
            if let Some(table) = folded_after_two_rounds {
                self.range_image = RangeImageTableState::Materialized(table);
                self.rounds_completed += 1;
                return;
            }
        }

        let folded_from_compact = match &self.range_image {
            RangeImageTableState::Compact(CompactRangeLeafState { source, .. }) => {
                let _span = tracing::info_span!(
                    "digit_range_materialize_range_image",
                    round = self.rounds_completed,
                    live_digits = source.live_len(),
                    explicit_pairs = source.pair_count(),
                )
                .entered();
                let folded_pairs = {
                    let _span = tracing::info_span!(
                        "digit_range_build_folded_pair_table",
                        round = self.rounds_completed,
                        class_count = source.class_count(),
                        lane_count = 1,
                    )
                    .entered();
                    FoldedRangeImagePairTable::new(source.class_count(), challenge)
                };
                let explicit = cfg_into_iter!(0..source.pair_count())
                    .map(|pair_index| {
                        folded_pairs.value_by_pair_index(source.ordered_pair_index(pair_index))
                    })
                    .collect();
                Some(
                    ExactPrefixTable::new(
                        source.domain_len() / 2,
                        explicit,
                        folded_pairs.value_by_pair_index(0),
                    )
                    .expect("compact source and Boolean domain were validated"),
                )
            }
            RangeImageTableState::FirstChallengeFolded(_)
            | RangeImageTableState::Materialized(_) => None,
        };
        if let Some(table) = folded_from_compact {
            self.range_image = RangeImageTableState::Materialized(table);
        } else if let RangeImageTableState::Materialized(table) = &mut self.range_image {
            let _span = tracing::info_span!(
                "digit_range_fold_range_image",
                round = self.rounds_completed,
                materialized_rows = table.explicit_len(),
                domain_len = table.domain_len(),
            )
            .entered();
            let fold_context = E::precompute_fold(challenge);
            table
                .fold_in_place(|left, right| E::fold_one(&fold_context, left, right))
                .expect("validated exact-prefix range-image state can fold");
        }
        self.rounds_completed += 1;
    }
}
