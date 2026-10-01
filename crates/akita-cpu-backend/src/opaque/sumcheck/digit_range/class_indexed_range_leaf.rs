//! Class-indexed range-polynomial leaf prover.

use super::class_indexed_state::ClassIndexedTableState;
use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::ExactPrefixTable;
use super::range_class_tables::{
    FoldedRangeImagePairTable, OrderedRangePairCoefficients, SecondRoundRangeQuartetCoefficients,
};
use super::round_accumulation::{
    accumulate_equality_weighted_pair_terms, accumulate_equality_weighted_values,
};
use super::{
    compose_small_poly_with_affine, MAX_QUARTET_TABLE_CLASS_COUNT, MAX_TREE_STAGE_Q_DEGREE,
};
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use akita_sumcheck::EqFactoredSumcheckInstanceProver;
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, Ring};
use jolt_field::{Fold, Unreduced};
use jolt_poly::OmittedConstantPoly;

struct CompactRangeLeafState<E: Field> {
    source: CompactDigitSource,
    pair_coefficients: OrderedRangePairCoefficients<E>,
}

struct FirstChallengeFoldedRangeLeafState<E: Field> {
    source: CompactDigitSource,
    folded_pairs: FoldedRangeImagePairTable<E>,
    cached_second_round_coefficients: [E; MAX_TREE_STAGE_Q_DEGREE + 1],
}

type RangeImageTableState<E> =
    ClassIndexedTableState<CompactRangeLeafState<E>, FirstChallengeFoldedRangeLeafState<E>, E>;

fn accumulate_round<E: Field + Unreduced>(
    equality_prefix_weights: &[E],
    equality_suffix_weights: &[E],
    explicit_pair_count: usize,
    padding_range_image: E,
    pair_at: impl Fn(usize) -> (E, E) + Sync,
    polynomial_coefficients: &[E],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let padding_coefficients =
        compose_small_poly_with_affine(polynomial_coefficients, padding_range_image, E::zero());
    accumulate_equality_weighted_values(
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

/// A quartic leaf polynomial in depressed monic form, for the nonconstant
/// round coefficients of the materialized rounds.
///
/// A quartic `P(y) = c4 y^4 + c3 y^3 + c2 y^2 + c1 y + c0` with `c4 != 0` is
/// `c4 M(y - s)` for `s = -c3 / (4 c4)` and a monic `M(t) = t^4 + m2 t^2 +
/// m1 t + m0` without a cubic term. With `t = left - s`, the nonconstant
/// coefficients of `P(left + X delta)` are `4 c4 (t^3 + (m2/2) t + m1/4)
/// delta`, `6 c4 (t^2 + m2/6) delta^2`, `4 c4 t delta^3` and `c4 delta^4`.
/// The integer factors and `c4` are applied once per round, and the weight
/// rides on the powers `weight * delta^k`, so a pair pays one squaring, four
/// multiplications and four unreduced products.
#[derive(Clone, Copy)]
struct DepressedQuartic<E> {
    shift: E,
    half_m2: E,
    sixth_m2: E,
    quarter_m1: E,
    /// `[4 c4, 6 c4, 4 c4, c4]`, the per-round scales of the four sums.
    scales: [E; MAX_TREE_STAGE_Q_DEGREE],
}

impl<E: Field + Ring> DepressedQuartic<E> {
    /// Return the depressed form of an exact quartic, or `None` when the
    /// polynomial has fewer than five coefficients or a zero leading one.
    fn new(coefficients: &[E]) -> Option<Self> {
        let &[_, c1, c2, c3, c4] = coefficients else {
            return None;
        };
        let lead_inverse = c4.inverse()?;
        let half = E::from_u64(2).inverse()?;
        let quarter = E::from_u64(4).inverse()?;
        let sixth = E::from_u64(6).inverse()?;
        let (m1, m2, m3) = (c1 * lead_inverse, c2 * lead_inverse, c3 * lead_inverse);
        let shift = -(m3 * quarter);
        let shift_squared = shift.square();
        // Taylor shift of the monic quartic to `t + s`; the cubic term cancels.
        let depressed_m2 = m2 + E::from_u64(3) * m3 * shift + E::from_u64(6) * shift_squared;
        let depressed_m1 = m1
            + E::from_u64(2) * m2 * shift
            + E::from_u64(3) * m3 * shift_squared
            + E::from_u64(4) * shift_squared * shift;
        Some(Self {
            shift,
            half_m2: depressed_m2 * half,
            sixth_m2: depressed_m2 * sixth,
            quarter_m1: depressed_m1 * quarter,
            scales: [4, 6, 4, 1].map(|factor| E::from_u64(factor) * c4),
        })
    }

    /// Factor pairs whose products are the pair's four nonconstant sums.
    #[inline(always)]
    fn pair_terms(
        &self,
        left: E,
        right: E,
        weight: E,
    ) -> ([E; MAX_TREE_STAGE_Q_DEGREE], [E; MAX_TREE_STAGE_Q_DEGREE]) {
        let delta = right - left;
        let shifted = left - self.shift;
        let shifted_squared = shifted.square();
        let weighted_delta = weight * delta;
        let weighted_delta_squared = weighted_delta * delta;
        let weighted_delta_cubed = weighted_delta_squared * delta;
        (
            [
                shifted * (shifted_squared + self.half_m2) + self.quarter_m1,
                shifted_squared + self.sixth_m2,
                shifted,
                delta,
            ],
            [
                weighted_delta,
                weighted_delta_squared,
                weighted_delta_cubed,
                weighted_delta_cubed,
            ],
        )
    }
}

/// Final equality-factored quartic over the virtual range-image table.
pub(crate) struct ClassIndexedRangeLeafProver<E: Field> {
    range_image: RangeImageTableState<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    polynomial_coefficients: Vec<E>,
    depressed_quartic: Option<DepressedQuartic<E>>,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: Field + Ring> ClassIndexedRangeLeafProver<E> {
    pub(crate) fn new(
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
            depressed_quartic: DepressedQuartic::new(&polynomial_coefficients),
            polynomial_coefficients,
            num_rounds: equality_point.len(),
            rounds_completed: 0,
        })
    }

    pub(crate) fn final_range_image_eval(&self) -> E {
        self.range_image
            .final_value()
            .expect("range-image leaf was not fully folded")
    }
}

impl<E: Field + Ring + Fold + Unreduced> ClassIndexedRangeLeafProver<E> {
    /// Compute the existing equality-factored leaf's inner polynomial in
    /// coefficient form. Both the ordinary eq-factored driver and the L2 fused
    /// driver consume this one kernel.
    pub(crate) fn round_q_coefficients(
        &mut self,
        round: usize,
        claim: E,
    ) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
        debug_assert_eq!(round, self.rounds_completed);
        let (equality_prefix_weights, equality_suffix_weights) =
            self.split_eq.remaining_eq_tables();
        match &self.range_image {
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
                let pair_count = table.explicit_len().div_ceil(2);
                let pair_at = |pair_index: usize| {
                    (
                        table.value_or_default(2 * pair_index),
                        table.value_or_default(2 * pair_index + 1),
                    )
                };
                match &self.depressed_quartic {
                    Some(quartic) => {
                        let sums = accumulate_equality_weighted_pair_terms(
                            equality_prefix_weights,
                            equality_suffix_weights,
                            pair_count,
                            |pair_index, weight| {
                                let (left, right) = pair_at(pair_index);
                                quartic.pair_terms(left, right, weight)
                            },
                        );
                        let nonconstant: [E; MAX_TREE_STAGE_Q_DEGREE] =
                            std::array::from_fn(|index| quartic.scales[index] * sums[index]);
                        // The normalized claim is `q(0) + tau (q(1) - q(0))`.
                        let nonconstant_sum: E = nonconstant.iter().copied().sum();
                        let constant = claim - self.split_eq.current_tau() * nonconstant_sum;
                        #[cfg(debug_assertions)]
                        {
                            let evaluate = |point: E| {
                                self.polynomial_coefficients
                                    .iter()
                                    .rev()
                                    .fold(E::zero(), |value, &coefficient| {
                                        value * point + coefficient
                                    })
                            };
                            let [direct_constant] = accumulate_equality_weighted_values(
                                equality_prefix_weights,
                                equality_suffix_weights,
                                pair_count,
                                |pair_index| {
                                    let (left, _) = pair_at(pair_index);
                                    [evaluate(left)]
                                },
                                [evaluate(table.default_value())],
                            );
                            debug_assert_eq!(constant, direct_constant);
                        }
                        let [q1, q2, q3, q4] = nonconstant;
                        [constant, q1, q2, q3, q4]
                    }
                    None => accumulate_round(
                        equality_prefix_weights,
                        equality_suffix_weights,
                        pair_count,
                        table.default_value(),
                        pair_at,
                        &self.polynomial_coefficients,
                    ),
                }
            }
        }
    }

    pub(crate) fn final_range_claim(&self) -> E {
        let range_image = self.final_range_image_eval();
        let leaf = self
            .polynomial_coefficients
            .iter()
            .rev()
            .fold(E::zero(), |acc, &coefficient| {
                acc * range_image + coefficient
            });
        self.split_eq.current_scalar() * leaf
    }

    /// Scalar-bearing equality-factor evaluations for the fused ordinary
    /// range-plus-L2 sum-check, which reconstructs the full `l(X)q(X)` term.
    pub(crate) fn current_full_eq_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }
}

impl<E: Field + Ring + Fold + Unreduced> EqFactoredSumcheckInstanceProver<E>
    for ClassIndexedRangeLeafProver<E>
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

    fn current_tau(&self) -> E {
        self.split_eq.current_tau()
    }

    fn compute_round_eq_factored(&mut self, round: usize, claim: E) -> OmittedConstantPoly<E> {
        let coefficients = self.round_q_coefficients(round, claim);
        OmittedConstantPoly::from_q_coefficients(coefficients[..=self.degree_bound()].to_vec())
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
                    let coefficients = accumulate_equality_weighted_values(
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
                    let fold_context = E::precompute(challenge);
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
            let fold_context = E::precompute(challenge);
            table
                .fold_in_place(|left, right| E::fold_one(&fold_context, left, right))
                .expect("validated exact-prefix range-image state can fold");
        }
        self.rounds_completed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::packed_digits::PackedSignedDigits;
    use akita_params::{DigitRangePlan, FlatBooleanDomain};
    use jolt_field::{Ext2, One, Prime64Offset59, Zero};

    type F = Ext2<Prime64Offset59>;

    #[test]
    fn depressed_quartic_matches_affine_composition() {
        let value = |seed: u64| F::from_u64(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 3);
        for polynomial in [
            [3, 5, 7, 11, 13].map(value).to_vec(),
            [17, 0, 19, 23, 1].map(value).to_vec(),
            vec![
                F::from_u64(105),
                -F::from_u64(64),
                -F::from_u64(42),
                F::zero(),
                F::one(),
            ],
        ] {
            let quartic =
                DepressedQuartic::new(&polynomial).expect("leading coefficient is nonzero");
            for (left, right, weight) in [(29, 31, 37), (41, 41, 43), (0, 47, 1)] {
                let (left, right, weight) = (value(left), value(right), value(weight));
                let (factors, weighted) = quartic.pair_terms(left, right, weight);
                let expected = compose_small_poly_with_affine(&polynomial, left, right - left);
                for index in 0..MAX_TREE_STAGE_Q_DEGREE {
                    assert_eq!(
                        quartic.scales[index] * factors[index] * weighted[index],
                        weight * expected[index + 1],
                    );
                }
            }
        }
    }

    #[test]
    fn depressed_quartic_requires_an_exact_quartic() {
        assert!(DepressedQuartic::<F>::new(&[F::one(), F::one(), F::one(), F::one()]).is_none());
        assert!(
            DepressedQuartic::<F>::new(&[F::one(), F::one(), F::one(), F::one(), F::zero()])
                .is_none()
        );
    }

    #[test]
    fn cubic_leaf_rounds_match_dense_reference() {
        let digits = [-7, -3, -1, 0, 2, 5, 7];
        let domain = FlatBooleanDomain::new(digits.len(), 3).unwrap();
        let source = CompactDigitSource::new(
            PackedSignedDigits::from_i8_digits_auto(digits.to_vec()),
            domain,
            DigitRangePlan::new(16).unwrap(),
        )
        .unwrap();
        let equality_point = [17, 19, 23].map(F::from_u64);
        let polynomial = [3, 5, 7, 11].map(F::from_u64).to_vec();
        let mut dense = digits
            .into_iter()
            .map(|digit| {
                let digit = F::from_i64(i64::from(digit));
                digit * (digit + F::one())
            })
            .chain(std::iter::once(F::zero()))
            .collect::<Vec<_>>();
        let mut reference_eq = GruenSplitEq::new(&equality_point).unwrap();
        let dense_round = |dense: &[F], split_eq: &GruenSplitEq<F>| {
            let (first, second) = split_eq.remaining_eq_tables();
            accumulate_round(
                first,
                second,
                dense.len() / 2,
                F::zero(),
                |pair_index| (dense[2 * pair_index], dense[2 * pair_index + 1]),
                &polynomial,
            )
        };
        let first_round = dense_round(&dense, &reference_eq);
        let mut claim = first_round[0]
            + reference_eq.current_tau() * first_round[1..].iter().copied().sum::<F>();
        let mut prover =
            ClassIndexedRangeLeafProver::new(source, &equality_point, claim, polynomial.clone())
                .unwrap();

        for round in 0..equality_point.len() {
            let expected = dense_round(&dense, &reference_eq);
            assert_eq!(prover.round_q_coefficients(round, claim), expected);

            let challenge = F::from_u64(29 + round as u64 * 2);
            claim = expected
                .iter()
                .rev()
                .fold(F::zero(), |value, &coefficient| {
                    value * challenge + coefficient
                });
            prover.ingest_challenge(round, challenge);
            reference_eq.bind(challenge);
            let fold_context = F::precompute(challenge);
            dense = dense
                .chunks_exact(2)
                .map(|pair| F::fold_one(&fold_context, pair[0], pair[1]))
                .collect();
        }
    }
}
