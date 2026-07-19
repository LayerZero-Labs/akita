use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::{ExactPrefixTable, SplitEqualitySuffixMass};
use super::round_accumulation::{add_scaled_round_coefficients, split_equality_weight};
use super::{compose_small_poly_with_affine, MAX_TREE_STAGE_Q_DEGREE};
use akita_algebra::split_eq::GruenSplitEq;
use akita_field::parallel::*;
use akita_field::unreduced::HasOptimizedFold;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, EqFactoredUniPoly};

enum RangeImageValues<E: FieldCore> {
    Compact(CompactDigitSource),
    Folded(ExactPrefixTable<E>),
}

fn accumulate_round<E: FieldCore>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    default: E,
    pair_at: impl Fn(usize) -> (E, E) + Sync,
    polynomial_coefficients: &[E],
) -> [E; MAX_TREE_STAGE_Q_DEGREE + 1] {
    let mut coefficients = cfg_fold_reduce!(
        0..explicit_pair_count,
        || [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1],
        |mut sum, pair_index| {
            let (left, right) = pair_at(pair_index);
            let pair_coefficients =
                compose_small_poly_with_affine(polynomial_coefficients, left, right - left);
            add_scaled_round_coefficients(
                &mut sum,
                &pair_coefficients,
                split_equality_weight(first, second, pair_index),
            );
            sum
        },
        |mut left, right| {
            for (left, right) in left.iter_mut().zip(right.iter()) {
                *left += *right;
            }
            left
        }
    );
    let default_coefficients =
        compose_small_poly_with_affine(polynomial_coefficients, default, E::zero());
    let suffix_weight = SplitEqualitySuffixMass::new(first, second)
        .and_then(|suffix| suffix.weight_from(explicit_pair_count))
        .expect("split equality and exact prefix were validated at construction");
    add_scaled_round_coefficients(&mut coefficients, &default_coefficients, suffix_weight);
    coefficients
}

/// Final equality-factored quartic over the virtual range-image table.
pub(super) struct StreamingRangeLeaf<E: FieldCore> {
    range_image: RangeImageValues<E>,
    split_eq: GruenSplitEq<E>,
    input_claim: E,
    polynomial_coefficients: Vec<E>,
    num_rounds: usize,
    rounds_completed: usize,
}

impl<E: FieldCore> StreamingRangeLeaf<E> {
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
        Ok(Self {
            range_image: RangeImageValues::Compact(source),
            split_eq: GruenSplitEq::new(equality_point)?,
            input_claim,
            polynomial_coefficients,
            num_rounds: equality_point.len(),
            rounds_completed: 0,
        })
    }

    pub(super) fn final_range_image_eval(&self) -> E {
        let RangeImageValues::Folded(table) = &self.range_image else {
            panic!("range-image leaf remained compact after its final round")
        };
        table
            .final_value()
            .expect("range-image leaf was not fully folded")
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasOptimizedFold> EqFactoredSumcheckInstanceProver<E>
    for StreamingRangeLeaf<E>
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

    fn compute_round_eq_factored(&mut self, _round: usize) -> EqFactoredUniPoly<E> {
        let (first, second) = self.split_eq.remaining_eq_tables();
        let coefficients = match &self.range_image {
            RangeImageValues::Compact(source) => {
                let _span = tracing::info_span!(
                    "digit_range_initial_round",
                    round = self.rounds_completed,
                    explicit_rows = source.live_len(),
                    kernel_strategy = "node-evaluation",
                )
                .entered();
                accumulate_round(
                    first,
                    second,
                    source.live_len().div_ceil(2),
                    E::zero(),
                    |pair_index| {
                        (
                            source.class_or_padding(2 * pair_index).range_image::<E>(),
                            source
                                .class_or_padding(2 * pair_index + 1)
                                .range_image::<E>(),
                        )
                    },
                    &self.polynomial_coefficients,
                )
            }
            RangeImageValues::Folded(table) => {
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
                    &self.polynomial_coefficients,
                )
            }
        };
        EqFactoredUniPoly::from_q_coeffs(coefficients[..=self.degree_bound()].to_vec())
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: E) {
        self.split_eq.bind(challenge);
        let folded_from_compact = match &self.range_image {
            RangeImageValues::Compact(source) => {
                let _span = tracing::info_span!(
                    "digit_range_materialize_range_image",
                    round = self.rounds_completed,
                    explicit_rows = source.live_len(),
                )
                .entered();
                let explicit = cfg_into_iter!(0..source.live_len().div_ceil(2))
                    .map(|pair_index| {
                        let left = source.class_or_padding(2 * pair_index).range_image::<E>();
                        let right = source
                            .class_or_padding(2 * pair_index + 1)
                            .range_image::<E>();
                        left + challenge * (right - left)
                    })
                    .collect();
                Some(
                    ExactPrefixTable::new(source.domain_len() / 2, explicit, E::zero())
                        .expect("compact source and Boolean domain were validated"),
                )
            }
            RangeImageValues::Folded(_) => None,
        };
        if let Some(table) = folded_from_compact {
            self.range_image = RangeImageValues::Folded(table);
        } else if let RangeImageValues::Folded(table) = &mut self.range_image {
            let _span = tracing::info_span!(
                "digit_range_fold_range_image",
                round = self.rounds_completed,
                explicit_rows = table.explicit_len(),
                domain_len = table.domain_len(),
            )
            .entered();
            table
                .fold_in_place(|left, right| left + challenge * (right - left))
                .expect("validated exact-prefix range-image state can fold");
        }
        self.rounds_completed += 1;
    }
}
