//! Sparse compact-geometry relation and restricted-binary terms.

use akita_algebra::{eq_poly::EqPolynomial, offset_eq::OffsetEqWindow, poly::trim_trailing_zeros};
use akita_error::AkitaError;
use akita_sumcheck::reduce_signed_accum;
use jolt_field::solinas::parallel::*;
use jolt_field::Unreduced;
use jolt_field::{Field, Ring, Zero};
use jolt_poly::UnivariatePoly;
use std::cmp::Ordering;
use std::mem;
use std::ops::Range;

use crate::opaque::sumcheck::sum_partials;
use crate::sources::packed_digits::{PackedSignedDigitView, PackedSignedDigits};

/// Weights per parallel task; a smaller support stays on one thread.
const TASK_WEIGHTS: usize = 1 << 12;

#[derive(Clone, Copy)]
struct SparseWeight<E: Field> {
    index: usize,
    linear: E,
    binary: E,
}

/// Consecutive ranges of about [`TASK_WEIGHTS`] sorted, distinct `weights`
/// that keep both children of every parent together.
fn parent_ranges<E: Field>(weights: &[SparseWeight<E>]) -> Vec<Range<usize>> {
    let mut ranges = Vec::with_capacity(weights.len().div_ceil(TASK_WEIGHTS));
    let mut start = 0;
    while start < weights.len() {
        let mut end = (start + TASK_WEIGHTS).min(weights.len());
        // Distinct sorted indices give a parent at most two entries.
        if end < weights.len() && weights[end].index >> 1 == weights[end - 1].index >> 1 {
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// `(parent, linear, binary)` for every parent of `weights`, with the even
/// child's weights first and a missing child's weights zero.
fn parent_pairs<E: Field>(
    weights: &[SparseWeight<E>],
) -> impl Iterator<Item = (usize, [E; 2], [E; 2])> + '_ {
    let mut cursor = 0usize;
    std::iter::from_fn(move || {
        let parent = weights.get(cursor)?.index >> 1;
        let mut linear = [E::zero(); 2];
        let mut binary = [E::zero(); 2];
        while let Some(weight) = weights.get(cursor).filter(|w| w.index >> 1 == parent) {
            let side = weight.index & 1;
            linear[side] = weight.linear;
            binary[side] = weight.binary;
            cursor += 1;
        }
        Some((parent, linear, binary))
    })
}

/// Sparse Stage-2 addend over the canonical witness table.
///
/// Only the compression relation and negative-binary weights are retained.
/// Witness values are read from `RelationRangeImageProver`'s existing compact
/// or folded table, avoiding a full-domain field copy and keeping the addend's
/// work proportional to its live support.
pub(crate) struct AdditionalRelationTerms<E: Field> {
    weights: Vec<SparseWeight<E>>,
    binary_batching: E,
    input_claim: E,
    domain_len: usize,
}

impl<E: Field + Ring> AdditionalRelationTerms<E> {
    pub(crate) fn new(
        compact_witness: &PackedSignedDigits,
        domain_len: usize,
        linear_weights: Vec<(usize, E)>,
        binary_intervals: &[Range<usize>],
        binary_equality_point: &[E],
        binary_batching: E,
    ) -> Result<Self, AkitaError> {
        if !domain_len.is_power_of_two() || compact_witness.len() > domain_len {
            return Err(AkitaError::InvalidSize {
                expected: domain_len,
                actual: compact_witness.len(),
            });
        }
        let expected_equality_variables = domain_len.trailing_zeros() as usize;
        if binary_equality_point.len() != expected_equality_variables {
            return Err(AkitaError::InvalidSize {
                expected: expected_equality_variables,
                actual: binary_equality_point.len(),
            });
        }
        let mut collapsed_linear = Vec::<(usize, E)>::with_capacity(linear_weights.len());
        for (index, value) in linear_weights {
            if index >= domain_len {
                return Err(AkitaError::InvalidSize {
                    expected: domain_len,
                    actual: index.saturating_add(1),
                });
            }
            if let Some((previous_index, previous_value)) = collapsed_linear.last_mut() {
                if index < *previous_index {
                    return Err(AkitaError::InvalidInput(
                        "compression relation weights are not sorted".into(),
                    ));
                }
                if index == *previous_index {
                    *previous_value += value;
                    continue;
                }
            }
            collapsed_linear.push((index, value));
        }
        collapsed_linear.retain(|(_, value)| !value.is_zero());

        let mut previous_end = 0usize;
        let mut binary_support_len = 0usize;
        for interval in binary_intervals {
            if interval.start >= interval.end
                || interval.start < previous_end
                || interval.end > domain_len
            {
                return Err(AkitaError::InvalidInput(
                    "negative-binary support interval is malformed".into(),
                ));
            }
            binary_support_len = binary_support_len
                .checked_add(interval.len())
                .ok_or_else(|| AkitaError::InvalidSetup("binary support length overflow".into()))?;
            previous_end = interval.end;
        }

        // Both sources are already sorted. Merge them directly instead of
        // paying one tree lookup and allocation per compression coordinate.
        let capacity = collapsed_linear
            .len()
            .checked_add(binary_support_len)
            .ok_or_else(|| AkitaError::InvalidSetup("sparse weight capacity overflow".into()))?;
        let binary_equality = OffsetEqWindow::new(binary_equality_point)?;
        let binary_indices = binary_intervals
            .iter()
            .flat_map(|interval| interval.clone())
            .collect::<Vec<_>>();
        let binary_values = cfg_iter!(binary_indices)
            .map(|&index| binary_equality.eval(index))
            .collect::<Vec<_>>();
        let mut weights = Vec::with_capacity(capacity);
        let mut linear = collapsed_linear.into_iter().peekable();
        let mut binary = binary_indices.into_iter().zip(binary_values).peekable();
        loop {
            match (linear.peek(), binary.peek()) {
                (Some(&(linear_index, _)), Some(&(binary_index, _))) => {
                    match linear_index.cmp(&binary_index) {
                        Ordering::Less => {
                            let (index, linear) = linear.next().ok_or(AkitaError::InvalidProof)?;
                            weights.push(SparseWeight {
                                index,
                                linear,
                                binary: E::zero(),
                            });
                        }
                        Ordering::Equal => {
                            let (index, linear) = linear.next().ok_or(AkitaError::InvalidProof)?;
                            let (_, binary) = binary.next().ok_or(AkitaError::InvalidProof)?;
                            weights.push(SparseWeight {
                                index,
                                linear,
                                binary,
                            });
                        }
                        Ordering::Greater => {
                            let (index, binary) = binary.next().ok_or(AkitaError::InvalidProof)?;
                            weights.push(SparseWeight {
                                index,
                                linear: E::zero(),
                                binary,
                            });
                        }
                    }
                }
                (Some(_), None) => {
                    weights.extend(linear.map(|(index, linear)| SparseWeight {
                        index,
                        linear,
                        binary: E::zero(),
                    }));
                    break;
                }
                (None, Some(_)) => {
                    weights.extend(binary.map(|(index, binary)| SparseWeight {
                        index,
                        linear: E::zero(),
                        binary,
                    }));
                    break;
                }
                (None, None) => break,
            }
        }
        let input_claim = cfg_iter!(weights)
            .map(|weight| {
                let witness = compact_witness
                    .get(weight.index)
                    .map_or_else(E::zero, |value| E::from_i64(i64::from(value)));
                witness * weight.linear
                    + binary_batching * weight.binary * witness * (witness + E::one())
            })
            .sum::<E>();
        Ok(Self {
            weights,
            binary_batching,
            input_claim,
            domain_len,
        })
    }

    pub(crate) fn input_claim(&self) -> E {
        self.input_claim
    }

    /// Compute the complete cubic directly in coefficient form.
    ///
    /// For one folded pair, write `w(t) = w0 + t dw`, and likewise for the
    /// linear and binary weights. Expanding
    /// `w(t) l(t) + rho b(t) w(t) (w(t) + 1)` once avoids four separate point
    /// evaluations followed by generic interpolation.
    fn round_polynomial_with(&self, witness_at: impl Fn(usize) -> E + Sync) -> UnivariatePoly<E> {
        let partials = cfg_into_iter!(parent_ranges(&self.weights))
            .map(|range| {
                let mut coefficients = [E::zero(); 4];
                for (parent, linear, binary) in parent_pairs(&self.weights[range]) {
                    let witness = [witness_at(2 * parent), witness_at(2 * parent + 1)];
                    let dw = witness[1] - witness[0];
                    let d_linear = linear[1] - linear[0];
                    let d_binary = binary[1] - binary[0];

                    let witness_square_constant = witness[0].square() + witness[0];
                    let witness_square_linear = dw * (witness[0] + witness[0] + E::one());
                    let witness_square_quadratic = dw.square();
                    let batched_binary = self.binary_batching * binary[0];
                    let batched_binary_delta = self.binary_batching * d_binary;

                    coefficients[0] +=
                        witness[0] * linear[0] + batched_binary * witness_square_constant;
                    coefficients[1] += witness[0] * d_linear
                        + dw * linear[0]
                        + batched_binary * witness_square_linear
                        + batched_binary_delta * witness_square_constant;
                    coefficients[2] += dw * d_linear
                        + batched_binary * witness_square_quadratic
                        + batched_binary_delta * witness_square_linear;
                    coefficients[3] += batched_binary_delta * witness_square_quadratic;
                }
                coefficients
            })
            .collect::<Vec<_>>();
        let mut coefficients = sum_partials(E::zero(), partials).to_vec();
        trim_trailing_zeros(&mut coefficients);
        UnivariatePoly::new(coefficients)
    }

    /// Round polynomial while the witness is still packed signed digits, after
    /// the coefficient challenges `bound` have been drawn.
    pub(crate) fn round_polynomial_compact(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        bound: &[E],
    ) -> UnivariatePoly<E>
    where
        E: Unreduced,
    {
        if bound.is_empty() {
            return self.round_polynomial_compact_initial(compact_witness);
        }
        let bound_weights =
            EqPolynomial::evals(bound).expect("compact prefix binds few coefficient challenges");
        let stride = bound_weights.len();
        self.round_polynomial_with(|index| {
            bound_weights
                .iter()
                .enumerate()
                .fold(E::zero(), |sum, (offset, &weight)| {
                    compact_witness
                        .get(index * stride + offset)
                        .map_or(sum, |value| sum + weight * E::from_i64(i64::from(value)))
                })
        })
    }

    /// First-round specialization while the witness is still packed signed digits.
    ///
    /// The witness-dependent factors are small integers here. Accumulate their
    /// products without reducing after every multiplication, matching the
    /// compact ordinary-relation kernel used by the surrounding Stage 2 prover.
    fn round_polynomial_compact_initial(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
    ) -> UnivariatePoly<E>
    where
        E: Unreduced,
    {
        let partials = cfg_into_iter!(parent_ranges(&self.weights))
            .map(|range| {
                let mut coefficients = [E::SmallProduct::zero(); 8];
                for (parent, linear, binary) in parent_pairs(&self.weights[range]) {
                    let witness_at = |index| compact_witness.get(index).map_or(0, i64::from);
                    let witness = witness_at(2 * parent);
                    let witness_delta = witness_at(2 * parent + 1) - witness;
                    let linear_delta = linear[1] - linear[0];
                    let binary_delta = binary[1] - binary[0];
                    let witness_square_constant = witness * (witness + 1);
                    let witness_square_linear = witness_delta * (2 * witness + 1);
                    let witness_square_quadratic = witness_delta * witness_delta;
                    let batched_binary = self.binary_batching * binary[0];
                    let batched_binary_delta = self.binary_batching * binary_delta;

                    let terms = [
                        (0, linear[0], witness),
                        (0, batched_binary, witness_square_constant),
                        (2, linear_delta, witness),
                        (2, linear[0], witness_delta),
                        (2, batched_binary, witness_square_linear),
                        (2, batched_binary_delta, witness_square_constant),
                        (4, linear_delta, witness_delta),
                        (4, batched_binary, witness_square_quadratic),
                        (4, batched_binary_delta, witness_square_linear),
                        (6, batched_binary_delta, witness_square_quadratic),
                    ];
                    for (slot, factor, small) in terms {
                        super::accum_small_signed(&mut coefficients, slot, factor, small);
                    }
                }
                std::array::from_fn::<E, 4, _>(|degree| {
                    reduce_signed_accum::<E>(coefficients[2 * degree], coefficients[2 * degree + 1])
                })
            })
            .collect::<Vec<_>>();
        let mut coefficients = sum_partials(E::zero(), partials).to_vec();
        trim_trailing_zeros(&mut coefficients);
        UnivariatePoly::new(coefficients)
    }

    pub(crate) fn round_polynomial_folded(&self, folded_witness: &[E]) -> UnivariatePoly<E> {
        self.round_polynomial_with(|index| {
            folded_witness.get(index).copied().unwrap_or_else(E::zero)
        })
    }

    /// Fold every parent's two children by `challenge`, dropping zero parents.
    ///
    /// Tasks compact their parent-aligned ranges in place, and the compacted
    /// ranges are then moved together in order.
    pub(crate) fn bind(&mut self, challenge: E) {
        let ranges = parent_ranges(&self.weights);
        let mut chunks = Vec::with_capacity(ranges.len());
        let mut rest = self.weights.as_mut_slice();
        for range in &ranges {
            let (chunk, tail) = mem::take(&mut rest).split_at_mut(range.len());
            chunks.push(chunk);
            rest = tail;
        }
        let even_scale = E::one() - challenge;
        let kept = cfg_into_iter!(chunks)
            .map(|chunk| {
                let (mut read, mut write) = (0usize, 0usize);
                while read < chunk.len() {
                    let parent = chunk[read].index >> 1;
                    let mut linear = E::zero();
                    let mut binary = E::zero();
                    while read < chunk.len() && chunk[read].index >> 1 == parent {
                        let weight = chunk[read];
                        let scale = if weight.index & 1 == 0 {
                            even_scale
                        } else {
                            challenge
                        };
                        linear += scale * weight.linear;
                        binary += scale * weight.binary;
                        read += 1;
                    }
                    if !linear.is_zero() || !binary.is_zero() {
                        chunk[write] = SparseWeight {
                            index: parent,
                            linear,
                            binary,
                        };
                        write += 1;
                    }
                }
                write
            })
            .collect::<Vec<_>>();
        let mut write = 0usize;
        for (range, kept) in ranges.into_iter().zip(kept) {
            self.weights
                .copy_within(range.start..range.start + kept, write);
            write += kept;
        }
        self.weights.truncate(write);
        self.domain_len /= 2;
    }

    pub(crate) fn final_claim(&self, witness: E) -> Result<E, AkitaError> {
        if self.domain_len != 1
            || self.weights.len() > 1
            || self.weights.first().is_some_and(|weight| weight.index != 0)
        {
            return Err(AkitaError::InvalidProof);
        }
        let Some(weight) = self.weights.first() else {
            return Ok(E::zero());
        };
        Ok(witness * weight.linear
            + self.binary_batching * weight.binary * witness * (witness + E::one()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::offset_eq::eq_eval_at_index;
    use jolt_field::One;
    use jolt_field::Prime128OffsetA7F7 as F;
    use std::collections::BTreeMap;

    fn equality_point(domain_len: usize) -> Vec<F> {
        (0..domain_len.trailing_zeros() as usize)
            .map(|index| F::from_u64(2 + index as u64))
            .collect()
    }

    fn packed(witness: &[i8]) -> PackedSignedDigits {
        PackedSignedDigits::from_i8_digits_auto(witness.to_vec())
    }

    fn reference_round_evaluation(
        terms: &AdditionalRelationTerms<F>,
        witness: &[i8],
        point: F,
    ) -> F {
        let mut evaluation = F::zero();
        let mut cursor = 0usize;
        while cursor < terms.weights.len() {
            let parent = terms.weights[cursor].index >> 1;
            let mut linear = [F::zero(); 2];
            let mut binary = [F::zero(); 2];
            while cursor < terms.weights.len() && terms.weights[cursor].index >> 1 == parent {
                let weight = terms.weights[cursor];
                let side = weight.index & 1;
                linear[side] = weight.linear;
                binary[side] = weight.binary;
                cursor += 1;
            }
            let witness_at = |index| {
                witness
                    .get(index)
                    .map_or_else(F::zero, |&value| F::from_i64(i64::from(value)))
            };
            let left = witness_at(2 * parent);
            let witness_at_point = left + point * (witness_at(2 * parent + 1) - left);
            let linear_at_point = linear[0] + point * (linear[1] - linear[0]);
            let binary_at_point = binary[0] + point * (binary[1] - binary[0]);
            evaluation += witness_at_point * linear_at_point
                + terms.binary_batching
                    * binary_at_point
                    * witness_at_point
                    * (witness_at_point + F::one());
        }
        evaluation
    }

    /// A support spanning several tasks, with parents across task range ends
    /// and parents whose bound weights cancel, matches a serial reference.
    #[test]
    fn multi_task_rounds_and_binds_match_serial_reference() {
        let domain_len = 1 << 16;
        let witness = (0..domain_len)
            .map(|index| ((index * 5 + 3) % 8) as i8 - 4)
            .collect::<Vec<_>>();
        // With the first challenge `2`, a parent with even weight `2c` and odd
        // weight `c` and no binary weight binds to zero.
        let cancelling = 50_000..50_400;
        let linear = (1..domain_len)
            .filter(|index| index % 3 != 1)
            .map(|index| {
                let value = if cancelling.contains(&index) {
                    ((index >> 1) % 7 + 1) as u64 * if index & 1 == 0 { 2 } else { 1 }
                } else {
                    index as u64 % 11 + 1
                };
                (index, F::from_u64(value))
            })
            .collect();
        let packed_witness = packed(&witness);
        let mut terms = AdditionalRelationTerms::new(
            &packed_witness,
            domain_len,
            linear,
            &[5..9_001, 20_000..40_000],
            &equality_point(domain_len),
            F::from_u64(13),
        )
        .unwrap();
        assert!(parent_ranges(&terms.weights)
            .iter()
            .any(|range| range.len() == TASK_WEIGHTS + 1));
        let mut folded = witness
            .iter()
            .map(|&digit| F::from_i64(i64::from(digit)))
            .collect::<Vec<_>>();
        let mut claim = terms.input_claim();
        for round in 0..6u64 {
            let polynomial = if round == 0 {
                terms.round_polynomial_compact(packed_witness.view(), &[])
            } else {
                terms.round_polynomial_folded(&folded)
            };
            assert_eq!(
                polynomial.evaluate(F::zero()) + polynomial.evaluate(F::one()),
                claim
            );
            let challenge = F::from_u64(if round == 0 { 2 } else { 29 + round });
            claim = polynomial.evaluate(challenge);

            let mut expected = BTreeMap::<usize, (F, F)>::new();
            for weight in &terms.weights {
                let scale = if weight.index & 1 == 0 {
                    F::one() - challenge
                } else {
                    challenge
                };
                let entry = expected
                    .entry(weight.index >> 1)
                    .or_insert((F::zero(), F::zero()));
                entry.0 += scale * weight.linear;
                entry.1 += scale * weight.binary;
            }
            let parents = expected.len();
            expected.retain(|_, (linear, binary)| !linear.is_zero() || !binary.is_zero());
            if round == 0 {
                assert!(expected.len() < parents);
            }
            terms.bind(challenge);
            assert_eq!(
                terms
                    .weights
                    .iter()
                    .map(|weight| (weight.index, (weight.linear, weight.binary)))
                    .collect::<Vec<_>>(),
                expected.into_iter().collect::<Vec<_>>()
            );
            folded = folded
                .chunks(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect();
        }
        let direct = terms
            .weights
            .iter()
            .map(|weight| {
                let witness = folded[weight.index];
                witness * weight.linear
                    + terms.binary_batching * weight.binary * witness * (witness + F::one())
            })
            .sum::<F>();
        assert_eq!(direct, claim);
    }

    #[test]
    fn round_polynomial_matches_boolean_sum_and_fold() {
        let witness = [-1, 0, 2, -2];
        let linear = vec![
            (0, F::from_u64(3)),
            (1, F::from_u64(5)),
            (2, F::from_u64(7)),
            (3, F::from_u64(11)),
        ];
        let rho = F::from_u64(13);
        let equality_point = equality_point(4);
        let claim = witness.iter().zip([3, 5, 7, 11]).enumerate().fold(
            F::zero(),
            |sum, (index, (&witness, linear))| {
                let witness = F::from_i64(i64::from(witness));
                let binary = if index < 2 {
                    eq_eval_at_index(&equality_point, index)
                } else {
                    F::zero()
                };
                sum + witness * F::from_u64(linear) + rho * binary * witness * (witness + F::one())
            },
        );
        let binary_interval = 0..2;
        let packed_witness = packed(&witness);
        let mut prover = AdditionalRelationTerms::new(
            &packed_witness,
            4,
            linear,
            std::slice::from_ref(&binary_interval),
            &equality_point,
            rho,
        )
        .unwrap();
        assert_eq!(prover.input_claim(), claim);
        let polynomial = prover.round_polynomial_compact(packed_witness.view(), &[]);
        assert_eq!(
            polynomial.evaluate(F::zero()) + polynomial.evaluate(F::one()),
            claim
        );
        let challenge = F::from_u64(17);
        let next_claim = polynomial.evaluate(challenge);
        prover.bind(challenge);
        let next = prover.round_polynomial_compact(packed_witness.view(), &[challenge]);
        assert_eq!(
            next.evaluate(F::zero()) + next.evaluate(F::one()),
            next_claim
        );
    }

    #[test]
    fn nonbinary_digit_inside_support_contributes_a_nonzero_constraint() {
        let rho = F::from_u64(13);
        let binary_interval = 0..1;
        let support = std::slice::from_ref(&binary_interval);
        let equality_point = equality_point(2);
        let invalid_witness = packed(&[2, 0]);
        let invalid = AdditionalRelationTerms::new(
            &invalid_witness,
            2,
            Vec::new(),
            support,
            &equality_point,
            rho,
        )
        .unwrap();
        assert_eq!(
            invalid.input_claim(),
            rho * eq_eval_at_index(&equality_point, 0) * F::from_u64(6)
        );

        let valid_witness = packed(&[-1, 0]);
        let valid = AdditionalRelationTerms::new(
            &valid_witness,
            2,
            Vec::new(),
            support,
            &equality_point,
            rho,
        )
        .unwrap();
        assert_eq!(valid.input_claim(), F::zero());
    }

    #[test]
    fn coefficient_kernel_matches_direct_cubic_evaluation() {
        let witness = [-1, 0, 2, -2, 1, 3, -4, 0];
        let linear = vec![
            (0, F::from_u64(3)),
            (1, F::from_u64(5)),
            (3, F::from_u64(7)),
            (6, F::from_u64(11)),
        ];
        let packed_witness = packed(&witness);
        let terms = AdditionalRelationTerms::new(
            &packed_witness,
            8,
            linear,
            &[1..4, 6..8],
            &equality_point(8),
            F::from_u64(13),
        )
        .unwrap();
        let polynomial = terms.round_polynomial_compact(packed_witness.view(), &[]);
        for point in 0..=5 {
            let point = F::from_u64(point);
            assert_eq!(
                polynomial.evaluate(point),
                reference_round_evaluation(&terms, &witness, point)
            );
        }
    }

    #[test]
    fn construction_linearly_merges_duplicates_and_binary_support() {
        let equality_point = equality_point(8);
        let packed_witness = packed(&[0; 8]);
        let terms = AdditionalRelationTerms::new(
            &packed_witness,
            8,
            vec![
                (0, F::from_u64(2)),
                (0, F::from_u64(3)),
                (3, F::from_u64(7)),
                (7, F::from_u64(11)),
            ],
            &[1..4, 6..8],
            &equality_point,
            F::from_u64(13),
        )
        .unwrap();
        assert_eq!(
            terms
                .weights
                .iter()
                .map(|weight| (weight.index, weight.linear, weight.binary))
                .collect::<Vec<_>>(),
            vec![
                (0, F::from_u64(5), F::zero()),
                (1, F::zero(), eq_eval_at_index(&equality_point, 1)),
                (2, F::zero(), eq_eval_at_index(&equality_point, 2)),
                (3, F::from_u64(7), eq_eval_at_index(&equality_point, 3)),
                (6, F::zero(), eq_eval_at_index(&equality_point, 6)),
                (7, F::from_u64(11), eq_eval_at_index(&equality_point, 7)),
            ]
        );
    }

    #[test]
    fn construction_rejects_unsorted_linear_weights() {
        let packed_witness = packed(&[0; 4]);
        assert!(AdditionalRelationTerms::new(
            &packed_witness,
            4,
            vec![(2, F::one()), (1, F::one())],
            &[],
            &equality_point(4),
            F::one(),
        )
        .is_err());
    }
}
