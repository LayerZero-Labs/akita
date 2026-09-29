//! Sparse compact-geometry relation and restricted-binary terms.

use akita_algebra::{eq_poly::EqPolynomial, offset_eq::OffsetEqWindow};
use akita_error::AkitaError;
use akita_sumcheck::reduce_signed_accum;
use jolt_field::solinas::parallel::*;
use jolt_field::Unreduced;
use jolt_field::{Field, Ring, Zero};
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
    domain_len: usize,
}

impl<E: Field + Ring> AdditionalRelationTerms<E> {
    #[tracing::instrument(skip_all, name = "additional_relation_new")]
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
        Ok(Self {
            weights,
            binary_batching,
            domain_len,
        })
    }

    #[cfg(any(debug_assertions, test))]
    pub(crate) fn input_claim(&self, compact_witness: &PackedSignedDigits) -> E {
        cfg_iter!(self.weights)
            .map(|weight| {
                let witness = compact_witness
                    .get(weight.index)
                    .map_or_else(E::zero, |value| E::from_i64(i64::from(value)));
                witness * weight.linear
                    + self.binary_batching * weight.binary * witness * (witness + E::one())
            })
            .sum::<E>()
    }

    #[cfg(debug_assertions)]
    pub(super) fn debug_round_at_zero(&self, witness_at: impl Fn(usize) -> E) -> E {
        self.weights
            .iter()
            .filter(|weight| weight.index % 2 == 0)
            .map(|weight| {
                let w = witness_at(weight.index);
                w * weight.linear + self.binary_batching * weight.binary * w * (w + E::one())
            })
            .sum()
    }

    /// Accumulate the value at one and the top two coefficients directly.
    /// The caller reconstructs the constant and linear coefficients from the
    /// running sumcheck claim, so this avoids their per-coordinate products.
    #[tracing::instrument(skip_all, name = "additional_relation_round")]
    fn round_message_with(&self, witness_at: impl Fn(usize) -> E + Sync) -> super::RoundMessage<E>
    where
        E: Unreduced,
    {
        let ranges = parent_ranges(&self.weights);
        let task = |range: Range<usize>| {
            // A task contains at most TASK_WEIGHTS + 1 entries. Even the
            // cubic still contributes only four products per pair, well
            // within ProductSum's accumulation bound.
            let mut coefficients = [super::ProductSum::<E>::zero(); 3];
            for (parent, linear, binary) in parent_pairs(&self.weights[range]) {
                let witness = [witness_at(2 * parent), witness_at(2 * parent + 1)];
                let dw = witness[1] - witness[0];
                let d_linear = linear[1] - linear[0];
                coefficients[0].add(witness[1], linear[1]);
                coefficients[1].add(dw, d_linear);

                // Compression and response-norm coordinates can have only
                // a linear weight. Their round polynomial is quadratic;
                // no witness squaring or binary products are needed.
                if !binary[0].is_zero() || !binary[1].is_zero() {
                    let witness_square_at_one = witness[1].square() + witness[1];
                    let witness_square_linear = dw * (witness[0] + witness[0] + E::one());
                    let witness_square_quadratic = dw.square();
                    let batched_binary = self.binary_batching * binary[0];
                    let batched_binary_at_one = self.binary_batching * binary[1];
                    let batched_binary_delta = self.binary_batching * (binary[1] - binary[0]);
                    coefficients[0].add(batched_binary_at_one, witness_square_at_one);
                    coefficients[1].add(batched_binary, witness_square_quadratic);
                    coefficients[1].add(batched_binary_delta, witness_square_linear);
                    coefficients[2].add(batched_binary_delta, witness_square_quadratic);
                }
            }
            coefficients.map(super::ProductSum::finish)
        };
        let partials = if ranges.len() <= 1 {
            ranges.into_iter().map(task).collect::<Vec<_>>()
        } else {
            cfg_into_iter!(ranges).map(task).collect::<Vec<_>>()
        };
        let totals = sum_partials(E::zero(), partials);
        super::RoundMessage {
            at_one: totals[0],
            quadratic: totals[1],
            cubic: totals[2],
        }
    }

    /// Round message while the witness is still packed signed digits, after
    /// the coefficient challenges `bound` have been drawn.
    #[tracing::instrument(skip_all, name = "additional_relation_compact_round")]
    pub(super) fn round_message_compact(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
        bound: &[E],
    ) -> super::RoundMessage<E>
    where
        E: Unreduced,
    {
        if bound.is_empty() {
            return self.round_message_compact_initial(compact_witness);
        }
        let bound_weights =
            EqPolynomial::evals(bound).expect("compact prefix binds few coefficient challenges");
        let stride = bound_weights.len();
        self.round_message_with(|index| {
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
    fn round_message_compact_initial(
        &self,
        compact_witness: PackedSignedDigitView<'_>,
    ) -> super::RoundMessage<E>
    where
        E: Unreduced,
    {
        let ranges = parent_ranges(&self.weights);
        let task = |range: Range<usize>| {
            let mut coefficients = [E::SmallProduct::zero(); 6];
            for (parent, linear, binary) in parent_pairs(&self.weights[range]) {
                let witness_at = |index| compact_witness.get(index).map_or(0, i64::from);
                let witness = witness_at(2 * parent);
                let witness_delta = witness_at(2 * parent + 1) - witness;
                let linear_delta = linear[1] - linear[0];
                let binary_delta = binary[1] - binary[0];
                let witness_at_one = witness + witness_delta;
                let witness_square_at_one = witness_at_one * (witness_at_one + 1);
                let witness_square_linear = witness_delta * (2 * witness + 1);
                let witness_square_quadratic = witness_delta * witness_delta;
                let batched_binary = self.binary_batching * binary[0];
                let batched_binary_at_one = self.binary_batching * binary[1];
                let batched_binary_delta = self.binary_batching * binary_delta;

                let terms = [
                    (0, linear[1], witness_at_one),
                    (0, batched_binary_at_one, witness_square_at_one),
                    (2, linear_delta, witness_delta),
                    (2, batched_binary, witness_square_quadratic),
                    (2, batched_binary_delta, witness_square_linear),
                    (4, batched_binary_delta, witness_square_quadratic),
                ];
                for (slot, factor, small) in terms {
                    super::accum_small_signed(&mut coefficients, slot, factor, small);
                }
            }
            std::array::from_fn::<E, 3, _>(|degree| {
                reduce_signed_accum::<E>(coefficients[2 * degree], coefficients[2 * degree + 1])
            })
        };
        let partials = if ranges.len() <= 1 {
            ranges.into_iter().map(task).collect::<Vec<_>>()
        } else {
            cfg_into_iter!(ranges).map(task).collect::<Vec<_>>()
        };
        let totals = sum_partials(E::zero(), partials);
        super::RoundMessage {
            at_one: totals[0],
            quadratic: totals[1],
            cubic: totals[2],
        }
    }

    pub(super) fn round_message_folded(&self, folded_witness: &[E]) -> super::RoundMessage<E>
    where
        E: Unreduced,
    {
        self.round_message_with(|index| folded_witness.get(index).copied().unwrap_or_else(E::zero))
    }

    /// Fold every parent's two children by `challenge`, dropping zero parents.
    ///
    /// Tasks compact their parent-aligned ranges in place, and the compacted
    /// ranges are then moved together in order.
    #[tracing::instrument(skip_all, name = "additional_relation_bind")]
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
        let task = |chunk: &mut [SparseWeight<E>]| {
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
        };
        let kept = if chunks.len() <= 1 {
            chunks.into_iter().map(task).collect::<Vec<_>>()
        } else {
            cfg_into_iter!(chunks).map(task).collect::<Vec<_>>()
        };
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
mod tests;
