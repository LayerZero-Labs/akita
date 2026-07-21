//! Prover-owned evaluation-trace support prepared for Stage 2.

use akita_field::{AkitaError, FieldCore};
use akita_types::{basis_weights_prefix, EvaluationTraceWeights};

/// One opening block/digit contribution over contiguous common-coordinate columns.
struct PreparedOpeningSupport<E: FieldCore> {
    first_column: usize,
    factor: E,
    inner_trace_index: usize,
}

#[derive(Clone)]
struct PreparedColumnTerm<E: FieldCore> {
    factor: E,
    source_index: usize,
    lane: usize,
}

struct PreparedTraceSource<E: FieldCore> {
    values: Vec<E>,
    lane_count: usize,
}

/// Canonical prover preparation of the evaluation trace's exact live E support.
///
/// Block, claim, and digit scalars are compiled once. The source-coordinate trace stays
/// factored while coefficient coordinates are folded; column challenges then merge the
/// prepared support directly. No full coefficient-domain trace table is materialized.
pub(crate) struct PreparedProverEvaluationTrace<E: FieldCore> {
    column_terms: Vec<Vec<PreparedColumnTerm<E>>>,
    sources: Vec<PreparedTraceSource<E>>,
    live_column_count: usize,
    coeff_count: usize,
}

impl<E: FieldCore> PreparedProverEvaluationTrace<E> {
    #[cfg(test)]
    pub(crate) fn from_dense(dense: Vec<E>, live_column_count: usize, coeff_count: usize) -> Self {
        assert_eq!(dense.len(), live_column_count * coeff_count);
        let mut column_terms = vec![Vec::new(); live_column_count];
        let sources = dense
            .chunks_exact(coeff_count)
            .enumerate()
            .map(|(column, values)| {
                column_terms[column].push(PreparedColumnTerm {
                    factor: E::one(),
                    source_index: column,
                    lane: 0,
                });
                PreparedTraceSource {
                    values: values.to_vec(),
                    lane_count: 1,
                }
            })
            .collect();
        Self {
            column_terms,
            sources,
            live_column_count,
            coeff_count,
        }
    }

    /// Compile checked semantic trace terms into exact opening support.
    #[tracing::instrument(
        skip_all,
        name = "PreparedProverEvaluationTrace::new",
        fields(
            terms = weights.terms().len(),
            coeff_count,
            physical_field_len = weights.physical_field_len()
        )
    )]
    pub(crate) fn new(
        weights: &EvaluationTraceWeights<E>,
        coeff_count: usize,
        output_scale: E,
    ) -> Result<Self, AkitaError> {
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || !weights.physical_field_len().is_multiple_of(coeff_count)
        {
            return Err(AkitaError::InvalidSetup(
                "evaluation-trace common-coordinate geometry is malformed".into(),
            ));
        }
        let live_column_count = weights.physical_field_len() / coeff_count;
        let opening_support_count =
            weights
                .terms()
                .iter()
                .try_fold(0usize, |term_count, term| {
                    term.segments()
                        .iter()
                        .try_fold(term_count, |count, segment| {
                            segment
                                .block_count()
                                .checked_mul(term.opening_digit_weights().len())
                                .and_then(|segment_count| count.checked_add(segment_count))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "evaluation-trace support count overflow".into(),
                                    )
                                })
                        })
                })?;
        let mut opening_support = Vec::new();
        opening_support
            .try_reserve_exact(opening_support_count)
            .map_err(|_| {
                AkitaError::InvalidInput("evaluation-trace support allocation failed".into())
            })?;
        let mut source_inner_traces = Vec::with_capacity(weights.terms().len());
        for term in weights.terms() {
            let source_ring_dimension = term.source_ring_dimension();
            if source_ring_dimension == 0
                || !source_ring_dimension.is_power_of_two()
                || !source_ring_dimension.is_multiple_of(coeff_count)
                || term.inner_trace().len() != source_ring_dimension
            {
                return Err(AkitaError::InvalidSetup(
                    "evaluation-trace source ring is incompatible with Stage 2".into(),
                ));
            }
            let block_weights = basis_weights_prefix(
                term.block_opening_point(),
                term.block_opening_basis(),
                term.group_block_count(),
            )?;
            let block_stride = term
                .opening_digit_weights()
                .len()
                .checked_mul(source_ring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("evaluation-trace block stride overflow".into())
                })?;
            let inner_trace_index = source_inner_traces.len();
            source_inner_traces.push(term.shared_inner_trace());
            for segment in term.segments() {
                for local_block in 0..segment.block_count() {
                    let global_block = segment
                        .global_block_start()
                        .checked_add(local_block)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "evaluation-trace global block overflow".into(),
                            )
                        })?;
                    let block_weight = *block_weights
                        .get(global_block)
                        .ok_or(AkitaError::InvalidProof)?;
                    let local_block_offset =
                        block_stride.checked_mul(local_block).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "evaluation-trace block offset overflow".into(),
                            )
                        })?;
                    let block_start = segment
                        .physical_coefficient_start()
                        .checked_add(local_block_offset)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "evaluation-trace block address overflow".into(),
                            )
                        })?;
                    for (digit, &digit_weight) in term.opening_digit_weights().iter().enumerate() {
                        let digit_offset =
                            source_ring_dimension.checked_mul(digit).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "evaluation-trace digit offset overflow".into(),
                                )
                            })?;
                        let coefficient_start =
                            block_start.checked_add(digit_offset).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "evaluation-trace digit address overflow".into(),
                                )
                            })?;
                        if !coefficient_start.is_multiple_of(coeff_count) {
                            return Err(AkitaError::InvalidSetup(
                                "evaluation-trace support is not common-coordinate aligned".into(),
                            ));
                        }
                        let first_column = coefficient_start / coeff_count;
                        let column_count = source_ring_dimension / coeff_count;
                        let support_end =
                            first_column.checked_add(column_count).ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "evaluation-trace support range overflow".into(),
                                )
                            })?;
                        if support_end > live_column_count {
                            return Err(AkitaError::InvalidProof);
                        }
                        opening_support.push(PreparedOpeningSupport {
                            first_column,
                            factor: output_scale * term.coefficient() * block_weight * digit_weight,
                            inner_trace_index,
                        });
                    }
                }
            }
        }
        if opening_support.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        if opening_support.len() != opening_support_count {
            return Err(AkitaError::InvalidProof);
        }
        let sources = source_inner_traces
            .into_iter()
            .map(|values| PreparedTraceSource {
                lane_count: values.len() / coeff_count,
                values: values.as_ref().to_vec(),
            })
            .collect::<Vec<_>>();
        let mut column_terms = vec![Vec::new(); live_column_count];
        for support in opening_support {
            let source = sources
                .get(support.inner_trace_index)
                .ok_or(AkitaError::InvalidProof)?;
            for lane in 0..source.lane_count {
                let column = support.first_column.checked_add(lane).ok_or_else(|| {
                    AkitaError::InvalidSetup("evaluation-trace column overflow".into())
                })?;
                column_terms
                    .get_mut(column)
                    .ok_or(AkitaError::InvalidProof)?
                    .push(PreparedColumnTerm {
                        factor: support.factor,
                        source_index: support.inner_trace_index,
                        lane,
                    });
            }
        }
        Ok(Self {
            column_terms,
            sources,
            live_column_count,
            coeff_count,
        })
    }

    #[inline]
    pub(crate) fn get(&self, column: usize, coefficient: usize, coeff_count: usize) -> E {
        debug_assert_eq!(self.coeff_count, coeff_count);
        let Some(terms) = self.column_terms.get(column) else {
            return E::zero();
        };
        terms.iter().fold(E::zero(), |evaluation, term| {
            let Some(source) = self.sources.get(term.source_index) else {
                return evaluation;
            };
            let index = term.lane * self.coeff_count + coefficient;
            evaluation
                + source
                    .values
                    .get(index)
                    .copied()
                    .map_or_else(E::zero, |value| term.factor * value)
        })
    }

    #[inline]
    pub(crate) fn pair_at_columns(
        &self,
        column0: usize,
        column1: usize,
        coefficient: usize,
        coeff_count: usize,
    ) -> (E, E) {
        (
            self.get(column0, coefficient, coeff_count),
            self.get(column1, coefficient, coeff_count),
        )
    }

    #[inline]
    pub(crate) fn pair_flat(&self, index0: usize, index1: usize, coeff_count: usize) -> (E, E) {
        (
            self.get(index0 / coeff_count, index0 % coeff_count, coeff_count),
            self.get(index1 / coeff_count, index1 % coeff_count, coeff_count),
        )
    }

    pub(crate) fn quad_at(&self, column: usize, base: usize, coeff_count: usize) -> [E; 4] {
        std::array::from_fn(|offset| self.get(column, base + offset, coeff_count))
    }

    pub(crate) fn validate_len(&self, witness_len: usize) -> Result<(), AkitaError> {
        let actual = self
            .live_column_count
            .checked_mul(self.coeff_count)
            .ok_or_else(|| AkitaError::InvalidSetup("evaluation-trace length overflow".into()))?;
        if actual != witness_len {
            return Err(AkitaError::InvalidSize {
                expected: witness_len,
                actual,
            });
        }
        if self.column_terms.len() != self.live_column_count
            || self.sources.iter().any(|source| {
                source.values.len() != source.lane_count.saturating_mul(self.coeff_count)
            })
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    pub(crate) fn fold_y(&mut self, challenge: E) {
        let next_coeff_count = self.coeff_count / 2;
        for source in &mut self.sources {
            let mut folded = Vec::with_capacity(source.lane_count * next_coeff_count);
            for lane in 0..source.lane_count {
                let start = lane * self.coeff_count;
                for coefficient in 0..next_coeff_count {
                    let left = source.values[start + 2 * coefficient];
                    let right = source.values[start + 2 * coefficient + 1];
                    folded.push(left + challenge * (right - left));
                }
            }
            source.values = folded;
        }
        self.coeff_count = next_coeff_count;
    }

    pub(crate) fn fold_y2(&mut self, r0: E, r1: E) {
        self.fold_y(r0);
        self.fold_y(r1);
    }

    pub(crate) fn fold_x(&mut self, challenge: E) {
        let next_live_column_count = self.live_column_count.div_ceil(2);
        let mut folded = vec![Vec::new(); next_live_column_count];
        for (column, terms) in self.column_terms.drain(..).enumerate() {
            let scale = if column % 2 == 0 {
                E::one() - challenge
            } else {
                challenge
            };
            let target = &mut folded[column / 2];
            target.extend(terms.into_iter().map(|mut term| {
                term.factor *= scale;
                term
            }));
        }
        self.column_terms = folded;
        self.live_column_count = next_live_column_count;
    }

    pub(crate) fn fold_for_w_update(&mut self, challenge: E, folding_x_round: bool) {
        if folding_x_round {
            self.fold_x(challenge);
        } else {
            self.fold_y(challenge);
        }
    }

    #[cfg(test)]
    pub(crate) fn materialize_dense(&self) -> Vec<E> {
        (0..self.live_column_count)
            .flat_map(|column| {
                (0..self.coeff_count)
                    .map(move |coefficient| self.get(column, coefficient, self.coeff_count))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
