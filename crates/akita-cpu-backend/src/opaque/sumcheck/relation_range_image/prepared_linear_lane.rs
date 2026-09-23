//! Per-lane views over prepared structured linear terms.

use super::evaluation_trace::{
    PreparedLaneTerm, PreparedLaneWeights, PreparedPackingSegment, PreparedProverLinearTerms,
    PreparedTraceSource,
};
use jolt_field::Field;

enum PreparedLinearLaneKind<'a, E: Field> {
    Dense(E),
    Packing {
        factor: E,
        values: &'a [E],
    },
    Sparse {
        terms: &'a [PreparedLaneTerm<E>],
        sources: &'a [PreparedTraceSource<E>],
        coeff_count: usize,
    },
    Zero,
}

/// One resolved linear-term lane.
///
/// Packing support is resolved once at the outer lane boundary. Source
/// coefficients stay factored so linear kernels can accumulate a batch before
/// applying each source's common factor.
pub(crate) struct PreparedLinearLane<'a, E: Field> {
    kind: PreparedLinearLaneKind<'a, E>,
}

fn packing_segment_values<'a, E: Field>(
    segment: &PreparedPackingSegment<E>,
    sources: &'a [PreparedTraceSource<E>],
    lane: usize,
    coeff_count: usize,
) -> Option<(E, &'a [E])> {
    let lane_offset = lane.checked_sub(segment.target_lane_start)?;
    if lane_offset >= segment.lane_count {
        return None;
    }
    let source = sources.get(segment.source_index)?;
    let source_lane = segment.source_lane_start + lane_offset;
    let source_lane_start = source_lane * coeff_count;
    source
        .values
        .get(source_lane_start..source_lane_start + coeff_count)
        .map(|values| (segment.factor, values))
}

impl<E: Field> PreparedLinearLane<'_, E> {
    #[inline]
    pub(super) fn evaluated_values<const N: usize>(&self, coefficients: [usize; N]) -> [E; N] {
        match &self.kind {
            PreparedLinearLaneKind::Dense(value) => [*value; N],
            PreparedLinearLaneKind::Packing { factor, values } => {
                std::array::from_fn(|idx| *factor * values[coefficients[idx]])
            }
            PreparedLinearLaneKind::Sparse {
                terms,
                sources,
                coeff_count,
            } => {
                let mut values = [E::zero(); N];
                for term in *terms {
                    let Some(source) = sources.get(term.source_index) else {
                        continue;
                    };
                    let source_lane_start = term.lane * coeff_count;
                    for (value, coefficient) in values.iter_mut().zip(coefficients) {
                        if let Some(source_value) =
                            source.values.get(source_lane_start + coefficient)
                        {
                            *value += term.factor * *source_value;
                        }
                    }
                }
                values
            }
            PreparedLinearLaneKind::Zero => [E::zero(); N],
        }
    }

    #[inline]
    pub(crate) fn pair(&self, left: usize) -> (E, E) {
        let [left, right] = self.evaluated_values([left, left + 1]);
        (left, right)
    }

    #[inline]
    pub(crate) fn for_each_factored(&self, mut visit: impl FnMut(E, &[E])) {
        match &self.kind {
            PreparedLinearLaneKind::Dense(value) => visit(E::one(), std::slice::from_ref(value)),
            PreparedLinearLaneKind::Packing { factor, values } => visit(*factor, values),
            PreparedLinearLaneKind::Sparse {
                terms,
                sources,
                coeff_count,
            } => {
                for term in *terms {
                    let Some(source) = sources.get(term.source_index) else {
                        continue;
                    };
                    let source_lane_start = term.lane * coeff_count;
                    if let Some(values) = source
                        .values
                        .get(source_lane_start..source_lane_start + coeff_count)
                    {
                        visit(term.factor, values);
                    }
                }
            }
            PreparedLinearLaneKind::Zero => {}
        }
    }
}

impl<E: Field> PreparedProverLinearTerms<E> {
    #[inline]
    pub(crate) fn resolve_lane(&self, lane: usize) -> PreparedLinearLane<'_, E> {
        let kind = match &self.lane_weights {
            PreparedLaneWeights::Dense(dense) => dense
                .get(lane)
                .copied()
                .map(PreparedLinearLaneKind::Dense)
                .unwrap_or(PreparedLinearLaneKind::Zero),
            PreparedLaneWeights::Packing(packing) => {
                let Some(segment) = packing
                    .lane_to_segment
                    .get(lane)
                    .and_then(|segment| *segment)
                    .and_then(|segment| packing.segments.get(segment.get() - 1))
                else {
                    return PreparedLinearLane {
                        kind: PreparedLinearLaneKind::Zero,
                    };
                };
                let Some((factor, values)) =
                    packing_segment_values(segment, &self.sources, lane, self.coeff_count)
                else {
                    return PreparedLinearLane {
                        kind: PreparedLinearLaneKind::Zero,
                    };
                };
                PreparedLinearLaneKind::Packing { factor, values }
            }
            PreparedLaneWeights::Sparse(lane_terms) => lane_terms
                .get(lane)
                .filter(|terms| !terms.is_empty())
                .map(|terms| PreparedLinearLaneKind::Sparse {
                    terms,
                    sources: &self.sources,
                    coeff_count: self.coeff_count,
                })
                .unwrap_or(PreparedLinearLaneKind::Zero),
        };
        PreparedLinearLane { kind }
    }
}
