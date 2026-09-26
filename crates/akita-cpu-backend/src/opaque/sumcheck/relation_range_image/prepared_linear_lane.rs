//! Per-lane views over prepared structured linear terms.

use super::evaluation_trace::{
    PreparedLaneTerm, PreparedLaneWeights, PreparedPackingLaneMap, PreparedPackingSegment,
    PreparedProverLinearTerms, PreparedTraceSource,
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
/// Packing support is resolved once at the outer lane boundary.
pub(crate) struct PreparedLinearLane<'a, E: Field> {
    kind: PreparedLinearLaneKind<'a, E>,
}

impl<E: Field> PreparedPackingLaneMap<E> {
    /// The segment covering witness lane `lane`, and the source lane it reads.
    fn source_lane(&self, lane: usize) -> Option<(&PreparedPackingSegment<E>, usize)> {
        let segment = self
            .segments
            .get(self.lane_to_segment.get(lane).copied().flatten()?.get() - 1)?;
        let lane_offset = lane
            .checked_sub(segment.target_lane_start)
            .filter(|&offset| offset < segment.lane_count)?;
        Some((segment, segment.source_lane_start + lane_offset))
    }
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
}

impl<E: Field> PreparedProverLinearTerms<E> {
    /// Visit `(factor, source_index, source_lane)` for every source term of
    /// witness lane `lane`. Dense weights have no source terms.
    #[inline]
    pub(crate) fn for_each_source_term(&self, lane: usize, mut visit: impl FnMut(E, usize, usize)) {
        let source_lane_count =
            |source_index: usize| self.sources.get(source_index).map_or(0, |s| s.lane_count);
        match &self.lane_weights {
            PreparedLaneWeights::Dense(_) => {}
            PreparedLaneWeights::Packing(packing) => {
                if let Some((segment, source_lane)) =
                    packing.source_lane(lane).filter(|&(segment, source_lane)| {
                        source_lane < source_lane_count(segment.source_index)
                    })
                {
                    visit(segment.factor, segment.source_index, source_lane);
                }
            }
            PreparedLaneWeights::Sparse(lane_terms) => {
                for term in lane_terms.get(lane).map_or(&[][..], Vec::as_slice) {
                    if term.lane < source_lane_count(term.source_index) {
                        visit(term.factor, term.source_index, term.lane);
                    }
                }
            }
        }
    }

    #[inline]
    pub(crate) fn resolve_lane(&self, lane: usize) -> PreparedLinearLane<'_, E> {
        let kind = match &self.lane_weights {
            PreparedLaneWeights::Dense(dense) => dense
                .get(lane)
                .copied()
                .map(PreparedLinearLaneKind::Dense)
                .unwrap_or(PreparedLinearLaneKind::Zero),
            PreparedLaneWeights::Packing(packing) => packing
                .source_lane(lane)
                .and_then(|(segment, source_lane)| {
                    let start = source_lane * self.coeff_count;
                    let values = self
                        .sources
                        .get(segment.source_index)?
                        .values
                        .get(start..start + self.coeff_count)?;
                    Some(PreparedLinearLaneKind::Packing {
                        factor: segment.factor,
                        values,
                    })
                })
                .unwrap_or(PreparedLinearLaneKind::Zero),
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
