use super::*;

pub(super) fn build_physical_b_weight_segments<E: Field>(
    geometry: &crate::CommitmentSliceGeometry,
    physical_rows: usize,
    logical_row_weights: &[E],
) -> Result<Vec<PhysicalBWeightSegment<E>>, AkitaError> {
    let logical_rows = geometry.logical_output_rows(physical_rows)?;
    if logical_row_weights.len() != logical_rows {
        return Err(AkitaError::InvalidSize {
            expected: logical_rows,
            actual: logical_row_weights.len(),
        });
    }
    let block_width = geometry.ring_elements_per_block_per_polynomial();
    let physical_polynomial_stride = geometry
        .max_blocks_per_slice()
        .checked_mul(block_width)
        .ok_or_else(|| AkitaError::InvalidSetup("physical B stride overflow".into()))?;
    let mut width_boundaries = geometry
        .block_ranges()
        .iter()
        .map(|range| {
            range
                .len()
                .checked_mul(block_width)
                .ok_or_else(|| AkitaError::InvalidSetup("B slice width overflow".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    width_boundaries.push(0);
    width_boundaries.sort_unstable();
    width_boundaries.dedup();

    let mut segments = Vec::new();
    for physical_row in 0..physical_rows {
        let physical_row_start = physical_row
            .checked_mul(geometry.physical_input_width())
            .ok_or_else(|| AkitaError::InvalidSetup("physical B row overflow".into()))?;
        for polynomial in 0..geometry.num_polynomials() {
            let physical_polynomial_start = physical_row_start
                .checked_add(
                    polynomial
                        .checked_mul(physical_polynomial_stride)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("physical B polynomial offset overflow".into())
                        })?,
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("physical B polynomial extent overflow".into())
                })?;
            for boundary in width_boundaries.windows(2) {
                let [start, end] = boundary else {
                    return Err(AkitaError::InvalidSetup(
                        "physical B width boundary is malformed".into(),
                    ));
                };
                if start == end {
                    continue;
                }
                let mut terms = Vec::new();
                for (slice_index, block_range) in geometry.block_ranges().iter().enumerate() {
                    let slice_width = block_range
                        .len()
                        .checked_mul(block_width)
                        .ok_or_else(|| AkitaError::InvalidSetup("B slice width overflow".into()))?;
                    if slice_width < *end {
                        continue;
                    }
                    let logical_row =
                        geometry.logical_row_index(slice_index, physical_row, physical_rows)?;
                    let row_weight = *logical_row_weights
                        .get(logical_row)
                        .ok_or(AkitaError::InvalidProof)?;
                    let logical_start = polynomial
                        .checked_mul(geometry.num_live_blocks())
                        .and_then(|base| base.checked_add(block_range.start))
                        .and_then(|block| block.checked_mul(block_width))
                        .and_then(|base| base.checked_add(*start))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("logical B column offset overflow".into())
                        })?;
                    terms.push(PhysicalBWeightTerm {
                        logical_start,
                        row_weight,
                    });
                }
                if terms.is_empty() {
                    return Err(AkitaError::InvalidSetup(
                        "physical B segment has no logical source".into(),
                    ));
                }
                segments.push(PhysicalBWeightSegment {
                    physical_start: physical_polynomial_start.checked_add(*start).ok_or_else(
                        || AkitaError::InvalidSetup("physical B segment offset overflow".into()),
                    )?,
                    len: end - start,
                    terms: terms.into(),
                });
            }
        }
    }
    Ok(segments)
}

impl<E: Field> PhysicalBSetupPlan<E> {
    /// Contract logical slice-major B row and column weights onto the one
    /// physical B matrix. The logical columns remain polynomial-major, while
    /// each slice is padded independently to the physical matrix width.
    pub fn contract_logical_column_weights(
        &self,
        logical_column_weights: &[E],
    ) -> Result<Vec<E>, AkitaError> {
        if logical_column_weights.len() != self.geometry().logical_input_width() {
            return Err(AkitaError::InvalidSize {
                expected: self.geometry().logical_input_width(),
                actual: logical_column_weights.len(),
            });
        }
        let mut physical = vec![E::zero(); self.physical_footprint()?];
        for segment in self.weight_segments() {
            let end = segment
                .physical_start
                .checked_add(segment.len)
                .ok_or_else(|| AkitaError::InvalidSetup("physical B segment overflow".into()))?;
            let target = physical
                .get_mut(segment.physical_start..end)
                .ok_or(AkitaError::InvalidProof)?;
            for (offset, target) in target.iter_mut().enumerate() {
                for term in segment.terms.iter() {
                    let logical = term
                        .logical_start
                        .checked_add(offset)
                        .and_then(|index| logical_column_weights.get(index))
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                    *target += term.row_weight * logical;
                }
            }
        }
        Ok(physical)
    }
}
