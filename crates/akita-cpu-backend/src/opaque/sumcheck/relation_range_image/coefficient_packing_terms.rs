//! CPU task planning for validated coefficient-packing Stage 2 semantics.

use std::ops::Range;

use akita_error::{checked, AkitaError};
use akita_types::CoefficientPackingGroupSemantics;
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

use super::PreparedProverLinearTerms;

/// Packing-Z positions each parallel CPU task materializes.
const PACKING_Z_POSITIONS_PER_TASK: usize = 1 << 10;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CpuCoefficientPackingSource {
    DirectOpening,
    PackingZ,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CpuCoefficientPackingSegment {
    physical_coefficients: Range<usize>,
    source_coefficients: Range<usize>,
}

impl CpuCoefficientPackingSegment {
    pub(super) fn physical_coefficients(&self) -> Range<usize> {
        self.physical_coefficients.clone()
    }

    pub(super) fn source_coefficients(&self) -> Range<usize> {
        self.source_coefficients.clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CpuCoefficientPackingTerm<E: Field> {
    source: CpuCoefficientPackingSource,
    factor: E,
    segments: Range<usize>,
}

impl<E: Field> CpuCoefficientPackingTerm<E> {
    pub(super) const fn source(&self) -> CpuCoefficientPackingSource {
        self.source
    }

    pub(super) const fn factor(&self) -> E {
        self.factor
    }

    pub(super) fn segments(&self) -> Range<usize> {
        self.segments.clone()
    }
}

/// CPU-owned source buffers and task-expanded support for one group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CpuCoefficientPackingTerms<E: Field> {
    sources: [Vec<E>; 2],
    segments: Vec<CpuCoefficientPackingSegment>,
    terms: Vec<CpuCoefficientPackingTerm<E>>,
    physical_field_len: usize,
    relation_coefficient_block_len: usize,
}

impl<E: Field> CpuCoefficientPackingTerms<E> {
    fn reserve<T>(values: &mut Vec<T>, count: usize, what: &str) -> Result<(), AkitaError> {
        values
            .try_reserve_exact(count)
            .map_err(|_| AkitaError::InvalidInput(format!("{what} allocation failed")))
    }

    fn new(semantics: &CoefficientPackingGroupSemantics<'_, E>) -> Result<Self, AkitaError> {
        let geometry = semantics.geometry();
        let prepared_point = semantics.prepared_point();
        let witness_units = semantics.witness_units();
        let group_claim_coefficients = semantics.group_claim_coefficients();
        let basis = semantics.basis();
        let opening_gadget = semantics.opening_gadget();
        let witness_gadget = semantics.witness_gadget();
        let fold_gadget = semantics.fold_gadget();
        let d_d = semantics.d_d();
        let d_a = geometry.a_ring_dimension();

        let mut direct_opening_source = Vec::new();
        Self::reserve(
            &mut direct_opening_source,
            geometry.partial_base_field_width(),
            "direct-opening source",
        )?;
        for &basis_element in basis {
            direct_opening_source.extend(
                prepared_point
                    .tail_weights()
                    .iter()
                    .map(|&tail_weight| basis_element * tail_weight),
            );
        }

        let mut packing_z_source = Vec::new();
        Self::reserve(&mut packing_z_source, d_a, "packing-Z source")?;
        packing_z_source.resize(d_a, E::zero());
        for (low_index, &packing_weight) in prepared_point.packing_weights().iter().enumerate() {
            for (subring_index, &alpha_power) in semantics.alpha_powers().iter().enumerate() {
                let physical = geometry.a_ring_coefficient_index(low_index, subring_index)?;
                *packing_z_source.get_mut(physical).ok_or_else(|| {
                    AkitaError::Internal(
                        "packing Z source coordinate is outside its geometry".into(),
                    )
                })? = packing_weight * alpha_power;
            }
        }

        let direct_term_capacity = checked::product([
            group_claim_coefficients.len(),
            prepared_point.num_live_blocks(),
            opening_gadget.len(),
        ])
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "coefficient-packing direct-opening term count overflow".into(),
            )
        })?;
        let segments_per_direct_term = geometry.partial_base_field_width() / d_d;
        let direct_segment_capacity = direct_term_capacity
            .checked_mul(segments_per_direct_term)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("direct-opening segment count overflow".into())
            })?;
        let terms_per_z_position = checked::product([witness_gadget.len(), fold_gadget.len()])
            .ok_or_else(|| {
                AkitaError::InvalidSetup("coefficient-packing packing-Z term count overflow".into())
            })?;
        let z_term_capacity = checked::product([
            witness_units.len(),
            prepared_point.position_weights().len(),
            terms_per_z_position,
        ])
        .ok_or_else(|| {
            AkitaError::InvalidSetup("coefficient-packing packing-Z term count overflow".into())
        })?;
        let segment_capacity = direct_segment_capacity
            .checked_add(z_term_capacity)
            .ok_or_else(|| AkitaError::InvalidSetup("packing segment count overflow".into()))?;
        let term_capacity = direct_term_capacity
            .checked_add(z_term_capacity)
            .ok_or_else(|| AkitaError::InvalidSetup("packing term count overflow".into()))?;
        let mut segments = Vec::new();
        Self::reserve(&mut segments, segment_capacity, "packing segment")?;
        let mut terms = Vec::new();
        Self::reserve(&mut terms, term_capacity, "packing term")?;

        for (claim, &claim_coefficient) in group_claim_coefficients.iter().enumerate() {
            for unit in witness_units {
                let first_segment = segments.len();
                let block_terms = cfg_into_iter!(unit.global_block_range())
                    .map(|global_block| {
                        let block_weight = *(prepared_point
                            .live_block_weights()
                            .get(global_block)
                            .ok_or_else(|| {
                                AkitaError::Internal(
                                    "packing opening block has no prepared weight".into(),
                                )
                            })?);
                        let block_segment_start = (global_block - unit.global_block_start())
                            .checked_mul(opening_gadget.len())
                            .and_then(|term| term.checked_mul(segments_per_direct_term))
                            .and_then(|offset| offset.checked_add(first_segment))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("direct-opening segment overflow".into())
                            })?;
                        let block_segment_capacity = opening_gadget
                            .len()
                            .checked_mul(segments_per_direct_term)
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "direct-opening segment count overflow".into(),
                                )
                            })?;
                        let mut block_segments = Vec::new();
                        Self::reserve(
                            &mut block_segments,
                            block_segment_capacity,
                            "direct-opening block segment",
                        )?;
                        let mut block_terms = Vec::new();
                        Self::reserve(
                            &mut block_terms,
                            opening_gadget.len(),
                            "direct-opening block term",
                        )?;
                        for (digit, &gadget) in opening_gadget.iter().enumerate() {
                            let segment_start = block_segment_start
                                .checked_add(block_segments.len())
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "direct-opening segment overflow".into(),
                                    )
                                })?;
                            for role_subcolumn in 0..segments_per_direct_term {
                                let physical_start = unit.e_coefficient_index(
                                    d_d,
                                    group_claim_coefficients.len(),
                                    semantics.num_digits_open(),
                                    claim,
                                    global_block,
                                    role_subcolumn,
                                    digit,
                                    0,
                                )?;
                                let source_start =
                                    role_subcolumn.checked_mul(d_d).ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "direct-opening source overflow".into(),
                                        )
                                    })?;
                                let physical_end =
                                    physical_start.checked_add(d_d).ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "direct-opening segment overflow".into(),
                                        )
                                    })?;
                                let source_end =
                                    source_start.checked_add(d_d).ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "direct-opening source overflow".into(),
                                        )
                                    })?;
                                block_segments.push(CpuCoefficientPackingSegment {
                                    physical_coefficients: physical_start..physical_end,
                                    source_coefficients: source_start..source_end,
                                });
                            }
                            let segment_end = segment_start
                                .checked_add(
                                    block_segments.len() - (digit * segments_per_direct_term),
                                )
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "direct-opening segment overflow".into(),
                                    )
                                })?;
                            block_terms.push(CpuCoefficientPackingTerm {
                                source: CpuCoefficientPackingSource::DirectOpening,
                                factor: semantics.scalar_claim_weight()
                                    * claim_coefficient
                                    * block_weight
                                    * gadget,
                                segments: segment_start..segment_end,
                            });
                        }
                        Ok((block_segments, block_terms))
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()?;
                for (block_segments, block_terms) in block_terms {
                    segments.extend(block_segments);
                    terms.extend(block_terms);
                }
            }
        }

        let position_weights = prepared_point.position_weights();
        for unit in witness_units {
            let first_segment = segments.len();
            let task_terms = cfg_into_iter!(
                0..position_weights
                    .len()
                    .div_ceil(PACKING_Z_POSITIONS_PER_TASK)
            )
            .map(|task| {
                let first_position =
                    task.checked_mul(PACKING_Z_POSITIONS_PER_TASK)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("packing-Z task position overflow".into())
                        })?;
                let end_position = first_position
                    .saturating_add(PACKING_Z_POSITIONS_PER_TASK)
                    .min(position_weights.len());
                let positions = first_position..end_position;
                let task_len = positions
                    .len()
                    .checked_mul(terms_per_z_position)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("packing-Z task size overflow".into())
                    })?;
                let mut task_segments = Vec::new();
                Self::reserve(&mut task_segments, task_len, "packing-Z task segment")?;
                let mut task_terms = Vec::new();
                Self::reserve(&mut task_terms, task_len, "packing-Z task term")?;
                for position in positions {
                    let position_weight = position_weights[position];
                    for (witness_digit, &witness_weight) in witness_gadget.iter().enumerate() {
                        for (fold_digit, &fold_weight) in fold_gadget.iter().enumerate() {
                            let physical_start = unit.z_coefficient_index(
                                d_a,
                                semantics.num_positions_per_block(),
                                semantics.num_digits_inner(),
                                semantics.num_digits_fold(),
                                position,
                                witness_digit,
                                fold_digit,
                                0,
                            )?;
                            let segment = first_segment
                                .checked_add(
                                    first_position
                                        .checked_mul(terms_per_z_position)
                                        .ok_or_else(|| {
                                            AkitaError::InvalidSetup(
                                                "packing-Z segment overflow".into(),
                                            )
                                        })?,
                                )
                                .and_then(|start| start.checked_add(task_segments.len()))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("packing-Z segment overflow".into())
                                })?;
                            let segment_end = physical_start.checked_add(d_a).ok_or_else(|| {
                                AkitaError::InvalidSetup("packing-Z segment overflow".into())
                            })?;
                            task_segments.push(CpuCoefficientPackingSegment {
                                physical_coefficients: physical_start..segment_end,
                                source_coefficients: 0..d_a,
                            });
                            let term_end = segment.checked_add(1).ok_or_else(|| {
                                AkitaError::InvalidSetup("packing-Z term segment overflow".into())
                            })?;
                            task_terms.push(CpuCoefficientPackingTerm {
                                source: CpuCoefficientPackingSource::PackingZ,
                                factor: -(semantics.consistency_weight()
                                    * position_weight
                                    * witness_weight
                                    * fold_weight),
                                segments: segment..term_end,
                            });
                        }
                    }
                }
                Ok((task_segments, task_terms))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
            for (task_segments, task_terms) in task_terms {
                segments.extend(task_segments);
                terms.extend(task_terms);
            }
        }

        Ok(Self {
            sources: [direct_opening_source, packing_z_source],
            segments,
            terms,
            physical_field_len: semantics.physical_field_len(),
            relation_coefficient_block_len: semantics.relation_coefficient_block_len(),
        })
    }

    pub(super) const fn physical_field_len(&self) -> usize {
        self.physical_field_len
    }

    pub(super) const fn relation_coefficient_block_len(&self) -> usize {
        self.relation_coefficient_block_len
    }

    pub(super) fn into_linear_parts(
        self,
    ) -> (
        [Vec<E>; 2],
        Vec<CpuCoefficientPackingSegment>,
        Vec<CpuCoefficientPackingTerm<E>>,
    ) {
        (self.sources, self.segments, self.terms)
    }
}

/// Materialize CPU-owned Stage 2 tasks from the validated shared semantics.
pub(crate) fn prepare_coefficient_packing_linear_terms<E: Field>(
    semantics: CoefficientPackingGroupSemantics<'_, E>,
) -> Result<PreparedProverLinearTerms<E>, AkitaError> {
    PreparedProverLinearTerms::from_coefficient_packing(CpuCoefficientPackingTerms::new(
        &semantics,
    )?)
}

#[cfg(test)]
#[path = "coefficient_packing_terms_tests.rs"]
mod tests;
