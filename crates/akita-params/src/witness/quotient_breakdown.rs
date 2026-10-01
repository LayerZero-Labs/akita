use akita_error::AkitaError;

use super::{RelationQuotientPlan, WitnessLayout};
use crate::{
    CommittedGroupParams, PolynomialGroupLayout, RelationWitnessGeometry, RingRelationMode,
};

/// Coefficients removed when a quotient-lift relation is realized by reduced
/// evaluation instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuotientCoefficientBreakdown {
    /// Quotient coefficients owned by ordinary opening-relation rows.
    pub ordinary: usize,
    /// Quotient coefficients owned by compression F/H relation rows.
    pub compression: usize,
}

impl QuotientCoefficientBreakdown {
    /// Derive the canonical quotient-lift counterfactual for reduced level
    /// parameters and classify the quotient coefficients by protocol owner.
    ///
    /// This keeps reporting and diagnostics on the same opening, compression,
    /// and row-ordering authority used to construct a runtime witness.
    pub fn for_reduced_counterfactual(
        params: &CommittedGroupParams,
        final_group: PolynomialGroupLayout,
        extension_degree: usize,
        field_bits: u32,
    ) -> Result<Self, AkitaError> {
        if !params.ring_relation_mode.is_reduced_evaluation() {
            return Err(AkitaError::InvalidInput(
                "quotient-lift counterfactual requires reduced-evaluation parameters".into(),
            ));
        }
        let mut lifted = params.clone();
        lifted.ring_relation_mode = RingRelationMode::QuotientLift;
        let opening_layout = lifted.opening_layout_for_final_group(final_group)?;
        let relation_geometry =
            RelationWitnessGeometry::for_level(&lifted, &opening_layout, extension_degree)?;
        let quotient_plan = RelationQuotientPlan::for_field_bits(&lifted, field_bits)?;
        let layout = WitnessLayout::new(
            &lifted,
            &opening_layout,
            &relation_geometry,
            lifted.witness_chunk.num_chunks,
            quotient_plan,
        )?;

        let mut compression_rows = vec![false; layout.r_rows().len()];
        for layer in layout.compression_layers() {
            for &(_, row) in layer.f_quotient_rows().into_iter().flatten() {
                let owned = compression_rows.get_mut(row).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "compression quotient owner is outside the witness row domain".into(),
                    )
                })?;
                *owned = true;
            }
            if let Some(row) = layer.h_quotient_row() {
                let owned = compression_rows.get_mut(row).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "compression quotient owner is outside the witness row domain".into(),
                    )
                })?;
                *owned = true;
            }
        }

        layout.r_rows().iter().enumerate().try_fold(
            Self::default(),
            |mut breakdown, (row, layout)| {
                let target = if compression_rows[row] {
                    &mut breakdown.compression
                } else {
                    &mut breakdown.ordinary
                };
                *target = target.checked_add(layout.range().len()).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "quotient coefficient breakdown length overflow".into(),
                    )
                })?;
                Ok(breakdown)
            },
        )
    }
}
