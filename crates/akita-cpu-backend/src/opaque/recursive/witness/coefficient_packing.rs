use super::{RecursiveWitnessFlat, SuffixWitnessBatchView};
use crate::arithmetic::coefficient_packing::{
    coefficient_packing_partials_from_digit_source, FusedPackingWeights, MAX_SUMMED_DIGIT,
};
use crate::opaque::{
    CpuBackend, SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use crate::sources::block_digits::{PackedDigits, RingValueSource};
use crate::sources::dense::seam_packing_partials;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

pub(crate) fn suffix_witness_coefficient_packing_partials<F, E, const D: usize>(
    witness: &RecursiveWitnessFlat,
    fused_weights: &FusedPackingWeights<'_, E>,
) -> Result<SubringCoefficientPackingPartials<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    let point = fused_weights.tables().point();
    let view = witness.view::<F, D>()?;
    if view.live_ring_elems != point.num_live_positions() {
        return Err(AkitaError::InvalidSize {
            expected: point.num_live_positions(),
            actual: view.live_ring_elems,
        });
    }
    let digits = PackedDigits {
        digits: view.digits,
        num_rings: view.live_ring_elems,
    };
    let bounds = view.digits.bounds();
    let digit_abs_bound = bounds.negative_abs_max().max(bounds.positive_max());
    let coordinates = if digit_abs_bound <= MAX_SUMMED_DIGIT {
        let live_coeff_len = view.live_coeff_len;
        coefficient_packing_partials_from_digit_source::<F, E, _, D>(
            fused_weights,
            view.num_vars(),
            live_coeff_len.div_ceil(D),
            digit_abs_bound,
            |position| RingValueSource::<F, D>::values(&digits, position),
            |position, coefficient, values: &[i8; D]| {
                // The last live ring may end inside the stored digits.
                if position * D + coefficient < live_coeff_len {
                    values[coefficient]
                } else {
                    0
                }
            },
        )?
    } else {
        seam_packing_partials::<F, E, _, D>(
            &digits,
            view.live_coeff_len,
            view.num_vars(),
            fused_weights,
        )?
    };
    SubringCoefficientPackingPartials::new(point.geometry(), point.num_live_blocks(), coordinates)
}

impl<F, E, const D: usize>
    SubringCoefficientPackingBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D>
    for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let fused_weights = FusedPackingWeights::new(plan.tables)?;
        source
            .polys
            .iter()
            .map(|witness| {
                suffix_witness_coefficient_packing_partials::<F, E, D>(witness, &fused_weights)
            })
            .collect()
    }
}
