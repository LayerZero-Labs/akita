use super::{RecursiveWitnessFlat, SuffixWitnessBatchView};
use crate::arithmetic::coefficient_packing::{
    coefficient_packing_partials_from_position_source, FusedPackingWeights,
};
use crate::opaque::{
    CpuBackend, SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
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
    let point = fused_weights.point();
    let view = witness.view::<F, D>()?;
    if view.live_ring_elems != point.num_live_positions() {
        return Err(AkitaError::InvalidSize {
            expected: point.num_live_positions(),
            actual: view.live_ring_elems,
        });
    }
    let coordinates = coefficient_packing_partials_from_position_source::<F, E, _, D>(
        fused_weights,
        view.num_vars(),
        |position| view.ring_elem(position).ok_or(AkitaError::InvalidProof),
        |position, coefficient_index, source| {
            let flat_index = position * D + coefficient_index;
            if flat_index < view.live_coeff_len {
                F::from_i8(source[coefficient_index])
            } else {
                F::zero()
            }
        },
    )?;
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
        let fused_weights = FusedPackingWeights::new(plan.point)?;
        source
            .polys
            .iter()
            .map(|witness| {
                suffix_witness_coefficient_packing_partials::<F, E, D>(witness, &fused_weights)
            })
            .collect()
    }
}
