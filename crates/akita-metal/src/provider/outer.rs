//! The outer (B) commitment stage on the device.

use std::time::Instant;

use akita_cpu_backend::commitment_backend::{
    InnerImage, InnerImageInput, OuterCommitOperation, StateOwnerCapability, UncompressedCommitPlan,
};
use akita_error::{checked, AkitaError};
use akita_types::RingVec;
use jolt_metal::runtime::DeviceBuffer;

use super::inner::MetalInnerImage;
use super::{record, CommitmentField, MetalCommitmentProvider};
use crate::decompose::decompose;
use crate::error::AkitaMetalError;

/// The outer stage operation.
pub(super) struct MetalOuterCommit<'a, F: CommitmentField> {
    provider: &'a MetalCommitmentProvider<F>,
    owner: StateOwnerCapability<InnerImage>,
}

impl<'a, F: CommitmentField> MetalOuterCommit<'a, F> {
    pub(super) fn new(
        provider: &'a MetalCommitmentProvider<F>,
        owner: StateOwnerCapability<InnerImage>,
    ) -> Self {
        Self { provider, owner }
    }
}

impl<F: CommitmentField> OuterCommitOperation<F> for MetalOuterCommit<'_, F> {
    fn supports_plan(&self, plan: &UncompressedCommitPlan) -> bool {
        super::outer_supported::<F>(
            plan.inner().ring_dimension,
            plan.outer().ring_dimension(),
            plan.outer().log_basis_outer(),
        )
    }

    fn commit_outer(
        &self,
        plan: &UncompressedCommitPlan,
        inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError> {
        let start = Instant::now();
        let (rows, device) = match inner {
            InnerImageInput::Owned(image) => {
                if image.binding().inner_plan() != plan.inner() {
                    return Err(AkitaError::InvalidInput(
                        "the inner image was committed for a different plan".into(),
                    ));
                }
                let image = self.owner.value::<MetalInnerImage<F>>(image)?;
                (
                    image.rows.as_slice(),
                    image.device.as_ref().map(|device| device.get()),
                )
            }
            InnerImageInput::HostRows(rows) => (rows, None),
        };
        let u = with_ring_degree!(plan.outer().ring_dimension(), |D_B| self
            .provider
            .commit_outer::<D_B>(plan, rows, device))?;
        let counters = &self.provider.counters;
        record(&counters.outer, &counters.outer_nanos, start);
        Ok(u)
    }
}

impl<F: CommitmentField> MetalCommitmentProvider<F> {
    /// Arranges `t` into the dyadic slices, decomposes each `D_A` ring as
    /// `D_A / D_B` subrings of outer digits, and multiplies by B.
    ///
    /// The slice input of slice `s` is, for every polynomial in order, its
    /// blocks in the slice's range followed by zero blocks up to the widest
    /// slice (`for_each_outer_slice_input`); zero coefficients decompose to
    /// zero digits, so padding `t` pads the digit planes. When `t` already has
    /// that layout (one polynomial, equal slices) the device copy is used as
    /// is.
    fn commit_outer<const D_B: usize>(
        &self,
        plan: &UncompressedCommitPlan,
        rows: &[RingVec<F>],
        device_rows: Option<&DeviceBuffer<F>>,
    ) -> Result<RingVec<F>, AkitaError> {
        let overflow = || AkitaError::InvalidSetup("outer commitment shape overflows".into());
        let invalid =
            |what: &str| AkitaError::InvalidSetup(format!("the Metal outer stage received {what}"));
        let (inner, outer) = (plan.inner(), plan.outer());
        let geometry = outer.geometry();
        let d_a = inner.ring_dimension;
        let subrings = checked::exact_div(d_a, D_B).ok_or_else(overflow)?;
        let digits = outer.num_digits_outer();
        let block_fields = checked::product([inner.n_a, d_a]).ok_or_else(overflow)?;
        let row_fields =
            checked::product([inner.num_live_blocks, block_fields]).ok_or_else(overflow)?;
        if geometry.num_polynomials() != rows.len()
            || geometry.num_live_blocks() != inner.num_live_blocks
            || checked::product([inner.n_a, subrings, digits])
                != Some(geometry.ring_elements_per_block_per_polynomial())
        {
            return Err(invalid(
                "a slice geometry that disagrees with its inner plan",
            ));
        }
        if rows
            .iter()
            .any(|row| row.ring_dim() != d_a || row.coeff_len() != row_fields)
        {
            return Err(invalid("inner rows of the wrong shape"));
        }
        let max_blocks = geometry.max_blocks_per_slice();
        let ranges = geometry.block_ranges();
        let slice_fields =
            checked::product([rows.len(), max_blocks, block_fields]).ok_or_else(overflow)?;
        let staged_len = checked::product([ranges.len(), slice_fields]).ok_or_else(overflow)?;
        let cols = checked::product([rows.len(), max_blocks, inner.n_a, subrings, digits])
            .ok_or_else(overflow)?;
        if cols != geometry.physical_input_width() {
            return Err(invalid("a physical B width that disagrees with its slices"));
        }

        let metal = self.metal.get();
        let unpadded = rows.len() == 1 && ranges.iter().all(|range| range.len() == max_blocks);
        let staged;
        let coefficients = match device_rows {
            Some(device) if unpadded && device.len() == staged_len => device,
            _ => {
                let mut host = vec![F::zero(); staged_len];
                for (slice, range) in ranges.iter().enumerate() {
                    for (polynomial, row) in rows.iter().enumerate() {
                        let source = checked::product([range.start, block_fields])
                            .zip(checked::product([range.end, block_fields]))
                            .and_then(|(start, end)| row.coeffs().get(start..end))
                            .ok_or_else(|| invalid("a slice past the inner rows"))?;
                        let start = checked::product([slice, rows.len()])
                            .and_then(|index| index.checked_add(polynomial))
                            .and_then(|index| index.checked_mul(max_blocks))
                            .and_then(|block| block.checked_mul(block_fields))
                            .ok_or_else(overflow)?;
                        host.get_mut(start..)
                            .and_then(|tail| tail.get_mut(..source.len()))
                            .ok_or_else(|| invalid("a slice wider than the widest slice"))?
                            .copy_from_slice(source);
                    }
                }
                staged = DeviceBuffer::from_slice(metal.device(), &host)
                    .map_err(AkitaMetalError::from)?;
                &staged
            }
        };
        let mut planes = DeviceBuffer::<i8>::zeroed(
            metal.device(),
            checked::product([staged_len, digits]).ok_or_else(overflow)?,
        )
        .map_err(AkitaMetalError::from)?;
        decompose(
            metal,
            coefficients,
            D_B,
            digits,
            outer.log_basis_outer(),
            &mut planes,
        )?;
        let matrix = self.matrices.ntt::<D_B>(
            metal,
            self.expanded.shared_matrix(),
            outer.n_b(),
            cols,
            outer.log_basis_outer(),
        )?;
        let expected = outer.output_coefficient_len()?;
        if checked::product([ranges.len(), outer.n_b(), D_B]) != Some(expected) {
            return Err(invalid("an output length that disagrees with its slices"));
        }
        let mut out =
            DeviceBuffer::<F>::zeroed(metal.device(), expected).map_err(AkitaMetalError::from)?;
        matrix.mat_vec_i8(metal, &planes, outer.log_basis_outer(), &mut out)?;
        let u = out.read().map_err(AkitaMetalError::from)?.to_vec();
        RingVec::from_coeffs_with_ring_dim(u, D_B)
    }
}
