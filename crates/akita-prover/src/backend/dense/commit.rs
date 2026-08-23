//! Dense polynomial inner commit.

use super::poly::DensePoly;
use crate::compute::{CpuBackend, CpuPreparedSetup, DenseCommitInput, PackedDenseCommitInput};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_field::{CanonicalField, FieldCore};

impl<F> DensePoly<F>
where
    F: FieldCore + CanonicalField,
{
    pub(super) fn commit_rows<const D: usize>(
        &self,
        backend: &CpuBackend,
        prepared: &CpuPreparedSetup<F>,
        n_a: usize,
        num_positions_per_block: usize,
        num_digits_inner: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        let coeffs = self.ring_coeffs::<D>()?;
        let n = coeffs.len();
        let num_live_blocks = n.div_ceil(num_positions_per_block);

        if let Some(digit_planes) = self.digit_planes_for::<D>(num_digits_inner, log_basis) {
            return backend.dense_commit_rows(
                prepared,
                n_a,
                DenseCommitInput::PackedDigits {
                    source: PackedDenseCommitInput::new::<D>(
                        digit_planes,
                        n,
                        num_positions_per_block,
                        num_digits_inner,
                    )?,
                    log_basis_inner: log_basis,
                },
            );
        }

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_live_blocks)
            .map(|i| {
                let start = i * num_positions_per_block;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &coeffs[start..(start + num_positions_per_block).min(n)]
                }
            })
            .collect();

        backend.dense_commit_rows(
            prepared,
            n_a,
            DenseCommitInput::CoeffBlocks {
                block_slices,
                num_digits_inner,
                log_basis_inner: log_basis,
            },
        )
    }
}
