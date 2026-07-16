//! Dense polynomial inner commit.

use super::poly::DensePoly;
use crate::compute::{CommitmentComputeBackend, DenseCommitInput, DenseCommitRowsPlan};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};

impl<F> DensePoly<F>
where
    F: FieldCore + CanonicalField,
{
    pub(super) fn commit_rows<B, const D: usize>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup,
        n_a: usize,
        positions_per_block: usize,
        num_digits_commit: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let coeffs = self.ring_coeffs::<D>()?;
        let n = coeffs.len();
        let live_block_count = n.div_ceil(positions_per_block);

        if let Some(digit_planes) = self.digit_planes_for::<D>(num_digits_commit, log_basis) {
            let digit_block_slices =
                digit_block_slices(digit_planes, n, positions_per_block, num_digits_commit);
            return backend.dense_commit_rows(
                prepared,
                DenseCommitRowsPlan {
                    n_a,
                    input: DenseCommitInput::CachedDigits {
                        digit_block_slices,
                        log_basis,
                    },
                },
            );
        }

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..live_block_count)
            .map(|i| {
                let start = i * positions_per_block;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &coeffs[start..(start + positions_per_block).min(n)]
                }
            })
            .collect();

        backend.dense_commit_rows(
            prepared,
            DenseCommitRowsPlan {
                n_a,
                input: DenseCommitInput::CoeffBlocks {
                    block_slices,
                    num_digits_commit,
                    log_basis,
                },
            },
        )
    }
}

pub(super) fn digit_block_slices<const D: usize>(
    digit_planes: &[[i8; D]],
    num_rings: usize,
    positions_per_block: usize,
    num_digits: usize,
) -> Vec<&[[i8; D]]> {
    let live_block_count = num_rings.div_ceil(positions_per_block);
    (0..live_block_count)
        .map(|block_idx| {
            let ring_start = block_idx * positions_per_block;
            let ring_end = (ring_start + positions_per_block).min(num_rings);
            let digit_start = ring_start * num_digits;
            let digit_end = ring_end * num_digits;
            &digit_planes[digit_start..digit_end]
        })
        .collect()
}
