//! Dense polynomial inner commit.

use super::poly::DensePoly;
use crate::compute::{CommitmentComputeBackend, DenseCommitInput, DenseCommitRowsPlan};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};

impl<F, const D: usize> DensePoly<F, D>
where
    F: FieldCore + CanonicalField,
{
    pub(super) fn commit_rows<B>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup,
        n_a: usize,
        block_len: usize,
        num_digits_commit: usize,
        log_basis: u32,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let n = self.coeffs.len();
        let num_blocks = n.div_ceil(block_len);

        if let Some(digit_planes) = self.digit_planes_for(num_digits_commit, log_basis) {
            let digit_block_slices =
                digit_block_slices(digit_planes, n, block_len, num_digits_commit);
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

        let block_slices: Vec<&[CyclotomicRing<F, D>]> = (0..num_blocks)
            .map(|i| {
                let start = i * block_len;
                if start >= n {
                    &[] as &[CyclotomicRing<F, D>]
                } else {
                    &self.coeffs[start..(start + block_len).min(n)]
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
    block_len: usize,
    num_digits: usize,
) -> Vec<&[[i8; D]]> {
    let num_blocks = num_rings.div_ceil(block_len);
    (0..num_blocks)
        .map(|block_idx| {
            let ring_start = block_idx * block_len;
            let ring_end = (ring_start + block_len).min(num_rings);
            let digit_start = ring_start * num_digits;
            let digit_end = ring_end * num_digits;
            &digit_planes[digit_start..digit_end]
        })
        .collect()
}
