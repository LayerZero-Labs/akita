//! Prover-owned commitment kernels.

use crate::crt_ntt::NttSlotCache;
use crate::linear::mat_vec_mul_ntt_single_i8;
use crate::{HachiPolyOps, HachiProverSetup};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore, HachiError};
use akita_types::{FlatDigitBlocks, HachiCommitmentHint, LevelParams, RingCommitment};

/// Commit a group of polynomials using already-selected level parameters.
///
/// Root config/schedule policy chooses `params`; this function owns only the
/// prover-side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if an inner witness commitment or hint allocation fails.
pub fn commit_with_params<F, const D: usize, P>(
    polys: &[P],
    setup: &HachiProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let t_hat_flat_len_per_poly =
        params.num_blocks * params.a_key.row_len() * params.num_digits_open;
    let mut t_hat_flat = vec![[0i8; D]; polys.len() * t_hat_flat_len_per_poly];
    let mut t_hat_vec: Vec<FlatDigitBlocks<D>> = (0..polys.len())
        .map(|_| FlatDigitBlocks::new(Vec::new(), Vec::new()))
        .collect::<Result<_, _>>()?;
    let mut t_vec: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = vec![Vec::new(); polys.len()];
    cfg_chunks_mut!(t_hat_flat, t_hat_flat_len_per_poly)
        .zip(cfg_iter!(polys))
        .zip(cfg_iter_mut!(t_hat_vec))
        .zip(cfg_iter_mut!(t_vec))
        .try_for_each(|(((dst, poly), t_hat), t)| -> Result<(), HachiError> {
            let inner = poly.commit_inner_witness(
                &setup.expanded.shared_matrix,
                &setup.ntt_shared,
                params.a_key.row_len(),
                params.block_len,
                params.num_digits_commit,
                params.num_digits_open,
                params.log_basis,
                setup.expanded.seed.max_stride,
            )?;
            dst.copy_from_slice(inner.t_hat.flat_digits());
            *t_hat = inner.t_hat;
            *t = inner.t;
            Ok(())
        })?;
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        &setup.ntt_shared,
        params.b_key.row_len(),
        setup.expanded.seed.max_stride,
        &t_hat_flat,
    );
    Ok((
        RingCommitment { u },
        HachiCommitmentHint::with_t(t_hat_vec, t_vec),
    ))
}
