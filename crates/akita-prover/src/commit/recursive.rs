//! Recursive `commit_w` core — the same `A`-commit → decompose → `B`/`F` shape
//! as the root path, with `SuffixWitness` as the opening source.

use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore, RandomSampling};
use akita_types::{AkitaCommitmentHint, AkitaExpandedSetup, LevelParams, RingCommitment};

use crate::backend::RecursiveWitnessFlat;
use crate::commit::ajtai::backend::CommitBackend;
use crate::commit::inner::commit_inner_one;
use crate::commit::outer::outer_commit;
use crate::commit::pipeline::{validate_commit_level_params, validate_commit_outer_input_nonempty};
#[cfg(feature = "zk")]
use crate::protocol::masking::sample_blinding_digits;

/// Commit the D-agnostic ring-switch witness `w` at the caller-selected ring
/// dimension.
///
/// # Errors
///
/// Returns an error if the witness length is not divisible by `D` or if the
/// recursive inner commitment fails.
#[tracing::instrument(skip_all, name = "commit_w")]
#[inline(never)]
pub fn commit_w<F, B, const D: usize>(
    w: &RecursiveWitnessFlat,
    expanded: &AkitaExpandedSetup<F>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    commit_layout: &LevelParams,
) -> Result<(RingCommitment<F, D>, AkitaCommitmentHint<F, D>), AkitaError>
where
    F: FieldCore + CanonicalField + RandomSampling + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    B: CommitBackend<F>,
{
    if commit_layout.ring_dimension != D {
        return Err(AkitaError::InvalidInput(format!(
            "commit_w layout D={} does not match target D={D}",
            commit_layout.ring_dimension
        )));
    }
    if !w.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: w.len(),
        });
    }
    backend.validate_prepared_setup::<D>(prepared, expanded)?;
    validate_commit_level_params::<F, D>(commit_layout, expanded)?;

    let num_ring_elems = w.len() / D;
    tracing::debug!(
        num_ring_elems,
        num_blocks = commit_layout.num_blocks,
        block_len = commit_layout.block_len,
        depth_commit = commit_layout.num_digits_commit,
        depth_open = commit_layout.num_digits_open,
        m_vars = commit_layout.m_vars,
        r_vars = commit_layout.r_vars,
        inner_width = commit_layout.inner_width(),
        pow2_block = 1usize << commit_layout.m_vars,
        "commit_w layout"
    );

    let w_view = w.view::<F, D>()?;
    let inner = commit_inner_one::<F, D, _, B>(&w_view, backend, prepared, commit_layout)?;

    let outer_input = inner.decomposed_inner_rows.flat_digits().to_vec();
    validate_commit_outer_input_nonempty(outer_input.len())?;

    #[cfg(feature = "zk")]
    let b_blinding_digits =
        sample_blinding_digits::<F, D>(commit_layout.b_key.row_len(), commit_layout.log_basis)?;
    #[cfg(feature = "zk")]
    let mut u: Vec<CyclotomicRing<F, D>> =
        outer_commit::<F, D, B>(backend, prepared, commit_layout, &outer_input)?;
    #[cfg(not(feature = "zk"))]
    let u: Vec<CyclotomicRing<F, D>> =
        outer_commit::<F, D, B>(backend, prepared, commit_layout, &outer_input)?;
    // ZK blinding only applies to the single-tier B image; tiered proofs are
    // exercised non-zk.
    #[cfg(feature = "zk")]
    if commit_layout.f_key.is_none() {
        let blinding_rows = backend.zk_b_digit_rows::<D>(
            prepared,
            commit_layout.b_key.row_len(),
            b_blinding_digits.flat_digits().len(),
            b_blinding_digits.flat_digits(),
        )?;
        for (row, blinding) in u.iter_mut().zip(blinding_rows) {
            *row += blinding;
        }
    }

    #[cfg(feature = "zk")]
    let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
        inner.decomposed_inner_rows,
        inner.recomposed_inner_rows,
        b_blinding_digits,
    );
    #[cfg(not(feature = "zk"))]
    let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
        inner.decomposed_inner_rows,
        inner.recomposed_inner_rows,
    );
    Ok((RingCommitment { u }, hint))
}
