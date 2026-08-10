use crate::compute::{CommitInnerPlan, ComputeBackendSetup, RootCommitKernel};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::CommitInnerWitness;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{DigitBlocks, RingVec};

#[tracing::instrument(skip_all, name = "validate_commit_inner_shape")]
pub(crate) fn validate_commit_inner_shape<F, const D: usize>(
    inner: &CommitInnerWitness<F>,
    num_live_blocks: usize,
    n_a: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    inner.ensure_ring_dim::<D>()?;

    let expected_rows = num_live_blocks
        .checked_mul(n_a)
        .ok_or_else(|| AkitaError::InvalidSetup("inner commitment row count overflow".into()))?;
    let actual_rows = inner.inner_rows.count();
    if actual_rows != expected_rows {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {actual_rows} inner commitment rows, expected {expected_rows}"
        )));
    }
    for block_idx in 0..num_live_blocks {
        let block_rows = inner.block_rows::<D>(block_idx, n_a)?;
        if block_rows.len() != n_a {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} A rows for inner commitment block {}, expected {}",
                block_rows.len(),
                block_idx,
                n_a
            )));
        }
    }
    Ok(())
}

fn validate_commit_inner_group_len(expected: usize, actual: usize) -> Result<(), AkitaError> {
    if actual != expected {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned {actual} inner commitments for {expected} sources"
        )));
    }
    Ok(())
}

/// Run and validate one same-shape inner commitment group, then decompose its
/// rows into the outer role's digits. This is the canonical transition from a
/// source-typed root kernel to outer commitment input.
pub(super) fn prepare_inner_commit_group<F, S, B, const D_A: usize, const D_B: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    sources: Vec<S>,
    plan: CommitInnerPlan,
    num_live_blocks: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<(RingVec<F>, DigitBlocks)>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: ComputeBackendSetup<F> + RootCommitKernel<S, F, D_A>,
{
    let source_count = sources.len();
    let n_a = plan.n_a;
    let inners = backend.commit_inner_group(prepared, sources, plan)?;
    validate_commit_inner_group_len(source_count, inners.len())?;
    cfg_into_iter!(inners)
        .map(|inner| -> Result<(RingVec<F>, DigitBlocks), AkitaError> {
            validate_commit_inner_shape::<F, D_A>(&inner, num_live_blocks, n_a)?;
            let blocks = (0..num_live_blocks)
                .map(|block| inner.block_rows::<D_A>(block, n_a))
                .collect::<Result<Vec<_>, _>>()?;
            let digits =
                decompose_commit_blocks_into::<F, D_A, D_B>(&blocks, num_digits_open, log_basis)?;
            Ok((inner.into_inner_rows(), digits))
        })
        .collect()
}
