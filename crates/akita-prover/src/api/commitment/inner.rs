use crate::compute::{
    CommitInnerPlan, ComputeBackendSetup, DigitRowsComputeBackend, RootCommitKernel,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::CommitInnerWitness;
use akita_algebra::ring::CyclotomicRing;
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

/// Apply one physical B matrix to every canonical slice and stack the images.
pub(crate) fn commit_outer_slices<F, B, const D_B: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    n_b: usize,
    polynomial_digits: &[&DigitBlocks],
    geometry: &akita_types::CommitmentSliceGeometry,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D_B>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let mut stacked = Vec::with_capacity(geometry.logical_output_rows(n_b)?);
    for input in outer_slice_inputs::<D_B>(polynomial_digits, geometry)? {
        let rows = backend.digit_rows::<D_B>(prepared, n_b, &input, log_basis)?;
        if rows.len() != n_b {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} B commitment rows, expected {n_b}",
                rows.len(),
            )));
        }
        stacked.extend(rows);
    }
    Ok(stacked)
}

pub(crate) fn outer_slice_inputs<const D_B: usize>(
    polynomial_digits: &[&DigitBlocks],
    geometry: &akita_types::CommitmentSliceGeometry,
) -> Result<Vec<Vec<[i8; D_B]>>, AkitaError> {
    let per_block = geometry.ring_elements_per_block_per_polynomial();
    let max_blocks = geometry.max_blocks_per_slice();
    let expected_width = geometry.physical_input_width();
    geometry
        .block_ranges()
        .iter()
        .map(|range| {
            let mut input = Vec::with_capacity(expected_width);
            for digits in polynomial_digits {
                digits.ensure_stride::<D_B>()?;
                if digits.block_count() < range.end
                    || digits.block_sizes().iter().any(|&size| size != per_block)
                {
                    return Err(AkitaError::InvalidSetup(
                        "B slice input does not match the frozen block geometry".into(),
                    ));
                }
                let blocks = digits.iter_blocks().collect::<Vec<_>>();
                for block in &blocks[range.clone()] {
                    let (planes, remainder) = block.as_chunks::<D_B>();
                    if !remainder.is_empty() {
                        return Err(AkitaError::InvalidSetup(
                            "B slice input has a partial digit plane".into(),
                        ));
                    }
                    input.extend_from_slice(planes);
                }
                input.resize(
                    input.len() + (max_blocks - range.len()) * per_block,
                    [0i8; D_B],
                );
            }
            if input.len() != expected_width {
                return Err(AkitaError::InvalidSetup(
                    "B slice input width does not match the physical matrix".into(),
                ));
            }
            Ok(input)
        })
        .collect()
}
