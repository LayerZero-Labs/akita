use crate::compute::{
    CommitInnerPlan, ComputeBackendSetup, DigitRowsComputeBackend, RootCommitKernel,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::CommitInnerWitness;
use akita_algebra::ring::CyclotomicRing;
use akita_error::AkitaError;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore};
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
pub(crate) fn commit_outer_slices<'a, F, B, const D_B: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    n_b: usize,
    polynomial_digits: impl IntoIterator<Item = &'a DigitBlocks>,
    geometry: &akita_types::CommitmentSliceGeometry,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D_B>>, AkitaError>
where
    F: FieldCore + CanonicalField,
    B: DigitRowsComputeBackend<F>,
{
    let mut stacked = Vec::with_capacity(geometry.logical_output_rows(n_b)?);
    let polynomial_planes = validate_outer_slice_digits::<D_B>(polynomial_digits, geometry)?;
    for_each_outer_slice_input::<D_B>(polynomial_planes, geometry, |input| {
        let rows = backend.digit_rows::<D_B>(prepared, n_b, input, log_basis)?;
        if rows.len() != n_b {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {} B commitment rows, expected {n_b}",
                rows.len(),
            )));
        }
        stacked.extend(rows);
        Ok(())
    })?;
    Ok(stacked)
}

/// Validate one committed group's per-polynomial plane counts, then stream its
/// canonical B slices through one reusable physical-width buffer.
pub(crate) fn for_each_outer_slice_input<'a, const D_B: usize>(
    polynomial_planes: impl IntoIterator<Item = &'a [[i8; D_B]]>,
    geometry: &akita_types::CommitmentSliceGeometry,
    mut consume: impl FnMut(&[[i8; D_B]]) -> Result<(), AkitaError>,
) -> Result<(), AkitaError> {
    let per_block = geometry.ring_elements_per_block_per_polynomial();
    let num_live_blocks = geometry
        .block_ranges()
        .last()
        .map(|range| range.end)
        .ok_or_else(|| AkitaError::InvalidSetup("B commitment has no slices".into()))?;
    let expected_planes = num_live_blocks
        .checked_mul(per_block)
        .ok_or_else(|| AkitaError::InvalidSetup("B slice plane count overflow".into()))?;
    let polynomial_planes = polynomial_planes.into_iter().collect::<Vec<_>>();
    if polynomial_planes.is_empty()
        || polynomial_planes
            .iter()
            .any(|planes| planes.len() != expected_planes)
    {
        return Err(AkitaError::InvalidSetup(
            "B slice input does not match the frozen block geometry".into(),
        ));
    }

    let max_blocks = geometry.max_blocks_per_slice();
    let expected_width = geometry.physical_input_width();
    let mut input = Vec::with_capacity(expected_width);
    for range in geometry.block_ranges() {
        input.clear();
        let plane_start = range
            .start
            .checked_mul(per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("B slice input offset overflow".into()))?;
        let plane_end = range
            .end
            .checked_mul(per_block)
            .ok_or_else(|| AkitaError::InvalidSetup("B slice input offset overflow".into()))?;
        for planes in &polynomial_planes {
            input.extend_from_slice(planes.get(plane_start..plane_end).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "B slice input does not match the frozen block geometry".into(),
                )
            })?);
            let padding = (max_blocks - range.len())
                .checked_mul(per_block)
                .ok_or_else(|| AkitaError::InvalidSetup("B slice padding overflow".into()))?;
            let padded_len = input
                .len()
                .checked_add(padding)
                .filter(|len| *len <= expected_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "B slice input width does not match the physical matrix".into(),
                    )
                })?;
            input.resize(padded_len, [0i8; D_B]);
        }
        if input.len() != expected_width {
            return Err(AkitaError::InvalidSetup(
                "B slice input width does not match the physical matrix".into(),
            ));
        }
        consume(&input)?;
    }
    Ok(())
}

fn validate_outer_slice_digits<'a, const D_B: usize>(
    polynomial_digits: impl IntoIterator<Item = &'a DigitBlocks>,
    geometry: &akita_types::CommitmentSliceGeometry,
) -> Result<Vec<&'a [[i8; D_B]]>, AkitaError> {
    let per_block = geometry.ring_elements_per_block_per_polynomial();
    let num_live_blocks = geometry
        .block_ranges()
        .last()
        .map(|range| range.end)
        .ok_or_else(|| AkitaError::InvalidSetup("B commitment has no slices".into()))?;
    polynomial_digits
        .into_iter()
        .map(|digits| {
            if digits.block_count() != num_live_blocks
                || digits.block_sizes().iter().any(|&size| size != per_block)
            {
                return Err(AkitaError::InvalidSetup(
                    "B slice input does not match the frozen block geometry".into(),
                ));
            }
            digits.typed_planes::<D_B>()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn outer_slice_inputs<const D_B: usize>(
    polynomial_digits: &[&DigitBlocks],
    geometry: &akita_types::CommitmentSliceGeometry,
) -> Result<Vec<Vec<[i8; D_B]>>, AkitaError> {
    let mut inputs = Vec::with_capacity(geometry.slice_count().get());
    let polynomial_planes =
        validate_outer_slice_digits::<D_B>(polynomial_digits.iter().copied(), geometry)?;
    for_each_outer_slice_input::<D_B>(polynomial_planes, geometry, |input| {
        inputs.push(input.to_vec());
        Ok(())
    })?;
    Ok(inputs)
}
