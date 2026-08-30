use crate::compute::{
    CommitInnerPlan, ComputeBackendSetup, DigitRowsComputeBackend, OperationCtx, RootCommitKernel,
    RootCommitSource, RuntimeCommitBackendFor, RuntimeCommitSource,
};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::CommitInnerWitness;
use akita_algebra::ring::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{
    dispatch_for_field, CommitmentRingDims, CommitmentSliceGeometry, DigitBlocks,
    GroupCommitPhaseParams, RingVec,
};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};

#[tracing::instrument(skip_all, name = "validate_commit_inner_shape")]
pub(crate) fn validate_commit_inner_shape<F, const D: usize>(
    inner: &CommitInnerWitness<F>,
    num_live_blocks: usize,
    n_a: usize,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
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

/// Compute one same-shape group of inner commitments `t_i`.
pub(super) fn compute_inner_commitment<F, P, B, const D_A: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    polys: &[P],
    plan: CommitInnerPlan,
) -> Result<Vec<CommitInnerWitness<F>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    P: RootCommitSource<F, D_A>,
    B: ComputeBackendSetup<F> + for<'a> RootCommitKernel<P::CommitView<'a>, F, D_A>,
{
    let views = polys
        .iter()
        .map(|poly| RootCommitSource::<F, D_A>::commit_view(poly))
        .collect::<Result<Vec<_>, _>>()?;
    backend.commit_inner_group(prepared, views, plan)
}

/// Validate and decompose `t_i`, apply the outer matrix, and erase its ring
/// dimension for compression.
pub(super) fn compute_outer_commitment<F, B, const D_A: usize, const D_B: usize>(
    backend: &B,
    prepared: &B::PreparedSetup,
    inners: Vec<CommitInnerWitness<F>>,
    profile: &GroupCommitPhaseParams,
    slice_geometry: &CommitmentSliceGeometry,
) -> Result<(Vec<RingVec<F>>, RingVec<F>), AkitaError>
where
    F: Field + CanonicalEncoding,
    B: DigitRowsComputeBackend<F>,
{
    let n_a = profile.inner.matrix.output_rank();
    let num_live_blocks = profile.blocks.live_blocks;
    let num_digits_open = profile.outer.digits.num_digits;
    let log_basis = profile.outer.digits.log_basis;
    let n_b = profile.outer.matrix.output_rank();

    validate_commit_inner_group_len(profile.group.num_polynomials(), inners.len())?;
    let prepared_polynomials = cfg_into_iter!(inners)
        .map(|inner| -> Result<(RingVec<F>, DigitBlocks), AkitaError> {
            validate_commit_inner_shape::<F, D_A>(&inner, num_live_blocks, n_a)?;
            let blocks = (0..num_live_blocks)
                .map(|block| inner.block_rows::<D_A>(block, n_a))
                .collect::<Result<Vec<_>, _>>()?;
            let digits =
                decompose_commit_blocks_into::<F, D_A, D_B>(&blocks, num_digits_open, log_basis)?;
            Ok((inner.into_inner_rows(), digits))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let u = commit_outer_slices::<F, _, D_B>(
        backend,
        prepared,
        n_b,
        prepared_polynomials.iter().map(|(_, digits)| digits),
        slice_geometry,
        log_basis,
    )?;
    let inner_rows = prepared_polynomials
        .into_iter()
        .map(|(rows, _)| rows)
        .collect();
    Ok((inner_rows, RingVec::from_ring_elems(&u)))
}

/// Compute the inner and outer commitment stages behind runtime role dispatch.
pub(super) fn compute_inner_outer_commitment<F, P, B>(
    polys: &[P],
    ctx: &OperationCtx<'_, F, B>,
    profile: GroupCommitPhaseParams,
) -> Result<(Vec<RingVec<F>>, RingVec<F>), AkitaError>
where
    F: Field + CanonicalEncoding,
    P: RuntimeCommitSource<F>,
    B: RuntimeCommitBackendFor<F, P>,
{
    let backend = ctx.backend();
    let prepared = ctx.prepared();
    let slice_geometry = profile.derive_slice_geometry()?;
    let dims = CommitmentRingDims {
        inner: profile.inner.matrix.ring_dimension(),
        outer: profile.outer.matrix.ring_dimension(),
        opening: profile.outer.matrix.ring_dimension(),
    };
    let plan = CommitInnerPlan {
        n_a: profile.inner.matrix.output_rank(),
        num_positions_per_block: profile.blocks.positions_per_block,
        num_digits_inner: profile.inner.digits.num_digits,
        log_basis_inner: profile.inner.digits.log_basis,
    };
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        dims.d_a(),
        |D_A| {
            let inners = compute_inner_commitment::<F, _, _, D_A>(backend, prepared, polys, plan)?;
            dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Outer),
                F,
                dims.d_b(),
                |D_B| compute_outer_commitment::<F, _, D_A, D_B>(
                    backend,
                    prepared,
                    inners,
                    &profile,
                    &slice_geometry,
                )
            )
        }
    )
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
    F: Field + CanonicalEncoding,
    B: DigitRowsComputeBackend<F>,
{
    let polynomial_planes = validate_outer_slice_digits::<D_B>(polynomial_digits, geometry)?;
    let mut inputs = Vec::with_capacity(geometry.slice_count().get());
    for_each_outer_slice_input::<D_B>(polynomial_planes, geometry, |input| {
        inputs.push(input.to_vec());
        Ok(())
    })?;
    let input_refs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let row_batches = backend.digit_rows::<D_B>(prepared, n_b, &input_refs, log_basis)?;
    if row_batches.len() != input_refs.len() || row_batches.iter().any(|rows| rows.len() != n_b) {
        return Err(AkitaError::InvalidSetup(format!(
            "backend returned B commitment row shape {:?}, expected {} batches of {n_b} rows",
            row_batches.iter().map(Vec::len).collect::<Vec<_>>(),
            input_refs.len(),
        )));
    }
    let mut stacked = Vec::with_capacity(geometry.logical_output_rows(n_b)?);
    stacked.extend(row_batches.into_iter().flatten());
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

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fp64;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn inner_witness(recomposed_blocks: usize, rows_per_block: usize) -> CommitInnerWitness<F> {
        CommitInnerWitness::from_rows(vec![
            vec![CyclotomicRing::<F, D>::zero(); rows_per_block];
            recomposed_blocks
        ])
    }

    #[test]
    fn commit_inner_shape_accepts_expected_layout() {
        let inner = inner_witness(2, 3);
        validate_commit_inner_shape::<F, D>(&inner, 2, 3).expect("shape should match");
    }

    #[test]
    fn commit_inner_shape_rejects_bad_block_count() {
        let inner = inner_witness(1, 3);
        assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
    }

    #[test]
    fn commit_inner_shape_rejects_bad_row_count() {
        let inner = inner_witness(2, 2);
        assert!(validate_commit_inner_shape::<F, D>(&inner, 2, 3).is_err());
    }

    #[test]
    fn commit_inner_shape_accepts_many_all_zero_blocks() {
        let num_live_blocks = 1024;
        let inner = inner_witness(num_live_blocks, 3);
        validate_commit_inner_shape::<F, D>(&inner, num_live_blocks, 3).expect("all-zero blocks");
    }

    #[test]
    fn outer_slice_inputs_are_polynomial_major_and_zero_padded() {
        let first =
            DigitBlocks::new(vec![10, 11, 12, 13, 14], vec![1; 5], 1).expect("first digit blocks");
        let second =
            DigitBlocks::new(vec![20, 21, 22, 23, 24], vec![1; 5], 1).expect("second digit blocks");
        let geometry = CommitmentSliceGeometry::try_new(
            akita_types::CommitmentSliceCount::TWO,
            5,
            2,
            1,
            1,
            1,
            1,
        )
        .expect("slice geometry");

        let inputs = outer_slice_inputs::<1>(&[&first, &second], &geometry).expect("slice inputs");
        assert_eq!(
            inputs,
            vec![
                vec![[10], [11], [0], [20], [21], [0]],
                vec![[12], [13], [14], [22], [23], [24]],
            ]
        );
    }

    #[test]
    fn outer_slice_stream_reuses_one_physical_width_buffer() {
        let digits = DigitBlocks::new((0..13).collect(), vec![1; 13], 1).expect("digit blocks");
        let geometry = CommitmentSliceGeometry::try_new(
            akita_types::CommitmentSliceCount::FOUR,
            13,
            1,
            1,
            1,
            1,
            1,
        )
        .expect("slice geometry");
        let planes = digits.typed_planes::<1>().expect("typed planes");
        let mut addresses = Vec::new();

        for_each_outer_slice_input::<1>(std::iter::once(planes), &geometry, |input| {
            assert_eq!(input.len(), geometry.physical_input_width());
            addresses.push(input.as_ptr());
            Ok(())
        })
        .expect("stream slices");

        assert_eq!(addresses.len(), 4);
        assert!(addresses.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn sliced_b_images_match_independent_block_diagonal_oracle_for_all_counts() {
        const BLOCKS: usize = 9;
        const POLYS: usize = 2;
        const PER_BLOCK: usize = 2;
        const ROWS: usize = 3;

        let polynomial_digits = (0..POLYS)
            .map(|polynomial| {
                let digits = (0..BLOCKS * PER_BLOCK)
                    .map(|index| (1 + polynomial * 31 + index) as i8)
                    .collect::<Vec<_>>();
                DigitBlocks::new(digits, vec![PER_BLOCK; BLOCKS], 1).unwrap()
            })
            .collect::<Vec<_>>();

        for slice_count in akita_types::CommitmentSliceCount::ALL {
            let geometry =
                CommitmentSliceGeometry::try_new(slice_count, BLOCKS, POLYS, PER_BLOCK, 1, 1, 1)
                    .unwrap();
            let matrix = (0..ROWS)
                .map(|row| {
                    (0..geometry.physical_input_width())
                        .map(|column| 1 + (row as i64 + 1) * 17 + column as i64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let production_inputs =
                outer_slice_inputs::<1>(&polynomial_digits.iter().collect::<Vec<_>>(), &geometry)
                    .unwrap();
            let production_image = production_inputs
                .iter()
                .flat_map(|input| {
                    let matrix = &matrix;
                    matrix.iter().map(move |row| {
                        row.iter()
                            .zip(input)
                            .map(|(&matrix_entry, digit)| matrix_entry * i64::from(digit[0]))
                            .sum::<i64>()
                    })
                })
                .collect::<Vec<_>>();

            let slices = slice_count.get();
            let max_blocks = BLOCKS.div_ceil(slices);
            let mut oracle_image = Vec::with_capacity(slices * ROWS);
            for slice_index in 0..slices {
                let start = BLOCKS * slice_index / slices;
                let end = BLOCKS * (slice_index + 1) / slices;
                for row in &matrix {
                    let mut image = 0i64;
                    for polynomial in 0..POLYS {
                        for global_block in start..end {
                            let local_block = global_block - start;
                            for offset in 0..PER_BLOCK {
                                let physical_column =
                                    (polynomial * max_blocks + local_block) * PER_BLOCK + offset;
                                let digit = 1 + polynomial * 31 + global_block * PER_BLOCK + offset;
                                image += row[physical_column] * digit as i64;
                            }
                        }
                    }
                    oracle_image.push(image);
                }
            }
            assert_eq!(production_image, oracle_image);

            let production_compressed = production_image
                .iter()
                .enumerate()
                .map(|(index, &value)| (index as i64 + 3) * value)
                .sum::<i64>();
            let oracle_compressed = oracle_image
                .iter()
                .enumerate()
                .map(|(index, &value)| (index as i64 + 3) * value)
                .sum::<i64>();
            assert_eq!(production_compressed, oracle_compressed);
        }
    }
}
