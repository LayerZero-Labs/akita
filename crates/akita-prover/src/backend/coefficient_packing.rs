//! Shared checked construction for coefficient-packing backend kernels.

use crate::compute::SubringCoefficientPackingPlan;
use akita_field::parallel::*;
use akita_field::{AkitaError, ExtField, FieldCore};
use akita_types::FpExtEncoding;

fn zero_vec<T: FieldCore>(len: usize) -> Result<Vec<T>, AkitaError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "coefficient-packing output allocation failed for {len} elements"
        ))
    })?;
    values.resize(len, T::zero());
    Ok(values)
}

/// Construct canonical partials from an indexed A-ring coefficient source.
///
/// The source index is `[position][A coefficient]`. This helper is shared by
/// dense and recursive representations; sparse representations may implement
/// a direct scatter while comparing against this path in tests.
#[tracing::instrument(skip_all, name = "coefficient_packing_partials")]
pub(crate) fn partials_from_indexed_source<F, E, const D: usize>(
    plan: SubringCoefficientPackingPlan<'_, E>,
    source_num_vars: usize,
    source_len: usize,
    coefficient_at: impl Fn(usize) -> Result<F, AkitaError> + Sync,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + FpExtEncoding<F>,
{
    plan.validate::<D>(source_num_vars)?;
    let point = plan.point;
    let geometry = point.geometry();
    if E::EXT_DEGREE != geometry.extension_degree() {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing field extension degree mismatch".into(),
        ));
    }
    let expected_source_len = point.num_live_positions().checked_mul(D).ok_or_else(|| {
        AkitaError::InvalidInput("coefficient-packing source length overflow".into())
    })?;
    if source_len != expected_source_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_source_len,
            actual: source_len,
        });
    }

    let num_blocks = point.num_live_blocks();
    let partial_width = geometry.partial_base_field_width();
    let output_len = num_blocks.checked_mul(partial_width).ok_or_else(|| {
        AkitaError::InvalidInput("coefficient-packing output length overflow".into())
    })?;
    let s = geometry.challenge_subring_dimension();
    let stride = geometry.subring_embedding_stride();

    let block_coordinates = cfg_into_iter!(0..num_blocks)
        .map(|block_index| {
            let first_position = block_index
                .checked_mul(point.num_positions_per_block())
                .ok_or_else(|| {
                    AkitaError::InvalidInput("coefficient-packing block offset overflow".into())
                })?;
            let live_in_block = point
                .num_live_positions()
                .checked_sub(first_position)
                .ok_or(AkitaError::InvalidProof)?
                .min(point.num_positions_per_block());
            let mut packed = zero_vec::<E>(s)?;
            for position_in_block in 0..live_in_block {
                let position = first_position
                    .checked_add(position_in_block)
                    .ok_or(AkitaError::InvalidProof)?;
                let source_offset = position.checked_mul(D).ok_or_else(|| {
                    AkitaError::InvalidInput("coefficient-packing source offset overflow".into())
                })?;
                let position_weight = point.position_weights()[position_in_block];
                for (subring_index, accumulator) in packed.iter_mut().enumerate() {
                    let subring_offset = subring_index.checked_mul(stride).ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "coefficient-packing subring offset overflow".into(),
                        )
                    })?;
                    let mut packed_position = E::zero();
                    for (low_index, &packing_weight) in point.packing_weights().iter().enumerate() {
                        let index = source_offset
                            .checked_add(subring_offset)
                            .and_then(|value| value.checked_add(low_index))
                            .ok_or_else(|| {
                                AkitaError::InvalidInput(
                                    "coefficient-packing source index overflow".into(),
                                )
                            })?;
                        let source = coefficient_at(index)?;
                        packed_position += packing_weight.mul_base(source);
                    }
                    *accumulator += position_weight * packed_position;
                }
            }

            let mut output_coordinates = zero_vec::<F>(partial_width)?;
            for (subring_index, coefficient) in packed.into_iter().enumerate() {
                let coordinates = coefficient.ext_coords();
                if coordinates.len() != geometry.extension_degree() {
                    return Err(AkitaError::InvalidSetup(
                        "coefficient-packing extension encoding width mismatch".into(),
                    ));
                }
                for (extension_coordinate, &coordinate) in coordinates.iter().enumerate() {
                    let local_index = geometry
                        .partial_base_field_coordinate_index(extension_coordinate, subring_index)?;
                    *output_coordinates
                        .get_mut(local_index)
                        .ok_or(AkitaError::InvalidProof)? = coordinate;
                }
            }
            Ok(output_coordinates)
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let mut output = Vec::new();
    output.try_reserve_exact(output_len).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "coefficient-packing output allocation failed for {output_len} elements"
        ))
    })?;
    for coordinates in block_coordinates {
        output.extend(coordinates);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::backend::{DensePoly, OneHotPoly, RecursiveWitnessFlat};
    use crate::compute::{
        CpuBackend, RootTensorSource, SubringCoefficientPackingBatchKernel,
        SubringCoefficientPackingPlan,
    };
    use crate::{RootTensorProjectionPoly, SparseRingPoly};
    use akita_algebra::CyclotomicRing;
    use akita_field::{
        AkitaError, CanonicalField, Ext2, ExtField, FieldCore, FpExt4, FromPrimitiveInt,
        Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59,
    };
    use akita_types::{
        coefficient_packing_partials, BasisMode, FpExtEncoding,
        PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingGeometry,
    };

    type F = Prime32Offset99;
    type E = FpExt4<F>;
    const D: usize = 256;

    fn prepared_point() -> PreparedSubringCoefficientPackingPoint<E> {
        let geometry = SubringCoefficientPackingGeometry::try_new(4, D, 64).unwrap();
        let point = (0..9)
            .map(|index| E::from_u64((index + 2) as u64))
            .collect::<Vec<_>>();
        PreparedSubringCoefficientPackingPoint::new(geometry, BasisMode::Lagrange, 2, 4, 9, &point)
            .unwrap()
    }

    fn assert_dense_matches_reference<T, U, const RING_D: usize>(s: usize)
    where
        T: FieldCore + CanonicalField + FromPrimitiveInt,
        U: ExtField<T> + FpExtEncoding<T> + FromPrimitiveInt,
    {
        let geometry =
            SubringCoefficientPackingGeometry::try_new(U::EXT_DEGREE, RING_D, s).unwrap();
        let point_len = (2 * RING_D).next_power_of_two().trailing_zeros() as usize;
        let public_point = (0..point_len)
            .map(|index| U::from_u64((index + 3) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            2,
            4,
            point_len,
            &public_point,
        )
        .unwrap();
        let rings = (0..2)
            .map(|position| {
                CyclotomicRing::<T, RING_D>::from_coefficients(std::array::from_fn(|coefficient| {
                    T::from_i64(((position * RING_D + coefficient) % 13) as i64 - 6)
                }))
            })
            .collect::<Vec<_>>();
        let source = rings
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<_>>();
        let expected = coefficient_packing_partials::<T, U>(
            geometry,
            2,
            4,
            &source,
            point.position_weights(),
            point.packing_weights(),
        )
        .unwrap();
        let poly = DensePoly::from_ring_coeffs(rings);
        let polys = [&poly];
        let batch = <DensePoly<T> as RootTensorSource<T, RING_D>>::tensor_batch(&polys).unwrap();
        let got = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { point: &point },
            )
            .unwrap();
        assert_eq!(got[0].coordinates(), expected);
    }

    #[test]
    fn dense_kernel_matches_flat_reference_with_signed_coefficients() {
        let point = prepared_point();
        let rings = (0..2)
            .map(|position| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64(((position * D + coefficient) % 11) as i64 - 5)
                }))
            })
            .collect::<Vec<_>>();
        let source = rings
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<_>>();
        let expected = coefficient_packing_partials::<F, E>(
            point.geometry(),
            point.num_live_positions(),
            point.num_positions_per_block(),
            &source,
            point.position_weights(),
            point.packing_weights(),
        )
        .unwrap();
        let poly = DensePoly::from_ring_coeffs(rings);
        let polys = [&poly];
        let batch = <DensePoly<F> as RootTensorSource<F, D>>::tensor_batch(&polys).unwrap();
        let got = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { point: &point },
            )
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].coordinates(), expected);
    }

    #[test]
    fn dense_kernel_covers_every_production_extension_degree() {
        assert_dense_matches_reference::<Prime32Offset99, FpExt4<Prime32Offset99>, 256>(64);
        assert_dense_matches_reference::<Prime32Offset99, FpExt4<Prime32Offset99>, 1024>(64);
        assert_dense_matches_reference::<Prime64Offset59, Ext2<Prime64Offset59>, 128>(64);
        assert_dense_matches_reference::<Prime128OffsetA7F7, Prime128OffsetA7F7, 128>(64);
    }

    #[test]
    fn recursive_batch_preserves_claim_order_and_partial_final_block() {
        let geometry = SubringCoefficientPackingGeometry::try_new(4, D, 64).unwrap();
        let public_point = (0..11)
            .map(|index| E::from_u64((index + 5) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            6,
            4,
            11,
            &public_point,
        )
        .unwrap();
        let source_digits = (0..2)
            .map(|claim| {
                (0..6 * D)
                    .map(|index| ((claim * 6 * D + index) % 17) as i8 - 8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let sources = source_digits
            .iter()
            .map(|digits| {
                RecursiveWitnessFlat::from_i8_digits(digits.clone())
                    .align_for_commitment_ring_dim(D)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let refs = sources.iter().collect::<Vec<_>>();
        let batch = <RecursiveWitnessFlat as RootTensorSource<F, D>>::tensor_batch(&refs).unwrap();
        let got = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { point: &point },
            )
            .unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|partials| {
            partials.num_live_blocks() == 2 && partials.coordinates().len() == 2 * 4 * 64
        }));
        for (claim, partials) in got.iter().enumerate() {
            let source = source_digits[claim]
                .iter()
                .copied()
                .map(F::from_i8)
                .collect::<Vec<_>>();
            let expected = coefficient_packing_partials::<F, E>(
                geometry,
                6,
                4,
                &source,
                point.position_weights(),
                point.packing_weights(),
            )
            .unwrap();
            assert_eq!(partials.coordinates(), expected);
        }
    }

    #[test]
    fn onehot_and_dense_kernels_emit_identical_coordinates() {
        let point = prepared_point();
        let hot = [17usize, 255usize];
        let rings = hot
            .iter()
            .map(|&hot_index| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_u64(u64::from(coefficient == hot_index))
                }))
            })
            .collect::<Vec<_>>();
        let dense = DensePoly::from_ring_coeffs(rings);
        let onehot = OneHotPoly::<F>::new(256, D, hot.map(Some).to_vec()).unwrap();
        let dense_refs = [&dense];
        let onehot_refs = [&onehot];
        let dense_batch =
            <DensePoly<F> as RootTensorSource<F, D>>::tensor_batch(&dense_refs).unwrap();
        let onehot_batch =
            <OneHotPoly<F> as RootTensorSource<F, D>>::tensor_batch(&onehot_refs).unwrap();
        let plan = SubringCoefficientPackingPlan { point: &point };
        let dense_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, dense_batch, plan)
            .unwrap();
        let onehot_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, onehot_batch, plan)
            .unwrap();
        assert_eq!(dense_partials, onehot_partials);
    }

    #[test]
    fn dense_and_onehot_reject_recursive_style_live_prefixes() {
        let geometry = SubringCoefficientPackingGeometry::try_new(4, D, 64).unwrap();
        let public_point = (0..11)
            .map(|index| E::from_u64((index + 3) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            6,
            4,
            11,
            &public_point,
        )
        .expect("recursive-style live prefix point");
        let dense = DensePoly::from_ring_coeffs::<D>(vec![CyclotomicRing::zero(); 8]);
        let onehot = OneHotPoly::<F>::new(
            D,
            D,
            vec![
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                Some(0),
                None,
                None,
            ],
        )
        .unwrap();
        let dense_refs = [&dense];
        let onehot_refs = [&onehot];
        let dense_batch =
            <DensePoly<F> as RootTensorSource<F, D>>::tensor_batch(&dense_refs).unwrap();
        let onehot_batch =
            <OneHotPoly<F> as RootTensorSource<F, D>>::tensor_batch(&onehot_refs).unwrap();
        let plan = SubringCoefficientPackingPlan { point: &point };

        assert!(matches!(
            CpuBackend::DEFAULT.coefficient_packing_partials_batch(None, dense_batch, plan),
            Err(AkitaError::InvalidSize {
                expected: 6,
                actual: 8,
            })
        ));
        assert!(matches!(
            CpuBackend::DEFAULT.coefficient_packing_partials_batch(None, onehot_batch, plan),
            Err(AkitaError::InvalidSize {
                expected,
                actual,
            }) if expected == 6 * D && actual == 8 * D
        ));
    }

    #[test]
    fn tensor_projection_dense_and_sparse_sources_pack_identically() {
        let point = prepared_point();
        let entries = vec![(0usize, 17usize, 1i8), (1, 255, -1)];
        let sparse = SparseRingPoly::from_signed_coeffs(9, D, 2, entries.clone()).unwrap();
        let dense = DensePoly::from_ring_coeffs(
            (0..2)
                .map(|position| {
                    CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                        entries
                            .iter()
                            .filter(|(ring, index, _)| *ring == position && *index == coefficient)
                            .map(|(_, _, value)| F::from_i8(*value))
                            .fold(F::zero(), |sum, value| sum + value)
                    }))
                })
                .collect(),
        );
        let dense_projection = RootTensorProjectionPoly::Dense(dense);
        let sparse_projection = RootTensorProjectionPoly::Sparse(std::sync::Arc::new(sparse));
        let refs = [&dense_projection, &sparse_projection];
        let batch =
            <RootTensorProjectionPoly<F> as RootTensorSource<F, D>>::tensor_batch(&refs).unwrap();
        let packed = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { point: &point },
            )
            .unwrap();
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], packed[1]);
    }

    #[test]
    fn onehot_scatter_matches_dense_with_nontrivial_embedding_stride() {
        const LARGE_D: usize = 1024;
        let geometry = SubringCoefficientPackingGeometry::try_new(4, LARGE_D, 64).unwrap();
        assert_eq!(geometry.subring_embedding_stride(), 16);
        let public_point = (0..11)
            .map(|index| E::from_u64((index + 7) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            2,
            4,
            11,
            &public_point,
        )
        .unwrap();
        let hot = [17usize, 1009usize];
        let dense = DensePoly::from_ring_coeffs(
            hot.iter()
                .map(|&hot_index| {
                    CyclotomicRing::<F, LARGE_D>::from_coefficients(std::array::from_fn(
                        |coefficient| F::from_u64(u64::from(coefficient == hot_index)),
                    ))
                })
                .collect(),
        );
        let onehot = OneHotPoly::<F>::new(LARGE_D, LARGE_D, hot.map(Some).to_vec()).unwrap();
        let dense_refs = [&dense];
        let onehot_refs = [&onehot];
        let dense_batch =
            <DensePoly<F> as RootTensorSource<F, LARGE_D>>::tensor_batch(&dense_refs).unwrap();
        let onehot_batch =
            <OneHotPoly<F> as RootTensorSource<F, LARGE_D>>::tensor_batch(&onehot_refs).unwrap();
        let plan = SubringCoefficientPackingPlan { point: &point };
        let dense_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, dense_batch, plan)
            .unwrap();
        let onehot_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, onehot_batch, plan)
            .unwrap();
        assert_eq!(dense_partials, onehot_partials);
    }

    #[test]
    fn onehot_chunk_spanning_packing_blocks_matches_dense_at_boundaries() {
        const ONEHOT_K: usize = 2048;
        const NUM_POSITIONS: usize = 16;
        const POSITIONS_PER_BLOCK: usize = 4;
        const _: () = assert!(ONEHOT_K > D * POSITIONS_PER_BLOCK);
        let geometry = SubringCoefficientPackingGeometry::try_new(4, D, 64).unwrap();
        let public_point = (0..12)
            .map(|index| E::from_u64((index + 11) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            NUM_POSITIONS,
            POSITIONS_PER_BLOCK,
            12,
            &public_point,
        )
        .unwrap();
        let hot_by_claim = [
            [Some(1023usize), Some(17usize)],
            [Some(1024usize), Some(2047usize)],
            [Some(1025usize), None],
        ];
        let onehot = hot_by_claim
            .iter()
            .map(|indices| OneHotPoly::<F>::new(ONEHOT_K, D, indices.to_vec()).unwrap())
            .collect::<Vec<_>>();
        let dense = hot_by_claim
            .iter()
            .map(|indices| {
                let hot_fields = indices
                    .iter()
                    .enumerate()
                    .filter_map(|(chunk, hot)| hot.map(|hot| chunk * ONEHOT_K + hot))
                    .collect::<Vec<_>>();
                DensePoly::from_ring_coeffs(
                    (0..NUM_POSITIONS)
                        .map(|position| {
                            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(
                                |coefficient| {
                                    let field = position * D + coefficient;
                                    F::from_u64(u64::from(hot_fields.contains(&field)))
                                },
                            ))
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let onehot_refs = onehot.iter().collect::<Vec<_>>();
        let dense_refs = dense.iter().collect::<Vec<_>>();
        let onehot_batch =
            <OneHotPoly<F> as RootTensorSource<F, D>>::tensor_batch(&onehot_refs).unwrap();
        let dense_batch =
            <DensePoly<F> as RootTensorSource<F, D>>::tensor_batch(&dense_refs).unwrap();
        let plan = SubringCoefficientPackingPlan { point: &point };
        let onehot_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, onehot_batch, plan)
            .unwrap();
        let dense_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, dense_batch, plan)
            .unwrap();

        assert_eq!(onehot_partials, dense_partials);
    }

    #[test]
    fn recursive_live_prefix_matches_zero_padded_dense_source() {
        let point = prepared_point();
        let live_digits = (0..300)
            .map(|index| match index % 5 {
                0 => -1,
                1 => 1,
                _ => 0,
            })
            .collect::<Vec<_>>();
        let recursive = RecursiveWitnessFlat::from_i8_digits(live_digits.clone())
            .align_for_commitment_ring_dim(D)
            .unwrap();
        let mut padded = live_digits;
        padded.resize(2 * D, 0);
        let rings = padded
            .chunks_exact(D)
            .map(|coefficients| {
                CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
                    F::from_i8(coefficients[index])
                }))
            })
            .collect::<Vec<_>>();
        let dense = DensePoly::from_ring_coeffs(rings);
        let dense_refs = [&dense];
        let recursive_refs = [&recursive];
        let dense_batch =
            <DensePoly<F> as RootTensorSource<F, D>>::tensor_batch(&dense_refs).unwrap();
        let recursive_batch =
            <RecursiveWitnessFlat as RootTensorSource<F, D>>::tensor_batch(&recursive_refs)
                .unwrap();
        let plan = SubringCoefficientPackingPlan { point: &point };
        let dense_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, dense_batch, plan)
            .unwrap();
        let recursive_partials = CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(None, recursive_batch, plan)
            .unwrap();
        assert_eq!(dense_partials, recursive_partials);
    }

    #[test]
    fn prepared_point_rejects_same_ring_count_with_different_source_arity() {
        type Base = Prime128OffsetA7F7;
        const RING_D: usize = 128;
        let geometry = SubringCoefficientPackingGeometry::try_new(1, RING_D, 64).unwrap();
        let public_point = (0..6)
            .map(|index| Base::from_u64((index + 1) as u64))
            .collect::<Vec<_>>();
        let point = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            1,
            1,
            6,
            &public_point,
        )
        .unwrap();
        let lower_arity = DensePoly::from_field_evals(5, RING_D, vec![Base::one(); 32]).unwrap();
        let refs = [&lower_arity];
        let batch =
            <DensePoly<Base> as RootTensorSource<Base, RING_D>>::tensor_batch(&refs).unwrap();
        assert!(CpuBackend::DEFAULT
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { point: &point },
            )
            .is_err());
    }
}
