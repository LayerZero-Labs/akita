use super::*;
use crate::DensePoly;
use akita_field::Prime128OffsetA7F7 as F;

#[test]
fn sparse_commitment_scratch_budget_preserves_rows_and_rejects_too_small() {
    const D: usize = 8;
    let n_a = 2;
    let positions = 4;
    let rows = (0..n_a * positions)
        .map(|value| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coeff| {
                F::from_u64(value as u64 + coeff as u64 + 1)
            }))
        })
        .collect::<Vec<_>>();
    let flat = akita_types::FlatMatrix::from_ring_slice(&rows);
    let view = flat.ring_view::<D>(n_a, positions).unwrap();
    let owned = [
        vec![
            SparseRingBlockEntry::new(0, 1, 1),
            SparseRingBlockEntry::new(2, 3, -1),
        ],
        vec![SparseRingBlockEntry::new(1, 2, 1)],
    ];
    let blocks = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();

    let default = column_sweep_sparse(&view, &blocks, n_a, positions, 1, 1 << 20).unwrap();
    let constrained = column_sweep_sparse(&view, &blocks, n_a, positions, 1, 4096).unwrap();
    assert_eq!(constrained, default);
    assert!(matches!(
        column_sweep_sparse(&view, &blocks, n_a, positions, 1, 1),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn sparse_ring_fold_matches_dense_reference() {
    const D: usize = 8;
    let sparse =
        SparseRingPoly::<F>::from_signed_coeffs(5, D, 4, vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)])
            .unwrap();
    let mut dense_coeffs = vec![CyclotomicRing::<F, D>::zero(); 4];
    dense_coeffs[0].coeffs[1] += F::one();
    dense_coeffs[1].coeffs[3] -= F::one();
    dense_coeffs[3].coeffs[2] += F::one();
    let dense = DensePoly::from_ring_coeffs(dense_coeffs);
    let scalars = (0..2)
        .map(|idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                F::from_u64(10 + idx * 10 + k as u64)
            }))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sparse.fold_blocks_ring::<D>(&scalars, 2),
        dense.fold_blocks_ring::<D>(&scalars, 2)
    );
}

#[test]
fn sparse_ring_fold_matches_dense_for_partial_final_slice() {
    const D: usize = 8;
    let sparse =
        SparseRingPoly::<F>::from_signed_coeffs(5, D, 4, vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)])
            .unwrap();
    let mut dense_coeffs = vec![CyclotomicRing::<F, D>::zero(); 4];
    dense_coeffs[0].coeffs[1] += F::one();
    dense_coeffs[1].coeffs[3] -= F::one();
    dense_coeffs[3].coeffs[2] += F::one();
    let dense = DensePoly::from_ring_coeffs(dense_coeffs);
    let num_positions_per_block = 8usize;
    let position_weights = (0..num_positions_per_block)
        .map(|idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                F::from_u64(10 + idx as u64 * 10 + k as u64)
            }))
        })
        .collect::<Vec<_>>();

    assert_eq!(
        sparse.fold_blocks_ring::<D>(&position_weights, num_positions_per_block),
        dense.fold_blocks_ring::<D>(&position_weights, num_positions_per_block)
    );
}

#[test]
fn sparse_ring_poly_caches_multiple_runtime_layouts() {
    let sparse =
        SparseRingPoly::<F>::from_signed_coeffs(8, 32, 8, vec![(0, 1, 1), (1, 3, -1), (7, 31, 1)])
            .unwrap();

    let d32_blocks = sparse.blocks_for(32, 4).unwrap();
    let d64_blocks = sparse.blocks_for(64, 2).unwrap();

    assert_eq!(d32_blocks.num_live_blocks(), 2);
    assert_eq!(d64_blocks.num_live_blocks(), 2);
    assert_eq!(sparse.block_cache.lock().unwrap().len(), 2);
}

#[test]
fn sorted_sparse_ring_constructor_rejects_unsorted_coeffs() {
    const D: usize = 8;
    let sorted =
        SparseRingPoly::<F>::from_sorted_signed_coeffs(5, D, 4, vec![(0, 1, 1), (2, 3, -1)])
            .unwrap();
    assert_eq!(sorted.num_ring_elems(), 4);

    assert!(
        SparseRingPoly::<F>::from_sorted_signed_coeffs(5, D, 4, vec![(2, 3, -1), (0, 1, 1)],)
            .is_err()
    );
}

#[test]
fn sparse_ring_constructor_rejects_non_signed_unit_coefficients() {
    const D: usize = 8;
    for value in [-2, 0, 2] {
        assert!(matches!(
            SparseRingPoly::<F>::from_signed_coeffs(5, D, 4, vec![(0, 1, value)]),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

#[test]
fn packed_sparse_ring_constructor_matches_tuple_constructor() {
    const D: usize = 8;
    let tuples = vec![(0, 1, 1), (1, 3, -1), (3, 2, 1)];
    let packed = tuples
        .iter()
        .copied()
        .map(|(ring_idx, coeff_idx, value)| {
            SparseRingCoeff::from_ring_coords(ring_idx, coeff_idx, D, value).unwrap()
        })
        .collect::<Vec<_>>();
    let from_tuples = SparseRingPoly::<F>::from_signed_coeffs(5, D, 4, tuples).unwrap();
    let from_packed = SparseRingPoly::<F>::from_packed_coeffs(5, D, 4, packed).unwrap();

    let scalars = (0..2)
        .map(|idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                F::from_u64(20 + idx * 10 + k as u64)
            }))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        from_packed.fold_blocks_ring::<D>(&scalars, 2),
        from_tuples.fold_blocks_ring::<D>(&scalars, 2)
    );
}
