use super::test_helpers::inner_ajtai_reference;
use super::*;
use crate::backend::test_support::aggregate_witnesses;
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{RootCommitSource, RootOpeningSource, RootPolyMeta, RootTensorSource};
use crate::DensePoly;
use akita_field::RandomSampling;
use akita_field::{Fp64, FpExt4, Prime128Offset275, Prime24Offset3, Prime32Offset99};
use akita_types::FlatMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn materialize_onehot_as_dense<F, const D: usize, I>(poly: &OneHotPoly<F, I>) -> DensePoly<F>
where
    F: FieldCore + CanonicalField,
    I: OneHotIndex,
{
    let mut coeffs = vec![CyclotomicRing::<F, D>::zero(); poly.total_ring_elems];
    for (chunk_idx, hot_idx) in poly.indices.iter().copied().enumerate() {
        let Some(raw) = hot_idx else {
            continue;
        };
        let field_pos = chunk_idx * poly.onehot_k + raw.as_usize();
        let ring_idx = field_pos / D;
        let coeff_idx = field_pos % D;
        coeffs[ring_idx].coeffs[coeff_idx] += F::one();
    }
    DensePoly::from_ring_coeffs(coeffs)
}

fn test_ring_scalar<F, const D: usize>(seed: u64) -> CyclotomicRing<F, D>
where
    F: CanonicalField,
{
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_canonical_u128_reduced(u128::from(seed + idx as u64 + 1))
    }))
}

fn block_entry(pos_in_block: usize, coeff_idx: usize) -> SparseRingBlockEntry {
    SparseRingBlockEntry::new(pos_in_block as u32, coeff_idx as u16, 1)
}

fn assert_flat_blocks_eq(
    left: &FlatBlocks<SparseRingBlockEntry>,
    right: &FlatBlocks<SparseRingBlockEntry>,
) {
    assert_eq!(left.num_live_blocks(), right.num_live_blocks());
    for block in 0..left.num_live_blocks() {
        assert_eq!(left.block(block), right.block(block));
    }
}

#[test]
fn tensor_column_partials_match_dense_reference() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let poly = OneHotPoly::<F>::new(
        8,
        D,
        vec![
            Some(0usize),
            Some(7),
            None,
            Some(3),
            Some(5),
            Some(1),
            None,
            Some(6),
        ],
    )
    .unwrap();
    let dense = materialize_onehot_as_dense::<F, D, _>(&poly);
    let point = (0..poly.num_vars())
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(3 * idx as u128 + 2),
                F::from_canonical_u128_reduced(3 * idx as u128 + 3),
                F::from_canonical_u128_reduced(3 * idx as u128 + 5),
                F::from_canonical_u128_reduced(3 * idx as u128 + 7),
            ])
        })
        .collect::<Vec<_>>();

    let sparse_partials = poly.tensor_extension_column_partials::<E>(&point).unwrap();
    let dense_partials = dense
        .tensor_extension_column_partials::<E, D>(&point)
        .unwrap();
    assert_eq!(sparse_partials, dense_partials);
}

#[test]
fn batched_tensor_column_partials_match_individual() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let polys = [
        OneHotPoly::<F>::new(
            8,
            D,
            vec![
                Some(0usize),
                Some(7),
                None,
                Some(3),
                Some(5),
                Some(1),
                None,
                Some(6),
            ],
        )
        .unwrap(),
        OneHotPoly::<F>::new(
            8,
            D,
            vec![
                Some(4usize),
                Some(2),
                Some(7),
                None,
                Some(1),
                None,
                Some(5),
                Some(0),
            ],
        )
        .unwrap(),
    ];
    let point = (0..polys[0].num_vars())
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(5 * idx as u128 + 2),
                F::from_canonical_u128_reduced(5 * idx as u128 + 3),
                F::from_canonical_u128_reduced(5 * idx as u128 + 5),
                F::from_canonical_u128_reduced(5 * idx as u128 + 7),
            ])
        })
        .collect::<Vec<_>>();
    let expected = polys
        .iter()
        .map(|poly| poly.tensor_extension_column_partials::<E>(&point).unwrap())
        .collect::<Vec<_>>();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let got =
        OneHotPoly::<F>::tensor_extension_column_partials_batch::<E>(&poly_refs, &point).unwrap();

    assert_eq!(got, expected);
}

/// Exercises the factorized sparse fast path across *multiple outer blocks*
/// (`num_vars - low_vars` exceeds the inner-bit cap, so the high weights are
/// genuinely split into more than one outer block) and on a power-of-two
/// `onehot_k`. The batched sparse partials must be byte-identical both to the
/// dense reference and to the per-poly path.
#[test]
fn batched_tensor_column_partials_multi_block_match_dense() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;
    const ONEHOT_K: usize = 8;
    // hi_vars = NUM_VARS - log2(ONEHOT_K) = 18 - 3 = 15 > inner-bit cap, so the
    // factorization produces several outer blocks.
    const NUM_VARS: usize = 18;

    let num_chunks = (1usize << NUM_VARS) / ONEHOT_K;
    let make_indices = |seed: usize| {
        (0..num_chunks)
            .map(|c| {
                let h = c
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(40_503));
                if h % 7 == 0 {
                    None
                } else {
                    Some(h % ONEHOT_K)
                }
            })
            .collect::<Vec<Option<usize>>>()
    };
    let polys = [
        OneHotPoly::<F>::new(ONEHOT_K, D, make_indices(1)).unwrap(),
        OneHotPoly::<F>::new(ONEHOT_K, D, make_indices(2)).unwrap(),
        OneHotPoly::<F>::new(ONEHOT_K, D, make_indices(3)).unwrap(),
    ];
    let point = (0..NUM_VARS)
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(7 * idx as u128 + 1),
                F::from_canonical_u128_reduced(7 * idx as u128 + 2),
                F::from_canonical_u128_reduced(7 * idx as u128 + 4),
                F::from_canonical_u128_reduced(7 * idx as u128 + 6),
            ])
        })
        .collect::<Vec<_>>();

    let dense_expected = polys
        .iter()
        .map(|poly| {
            materialize_onehot_as_dense::<F, D, _>(poly)
                .tensor_extension_column_partials::<E, D>(&point)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let individual = polys
        .iter()
        .map(|poly| poly.tensor_extension_column_partials::<E>(&point).unwrap())
        .collect::<Vec<_>>();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let batched =
        OneHotPoly::<F>::tensor_extension_column_partials_batch::<E>(&poly_refs, &point).unwrap();

    assert_eq!(batched, dense_expected);
    assert_eq!(batched, individual);
}

#[test]
fn batched_tensor_column_partials_match_dense_for_fp_ext4() {
    type F = Prime32Offset99;
    type E = FpExt4<F>;
    const D: usize = 32;
    const ONEHOT_K: usize = 16;
    const NUM_VARS: usize = 10;

    let num_chunks = (1usize << NUM_VARS) / ONEHOT_K;
    let make_indices = |seed: usize| {
        (0..num_chunks)
            .map(|chunk| {
                let h = chunk
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(seed.wrapping_mul(12_345));
                if h % 11 == 0 {
                    None
                } else {
                    Some(h % ONEHOT_K)
                }
            })
            .collect::<Vec<Option<usize>>>()
    };
    let polys = [
        OneHotPoly::<F>::new(ONEHOT_K, D, make_indices(1)).unwrap(),
        OneHotPoly::<F>::new(ONEHOT_K, D, make_indices(2)).unwrap(),
        OneHotPoly::<F>::new(ONEHOT_K, D, make_indices(3)).unwrap(),
    ];
    let point = (0..NUM_VARS)
        .map(|idx| {
            E::from_base_slice(&[
                F::from_canonical_u128_reduced(7 * idx as u128 + 1),
                F::from_canonical_u128_reduced(7 * idx as u128 + 2),
                F::from_canonical_u128_reduced(7 * idx as u128 + 4),
                F::from_canonical_u128_reduced(7 * idx as u128 + 8),
            ])
        })
        .collect::<Vec<_>>();

    let dense_expected = polys
        .iter()
        .map(|poly| {
            materialize_onehot_as_dense::<F, D, _>(poly)
                .tensor_extension_column_partials::<E, D>(&point)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let batched =
        OneHotPoly::<F>::tensor_extension_column_partials_batch::<E>(&poly_refs, &point).unwrap();

    assert_eq!(batched, dense_expected);
}

#[test]
fn tensor_packed_sparse_linear_combination_matches_individual_witnesses() {
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let polys = [
        OneHotPoly::<F>::new(
            8,
            D,
            vec![
                Some(0usize),
                Some(7),
                None,
                Some(3),
                Some(5),
                Some(1),
                None,
                Some(6),
            ],
        )
        .unwrap(),
        OneHotPoly::<F>::new(
            8,
            D,
            vec![
                Some(4usize),
                Some(2),
                Some(7),
                None,
                Some(1),
                None,
                Some(5),
                Some(0),
            ],
        )
        .unwrap(),
    ];
    let coeffs = vec![
        E::from_base_slice(&[
            F::from_canonical_u128_reduced(3),
            F::from_canonical_u128_reduced(5),
            F::from_canonical_u128_reduced(7),
            F::from_canonical_u128_reduced(11),
        ]),
        E::from_base_slice(&[
            F::from_canonical_u128_reduced(13),
            F::from_canonical_u128_reduced(17),
            F::from_canonical_u128_reduced(19),
            F::from_canonical_u128_reduced(23),
        ]),
    ];
    let witnesses = polys
        .iter()
        .map(|poly| {
            poly.tensor_packed_extension_sparse_evals::<E>()
                .unwrap()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let expected =
        SparseExtensionOpeningWitness::linear_combination(coeffs.iter().copied().zip(&witnesses))
            .unwrap();
    let poly_refs = polys.iter().collect::<Vec<_>>();
    let got = OneHotPoly::<F>::tensor_packed_extension_sparse_linear_combination::<E>(
        &poly_refs, &coeffs,
    )
    .unwrap()
    .unwrap();

    assert_eq!(got.table_len(), expected.table_len());
    assert_eq!(got.indices(), expected.indices());
    assert_eq!(got.num_entries(), expected.num_entries());
    for row in 0..got.num_entries() {
        assert_eq!(got.value(row), expected.value(row));
    }
}

/// Diagnostic for the EOR `np = 1` plateau: dump the within-chunk hot-position
/// distribution (`raw >> lw`, equivalently `tail % stride`) read off a *real*
/// tensor-packed witness, and show the fold plateau is `log2(stride)` rounds
/// long *regardless* of how that distribution looks.
///
/// The hot positions are uniformly spread (random `raw`, exactly like
/// `examples/profile/workload.rs`: seed `0xbeef_cafe`, `gen_range(0..onehot_k)`,
/// every chunk active), yet `entries_len` is provably flat for `log2(stride)`
/// rounds. So the plateau comes from the one-entry-per-power-of-two-window
/// layout, not from any "alignment" of the hot positions.
///
/// `onehot_k = 256` and `width = [E:F] = 4` reproduce the `fp32 onehot_d32`
/// shape (`stride = 64`, expected plateau `log2(64) = 6`). The arity is
/// downscaled (2^14 chunks) purely for test speed; the per-chunk structure is
/// identical to the profiled run.
///
/// See the histogram with:
///   cargo test -p akita-prover np1_offset_distribution_and_plateau -- --nocapture
#[test]
fn np1_offset_distribution_and_plateau() {
    use rand::Rng;
    type F = Prime24Offset3;
    type E = FpExt4<F>;
    const D: usize = 16;

    let onehot_k = 256usize;
    let log_chunks = 14usize;
    let num_chunks = 1usize << log_chunks;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..num_chunks)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();
    let poly = OneHotPoly::<F>::new(onehot_k, D, indices).unwrap();

    let witness = poly.tensor_packed_sparse_witness::<E>().unwrap();

    let (lw, width) = akita_types::tensor_opening_split::<F, E>().unwrap();
    assert!(onehot_k.is_multiple_of(width));
    let stride = onehot_k / width;
    assert!(stride.is_power_of_two() && stride >= 2);
    let s = stride.trailing_zeros() as usize;

    // (a) offset = raw >> lw, recovered from the real witness as tail % stride
    //     because tail = chunk_idx * stride + (raw >> lw).
    let tails: Vec<usize> = witness.indices().to_vec();
    assert_eq!(
        tails.len(),
        num_chunks,
        "all chunks active => one entry each"
    );
    let mut hist = vec![0usize; stride];
    for &t in &tails {
        hist[t % stride] += 1;
    }
    let occupied = hist.iter().filter(|&&c| c > 0).count();
    let min = *hist.iter().min().unwrap();
    let max = *hist.iter().max().unwrap();
    eprintln!(
        "np=1 offset (raw>>lw) distribution: onehot_k={onehot_k} width={width} lw={lw} \
         stride={stride} entries={} occupied_buckets={occupied}/{stride} \
         per-bucket min={min} max={max} mean={}",
        tails.len(),
        tails.len() / stride,
    );
    eprintln!("  histogram[offset 0..{stride}] = {hist:?}");
    assert!(
        occupied > stride / 2,
        "hot positions are spread, not aligned (occupied {occupied}/{stride})"
    );

    // (b) entries_len after r folds == #distinct(tail >> r): flat for r=0..=s,
    //     then halves at r=s+1 — independent of the spread distribution above.
    let distinct_after = |r: usize| -> usize {
        let mut v: Vec<usize> = tails.iter().map(|&t| t >> r).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    eprintln!("plateau (expected entries_len flat for r=0..={s}):");
    for r in 0..=(s + 2) {
        eprintln!(
            "  round r={r:2}: table_len=2^{:<2} entries_len={}",
            log_chunks + s - r,
            distinct_after(r),
        );
    }
    for r in 0..=s {
        assert_eq!(
            distinct_after(r),
            num_chunks,
            "entries_len must stay flat across the log2(stride) plateau (round {r})",
        );
    }
    assert_eq!(
        distinct_after(s + 1),
        num_chunks / 2,
        "first merge halves entries_len exactly one round after the plateau",
    );
}

#[test]
fn wide_matches_reference() {
    type F = Fp64<4294967197>;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let n_a = 3;
    let num_positions_per_block = 4;
    let num_digits = 5;
    let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
        .map(|_| {
            (0..num_positions_per_block * num_digits)
                .map(|_| CyclotomicRing::random(&mut rng))
                .collect()
        })
        .collect();

    let entries = vec![
        block_entry(0, 1),
        block_entry(0, 7),
        block_entry(0, 15),
        block_entry(2, 0),
        block_entry(2, 63),
    ];

    let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
    let a_view = a_flat
        .ring_view::<D>(n_a, num_positions_per_block * num_digits)
        .unwrap();
    let ref_result = inner_ajtai_reference(&a_matrix, &entries, num_digits);
    let wide_result = inner_ajtai_wide_onehot(&a_view, &entries, num_digits);

    assert_eq!(ref_result.len(), wide_result.len());
    for (r, w) in ref_result.iter().zip(wide_result.iter()) {
        assert_eq!(r, w, "wide result must match reference");
    }
}

#[test]
fn wide_matches_reference_fp128() {
    type F = Prime128Offset275;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0xcafe_1234);
    let n_a = 2;
    let num_positions_per_block = 2;
    let num_digits = 3;
    let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
        .map(|_| {
            (0..num_positions_per_block * num_digits)
                .map(|_| CyclotomicRing::random(&mut rng))
                .collect()
        })
        .collect();

    let entries = vec![
        block_entry(0, 0),
        block_entry(0, 5),
        block_entry(0, 32),
        block_entry(0, 63),
        block_entry(1, 10),
    ];

    let a_flat_elems: Vec<CyclotomicRing<F, D>> = a_matrix
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let a_flat = FlatMatrix::from_ring_slice(&a_flat_elems);
    let a_view = a_flat
        .ring_view::<D>(n_a, num_positions_per_block * num_digits)
        .unwrap();
    let ref_result = inner_ajtai_reference(&a_matrix, &entries, num_digits);
    let wide_result = inner_ajtai_wide_onehot(&a_view, &entries, num_digits);

    assert_eq!(ref_result.len(), wide_result.len());
    for (r, w) in ref_result.iter().zip(wide_result.iter()) {
        assert_eq!(r, w, "wide result must match reference (Fp128)");
    }
}

#[test]
fn counting_column_sweep_matches_per_block_reference() {
    type F = Fp64<4294967197>;
    const D: usize = 64;

    let mut rng = StdRng::seed_from_u64(0x51ee_7eed);
    let n_a = 2;
    let num_positions_per_block = 4;
    let num_digits_inner = 3;
    // The production sweep threshold is 32 blocks per worker.
    const BLOCKS_PER_THREAD: usize = 33;
    #[cfg(feature = "parallel")]
    let num_live_blocks = rayon::current_num_threads() * BLOCKS_PER_THREAD;
    #[cfg(not(feature = "parallel"))]
    let num_live_blocks = BLOCKS_PER_THREAD;
    let active_a_cols = num_positions_per_block * num_digits_inner;
    let a_matrix: Vec<Vec<CyclotomicRing<F, D>>> = (0..n_a)
        .map(|_| {
            (0..active_a_cols)
                .map(|_| CyclotomicRing::random(&mut rng))
                .collect()
        })
        .collect();
    let buckets = (0..num_live_blocks)
        .map(|block| {
            vec![
                block_entry(block % num_positions_per_block, 0),
                block_entry(block % num_positions_per_block, 7),
                block_entry(block % num_positions_per_block, 31),
                block_entry((block + 1) % num_positions_per_block, 5),
                block_entry((block + 1) % num_positions_per_block, 19),
            ]
        })
        .collect::<Vec<_>>();
    let blocks = super::test_helpers::from_buckets(buckets.clone());
    let block_views = (0..num_live_blocks)
        .map(|block| blocks.block(block))
        .collect::<Vec<_>>();
    let a_flat =
        FlatMatrix::from_ring_slice(&a_matrix.iter().flatten().copied().collect::<Vec<_>>());
    let a_view = a_flat.ring_view::<D>(n_a, active_a_cols).unwrap();

    let got = column_sweep_ajtai_onehot::<F, D>(
        &a_view,
        &block_views,
        n_a,
        active_a_cols,
        num_digits_inner,
    );
    let expected = buckets
        .iter()
        .map(|entries| inner_ajtai_reference::<F, D>(&a_matrix, entries, num_digits_inner))
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

// -------------------------------------------------------------------------
// Tests that exercise the column-sweep kernels and the OneHotPoly-level
// behaviour defined above.
// -------------------------------------------------------------------------

#[test]
fn single_chunk_onehot_large_block_uses_safe_accumulator_path() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let num_positions_per_block = F::MAX_COMMIT_ACCUMULATIONS + 1;
    let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
    let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
    let a_matrix = [vec![dense_ring; num_positions_per_block]];
    let bucket: Vec<SparseRingBlockEntry> = (0..num_positions_per_block)
        .map(|pos| block_entry(pos, pos % D))
        .collect();
    let single_chunk_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);

    let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
    let a_view = a_flat.ring_view::<D>(1, num_positions_per_block).unwrap();

    let single_chunk_views: Vec<&[SparseRingBlockEntry]> = (0..single_chunk_blocks
        .num_live_blocks())
        .map(|i| single_chunk_blocks.block(i))
        .collect();
    let got = column_sweep_ajtai_onehot::<F, D>(
        &a_view,
        &single_chunk_views,
        1,
        num_positions_per_block,
        1,
    );
    let expected = inner_ajtai_wide_single_chunk_tiled::<F, D>(&a_view, &bucket, 1);

    assert_eq!(got.len(), 1);
    assert_eq!(got[0], expected);
}

#[test]
fn multi_chunk_onehot_large_block_uses_safe_accumulator_path() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let coeffs_per_entry: usize = D / 2;
    let num_entries: usize = F::MAX_COMMIT_ACCUMULATIONS / coeffs_per_entry + 1;
    let total_shift_accumulates: usize = num_entries * coeffs_per_entry;
    assert!(total_shift_accumulates > F::MAX_COMMIT_ACCUMULATIONS);

    let n_a = 1;
    let num_digits_inner = 1;
    let num_positions_per_block = num_entries;

    let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
    let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
    let a_matrix = [vec![dense_ring; num_positions_per_block * num_digits_inner]];

    let nonzero_coeffs: Vec<u16> = (0..coeffs_per_entry as u16).collect();
    let bucket: Vec<SparseRingBlockEntry> = (0..num_positions_per_block)
        .flat_map(|pos| {
            nonzero_coeffs
                .iter()
                .map(move |&coeff| block_entry(pos, usize::from(coeff)))
        })
        .collect();
    let multi_chunk_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);

    let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
    let a_view = a_flat
        .ring_view::<D>(n_a, num_positions_per_block * num_digits_inner)
        .unwrap();

    let views: Vec<&[SparseRingBlockEntry]> = (0..multi_chunk_blocks.num_live_blocks())
        .map(|i| multi_chunk_blocks.block(i))
        .collect();

    let got = column_sweep_ajtai_onehot::<F, D>(
        &a_view,
        &views,
        n_a,
        num_positions_per_block * num_digits_inner,
        num_digits_inner,
    );
    let reference = inner_ajtai_reference::<F, D>(&a_matrix, &bucket, num_digits_inner);

    assert_eq!(got.len(), 1, "single-block test: expected one output row");
    assert_eq!(
        got[0], reference,
        "column_sweep_ajtai_onehot must agree with the non-wide \
         reference above the field's commitment accumulation cap"
    );
}

#[test]
fn repeated_onehot_position_overflow_splits_entries() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let n_a = 1;
    let num_digits_inner = 1;
    let max_coeff = F::from_canonical_u128_reduced((1u128 << 24) - 4);
    let dense_ring = CyclotomicRing::from_coefficients([max_coeff; D]);
    let a_matrix = [vec![dense_ring]];

    let bucket = vec![block_entry(0, 0); F::MAX_COMMIT_ACCUMULATIONS + 1];
    let multi_chunk_blocks = super::test_helpers::from_buckets(vec![bucket.clone()]);
    let views: Vec<&[SparseRingBlockEntry]> = (0..multi_chunk_blocks.num_live_blocks())
        .map(|i| multi_chunk_blocks.block(i))
        .collect();

    let a_flat = FlatMatrix::from_ring_slice(&a_matrix[0]);
    let a_view = a_flat.ring_view::<D>(n_a, num_digits_inner).unwrap();

    let got =
        column_sweep_ajtai_onehot::<F, D>(&a_view, &views, n_a, num_digits_inner, num_digits_inner);
    let reference = inner_ajtai_reference::<F, D>(&a_matrix, &bucket, num_digits_inner);

    assert_eq!(got[0], reference);
}

#[test]
fn batched_single_chunk_onehot_decompose_fold_matches_individual_aggregation() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let num_positions_per_block = 64;
    let mut indices0 = vec![None; 128];
    indices0[0] = Some(1usize);
    indices0[17] = Some(5usize);
    indices0[64] = Some(9usize);
    indices0[91] = Some(33usize);
    let mut indices1 = vec![None; 128];
    indices1[3] = Some(7usize);
    indices1[29] = Some(11usize);
    indices1[64] = Some(19usize);
    indices1[100] = Some(21usize);
    let polys = [
        OneHotPoly::<F>::new(num_positions_per_block, D, indices0).unwrap(),
        OneHotPoly::<F>::new(num_positions_per_block, D, indices1).unwrap(),
    ];
    let challenges = vec![
        SparseChallenge {
            positions: vec![0, 5].into(),
            coeffs: vec![1, -1].into(),
        },
        SparseChallenge {
            positions: vec![2, 7].into(),
            coeffs: vec![1, 1].into(),
        },
        SparseChallenge {
            positions: vec![4, 11].into(),
            coeffs: vec![-1, 2].into(),
        },
        SparseChallenge {
            positions: vec![8, 13].into(),
            coeffs: vec![1, -2].into(),
        },
    ];

    let expected = aggregate_witnesses::<F, D>(
        &polys
            .iter()
            .zip(challenges.chunks(2))
            .map(|(poly, poly_challenges)| {
                poly.decompose_fold::<D>(poly_challenges, num_positions_per_block, 1, 0)
            })
            .collect::<Vec<_>>(),
    );
    let poly_refs: Vec<&OneHotPoly<F>> = polys.iter().collect();
    let got = OneHotPoly::<F>::decompose_fold_batched::<D>(
        &poly_refs,
        &challenges,
        num_positions_per_block,
        1,
        0,
    )
    .expect("onehot batched path should apply");

    assert_eq!(got, expected);
}

#[test]
fn single_chunk_onehot_evaluate_and_fold_matches_factorized_eval() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let poly =
        OneHotPoly::<F>::new(64, D, vec![Some(1usize), None, Some(9usize), Some(17usize)]).unwrap();
    let num_positions_per_block = 2usize;
    let position_weights = vec![F::from_u64(3), F::from_u64(5)];
    let live_block_weights = vec![F::from_u64(7), F::from_u64(11)];

    let (eval, folded) = poly.evaluate_and_fold::<D>(
        &live_block_weights,
        &position_weights,
        num_positions_per_block,
    );
    let expected_folded = poly.fold_blocks::<D>(&position_weights, num_positions_per_block);
    assert_eq!(folded, expected_folded);

    let full_scalars: Vec<F> = live_block_weights
        .iter()
        .flat_map(|outer| position_weights.iter().map(move |inner| *outer * *inner))
        .collect();
    let expected_eval = super::test_helpers::evaluate_ring_onehot::<F, D, _>(&poly, &full_scalars);
    assert_eq!(eval, expected_eval);
}

#[test]
fn single_chunk_onehot_ring_fold_matches_dense_materialization() {
    type F = Prime24Offset3;
    const D: usize = 8;

    let poly =
        OneHotPoly::<F>::new(16, D, vec![Some(1usize), None, Some(13usize), Some(7usize)]).unwrap();
    let dense = materialize_onehot_as_dense::<F, D, _>(&poly);
    let num_positions_per_block = 4usize;
    let position_weights = vec![
        test_ring_scalar::<F, D>(10),
        test_ring_scalar::<F, D>(40),
        test_ring_scalar::<F, D>(90),
        test_ring_scalar::<F, D>(120),
    ];

    assert_eq!(
        poly.fold_blocks_ring(&position_weights, num_positions_per_block),
        dense.fold_blocks_ring(&position_weights, num_positions_per_block)
    );
}

#[test]
fn onehot_ring_fold_matches_dense_for_partial_final_slice() {
    type F = Prime24Offset3;
    const D: usize = 8;

    let poly =
        OneHotPoly::<F>::new(16, D, vec![Some(1usize), None, Some(13usize), Some(7usize)]).unwrap();
    let dense = materialize_onehot_as_dense::<F, D, _>(&poly);
    let num_positions_per_block = 16usize;
    let position_weights = (0..num_positions_per_block)
        .map(|index| test_ring_scalar::<F, D>(10 + index as u64))
        .collect::<Vec<_>>();

    assert_eq!(
        poly.fold_blocks_ring(&position_weights, num_positions_per_block),
        dense.fold_blocks_ring(&position_weights, num_positions_per_block)
    );
}

#[test]
fn multi_chunk_onehot_evaluate_and_fold_matches_factorized_eval() {
    type F = Prime24Offset3;
    const D: usize = 64;

    let poly = OneHotPoly::<F>::new(
        32,
        D,
        vec![
            Some(1usize),
            None,
            Some(7usize),
            Some(12usize),
            None,
            Some(3usize),
            None,
            Some(15usize),
        ],
    )
    .unwrap();
    let num_positions_per_block = 2usize;
    let position_weights = vec![F::from_u64(2), F::from_u64(4)];
    let live_block_weights = vec![F::from_u64(3), F::from_u64(5)];

    let (eval, folded) = poly.evaluate_and_fold::<D>(
        &live_block_weights,
        &position_weights,
        num_positions_per_block,
    );
    let expected_folded = poly.fold_blocks::<D>(&position_weights, num_positions_per_block);
    assert_eq!(folded, expected_folded);

    let full_scalars: Vec<F> = live_block_weights
        .iter()
        .flat_map(|outer| position_weights.iter().map(move |inner| *outer * *inner))
        .collect();
    let expected_eval = super::test_helpers::evaluate_ring_onehot::<F, D, _>(&poly, &full_scalars);
    assert_eq!(eval, expected_eval);
}

#[test]
fn multi_chunk_onehot_ring_fold_matches_dense_materialization() {
    type F = Prime24Offset3;
    const D: usize = 16;

    let poly = OneHotPoly::<F>::new(
        4,
        D,
        vec![
            Some(0usize),
            Some(3usize),
            None,
            Some(2usize),
            Some(1usize),
            None,
            Some(3usize),
            Some(0usize),
            None,
            Some(2usize),
            Some(1usize),
            None,
            Some(3usize),
            None,
            Some(0usize),
            Some(2usize),
        ],
    )
    .unwrap();
    let dense = materialize_onehot_as_dense::<F, D, _>(&poly);
    let num_positions_per_block = 2usize;
    let position_weights = vec![test_ring_scalar::<F, D>(7), test_ring_scalar::<F, D>(80)];

    assert_eq!(
        poly.fold_blocks_ring(&position_weights, num_positions_per_block),
        dense.fold_blocks_ring(&position_weights, num_positions_per_block)
    );
}

mod layout_and_ownership;
mod optimized_commit;
