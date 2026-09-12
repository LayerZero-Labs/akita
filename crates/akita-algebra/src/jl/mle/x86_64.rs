//! x86-64 selector decoding for generic field contractions.

use super::TERNARY4_PATTERN_COUNT;
use jolt_field::Field;
use std::arch::x86_64::*;

const AVX2_ROW_PAIRS: usize = 32;
const AVX512_ROW_PAIRS: usize = 64;

/// A nibble contributes one bit to each of four base-three digits. Two
/// Rademacher planes add their contributions to obtain an index in `0..81`.
const TERNARY_NIBBLE_WEIGHT: [u8; 16] = [0, 27, 9, 36, 3, 30, 12, 39, 1, 28, 10, 37, 4, 31, 13, 40];

pub(super) type RowKernel<F> =
    unsafe fn(&[u8], &[u8], &mut [F], &[F; TERNARY4_PATTERN_COUNT]) -> usize;

/// Select once per contraction panel, outside the column-group loop.
pub(super) fn selected_row_kernel<F: Field>(complete_pairs: usize) -> Option<RowKernel<F>> {
    if complete_pairs >= AVX512_ROW_PAIRS
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
    {
        Some(contract_rows_avx512::<F>)
    } else if complete_pairs >= AVX2_ROW_PAIRS && std::arch::is_x86_feature_detected!("avx2") {
        Some(contract_rows_avx2::<F>)
    } else {
        None
    }
}

#[target_feature(enable = "avx2")]
unsafe fn contract_rows_avx2<F: Field>(
    first: &[u8],
    second: &[u8],
    rows: &mut [F],
    lut: &[F; TERNARY4_PATTERN_COUNT],
) -> usize {
    let complete_pairs = first.len().min(second.len()).min(rows.len() / 2);
    let vector_pairs = complete_pairs / AVX2_ROW_PAIRS * AVX2_ROW_PAIRS;
    let weights =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(TERNARY_NIBBLE_WEIGHT.as_ptr().cast()));
    let nibble_mask = _mm256_set1_epi8(0x0f);

    for pair in (0..vector_pairs).step_by(AVX2_ROW_PAIRS) {
        let first_bytes = _mm256_loadu_si256(first[pair..].as_ptr().cast());
        let second_bytes = _mm256_loadu_si256(second[pair..].as_ptr().cast());
        let even = selector_indices_avx2(first_bytes, second_bytes, weights, nibble_mask);
        let odd = selector_indices_avx2(
            _mm256_srli_epi16::<4>(first_bytes),
            _mm256_srli_epi16::<4>(second_bytes),
            weights,
            nibble_mask,
        );
        let mut even_indices = [0u8; AVX2_ROW_PAIRS];
        let mut odd_indices = [0u8; AVX2_ROW_PAIRS];
        _mm256_storeu_si256(even_indices.as_mut_ptr().cast(), even);
        _mm256_storeu_si256(odd_indices.as_mut_ptr().cast(), odd);
        accumulate_indices(
            &mut rows[2 * pair..2 * (pair + AVX2_ROW_PAIRS)],
            &even_indices,
            &odd_indices,
            lut,
        );
    }
    vector_pairs
}

#[target_feature(enable = "avx2")]
unsafe fn selector_indices_avx2(
    first: __m256i,
    second: __m256i,
    weights: __m256i,
    nibble_mask: __m256i,
) -> __m256i {
    let first = _mm256_and_si256(first, nibble_mask);
    let second = _mm256_and_si256(second, nibble_mask);
    _mm256_add_epi8(
        _mm256_shuffle_epi8(weights, first),
        _mm256_shuffle_epi8(weights, second),
    )
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn contract_rows_avx512<F: Field>(
    first: &[u8],
    second: &[u8],
    rows: &mut [F],
    lut: &[F; TERNARY4_PATTERN_COUNT],
) -> usize {
    let complete_pairs = first.len().min(second.len()).min(rows.len() / 2);
    let vector_pairs = complete_pairs / AVX512_ROW_PAIRS * AVX512_ROW_PAIRS;
    let weights = _mm512_broadcast_i32x4(_mm_loadu_si128(TERNARY_NIBBLE_WEIGHT.as_ptr().cast()));
    let nibble_mask = _mm512_set1_epi8(0x0f);

    for pair in (0..vector_pairs).step_by(AVX512_ROW_PAIRS) {
        let first_bytes = _mm512_loadu_si512(first[pair..].as_ptr().cast());
        let second_bytes = _mm512_loadu_si512(second[pair..].as_ptr().cast());
        let even = selector_indices_avx512(first_bytes, second_bytes, weights, nibble_mask);
        let odd = selector_indices_avx512(
            _mm512_srli_epi16::<4>(first_bytes),
            _mm512_srli_epi16::<4>(second_bytes),
            weights,
            nibble_mask,
        );
        let mut even_indices = [0u8; AVX512_ROW_PAIRS];
        let mut odd_indices = [0u8; AVX512_ROW_PAIRS];
        _mm512_storeu_si512(even_indices.as_mut_ptr().cast(), even);
        _mm512_storeu_si512(odd_indices.as_mut_ptr().cast(), odd);
        accumulate_indices(
            &mut rows[2 * pair..2 * (pair + AVX512_ROW_PAIRS)],
            &even_indices,
            &odd_indices,
            lut,
        );
    }
    vector_pairs
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn selector_indices_avx512(
    first: __m512i,
    second: __m512i,
    weights: __m512i,
    nibble_mask: __m512i,
) -> __m512i {
    let first = _mm512_and_si512(first, nibble_mask);
    let second = _mm512_and_si512(second, nibble_mask);
    _mm512_add_epi8(
        _mm512_shuffle_epi8(weights, first),
        _mm512_shuffle_epi8(weights, second),
    )
}

#[inline]
fn accumulate_indices<F: Field>(
    rows: &mut [F],
    even_indices: &[u8],
    odd_indices: &[u8],
    lut: &[F; TERNARY4_PATTERN_COUNT],
) {
    for ((pair, &even), &odd) in rows.chunks_exact_mut(2).zip(even_indices).zip(odd_indices) {
        pair[0] += lut[usize::from(even)];
        pair[1] += lut[usize::from(odd)];
    }
}

#[cfg(test)]
mod tests {
    use super::super::SELECTORS_TO_TERNARY4;
    use super::*;
    use jolt_field::{
        Ext2, Fp64, FpExt4, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59, Ring, Zero,
    };
    use std::hint::black_box;
    use std::time::Instant;

    type F = Fp64<4_294_967_197>;
    type F32Ext = FpExt4<Prime32Offset99>;
    type F64Ext = Ext2<Prime64Offset59>;
    type F128 = Prime128OffsetA7F7;

    fn lut() -> [F; TERNARY4_PATTERN_COUNT] {
        std::array::from_fn(|index| F::from_u64(index as u64))
    }

    fn expected(first: u8, second: u8, odd: bool) -> F {
        let selector = if odd {
            (first >> 4) | (second & 0xf0)
        } else {
            (first & 0x0f) | ((second & 0x0f) << 4)
        };
        F::from_u64(u64::from(SELECTORS_TO_TERNARY4[usize::from(selector)]))
    }

    fn assert_forced_kernel(
        pairs: usize,
        kernel: unsafe fn(&[u8], &[u8], &mut [F], &[F; 81]) -> usize,
    ) {
        let table = lut();
        for block in (0..=u16::MAX as usize).step_by(pairs) {
            let mut first = vec![0u8; pairs];
            let mut second = vec![0u8; pairs];
            for (lane, (first, second)) in first.iter_mut().zip(&mut second).enumerate() {
                let selector_pair = (block + lane) as u16;
                *first = selector_pair as u8;
                *second = (selector_pair >> 8) as u8;
            }
            let mut rows = vec![F::from_u64(97); 2 * pairs];
            // SAFETY: each caller checks the target features required by the
            // forced kernel, and all slices have one complete vector batch.
            let consumed = unsafe { kernel(&first, &second, &mut rows, &table) };
            assert_eq!(consumed, pairs);
            for lane in 0..pairs {
                assert_eq!(
                    rows[2 * lane],
                    F::from_u64(97) + expected(first[lane], second[lane], false)
                );
                assert_eq!(
                    rows[2 * lane + 1],
                    F::from_u64(97) + expected(first[lane], second[lane], true)
                );
            }
        }
    }

    fn assert_field_contraction<G: Field + std::fmt::Debug>(
        kernel: unsafe fn(&[u8], &[u8], &mut [G], &[G; 81]) -> usize,
    ) {
        const ROWS: usize = 131;
        const ROW_PAIRS: usize = ROWS.div_ceil(2);
        let first: Vec<u8> = (0..ROW_PAIRS)
            .map(|pair| pair.wrapping_mul(73).wrapping_add(19) as u8)
            .collect();
        let second: Vec<u8> = (0..ROW_PAIRS)
            .map(|pair| pair.wrapping_mul(151).wrapping_add(41) as u8)
            .collect();
        let table: [G; TERNARY4_PATTERN_COUNT] =
            std::array::from_fn(|index| G::from_u64((index * 17 + 5) as u64));
        let initial: Vec<G> = (0..ROWS)
            .map(|row| G::from_u64((row * 29 + 7) as u64))
            .collect();
        let mut expected = initial.clone();
        super::super::accumulate_selector_pairs(&first, &second, &mut expected, &table, 0);

        let mut actual = initial;
        // SAFETY: each caller checks the target features required by the
        // forced kernel, and these slices contain at least one full batch.
        let first_scalar_pair = unsafe { kernel(&first, &second, &mut actual, &table) };
        super::super::accumulate_selector_pairs(
            &first,
            &second,
            &mut actual,
            &table,
            first_scalar_pair,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn forced_avx2_decoding_exhausts_both_selector_bytes() {
        if std::arch::is_x86_feature_detected!("avx2") {
            assert_forced_kernel(AVX2_ROW_PAIRS, contract_rows_avx2::<F>);
            assert_field_contraction::<F32Ext>(contract_rows_avx2::<F32Ext>);
            assert_field_contraction::<F64Ext>(contract_rows_avx2::<F64Ext>);
            assert_field_contraction::<F128>(contract_rows_avx2::<F128>);
        }
    }

    #[test]
    fn forced_avx512_decoding_exhausts_both_selector_bytes() {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            assert_forced_kernel(AVX512_ROW_PAIRS, contract_rows_avx512::<F>);
            assert_field_contraction::<F32Ext>(contract_rows_avx512::<F32Ext>);
            assert_field_contraction::<F64Ext>(contract_rows_avx512::<F64Ext>);
            assert_field_contraction::<F128>(contract_rows_avx512::<F128>);
        }
    }

    #[test]
    #[ignore = "private x86 selector-kernel microbenchmark"]
    fn selector_kernel_microbenchmark() {
        const PAIRS: usize = 256;
        const ITERATIONS: usize = 20_000;
        let first: Vec<u8> = (0..PAIRS).map(|i| i.wrapping_mul(73) as u8).collect();
        let second: Vec<u8> = (0..PAIRS).map(|i| i.wrapping_mul(151) as u8).collect();
        let table = lut();

        let elapsed = time_kernel(
            ITERATIONS,
            &first,
            &second,
            &table,
            contract_rows_scalar::<F>,
        );
        eprintln!("scalar: {elapsed:?}");

        if std::arch::is_x86_feature_detected!("avx2") {
            let elapsed = time_kernel(ITERATIONS, &first, &second, &table, contract_rows_avx2::<F>);
            eprintln!("avx2:   {elapsed:?}");
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            let elapsed = time_kernel(
                ITERATIONS,
                &first,
                &second,
                &table,
                contract_rows_avx512::<F>,
            );
            eprintln!("avx512: {elapsed:?}");
        }
    }

    #[test]
    #[ignore = "private x86 end-to-end MLE microbenchmark"]
    fn matrix_mle_microbenchmark() {
        const ROWS: usize = 256;
        const COLS: usize = 1 << 14;
        const ITERATIONS: usize = 200;
        let shape = crate::jl::TernaryProjectionShape::new(ROWS, COLS).unwrap();
        let first: Vec<u8> = (0..shape.plane_len())
            .map(|index| index.wrapping_mul(73).wrapping_add(19) as u8)
            .collect();
        let second: Vec<u8> = (0..shape.plane_len())
            .map(|index| index.wrapping_mul(151).wrapping_add(41) as u8)
            .collect();
        let matrix =
            crate::jl::TernaryProjectionMatrix::from_rademacher_bitplanes(shape, first, second)
                .unwrap();
        let col_weights: Vec<F> = (0..COLS)
            .map(|index| F::from_u64((index * 17 + 5) as u64))
            .collect();
        let row_eq: Vec<F> = (0..ROWS)
            .map(|index| F::from_u64((index * 29 + 7) as u64))
            .collect();
        let row_point: Vec<F> = (0..shape.row_num_vars().unwrap())
            .map(|index| F::from_u64((index * 31 + 11) as u64))
            .collect();
        let col_point: Vec<F> = (0..shape.col_num_vars().unwrap())
            .map(|index| F::from_u64((index * 37 + 13) as u64))
            .collect();

        let scalar = time_contraction(ITERATIONS, &matrix, &col_weights, None);
        eprintln!("full contraction scalar: {scalar:?}");
        if std::arch::is_x86_feature_detected!("avx2") {
            let avx2 = time_contraction(
                ITERATIONS,
                &matrix,
                &col_weights,
                Some(contract_rows_avx2::<F>),
            );
            eprintln!("full contraction avx2:   {avx2:?}");
        }
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            let avx512 = time_contraction(
                ITERATIONS,
                &matrix,
                &col_weights,
                Some(contract_rows_avx512::<F>),
            );
            eprintln!("full contraction avx512: {avx512:?}");
        }

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(
                super::super::eval_ternary_matrix_mle_from_eq_tables(
                    &matrix,
                    &row_eq,
                    &col_weights,
                )
                .unwrap(),
            );
        }
        eprintln!("cached matrix MLE:          {:?}", start.elapsed());

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(super::super::build_ternary_column_weights(&matrix, &row_point).unwrap());
        }
        eprintln!("column weights + row Eq:    {:?}", start.elapsed());

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(crate::EqPolynomial::evals(&col_point).unwrap());
        }
        eprintln!("column Eq table:            {:?}", start.elapsed());

        report_contraction_field::<F32Ext>("fp32ext", 50, &matrix);
        report_contraction_field::<F64Ext>("fp64ext", 50, &matrix);
        report_contraction_field::<F128>("fp128", 50, &matrix);
    }

    fn report_contraction_field<G: Field>(
        label: &str,
        iterations: usize,
        matrix: &crate::jl::TernaryProjectionMatrix,
    ) {
        let col_weights: Vec<G> = (0..matrix.shape().cols())
            .map(|index| G::from_u64((index * 17 + 5) as u64))
            .collect();
        let scalar = time_contraction(iterations, matrix, &col_weights, None);
        let avx2 = if std::arch::is_x86_feature_detected!("avx2") {
            Some(time_contraction(
                iterations,
                matrix,
                &col_weights,
                Some(contract_rows_avx2::<G>),
            ))
        } else {
            None
        };
        let avx512 = if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            Some(time_contraction(
                iterations,
                matrix,
                &col_weights,
                Some(contract_rows_avx512::<G>),
            ))
        } else {
            None
        };
        eprintln!("{label}: scalar={scalar:?}, avx2={avx2:?}, avx512={avx512:?}");
    }

    fn time_contraction<G: Field>(
        iterations: usize,
        matrix: &crate::jl::TernaryProjectionMatrix,
        col_weights: &[G],
        kernel: Option<unsafe fn(&[u8], &[u8], &mut [G], &[G; 81]) -> usize>,
    ) -> std::time::Duration {
        let mut rows = vec![G::zero(); matrix.shape().rows()];
        let start = Instant::now();
        for _ in 0..iterations {
            rows.fill(G::zero());
            contract_for_benchmark(matrix, col_weights, &mut rows, kernel);
            black_box(&rows);
        }
        start.elapsed()
    }

    fn contract_for_benchmark<G: Field>(
        matrix: &crate::jl::TernaryProjectionMatrix,
        col_weights: &[G],
        row_acc: &mut [G],
        kernel: Option<unsafe fn(&[u8], &[u8], &mut [G], &[G; 81]) -> usize>,
    ) {
        let shape = matrix.shape();
        for group in 0..shape.col_groups() {
            let col_start = group * 4;
            let mut values = [G::zero(); 4];
            values.copy_from_slice(&col_weights[col_start..col_start + 4]);
            let table = super::super::build_ternary4_weight_lut(&values);
            let (first, second) = matrix.sign_groups_unchecked(group);
            let first_scalar_pair = kernel.map_or(0, |kernel| {
                // SAFETY: benchmark callers feature-check the forced kernel.
                unsafe { kernel(first, second, row_acc, &table) }
            });
            super::super::accumulate_selector_pairs(
                first,
                second,
                row_acc,
                &table,
                first_scalar_pair,
            );
        }
    }

    unsafe fn contract_rows_scalar<G: Field>(
        first: &[u8],
        second: &[u8],
        rows: &mut [G],
        lut: &[G; TERNARY4_PATTERN_COUNT],
    ) -> usize {
        let complete_pairs = first.len().min(second.len()).min(rows.len() / 2);
        super::super::accumulate_selector_pairs(first, second, rows, lut, 0);
        complete_pairs
    }

    fn time_kernel(
        iterations: usize,
        first: &[u8],
        second: &[u8],
        table: &[F; TERNARY4_PATTERN_COUNT],
        kernel: unsafe fn(&[u8], &[u8], &mut [F], &[F; 81]) -> usize,
    ) -> std::time::Duration {
        let start = Instant::now();
        let mut rows = vec![F::zero(); 2 * first.len()];
        for _ in 0..iterations {
            // SAFETY: the caller checks the target features before selecting
            // this forced benchmark kernel.
            black_box(unsafe { kernel(first, second, &mut rows, table) });
        }
        black_box(rows);
        start.elapsed()
    }
}
