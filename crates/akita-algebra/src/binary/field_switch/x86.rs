//! AVX-512/GFNI bit-matrix kernels for the binary field switch.

use std::arch::x86_64::{
    __m512i, _mm512_and_si512, _mm512_gf2p8affine_epi64_epi8, _mm512_i64gather_epi64,
    _mm512_loadu_si512, _mm512_permutex2var_epi64, _mm512_permutex2var_epi8, _mm512_set1_epi64,
    _mm512_setr_epi64, _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_slli_epi64,
    _mm512_srli_epi64, _mm512_storeu_si512, _mm512_xor_si512,
};

use super::x86_common::{
    coefficient_matrix, transpose8, LIMB_BYTES, MAX_HOST_BYTES, MAX_SOURCE_BYTES, TILE,
};
use super::{SwitchField, F};

const fn transpose_indices(stage: usize, output_vector_bit: usize) -> [u8; 64] {
    let mut indices = [0u8; 64];
    let mut position = 0;
    while position < 64 {
        let byte = position & 7;
        let source_vector = (byte >> stage) & 1;
        let source_byte = (byte & !(1 << stage)) | (output_vector_bit << stage);
        indices[position] = ((position & !7) | source_byte | (source_vector << 6)) as u8;
        position += 1;
    }
    indices
}

const TRANSPOSE_INDICES: [[[u8; 64]; 2]; 3] = [
    [transpose_indices(0, 0), transpose_indices(0, 1)],
    [transpose_indices(1, 0), transpose_indices(1, 1)],
    [transpose_indices(2, 0), transpose_indices(2, 1)],
];

const REVERSE_QWORD_BYTES: [u8; 64] = [
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12,
    11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15,
    14, 13, 12, 11, 10, 9, 8,
];

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
fn transpose8x8(value: __m512i) -> __m512i {
    let mut value = value;
    let mut swap = _mm512_and_si512(
        _mm512_xor_si512(value, _mm512_srli_epi64::<7>(value)),
        _mm512_set1_epi64(0x00aa_00aa_00aa_00aa),
    );
    value = _mm512_xor_si512(value, _mm512_xor_si512(swap, _mm512_slli_epi64::<7>(swap)));
    swap = _mm512_and_si512(
        _mm512_xor_si512(value, _mm512_srli_epi64::<14>(value)),
        _mm512_set1_epi64(0x0000_cccc_0000_cccc),
    );
    value = _mm512_xor_si512(value, _mm512_xor_si512(swap, _mm512_slli_epi64::<14>(swap)));
    swap = _mm512_and_si512(
        _mm512_xor_si512(value, _mm512_srli_epi64::<28>(value)),
        _mm512_set1_epi64(0x0000_0000_f0f0_f0f0),
    );
    _mm512_xor_si512(value, _mm512_xor_si512(swap, _mm512_slli_epi64::<28>(swap)))
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn swap_dimension(values: [__m512i; 8], stage: usize) -> [__m512i; 8] {
    let mask = 1 << stage;
    let indices = TRANSPOSE_INDICES[stage].map(|bytes| {
        // SAFETY: each index table is exactly one vector load.
        unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) }
    });
    std::array::from_fn(|vector| {
        let low = vector & !mask;
        _mm512_permutex2var_epi8(
            values[low],
            indices[(vector >> stage) & 1],
            values[low | mask],
        )
    })
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn byte_planes(mut values: [__m512i; 8]) -> [__m512i; 8] {
    for stage in 0..3 {
        // SAFETY: inherited target features.
        values = unsafe { swap_dimension(values, stage) };
    }
    values
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn qword_vectors(mut planes: [__m512i; 8]) -> [__m512i; 8] {
    for stage in (0..3).rev() {
        // SAFETY: each swap is an involution, so reversing the stages inverts
        // the byte-plane transpose.
        planes = unsafe { swap_dimension(planes, stage) };
    }
    planes
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn load_limb_planes(base: *const i64, words_per_value: usize, limb: usize) -> [__m512i; 8] {
    let values = std::array::from_fn(|group| {
        if words_per_value == 1 {
            // SAFETY: the caller provides 64 complete one-word values.
            unsafe { _mm512_loadu_si512(base.add(8 * group).cast()) }
        } else if words_per_value == 2 {
            let indices = if limb == 0 {
                _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14)
            } else {
                _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15)
            };
            // SAFETY: two contiguous loads cover eight complete two-word
            // values. The permute extracts the requested limb from each.
            unsafe {
                let start = base.add(16 * group);
                let first = _mm512_loadu_si512(start.cast());
                let second = _mm512_loadu_si512(start.add(8).cast());
                _mm512_permutex2var_epi64(first, indices, second)
            }
        } else {
            let offsets: [i64; 8] =
                std::array::from_fn(|index| ((8 * group + index) * words_per_value + limb) as i64);
            // SAFETY: every generated offset addresses a limb of one of 64
            // complete values supplied by the caller.
            unsafe {
                let offsets = _mm512_loadu_si512(offsets.as_ptr().cast());
                _mm512_i64gather_epi64::<8>(offsets, base)
            }
        }
    });
    // SAFETY: inherited target features.
    unsafe { byte_planes(values) }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn reverse_qword_bytes(value: __m512i) -> __m512i {
    // SAFETY: the shuffle table is exactly one vector load.
    let indices = unsafe { _mm512_loadu_si512(REVERSE_QWORD_BYTES.as_ptr().cast()) };
    _mm512_shuffle_epi8(value, indices)
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn store_qwords(value: __m512i) -> [u64; 8] {
    let mut words = [0u64; 8];
    // SAFETY: `words` contains exactly one unaligned 512-bit store.
    unsafe { _mm512_storeu_si512(words.as_mut_ptr().cast(), value) };
    words
}

/// Compute `p_k = XOR_j bit_k(weights[j]) * source[j]` for complete tiles.
///
/// # Safety
///
/// The caller must establish AVX-512F, AVX-512BW, AVX-512VBMI, and GFNI at
/// runtime. The two inputs must have equal lengths divisible by 64. Only the
/// sealed 64/192 and 128/128 switch profiles may instantiate this function.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
pub(super) unsafe fn partials<H: SwitchField>(source: &[H::Source], weights: &[H]) -> [u128; 192] {
    debug_assert_eq!(source.len(), weights.len());
    debug_assert_eq!(source.len() % TILE, 0);
    let host_bytes = H::ROWS / 8;
    let source_bytes = std::mem::size_of::<H::Source>();
    debug_assert!(host_bytes <= MAX_HOST_BYTES);
    debug_assert!(matches!(source_bytes, 8 | 16));

    let zero = _mm512_setzero_si512();
    let mut accumulators = [zero; MAX_HOST_BYTES * MAX_SOURCE_BYTES];
    for tile in (0..source.len()).step_by(TILE) {
        let source_base = unsafe { source.as_ptr().add(tile).cast::<i64>() };
        let mut source_columns = [zero; MAX_SOURCE_BYTES];
        for limb in 0..source_bytes / 8 {
            // SAFETY: this tile contains 64 complete source values.
            let planes = unsafe { load_limb_planes(source_base, source_bytes / 8, limb) };
            for (byte, &plane) in planes.iter().enumerate() {
                source_columns[8 * limb + byte] = transpose8x8(plane);
            }
        }

        let host_base = unsafe { weights.as_ptr().add(tile).cast::<i64>() };
        let host_words = H::ROWS / 64;
        let mut host_matrices = [zero; MAX_HOST_BYTES];
        for limb in 0..host_words {
            // SAFETY: the sealed host types are transparent arrays of two or
            // three u64 limbs, and this tile contains 64 complete values.
            let planes = unsafe { load_limb_planes(host_base, host_words, limb) };
            for (byte, &plane) in planes.iter().enumerate() {
                // Transpose values into selector rows, then transpose once
                // more to reverse their byte order for GFNI's row convention.
                // SAFETY: inherited target features.
                host_matrices[8 * limb + byte] =
                    unsafe { reverse_qword_bytes(transpose8x8(plane)) };
            }
        }

        for (host_byte, &host_matrix) in host_matrices[..host_bytes].iter().enumerate() {
            for (source_byte, &source_column) in source_columns[..source_bytes].iter().enumerate() {
                let products = _mm512_gf2p8affine_epi64_epi8::<0>(source_column, host_matrix);
                let index = host_byte * MAX_SOURCE_BYTES + source_byte;
                accumulators[index] = _mm512_xor_si512(accumulators[index], products);
            }
        }
    }

    let mut output = [0u128; 192];
    for (host_byte, host_accumulators) in accumulators
        .chunks_exact(MAX_SOURCE_BYTES)
        .take(host_bytes)
        .enumerate()
    {
        for (source_byte, &accumulator) in host_accumulators[..source_bytes].iter().enumerate() {
            // SAFETY: inherited target features.
            let lanes = unsafe { store_qwords(accumulator) };
            for product in lanes {
                // Product bytes are indexed by source bit and contain host
                // bits. Transposing recovers one source byte per host row.
                let rows = transpose8(product);
                for bit in 0..8 {
                    output[8 * host_byte + bit] ^=
                        u128::from((rows >> (8 * bit)) & 0xff) << (8 * source_byte);
                }
            }
        }
    }
    output
}

/// Map host equality weights through the F162 row weights into SoA output.
///
/// # Safety
///
/// The caller must establish AVX-512F, AVX-512BW, AVX-512VBMI, and GFNI at
/// runtime. `weights.len()` must be divisible by 64, every output slice must
/// have exactly that length, and `rows` must contain the caller-validated
/// canonical row weights for `H` (including zero padding where prescribed).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
pub(super) unsafe fn coefficients<H: SwitchField>(
    weights: &[H],
    rows: &[F; 256],
    output: [&mut [u64]; 3],
) {
    debug_assert_eq!(weights.len() % TILE, 0);
    debug_assert!(output.iter().all(|words| words.len() == weights.len()));
    let host_bytes = H::ROWS / 8;
    debug_assert!(host_bytes <= MAX_HOST_BYTES);

    let matrices: [[[u64; 8]; 3]; MAX_HOST_BYTES] = std::array::from_fn(|host_byte| {
        std::array::from_fn(|limb| {
            std::array::from_fn(|byte| {
                if host_byte < host_bytes && byte < LIMB_BYTES[limb] {
                    coefficient_matrix(host_byte, 64 * limb + 8 * byte, rows)
                } else {
                    0
                }
            })
        })
    });

    let [output0, output1, output2] = output;
    let output = [output0, output1, output2];
    for tile in (0..weights.len()).step_by(TILE) {
        let zero = _mm512_setzero_si512();
        let host_words = H::ROWS / 64;
        let host_base = unsafe { weights.as_ptr().add(tile).cast::<i64>() };
        let mut host_planes = [zero; MAX_HOST_BYTES];
        for limb in 0..host_words {
            // SAFETY: the sealed host types are transparent arrays of two or
            // three u64 limbs, and this tile contains 64 complete values.
            let planes = unsafe { load_limb_planes(host_base, host_words, limb) };
            host_planes[8 * limb..8 * limb + 8].copy_from_slice(&planes);
        }

        let mut accumulators = [[zero; 8]; 3];
        for (host_byte, &masks) in host_planes[..host_bytes].iter().enumerate() {
            for limb in 0..3 {
                for byte in 0..LIMB_BYTES[limb] {
                    let matrix = _mm512_set1_epi64(matrices[host_byte][limb][byte] as i64);
                    let mapped = _mm512_gf2p8affine_epi64_epi8::<0>(masks, matrix);
                    accumulators[limb][byte] = _mm512_xor_si512(accumulators[limb][byte], mapped);
                }
            }
        }

        for limb in 0..3 {
            // SAFETY: reversing the three transpose stages reconstructs eight
            // vectors of eight contiguous output values.
            let vectors = unsafe { qword_vectors(accumulators[limb]) };
            for (group, &values) in vectors.iter().enumerate() {
                // SAFETY: each group writes eight complete u64 output words.
                unsafe {
                    _mm512_storeu_si512(
                        output[limb][tile + 8 * group..].as_mut_ptr().cast(),
                        values,
                    )
                };
            }
        }
    }
}
