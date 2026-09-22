//! AVX2/GFNI bit-matrix kernels for the binary field switch.

use std::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_blendv_epi8, _mm256_gf2p8affine_epi64_epi8,
    _mm256_i64gather_epi64, _mm256_loadu_si256, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256, _mm256_xor_si256,
};

use super::bit_matrix::transpose8;
use super::x86_common::{coefficient_matrix, LIMB_BYTES, MAX_HOST_BYTES, MAX_SOURCE_BYTES, TILE};
use super::{SwitchField, F};

const HALF_TILE: usize = 32;

const fn transpose_indices(stage: usize, output_vector_bit: usize) -> [u8; 32] {
    let mut indices = [0u8; 32];
    let mut position = 0;
    while position < 32 {
        let byte = position & 7;
        let source_byte = (byte & !(1 << stage)) | (output_vector_bit << stage);
        indices[position] = ((position & !7) | source_byte) as u8;
        position += 1;
    }
    indices
}

const fn transpose_mask(stage: usize) -> [u8; 32] {
    let mut mask = [0u8; 32];
    let mut position = 0;
    while position < 32 {
        mask[position] = if ((position >> stage) & 1) != 0 {
            0xff
        } else {
            0
        };
        position += 1;
    }
    mask
}

const TRANSPOSE_INDICES: [[[u8; 32]; 2]; 3] = [
    [transpose_indices(0, 0), transpose_indices(0, 1)],
    [transpose_indices(1, 0), transpose_indices(1, 1)],
    [transpose_indices(2, 0), transpose_indices(2, 1)],
];
const TRANSPOSE_MASKS: [[u8; 32]; 3] = [transpose_mask(0), transpose_mask(1), transpose_mask(2)];
const REVERSE_QWORD_BYTES: [u8; 32] = [
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12,
    11, 10, 9, 8,
];

#[inline]
#[target_feature(enable = "avx2,gfni")]
fn transpose8x8(mut value: __m256i) -> __m256i {
    let mut swap = _mm256_and_si256(
        _mm256_xor_si256(value, _mm256_srli_epi64::<7>(value)),
        _mm256_set1_epi64x(0x00aa_00aa_00aa_00aa),
    );
    value = _mm256_xor_si256(value, _mm256_xor_si256(swap, _mm256_slli_epi64::<7>(swap)));
    swap = _mm256_and_si256(
        _mm256_xor_si256(value, _mm256_srli_epi64::<14>(value)),
        _mm256_set1_epi64x(0x0000_cccc_0000_cccc),
    );
    value = _mm256_xor_si256(value, _mm256_xor_si256(swap, _mm256_slli_epi64::<14>(swap)));
    swap = _mm256_and_si256(
        _mm256_xor_si256(value, _mm256_srli_epi64::<28>(value)),
        _mm256_set1_epi64x(0x0000_0000_f0f0_f0f0),
    );
    _mm256_xor_si256(value, _mm256_xor_si256(swap, _mm256_slli_epi64::<28>(swap)))
}

#[inline]
#[target_feature(enable = "avx2,gfni")]
unsafe fn swap_dimension(values: [__m256i; 8], stage: usize) -> [__m256i; 8] {
    let indices = TRANSPOSE_INDICES[stage].map(|bytes| {
        // SAFETY: each index table is exactly one vector load.
        unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
    });
    // SAFETY: each mask is exactly one vector load.
    let blend_mask = unsafe { _mm256_loadu_si256(TRANSPOSE_MASKS[stage].as_ptr().cast()) };
    let pair_mask = 1 << stage;
    std::array::from_fn(|vector| {
        let low = vector & !pair_mask;
        let selected = indices[(vector >> stage) & 1];
        _mm256_blendv_epi8(
            _mm256_shuffle_epi8(values[low], selected),
            _mm256_shuffle_epi8(values[low | pair_mask], selected),
            blend_mask,
        )
    })
}

#[inline]
#[target_feature(enable = "avx2,gfni")]
unsafe fn byte_planes(mut values: [__m256i; 8]) -> [__m256i; 8] {
    for stage in 0..3 {
        // SAFETY: inherited target features.
        values = unsafe { swap_dimension(values, stage) };
    }
    values
}

#[inline]
#[target_feature(enable = "avx2,gfni")]
unsafe fn qword_vectors(mut planes: [__m256i; 8]) -> [__m256i; 8] {
    for stage in (0..3).rev() {
        // SAFETY: each swap is an involution, so reversing stages undoes it.
        planes = unsafe { swap_dimension(planes, stage) };
    }
    planes
}

#[inline]
#[target_feature(enable = "avx2,gfni")]
unsafe fn load_limb_planes(base: *const i64, words_per_value: usize, limb: usize) -> [__m256i; 8] {
    let values = std::array::from_fn(|group| {
        if words_per_value == 1 {
            // SAFETY: the caller provides 32 complete one-word values.
            unsafe { _mm256_loadu_si256(base.add(4 * group).cast()) }
        } else if words_per_value == 2 {
            // SAFETY: two contiguous loads cover four complete two-word
            // values. The two permutes extract the requested limb.
            unsafe {
                let start = base.add(8 * group);
                let first = _mm256_loadu_si256(start.cast());
                let second = _mm256_loadu_si256(start.add(4).cast());
                // Each immediate picks the even or odd limb of the two loads.
                let (first, second) = if limb == 0 {
                    (
                        _mm256_permute4x64_epi64::<0x88>(first),
                        _mm256_permute4x64_epi64::<0x88>(second),
                    )
                } else {
                    (
                        _mm256_permute4x64_epi64::<0xdd>(first),
                        _mm256_permute4x64_epi64::<0xdd>(second),
                    )
                };
                _mm256_permute2x128_si256::<0x20>(first, second)
            }
        } else {
            let offsets: [i64; 4] =
                std::array::from_fn(|index| ((4 * group + index) * words_per_value + limb) as i64);
            // SAFETY: all four offsets address limbs of complete host values.
            unsafe {
                let offsets = _mm256_loadu_si256(offsets.as_ptr().cast());
                _mm256_i64gather_epi64::<8>(base, offsets)
            }
        }
    });
    // SAFETY: inherited target features.
    unsafe { byte_planes(values) }
}

#[inline]
#[target_feature(enable = "avx2,gfni")]
unsafe fn reverse_qword_bytes(value: __m256i) -> __m256i {
    // SAFETY: the shuffle table is exactly one vector load.
    let indices = unsafe { _mm256_loadu_si256(REVERSE_QWORD_BYTES.as_ptr().cast()) };
    _mm256_shuffle_epi8(value, indices)
}

#[inline]
#[target_feature(enable = "avx2,gfni")]
unsafe fn store_qwords(value: __m256i) -> [u64; 4] {
    let mut words = [0u64; 4];
    // SAFETY: words contains exactly one unaligned vector store.
    unsafe { _mm256_storeu_si256(words.as_mut_ptr().cast(), value) };
    words
}

/// Compute `p_k = XOR_j bit_k(weights[j]) * source[j]` for complete tiles.
///
/// # Safety
///
/// The caller must establish AVX2 and GFNI at runtime. The two inputs must
/// have equal lengths divisible by 64. Only the sealed 64/192 and 128/128
/// switch profiles may instantiate this function.
#[target_feature(enable = "avx2,gfni")]
pub(super) unsafe fn partials<H: SwitchField>(source: &[H::Source], weights: &[H]) -> [u128; 192] {
    debug_assert_eq!(source.len(), weights.len());
    debug_assert_eq!(source.len() % TILE, 0);
    let host_bytes = H::ROWS / 8;
    let source_bytes = std::mem::size_of::<H::Source>();
    debug_assert!(host_bytes <= MAX_HOST_BYTES);
    debug_assert!(matches!(source_bytes, 8 | 16));

    let zero = _mm256_setzero_si256();
    let mut accumulators = [zero; MAX_HOST_BYTES * MAX_SOURCE_BYTES];
    for half in (0..source.len()).step_by(HALF_TILE) {
        let source_base = unsafe { source.as_ptr().add(half).cast::<i64>() };
        let mut source_columns = [zero; MAX_SOURCE_BYTES];
        for limb in 0..source_bytes / 8 {
            // SAFETY: this half-tile contains 32 complete source values.
            let planes = unsafe { load_limb_planes(source_base, source_bytes / 8, limb) };
            for (byte, &plane) in planes.iter().enumerate() {
                source_columns[8 * limb + byte] = transpose8x8(plane);
            }
        }

        let host_base = unsafe { weights.as_ptr().add(half).cast::<i64>() };
        let host_words = H::ROWS / 64;
        let mut host_matrices = [zero; MAX_HOST_BYTES];
        for limb in 0..host_words {
            // SAFETY: sealed hosts are arrays of two or three u64 limbs.
            let planes = unsafe { load_limb_planes(host_base, host_words, limb) };
            for (byte, &plane) in planes.iter().enumerate() {
                // SAFETY: inherited target features.
                host_matrices[8 * limb + byte] =
                    unsafe { reverse_qword_bytes(transpose8x8(plane)) };
            }
        }

        for (host_byte, &host_matrix) in host_matrices[..host_bytes].iter().enumerate() {
            for (source_byte, &source_column) in source_columns[..source_bytes].iter().enumerate() {
                let products = _mm256_gf2p8affine_epi64_epi8::<0>(source_column, host_matrix);
                let index = host_byte * MAX_SOURCE_BYTES + source_byte;
                accumulators[index] = _mm256_xor_si256(accumulators[index], products);
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

/// Map host equality weights through F162 row weights into SoA output.
///
/// # Safety
///
/// The caller must establish AVX2 and GFNI at runtime. `weights.len()` must
/// be divisible by 64, every output slice must have exactly that length, and
/// `rows` must contain the caller-validated canonical row weights for `H`.
#[target_feature(enable = "avx2,gfni")]
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
    for half in (0..weights.len()).step_by(HALF_TILE) {
        let zero = _mm256_setzero_si256();
        let host_words = H::ROWS / 64;
        let host_base = unsafe { weights.as_ptr().add(half).cast::<i64>() };
        let mut host_planes = [zero; MAX_HOST_BYTES];
        for limb in 0..host_words {
            // SAFETY: sealed hosts are arrays of two or three u64 limbs.
            let planes = unsafe { load_limb_planes(host_base, host_words, limb) };
            host_planes[8 * limb..8 * limb + 8].copy_from_slice(&planes);
        }

        let mut accumulators = [[zero; 8]; 3];
        for (host_byte, &masks) in host_planes[..host_bytes].iter().enumerate() {
            for limb in 0..3 {
                for byte in 0..LIMB_BYTES[limb] {
                    let matrix = _mm256_set1_epi64x(matrices[host_byte][limb][byte] as i64);
                    let mapped = _mm256_gf2p8affine_epi64_epi8::<0>(masks, matrix);
                    accumulators[limb][byte] = _mm256_xor_si256(accumulators[limb][byte], mapped);
                }
            }
        }

        for limb in 0..3 {
            // SAFETY: reverse transpose reconstructs eight vectors of four
            // contiguous output values.
            let vectors = unsafe { qword_vectors(accumulators[limb]) };
            for (group, &values) in vectors.iter().enumerate() {
                // SAFETY: each group writes four complete u64 words.
                unsafe {
                    _mm256_storeu_si256(
                        output[limb][half + 4 * group..].as_mut_ptr().cast(),
                        values,
                    )
                };
            }
        }
    }
}
