use super::{
    aligned_i8_tile_width, balanced_digit_abs_bound, base_tile_width,
    centered_quotient_rows_with_i16_tail, decompose_block_i8, fused_split_eq_quotients,
    mat_vec_mul_crt_ntt, mat_vec_mul_crt_ntt_many, mat_vec_mul_digits_i8_block_parallel,
    mat_vec_mul_digits_i8_with_params, mat_vec_mul_ntt_digits_i8, mat_vec_mul_ntt_single_i8_cyclic,
    mat_vec_mul_unchecked, precompute_dense_mat_ntt_with_params,
};
use akita_params::layout::{FlatMatrix, RingMatrixView};
use akita_types::{
    prepare_ntt_cache, select_crt_ntt_params, FieldCrtBases, NttCacheMode, PreparedNttCache,
    ProtocolCrtNttParams,
};
use jolt_algebra::ntt::{
    tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES, Q64_NUM_PRIMES},
    PrimeWidth,
};
use jolt_algebra::CrtNttParamSet;
use jolt_algebra::CyclotomicCrtNtt;
use jolt_algebra::CyclotomicRing;
use jolt_algebra::{CanonicalEncoding, Field, Fp64, One, Prime128Offset275, Prime64Offset59, Ring};

fn field_source<F: Field + CanonicalEncoding, const D: usize>(
    blocks: &[&[CyclotomicRing<F, D>]],
) -> (Vec<CyclotomicRing<F, D>>, Vec<std::ops::Range<usize>>) {
    let rings = blocks
        .iter()
        .flat_map(|block| block.iter().copied())
        .collect();
    let mut start = 0;
    let ranges = blocks
        .iter()
        .map(|block| {
            let r = start..start + block.len();
            start = r.end;
            r
        })
        .collect();
    (rings, ranges)
}

fn mat_vec_mul_i8_with_params<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    matrix: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, akita_error::AkitaError> {
    field_matvec::<F, W, K, D, true>(matrix, blocks, num_digits, log_basis, params)
}
fn mat_vec_mul_i8_dense_with_params<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    matrix: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, akita_error::AkitaError> {
    field_matvec::<F, W, K, D, false>(matrix, blocks, num_digits, log_basis, params)
}
fn field_matvec<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const ZERO: bool,
>(
    matrix: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, akita_error::AkitaError> {
    let (rings, ranges) = field_source(blocks);
    let decomposition =
        jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params::new(num_digits, log_basis);
    let source = super::LiveRings {
        coeffs: &rings,
        params: &decomposition,
        basis: crate::sources::poly_helpers::SignedDigitBasis::new(log_basis)?,
    };
    super::mat_vec_mul_blocks_i8_with_params_impl::<F, _, W, K, D, ZERO>(
        matrix,
        &source,
        &ranges,
        super::DigitShape {
            num_digits,
            log_basis,
        },
        super::I8CommitPolicy::Dense,
        "i8 matvec CRT capacity cannot fit a single term",
        params,
    )
}
fn mat_vec_mul_i8_dense_single_row_with_params<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    matrix: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, akita_error::AkitaError> {
    let (rings, ranges) = field_source(blocks);
    let decomposition =
        jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params::new(num_digits, log_basis);
    let source = super::LiveRings {
        coeffs: &rings,
        params: &decomposition,
        basis: crate::sources::poly_helpers::SignedDigitBasis::new(log_basis)?,
    };
    super::mat_vec_mul_blocks_i8_single_row_with_params(
        matrix,
        &source,
        &ranges,
        super::DigitShape {
            num_digits,
            log_basis,
        },
        params,
    )
}
fn mat_vec_mul_ntt_i8_dense_single_row<F: Field + CanonicalEncoding, const D: usize>(
    slot: &PreparedNttCache<D>,
    width: usize,
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, akita_error::AkitaError> {
    let (rings, ranges) = field_source(blocks);
    let decomposition =
        jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params::new(num_digits, log_basis);
    let source = super::LiveRings {
        coeffs: &rings,
        params: &decomposition,
        basis: crate::sources::poly_helpers::SignedDigitBasis::new(log_basis)?,
    };
    Ok(super::mat_vec_mul_ntt_blocks_i8(
        slot,
        1,
        width,
        &source,
        &ranges,
        super::DigitShape {
            num_digits,
            log_basis,
        },
        false,
        super::I8CommitPolicy::Dense,
    )?
    .into_iter()
    .map(|mut row| row.remove(0))
    .collect())
}

fn prepare_both_transforms<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
) -> Result<PreparedNttCache<D>, akita_error::AkitaError> {
    prepare_ntt_cache(matrix, NttCacheMode::BothTransforms)
}

fn build_negacyclic_ntt_slot<F: Field + CanonicalEncoding, const D: usize>(
    matrix: RingMatrixView<'_, F, D>,
) -> Result<PreparedNttCache<D>, akita_error::AkitaError> {
    prepare_ntt_cache(
        matrix,
        NttCacheMode::ExactNegacyclic {
            width: 1,
            rhs_abs_bound: 1 << 7,
        },
    )
}

fn centered_i32_ring<F: jolt_algebra::Field + jolt_algebra::CanonicalEncoding, const D: usize>(
    coeffs: &[i32; D],
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| F::from_i64(coeffs[idx] as i64)))
}

fn cyclic_product<F: jolt_algebra::Field, const D: usize>(
    lhs: &CyclotomicRing<F, D>,
    rhs: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let mut out = CyclotomicRing::<F, D>::zero();
    for (i, &a) in lhs.coefficients().iter().enumerate() {
        if a.is_zero() {
            continue;
        }
        for (j, &b) in rhs.coefficients().iter().enumerate() {
            if !b.is_zero() {
                out.coefficients_mut()[(i + j) % D] += a * b;
            }
        }
    }
    out
}

fn mat_vec_mul_i8_with_params_for_log_basis<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_with_params(ntt_mat, blocks, num_digits, log_basis, params).unwrap()
}

fn mat_vec_mul_i8_dense_with_params_for_log_basis<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_dense_with_params(ntt_mat, blocks, num_digits, log_basis, params).unwrap()
}

fn mat_vec_mul_digits_i8_with_params_for_log_basis<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params(ntt_mat, blocks, log_basis, params).unwrap()
}

fn quotient_from_cyclic_and_negacyclic<F: jolt_algebra::Field, const D: usize>(
    cyclic: &CyclotomicRing<F, D>,
    negacyclic: &CyclotomicRing<F, D>,
) -> CyclotomicRing<F, D> {
    let cyc = cyclic.coefficients();
    let neg = negacyclic.coefficients();
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| (cyc[idx] - neg[idx]).half()))
}

fn schoolbook_digit_mat_vec<F: Field + CanonicalEncoding, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    blocks: &[Vec<[i8; D]>],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    blocks
        .iter()
        .map(|block| {
            mat.iter()
                .map(|row| {
                    row.iter().zip(block.iter()).fold(
                        CyclotomicRing::<F, D>::zero(),
                        |mut acc, (lhs, digit)| {
                            let rhs = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                                F::from_i64(i64::from(digit[k]))
                            }));
                            acc += *lhs * rhs;
                            acc
                        },
                    )
                })
                .collect()
        })
        .collect()
}

mod api;
mod chunking;
mod compression;
mod crt_dense;
mod digit_matvec;
mod fused;
mod reduced_profiles;

#[test]
fn raw_block_bound_uses_bounded_scratch_and_live_columns() {
    use crate::sources::block_digits::{I8Rows, PackedDigits};
    use crate::sources::packed_digits::PackedSignedDigits;

    const D: usize = 8;
    let mut digits = vec![1; 23 * D];
    digits[16 * D] = -128;
    digits[21 * D] = 127;
    let packed = PackedSignedDigits::from_i8_digits_auto(digits);
    let source = PackedDigits {
        digits: packed.view(),
        num_rings: 23,
    };
    let shape = super::DigitShape {
        num_digits: 1,
        log_basis: 3,
    };
    let mut scratch = <PackedDigits as I8Rows<D>>::i8_scratch(&source, shape, 4);
    let bound = |width, scratch: &mut Vec<[i8; D]>| {
        super::i8_block_abs_bound(&source, 2, &(5..23), shape, width, 4, scratch).unwrap()
    };
    assert_eq!(bound(11, &mut scratch), 1);
    assert_eq!(bound(12, &mut scratch), 128);
    assert_eq!(bound(18, &mut scratch), 128);
    assert_eq!(scratch.len(), 4);
}
