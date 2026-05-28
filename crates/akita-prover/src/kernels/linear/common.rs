use super::*;

#[inline]
pub(super) fn accumulate_pointwise_product_into<W: PrimeWidth, const K: usize, const D: usize>(
    acc: &mut CyclotomicCrtNtt<W, K, D>,
    lhs: &CyclotomicCrtNtt<W, K, D>,
    rhs: &CyclotomicCrtNtt<W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    acc.add_assign_pointwise_mul_with_params(lhs, rhs, params);
}

#[inline]
pub(super) fn is_zero_plane<const D: usize>(plane: &[i8; D]) -> bool {
    plane.iter().all(|&d| d == 0)
}

#[inline]
pub(super) fn is_zero_centered_row<const D: usize>(row: &[i32; D]) -> bool {
    row.iter().all(|&d| d == 0)
}

pub(super) const I8_RHS_MAX_ABS: u128 = 32;
#[cfg(target_arch = "aarch64")]
pub(super) const TARGET_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(target_arch = "x86_64")]
pub(super) const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(super) const TARGET_L2_CACHE_BYTES: usize = 1024 * 1024;
pub(super) const CENTERED_LUT_MAX_ABS: u32 = (1 << 16) - 1;
pub(super) const SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS: usize = 4;
pub(super) const SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS: usize = 16;

#[inline]
fn bit_len_u128(x: u128) -> u32 {
    if x == 0 {
        0
    } else {
        u128::BITS - x.leading_zeros()
    }
}

#[inline]
fn bit_len_usize(x: usize) -> u32 {
    if x == 0 {
        0
    } else {
        usize::BITS - x.leading_zeros()
    }
}

#[inline]
fn crt_accumulation_chunk_width_with_terms<F: CanonicalField, W: PrimeWidth, const K: usize>(
    rhs_max_abs: u128,
    coefficient_terms: usize,
    max_width: usize,
) -> usize {
    let q = (-F::one()).to_canonical_u128() + 1;
    let q_half_bits = bit_len_u128(q / 2);
    let prime_bits = match size_of::<W>() {
        2 => 14,
        4 => 30,
        _ => (size_of::<W>() as u32) * 8,
    };
    let product_half_bits = (K as u32).saturating_mul(prime_bits).saturating_sub(1);
    let rhs_bits = bit_len_u128(rhs_max_abs.max(1));
    let terms_bits = bit_len_usize(coefficient_terms.max(1));
    let used_bits = q_half_bits + rhs_bits + terms_bits;
    let chunk_bits = product_half_bits.saturating_sub(used_bits);
    let width = if chunk_bits >= usize::BITS {
        usize::MAX
    } else {
        1usize << chunk_bits
    };
    width.clamp(1, max_width.max(1))
}

#[inline]
pub(super) fn crt_accumulation_chunk_width<
    F: CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    rhs_max_abs: u128,
    max_width: usize,
) -> usize {
    crt_accumulation_chunk_width_with_terms::<F, W, K>(rhs_max_abs, D.max(1), max_width)
}

#[inline]
pub(super) fn crt_field_rhs_accumulation_chunk_width<
    F: CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    rhs_max_abs: u128,
    max_width: usize,
) -> usize {
    crt_accumulation_chunk_width::<F, W, K, D>(rhs_max_abs, max_width)
}

#[inline]
pub(super) fn add_ring_into<F: FieldCore, const D: usize>(
    lhs: &mut CyclotomicRing<F, D>,
    rhs: CyclotomicRing<F, D>,
) {
    *lhs += rhs;
}

pub(super) fn max_centered_abs<F: CanonicalField, const D: usize>(
    rings: &[CyclotomicRing<F, D>],
) -> u128 {
    let q = (-F::one()).to_canonical_u128() + 1;
    let half_q = q / 2;
    let mut max_abs = 0u128;
    for ring in rings {
        for coeff in ring.coefficients() {
            let canonical = coeff.to_canonical_u128();
            let abs = if canonical > half_q {
                q - canonical
            } else {
                canonical
            };
            max_abs = max_abs.max(abs);
        }
    }
    max_abs
}

#[inline]
#[cfg(all(test, not(feature = "zk")))]
pub(super) fn aligned_i8_tile_width(
    raw_width: usize,
    inner_width: usize,
    num_digits: usize,
) -> usize {
    debug_assert!(inner_width > 0);
    debug_assert!(num_digits > 0);

    if inner_width <= num_digits {
        return inner_width;
    }

    let clamped = raw_width.min(inner_width).max(num_digits);
    ((clamped / num_digits).max(1)) * num_digits
}

#[inline]
pub(super) fn bounded_i8_tile_width(
    raw_width: usize,
    inner_width: usize,
    num_digits: usize,
) -> usize {
    debug_assert!(inner_width > 0);
    debug_assert!(num_digits > 0);

    let clamped = raw_width.min(inner_width).max(1);
    if clamped < num_digits {
        clamped
    } else {
        ((clamped / num_digits).max(1)) * num_digits
    }
}
