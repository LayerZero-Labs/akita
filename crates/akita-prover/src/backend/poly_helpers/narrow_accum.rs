//! Proven-safe i16 partial accumulators for sparse negacyclic products.

use akita_challenges::SparseChallenge;

#[inline(always)]
fn accumulate_i8_segment(src: &[i8], dst: &mut [i16], scale: i16) {
    debug_assert_eq!(src.len(), dst.len());
    match scale {
        1 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += i16::from(value)),
        -1 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc -= i16::from(value)),
        2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += i16::from(value) << 1),
        -2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc -= i16::from(value) << 1),
        _ => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += i16::from(value) * scale),
    }
}

#[inline(always)]
fn accumulate_i16_segment(src: &[i16], dst: &mut [i16], scale: i16) {
    debug_assert_eq!(src.len(), dst.len());
    match scale {
        1 => src.iter().zip(dst).for_each(|(&value, acc)| *acc += value),
        -1 => src.iter().zip(dst).for_each(|(&value, acc)| *acc -= value),
        2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += value << 1),
        -2 => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc -= value << 1),
        _ => src
            .iter()
            .zip(dst)
            .for_each(|(&value, acc)| *acc += value * scale),
    }
}

/// Accumulate one sparse negacyclic product into a proven-safe i16 partial sum.
///
/// The caller must prove that the complete partial sum stays in the i16 range.
#[inline(always)]
pub(super) fn sparse_mul_acc<const D: usize>(
    digit_plane: &[i8; D],
    challenge: &SparseChallenge,
    acc: &mut [i16; D],
) {
    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());
    sparse_mul_acc_terms(digit_plane, &challenge.positions, &challenge.coeffs, acc);
}

#[inline(always)]
pub(super) fn sparse_mul_acc_terms<const D: usize>(
    digit_plane: &[i8; D],
    positions: &[u32],
    coefficients: &[i8],
    acc: &mut [i16; D],
) {
    debug_assert_eq!(positions.len(), coefficients.len());
    for (&position, &coefficient) in positions.iter().zip(coefficients) {
        debug_assert!(position < D as u32);
        let position = position as usize;
        let split = D - position;
        let scale = i16::from(coefficient);
        accumulate_i8_segment(&digit_plane[..split], &mut acc[position..], scale);
        if position > 0 {
            accumulate_i8_segment(&digit_plane[split..], &mut acc[..position], -scale);
        }
    }
}

/// Signed-i16 source variant of [`sparse_mul_acc`].
#[inline(always)]
pub(super) fn sparse_mul_acc_i16<const D: usize>(
    digit_plane: &[i16; D],
    challenge: &SparseChallenge,
    acc: &mut [i16; D],
) {
    debug_assert_eq!(challenge.positions.len(), challenge.coeffs.len());
    sparse_mul_acc_i16_terms(digit_plane, &challenge.positions, &challenge.coeffs, acc);
}

#[inline(always)]
pub(super) fn sparse_mul_acc_i16_terms<const D: usize>(
    digit_plane: &[i16; D],
    positions: &[u32],
    coefficients: &[i8],
    acc: &mut [i16; D],
) {
    debug_assert_eq!(positions.len(), coefficients.len());
    for (&position, &coefficient) in positions.iter().zip(coefficients) {
        debug_assert!(position < D as u32);
        let position = position as usize;
        let split = D - position;
        let scale = i16::from(coefficient);
        accumulate_i16_segment(&digit_plane[..split], &mut acc[position..], scale);
        if position > 0 {
            accumulate_i16_segment(&digit_plane[split..], &mut acc[..position], -scale);
        }
    }
}
