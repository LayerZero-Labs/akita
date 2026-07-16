//! Rotated-challenge accumulation for decompose-fold (dense D32/D64 high-weight path).

use super::{extract_balanced_digit, peel_first_balanced_digit_i32, to_signed, DecomposeParams};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use jolt_field::CanonicalField;

const D32_ROTATED_CHALLENGE_MIN_WEIGHT: usize = 24;
const D64_ROTATED_CHALLENGE_MIN_WEIGHT: usize = 42;

#[inline(always)]
pub(crate) fn should_use_rotated_challenge<const D: usize>(challenge: &SparseChallenge) -> bool {
    (D == 32 && challenge.positions.len() >= D32_ROTATED_CHALLENGE_MIN_WEIGHT
        || D == 64 && challenge.positions.len() >= D64_ROTATED_CHALLENGE_MIN_WEIGHT)
        && challenge.positions.len() == challenge.coeffs.len()
}

#[inline(always)]
fn add_scaled_rotated_row<const D: usize>(acc: &mut [i32; D], row: &[i16; D], scale: i32) {
    match scale {
        1 => {
            for k in 0..D {
                acc[k] += row[k] as i32;
            }
        }
        -1 => {
            for k in 0..D {
                acc[k] -= row[k] as i32;
            }
        }
        2 => {
            for k in 0..D {
                acc[k] += (row[k] as i32) << 1;
            }
        }
        -2 => {
            for k in 0..D {
                acc[k] -= (row[k] as i32) << 1;
            }
        }
        _ => {
            for k in 0..D {
                acc[k] += scale * row[k] as i32;
            }
        }
    }
}

#[inline(always)]
fn add_scaled_rotated_rows_triplet<const D: usize>(
    acc: &mut [i32; D],
    rows: [&[i16; D]; 3],
    scales: [i32; 3],
) {
    for (k, acc_coeff) in acc.iter_mut().enumerate() {
        *acc_coeff += scales[0] * rows[0][k] as i32
            + scales[1] * rows[1][k] as i32
            + scales[2] * rows[2][k] as i32;
    }
}

#[inline(always)]
fn accumulate_rotated_triplet<const D: usize>(
    acc: &mut [i32; D],
    rots: [&[i16; D]; 3],
    digits: [i32; 3],
) {
    match (digits[0] != 0, digits[1] != 0, digits[2] != 0) {
        (false, false, false) => {}
        (true, false, false) => add_scaled_rotated_row(acc, rots[0], digits[0]),
        (false, true, false) => add_scaled_rotated_row(acc, rots[1], digits[1]),
        (false, false, true) => add_scaled_rotated_row(acc, rots[2], digits[2]),
        _ => add_scaled_rotated_rows_triplet(acc, rots, digits),
    }
}

/// Single-plane rotated accumulation from a pre-materialized i8 digit plane.
#[inline(always)]
pub(super) fn accumulate_rotated_digit_plane<const D: usize>(
    digit_plane: &[i8; D],
    rotated: &[[i16; D]],
    acc: &mut [i32; D],
) {
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        accumulate_rotated_triplet(
            acc,
            [&rotated[base], &rotated[base + 1], &rotated[base + 2]],
            [
                i32::from(digit_plane[base]),
                i32::from(digit_plane[base + 1]),
                i32::from(digit_plane[base + 2]),
            ],
        );
    }

    for (idx, rot) in rotated.iter().enumerate().take(D).skip(bulk_end) {
        let digit = i32::from(digit_plane[idx]);
        if digit != 0 {
            add_scaled_rotated_row(acc, rot, digit);
        }
    }
}

#[inline(always)]
pub(crate) fn decompose_ring_full_challenge_accumulate<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    rotated: &[[i16; D]],
    acc: &mut [[i32; D]],
    p: &DecomposeParams,
) {
    if p.overflow_possible {
        decompose_ring_full_challenge_accumulate_overflow(ring, rotated, acc, p);
    } else {
        decompose_ring_full_challenge_accumulate_fast(ring, rotated, acc, p);
    }
}

#[inline(always)]
fn decompose_ring_full_challenge_accumulate_fast<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    rotated: &[[i16; D]],
    acc: &mut [[i32; D]],
    p: &DecomposeParams,
) {
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        let mut c0 = to_signed(ring.coeffs[base].to_canonical_u128(), p);
        let mut c1 = to_signed(ring.coeffs[base + 1].to_canonical_u128(), p);
        let mut c2 = to_signed(ring.coeffs[base + 2].to_canonical_u128(), p);
        let rot0 = &rotated[base];
        let rot1 = &rotated[base + 1];
        let rot2 = &rotated[base + 2];

        for plane in acc.iter_mut() {
            let d0 = extract_balanced_digit(&mut c0, p);
            let d1 = extract_balanced_digit(&mut c1, p);
            let d2 = extract_balanced_digit(&mut c2, p);
            match (d0 != 0, d1 != 0, d2 != 0) {
                (false, false, false) => {}
                (true, false, false) => add_scaled_rotated_row(plane, rot0, d0),
                (false, true, false) => add_scaled_rotated_row(plane, rot1, d1),
                (false, false, true) => add_scaled_rotated_row(plane, rot2, d2),
                _ => add_scaled_rotated_rows_triplet(plane, [rot0, rot1, rot2], [d0, d1, d2]),
            }
        }
    }

    for (idx, rot) in rotated.iter().enumerate().take(D).skip(bulk_end) {
        let mut c = to_signed(ring.coeffs[idx].to_canonical_u128(), p);
        for plane in acc.iter_mut() {
            let digit = extract_balanced_digit(&mut c, p);
            if digit != 0 {
                add_scaled_rotated_row(plane, rot, digit);
            }
        }
    }
}

#[inline(always)]
fn decompose_ring_full_challenge_accumulate_overflow<F: CanonicalField, const D: usize>(
    ring: &CyclotomicRing<F, D>,
    rotated: &[[i16; D]],
    acc: &mut [[i32; D]],
    p: &DecomposeParams,
) {
    let (first_acc, remaining_acc) = acc
        .split_first_mut()
        .expect("decompose_ring_full_challenge_accumulate_overflow requires at least one plane");
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        let rot0 = &rotated[base];
        let rot1 = &rotated[base + 1];
        let rot2 = &rotated[base + 2];

        let canonical0 = ring.coeffs[base].to_canonical_u128();
        let canonical1 = ring.coeffs[base + 1].to_canonical_u128();
        let canonical2 = ring.coeffs[base + 2].to_canonical_u128();

        let (mut c0, d0) = peel_first_balanced_digit_i32(canonical0, p);
        let (mut c1, d1) = peel_first_balanced_digit_i32(canonical1, p);
        let (mut c2, d2) = peel_first_balanced_digit_i32(canonical2, p);

        if d0 != 0 {
            add_scaled_rotated_row(first_acc, rot0, d0);
        }
        if d1 != 0 {
            add_scaled_rotated_row(first_acc, rot1, d1);
        }
        if d2 != 0 {
            add_scaled_rotated_row(first_acc, rot2, d2);
        }

        for plane in remaining_acc.iter_mut() {
            let d0 = extract_balanced_digit(&mut c0, p);
            let d1 = extract_balanced_digit(&mut c1, p);
            let d2 = extract_balanced_digit(&mut c2, p);
            match (d0 != 0, d1 != 0, d2 != 0) {
                (false, false, false) => {}
                (true, false, false) => add_scaled_rotated_row(plane, rot0, d0),
                (false, true, false) => add_scaled_rotated_row(plane, rot1, d1),
                (false, false, true) => add_scaled_rotated_row(plane, rot2, d2),
                _ => add_scaled_rotated_rows_triplet(plane, [rot0, rot1, rot2], [d0, d1, d2]),
            }
        }
    }

    for (idx, rot) in rotated.iter().enumerate().take(D).skip(bulk_end) {
        let canonical = ring.coeffs[idx].to_canonical_u128();
        let (mut c, d0) = peel_first_balanced_digit_i32(canonical, p);
        if d0 != 0 {
            add_scaled_rotated_row(first_acc, rot, d0);
        }
        for plane in remaining_acc.iter_mut() {
            let digit = extract_balanced_digit(&mut c, p);
            if digit != 0 {
                add_scaled_rotated_row(plane, rot, digit);
            }
        }
    }
}
