//! Rotated-challenge accumulation for decompose-fold (dense D64 high-weight path).

use akita_challenges::SparseChallenge;

const D64_ROTATED_CHALLENGE_MIN_WEIGHT: usize = 42;

#[inline(always)]
pub(crate) fn should_use_rotated_challenge<const D: usize>(challenge: &SparseChallenge) -> bool {
    (D == 64 && challenge.positions.len() >= D64_ROTATED_CHALLENGE_MIN_WEIGHT)
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

/// Single-plane rotated accumulation from a pre-materialized signed digit plane.
#[inline(always)]
pub(super) fn accumulate_rotated_digit_plane<T: Copy + Into<i32>, const D: usize>(
    digit_plane: &[T; D],
    rotated: &[[i16; D]],
    acc: &mut [i32; D],
) {
    let bulk_end = D - (D % 3);

    for base in (0..bulk_end).step_by(3) {
        accumulate_rotated_triplet(
            acc,
            [&rotated[base], &rotated[base + 1], &rotated[base + 2]],
            [
                digit_plane[base].into(),
                digit_plane[base + 1].into(),
                digit_plane[base + 2].into(),
            ],
        );
    }

    for (idx, rot) in rotated.iter().enumerate().take(D).skip(bulk_end) {
        let digit: i32 = digit_plane[idx].into();
        if digit != 0 {
            add_scaled_rotated_row(acc, rot, digit);
        }
    }
}
