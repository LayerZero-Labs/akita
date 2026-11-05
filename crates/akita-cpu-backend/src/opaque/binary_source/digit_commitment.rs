//! Public digit commitments through signed 16-bit limb products.
//!
//! [`binary_public_digit_commitment`] multiplies every setup entry, 64
//! canonical residues below `P = 2^128 - 275`, by its chunk of balanced
//! digits in `Z[X]/(X^64 + 1)` with exact 192-bit integer additions. Here a
//! residue `a` is read as eight little-endian 16-bit limbs `a_j`, shifted to
//! signed limbs `s_j = a_j - 2^15`, so `a = sum_j 2^(16 j) s_j + H` with
//! `H = sum_j 2^(16 j + 15)`. Output coefficient `k` of one column's product
//! is `sum_t c_k[t] a[t]`, where `c_k[t]` is the digit `d[k - t]`, or
//! `-d[k - t + 64]` when `t > k`. It therefore equals
//! `sum_j 2^(16 j) L_j[k] + H Z[k]` with the limb sums
//! `L_j[k] = sum_t c_k[t] s_j[t]` and the digit sum `Z[k] = sum_t c_k[t]`.
//! Both are exact `i32` sums over at most [`BLOCK_COLUMNS`] columns, and the
//! payload adds their combination in the field. That is the reference
//! payload's integer modulo `P`, so the payload is identical.

use akita_error::{checked, AkitaError};
use akita_params::binary_setup::SetupStreamLabel;
use akita_types::binary_source::{
    binary_public_digit_commitment, digit_commitment_setup_column, validate_digit_commitment_label,
    BINARY_DIGIT_COMMITMENT_DEGREE,
};
use jolt_field::Field;

const DEGREE: usize = BINARY_DIGIT_COMMITMENT_DEGREE;
const LIMBS: usize = 8;
/// Every digit `c_k[t]` has magnitude at most 4, every signed limb at most
/// `2^15`, so one column adds at most `64 * 4 * 2^15 = 2^23` to a limb sum,
/// and this many columns add at most `2^30`, well inside `i32`.
const BLOCK_COLUMNS: usize = 128;

/// Setup columns in one parallel piece of a public digit commitment.
const PIECE_COLUMNS: usize = 256;

/// Signed limbs `s_j[t]` of one setup entry, coefficient-major.
type Limbs = [[i16; LIMBS]; DEGREE];
/// Limb sums `L_j[k]` of one row, coefficient-major.
type LimbSums = [[i32; LIMBS]; DEGREE];

/// Compute [`binary_public_digit_commitment`] as the field sum of disjoint
/// column-aligned pieces, which run in parallel under the `parallel` feature.
/// The sum is exact, so the payload equals the one-piece commitment.
pub(super) fn digit_commitment<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    digits: &[i8],
) -> Result<Vec<F>, AkitaError> {
    let rows = validate_digit_commitment_label::<F>(label)?;
    if digits.is_empty() {
        // The reference rejects the empty table with its own message.
        return binary_public_digit_commitment(seed, label, 0, digits);
    }
    let piece = checked::product([PIECE_COLUMNS, DEGREE]).ok_or(AkitaError::InvalidProof)?;
    let commit = |(index, chunk): (usize, &[i8])| {
        let first = checked::product([index, PIECE_COLUMNS]).ok_or(AkitaError::InvalidProof)?;
        piece_commitment::<F>(seed, label, rows, first, chunk)
    };
    #[cfg(feature = "parallel")]
    let pieces = {
        use rayon::prelude::*;
        digits
            .par_chunks(piece)
            .enumerate()
            .map(commit)
            .collect::<Result<Vec<_>, _>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let pieces = digits
        .chunks(piece)
        .enumerate()
        .map(commit)
        .collect::<Result<Vec<_>, _>>()?;
    let mut pieces = pieces.into_iter();
    let mut payload = pieces.next().ok_or(AkitaError::InvalidProof)?;
    for other in pieces {
        if other.len() != payload.len() {
            return Err(AkitaError::InvalidProof);
        }
        for (sum, value) in payload.iter_mut().zip(other) {
            *sum += value;
        }
    }
    Ok(payload)
}

/// The payload of the digits starting at setup column `first_column`.
fn piece_commitment<F: Field>(
    seed: &[u8; 32],
    label: SetupStreamLabel,
    rows: usize,
    first_column: usize,
    digits: &[i8],
) -> Result<Vec<F>, AkitaError> {
    if digits.iter().any(|digit| !(-4..=3).contains(digit)) {
        return Err(AkitaError::InvalidInput(
            "binary commitment needs nonempty balanced digits".into(),
        ));
    }
    let mut payload =
        vec![F::zero(); checked::product([rows, DEGREE]).ok_or(AkitaError::InvalidProof)?];
    let mut sums = vec![[[0i32; LIMBS]; DEGREE]; rows];
    let mut digit_sums = [0i32; DEGREE];
    for (block_index, block) in digits.chunks(BLOCK_COLUMNS * DEGREE).enumerate() {
        let block_first = checked::mul_add(block_index, BLOCK_COLUMNS, first_column)
            .ok_or(AkitaError::InvalidProof)?;
        for (index, chunk) in block.chunks(DEGREE).enumerate() {
            let column = checked::sum([block_first, index]).ok_or(AkitaError::InvalidProof)?;
            let window = digit_window(chunk);
            add_digit_sums(chunk, &mut digit_sums);
            let entries = digit_commitment_setup_column(seed, label, column)?;
            for (entry, row_sums) in entries.iter().zip(sums.iter_mut()) {
                accumulate_column(&signed_limbs(entry), &window, row_sums);
            }
        }
        fold_block(&mut sums, &mut digit_sums, &mut payload);
    }
    Ok(payload)
}

/// The digits `c_k[t] = window[63 - k + t]` of one column's chunk.
///
/// Index `x = 63 - (k - t)` holds `d[63 - x]` for `x <= 63`, and
/// `-d[127 - x]` for `64 <= x <= 126`. A short final chunk is zero-filled.
fn digit_window(chunk: &[i8]) -> [i16; 2 * DEGREE] {
    let mut window = [0i16; 2 * DEGREE];
    for (s, &digit) in chunk.iter().enumerate().take(DEGREE) {
        window[DEGREE - 1 - s] = i16::from(digit);
        if s > 0 {
            window[2 * DEGREE - 1 - s] = -i16::from(digit);
        }
    }
    window
}

/// Add `Z[k] = sum_t c_k[t] = sum_(s <= k) d[s] - sum_(s > k) d[s]`.
fn add_digit_sums(chunk: &[i8], digit_sums: &mut [i32; DEGREE]) {
    let total: i32 = chunk.iter().map(|&digit| i32::from(digit)).sum();
    let mut prefix = 0i32;
    for (k, sum) in digit_sums.iter_mut().enumerate() {
        prefix += chunk.get(k).map_or(0, |&digit| i32::from(digit));
        *sum += 2 * prefix - total;
    }
}

/// Signed limbs `a_j - 2^15` of every residue of one setup entry.
fn signed_limbs(entry: &[u128; DEGREE]) -> Limbs {
    entry.map(|residue| {
        core::array::from_fn(|limb| ((residue >> (16 * limb)) as u16 ^ 0x8000) as i16)
    })
}

/// Add every residue's limb sums to the field payload and clear the sums.
fn fold_block<F: Field>(sums: &mut [LimbSums], digit_sums: &mut [i32; DEGREE], payload: &mut [F]) {
    let limb_weights: [F; LIMBS] = core::array::from_fn(|limb| F::from_u128(1 << (16 * limb)));
    let offset = F::from_u128(0x8000_8000_8000_8000_8000_8000_8000_8000);
    for (row_sums, row_payload) in sums.iter_mut().zip(payload.chunks_exact_mut(DEGREE)) {
        for ((limb_sums, slot), &digit_sum) in
            row_sums.iter_mut().zip(row_payload).zip(digit_sums.iter())
        {
            let mut value = offset * F::from_i128(i128::from(digit_sum));
            for (&limb_sum, &weight) in limb_sums.iter().zip(&limb_weights) {
                value += weight * F::from_i128(i128::from(limb_sum));
            }
            *slot += value;
            *limb_sums = [0; LIMBS];
        }
    }
    *digit_sums = [0; DEGREE];
}

/// Add `L_j[k] += sum_t c_k[t] s_j[t]` for one column.
#[inline]
fn accumulate_column(limbs: &Limbs, window: &[i16; 2 * DEGREE], sums: &mut LimbSums) {
    #[cfg(target_arch = "aarch64")]
    neon::accumulate_column(limbs, window, sums);
    #[cfg(not(target_arch = "aarch64"))]
    accumulate_column_portable(limbs, window, sums);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn accumulate_column_portable(limbs: &Limbs, window: &[i16; 2 * DEGREE], sums: &mut LimbSums) {
    for (k, limb_sums) in sums.iter_mut().enumerate() {
        for (residue, &digit) in limbs.iter().zip(&window[DEGREE - 1 - k..][..DEGREE]) {
            for (sum, &limb) in limb_sums.iter_mut().zip(residue) {
                *sum += i32::from(digit) * i32::from(limb);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{LimbSums, Limbs, DEGREE, LIMBS};
    use std::arch::aarch64::*;

    /// Output coefficients per register tile.
    const TILE: usize = 4;

    pub(super) fn accumulate_column(
        limbs: &Limbs,
        window: &[i16; 2 * DEGREE],
        sums: &mut LimbSums,
    ) {
        for first in (0..DEGREE).step_by(TILE) {
            // SAFETY: `first + TILE <= DEGREE`; see `tile`.
            unsafe { tile(limbs, window, sums, first) }
        }
    }

    /// Accumulate output coefficients `first..first + TILE`.
    ///
    /// Each 16-bit limb vector of residue `t` is multiplied by the digit
    /// `c_k[t]`, taken as lane `t % 8` of the window vector at
    /// `63 - k + 8 (t / 8)`, and added widened to the `i32` sums: the same
    /// products and sums as `accumulate_column_portable`.
    ///
    /// # Safety
    ///
    /// `first + TILE <= DEGREE`. NEON is baseline on aarch64.
    #[inline(always)]
    unsafe fn tile(limbs: &Limbs, window: &[i16; 2 * DEGREE], sums: &mut LimbSums, first: usize) {
        let mut acc = [[vdupq_n_s32(0); 2]; TILE];
        for (lane, acc) in acc.iter_mut().enumerate() {
            let sum = sums[first + lane].as_ptr();
            acc[0] = vld1q_s32(sum);
            acc[1] = vld1q_s32(sum.add(LIMBS / 2));
        }
        for octet in 0..DEGREE / 8 {
            // The window read at `63 - k + 8 octet` for `k >= first` ends at
            // most at `63 + 56 + 8 = 127 < 2 * DEGREE`.
            let digits: [int16x8_t; TILE] = core::array::from_fn(|lane| {
                vld1q_s16(window.as_ptr().add(DEGREE - 1 - (first + lane) + 8 * octet))
            });
            macro_rules! term {
                ($t:literal) => {
                    let residue = vld1q_s16(limbs[8 * octet + $t].as_ptr());
                    let low = vget_low_s16(residue);
                    for (acc, &digits) in acc.iter_mut().zip(&digits) {
                        acc[0] = vmlal_laneq_s16::<$t>(acc[0], low, digits);
                        acc[1] = vmlal_high_laneq_s16::<$t>(acc[1], residue, digits);
                    }
                };
            }
            term!(0);
            term!(1);
            term!(2);
            term!(3);
            term!(4);
            term!(5);
            term!(6);
            term!(7);
        }
        for (lane, acc) in acc.iter().enumerate() {
            let sum = sums[first + lane].as_mut_ptr();
            vst1q_s32(sum, acc[0]);
            vst1q_s32(sum.add(LIMBS / 2), acc[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_params::binary_setup::{SetupStreamModulus, SetupStreamRing, SetupStreamRole};
    use jolt_field::Prime128Offset275 as F;
    use rand::{Rng, SeedableRng};

    fn label() -> SetupStreamLabel {
        SetupStreamLabel {
            role: SetupStreamRole::WitnessCommitment,
            group: 3,
            level: 0,
            rank: 2,
            row: 0,
            column: 0,
            ring: SetupStreamRing::Negacyclic,
            degree: DEGREE as u32,
            modulus: SetupStreamModulus::Prime128Offset275,
        }
    }

    #[test]
    fn column_kernel_matches_portable_sums() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for case in 0..64 {
            let limbs: Limbs = core::array::from_fn(|_| {
                core::array::from_fn(|_| match case % 3 {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => rng.gen(),
                })
            });
            let chunk: Vec<i8> = (0..DEGREE - case % 5)
                .map(|_| {
                    if case % 4 == 0 {
                        -4
                    } else {
                        rng.gen_range(-4..=3)
                    }
                })
                .collect();
            let window = digit_window(&chunk);
            let start: LimbSums = core::array::from_fn(|_| {
                core::array::from_fn(|_| rng.gen_range(-1 << 20..1 << 20))
            });
            let (mut kernel, mut portable) = (start, start);
            accumulate_column(&limbs, &window, &mut kernel);
            accumulate_column_portable(&limbs, &window, &mut portable);
            assert_eq!(kernel, portable);
        }
    }

    #[test]
    fn digit_window_is_the_negacyclic_digit() {
        let chunk: Vec<i8> = (0..DEGREE as i8).map(|s| s % 8 - 4).collect();
        let window = digit_window(&chunk);
        for k in 0..DEGREE {
            for t in 0..DEGREE {
                let expected = if t <= k {
                    i16::from(chunk[k - t])
                } else {
                    -i16::from(chunk[k + DEGREE - t])
                };
                assert_eq!(window[DEGREE - 1 - k + t], expected);
            }
        }
    }

    #[test]
    fn payload_matches_reference_commitment() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let seed = [5u8; 32];
        // Lengths cover one partial column, several blocks and several pieces.
        for len in [
            1,
            63,
            64,
            1000,
            BLOCK_COLUMNS * DEGREE + 17,
            PIECE_COLUMNS * DEGREE * 2 + 5,
        ] {
            for extreme in [None, Some(-4i8), Some(3)] {
                let digits: Vec<i8> = (0..len)
                    .map(|_| extreme.unwrap_or_else(|| rng.gen_range(-4..=3)))
                    .collect();
                let fast = digit_commitment::<F>(&seed, label(), &digits).unwrap();
                let reference =
                    binary_public_digit_commitment::<F>(&seed, label(), 0, &digits).unwrap();
                assert_eq!(fast, reference, "len {len} extreme {extreme:?}");
            }
        }
        assert!(digit_commitment::<F>(&seed, label(), &[]).is_err());
        assert!(digit_commitment::<F>(&seed, label(), &[4]).is_err());
        assert!(digit_commitment::<F>(&seed, label(), &[-5]).is_err());
        let wrong_rank = SetupStreamLabel { rank: 3, ..label() };
        assert!(digit_commitment::<F>(&seed, wrong_rank, &[1]).is_err());
    }
}
