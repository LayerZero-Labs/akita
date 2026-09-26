//! Signed-digit multiples of field elements, accumulated in wide lanes.
//!
//! The compact-prefix scan contracts every witness lane against its relation
//! and linear-source factors: `mass[c] += factor * digit(c)` for each of the
//! lane's coefficients `c`. Rows that share a mass block are added together, so
//! a kernel can keep a tile of the block's accumulators in registers across
//! all of them.

use jolt_field::{Field, Unreduced, Zero};

/// Largest digit basis the scan serves.
pub(super) const MAX_DIGIT_BASIS: usize = 8;

/// Mass slots in blocks that always receive whole digit rows.
///
/// A block is reduced before any of its `i32` lanes can overflow.
pub(super) struct WideMass<E: Field + Unreduced> {
    wide: Vec<E::Wide>,
    reduced: Vec<E>,
    pending: Vec<usize>,
    block_len: usize,
    max_pending: usize,
    /// Digits lie in `[-half, half)`.
    half: usize,
    #[cfg(test)]
    force_portable: bool,
}

impl<E: Field + Unreduced> WideMass<E> {
    pub(super) fn new(block_count: usize, block_len: usize, b: usize) -> Self {
        debug_assert!(b.is_power_of_two() && (2..=MAX_DIGIT_BASIS).contains(&b));
        Self {
            wide: vec![E::Wide::zero(); block_count * block_len],
            reduced: vec![E::zero(); block_count * block_len],
            pending: vec![0; block_count],
            block_len,
            // A lane of `E::Wide::from(factor)` is below `2^16`, and a digit's
            // magnitude is at most `b / 2`.
            max_pending: (i32::MAX as usize) / (usize::from(u16::MAX) * (b / 2)),
            half: b / 2,
            #[cfg(test)]
            force_portable: false,
        }
    }

    /// Largest row count one [`add_rows`](Self::add_rows) call accepts.
    #[inline]
    pub(super) fn max_rows(&self) -> usize {
        self.max_pending
    }

    /// Add `factor * digits[row * block_len..][..block_len]` to block `block`
    /// for every `(factor, row)` of `rows`.
    pub(super) fn add_rows(&mut self, block: usize, rows: &[(E, u32)], digits: &[i8])
    where
        E: 'static,
    {
        assert!(rows.len() <= self.max_pending);
        if rows.is_empty() {
            return;
        }
        if self.pending[block] + rows.len() > self.max_pending {
            self.flush(block);
        }
        self.pending[block] += rows.len();
        let block_len = self.block_len;
        let slots = block * block_len..(block + 1) * block_len;
        #[cfg(target_arch = "aarch64")]
        let use_neon = block_len.is_multiple_of(neon::TILE);
        #[cfg(all(test, target_arch = "aarch64"))]
        let use_neon = use_neon && !self.force_portable;
        #[cfg(target_arch = "aarch64")]
        if use_neon {
            if let Some(wide) = (&mut self.wide as &mut dyn std::any::Any)
                .downcast_mut::<Vec<jolt_field::Fp128x8i32>>()
            {
                neon::add_rows::<E>(&mut wide[slots], self.half, rows, digits);
                return;
            }
        }
        let masses = &mut self.wide[slots];
        for &(factor, row) in rows {
            let row = row as usize;
            add_row(
                masses,
                self.half,
                factor,
                &digits[row * block_len..][..block_len],
                #[cfg(test)]
                self.force_portable,
            );
        }
    }

    fn flush(&mut self, block: usize) {
        let slots = block * self.block_len..(block + 1) * self.block_len;
        for (reduced, wide) in self.reduced[slots.clone()]
            .iter_mut()
            .zip(&mut self.wide[slots])
        {
            *reduced += E::reduce_wide(*wide);
            *wide = E::Wide::zero();
        }
        self.pending[block] = 0;
    }

    pub(super) fn finish(mut self) -> Vec<E> {
        for block in 0..self.pending.len() {
            if self.pending[block] != 0 {
                self.flush(block);
            }
        }
        self.reduced
    }
}

/// Add `factor * digits[i]` to `masses[i]`.
#[inline]
fn add_row<E: Field + Unreduced>(
    masses: &mut [E::Wide],
    half: usize,
    factor: E,
    digits: &[i8],
    #[cfg(test)] force_portable: bool,
) {
    let use_scale = cfg!(target_arch = "aarch64");
    #[cfg(test)]
    let use_scale = use_scale && !force_portable;
    if use_scale {
        // NEON multiplies `i32` lanes natively.
        for (mass, &digit) in masses.iter_mut().zip(digits) {
            *mass += factor.scale_wide(i32::from(digit));
        }
        return;
    }
    // Without a vector `i32` multiply (baseline x86-64 has none) every lane
    // would be scaled in scalar, so build the row's digit multiples of
    // `factor` by lane-wise additions and add one per digit.
    let base = E::Wide::from(factor);
    let mut multiples = [E::Wide::zero(); MAX_DIGIT_BASIS];
    for offset in 1..=half {
        if half + offset < 2 * half {
            multiples[half + offset] = multiples[half + offset - 1] + base;
        }
        multiples[half - offset] = multiples[half + 1 - offset] - base;
    }
    for (mass, &digit) in masses.iter_mut().zip(digits) {
        debug_assert!((-(half as i16)..half as i16).contains(&i16::from(digit)));
        let index = (i16::from(digit) + half as i16) as usize & (MAX_DIGIT_BASIS - 1);
        *mass += multiples[index];
    }
}

/// `Fp128x8i32` rows on NEON: each factor splits into eight 16-bit limbs, and
/// a widening `u16` multiply-accumulate by biased digit `digit + half`
/// accumulates a tile of eight slots in registers across every row. The bias
/// is removed once per tile as `half` times the row limb sums.
///
/// Lanes wrap modulo `2^32`. The true signed lane sums fit in `i32` while the
/// block is within `max_pending` rows, so the corrected lanes are exact.
#[cfg(target_arch = "aarch64")]
mod neon {
    use jolt_field::{Field, Fp128x8i32, Unreduced};
    use std::any::Any;
    use std::arch::aarch64::*;

    // Raw tile loads assume contiguous eight-i32 elements with no padding.
    const _: () = {
        assert!(std::mem::size_of::<Fp128x8i32>() == std::mem::size_of::<[i32; 8]>());
        assert!(std::mem::align_of::<Fp128x8i32>() == std::mem::align_of::<[i32; 8]>());
    };

    /// Slots per register tile.
    pub(super) const TILE: usize = 8;

    /// Rows whose limbs are staged at once.
    const ROW_BATCH: usize = 64;

    pub(super) fn add_rows<E: Field + Unreduced + 'static>(
        masses: &mut [Fp128x8i32],
        half: usize,
        rows: &[(E, u32)],
        digits: &[i8],
    ) {
        let row_len = masses.len();
        debug_assert!(row_len.is_multiple_of(TILE));
        debug_assert!(half < 8);
        let mut limbs = [[0u16; 8]; ROW_BATCH];
        let mut offsets = [0usize; ROW_BATCH];
        for batch in rows.chunks(ROW_BATCH) {
            for ((limbs, offset), &(factor, row)) in limbs.iter_mut().zip(&mut offsets).zip(batch) {
                let wide = E::Wide::from(factor);
                let lanes = (&wide as &dyn Any)
                    .downcast_ref::<Fp128x8i32>()
                    .expect("wide type was matched by the caller");
                *limbs = lanes.0.map(|lane| {
                    debug_assert!((0..=i32::from(u16::MAX)).contains(&lane));
                    lane as u16
                });
                *offset = row as usize * row_len;
                // Bounds for the unchecked row loads below.
                assert!(*offset + row_len <= digits.len());
            }
            // SAFETY: every staged row lies inside `digits` (asserted above),
            // and every tile lies inside `masses` because `row_len` is a
            // multiple of `TILE`.
            unsafe {
                add_batch(
                    masses,
                    half as u8,
                    &limbs[..batch.len()],
                    &offsets[..batch.len()],
                    digits,
                );
            }
        }
    }

    /// # Safety
    ///
    /// `masses.len()` is a multiple of [`TILE`], and every
    /// `offsets[k] + masses.len() <= digits.len()`.
    #[inline]
    unsafe fn add_batch(
        masses: &mut [Fp128x8i32],
        half: u8,
        limbs: &[[u16; 8]],
        offsets: &[usize],
        digits: &[i8],
    ) {
        let bias = vdup_n_u8(half);
        let (mut sum_low, mut sum_high) = (vdupq_n_u32(0), vdupq_n_u32(0));
        for limbs in limbs {
            let factor = vld1q_u16(limbs.as_ptr());
            sum_low = vaddw_u16(sum_low, vget_low_u16(factor));
            sum_high = vaddw_high_u16(sum_high, factor);
        }
        let correction_low = vmulq_n_u32(sum_low, u32::from(half));
        let correction_high = vmulq_n_u32(sum_high, u32::from(half));
        for tile in 0..masses.len() / TILE {
            let base = masses.as_mut_ptr().add(TILE * tile).cast::<u32>();
            let mut acc: [[uint32x4_t; 2]; TILE] = std::array::from_fn(|slot| {
                [
                    vld1q_u32(base.add(8 * slot)),
                    vld1q_u32(base.add(8 * slot + 4)),
                ]
            });
            for (limbs, &offset) in limbs.iter().zip(offsets) {
                let factor = vld1q_u16(limbs.as_ptr());
                let low = vget_low_u16(factor);
                let row = digits.as_ptr().add(offset + TILE * tile).cast::<u8>();
                let biased = vmovl_u8(vadd_u8(vld1_u8(row), bias));
                macro_rules! slot {
                    ($slot:literal) => {
                        acc[$slot][0] = vmlal_laneq_u16::<$slot>(acc[$slot][0], low, biased);
                        acc[$slot][1] =
                            vmlal_high_laneq_u16::<$slot>(acc[$slot][1], factor, biased);
                    };
                }
                slot!(0);
                slot!(1);
                slot!(2);
                slot!(3);
                slot!(4);
                slot!(5);
                slot!(6);
                slot!(7);
            }
            for (slot, [low, high]) in acc.into_iter().enumerate() {
                vst1q_u32(base.add(8 * slot), vsubq_u32(low, correction_low));
                vst1q_u32(base.add(8 * slot + 4), vsubq_u32(high, correction_high));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Prime128Offset275, Prime64Offset59};

    /// Rows added in shuffled groups match the direct field sum, across
    /// overflow flushes and with extreme factor limbs and digits.
    fn check<E: Field + Unreduced + 'static>(b: usize, block_len: usize) {
        let half = (b / 2) as i8;
        let row_count = 2 * WideMass::<E>::new(1, block_len, b).max_rows() + 37;
        let digits = (0..row_count * block_len)
            .map(|index| match index % 5 {
                0 | 2 => -half,
                1 => half - 1,
                _ => ((index * 7 + 3) % b) as i8 - half,
            })
            .collect::<Vec<_>>();
        // `-1 - k` has all-ones high limbs.
        let factors = (0..row_count)
            .map(|row| -E::from_u64(row as u64 + 1))
            .collect::<Vec<_>>();
        let order = (0..row_count as u32)
            .map(|row| (row * 7919) % row_count as u32)
            .collect::<Vec<_>>();

        let mut mass = WideMass::<E>::new(2, block_len, b);
        let rows = order
            .iter()
            .map(|&row| (factors[row as usize], row))
            .collect::<Vec<_>>();
        for group in rows.chunks(301) {
            mass.add_rows(1, group, &digits);
        }
        let masses = mass.finish();

        let mut expected = vec![E::zero(); block_len];
        for (row, &factor) in factors.iter().enumerate() {
            for (slot, &digit) in expected
                .iter_mut()
                .zip(&digits[row * block_len..][..block_len])
            {
                *slot += factor * E::from_i64(i64::from(digit));
            }
        }
        assert!(masses[..block_len].iter().all(|mass| mass.is_zero()));
        assert_eq!(&masses[block_len..], &expected[..]);
    }

    #[test]
    fn native_and_portable_match_at_flush_boundary() {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        type E = Prime128Offset275;
        let mut rng = StdRng::seed_from_u64(0x5749_4445);
        for b in [4, 8] {
            for block_count in [1, 3] {
                // Four slots exercise scale_wide on ARM; eight and sixteen
                // exercise the tiled NEON kernel against digit multiples.
                for block_len in [4, 8, 16] {
                    let mut native = WideMass::<E>::new(block_count, block_len, b);
                    let mut portable = WideMass::<E>::new(block_count, block_len, b);
                    portable.force_portable = true;
                    let limit = native.max_rows();
                    let digits = (0..limit * block_len)
                        .map(|_| rng.gen_range(-(b as i8 / 2)..b as i8 / 2))
                        .collect::<Vec<_>>();
                    let rows = (0..limit)
                        .map(|row| (E::random(&mut rng), row as u32))
                        .collect::<Vec<_>>();
                    for block in 0..block_count {
                        for (count, pending) in
                            [(limit - 1, limit - 1), (1, limit), (1, 1), (limit, limit)]
                        {
                            native.add_rows(block, &rows[..count], &digits);
                            portable.add_rows(block, &rows[..count], &digits);
                            assert_eq!(native.pending[block], pending);
                            assert_eq!(portable.pending[block], pending);
                            assert_eq!(native.wide, portable.wide);
                            assert_eq!(native.reduced, portable.reduced);
                        }
                    }
                    assert_eq!(native.finish(), portable.finish());
                }
            }
        }
    }

    #[test]
    fn grouped_rows_match_direct_sum() {
        for block_len in [4, 8, 64] {
            check::<Prime128Offset275>(8, block_len);
            check::<Prime128Offset275>(4, block_len);
            check::<Prime64Offset59>(8, block_len);
        }
    }
}
