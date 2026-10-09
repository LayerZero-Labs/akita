//! Table transform for interleaved binary words with alternating word-bit signs.

use super::{gather, use_neon, TrinomialError, TrinomialWideLimbDomain, TrinomialWideLimbSlots};
use super::{DEGREE, LANES, PIECE};

#[cfg(test)]
mod tests;

impl TrinomialWideLimbDomain {
    /// Transform coefficient-order bits with coefficient `n` negated when
    /// `floor(n / 4)` is odd.
    ///
    /// Bits are little-endian within and across eleven words, as for
    /// [`Self::forward_bits`]. This represents four interleaved binary words:
    /// coefficient `4*s+c` equals `(-1)^s` times bit `s` of word `c`.
    /// The signs are folded into lazily initialized fused tables; no signed
    /// coefficient array is unpacked. Subsequent calls allocate nothing.
    ///
    /// Reject wrong length, nonzero tail bits, or a mismatched output prime
    /// before initializing tables or changing the output.
    pub fn forward_interleaved_bits(
        &self,
        bits: &[u64],
        out: &mut TrinomialWideLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.forward_interleaved_bits_backend(bits, out, use_neon())
    }

    /// Heap bytes occupied by the bit transform lookup tables.
    ///
    /// Includes the unsigned tables and, after the first interleaved transform,
    /// its signed tables. Excludes the domain struct and the split-plan vector.
    pub fn table_storage_bytes(&self) -> usize {
        std::mem::size_of_val(self.table.as_ref())
            + std::mem::size_of_val(self.first_level.as_ref())
            + self
                .interleaved_first_level
                .get()
                .map_or(0, |tables| std::mem::size_of_val(tables.as_ref()))
    }

    fn forward_interleaved_bits_backend(
        &self,
        bits: &[u64],
        out: &mut TrinomialWideLimbSlots,
        neon: bool,
    ) -> Result<(), TrinomialError> {
        Self::check_bits(bits)?;
        self.check_slots(out)?;
        let mut indices = [0; PIECE];
        gather::indices(bits, &mut indices, true);
        let tables = self.interleaved_first_level.get_or_init(|| {
            // Four sign phases for each of the three first-level sources.
            // A phase offset of four negates every entry. Allocate in blocks
            // of 24 KiB rather than constructing a 288 KiB stack temporary.
            let mut tables = vec![[[0; 3 * LANES]; 256]; 4 * 3].into_boxed_slice();
            for phase in 0..4 {
                // At gather position t+81*j, sign is determined by bit two
                // of t+j, because 81 == 1 (mod 8).
                let mask =
                    (0..LANES).fold(0, |mask, j| mask | (usize::from((phase + j) & 4 != 0) << j));
                for source in 0..3 {
                    for byte in 0..256 {
                        for lane in 0..3 * LANES {
                            // XOR inserts the negative baseline coefficients;
                            // subtracting that baseline restores zero bits.
                            let difference = self.first_level[source][byte ^ mask][lane]
                                - self.first_level[source][mask][lane];
                            tables[phase * 3 + source][byte][lane] =
                                self.arithmetic.center(i64::from(difference));
                        }
                    }
                }
            }
            tables
        });
        #[cfg(target_arch = "aarch64")]
        if neon {
            table_neon(tables, &indices, &mut out.values);
        } else {
            table_scalar(tables, &indices, &mut out.values);
        }
        #[cfg(not(target_arch = "aarch64"))]
        table_scalar(tables, &indices, &mut out.values);
        self.transform(&mut out.values, false, neon, true);
        Ok(())
    }
}

fn table_scalar(
    tables: &[[[i32; 3 * LANES]; 256]],
    indices: &[u8; PIECE],
    values: &mut [i32; DEGREE],
) {
    // Entries are centered; the signed sum is bounded by 3*floor(p/2),
    // exactly the same bound as the existing unsigned fused first level.
    for t in 0..27 {
        let rows: [_; 3] = std::array::from_fn(|source| {
            let phase = (t + source * 27) & 7;
            (
                &tables[(phase & 3) * 3 + source][usize::from(indices[t + source * 27])],
                if phase & 4 == 0 { 1 } else { -1 },
            )
        });
        for branch in 0..3 {
            for lane in 0..LANES {
                let i = branch * LANES + lane;
                values[(t + branch * 27) * LANES + lane] =
                    rows[0].0[i] * rows[0].1 + rows[1].0[i] * rows[1].1 + rows[2].0[i] * rows[2].1;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn table_neon(
    tables: &[[[i32; 3 * LANES]; 256]],
    indices: &[u8; PIECE],
    values: &mut [i32; DEGREE],
) {
    use std::arch::aarch64::*;
    // SAFETY: AArch64 guarantees NEON. Each indexed table has 256 complete
    // 24-lane rows and each load reads four lanes within that row. Stores
    // cover disjoint four-lane halves of the output. Centered table entries
    // and their signed sum are bounded by 3*floor(p/2), fitting i32.
    unsafe {
        for t in 0..27 {
            let rows: [_; 3] = std::array::from_fn(|source| {
                let phase = (t + source * 27) & 7;
                (
                    tables[(phase & 3) * 3 + source][usize::from(indices[t + source * 27])]
                        .as_ptr(),
                    vdupq_n_s32(if phase & 4 == 0 { 0 } else { -1 }),
                )
            });
            for branch in 0..3 {
                for lane in [0, 4] {
                    let index = branch * LANES + lane;
                    // Two's-complement conditional negation: (x XOR mask)-mask.
                    let a = vsubq_s32(
                        veorq_s32(vld1q_s32(rows[0].0.add(index)), rows[0].1),
                        rows[0].1,
                    );
                    let b = vsubq_s32(
                        veorq_s32(vld1q_s32(rows[1].0.add(index)), rows[1].1),
                        rows[1].1,
                    );
                    let c = vsubq_s32(
                        veorq_s32(vld1q_s32(rows[2].0.add(index)), rows[2].1),
                        rows[2].1,
                    );
                    vst1q_s32(
                        values.as_mut_ptr().add((t + branch * 27) * LANES + lane),
                        vaddq_s32(vaddq_s32(a, b), c),
                    );
                }
            }
        }
    }
}
