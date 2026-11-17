//! One-pass dot products of a batch of key positions against several columns.
//!
//! A commitment to many columns under one key multiplies the same key limbs
//! by every column's transformed input. Walking one limb and one column at a
//! time reads each input once per limb and each key limb once per column, and
//! reduces every short batch of products. Here the operands of [`SLOTS`]
//! consecutive key positions are laid out so that one sequential pass serves
//! every limb of every column in a tile:
//!
//! - a key batch is `[lane group][limb][slot][lane]`, read once front to back,
//! - a column's input is one ring per slot, as its transform writes it,
//! - a column's sums are `[lane group][limb][lane]`, one `i64` each,
//!
//! with [`LANES`] coefficients per lane group. Each slot of an input is
//! followed by [`INPUT_PAD`] unused words: a pass reads every slot of every
//! column of its tile at one offset, and rings a power of two apart would
//! put those reads in a handful of cache sets. A pass adds to every sum an
//! integer congruent to that limb's dot product over the batch, with no
//! Montgomery reduction; [`reduce`] turns a limb's sums into residues after
//! at most [`max_passes`] passes.

use super::prime::{MontCoeff, NttPrime};
use super::NttKernelPlan;

/// Key positions per batch. With key coefficients centered in `[-p/2, p/2]`
/// and input coefficients in `[-p, p]`, this many products sum below `2^63`
/// for every prime below `2^30`.
pub const SLOTS: usize = 12;

/// Coefficients per lane group.
pub const LANES: usize = 4;

/// Words of one key batch of `limbs` limbs of degree `d`.
#[must_use]
pub const fn key_len(limbs: usize, d: usize) -> usize {
    d * limbs * SLOTS
}

/// Unused words after each slot of a column's input: one 128-byte cache
/// line, which puts the slots of a four-column tile of degree-512 inputs
/// laid back to back in distinct cache sets.
pub const INPUT_PAD: usize = 32;

/// Words from one slot of a column's input to the next.
#[must_use]
pub const fn input_stride(d: usize) -> usize {
    d + INPUT_PAD
}

/// Words of one column's input for a batch.
#[must_use]
pub const fn input_len(d: usize) -> usize {
    input_stride(d) * SLOTS
}

/// Sums one column keeps for `limbs` limbs.
#[must_use]
pub const fn sums_len(limbs: usize, d: usize) -> usize {
    d * limbs
}

/// Whether [`accumulate`] and [`reduce`] run vector kernels under `plan`;
/// otherwise they run the portable loops.
#[must_use]
pub const fn is_vectorized(plan: NttKernelPlan) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        plan.uses_neon()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = plan;
        false
    }
}

/// `2^32 mod p`, the weight of a sum's high word.
fn high_word_weight(prime: NttPrime<i32>) -> i64 {
    (1_i64 << 32) % i64::from(prime.p)
}

/// Passes one sum takes before [`reduce`] must run: each adds less than
/// `2^31 * (2^32 mod p) + 2^32` in magnitude, and [`reduce`] accepts sums
/// below `2^61`.
#[must_use]
pub fn max_passes(prime: NttPrime<i32>) -> usize {
    let per_pass = (1_i64 << 31) * high_word_weight(prime) + (1_i64 << 32);
    ((1_i64 << 61) / per_pass) as usize
}

/// Writes `ring` as limb `limb` of position `slot` of a key batch, every
/// coefficient replaced by the representative of its residue in
/// `[-p/2, p/2]`. Coefficients must be in `(-2p, 2p)`.
///
/// # Panics
///
/// Panics if the batch is not [`key_len`] words or an index is out of range.
pub fn write_key<const D: usize>(
    batch: &mut [i32],
    limbs: usize,
    limb: usize,
    slot: usize,
    ring: &[MontCoeff<i32>; D],
    prime: NttPrime<i32>,
) {
    assert_eq!(batch.len(), key_len(limbs, D), "key batch length");
    assert!(limb < limbs && slot < SLOTS, "key batch index");
    for (group, lanes) in ring.as_chunks::<LANES>().0.iter().enumerate() {
        let start = ((group * limbs + limb) * SLOTS + slot) * LANES;
        for (word, coefficient) in batch[start..start + LANES].iter_mut().zip(lanes) {
            *word = prime.center(prime.reduce_range(*coefficient).raw());
        }
    }
}

/// Writes `ring` as position `slot` of a column's input. Coefficients must
/// be in `[-p, p]`.
///
/// # Panics
///
/// Panics if the input is not [`input_len`] words or `slot` is out of range.
pub fn write_input<const D: usize>(input: &mut [i32], slot: usize, ring: &[MontCoeff<i32>; D]) {
    for (word, coefficient) in input_slot_mut(input, slot, D).iter_mut().zip(ring) {
        *word = coefficient.raw();
    }
}

/// Position `slot` of a column's input of degree `d`, for a transform to
/// write in place. Values must be in `[-p, p]`.
///
/// # Panics
///
/// Panics if the input is not [`input_len`] words or `slot` is out of range.
pub fn input_slot_mut(input: &mut [i32], slot: usize, d: usize) -> &mut [i32] {
    assert_eq!(input.len(), input_len(d), "column input length");
    assert!(slot < SLOTS, "column input slot");
    &mut input[slot * input_stride(d)..][..d]
}

/// Copies position `from` of a column's input of degree `d` to position `to`.
///
/// # Panics
///
/// Panics if the input is not [`input_len`] words or a slot is out of range.
pub fn copy_input(input: &mut [i32], from: usize, to: usize, d: usize) {
    assert_eq!(input.len(), input_len(d), "column input length");
    assert!(from < SLOTS && to < SLOTS, "column input slot");
    let source = from * input_stride(d);
    input.copy_within(source..source + d, to * input_stride(d));
}

/// Zeroes position `slot` of a column's input of degree `d`.
///
/// # Panics
///
/// Panics if the input is not [`input_len`] words or `slot` is out of range.
pub fn clear_input(input: &mut [i32], slot: usize, d: usize) {
    input_slot_mut(input, slot, d).fill(0);
}

/// Adds one key batch's dot products to every column of a tile: for each
/// column `t`, limb and coefficient, `sums[t]` gains an integer congruent
/// modulo `p` to the sum over slots of `key * inputs[t]`.
///
/// `inputs` holds the columns' inputs back to back. The key must have been
/// written by [`write_key`], every slot of the inputs written or cleared, and
/// no sum may take more than [`max_passes`] passes between reductions.
///
/// # Panics
///
/// Panics if a length disagrees with `limbs`, `d` and the number of columns,
/// or `d` is not whole lane groups.
pub fn accumulate(
    sums: &mut [&mut [i64]],
    key: &[i32],
    inputs: &[i32],
    limbs: usize,
    d: usize,
    prime: NttPrime<i32>,
    plan: NttKernelPlan,
) {
    assert!(d.is_multiple_of(LANES), "degree is not whole lane groups");
    assert_eq!(key.len(), key_len(limbs, d), "key batch length");
    assert_eq!(inputs.len(), sums.len() * input_len(d), "tile input length");
    assert!(
        sums.iter().all(|sums| sums.len() == sums_len(limbs, d)),
        "column sums length"
    );
    let weight = high_word_weight(prime);

    #[cfg(target_arch = "aarch64")]
    if plan.uses_neon() {
        // The kernel keeps a tile's pointers on the stack.
        const MAX_TILE: usize = 16;
        for (sums, inputs) in sums
            .chunks_mut(MAX_TILE)
            .zip(inputs.chunks(MAX_TILE * input_len(d)))
        {
            let mut sum_pointers = [core::ptr::null_mut::<i64>(); MAX_TILE];
            let mut input_pointers = [core::ptr::null::<i32>(); MAX_TILE];
            for ((sum_pointer, input_pointer), (sums, input)) in sum_pointers
                .iter_mut()
                .zip(&mut input_pointers)
                .zip(sums.iter_mut().zip(inputs.chunks_exact(input_len(d))))
            {
                *sum_pointer = sums.as_mut_ptr();
                *input_pointer = input.as_ptr();
            }
            // SAFETY: the plan proves NEON support. The lengths asserted above
            // give the key `d / LANES * limbs * SLOTS` vectors, every column
            // `SLOTS` rings of `d / LANES` vectors `input_stride(d)` words
            // apart and `d / LANES * limbs` sum vectors; the sums are distinct
            // exclusive borrows.
            unsafe {
                super::neon::tile_dot_acc_i32(
                    &sum_pointers[..sums.len()],
                    key.as_ptr(),
                    &input_pointers[..sums.len()],
                    input_stride(d),
                    limbs,
                    d / LANES,
                    weight as i32,
                );
            }
        }
        return;
    }
    let _ = plan;

    for (sums, input) in sums.iter_mut().zip(inputs.chunks_exact(input_len(d))) {
        for (group, (key, sums)) in key
            .chunks_exact(limbs * SLOTS * LANES)
            .zip(sums.chunks_exact_mut(limbs * LANES))
            .enumerate()
        {
            for (key, sums) in key
                .chunks_exact(SLOTS * LANES)
                .zip(sums.chunks_exact_mut(LANES))
            {
                for (lane, sum) in sums.iter_mut().enumerate() {
                    let products: i64 = (0..SLOTS)
                        .map(|slot| {
                            i64::from(key[slot * LANES + lane])
                                * i64::from(input[slot * input_stride(d) + group * LANES + lane])
                        })
                        .sum();
                    *sum += (products >> 32) * weight + i64::from(products as u32);
                }
            }
        }
    }
}

/// Writes limb `limb` of a column's sums as Montgomery residues in `(-p, p)`:
/// coefficient `i` of `out` is the Montgomery product the accumulated dot
/// products stand for.
///
/// # Panics
///
/// Panics if `sums` is not [`sums_len`] words or `limb` is out of range.
pub fn reduce<const D: usize>(
    sums: &[i64],
    limbs: usize,
    limb: usize,
    out: &mut [MontCoeff<i32>; D],
    prime: NttPrime<i32>,
    plan: NttKernelPlan,
) {
    assert!(D.is_multiple_of(LANES), "degree is not whole lane groups");
    assert_eq!(sums.len(), sums_len(limbs, D), "column sums length");
    assert!(limb < limbs, "limb index");

    #[cfg(target_arch = "aarch64")]
    if plan.uses_neon() {
        // SAFETY: the plan proves NEON support; `sums` holds `D / LANES`
        // groups of `limbs` four-lane vectors and `out` holds `D` words.
        unsafe {
            super::neon::tile_dot_reduce_i32(
                sums.as_ptr().add(limb * LANES),
                limbs * LANES,
                out.as_mut_ptr().cast::<i32>(),
                D / LANES,
                prime.p,
                prime.pinv,
            );
        }
        return;
    }
    let _ = plan;

    let one = MontCoeff::from_raw(1);
    for (out, sums) in out
        .as_chunks_mut::<LANES>()
        .0
        .iter_mut()
        .zip(sums.chunks_exact(limbs * LANES))
    {
        for (out, &sum) in out.iter_mut().zip(&sums[limb * LANES..(limb + 1) * LANES]) {
            // sum = high * 2^32 + low and low = signed_low + 2^32 * carry, so
            // sum / 2^32 is high + carry + signed_low / 2^32 modulo p.
            let low = sum as u32;
            let residue = (sum >> 32)
                + i64::from(low >> 31)
                + i64::from(prime.mul(MontCoeff::from_raw(low as i32), one).raw());
            *out = MontCoeff::from_raw((residue % i64::from(prime.p)) as i32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::tables::Q128_RAW_PRIMES;

    const D: usize = 32;
    const LIMBS: usize = 3;

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, bound: i32) -> i32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % bound as u64) as i32
        }
    }

    /// Runs `passes` passes of a three-column tile and checks every reduced
    /// sum against per-product Montgomery arithmetic.
    fn check(plan: NttKernelPlan, raw_prime: i32, passes: usize, extreme: bool) {
        let prime = NttPrime::new(raw_prime);
        let p = prime.p;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ raw_prime as u64);
        let columns = 3;
        let mut sums = vec![vec![0_i64; sums_len(LIMBS, D)]; columns];
        let mut expected = vec![[[0_i64; D]; LIMBS]; columns];
        for pass in 0..passes {
            let mut key = vec![0; key_len(LIMBS, D)];
            let mut key_rings = vec![[[MontCoeff::from_raw(0); D]; SLOTS]; LIMBS];
            for (limb, rings) in key_rings.iter_mut().enumerate() {
                for (slot, ring) in rings.iter_mut().enumerate() {
                    for (index, coefficient) in ring.iter_mut().enumerate() {
                        // The largest products all share one sign in lanes 0
                        // and 1 of the extreme case.
                        let value = if extreme && index < 2 {
                            p / 2
                        } else {
                            rng.below(2 * p - 1) - (p - 1)
                        };
                        *coefficient = MontCoeff::from_raw(value);
                    }
                    write_key(&mut key, LIMBS, limb, slot, ring, prime);
                }
            }
            let mut inputs = vec![0; columns * input_len(D)];
            for (column, input) in inputs.chunks_exact_mut(input_len(D)).enumerate() {
                for slot in 0..SLOTS {
                    let ring: [MontCoeff<i32>; D] = std::array::from_fn(|index| {
                        MontCoeff::from_raw(match (extreme, index) {
                            (true, 0) => p - 1,
                            (true, 1) => 1 - p,
                            _ => rng.below(2 * p - 1) - (p - 1),
                        })
                    });
                    if slot == 3 && pass % 2 == 1 {
                        copy_input(input, 2, 3, D);
                    } else if slot == 5 && pass % 3 == 2 {
                        clear_input(input, 5, D);
                    } else if slot % 2 == 0 {
                        write_input(input, slot, &ring);
                    } else {
                        let raw = ring.map(MontCoeff::raw);
                        input_slot_mut(input, slot, D).copy_from_slice(&raw);
                    }
                    for (limb, rings) in key_rings.iter().enumerate() {
                        for index in 0..D {
                            let word = input[slot * input_stride(D) + index];
                            let product = prime.mul(rings[slot][index], MontCoeff::from_raw(word));
                            expected[column][limb][index] += i64::from(product.raw());
                        }
                    }
                }
            }
            let mut borrowed: Vec<&mut [i64]> = sums.iter_mut().map(Vec::as_mut_slice).collect();
            accumulate(&mut borrowed, &key, &inputs, LIMBS, D, prime, plan);
        }
        for (column, (sums, expected)) in sums.iter().zip(&expected).enumerate() {
            for (limb, expected) in expected.iter().enumerate() {
                let mut out = [MontCoeff::from_raw(0); D];
                reduce(sums, LIMBS, limb, &mut out, prime, plan);
                for (index, (value, expected)) in out.iter().zip(expected).enumerate() {
                    assert!(
                        value.raw().abs() < p,
                        "prime {raw_prime} column {column} limb {limb} index {index}: out of range"
                    );
                    assert_eq!(
                        (i64::from(value.raw()) - expected).rem_euclid(i64::from(p)),
                        0,
                        "prime {raw_prime} column {column} limb {limb} index {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn passes_match_per_product_montgomery_arithmetic() {
        for raw_prime in Q128_RAW_PRIMES {
            for plan in [NttKernelPlan::SCALAR, NttKernelPlan::detect::<i32>()] {
                check(plan, raw_prime, 1, false);
                check(plan, raw_prime, 5, false);
                check(plan, raw_prime, 64, true);
            }
        }
    }

    #[test]
    fn every_supported_prime_admits_a_reconstruction_span() {
        for raw_prime in Q128_RAW_PRIMES {
            let prime = NttPrime::new(raw_prime);
            let modulus = i128::from(raw_prime);
            // Twelve products of a centered and a reduced operand fit i64.
            assert!(SLOTS as i128 * (modulus / 2) * (modulus - 1) < 1_i128 << 63);
            assert!(max_passes(prime) >= 64, "prime {raw_prime}");
        }
    }
}
