//! Sixteen-point decimation-in-frequency transform across blocks of lanes.
//!
//! The slots of a ring transform split into classes that share a power of the
//! root, and within a class the slots are one short transform of that class's
//! sixteen partial sums. Every class runs it with the same twiddles, so point
//! `v` is stored as a block of lanes, one per class, and each butterfly is
//! plain vector arithmetic with a broadcast twiddle: no stage works inside a
//! vector.

use super::prime::NttPrime;
use super::NttKernelPlan;

/// Points of the transform.
pub const POINTS: usize = 16;

/// The transform for one prime and one primitive sixteenth root of unity.
pub struct LaneDif {
    p: u32,
    /// `omega^j` for `j < POINTS / 2` with its Shoup quotient
    /// `floor(omega^j * 2^32 / p)`.
    shoup: [(u32, u32); POINTS / 2],
    /// `omega^j` for `j < POINTS / 2` in the NEON kernels' Barrett form.
    #[cfg(target_arch = "aarch64")]
    barrett: super::neon::BarrettTable<i32, { POINTS / 2 }>,
}

impl LaneDif {
    /// `omega` must be a primitive sixteenth root of unity modulo the prime,
    /// in `[0, p)`, and the prime below `2^30`; otherwise `None`.
    #[must_use]
    pub fn new(prime: NttPrime<i32>, omega: u32) -> Option<Self> {
        let p = u32::try_from(prime.p).ok()?;
        if p >= 1 << 30 || omega >= p {
            return None;
        }
        let mul = |a: u32, b: u32| (u64::from(a) * u64::from(b) % u64::from(p)) as u32;
        let mut powers = [1u32; POINTS / 2];
        for j in 1..POINTS / 2 {
            powers[j] = mul(powers[j - 1], omega);
        }
        // omega^8 = -1 makes omega a primitive sixteenth root.
        if mul(powers[POINTS / 2 - 1], omega) != p - 1 {
            return None;
        }
        let shoup = powers.map(|w| (w, ((u64::from(w) << 32) / u64::from(p)) as u32));
        Some(Self {
            p,
            shoup,
            #[cfg(target_arch = "aarch64")]
            barrett: super::neon::BarrettTable::from_fn(i64::from(p), |j| i64::from(powers[j])),
        })
    }

    /// Transforms the [`POINTS`] blocks of `points`, block `v` holding lane
    /// `l` of point `v` at `v * lanes + l`, into `out` of the same shape:
    /// lane `l` of block `i` becomes a value in `[-p, p)` congruent to the
    /// sum over `v` of `omega^(rev(i) * v)` times lane `l` of point `v`,
    /// where `rev` reverses the four bits of `i`. Inputs must be in
    /// `[0, 2p)`; `points` is left unspecified.
    ///
    /// # Panics
    ///
    /// Panics if `points` is not [`POINTS`] equal blocks or `out` has
    /// another length.
    pub fn transform(&self, points: &mut [u32], out: &mut [i32], plan: NttKernelPlan) {
        assert!(
            points.len().is_multiple_of(POINTS),
            "points are not whole blocks"
        );
        assert_eq!(points.len(), out.len(), "output length");
        let lanes = points.len() / POINTS;
        if lanes == 0 {
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if plan.uses_neon() && lanes.is_multiple_of(4) {
            // SAFETY: the plan proves NEON support; both slices hold sixteen
            // blocks of `lanes` words, a whole number of four-lane vectors,
            // and `new` checked the prime.
            unsafe {
                super::neon::lane_dif16_i32(
                    points.as_ptr().cast(),
                    out.as_mut_ptr(),
                    lanes,
                    self.p as i32,
                    &self.barrett,
                );
            }
            return;
        }
        let _ = plan;
        self.transform_portable(points, out, lanes);
    }

    fn transform_portable(&self, points: &mut [u32], out: &mut [i32], lanes: usize) {
        for half in [8, 4, 2] {
            let stride = POINTS / (2 * half);
            for group in points.chunks_exact_mut(2 * half * lanes) {
                let (low, high) = group.split_at_mut(half * lanes);
                for (j, (a, b)) in low
                    .chunks_exact_mut(lanes)
                    .zip(high.chunks_exact_mut(lanes))
                    .enumerate()
                {
                    self.butterfly(a, b, self.shoup[j * stride]);
                }
            }
        }
        for (pair, out) in points
            .chunks_exact(2 * lanes)
            .zip(out.chunks_exact_mut(2 * lanes))
        {
            let (a, b) = pair.split_at(lanes);
            let (out_a, out_b) = out.split_at_mut(lanes);
            self.last_butterfly(a, b, out_a, out_b);
        }
    }

    // The block helpers stay out of line: inlined, their loops are unrolled
    // before vectorizing and come out scalar.

    /// `(a + b, (a - b) * w)` with outputs in `[0, 2p)`; the unit twiddle
    /// skips the product.
    #[inline(never)]
    fn butterfly(&self, a: &mut [u32], b: &mut [u32], (w, w_shoup): (u32, u32)) {
        let (p, p2) = (self.p, 2 * self.p);
        if w == 1 {
            for (a, b) in a.iter_mut().zip(b) {
                let (x, y) = (*a, *b);
                *a = fold(x + y, p2);
                *b = fold(x + p2 - y, p2);
            }
            return;
        }
        for (a, b) in a.iter_mut().zip(b) {
            let (x, y) = (*a, *b);
            let difference = x + p2 - y;
            *a = fold(x + y, p2);
            let quotient = ((u64::from(difference) * u64::from(w_shoup)) >> 32) as u32;
            *b = difference
                .wrapping_mul(w)
                .wrapping_sub(quotient.wrapping_mul(p));
        }
    }

    /// The last stage, whose twiddles are all one, written in `[0, p)`.
    #[inline(never)]
    fn last_butterfly(&self, a: &[u32], b: &[u32], out_a: &mut [i32], out_b: &mut [i32]) {
        let (p, p2) = (self.p, 2 * self.p);
        for (((&x, &y), out_a), out_b) in a.iter().zip(b).zip(out_a).zip(out_b) {
            *out_a = fold(fold(x + y, p2), p) as i32;
            *out_b = fold(fold(x + p2 - y, p2), p) as i32;
        }
    }
}

/// `value` or `value - bound`, whichever does not wrap.
#[inline(always)]
fn fold(value: u32, bound: u32) -> u32 {
    value.min(value.wrapping_sub(bound))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "test parameters are valid")]

    use super::*;
    use crate::tables::q128_primes;

    fn pow(base: u64, mut exponent: u64, p: u64) -> u64 {
        let (mut result, mut base) = (1, base % p);
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result * base % p;
            }
            base = base * base % p;
            exponent >>= 1;
        }
        result
    }

    #[test]
    fn matches_the_defining_sum() {
        let prime = q128_primes()[0];
        let p = prime.p as u64;
        // A sixteenth root of unity from any element whose power is not +-1.
        let omega = (2..)
            .map(|g| pow(g, (p - 1) / 16, p))
            .find(|&w| pow(w, 8, p) == p - 1)
            .unwrap();
        let dif = LaneDif::new(prime, omega as u32).unwrap();
        assert!(LaneDif::new(prime, pow(omega, 2, p) as u32).is_none());
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        for lanes in [0, 4, 32, 3] {
            for extreme in [false, true] {
                let points: Vec<u32> = (0..POINTS * lanes)
                    .map(|_| {
                        if extreme {
                            2 * p as u32 - 1
                        } else {
                            next(2 * p) as u32
                        }
                    })
                    .collect();
                for plan in [NttKernelPlan::detect::<i32>(), NttKernelPlan::SCALAR] {
                    let mut scratch = points.clone();
                    let mut out = vec![0i32; POINTS * lanes];
                    dif.transform(&mut scratch, &mut out, plan);
                    for i in 0..POINTS {
                        let frequency = (i as u64).reverse_bits() >> 60;
                        for lane in 0..lanes {
                            let expected = (0..POINTS as u64).fold(0, |sum, v| {
                                let point = u64::from(points[v as usize * lanes + lane]);
                                (sum + pow(omega, frequency * v, p) * (point % p)) % p
                            });
                            let actual = i64::from(out[i * lanes + lane]);
                            assert!((-(p as i64)..p as i64).contains(&actual));
                            assert_eq!(actual.rem_euclid(p as i64) as u64, expected);
                        }
                    }
                }
            }
        }
    }
}
