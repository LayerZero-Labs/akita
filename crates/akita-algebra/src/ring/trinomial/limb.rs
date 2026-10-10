//! Exact arithmetic for `Z_p[Y]/(Y^648 - Y^324 + 1)` in signed 32-bit lanes.
//!
//! Centered matrix coefficients enter through [`TrinomialLimbDomain::forward_centered`].
//! The single bit transform, [`TrinomialLimbDomain::forward_interleaved_bits`],
//! maps bit `n` to coefficient `(-1)^floor(n/4)` times that bit. Gather byte `t`
//! contains bit `t+81*j` at position `j`; its sign is negative exactly when
//! `(t+j)&4 != 0`. Four signed table phases and conditional negation implement
//! all eight phases without unpacking a signed coefficient array.
//!
//! NEON runs only on `cfg(all(target_arch = "aarch64", target_feature = "neon"))`.
//! All other targets, including AArch64 without NEON, use the portable backend.
//! `AKITA_SCALAR_NTT=1` also selects the portable backend on NEON targets.
//! Both backends return the same lazy representatives for the same path.
//! Different paths can differ by a multiple of `p`; slots have no structural
//! equality. Compare canonical residues or the inverse-transformed coefficients.
//!
//! # Magnitude bounds
//!
//! Let `h=floor(p/2)` and `C(B)=ceil(p/2+B*p/2^32)`. A rounded-quotient
//! twiddle product of an operand with magnitude at most `B` has magnitude at
//! most `C(B)`. For a lazy forward level define
//! `F(B)=max(B+2*C(B), B+C(B)+C(2*C(B)))`.
//! This bounds all transient additions/subtractions as well as stored outputs.
//! The centered path starts with `B0=h` after eight-term evaluation and uses
//! `B1=F(B0)`, `B2=F(B1)`, `B3=F(B2)`, `B4=F(B3)` at strides 27,9,3,1.
//! Signed table construction sums at most eight centered weights (`8h` in
//! i64); each fused entry is centered to `h`. Its two-entry partial sum is
//! bounded by `2h`, and its three-entry first-level result by `3h<=B1`.
//! The signed path therefore skips stride 27 and retains the bounds B2–B4.
//! Multiplication by one finishes either forward path at `C(B4)<p`.
//!
//! The inverse starts by centering every slot to `h`. At each stride 1,3,9,27,
//! `|b-c|<=2h`, its product is bounded by `C(2h)`, and every intermediate is
//! bounded by `I=max(3h,2h+C(2h))`. A final product is bounded by `C(I)<p`;
//! centering restores `h` at every inverse level. Initial evaluation and final
//! interpolation sum eight centered products, bounded by `8h^2` in i64;
//! their stored output is centered to `h`.
//!
//! Accumulation checkpoints every 128 products. After `k<=128` terms since a
//! checkpoint its i64 magnitude is at most `h+k*(p-1)^2`; the carry is at most
//! `h`. The checkpoint and [`TrinomialLimbAccumulator::finish`] both center.
//! This invariant permits an arbitrary number of products.
//!
//! Evaluated at the largest admitted prime, `p=268433353`:
//!
//! | Retained stage | Formula | Maximum magnitude | Limit |
//! |---|---|---:|---:|
//! | Centered input/evaluation, inverse stored output | h | 134216676 | i32::MAX=2147483647 |
//! | Signed table construction sum | 8h | 1073733408 | i64::MAX=9223372036854775807 |
//! | Signed fused entry / partial / result | h / 2h / 3h | 134216676 / 268433352 / 402650028 | i32::MAX |
//! | Centered forward stride 27 | B1 | 428864012 | i32::MAX |
//! | Either forward stride 9 | B2 | 750904948 | i32::MAX |
//! | Either forward stride 3 | B3 | 1113200686 | i32::MAX |
//! | Either forward stride 1 | B4 | 1520783036 | i32::MAX |
//! | Final slots | C(B4) | 229264872 | p=268433353 (strict) |
//! | Inverse difference product, each level | C(2h) | 150993630 | i32::MAX |
//! | Inverse transient, each level | I | 419426982 | i32::MAX |
//! | Inverse product before centering | C(I) | 160430658 | p (strict) |
//! | Evaluation/interpolation dot | 8h^2 | 144112928931911808 | 2^57=144115188075855872 (strict) |
//! | Accumulator, full checkpoint batch | h+128(p-1)^2 | 9223227451776572388 | i64::MAX |
//!
//! Every formula above is nondecreasing in `p`, so each admitted prime obeys
//! the inequalities stated at the largest one. Admission rechecks B4;
//! checked shadows exercise every retained forward/inverse level, table phase,
//! and accumulator checkpoint using extremal signs for every admitted prime.

use super::TrinomialError;
use std::sync::OnceLock;

pub(super) mod arithmetic;
mod gather;
mod interleaved;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(test)]
mod tests;

use arithmetic::{Arithmetic, Split, Twiddle};

const DEGREE: usize = 648;
const LANES: usize = 8;
const PIECE: usize = 81;
const EXPONENTS: [u32; LANES] = [1, 5, 7, 11, 13, 17, 19, 23];

fn use_neon() -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        crate::ntt::neon::use_neon_ntt()
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        false
    }
}

#[cfg(test)]
fn canonical_slots(slots: &TrinomialLimbSlots) -> [i32; DEGREE] {
    slots
        .values
        .map(|v| Arithmetic { prime: slots.prime }.center(i64::from(v)))
}

/// One admitted prime below `2^28` and its degree-648 transform plan.
#[derive(Clone, Debug)]
pub struct TrinomialLimbDomain {
    arithmetic: Arithmetic,
    interleaved_first_level: OnceLock<Box<[[[i32; 3 * LANES]; 256]]>>,
    evaluation: [[i32; LANES]; LANES],
    interpolation: [[i32; LANES]; LANES],
    splits: Vec<Split>,
    omega: Twiddle,
    inverse_omega: Twiddle,
    one: Twiddle,
}

/// 648 plain signed residues, with magnitude strictly below the tagged prime.
///
/// Values are lazy representatives; algebraically equal slots can differ by
/// the prime. Use the inverse transform to compare polynomials.
#[derive(Clone, Debug)]
pub struct TrinomialLimbSlots {
    values: [i32; DEGREE],
    prime: u32,
}

/// Lazily accumulated plain slot products in signed 64-bit lanes.
///
/// A checkpoint after 128 products permits arbitrarily long columns:
/// `128*(p-1)^2 + floor(p/2) < i64::MAX` for every `p < 2^28`.
#[derive(Clone, Debug)]
pub struct TrinomialLimbAccumulator {
    arithmetic: Arithmetic,
    sums: [i64; DEGREE],
    terms: u16,
}

impl TrinomialLimbDomain {
    /// The LaBinius commitment prime `33_568_993 = 1 + 23328 * 1439`, the smallest
    /// prime above `2^25` congruent to 1 modulo `23328 = lcm(1944, 3888, 5832, 7776)`,
    /// then the largest prime below each of `2^26`, `2^27`, `2^28` congruent to 1
    /// modulo 1944.
    pub const ADMITTED_PRIMES: [u32; 4] = [33_568_993, 67_091_329, 134_217_649, 268_433_353];

    /// Build the transform plan; reject primes outside [`Self::ADMITTED_PRIMES`].
    pub fn new(prime: u32) -> Result<Self, TrinomialError> {
        if !Self::ADMITTED_PRIMES.contains(&prime) {
            return Err(TrinomialError::LimbInput {
                reason: "prime is not an admitted limb prime",
            });
        }
        let arithmetic = Arithmetic { prime };
        let root = (2..prime)
            .map(|candidate| arithmetic.pow(candidate, (prime - 1) / 1944))
            .find(|&root| arithmetic.pow(root, 972) != 1 && arithmetic.pow(root, 648) != 1)
            .ok_or(TrinomialError::LimbInput {
                reason: "prime does not supply a primitive 1944th root",
            })?;
        let twiddle = |exponent| {
            Twiddle::new(
                arithmetic,
                arithmetic.center(i64::from(arithmetic.pow(root, exponent))),
            )
        };
        let omega = twiddle(648);
        let inverse_omega = twiddle(1296);
        let evaluation = std::array::from_fn(|lane| {
            std::array::from_fn(|j| twiddle(81 * EXPONENTS[lane] * j as u32).value)
        });
        let interpolation = arithmetic::invert_evaluation(arithmetic, &evaluation)?;
        let mut splits = Vec::with_capacity(40);
        let mut exponents = vec![EXPONENTS.map(|e| 81 * e)];
        let mut len = PIECE;
        while len > 1 {
            let stride = len / 3;
            let mut children = Vec::with_capacity(exponents.len() * 3);
            for (group, exponents) in exponents.iter().enumerate() {
                let roots = exponents.map(|e| e / 3);
                splits.push(Split {
                    start: group * len,
                    stride,
                    r: roots.map(twiddle),
                    r2: roots.map(|e| twiddle(2 * e)),
                    ri: roots.map(|e| twiddle((1944 - e % 1944) % 1944)),
                    ri2: roots.map(|e| twiddle((1944 - 2 * e % 1944) % 1944)),
                });
                for branch in 0..3 {
                    children.push(roots.map(|e| e + 648 * branch));
                }
            }
            exponents = children;
            len = stride;
        }
        // These recurrence bounds include every add/subtract, not only output
        // values. At p<2^28 the final B4 is <5.67p<2^31; no butterfly wraps.
        // B0=floor(p/2); C(B)=ceil(p/2+Bp/2^32);
        // Bnext=max(B+2C(B), B+C(B)+C(2C(B))).
        let bounds = arithmetic::forward_bounds(prime);
        if bounds[4] >= i64::from(i32::MAX) {
            return Err(TrinomialError::LimbInput {
                reason: "limb lazy bound exceeds signed lanes",
            });
        }
        Ok(Self {
            arithmetic,
            interleaved_first_level: OnceLock::new(),
            evaluation,
            interpolation,
            splits,
            omega,
            inverse_omega,
            one: Twiddle::new(arithmetic, 1),
        })
    }

    /// The modulus of this domain.
    pub fn prime(&self) -> u32 {
        self.arithmetic.prime
    }

    /// Create zero slots with this domain's prime tag.
    pub fn zero_slots(&self) -> TrinomialLimbSlots {
        TrinomialLimbSlots {
            values: [0; DEGREE],
            prime: self.prime(),
        }
    }

    fn check_bits(bits: &[u64]) -> Result<(), TrinomialError> {
        if bits.len() != 11 {
            return Err(TrinomialError::CoefficientLength {
                expected: 11,
                actual: bits.len(),
            });
        }
        if bits[10] >> 8 != 0 {
            return Err(TrinomialError::LimbInput {
                reason: "unused tail bits must be zero",
            });
        }
        Ok(())
    }

    fn check_slots(&self, slots: &TrinomialLimbSlots) -> Result<(), TrinomialError> {
        if slots.prime != self.prime() {
            return Err(TrinomialError::LimbInput {
                reason: "slot prime does not match the domain",
            });
        }
        Ok(())
    }

    /// Transform 648 centered coefficients in `[-(p-1)/2,(p-1)/2]`.
    ///
    /// Reject wrong length, noncentered coefficients, or a mismatched prime.
    pub fn forward_centered(
        &self,
        coefficients: &[i32],
        out: &mut TrinomialLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.forward_centered_backend(coefficients, out, use_neon())
    }

    fn forward_centered_backend(
        &self,
        coefficients: &[i32],
        out: &mut TrinomialLimbSlots,
        neon: bool,
    ) -> Result<(), TrinomialError> {
        if coefficients.len() != DEGREE {
            return Err(TrinomialError::CoefficientLength {
                expected: DEGREE,
                actual: coefficients.len(),
            });
        }
        let half = (self.prime() / 2) as i32;
        if coefficients.iter().any(|&v| v < -half || v > half) {
            return Err(TrinomialError::LimbInput {
                reason: "coefficient must be centered below the limb prime",
            });
        }
        self.check_slots(out)?;
        for t in 0..PIECE {
            let input: [i32; LANES] = std::array::from_fn(|j| coefficients[t + PIECE * j]);
            for (lane, weights) in self.evaluation.iter().enumerate() {
                out.values[t * LANES + lane] = self.arithmetic.dot(&input, weights);
            }
        }
        self.transform(&mut out.values, false, neon, false);
        Ok(())
    }

    /// Invert to 648 centered coefficients, rejecting wrong length or prime.
    pub fn inverse_centered(
        &self,
        slots: &TrinomialLimbSlots,
        out: &mut [i32],
    ) -> Result<(), TrinomialError> {
        self.inverse_centered_backend(slots, out, use_neon())
    }

    fn inverse_centered_backend(
        &self,
        slots: &TrinomialLimbSlots,
        out: &mut [i32],
        neon: bool,
    ) -> Result<(), TrinomialError> {
        if out.len() != DEGREE {
            return Err(TrinomialError::CoefficientLength {
                expected: DEGREE,
                actual: out.len(),
            });
        }
        self.check_slots(slots)?;
        let mut values = slots.values.map(|v| self.arithmetic.center(i64::from(v)));
        self.transform(&mut values, true, neon, false);
        for (t, row) in values.chunks_exact(LANES).enumerate() {
            for (j, weights) in self.interpolation.iter().enumerate() {
                out[t + PIECE * j] = self.arithmetic.dot(row, weights);
            }
        }
        Ok(())
    }

    fn transform(&self, values: &mut [i32; DEGREE], inverse: bool, neon: bool, skip_first: bool) {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if neon {
            return neon::transform(self, values, inverse, skip_first);
        }
        let _ = neon;
        if inverse {
            for split in self.splits.iter().rev() {
                arithmetic::butterfly(
                    self.arithmetic,
                    self.omega,
                    self.inverse_omega,
                    values,
                    split,
                    true,
                );
            }
        } else {
            for split in &self.splits[usize::from(skip_first)..] {
                arithmetic::butterfly(
                    self.arithmetic,
                    self.omega,
                    self.inverse_omega,
                    values,
                    split,
                    false,
                );
            }
            // C(B4)<p: normalization gives the accumulator's strict |slot|<p bound.
            for value in values {
                *value = self.arithmetic.mul(*value, self.one);
            }
        }
    }
}

impl TrinomialLimbAccumulator {
    pub(super) const REDUCTION_TERMS: u16 = 128;

    /// Create an empty accumulator tagged with this domain's prime.
    pub fn new(domain: &TrinomialLimbDomain) -> Self {
        Self {
            arithmetic: domain.arithmetic,
            sums: [0; DEGREE],
            terms: 0,
        }
    }

    /// Add one product; reject mismatched primes before changing the sum.
    pub fn add_product(
        &mut self,
        lhs: &TrinomialLimbSlots,
        rhs: &TrinomialLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.add_product_backend(lhs, rhs, use_neon())
    }

    fn add_product_backend(
        &mut self,
        lhs: &TrinomialLimbSlots,
        rhs: &TrinomialLimbSlots,
        neon: bool,
    ) -> Result<(), TrinomialError> {
        if lhs.prime != self.arithmetic.prime || rhs.prime != self.arithmetic.prime {
            return Err(TrinomialError::LimbInput {
                reason: "product prime does not match the accumulator",
            });
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        if neon {
            neon::accumulate(&mut self.sums, &lhs.values, &rhs.values);
        } else {
            arithmetic::accumulate(&mut self.sums, &lhs.values, &rhs.values);
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            let _ = neon;
            arithmetic::accumulate(&mut self.sums, &lhs.values, &rhs.values);
        }
        self.advance();
        Ok(())
    }

    fn advance(&mut self) {
        self.terms += 1;
        if self.terms == Self::REDUCTION_TERMS {
            for sum in &mut self.sums {
                *sum = i64::from(self.arithmetic.center(*sum));
            }
            self.terms = 0;
        }
    }

    /// Write the centered sum and clear it, rejecting a mismatched prime.
    pub fn finish(&mut self, out: &mut TrinomialLimbSlots) -> Result<(), TrinomialError> {
        if out.prime != self.arithmetic.prime {
            return Err(TrinomialError::LimbInput {
                reason: "output prime does not match the accumulator",
            });
        }
        for (value, &sum) in out.values.iter_mut().zip(&self.sums) {
            *value = self.arithmetic.center(sum);
        }
        self.sums.fill(0);
        self.terms = 0;
        Ok(())
    }
}
