//! Plain signed 32-bit residues for one prime below 2^28.
//!
//! Four radix-three levels are fully lazy. Twiddle products use a rounded
//! reciprocal and the signed rounded high multiply, without Montgomery form.
//! The final multiplication by one bounds every slot strictly below p.

use super::{use_neon, TrinomialError, DEGREE, EXPONENTS, LANES, PIECE};
use std::sync::OnceLock;

pub(super) mod arithmetic;
mod gather;
mod interleaved;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(test)]
mod tests;

use arithmetic::{Arithmetic, Split, Twiddle};

/// One admitted prime below `2^28` and its degree-648 transform plan.
#[derive(Clone, Debug)]
pub struct TrinomialWideLimbDomain {
    arithmetic: Arithmetic,
    table: Box<[[i32; LANES]; 256]>,
    first_level: Box<[[[i32; 3 * LANES]; 256]; 3]>,
    interleaved_first_level: OnceLock<Box<[[[i32; 3 * LANES]; 256]]>>,
    evaluation: [[i32; LANES]; LANES],
    interpolation: [[i32; LANES]; LANES],
    splits: Vec<Split>,
    omega: Twiddle,
    inverse_omega: Twiddle,
    one: Twiddle,
}

/// 648 plain signed residues, with magnitude strictly below the tagged prime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrinomialWideLimbSlots {
    values: [i32; DEGREE],
    prime: u32,
}

/// Lazily accumulated plain slot products in signed 64-bit lanes.
///
/// A checkpoint after 128 products permits arbitrarily long columns:
/// `128*(p-1)^2 + floor(p/2) < i64::MAX` for every `p < 2^28`.
#[derive(Clone, Debug)]
pub struct TrinomialWideLimbAccumulator {
    arithmetic: Arithmetic,
    sums: [i64; DEGREE],
    terms: u16,
}

impl TrinomialWideLimbDomain {
    /// Largest prime below each of `2^26`, `2^27`, `2^28`, congruent to 1 modulo 1944.
    pub const ADMITTED_PRIMES: [u32; 3] = [67_091_329, 134_217_649, 268_433_353];

    /// Build the transform plan; reject primes outside [`Self::ADMITTED_PRIMES`].
    pub fn new(prime: u32) -> Result<Self, TrinomialError> {
        if !Self::ADMITTED_PRIMES.contains(&prime) {
            return Err(TrinomialError::LimbInput {
                reason: "prime is not an admitted wide limb prime",
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
        let mut table = Box::new([[0; LANES]; 256]);
        for (byte, row) in table.iter_mut().enumerate() {
            for (lane, value) in row.iter_mut().enumerate() {
                *value = arithmetic.center(
                    evaluation[lane]
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| byte & (1 << j) != 0)
                        .map(|(_, &w)| i64::from(w))
                        .sum(),
                );
            }
        }
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
        let mut first_level = Box::new([[[0; 3 * LANES]; 256]; 3]);
        for (source, rows) in first_level.iter_mut().enumerate() {
            for (byte, row) in rows.iter_mut().enumerate() {
                let mut values = [0; DEGREE];
                values[source * 27 * LANES..source * 27 * LANES + LANES]
                    .copy_from_slice(&table[byte]);
                arithmetic::butterfly(
                    arithmetic,
                    omega,
                    inverse_omega,
                    &mut values,
                    &splits[0],
                    false,
                );
                for branch in 0..3 {
                    for lane in 0..LANES {
                        row[branch * LANES + lane] =
                            arithmetic.center(i64::from(values[branch * 27 * LANES + lane]));
                    }
                }
            }
        }
        // These recurrence bounds include every add/subtract, not only output
        // values. At p<2^28 the final B4 is <5.67p<2^31; no butterfly wraps.
        // B0=floor(p/2); C(B)=ceil(p/2+Bp/2^32);
        // Bnext=max(B+2C(B), B+C(B)+C(2C(B))).
        let bounds = arithmetic::forward_bounds(prime);
        if bounds[4] >= i64::from(i32::MAX) {
            return Err(TrinomialError::LimbInput {
                reason: "wide limb lazy bound exceeds signed lanes",
            });
        }
        Ok(Self {
            arithmetic,
            table,
            first_level,
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
    pub fn zero_slots(&self) -> TrinomialWideLimbSlots {
        TrinomialWideLimbSlots {
            values: [0; DEGREE],
            prime: self.prime(),
        }
    }

    /// Gather 648 bits into 81 bytes, one bit from each 81-bit piece.
    ///
    /// Reject a word count other than eleven or nonzero tail bits.
    pub fn gather_bits(bits: &[u64], out: &mut [u8; PIECE]) -> Result<(), TrinomialError> {
        Self::check_bits(bits)?;
        gather::indices(bits, out, true);
        Ok(())
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

    fn check_slots(&self, slots: &TrinomialWideLimbSlots) -> Result<(), TrinomialError> {
        if slots.prime != self.prime() {
            return Err(TrinomialError::LimbInput {
                reason: "slot prime does not match the domain",
            });
        }
        Ok(())
    }

    /// Transform 648 bits, little-endian within and across eleven words.
    ///
    /// Reject wrong length, nonzero tail bits, or a mismatched output prime.
    pub fn forward_bits(
        &self,
        bits: &[u64],
        out: &mut TrinomialWideLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.forward_bits_backend(bits, out, use_neon(), true)
    }

    fn forward_bits_backend(
        &self,
        bits: &[u64],
        out: &mut TrinomialWideLimbSlots,
        neon: bool,
        fused_table: bool,
    ) -> Result<(), TrinomialError> {
        self.bits_transform(bits, out, neon, fused_table, true, true, false)
    }

    /// Select table, gather, and final-level variants for kernel measurements.
    #[doc(hidden)]
    pub fn forward_bits_variant(
        &self,
        bits: &[u64],
        out: &mut TrinomialWideLimbSlots,
        fused_table: bool,
        transpose: bool,
        unroll_last: bool,
    ) -> Result<(), TrinomialError> {
        self.bits_transform(
            bits,
            out,
            use_neon(),
            fused_table,
            transpose,
            unroll_last,
            false,
        )
    }

    /// Benchmark the same butterfly with centered corrections after each level.
    #[doc(hidden)]
    pub fn forward_bits_corrected(
        &self,
        bits: &[u64],
        out: &mut TrinomialWideLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.bits_transform(bits, out, use_neon(), true, true, true, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn bits_transform(
        &self,
        bits: &[u64],
        out: &mut TrinomialWideLimbSlots,
        neon: bool,
        fused_table: bool,
        transpose: bool,
        unroll_last: bool,
        corrected: bool,
    ) -> Result<(), TrinomialError> {
        Self::check_bits(bits)?;
        self.check_slots(out)?;
        let mut indices = [0; PIECE];
        gather::indices(bits, &mut indices, transpose);
        if fused_table {
            #[cfg(target_arch = "aarch64")]
            if neon {
                neon::table(self, &indices, &mut out.values);
            } else {
                self.table_scalar(&indices, &mut out.values);
            }
            #[cfg(not(target_arch = "aarch64"))]
            self.table_scalar(&indices, &mut out.values);
        } else {
            for (t, row) in out.values.chunks_exact_mut(LANES).enumerate() {
                row.copy_from_slice(&self.table[usize::from(indices[t])]);
            }
        }
        self.transform_variant(
            &mut out.values,
            false,
            neon,
            fused_table,
            unroll_last,
            corrected,
        );
        Ok(())
    }

    fn table_scalar(&self, indices: &[u8; PIECE], values: &mut [i32; DEGREE]) {
        // Each table entry is centered, so these two additions give |s|<=3p/2,
        // below the ordinary first-level bound B1. Thus both table paths use
        // the same bounds for the remaining three lazy levels.
        for t in 0..27 {
            let a = &self.first_level[0][usize::from(indices[t])];
            let b = &self.first_level[1][usize::from(indices[t + 27])];
            let c = &self.first_level[2][usize::from(indices[t + 54])];
            for branch in 0..3 {
                for lane in 0..LANES {
                    let i = branch * LANES + lane;
                    values[(t + branch * 27) * LANES + lane] = a[i] + b[i] + c[i];
                }
            }
        }
    }

    /// Transform 648 centered coefficients in `[-(p-1)/2,(p-1)/2]`.
    ///
    /// Reject wrong length, noncentered coefficients, or a mismatched prime.
    pub fn forward_centered(
        &self,
        coefficients: &[i32],
        out: &mut TrinomialWideLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.forward_centered_backend(coefficients, out, use_neon())
    }

    fn forward_centered_backend(
        &self,
        coefficients: &[i32],
        out: &mut TrinomialWideLimbSlots,
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
                reason: "coefficient must be centered below the wide limb prime",
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
        slots: &TrinomialWideLimbSlots,
        out: &mut [i32],
    ) -> Result<(), TrinomialError> {
        self.inverse_centered_backend(slots, out, use_neon())
    }

    fn inverse_centered_backend(
        &self,
        slots: &TrinomialWideLimbSlots,
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
        self.transform_variant(values, inverse, neon, skip_first, true, false)
    }

    fn transform_variant(
        &self,
        values: &mut [i32; DEGREE],
        inverse: bool,
        neon: bool,
        skip_first: bool,
        unroll_last: bool,
        corrected: bool,
    ) {
        #[cfg(target_arch = "aarch64")]
        if neon {
            return neon::transform(self, values, inverse, skip_first, unroll_last, corrected);
        }
        let _ = (neon, unroll_last);
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
            if corrected && skip_first {
                for value in values.iter_mut() {
                    *value = self.arithmetic.center(i64::from(*value));
                }
            }
            for split in &self.splits[usize::from(skip_first)..] {
                arithmetic::butterfly(
                    self.arithmetic,
                    self.omega,
                    self.inverse_omega,
                    values,
                    split,
                    false,
                );
                if corrected {
                    for value in
                        &mut values[split.start * LANES..(split.start + 3 * split.stride) * LANES]
                    {
                        *value = self.arithmetic.center(i64::from(*value));
                    }
                }
            }
            // C(B4)<p for B4<5.67p and p<2^28. One multiplication by
            // one therefore gives the accumulator's strict |slot|<p bound.
            for value in values {
                *value = self.arithmetic.mul(*value, self.one);
            }
        }
    }
}

impl TrinomialWideLimbAccumulator {
    pub(super) const REDUCTION_TERMS: u16 = 128;

    /// Create an empty accumulator tagged with this domain's prime.
    pub fn new(domain: &TrinomialWideLimbDomain) -> Self {
        Self {
            arithmetic: domain.arithmetic,
            sums: [0; DEGREE],
            terms: 0,
        }
    }

    /// Add one product; reject mismatched primes before changing the sum.
    pub fn add_product(
        &mut self,
        lhs: &TrinomialWideLimbSlots,
        rhs: &TrinomialWideLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.add_product_backend(lhs, rhs, use_neon())
    }

    fn add_product_backend(
        &mut self,
        lhs: &TrinomialWideLimbSlots,
        rhs: &TrinomialWideLimbSlots,
        neon: bool,
    ) -> Result<(), TrinomialError> {
        if lhs.prime != self.arithmetic.prime || rhs.prime != self.arithmetic.prime {
            return Err(TrinomialError::LimbInput {
                reason: "product prime does not match the accumulator",
            });
        }
        #[cfg(target_arch = "aarch64")]
        if neon {
            neon::accumulate(&mut self.sums, &lhs.values, &rhs.values);
        } else {
            arithmetic::accumulate(&mut self.sums, &lhs.values, &rhs.values);
        }
        #[cfg(not(target_arch = "aarch64"))]
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
    pub fn finish(&mut self, out: &mut TrinomialWideLimbSlots) -> Result<(), TrinomialError> {
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
