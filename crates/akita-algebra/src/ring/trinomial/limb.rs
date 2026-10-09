//! Exact small-prime arithmetic for the degree-648 packed commitment ring.
//!
//! Slots are centered signed Montgomery residues, interleaved across the eight
//! roots of `Y^8 - Y^4 + 1`, where `Y = X^81`. A 256-by-eight table replaces
//! that first split for bits; four ternary splits finish the evaluation map.

use super::TrinomialError;

mod arithmetic;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(test)]
mod tests;

use arithmetic::{Arithmetic, Split};

const DEGREE: usize = 648;
const LANES: usize = 8;
const PIECE: usize = 81;
const EXPONENTS: [u32; LANES] = [1, 5, 7, 11, 13, 17, 19, 23];

/// One admitted prime and its transform plan for `X^648 - X^324 + 1`.
#[derive(Clone, Debug)]
pub struct TrinomialLimbDomain {
    arithmetic: Arithmetic,
    table: Box<[[i16; LANES]; 256]>,
    evaluation: [[i16; LANES]; LANES],
    interpolation: [[i16; LANES]; LANES],
    splits: Vec<Split>,
    omega: i16,
    inverse_omega: i16,
}

/// 648 private, centered Montgomery transform values of one limb.
///
/// The prime tag prevents accidentally mixing rings. Construct a zero vector
/// with [`TrinomialLimbDomain::zero_slots`], then fill it through a transform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrinomialLimbSlots {
    values: [i16; DEGREE],
    prime: u16,
}

/// A lazily reduced sum of slot-wise products.
///
/// Products use centered Montgomery factors but are accumulated *before*
/// Montgomery reduction in signed 32-bit lanes. After 15 terms the accumulator
/// reduces internally. Thus any number of additions is supported, including
/// 8192, without caller-managed reductions.
#[derive(Clone, Debug)]
pub struct TrinomialLimbAccumulator {
    arithmetic: Arithmetic,
    sums: [i32; DEGREE],
    terms: u8,
}

fn use_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        crate::ntt::neon::use_neon_ntt()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

impl TrinomialLimbDomain {
    /// Every prime below `2^15` with `1944 | p - 1`, ascending.
    pub const ADMITTED_PRIMES: [u16; 4] = [3889, 9721, 17497, 19441];

    /// Prepare the bit table and full-split transform for an admitted prime.
    ///
    /// Returns `LimbInput` for a prime outside [`Self::ADMITTED_PRIMES`].
    pub fn new(prime: u16) -> Result<Self, TrinomialError> {
        if !Self::ADMITTED_PRIMES.contains(&prime) {
            return Err(TrinomialError::LimbInput {
                reason: "prime is not an admitted limb prime",
            });
        }
        let arithmetic = Arithmetic::new(prime);
        // Order 1944 = 2^3 * 3^5: checking its two maximal proper divisors
        // proves exact order. The admitted-prime check guarantees existence.
        let root = (2..prime)
            .map(|candidate| arithmetic.pow(candidate, u32::from(prime - 1) / 1944))
            .find(|&root| arithmetic.pow(root, 972) != 1 && arithmetic.pow(root, 648) != 1)
            .ok_or(TrinomialError::LimbInput {
                reason: "prime does not supply a primitive 1944th root",
            })?;
        let omega = arithmetic.encode(arithmetic.pow(root, 648));
        let inverse_omega = arithmetic.encode(arithmetic.pow(root, 1296));
        let mut evaluation = [[0; LANES]; LANES];
        for (row, &exponent) in evaluation.iter_mut().zip(&EXPONENTS) {
            for (j, value) in row.iter_mut().enumerate() {
                *value = arithmetic.encode(arithmetic.pow(root, 81 * exponent * j as u32));
            }
        }
        let interpolation = arithmetic::invert_evaluation(&arithmetic, &evaluation)?;
        let mut table = Box::new([[0; LANES]; 256]);
        for (byte, row) in table.iter_mut().enumerate() {
            for (lane, value) in row.iter_mut().enumerate() {
                for (j, &weight) in evaluation[lane].iter().enumerate() {
                    if byte & (1 << j) != 0 {
                        *value = arithmetic.add(*value, weight);
                    }
                }
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
                let r = roots.map(|e| arithmetic.encode(arithmetic.pow(root, e)));
                let r2 = r.map(|r| arithmetic.mul(r, r));
                let ri = roots
                    .map(|e| arithmetic.encode(arithmetic.pow(root, (1944 - e % 1944) % 1944)));
                let ri2 = ri.map(|r| arithmetic.mul(r, r));
                splits.push(Split {
                    start: group * len,
                    stride,
                    r,
                    r2,
                    ri,
                    ri2,
                });
                for branch in 0..3 {
                    children.push(roots.map(|e| e + 648 * branch));
                }
            }
            exponents = children;
            len = stride;
        }
        Ok(Self {
            arithmetic,
            table,
            evaluation,
            interpolation,
            splits,
            omega,
            inverse_omega,
        })
    }

    /// The modulus of this limb.
    pub fn prime(&self) -> u16 {
        self.arithmetic.prime
    }

    /// Construct the additive identity with this domain's prime tag.
    pub fn zero_slots(&self) -> TrinomialLimbSlots {
        TrinomialLimbSlots {
            values: [0; DEGREE],
            prime: self.prime(),
        }
    }

    /// Transform 648 bits, little-endian within and across eleven words.
    ///
    /// Returns `CoefficientLength` for a word count other than eleven and
    /// `LimbInput` for nonzero unused tail bits or a mismatched output prime.
    pub fn forward_bits(
        &self,
        bits: &[u64],
        out: &mut TrinomialLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.forward_bits_backend(bits, out, use_neon())
    }

    fn forward_bits_backend(
        &self,
        bits: &[u64],
        out: &mut TrinomialLimbSlots,
        neon: bool,
    ) -> Result<(), TrinomialError> {
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
        self.check_slots(out)?;
        for (t, row) in out.values.chunks_exact_mut(LANES).enumerate() {
            let mut byte = 0;
            for j in 0..LANES {
                let position = t + PIECE * j;
                byte |= (((bits[position / 64] >> (position % 64)) & 1) as usize) << j;
            }
            row.copy_from_slice(&self.table[byte]);
        }
        self.transform(&mut out.values, false, neon);
        Ok(())
    }

    /// Transform exactly 648 canonical coefficients in `[0, p)`.
    ///
    /// Returns `CoefficientLength` for a wrong length and `LimbInput` for a
    /// noncanonical coefficient or mismatched output prime. Errors leave the
    /// destination unchanged.
    pub fn forward_canonical(
        &self,
        coefficients: &[u16],
        out: &mut TrinomialLimbSlots,
    ) -> Result<(), TrinomialError> {
        self.forward_canonical_backend(coefficients, out, use_neon())
    }

    fn forward_canonical_backend(
        &self,
        coefficients: &[u16],
        out: &mut TrinomialLimbSlots,
        neon: bool,
    ) -> Result<(), TrinomialError> {
        if coefficients.len() != DEGREE {
            return Err(TrinomialError::CoefficientLength {
                expected: DEGREE,
                actual: coefficients.len(),
            });
        }
        if coefficients.iter().any(|&value| value >= self.prime()) {
            return Err(TrinomialError::LimbInput {
                reason: "coefficient must be below the limb prime",
            });
        }
        self.check_slots(out)?;
        for t in 0..PIECE {
            let input: [i16; LANES] =
                std::array::from_fn(|j| self.arithmetic.encode(coefficients[t + PIECE * j]));
            for (lane, weights) in self.evaluation.iter().enumerate() {
                out.values[t * LANES + lane] = self.arithmetic.dot(&input, weights);
            }
        }
        self.transform(&mut out.values, false, neon);
        Ok(())
    }

    /// Invert slots to exactly 648 canonical coefficients in `[0, p)`.
    ///
    /// Returns `CoefficientLength` for a wrong output length and `LimbInput`
    /// for slots from a different prime. Errors leave the output unchanged.
    pub fn inverse(
        &self,
        slots: &TrinomialLimbSlots,
        out: &mut [u16],
    ) -> Result<(), TrinomialError> {
        self.inverse_backend(slots, out, use_neon())
    }

    fn inverse_backend(
        &self,
        slots: &TrinomialLimbSlots,
        out: &mut [u16],
        neon: bool,
    ) -> Result<(), TrinomialError> {
        if out.len() != DEGREE {
            return Err(TrinomialError::CoefficientLength {
                expected: DEGREE,
                actual: out.len(),
            });
        }
        self.check_slots(slots)?;
        let mut values = slots.values;
        self.transform(&mut values, true, neon);
        for (t, row) in values.chunks_exact(LANES).enumerate() {
            for (j, weights) in self.interpolation.iter().enumerate() {
                out[t + PIECE * j] = self.arithmetic.decode(self.arithmetic.dot(row, weights));
            }
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

    fn transform(&self, values: &mut [i16; DEGREE], inverse: bool, neon: bool) {
        #[cfg(target_arch = "aarch64")]
        if neon {
            return neon::transform(self, values, inverse);
        }
        let _ = neon;
        if inverse {
            for split in self.splits.iter().rev() {
                arithmetic::butterfly(self, values, split, true);
            }
        } else {
            for split in &self.splits {
                arithmetic::butterfly(self, values, split, false);
            }
        }
    }
}

impl TrinomialLimbAccumulator {
    /// Construct an empty sum for this domain.
    pub fn new(domain: &TrinomialLimbDomain) -> Self {
        Self {
            arithmetic: domain.arithmetic,
            sums: [0; DEGREE],
            terms: 0,
        }
    }

    /// Add a slot-wise product, reducing internally every fifteen products.
    ///
    /// Returns `LimbInput` on mismatched prime tags, without changing the sum.
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
        self.terms += 1;
        if self.terms == 15 {
            self.reduce();
        }
        Ok(())
    }

    fn reduce(&mut self) {
        // Each raw product is <= floor(p/2)^2. With fifteen products and
        // a carried residue times R, |sum| <= 15*9720^2 + 9720*65536
        // = 2,054,185,920 < i32::MAX. Reduction uses i64 intermediates.
        for sum in &mut self.sums {
            *sum = i32::from(self.arithmetic.reduce_wide(i64::from(*sum))) * 65536;
        }
        self.terms = 0;
    }

    /// Write the reduced sum, then clear it for another column.
    ///
    /// Returns `LimbInput` for a mismatched output prime without clearing.
    pub fn finish(&mut self, out: &mut TrinomialLimbSlots) -> Result<(), TrinomialError> {
        if out.prime != self.arithmetic.prime {
            return Err(TrinomialError::LimbInput {
                reason: "output prime does not match the accumulator",
            });
        }
        for (value, &sum) in out.values.iter_mut().zip(&self.sums) {
            *value = self.arithmetic.reduce_wide(i64::from(sum));
        }
        self.sums.fill(0);
        self.terms = 0;
        Ok(())
    }
}
