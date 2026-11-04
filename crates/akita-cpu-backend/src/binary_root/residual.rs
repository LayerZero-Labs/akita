//! Exact residual remainders by Chinese remaindering over limb primes.
//!
//! Every operand of `sum A*z - sum iota(c)*T` is a centered integer below the
//! smallest admitted limb prime, so each limb domain multiplies them in
//! `Z_p[X]/(X^648 - X^324 + 1)` without reduction. The remainder modulo Psi is
//! linear, so its residue modulo each prime is the limb result. Three primes
//! whose product exceeds twice the relation bound recover the unique centered
//! integer. The first prime is q24, so q24-divisibility is the vanishing of
//! the first residue. The result equals the remainder of
//! [`BinaryRootRelation::from_response`] exactly.

use super::{dimensions, embed_challenges, invalid, relation_bound, BinaryRootRelation, D};
use akita_algebra::ring::trinomial::image::{Q24Image, Q24};
use akita_algebra::{TrinomialLimbAccumulator, TrinomialLimbDomain, TrinomialLimbSlots};
use akita_challenges::{Challenges, FoldChallengeDistribution};
use akita_error::{checked, AkitaError};
use akita_types::binary_source::BinarySourceDescriptor;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// q24 and the largest admitted primes below `2^28` and `2^27`.
const PRIMES: [u32; 3] = [Q24, 268_433_353, 134_217_649];
/// Source rows per parallel task.
const ROW_CHUNK: usize = 128;

type Coefficients = [i32; D];

impl BinaryRootRelation {
    /// The residual remainders `rem_Psi(sum_j A_ij z_j - sum_c iota(c_c) T_ci)`
    /// in polynomial/matrix-row order, without quotients.
    ///
    /// Inputs are validated as in [`Self::from_response`]. Source-row chunks
    /// run in parallel under the `parallel` feature; their residues are summed
    /// exactly, so the result does not depend on the schedule.
    pub fn remainders(
        descriptor: &BinarySourceDescriptor,
        matrix: &[Vec<[i128; D]>],
        response: &[[i128; D]],
        images: &[[Q24Image; D]],
        challenges: &Challenges,
        family: FoldChallengeDistribution,
    ) -> Result<Vec<[i128; D]>, AkitaError> {
        let (polynomials, columns, source_rows) = dimensions(descriptor, matrix)?;
        let rank = matrix.len();
        let bound = relation_bound(descriptor, matrix, response, images, family)?;
        let modulus = PRIMES
            .iter()
            .try_fold(1u128, |product, &prime| {
                product.checked_mul(u128::from(prime))
            })
            .filter(|&modulus| bound.checked_mul(2).is_some_and(|twice| twice < modulus))
            .ok_or_else(|| invalid("binary relation exceeds its limb-prime lift"))?;
        let embedded = embed_challenges(descriptor, challenges, family)?;
        let narrow = |element: &[i128; D]| -> Result<Coefficients, AkitaError> {
            let mut out = [0i32; D];
            for (slot, &value) in out.iter_mut().zip(element) {
                *slot = i32::try_from(value).map_err(|_| invalid("binary operand exceeds i32"))?;
            }
            Ok(out)
        };
        let matrix = matrix
            .iter()
            .map(|row| row.iter().map(narrow).collect::<Result<Vec<_>, _>>())
            .collect::<Result<Vec<_>, _>>()?;
        let response = response.iter().map(narrow).collect::<Result<Vec<_>, _>>()?;
        let embedded = embedded.iter().map(narrow).collect::<Result<Vec<_>, _>>()?;
        let images = images
            .iter()
            .map(|image| {
                let mut out = [0i32; D];
                for (slot, value) in out.iter_mut().zip(image) {
                    *slot = value.centered();
                }
                out
            })
            .collect::<Vec<_>>();
        let domains = PRIMES
            .iter()
            .map(|&prime| TrinomialLimbDomain::new(prime).map_err(limb))
            .collect::<Result<Vec<_>, _>>()?;
        let chunks = source_rows.div_ceil(ROW_CHUNK);
        // One task per (prime, polynomial, row chunk); the image terms of a
        // (prime, polynomial) pair ride on its first chunk with a minus sign.
        let tasks = checked::product([PRIMES.len(), polynomials, chunks])
            .ok_or(AkitaError::InvalidProof)?;
        let task = |task: usize| -> Result<Vec<Coefficients>, AkitaError> {
            let prime = task / (polynomials * chunks);
            let polynomial = task / chunks % polynomials;
            let chunk = task % chunks;
            let domain = domains.get(prime).ok_or(AkitaError::InvalidProof)?;
            let mut left = domain.zero_slots();
            let mut right = domain.zero_slots();
            let mut sums = (0..rank)
                .map(|_| TrinomialLimbAccumulator::new(domain))
                .collect::<Vec<_>>();
            let start = checked::product([chunk, ROW_CHUNK]).ok_or(AkitaError::InvalidProof)?;
            let end = source_rows.min(start + ROW_CHUNK);
            for row in start..end {
                let z = response
                    .get(polynomial * source_rows + row)
                    .ok_or(AkitaError::InvalidProof)?;
                domain.forward_centered(z, &mut right).map_err(limb)?;
                for (sum, matrix_row) in sums.iter_mut().zip(&matrix) {
                    let a = matrix_row.get(row).ok_or(AkitaError::InvalidProof)?;
                    domain.forward_centered(a, &mut left).map_err(limb)?;
                    sum.add_product(&left, &right).map_err(limb)?;
                }
            }
            let mut out = finish(domain, &mut sums)?;
            if chunk == 0 {
                for column in 0..columns {
                    let index = polynomial * columns + column;
                    let challenge = embedded.get(index).ok_or(AkitaError::InvalidProof)?;
                    domain
                        .forward_centered(challenge, &mut right)
                        .map_err(limb)?;
                    for (row, sum) in sums.iter_mut().enumerate() {
                        let image = images
                            .get(index * rank + row)
                            .ok_or(AkitaError::InvalidProof)?;
                        domain.forward_centered(image, &mut left).map_err(limb)?;
                        sum.add_product(&left, &right).map_err(limb)?;
                    }
                }
                for (out, images) in out.iter_mut().zip(finish(domain, &mut sums)?) {
                    for (value, image) in out.iter_mut().zip(images) {
                        *value -= image;
                    }
                }
            }
            Ok(out)
        };
        #[cfg(feature = "parallel")]
        let partials = (0..tasks)
            .into_par_iter()
            .map(task)
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(not(feature = "parallel"))]
        let partials = (0..tasks).map(task).collect::<Result<Vec<_>, _>>()?;
        let count = checked::product([polynomials, rank]).ok_or(AkitaError::InvalidProof)?;
        let mut residues = vec![[[0i64; D]; PRIMES.len()]; count];
        for (task, partial) in partials.iter().enumerate() {
            let prime = task / (polynomials * chunks);
            let polynomial = task / chunks % polynomials;
            for (row, coefficients) in partial.iter().enumerate() {
                let sums = residues
                    .get_mut(polynomial * rank + row)
                    .and_then(|row| row.get_mut(prime))
                    .ok_or(AkitaError::InvalidProof)?;
                for (sum, &value) in sums.iter_mut().zip(coefficients) {
                    *sum += i64::from(value);
                }
            }
        }
        let crt = Garner::new(modulus)?;
        residues
            .iter()
            .map(|residues| {
                let mut row = [0i128; D];
                for (t, value) in row.iter_mut().enumerate() {
                    *value = crt.lift(residues.map(|residues| residues[t]), bound)?;
                }
                Ok(row)
            })
            .collect()
    }
}

/// Centered residues of each accumulator, in row order; clears the sums.
fn finish(
    domain: &TrinomialLimbDomain,
    sums: &mut [TrinomialLimbAccumulator],
) -> Result<Vec<Coefficients>, AkitaError> {
    let mut slots: TrinomialLimbSlots = domain.zero_slots();
    sums.iter_mut()
        .map(|sum| {
            sum.finish(&mut slots).map_err(limb)?;
            let mut out = [0i32; D];
            domain.inverse_centered(&slots, &mut out).map_err(limb)?;
            Ok(out)
        })
        .collect()
}

/// Garner reconstruction over [`PRIMES`].
struct Garner {
    modulus: u128,
    /// `q24^-1 mod p1`, `q24^-1 mod p2` and `p1^-1 mod p2`.
    inverses: [u64; 3],
}

impl Garner {
    fn new(modulus: u128) -> Result<Self, AkitaError> {
        let [q, p1, p2] = PRIMES.map(u64::from);
        Ok(Self {
            modulus,
            inverses: [
                inverse(q % p1, p1)?,
                inverse(q % p2, p2)?,
                inverse(p1 % p2, p2)?,
            ],
        })
    }

    /// The centered integer with these residue sums, which must vanish modulo
    /// q24 and lie within `bound`.
    fn lift(&self, sums: [i64; 3], bound: u128) -> Result<i128, AkitaError> {
        let [q, p1, p2] = PRIMES.map(u64::from);
        let [a0, a1, a2] = [(sums[0], q), (sums[1], p1), (sums[2], p2)]
            .map(|(sum, prime)| sum.rem_euclid(prime as i64) as u64);
        if a0 != 0 {
            return Err(invalid(
                "binary commitment residual is not divisible by q24",
            ));
        }
        // x = q*v1 + q*p1*v2 with v1 = a1/q mod p1, v2 = (a2/q - v1)/p1 mod p2.
        let v1 = a1 * self.inverses[0] % p1;
        let v2 = (a2 * self.inverses[1] % p2 + p2 - v1 % p2) % p2 * self.inverses[2] % p2;
        let x = u128::from(q) * (u128::from(v1) + u128::from(p1) * u128::from(v2));
        let (negative, magnitude) = if x > self.modulus / 2 {
            (true, self.modulus - x)
        } else {
            (false, x)
        };
        if magnitude > bound {
            return Err(invalid(
                "binary transform coefficient exceeds its verified integer bound",
            ));
        }
        let magnitude = i128::try_from(magnitude).map_err(|_| AkitaError::InvalidProof)?;
        Ok(if negative { -magnitude } else { magnitude })
    }
}

/// `value^-1 mod prime` by Fermat's little theorem.
fn inverse(value: u64, prime: u64) -> Result<u64, AkitaError> {
    if value == 0 {
        return Err(AkitaError::InvalidProof);
    }
    let (mut base, mut exponent, mut result) = (value, prime - 2, 1u64);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base % prime;
        }
        base = base * base % prime;
        exponent >>= 1;
    }
    Ok(result)
}

fn limb(error: akita_algebra::TrinomialError) -> AkitaError {
    invalid(&error.to_string())
}
