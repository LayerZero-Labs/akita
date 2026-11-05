//! Exact adjoint sums in `Z[Y]/(Y^648 - Y^324 + 1)` as ring products.
//!
//! The modulus is the cyclotomic polynomial `Phi_1944`. Multiplication by `v`
//! is self-adjoint under the trace form `Tr(xy) = x^T G y`, so its coefficient
//! adjoint is `adj(rho, v) = G (v * G^-1 rho)`. `Tr(Y^n)` vanishes unless 324
//! divides `n`, where it is `324 * (2, 1, -1, -2)[n / 324]` for `n <= 1294`.
//! The scaled Gram matrix `G' = G / 324` therefore has two nonzero entries
//! per row, and its inverse acts on components of at most four coefficients.
//!
//! `w = G'^-1 rho` is computed once per functional and split into three
//! 43-bit limbs. For every column, each limb product `sum_i rows[i][j] * w_i`
//! runs in the 648-point limb domains of three primes and is recovered
//! exactly by Chinese remaindering; `G'` and the limb weights are applied to
//! the recovered integers. The result equals [`TrinomialAdjoint::apply`]
//! summed over the rows, exactly.

use akita_error::AkitaError;
use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::internal;
use crate::ring::trinomial::{TrinomialLimbAccumulator, TrinomialLimbDomain, TrinomialLimbSlots};

/// The ring degree of the limb domains.
const DEGREE: usize = 648;
const HALF: usize = DEGREE / 2;
/// The three largest admitted limb primes; their product exceeds `2^80`.
const PRIMES: [u32; 3] = [67_091_329, 134_217_649, 268_433_353];
const LIMB_BITS: u32 = 43;
const LIMBS: usize = 3;
/// `Tr(Y^(324 m)) / 324` for `m = 0..4`.
const TRACE: [i8; 4] = [2, 1, -1, -2];

/// The product of [`PRIMES`].
const MODULUS: u128 = PRIMES[0] as u128 * PRIMES[1] as u128 * PRIMES[2] as u128;

const fn power(base: u64, mut exponent: u64, prime: u64) -> u64 {
    let (mut base, mut result) = (base % prime, 1u64);
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base % prime;
        }
        base = base * base % prime;
        exponent >>= 1;
    }
    result
}

/// `p0^-1 mod p1`, `p0^-1 mod p2` and `p1^-1 mod p2`, by Fermat.
const INVERSES: [u64; 3] = {
    let [p0, p1, p2] = [PRIMES[0] as u64, PRIMES[1] as u64, PRIMES[2] as u64];
    [
        power(p0, p1 - 2, p1),
        power(p0, p2 - 2, p2),
        power(p1, p2 - 2, p2),
    ]
};

/// The largest magnitude of a limb product coefficient: `rank` products of
/// `D` terms, each a coefficient times a limb, and at most three linear
/// coefficients per remainder coefficient. `None` unless the degree is 648,
/// the limb domains represent the operands and the CRT recovers the result.
pub(super) fn limb_bound(rank: usize, bound: u128, degree: usize) -> Option<u128> {
    if degree != DEGREE {
        return None;
    }
    let smallest_half = u128::from(PRIMES[0] / 2);
    let limb_bound = 3u128
        .checked_mul(rank as u128)?
        .checked_mul(DEGREE as u128)?
        .checked_mul(bound)?
        .checked_mul(1u128 << LIMB_BITS)?;
    (bound <= smallest_half && limb_bound.checked_mul(2)? < MODULUS).then_some(limb_bound)
}

/// `G'_{ab}`, nonzero only when `324` divides `a + b`.
fn gram(a: usize, b: usize) -> i8 {
    let n = a + b;
    if n.is_multiple_of(HALF) {
        TRACE.get(n / HALF).copied().unwrap_or(0)
    } else {
        0
    }
}

/// The coefficients coupled to `a` by `G'`.
fn neighbours(a: usize) -> impl Iterator<Item = usize> {
    (0..TRACE.len())
        .filter_map(move |m| (m * HALF).checked_sub(a))
        .filter(|&b| b < DEGREE)
}

/// `G'^-1 rho`, solving each connected component of `G'` directly.
fn gram_solve<F: Field>(rho: &[F]) -> Result<Vec<F>, AkitaError> {
    if rho.len() != DEGREE {
        return Err(internal("trace functional"));
    }
    let mut w = vec![F::zero(); DEGREE];
    let mut seen = [false; DEGREE];
    for start in 0..DEGREE {
        if seen[start] {
            continue;
        }
        let mut component = vec![start];
        seen[start] = true;
        let mut next = 0;
        while let Some(&a) = component.get(next) {
            for b in neighbours(a) {
                if !seen[b] {
                    seen[b] = true;
                    component.push(b);
                }
            }
            next += 1;
        }
        // Gaussian elimination on the component's rows of `G' w = rho`.
        let size = component.len();
        let mut rows = component
            .iter()
            .map(|&a| {
                let mut row = component
                    .iter()
                    .map(|&b| F::from_i64(i64::from(gram(a, b))))
                    .collect::<Vec<_>>();
                row.push(rho[a]);
                row
            })
            .collect::<Vec<_>>();
        for column in 0..size {
            let pivot = (column..size)
                .find(|&r| !rows[r][column].is_zero())
                .ok_or_else(|| internal("trace Gram pivot"))?;
            rows.swap(column, pivot);
            let inverse = rows[column][column]
                .inverse()
                .ok_or_else(|| internal("trace Gram inverse"))?;
            for value in &mut rows[column] {
                *value *= inverse;
            }
            let pivot_row = rows[column].clone();
            for (r, row) in rows.iter_mut().enumerate() {
                if r != column && !row[column].is_zero() {
                    let factor = row[column];
                    for (value, &p) in row.iter_mut().zip(&pivot_row) {
                        *value -= factor * p;
                    }
                }
            }
        }
        for (row, &a) in rows.iter().zip(&component) {
            w[a] = row[size];
        }
    }
    Ok(w)
}

/// Per-worker accumulators and buffers, kept off the worker stacks.
struct Scratch {
    sums: Vec<[TrinomialLimbAccumulator; LIMBS]>,
    slots: Vec<TrinomialLimbSlots>,
    operand: Box<[i32; DEGREE]>,
    residues: Box<[[[i32; DEGREE]; LIMBS]; PRIMES.len()]>,
    products: Box<[[i128; DEGREE]; LIMBS]>,
}

/// The exact centered integer with residues `r` modulo [`PRIMES`].
#[inline]
fn garner(r: [i32; 3]) -> i128 {
    let [p0, p1, p2] = PRIMES.map(u64::from);
    let [a0, a1, a2] = [(r[0], p0), (r[1], p1), (r[2], p2)]
        .map(|(value, prime)| i64::from(value).rem_euclid(prime as i64) as u64);
    let v1 = (a1 + p1 - a0 % p1) % p1 * INVERSES[0] % p1;
    let v2 = ((a2 + p2 - a0 % p2) % p2 * INVERSES[1] % p2 + p2 - v1 % p2) % p2 * INVERSES[2] % p2;
    let x = u128::from(a0) + u128::from(p0) * (u128::from(v1) + u128::from(p1) * u128::from(v2));
    if x > MODULUS / 2 {
        -((MODULUS - x) as i128)
    } else {
        x as i128
    }
}

/// `h_j = sum_i adj(rho[i], rows[i][j])` through limb-domain ring products.
///
/// The caller has validated the shapes and that every coefficient is at most
/// `bound`, and [`limb_bound`] admitted `bound`; `D` must be 648.
pub(super) fn apply_rows<F: Field, const D: usize>(
    rho: &[[F; D]],
    rows: &[Vec<[i128; D]>],
    bound: u128,
    canonical: fn(&F) -> Option<u128>,
) -> Result<Vec<[F; D]>, AkitaError> {
    let limb_bound =
        limb_bound(rows.len(), bound, D).ok_or_else(|| internal("trace limb bound"))?;
    let columns = rows.first().map_or(0, Vec::len);
    let limb = |error: crate::TrinomialError| AkitaError::Internal(format!("trace limb: {error}"));
    let domains = PRIMES
        .iter()
        .map(|&prime| TrinomialLimbDomain::new(prime).map_err(limb))
        .collect::<Result<Vec<_>, _>>()?;
    // weights[prime][row][limb]: the transformed limbs of `G'^-1 rho[row]`.
    let mut weights = domains
        .iter()
        .map(|_| Vec::with_capacity(rho.len()))
        .collect::<Vec<_>>();
    for functional in rho {
        let w = gram_solve(functional)?
            .iter()
            .map(|value| canonical(value).ok_or_else(|| internal("trace functional lift")))
            .collect::<Result<Vec<u128>, _>>()?;
        for (domain, weights) in domains.iter().zip(&mut weights) {
            let prime = u64::from(domain.prime());
            let mut limbs: [TrinomialLimbSlots; LIMBS] =
                core::array::from_fn(|_| domain.zero_slots());
            let mut operand = [0i32; DEGREE];
            for (index, slots) in limbs.iter_mut().enumerate() {
                for (slot, &value) in operand.iter_mut().zip(&w) {
                    let digit =
                        (value >> (LIMB_BITS as usize * index)) as u64 & ((1 << LIMB_BITS) - 1);
                    let residue = digit % prime;
                    *slot = if residue > prime / 2 {
                        residue as i32 - prime as i32
                    } else {
                        residue as i32
                    };
                }
                domain.forward_centered(&operand, slots).map_err(limb)?;
            }
            weights.push(limbs);
        }
    }
    let high = F::from_u128(1 << (2 * LIMB_BITS));
    let new_scratch = || Scratch {
        sums: domains
            .iter()
            .map(|domain| core::array::from_fn(|_| TrinomialLimbAccumulator::new(domain)))
            .collect(),
        slots: domains
            .iter()
            .map(TrinomialLimbDomain::zero_slots)
            .collect(),
        operand: Box::new([0; DEGREE]),
        residues: Box::new([[[0; DEGREE]; LIMBS]; PRIMES.len()]),
        products: Box::new([[0; DEGREE]; LIMBS]),
    };
    let column = |scratch: &mut Scratch, j: usize| -> Result<[F; D], AkitaError> {
        let Scratch {
            sums,
            slots,
            operand,
            residues,
            products,
        } = scratch;
        for ((((domain, weights), sums), slots), residues) in domains
            .iter()
            .zip(&weights)
            .zip(sums.iter_mut())
            .zip(slots.iter_mut())
            .zip(residues.iter_mut())
        {
            for (row, limbs) in rows.iter().zip(weights) {
                let element = row.get(j).ok_or_else(|| internal("trace column"))?;
                for (slot, &value) in operand.iter_mut().zip(element) {
                    *slot = value as i32;
                }
                domain
                    .forward_centered(operand.as_slice(), slots)
                    .map_err(limb)?;
                for (sum, w) in sums.iter_mut().zip(limbs) {
                    sum.add_product(slots, w).map_err(limb)?;
                }
            }
            for (sum, residues) in sums.iter_mut().zip(residues.iter_mut()) {
                sum.finish(slots).map_err(limb)?;
                domain.inverse_centered(slots, residues).map_err(limb)?;
            }
        }
        // u_l = sum_i rows[i][j] * limb_l(w_i) exactly; then h = G' u.
        for (l, product) in products.iter_mut().enumerate() {
            for (t, slot) in product.iter_mut().enumerate() {
                *slot = garner([residues[0][l][t], residues[1][l][t], residues[2][l][t]]);
                if slot.unsigned_abs() > limb_bound {
                    return Err(internal("trace limb product"));
                }
            }
        }
        let mut h = [F::zero(); D];
        for (a, slot) in h.iter_mut().enumerate() {
            let mut limbs = [0i128; LIMBS];
            for b in neighbours(a) {
                let entry = i128::from(gram(a, b));
                for (sum, product) in limbs.iter_mut().zip(products.iter()) {
                    *sum += entry * product[b];
                }
            }
            // A row of G' has absolute sum at most 3, so every limb is below
            // 3 * 2^80 and the two low limbs combine exactly in an i128.
            let low = limbs[0] + (limbs[1] << LIMB_BITS);
            *slot = F::from_i128(low) + high * F::from_i128(limbs[2]);
        }
        Ok(h)
    };
    #[cfg(feature = "parallel")]
    let output = (0..columns)
        .into_par_iter()
        .map_init(new_scratch, column)
        .collect();
    #[cfg(not(feature = "parallel"))]
    let output = {
        let mut scratch = new_scratch();
        (0..columns).map(|j| column(&mut scratch, j)).collect()
    };
    output
}
