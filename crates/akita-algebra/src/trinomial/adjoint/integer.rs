//! Exact adjoint sums against bounded integer rows, for prover tables.
//!
//! The proof field need not have smooth roots of unity. Each extended row is
//! split into two 64-bit limbs of its canonical integers, and every Hankel
//! product is computed exactly over the integers in the smooth auxiliary
//! field: a linear product of length `3D-2` fits a length-`3D` transform with
//! no wrap-around. The checked bound below makes the centered lift unique,
//! after which the limbs recombine in the proof field. The result equals the
//! field-native [`TrinomialAdjoint::apply`] sum exactly.

use akita_error::AkitaError;

use jolt_algebra::checked;
use jolt_algebra::{CanonicalEncoding, Field, One, Ring, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{internal, TrinomialAdjoint};
use crate::trinomial::TrinomialModulus;
use jolt_algebra::fft::{FftWorkspace, SmoothDomain};

use crate::trinomial::TrinomialAuxiliary as Auxiliary;
use crate::trinomial::TrinomialField;

/// Per-worker transform buffers.
struct Scratch {
    workspace: FftWorkspace<Auxiliary>,
    padded: Vec<Auxiliary>,
    transform: Vec<Auxiliary>,
    sums: [Vec<Auxiliary>; 2],
    products: [Vec<Auxiliary>; 2],
}

impl Scratch {
    fn new(domain: &SmoothDomain<Auxiliary>) -> Self {
        let buffer = || vec![Auxiliary::zero(); domain.n];
        Self {
            workspace: domain.workspace(),
            padded: buffer(),
            transform: buffer(),
            sums: [buffer(), buffer()],
            products: [buffer(), buffer()],
        }
    }
}

impl<const D: usize, M: TrinomialModulus> TrinomialAdjoint<D, M> {
    /// Return `h_j = sum_i adj(rho[i], rows[i][j])` for every column `j`.
    ///
    /// Every row holds the same number of columns, and every coefficient has
    /// absolute value at most `bound`. `canonical` returns the canonical
    /// integer of a proof-field element, which must be below `2^128`. This is
    /// a prover kernel: columns run in parallel under the `parallel` feature.
    pub fn apply_integer_rows<F: Field>(
        &self,
        rho: &[[F; D]],
        rows: &[Vec<[i128; D]>],
        bound: u128,
        canonical: fn(&F) -> Option<u128>,
    ) -> Result<Vec<[F; D]>, AkitaError> {
        let columns = rows.first().map_or(0, Vec::len);
        if rho.is_empty() || rho.len() != rows.len() || rows.iter().any(|r| r.len() != columns) {
            return Err(AkitaError::InvalidInput(
                "integer adjoint rows differ from their functionals".into(),
            ));
        }
        let length = checked::product([3, D])
            .filter(|&n| Auxiliary::SMOOTH_SUBGROUP_ORDER.is_multiple_of(n))
            .ok_or_else(|| {
                AkitaError::InvalidInput("integer adjoint degree has no auxiliary domain".into())
            })?;
        let modulus = (-Auxiliary::one())
            .to_u128_checked()
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| internal("auxiliary modulus"))?;
        // A limb is below 2^64; each output sums rank * D products of a limb
        // and a coefficient. Twice that bound must stay below the modulus.
        let output_bound = (rows.len() as u128)
            .checked_mul(D as u128)
            .and_then(|x| x.checked_mul(bound))
            .and_then(|x| x.checked_mul(u128::from(u64::MAX)))
            .filter(|&x| x < modulus / 2)
            .ok_or_else(|| {
                AkitaError::InvalidInput("integer adjoint exceeds its exact lift bound".into())
            })?;
        if rows
            .iter()
            .flatten()
            .flatten()
            .any(|c| c.unsigned_abs() > bound)
        {
            return Err(AkitaError::InvalidInput(
                "integer adjoint coefficient exceeds its bound".into(),
            ));
        }
        let root = Auxiliary::primitive_root(length)
            .map_err(|_| internal("integer adjoint auxiliary root"))?;
        let domain = SmoothDomain::new(root, length)
            .map_err(|_| internal("integer adjoint auxiliary domain"))?;
        let mut scratch = Scratch::new(&domain);
        let mut limbs = Vec::with_capacity(rho.len());
        for functional in rho {
            let extended = self.extend_row(functional)?;
            let mut pair = [
                vec![Auxiliary::zero(); length],
                vec![Auxiliary::zero(); length],
            ];
            for (limb, transform) in pair.iter_mut().enumerate() {
                for (slot, value) in scratch.padded.iter_mut().zip(&extended) {
                    let value = canonical(value).ok_or_else(|| {
                        AkitaError::InvalidInput("proof field exceeds 128-bit lift".into())
                    })?;
                    *slot = Auxiliary::from_u64((value >> (64 * limb)) as u64);
                }
                domain
                    .forward_into(&scratch.padded, transform, &mut scratch.workspace)
                    .map_err(|_| internal("integer adjoint auxiliary transform"))?;
            }
            limbs.push(pair);
        }
        let limb_scale = F::from_u128(1 << 64);
        let column = |scratch: &mut Scratch, j: usize| -> Result<[F; D], AkitaError> {
            for sum in &mut scratch.sums {
                sum.fill(Auxiliary::zero());
            }
            for (row, pair) in rows.iter().zip(&limbs) {
                let element = row
                    .get(j)
                    .ok_or_else(|| internal("integer adjoint column"))?;
                scratch.padded.fill(Auxiliary::zero());
                for (slot, &value) in scratch.padded.iter_mut().zip(element.iter().rev()) {
                    *slot = Auxiliary::from_i128(value);
                }
                domain
                    .forward_into(
                        &scratch.padded,
                        &mut scratch.transform,
                        &mut scratch.workspace,
                    )
                    .map_err(|_| internal("integer adjoint auxiliary transform"))?;
                for (sum, limb) in scratch.sums.iter_mut().zip(pair) {
                    for ((slot, &a), &g) in sum.iter_mut().zip(&scratch.transform).zip(limb) {
                        *slot += a * g;
                    }
                }
            }
            for (sum, product) in scratch.sums.iter().zip(&mut scratch.products) {
                domain
                    .inverse_into(sum, product, &mut scratch.workspace)
                    .map_err(|_| internal("integer adjoint auxiliary transform"))?;
            }
            let lift = |value: Auxiliary| -> Result<F, AkitaError> {
                let value = value
                    .to_u128_checked()
                    .ok_or_else(|| internal("auxiliary lift"))?;
                let (negative, magnitude) = if value > modulus / 2 {
                    (true, modulus - value)
                } else {
                    (false, value)
                };
                if magnitude > output_bound {
                    return Err(internal("integer adjoint lift bound"));
                }
                let magnitude = F::from_u128(magnitude);
                Ok(if negative { -magnitude } else { magnitude })
            };
            let [low, high] = &scratch.products;
            let mut h = [F::zero(); D];
            for (s, slot) in h.iter_mut().enumerate() {
                let at = |product: &[Auxiliary]| {
                    product
                        .get(D - 1 + s)
                        .copied()
                        .ok_or_else(|| internal("integer adjoint product"))
                };
                *slot = lift(at(low)?)? + limb_scale * lift(at(high)?)?;
            }
            Ok(h)
        };
        #[cfg(feature = "parallel")]
        let output = (0..columns)
            .into_par_iter()
            .map_init(|| Scratch::new(&domain), column)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let output = (0..columns).map(|j| column(&mut scratch, j)).collect();
        output
    }
}
