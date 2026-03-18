//! Prototype `k=32` partial-split NTT for `F_p[X]/(X^64 + 1)`.
//!
//! For primes `p` with `64 | (p - 1)` but `128 ∤ (p - 1)`, the usual size-64
//! negacyclic NTT does not exist over `F_p`. We can still exploit the
//! `k=32` split:
//!
//! `X^64 + 1 = \prod_{j=0}^{31} (X^2 - r_j)`,
//!
//! where `r_j` ranges over the roots of `Y^32 + 1` with `Y = X^2`.
//!
//! Writing a ring element as `a(X) = a_0(Y) + X a_1(Y)`, multiplication in the
//! quotient ring reduces to:
//!
//! 1. two size-32 negacyclic transforms over `F_p` (for `a_0` and `a_1`),
//! 2. pointwise multiplication in the quadratic factors `F_p[X]/(X^2 - r_j)`,
//! 3. two inverse size-32 negacyclic transforms.
//!
//! This module is intentionally a prototype helper for benchmarking the
//! split-native path against the existing multi-CRT NTT implementation.

use super::CyclotomicRing;
use crate::algebra::PackedField;
use crate::{CanonicalField, FieldCore};
use std::array::from_fn;

const CLASS_D: usize = 32;
const RING_D: usize = 64;

/// Cached `k=32` split-domain representation of a `D=64` ring element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartialSplitEval32<F: CanonicalField> {
    even: [F; CLASS_D],
    odd: [F; CLASS_D],
}

impl<F: CanonicalField> PartialSplitEval32<F> {
    /// The additive identity in split-evaluation form.
    pub fn zero() -> Self {
        Self {
            even: [F::zero(); CLASS_D],
            odd: [F::zero(); CLASS_D],
        }
    }

    /// Convert a `D=64` ring element into the cached split-domain form.
    pub fn from_ring(split: &PartialSplitNtt32<F>, ring: &CyclotomicRing<F, RING_D>) -> Self {
        let (mut even, mut odd) = split_even_odd(ring.coefficients());
        split.forward_class_pair(&mut even, &mut odd);
        Self { even, odd }
    }

    /// Convert centered `i8` coefficients into the cached split-domain form.
    pub fn from_i8(split: &PartialSplitNtt32<F>, coeffs: &[i8; RING_D]) -> Self {
        let (even_i8, odd_i8) = split_even_odd_i8(coeffs);
        let (even, odd) = split.forward_class_i8_pair(&even_i8, &odd_i8);
        Self { even, odd }
    }

    /// Convert the cached split-domain form back to coefficient form.
    pub fn to_ring(&self, split: &PartialSplitNtt32<F>) -> CyclotomicRing<F, RING_D> {
        let mut even = self.even;
        let mut odd = self.odd;
        split.inverse_class_pair(&mut even, &mut odd);
        merge_even_odd(even, odd)
    }

    /// Pointwise multiplication in split-evaluation form.
    pub fn pointwise_mul(&self, rhs: &Self, split: &PartialSplitNtt32<F>) -> Self {
        let mut even = [F::zero(); CLASS_D];
        let mut odd = [F::zero(); CLASS_D];
        for i in 0..CLASS_D {
            (even[i], odd[i]) = mul_quadratic_karatsuba(
                self.even[i],
                self.odd[i],
                rhs.even[i],
                rhs.odd[i],
                split.eval_roots[i],
            );
        }
        Self { even, odd }
    }

    /// Add another split-evaluation element in place.
    pub fn add_assign(&mut self, rhs: &Self) {
        for i in 0..CLASS_D {
            self.even[i] += rhs.even[i];
            self.odd[i] += rhs.odd[i];
        }
    }

    /// Accumulate a pointwise product directly into `self`.
    pub fn add_mul_assign(&mut self, lhs: &Self, rhs: &Self, split: &PartialSplitNtt32<F>) {
        for i in 0..CLASS_D {
            let (even, odd) = mul_quadratic_karatsuba(
                lhs.even[i],
                lhs.odd[i],
                rhs.even[i],
                rhs.odd[i],
                split.eval_roots[i],
            );
            self.even[i] += even;
            self.odd[i] += odd;
        }
    }
}

/// Packed cached `k=32` split-domain representation of one or more `D=64`
/// ring elements, grouped across SIMD lanes.
#[derive(Clone, Copy)]
pub struct PackedPartialSplitEval32<PF: PackedField> {
    even: [PF; CLASS_D],
    odd: [PF; CLASS_D],
}

impl<PF: PackedField> PackedPartialSplitEval32<PF> {
    /// Number of scalar lanes grouped into this packed value.
    pub const WIDTH: usize = PF::WIDTH;

    /// The additive identity in packed split-evaluation form.
    pub fn zero() -> Self {
        let zero = PF::broadcast(PF::Scalar::zero());
        Self {
            even: [zero; CLASS_D],
            odd: [zero; CLASS_D],
        }
    }

    /// Pack one scalar split-domain value per lane.
    pub fn from_fn<FN>(mut f: FN) -> Self
    where
        PF::Scalar: CanonicalField,
        FN: FnMut(usize) -> PartialSplitEval32<PF::Scalar>,
    {
        let lanes: Vec<PartialSplitEval32<PF::Scalar>> = (0..PF::WIDTH).map(&mut f).collect();
        Self::from_chunk(&lanes)
    }

    /// Broadcast one scalar split-domain value to every SIMD lane.
    pub fn broadcast(value: &PartialSplitEval32<PF::Scalar>) -> Self
    where
        PF::Scalar: CanonicalField,
    {
        Self {
            even: from_fn(|i| PF::broadcast(value.even[i])),
            odd: from_fn(|i| PF::broadcast(value.odd[i])),
        }
    }

    /// Pack a scalar slice into SIMD groups, returning any scalar suffix.
    pub fn pack_slice_with_suffix(
        buf: &[PartialSplitEval32<PF::Scalar>],
    ) -> (Vec<Self>, &[PartialSplitEval32<PF::Scalar>])
    where
        PF::Scalar: CanonicalField,
    {
        let split = buf.len() - (buf.len() % PF::WIDTH);
        let (packed, suffix) = buf.split_at(split);
        let out = packed
            .chunks_exact(PF::WIDTH)
            .map(Self::from_chunk)
            .collect();
        (out, suffix)
    }

    /// Accumulate a packed pointwise product directly into `self`.
    pub fn add_mul_assign(&mut self, lhs: &Self, rhs: &Self, split: &PartialSplitNtt32<PF::Scalar>)
    where
        PF::Scalar: CanonicalField,
    {
        for i in 0..CLASS_D {
            let p0 = lhs.even[i] * rhs.even[i];
            let p1 = lhs.odd[i] * rhs.odd[i];
            let p2 = (lhs.even[i] + lhs.odd[i]) * (rhs.even[i] + rhs.odd[i]);
            let r = PF::broadcast(split.eval_roots[i]);
            self.even[i] = self.even[i] + p0 + r * p1;
            self.odd[i] = self.odd[i] + p2 - p0 - p1;
        }
    }

    /// Convert the packed split-domain value back to coefficient form.
    pub fn to_rings(
        &self,
        split: &PartialSplitNtt32<PF::Scalar>,
    ) -> Vec<CyclotomicRing<PF::Scalar, RING_D>>
    where
        PF::Scalar: CanonicalField,
    {
        let mut even = self.even;
        let mut odd = self.odd;
        split.inverse_class_pair_packed(&mut even, &mut odd);
        (0..PF::WIDTH)
            .map(|lane| {
                let even_lane = from_fn(|i| even[i].extract(lane));
                let odd_lane = from_fn(|i| odd[i].extract(lane));
                merge_even_odd(even_lane, odd_lane)
            })
            .collect()
    }

    fn from_chunk(chunk: &[PartialSplitEval32<PF::Scalar>]) -> Self
    where
        PF::Scalar: CanonicalField,
    {
        assert_eq!(
            chunk.len(),
            PF::WIDTH,
            "chunk length {} must equal packed width {}",
            chunk.len(),
            PF::WIDTH
        );
        Self {
            even: from_fn(|i| PF::from_fn(|lane| chunk[lane].even[i])),
            odd: from_fn(|i| PF::from_fn(|lane| chunk[lane].odd[i])),
        }
    }
}

/// Pre-broadcast split-NTT tables for packed SIMD execution.
#[derive(Clone, Copy)]
pub struct PackedPartialSplitNtt32<PF: PackedField> {
    eval_roots: [PF; CLASS_D],
    inv_stage_roots: [PF; CLASS_D],
    d_inv_psi_inv: [PF; CLASS_D],
}

impl<PF: PackedField> PackedPartialSplitNtt32<PF> {
    /// Accumulate a packed pointwise product directly into `acc`.
    pub fn add_mul_assign(
        &self,
        acc: &mut PackedPartialSplitEval32<PF>,
        lhs: &PackedPartialSplitEval32<PF>,
        rhs: &PackedPartialSplitEval32<PF>,
    ) where
        PF::Scalar: CanonicalField,
    {
        for i in 0..CLASS_D {
            let p0 = lhs.even[i] * rhs.even[i];
            let p1 = lhs.odd[i] * rhs.odd[i];
            let p2 = (lhs.even[i] + lhs.odd[i]) * (rhs.even[i] + rhs.odd[i]);
            acc.even[i] = acc.even[i] + p0 + self.eval_roots[i] * p1;
            acc.odd[i] = acc.odd[i] + p2 - p0 - p1;
        }
    }

    /// Append the coefficient-form outputs for every packed lane.
    pub fn append_rings(
        &self,
        eval: &PackedPartialSplitEval32<PF>,
        out: &mut Vec<CyclotomicRing<PF::Scalar, RING_D>>,
    ) where
        PF::Scalar: CanonicalField,
    {
        let mut even = eval.even;
        let mut odd = eval.odd;
        inverse_cyclic_dit_pair_prebroadcast(&mut even, &mut odd, &self.inv_stage_roots);
        for i in 0..CLASS_D {
            even[i] = even[i] * self.d_inv_psi_inv[i];
            odd[i] = odd[i] * self.d_inv_psi_inv[i];
        }
        out.reserve(PF::WIDTH);
        for lane in 0..PF::WIDTH {
            let even_lane = from_fn(|i| even[i].extract(lane));
            let odd_lane = from_fn(|i| odd[i].extract(lane));
            out.push(merge_even_odd(even_lane, odd_lane));
        }
    }

    /// Convert the packed split-domain value back to coefficient form.
    pub fn to_rings(
        &self,
        eval: &PackedPartialSplitEval32<PF>,
    ) -> Vec<CyclotomicRing<PF::Scalar, RING_D>>
    where
        PF::Scalar: CanonicalField,
    {
        let mut out = Vec::with_capacity(PF::WIDTH);
        self.append_rings(eval, &mut out);
        out
    }
}

/// Precomputed `k=32` split-NTT data for `F_p[X]/(X^64 + 1)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialSplitNtt32<F: CanonicalField> {
    fwd_stage_roots: [F; CLASS_D],
    inv_stage_roots: [F; CLASS_D],
    psi_pows: [F; CLASS_D],
    d_inv_psi_inv: [F; CLASS_D],
    eval_roots: [F; CLASS_D],
    cyclic_fwd_stage_roots: [F; RING_D],
    cyclic_inv_stage_roots: [F; RING_D],
    cyclic_d_inv: F,
    centered_i8_lut: [F; 256],
}

impl<F: CanonicalField> PartialSplitNtt32<F> {
    /// Build the size-32 negacyclic transform tables from the field modulus.
    ///
    /// # Panics
    ///
    /// Panics if `64 ∤ (p - 1)`, or if no primitive `64`-th root is found by
    /// the small-generator search.
    pub fn compute() -> Self {
        let q = modulus::<F>();
        assert!(
            (q - 1) % (2 * CLASS_D as u128) == 0,
            "64 must divide q - 1 for the k=32 split transform"
        );

        let psi = find_primitive_root_2d::<F>(q, CLASS_D);
        let omega = psi.square();
        let omega_inv = omega
            .inv()
            .expect("primitive root must be invertible in the base field");
        let psi_inv = psi
            .inv()
            .expect("primitive root must be invertible in the base field");
        let d_inv = F::from_u64(CLASS_D as u64)
            .inv()
            .expect("transform size must be invertible in the base field");
        let cyclic_d_inv = F::from_u64(RING_D as u64)
            .inv()
            .expect("transform size must be invertible in the base field");

        let psi_pows = powers(psi);
        let psi_inv_pows = powers(psi_inv);
        let d_inv_psi_inv = from_fn(|i| d_inv * psi_inv_pows[i]);
        let eval_roots = {
            let mut natural = [F::zero(); CLASS_D];
            let mut cur = psi;
            for root in &mut natural {
                *root = cur;
                cur = cur * omega;
            }
            let log_n = CLASS_D.trailing_zeros() as usize;
            from_fn(|i| natural[bit_reverse_index(i, log_n)])
        };

        let mut fwd_stage_roots = [F::zero(); CLASS_D];
        let mut inv_stage_roots = [F::zero(); CLASS_D];
        let mut len = 1usize;
        while len < CLASS_D {
            let exp = (CLASS_D / (2 * len)) as u128;
            fwd_stage_roots[len] = pow_field(omega, exp);
            inv_stage_roots[len] = pow_field(omega_inv, exp);
            len *= 2;
        }

        let mut cyclic_fwd_stage_roots = [F::zero(); RING_D];
        let mut cyclic_inv_stage_roots = [F::zero(); RING_D];
        let mut len = 1usize;
        while len < RING_D {
            let exp = (RING_D / (2 * len)) as u128;
            cyclic_fwd_stage_roots[len] = pow_field(psi, exp);
            cyclic_inv_stage_roots[len] = pow_field(psi_inv, exp);
            len *= 2;
        }

        let centered_i8_lut = from_fn(|idx| F::from_i64(idx as i64 - 128));

        Self {
            fwd_stage_roots,
            inv_stage_roots,
            psi_pows,
            d_inv_psi_inv,
            eval_roots,
            cyclic_fwd_stage_roots,
            cyclic_inv_stage_roots,
            cyclic_d_inv,
            centered_i8_lut,
        }
    }

    /// Roots `r_j` of `Y^32 + 1` in the same slot order used by the transform.
    pub fn eval_roots(&self) -> &[F; CLASS_D] {
        &self.eval_roots
    }

    /// Broadcast the split-domain tables into a packed SIMD representation.
    pub fn packed<PF: PackedField<Scalar = F>>(&self) -> PackedPartialSplitNtt32<PF> {
        PackedPartialSplitNtt32 {
            eval_roots: from_fn(|i| PF::broadcast(self.eval_roots[i])),
            inv_stage_roots: from_fn(|i| PF::broadcast(self.inv_stage_roots[i])),
            d_inv_psi_inv: from_fn(|i| PF::broadcast(self.d_inv_psi_inv[i])),
        }
    }

    /// Forward size-32 negacyclic transform on the class ring `F_p[Y]/(Y^32+1)`.
    pub fn forward_class(&self, coeffs: &mut [F; CLASS_D]) {
        for (coeff, psi) in coeffs.iter_mut().zip(self.psi_pows.iter()) {
            *coeff = *coeff * *psi;
        }
        forward_cyclic_dif(coeffs, &self.fwd_stage_roots);
    }

    /// Forward size-32 negacyclic transform on two class polynomials at once.
    pub fn forward_class_pair(&self, lhs: &mut [F; CLASS_D], rhs: &mut [F; CLASS_D]) {
        for ((lhs_i, rhs_i), psi) in lhs.iter_mut().zip(rhs.iter_mut()).zip(self.psi_pows.iter()) {
            *lhs_i = *lhs_i * *psi;
            *rhs_i = *rhs_i * *psi;
        }
        forward_cyclic_dif_pair(lhs, rhs, &self.fwd_stage_roots);
    }

    /// Forward size-32 negacyclic transform for centered `i8` coefficients.
    ///
    /// This uses a full `[-128, 127]` lookup table to avoid repeated
    /// `i8 -> field` conversions on the hot path.
    pub fn forward_class_i8(&self, coeffs: &[i8; CLASS_D]) -> [F; CLASS_D] {
        let mut out = from_fn(|i| self.i8_to_field(coeffs[i]));
        self.forward_class(&mut out);
        out
    }

    /// Forward size-32 negacyclic transform for two centered-`i8` classes.
    pub fn forward_class_i8_pair(
        &self,
        lhs: &[i8; CLASS_D],
        rhs: &[i8; CLASS_D],
    ) -> ([F; CLASS_D], [F; CLASS_D]) {
        let mut out_lhs = from_fn(|i| self.i8_to_field(lhs[i]));
        let mut out_rhs = from_fn(|i| self.i8_to_field(rhs[i]));
        self.forward_class_pair(&mut out_lhs, &mut out_rhs);
        (out_lhs, out_rhs)
    }

    /// Inverse size-32 negacyclic transform back to class-ring coefficients.
    pub fn inverse_class(&self, evals: &mut [F; CLASS_D]) {
        inverse_cyclic_dit(evals, &self.inv_stage_roots);
        for (eval, fused) in evals.iter_mut().zip(self.d_inv_psi_inv.iter()) {
            *eval = *eval * *fused;
        }
    }

    /// Inverse size-32 negacyclic transform on two slot vectors at once.
    pub fn inverse_class_pair(&self, lhs: &mut [F; CLASS_D], rhs: &mut [F; CLASS_D]) {
        inverse_cyclic_dit_pair(lhs, rhs, &self.inv_stage_roots);
        for ((lhs_i, rhs_i), fused) in lhs
            .iter_mut()
            .zip(rhs.iter_mut())
            .zip(self.d_inv_psi_inv.iter())
        {
            *lhs_i = *lhs_i * *fused;
            *rhs_i = *rhs_i * *fused;
        }
    }

    /// Inverse size-32 negacyclic transform on two packed slot vectors.
    pub fn inverse_class_pair_packed<PF: PackedField<Scalar = F>>(
        &self,
        lhs: &mut [PF; CLASS_D],
        rhs: &mut [PF; CLASS_D],
    ) {
        inverse_cyclic_dit_pair_packed(lhs, rhs, &self.inv_stage_roots);
        for ((lhs_i, rhs_i), fused) in lhs
            .iter_mut()
            .zip(rhs.iter_mut())
            .zip(self.d_inv_psi_inv.iter())
        {
            let fused = PF::broadcast(*fused);
            *lhs_i = *lhs_i * fused;
            *rhs_i = *rhs_i * fused;
        }
    }

    /// Forward size-64 cyclic transform over `F_p[X]/(X^64 - 1)`.
    pub fn forward_cyclic_ring(&self, coeffs: &mut [F; RING_D]) {
        forward_cyclic_dif64(coeffs, &self.cyclic_fwd_stage_roots);
    }

    /// Inverse size-64 cyclic transform over `F_p[X]/(X^64 - 1)`.
    pub fn inverse_cyclic_ring(&self, evals: &mut [F; RING_D]) {
        inverse_cyclic_dit64(evals, &self.cyclic_inv_stage_roots);
        for value in evals.iter_mut() {
            *value = *value * self.cyclic_d_inv;
        }
    }

    /// Multiply two `D=64` ring elements using the `k=32` partial split.
    pub fn multiply_d64(
        &self,
        lhs: &CyclotomicRing<F, RING_D>,
        rhs: &CyclotomicRing<F, RING_D>,
    ) -> CyclotomicRing<F, RING_D> {
        let (mut lhs_even, mut lhs_odd) = split_even_odd(lhs.coefficients());
        let (mut rhs_even, mut rhs_odd) = split_even_odd(rhs.coefficients());

        self.forward_class_pair(&mut lhs_even, &mut lhs_odd);
        self.forward_class_pair(&mut rhs_even, &mut rhs_odd);

        let mut out_even = [F::zero(); CLASS_D];
        let mut out_odd = [F::zero(); CLASS_D];
        for i in 0..CLASS_D {
            (out_even[i], out_odd[i]) = mul_quadratic_karatsuba(
                lhs_even[i],
                lhs_odd[i],
                rhs_even[i],
                rhs_odd[i],
                self.eval_roots[i],
            );
        }

        self.inverse_class_pair(&mut out_even, &mut out_odd);
        merge_even_odd(out_even, out_odd)
    }

    /// Multiply a full field-valued `D=64` ring element by centered `i8`
    /// coefficients using the `k=32` partial split.
    pub fn multiply_d64_rhs_i8(
        &self,
        lhs: &CyclotomicRing<F, RING_D>,
        rhs: &[i8; RING_D],
    ) -> CyclotomicRing<F, RING_D> {
        let (mut lhs_even, mut lhs_odd) = split_even_odd(lhs.coefficients());
        let (rhs_even, rhs_odd) = split_even_odd_i8(rhs);

        self.forward_class_pair(&mut lhs_even, &mut lhs_odd);
        let (rhs_even_eval, rhs_odd_eval) = self.forward_class_i8_pair(&rhs_even, &rhs_odd);

        let mut out_even = [F::zero(); CLASS_D];
        let mut out_odd = [F::zero(); CLASS_D];
        for i in 0..CLASS_D {
            (out_even[i], out_odd[i]) = mul_quadratic_karatsuba(
                lhs_even[i],
                lhs_odd[i],
                rhs_even_eval[i],
                rhs_odd_eval[i],
                self.eval_roots[i],
            );
        }

        self.inverse_class_pair(&mut out_even, &mut out_odd);
        merge_even_odd(out_even, out_odd)
    }

    /// Multiply two `D=64` coefficient arrays modulo `X^64 - 1`.
    pub fn multiply_cyclic_d64(
        &self,
        lhs: &CyclotomicRing<F, RING_D>,
        rhs: &CyclotomicRing<F, RING_D>,
    ) -> [F; RING_D] {
        let mut lhs_eval = *lhs.coefficients();
        let mut rhs_eval = *rhs.coefficients();
        self.forward_cyclic_ring(&mut lhs_eval);
        self.forward_cyclic_ring(&mut rhs_eval);
        let mut out = [F::zero(); RING_D];
        for i in 0..RING_D {
            out[i] = lhs_eval[i] * rhs_eval[i];
        }
        self.inverse_cyclic_ring(&mut out);
        out
    }

    /// Compute the high-half quotient `H` such that
    /// `lhs * rhs = H * X^64 + reduced`, with `deg(H) < 64`.
    pub fn unreduced_quotient_d64(
        &self,
        lhs: &CyclotomicRing<F, RING_D>,
        rhs: &CyclotomicRing<F, RING_D>,
    ) -> CyclotomicRing<F, RING_D> {
        let neg = self.multiply_d64(lhs, rhs);
        let cyc = self.multiply_cyclic_d64(lhs, rhs);
        let neg_coeffs = neg.coefficients();
        let quotient = from_fn(|i| (cyc[i] - neg_coeffs[i]) * F::TWO_INV);
        CyclotomicRing::from_coefficients(quotient)
    }

    #[inline]
    fn i8_to_field(&self, coeff: i8) -> F {
        self.centered_i8_lut[(coeff as i16 + 128) as usize]
    }
}

fn modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

fn powers<F: CanonicalField>(base: F) -> [F; CLASS_D] {
    let mut out = [F::zero(); CLASS_D];
    let mut cur = F::one();
    for value in &mut out {
        *value = cur;
        cur = cur * base;
    }
    out
}

fn pow_field<F: FieldCore>(mut base: F, mut exp: u128) -> F {
    let mut acc = F::one();
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base;
        }
        exp >>= 1;
        if exp > 0 {
            base = base.square();
        }
    }
    acc
}

fn find_primitive_root_2d<F: CanonicalField>(q: u128, d: usize) -> F {
    let half = (q - 1) / 2;
    let exp = (q - 1) / (2 * d as u128);
    let minus_one = -F::one();

    for candidate in 2u64..(1 << 20) {
        let probe = F::from_u64(candidate);
        if pow_field(probe, half) == minus_one {
            let psi = pow_field(probe, exp);
            debug_assert!(pow_field(psi, d as u128) == minus_one, "psi^D != -1");
            debug_assert!(pow_field(psi, 2 * d as u128) == F::one(), "psi^(2D) != 1");
            return psi;
        }
    }

    panic!("no primitive {}-th root found", 2 * d);
}

fn split_even_odd<F: CanonicalField>(coeffs: &[F; RING_D]) -> ([F; CLASS_D], [F; CLASS_D]) {
    let even = from_fn(|i| coeffs[2 * i]);
    let odd = from_fn(|i| coeffs[2 * i + 1]);
    (even, odd)
}

fn split_even_odd_i8(coeffs: &[i8; RING_D]) -> ([i8; CLASS_D], [i8; CLASS_D]) {
    let even = from_fn(|i| coeffs[2 * i]);
    let odd = from_fn(|i| coeffs[2 * i + 1]);
    (even, odd)
}

#[inline(always)]
fn mul_quadratic_karatsuba<F: CanonicalField>(a0: F, a1: F, b0: F, b1: F, r: F) -> (F, F) {
    let p0 = a0 * b0;
    let p1 = a1 * b1;
    let p2 = (a0 + a1) * (b0 + b1);
    let c0 = p0 + r * p1;
    let c1 = p2 - p0 - p1;
    (c0, c1)
}

fn merge_even_odd<F: CanonicalField>(
    even: [F; CLASS_D],
    odd: [F; CLASS_D],
) -> CyclotomicRing<F, RING_D> {
    CyclotomicRing::from_coefficients(from_fn(
        |i| {
            if i % 2 == 0 {
                even[i / 2]
            } else {
                odd[i / 2]
            }
        },
    ))
}

fn forward_cyclic_dif<F: CanonicalField>(a: &mut [F; CLASS_D], stage_roots: &[F; CLASS_D]) {
    let mut len = CLASS_D / 2;
    while len > 0 {
        let step = stage_roots[len];
        let mut start = 0usize;
        while start < CLASS_D {
            let mut w = F::one();
            for j in 0..len {
                let u = a[start + j];
                let v = a[start + j + len];
                a[start + j] = u + v;
                a[start + j + len] = (u - v) * w;
                w = w * step;
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

fn forward_cyclic_dif_pair<F: CanonicalField>(
    lhs: &mut [F; CLASS_D],
    rhs: &mut [F; CLASS_D],
    stage_roots: &[F; CLASS_D],
) {
    let mut len = CLASS_D / 2;
    while len > 0 {
        let step = stage_roots[len];
        let mut start = 0usize;
        while start < CLASS_D {
            let mut w = F::one();
            for j in 0..len {
                let lhs_u = lhs[start + j];
                let lhs_v = lhs[start + j + len];
                lhs[start + j] = lhs_u + lhs_v;
                lhs[start + j + len] = (lhs_u - lhs_v) * w;

                let rhs_u = rhs[start + j];
                let rhs_v = rhs[start + j + len];
                rhs[start + j] = rhs_u + rhs_v;
                rhs[start + j + len] = (rhs_u - rhs_v) * w;

                w = w * step;
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

fn inverse_cyclic_dit<F: CanonicalField>(a: &mut [F; CLASS_D], stage_roots: &[F; CLASS_D]) {
    let mut len = 1usize;
    while len < CLASS_D {
        let step = stage_roots[len];
        let mut start = 0usize;
        while start < CLASS_D {
            let mut w = F::one();
            for j in 0..len {
                let u = a[start + j];
                let v = a[start + j + len] * w;
                a[start + j] = u + v;
                a[start + j + len] = u - v;
                w = w * step;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

fn inverse_cyclic_dit_pair<F: CanonicalField>(
    lhs: &mut [F; CLASS_D],
    rhs: &mut [F; CLASS_D],
    stage_roots: &[F; CLASS_D],
) {
    let mut len = 1usize;
    while len < CLASS_D {
        let step = stage_roots[len];
        let mut start = 0usize;
        while start < CLASS_D {
            let mut w = F::one();
            for j in 0..len {
                let lhs_u = lhs[start + j];
                let lhs_v = lhs[start + j + len] * w;
                lhs[start + j] = lhs_u + lhs_v;
                lhs[start + j + len] = lhs_u - lhs_v;

                let rhs_u = rhs[start + j];
                let rhs_v = rhs[start + j + len] * w;
                rhs[start + j] = rhs_u + rhs_v;
                rhs[start + j + len] = rhs_u - rhs_v;

                w = w * step;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

fn inverse_cyclic_dit_pair_packed<F: CanonicalField, PF: PackedField<Scalar = F>>(
    lhs: &mut [PF; CLASS_D],
    rhs: &mut [PF; CLASS_D],
    stage_roots: &[F; CLASS_D],
) {
    let mut len = 1usize;
    while len < CLASS_D {
        let step = PF::broadcast(stage_roots[len]);
        let mut start = 0usize;
        while start < CLASS_D {
            let mut w = PF::broadcast(F::one());
            for j in 0..len {
                let lhs_u = lhs[start + j];
                let lhs_v = lhs[start + j + len] * w;
                lhs[start + j] = lhs_u + lhs_v;
                lhs[start + j + len] = lhs_u - lhs_v;

                let rhs_u = rhs[start + j];
                let rhs_v = rhs[start + j + len] * w;
                rhs[start + j] = rhs_u + rhs_v;
                rhs[start + j + len] = rhs_u - rhs_v;

                w = w * step;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

fn inverse_cyclic_dit_pair_prebroadcast<PF: PackedField>(
    lhs: &mut [PF; CLASS_D],
    rhs: &mut [PF; CLASS_D],
    stage_roots: &[PF; CLASS_D],
) where
    PF::Scalar: CanonicalField,
{
    let mut len = 1usize;
    while len < CLASS_D {
        let step = stage_roots[len];
        let mut start = 0usize;
        while start < CLASS_D {
            let mut w = PF::broadcast(PF::Scalar::one());
            for j in 0..len {
                let lhs_u = lhs[start + j];
                let lhs_v = lhs[start + j + len] * w;
                lhs[start + j] = lhs_u + lhs_v;
                lhs[start + j + len] = lhs_u - lhs_v;

                let rhs_u = rhs[start + j];
                let rhs_v = rhs[start + j + len] * w;
                rhs[start + j] = rhs_u + rhs_v;
                rhs[start + j + len] = rhs_u - rhs_v;

                w = w * step;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

fn forward_cyclic_dif64<F: CanonicalField>(a: &mut [F; RING_D], stage_roots: &[F; RING_D]) {
    let mut len = RING_D / 2;
    while len > 0 {
        let step = stage_roots[len];
        let mut start = 0usize;
        while start < RING_D {
            let mut w = F::one();
            for j in 0..len {
                let u = a[start + j];
                let v = a[start + j + len];
                a[start + j] = u + v;
                a[start + j + len] = (u - v) * w;
                w = w * step;
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

fn inverse_cyclic_dit64<F: CanonicalField>(a: &mut [F; RING_D], stage_roots: &[F; RING_D]) {
    let mut len = 1usize;
    while len < RING_D {
        let step = stage_roots[len];
        let mut start = 0usize;
        while start < RING_D {
            let mut w = F::one();
            for j in 0..len {
                let u = a[start + j];
                let v = a[start + j + len] * w;
                a[start + j] = u + v;
                a[start + j + len] = u - v;
                w = w * step;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

fn bit_reverse_index(idx: usize, log_n: usize) -> usize {
    idx.reverse_bits() >> (usize::BITS as usize - log_n)
}
