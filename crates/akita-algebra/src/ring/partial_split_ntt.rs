//! Prototype `k=16` partial-split NTT for `F_p[X]/(X^32 + 1)`.
//!
//! For primes `p` with `32 | (p - 1)` but `64 ∤ (p - 1)`, the usual size-32
//! negacyclic NTT does not exist over `F_p`. We can still exploit the
//! `k=16` split:
//!
//! `X^32 + 1 = \prod_{j=0}^{15} (X^2 - r_j)`,
//!
//! where `r_j` ranges over the roots of `Y^16 + 1` with `Y = X^2`.
//!
//! Writing a ring element as `a(X) = a_0(Y) + X a_1(Y)`, multiplication in the
//! quotient ring reduces to:
//!
//! 1. two size-16 negacyclic transforms over `F_p` (for `a_0` and `a_1`),
//! 2. pointwise multiplication in the quadratic factors `F_p[X]/(X^2 - r_j)`,
//! 3. two inverse size-16 negacyclic transforms.
//!
//! This module is intentionally a prototype helper for benchmarking the
//! split-native path against the existing multi-CRT NTT implementation.

use super::CyclotomicRing;
use crate::{CanonicalField, FieldCore, HalvingField};
use akita_field::packed::PackedField;
use akita_field::Zero;
use core::ops::{Add, Mul, Sub};
use std::array::from_fn;

const CLASS_D: usize = 16;
const RING_D: usize = 32;
const CLASS_LOG_D: usize = CLASS_D.trailing_zeros() as usize;
const CENTERED_I8_LUT_OFFSET: i16 = 128;

/// Cached `k=16` split-domain representation of a `D=32` ring element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartialSplitEval16<F: CanonicalField> {
    even: [F; CLASS_D],
    odd: [F; CLASS_D],
}

impl<F: CanonicalField> PartialSplitEval16<F> {
    /// The additive identity in split-evaluation form.
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            even: [F::zero(); CLASS_D],
            odd: [F::zero(); CLASS_D],
        }
    }

    /// Convert a `D=32` ring element into the cached split-domain form.
    #[inline(always)]
    pub fn from_ring(split: &PartialSplitNtt16<F>, ring: &CyclotomicRing<F, RING_D>) -> Self {
        let (mut even, mut odd) = split_even_odd(ring.coefficients());
        split.forward_class_pair(&mut even, &mut odd);
        Self { even, odd }
    }

    /// Convert centered `i8` coefficients into the cached split-domain form.
    #[inline(always)]
    pub fn from_i8(split: &PartialSplitNtt16<F>, coeffs: &[i8; RING_D]) -> Self {
        let (even_i8, odd_i8) = split_even_odd_i8(coeffs);
        let (even, odd) = split.forward_class_i8_pair(&even_i8, &odd_i8);
        Self { even, odd }
    }

    /// Convert the cached split-domain form back to coefficient form.
    #[inline(always)]
    pub fn to_ring(&self, split: &PartialSplitNtt16<F>) -> CyclotomicRing<F, RING_D> {
        let mut even = self.even;
        let mut odd = self.odd;
        split.inverse_class_pair(&mut even, &mut odd);
        merge_even_odd(even, odd)
    }

    /// Pointwise multiplication in split-evaluation form.
    #[inline(always)]
    pub fn pointwise_mul(&self, rhs: &Self, split: &PartialSplitNtt16<F>) -> Self {
        let mut even = [F::zero(); CLASS_D];
        let mut odd = [F::zero(); CLASS_D];
        mul_quadratic_pairs(
            &mut even,
            &mut odd,
            &self.even,
            &self.odd,
            &rhs.even,
            &rhs.odd,
            &split.eval_roots,
        );
        Self { even, odd }
    }

    /// Accumulate a pointwise product directly into `self`.
    #[inline(always)]
    pub fn add_mul_assign(&mut self, lhs: &Self, rhs: &Self, split: &PartialSplitNtt16<F>) {
        add_mul_quadratic_pairs(
            &mut self.even,
            &mut self.odd,
            &lhs.even,
            &lhs.odd,
            &rhs.even,
            &rhs.odd,
            &split.eval_roots,
        );
    }
}

/// Packed cached `k=16` split-domain representation of one or more `D=32`
/// ring elements, grouped across SIMD lanes.
#[derive(Clone, Copy)]
pub struct PackedPartialSplitEval16<PF: PackedField> {
    even: [PF; CLASS_D],
    odd: [PF; CLASS_D],
}

impl<PF: PackedField> PackedPartialSplitEval16<PF> {
    /// Number of scalar lanes grouped into this packed value.
    pub const WIDTH: usize = PF::WIDTH;

    /// The additive identity in packed split-evaluation form.
    #[inline(always)]
    pub fn zero() -> Self {
        let zero = PF::broadcast(PF::Scalar::zero());
        Self {
            even: [zero; CLASS_D],
            odd: [zero; CLASS_D],
        }
    }

    /// Pack one scalar split-domain value per lane.
    #[inline(always)]
    pub fn from_fn<FN>(mut f: FN) -> Self
    where
        PF::Scalar: CanonicalField,
        FN: FnMut(usize) -> PartialSplitEval16<PF::Scalar>,
    {
        let lanes: Vec<PartialSplitEval16<PF::Scalar>> = (0..PF::WIDTH).map(&mut f).collect();
        Self::from_chunk(&lanes)
    }

    /// Broadcast one scalar split-domain value to every SIMD lane.
    #[inline(always)]
    pub fn broadcast(value: &PartialSplitEval16<PF::Scalar>) -> Self
    where
        PF::Scalar: CanonicalField,
    {
        Self {
            even: from_fn(|i| PF::broadcast(value.even[i])),
            odd: from_fn(|i| PF::broadcast(value.odd[i])),
        }
    }

    #[inline(always)]
    fn from_chunk(chunk: &[PartialSplitEval16<PF::Scalar>]) -> Self
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
pub struct PackedPartialSplitNtt16<PF: PackedField> {
    eval_roots: [PF; CLASS_D],
    inv_twiddles: [PF; CLASS_D],
    d_inv_psi_inv: [PF; CLASS_D],
}

impl<PF: PackedField> PackedPartialSplitNtt16<PF> {
    /// Multiply two packed split-domain values lane-wise.
    #[inline(always)]
    pub fn pointwise_mul(
        &self,
        lhs: &PackedPartialSplitEval16<PF>,
        rhs: &PackedPartialSplitEval16<PF>,
    ) -> PackedPartialSplitEval16<PF>
    where
        PF::Scalar: CanonicalField,
    {
        let zero = PF::broadcast(PF::Scalar::zero());
        let mut even = [zero; CLASS_D];
        let mut odd = [zero; CLASS_D];
        mul_quadratic_pairs(
            &mut even,
            &mut odd,
            &lhs.even,
            &lhs.odd,
            &rhs.even,
            &rhs.odd,
            &self.eval_roots,
        );
        PackedPartialSplitEval16 { even, odd }
    }

    /// Accumulate a packed pointwise product directly into `acc`.
    #[inline(always)]
    pub fn add_mul_assign(
        &self,
        acc: &mut PackedPartialSplitEval16<PF>,
        lhs: &PackedPartialSplitEval16<PF>,
        rhs: &PackedPartialSplitEval16<PF>,
    ) where
        PF::Scalar: CanonicalField,
    {
        add_mul_quadratic_pairs(
            &mut acc.even,
            &mut acc.odd,
            &lhs.even,
            &lhs.odd,
            &rhs.even,
            &rhs.odd,
            &self.eval_roots,
        );
    }

    /// Append the coefficient-form outputs for every packed lane.
    #[inline(always)]
    pub fn append_rings(
        &self,
        eval: &PackedPartialSplitEval16<PF>,
        out: &mut Vec<CyclotomicRing<PF::Scalar, RING_D>>,
    ) where
        PF::Scalar: CanonicalField,
    {
        let mut even = eval.even;
        let mut odd = eval.odd;
        inverse_cyclic_dit_pair_prebroadcast(&mut even, &mut odd, &self.inv_twiddles);
        scale_pair_in_place(&mut even, &mut odd, &self.d_inv_psi_inv);
        out.reserve(PF::WIDTH);
        for lane in 0..PF::WIDTH {
            let even_lane = from_fn(|i| even[i].extract(lane));
            let odd_lane = from_fn(|i| odd[i].extract(lane));
            out.push(merge_even_odd(even_lane, odd_lane));
        }
    }
}

/// Precomputed `k=16` split-NTT data for `F_p[X]/(X^32 + 1)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialSplitNtt16<F: CanonicalField> {
    fwd_twiddles: [F; CLASS_D],
    inv_twiddles: [F; CLASS_D],
    psi_pows: [F; CLASS_D],
    d_inv_psi_inv: [F; CLASS_D],
    eval_roots: [F; CLASS_D],
    cyclic_fwd_twiddles: [F; RING_D],
    cyclic_inv_twiddles: [F; RING_D],
    cyclic_d_inv: F,
    centered_i8_lut: [F; 256],
}

impl<F: CanonicalField> PartialSplitNtt16<F> {
    /// Build the size-16 negacyclic transform tables from the field modulus.
    ///
    /// # Panics
    ///
    /// Panics if `32 ∤ (p - 1)`, or if no primitive `32`-th root is found by
    /// the small-generator search.
    pub fn compute() -> Self {
        let q = modulus::<F>();
        assert!(
            (q - 1).is_multiple_of(2 * CLASS_D as u128),
            "32 must divide q - 1 for the k=16 split transform"
        );

        let psi = find_primitive_root_2d::<F>(q, CLASS_D);
        let omega = psi.square();
        let omega_inv = omega
            .inverse()
            .expect("primitive root must be invertible in the base field");
        let psi_inv = psi
            .inverse()
            .expect("primitive root must be invertible in the base field");
        let d_inv = F::from_u64(CLASS_D as u64)
            .inverse()
            .expect("transform size must be invertible in the base field");
        let cyclic_d_inv = F::from_u64(RING_D as u64)
            .inverse()
            .expect("transform size must be invertible in the base field");

        let psi_pows = powers(psi);
        let psi_inv_pows = powers(psi_inv);
        let d_inv_psi_inv = from_fn(|i| d_inv * psi_inv_pows[i]);
        let eval_roots = {
            let mut natural = [F::zero(); CLASS_D];
            let mut cur = psi;
            for root in &mut natural {
                *root = cur;
                cur *= omega;
            }
            from_fn(|i| natural[bit_reverse_index(i, CLASS_LOG_D)])
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
        let fwd_twiddles = expand_stage_roots(&fwd_stage_roots);
        let inv_twiddles = expand_stage_roots(&inv_stage_roots);

        let mut cyclic_fwd_stage_roots = [F::zero(); RING_D];
        let mut cyclic_inv_stage_roots = [F::zero(); RING_D];
        let mut len = 1usize;
        while len < RING_D {
            let exp = (RING_D / (2 * len)) as u128;
            cyclic_fwd_stage_roots[len] = pow_field(psi, exp);
            cyclic_inv_stage_roots[len] = pow_field(psi_inv, exp);
            len *= 2;
        }
        let cyclic_fwd_twiddles = expand_stage_roots(&cyclic_fwd_stage_roots);
        let cyclic_inv_twiddles = expand_stage_roots(&cyclic_inv_stage_roots);

        let centered_i8_lut = from_fn(|idx| F::from_i64(idx as i64 - 128));

        Self {
            fwd_twiddles,
            inv_twiddles,
            psi_pows,
            d_inv_psi_inv,
            eval_roots,
            cyclic_fwd_twiddles,
            cyclic_inv_twiddles,
            cyclic_d_inv,
            centered_i8_lut,
        }
    }

    /// Roots `r_j` of `Y^16 + 1` in the same slot order used by the transform.
    #[inline(always)]
    pub fn eval_roots(&self) -> &[F; CLASS_D] {
        &self.eval_roots
    }

    /// Broadcast the split-domain tables into a packed SIMD representation.
    #[inline(always)]
    pub fn packed<PF: PackedField<Scalar = F>>(&self) -> PackedPartialSplitNtt16<PF> {
        PackedPartialSplitNtt16 {
            eval_roots: from_fn(|i| PF::broadcast(self.eval_roots[i])),
            inv_twiddles: from_fn(|i| PF::broadcast(self.inv_twiddles[i])),
            d_inv_psi_inv: from_fn(|i| PF::broadcast(self.d_inv_psi_inv[i])),
        }
    }

    /// Forward size-16 negacyclic transform on the class ring `F_p[Y]/(Y^16+1)`.
    #[inline(always)]
    pub fn forward_class(&self, coeffs: &mut [F; CLASS_D]) {
        for (coeff, psi) in coeffs.iter_mut().zip(self.psi_pows.iter()) {
            *coeff *= *psi;
        }
        forward_cyclic_dif(coeffs, &self.fwd_twiddles);
    }

    /// Forward size-16 negacyclic transform on two class polynomials at once.
    #[inline(always)]
    pub fn forward_class_pair(&self, lhs: &mut [F; CLASS_D], rhs: &mut [F; CLASS_D]) {
        scale_pair_in_place(lhs, rhs, &self.psi_pows);
        forward_cyclic_dif_pair(lhs, rhs, &self.fwd_twiddles);
    }

    /// Forward size-16 negacyclic transform for two centered-`i8` classes.
    #[inline(always)]
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

    /// Inverse size-16 negacyclic transform on two slot vectors at once.
    #[inline(always)]
    pub fn inverse_class_pair(&self, lhs: &mut [F; CLASS_D], rhs: &mut [F; CLASS_D]) {
        inverse_cyclic_dit_pair(lhs, rhs, &self.inv_twiddles);
        scale_pair_in_place(lhs, rhs, &self.d_inv_psi_inv);
    }

    /// Forward size-32 cyclic transform over `F_p[X]/(X^32 - 1)`.
    #[inline(always)]
    pub fn forward_cyclic_ring(&self, coeffs: &mut [F; RING_D]) {
        forward_cyclic_dif_ring(coeffs, &self.cyclic_fwd_twiddles);
    }

    /// Inverse size-32 cyclic transform over `F_p[X]/(X^32 - 1)`.
    #[inline(always)]
    pub fn inverse_cyclic_ring(&self, evals: &mut [F; RING_D]) {
        inverse_cyclic_dit_ring(evals, &self.cyclic_inv_twiddles);
        for value in evals.iter_mut() {
            *value *= self.cyclic_d_inv;
        }
    }

    /// Multiply two `D=32` ring elements using the `k=16` partial split.
    #[inline(always)]
    pub fn multiply_d32(
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
        mul_quadratic_pairs(
            &mut out_even,
            &mut out_odd,
            &lhs_even,
            &lhs_odd,
            &rhs_even,
            &rhs_odd,
            &self.eval_roots,
        );

        self.inverse_class_pair(&mut out_even, &mut out_odd);
        merge_even_odd(out_even, out_odd)
    }

    /// Multiply a full field-valued `D=32` ring element by centered `i8`
    /// coefficients using the `k=16` partial split.
    #[inline(always)]
    pub fn multiply_d32_rhs_i8(
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
        mul_quadratic_pairs(
            &mut out_even,
            &mut out_odd,
            &lhs_even,
            &lhs_odd,
            &rhs_even_eval,
            &rhs_odd_eval,
            &self.eval_roots,
        );

        self.inverse_class_pair(&mut out_even, &mut out_odd);
        merge_even_odd(out_even, out_odd)
    }

    /// Multiply two `D=32` coefficient arrays modulo `X^32 - 1`.
    #[inline(always)]
    pub fn multiply_cyclic_d32(
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
    /// `lhs * rhs = H * X^32 + reduced`, with `deg(H) < 32`.
    #[inline(always)]
    pub fn unreduced_quotient_d32(
        &self,
        lhs: &CyclotomicRing<F, RING_D>,
        rhs: &CyclotomicRing<F, RING_D>,
    ) -> CyclotomicRing<F, RING_D>
    where
        F: HalvingField,
    {
        let neg = self.multiply_d32(lhs, rhs);
        let cyc = self.multiply_cyclic_d32(lhs, rhs);
        let neg_coeffs = neg.coefficients();
        let quotient = from_fn(|i| (cyc[i] - neg_coeffs[i]).half());
        CyclotomicRing::from_coefficients(quotient)
    }

    #[inline(always)]
    fn i8_to_field(&self, coeff: i8) -> F {
        self.centered_i8_lut[(coeff as i16 + CENTERED_I8_LUT_OFFSET) as usize]
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
        cur *= base;
    }
    out
}

fn expand_stage_roots<F: CanonicalField, const N: usize>(stage_roots: &[F; N]) -> [F; N] {
    let mut twiddles = [F::zero(); N];
    let mut len = 1usize;
    while len < N {
        let base = len - 1;
        let step = stage_roots[len];
        let mut w = F::one();
        for j in 0..len {
            twiddles[base + j] = w;
            w *= step;
        }
        len *= 2;
    }
    twiddles
}

fn pow_field<F: FieldCore>(mut base: F, mut exp: u128) -> F {
    let mut acc = F::one();
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= base;
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

#[inline(always)]
fn split_even_odd<F: CanonicalField>(coeffs: &[F; RING_D]) -> ([F; CLASS_D], [F; CLASS_D]) {
    let even = from_fn(|i| coeffs[2 * i]);
    let odd = from_fn(|i| coeffs[2 * i + 1]);
    (even, odd)
}

#[inline(always)]
fn split_even_odd_i8(coeffs: &[i8; RING_D]) -> ([i8; CLASS_D], [i8; CLASS_D]) {
    let even = from_fn(|i| coeffs[2 * i]);
    let odd = from_fn(|i| coeffs[2 * i + 1]);
    (even, odd)
}

#[inline(always)]
fn mul_quadratic_karatsuba<T>(a0: T, a1: T, b0: T, b1: T, r: T) -> (T, T)
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    let p0 = a0 * b0;
    let p1 = a1 * b1;
    let p2 = (a0 + a1) * (b0 + b1);
    let c0 = p0 + r * p1;
    let c1 = p2 - p0 - p1;
    (c0, c1)
}

#[inline(always)]
fn mul_quadratic_pairs<T>(
    out_even: &mut [T; CLASS_D],
    out_odd: &mut [T; CLASS_D],
    lhs_even: &[T; CLASS_D],
    lhs_odd: &[T; CLASS_D],
    rhs_even: &[T; CLASS_D],
    rhs_odd: &[T; CLASS_D],
    roots: &[T; CLASS_D],
) where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    for i in 0..CLASS_D {
        (out_even[i], out_odd[i]) =
            mul_quadratic_karatsuba(lhs_even[i], lhs_odd[i], rhs_even[i], rhs_odd[i], roots[i]);
    }
}

#[inline(always)]
fn add_mul_quadratic_pairs<T>(
    acc_even: &mut [T; CLASS_D],
    acc_odd: &mut [T; CLASS_D],
    lhs_even: &[T; CLASS_D],
    lhs_odd: &[T; CLASS_D],
    rhs_even: &[T; CLASS_D],
    rhs_odd: &[T; CLASS_D],
    roots: &[T; CLASS_D],
) where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    for i in 0..CLASS_D {
        let (even, odd) =
            mul_quadratic_karatsuba(lhs_even[i], lhs_odd[i], rhs_even[i], rhs_odd[i], roots[i]);
        acc_even[i] = acc_even[i] + even;
        acc_odd[i] = acc_odd[i] + odd;
    }
}

#[inline(always)]
fn scale_pair_in_place<T>(lhs: &mut [T; CLASS_D], rhs: &mut [T; CLASS_D], scales: &[T; CLASS_D])
where
    T: Copy + Mul<Output = T>,
{
    for ((lhs_i, rhs_i), scale) in lhs.iter_mut().zip(rhs.iter_mut()).zip(scales.iter()) {
        *lhs_i = *lhs_i * *scale;
        *rhs_i = *rhs_i * *scale;
    }
}

#[inline(always)]
fn merge_even_odd<F: CanonicalField>(
    even: [F; CLASS_D],
    odd: [F; CLASS_D],
) -> CyclotomicRing<F, RING_D> {
    let mut coeffs = [F::zero(); RING_D];
    for i in 0..CLASS_D {
        coeffs[2 * i] = even[i];
        coeffs[2 * i + 1] = odd[i];
    }
    CyclotomicRing::from_coefficients(coeffs)
}

#[inline(always)]
fn forward_cyclic_dif<F: CanonicalField>(a: &mut [F; CLASS_D], twiddles: &[F; CLASS_D]) {
    let mut len = CLASS_D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < CLASS_D {
            for j in 0..len {
                let w = twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = a[start + j + len];
                a[start + j] = u + v;
                a[start + j + len] = (u - v) * w;
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

#[inline(always)]
fn forward_cyclic_dif_pair<F: CanonicalField>(
    lhs: &mut [F; CLASS_D],
    rhs: &mut [F; CLASS_D],
    twiddles: &[F; CLASS_D],
) {
    let mut len = CLASS_D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < CLASS_D {
            for j in 0..len {
                let w = twiddles[twiddle_base + j];
                let lhs_u = lhs[start + j];
                let lhs_v = lhs[start + j + len];
                lhs[start + j] = lhs_u + lhs_v;
                lhs[start + j + len] = (lhs_u - lhs_v) * w;

                let rhs_u = rhs[start + j];
                let rhs_v = rhs[start + j + len];
                rhs[start + j] = rhs_u + rhs_v;
                rhs[start + j + len] = (rhs_u - rhs_v) * w;
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

#[inline(always)]
fn inverse_cyclic_dit_pair<F: CanonicalField>(
    lhs: &mut [F; CLASS_D],
    rhs: &mut [F; CLASS_D],
    twiddles: &[F; CLASS_D],
) {
    let mut len = 1usize;
    while len < CLASS_D {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < CLASS_D {
            for j in 0..len {
                let w = twiddles[twiddle_base + j];
                let lhs_u = lhs[start + j];
                let lhs_v = lhs[start + j + len] * w;
                lhs[start + j] = lhs_u + lhs_v;
                lhs[start + j + len] = lhs_u - lhs_v;

                let rhs_u = rhs[start + j];
                let rhs_v = rhs[start + j + len] * w;
                rhs[start + j] = rhs_u + rhs_v;
                rhs[start + j + len] = rhs_u - rhs_v;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

#[inline(always)]
fn inverse_cyclic_dit_pair_prebroadcast<PF: PackedField>(
    lhs: &mut [PF; CLASS_D],
    rhs: &mut [PF; CLASS_D],
    twiddles: &[PF; CLASS_D],
) where
    PF::Scalar: CanonicalField,
{
    let mut len = 1usize;
    while len < CLASS_D {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < CLASS_D {
            for j in 0..len {
                let w = twiddles[twiddle_base + j];
                let lhs_u = lhs[start + j];
                let lhs_v = lhs[start + j + len] * w;
                lhs[start + j] = lhs_u + lhs_v;
                lhs[start + j + len] = lhs_u - lhs_v;

                let rhs_u = rhs[start + j];
                let rhs_v = rhs[start + j + len] * w;
                rhs[start + j] = rhs_u + rhs_v;
                rhs[start + j + len] = rhs_u - rhs_v;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

#[inline(always)]
fn forward_cyclic_dif_ring<F: CanonicalField>(a: &mut [F; RING_D], twiddles: &[F; RING_D]) {
    let mut len = RING_D / 2;
    while len > 0 {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < RING_D {
            for j in 0..len {
                let w = twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = a[start + j + len];
                a[start + j] = u + v;
                a[start + j + len] = (u - v) * w;
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

#[inline(always)]
fn inverse_cyclic_dit_ring<F: CanonicalField>(a: &mut [F; RING_D], twiddles: &[F; RING_D]) {
    let mut len = 1usize;
    while len < RING_D {
        let twiddle_base = len - 1;
        let mut start = 0usize;
        while start < RING_D {
            for j in 0..len {
                let w = twiddles[twiddle_base + j];
                let u = a[start + j];
                let v = a[start + j + len] * w;
                a[start + j] = u + v;
                a[start + j + len] = u - v;
            }
            start += 2 * len;
        }
        len *= 2;
    }
}

#[inline(always)]
fn bit_reverse_index(idx: usize, log_n: usize) -> usize {
    idx.reverse_bits() >> (usize::BITS as usize - log_n)
}
