//! Exact coefficient adjoints in `Y^D + middle*Y^(D/2) + 1`.
//!
//! A row extends to `g(l) = <rho, rem(Y^l)>` for `l < 2D-1`.
//! The Hankel product `h(s) = sum_k v[k] g(s+k)` is the middle of
//! `g * reverse(v)`, whose full length would be `3D-2`. Splitting `g`
//! gives two products of length `2D-1`, with no roots of unity required.

use core::marker::PhantomData;

use akita_error::{checked, AkitaError};
use jolt_field::Field;

use super::TrinomialModulus;
use crate::offset_eq::eq_eval_at_index;

/// A checked coefficient-adjoint shape; the modulus sign is part of its type.
#[derive(Clone, Copy, Debug)]
pub struct TrinomialAdjoint<const D: usize, M: TrinomialModulus> {
    product_len: usize,
    modulus: PhantomData<M>,
}

impl<const D: usize, M: TrinomialModulus> TrinomialAdjoint<D, M> {
    /// Validate the positive even degree and its convolution size.
    pub fn new() -> Result<Self, AkitaError> {
        if D == 0 || !D.is_multiple_of(2) {
            return Err(AkitaError::InvalidInput(
                "trinomial adjoint degree must be positive and even".into(),
            ));
        }
        let product_len = checked::product([2, D])
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| AkitaError::InvalidInput("trinomial adjoint size overflow".into()))?;
        Ok(Self {
            product_len,
            modulus: PhantomData,
        })
    }

    /// Extend a length-`D` row through signed reduction in linear time.
    ///
    /// Above degree `D-1`, `g(l) = -middle*g(l-D/2) - g(l-D)`.
    /// No field multiplications or individual monomial reductions are needed.
    pub fn extend_row<F: Field>(&self, rho: &[F]) -> Result<Vec<F>, AkitaError> {
        if rho.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: rho.len(),
            });
        }
        let mut g = field_buffer(self.product_len)?;
        for (slot, &value) in g.iter_mut().zip(rho) {
            *slot = value;
        }
        for l in D..self.product_len {
            let high = *g.get(l - D / 2).ok_or(AkitaError::InvalidProof)?;
            let low = *g.get(l - D).ok_or(AkitaError::InvalidProof)?;
            *g.get_mut(l).ok_or(AkitaError::InvalidProof)? = if M::MIDDLE_COEFFICIENT < 0 {
                high - low
            } else {
                -high - low
            };
        }
        Ok(g)
    }

    /// Compute `h(s) = sum_k v[k] g(s+k)` using two exact convolutions.
    ///
    /// Both inputs must have length `D`. Scratch and output occupy `O(D)`
    /// field elements, independent of the number of setup matrix columns.
    pub fn apply<F: Field>(&self, rho: &[F], v: &[F]) -> Result<Vec<F>, AkitaError> {
        if v.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: v.len(),
            });
        }
        let g = self.extend_row(rho)?;
        let (g0, tail) = g.split_at_checked(D).ok_or(AkitaError::InvalidProof)?;
        let mut g1 = field_buffer(D)?;
        for (slot, &value) in g1.iter_mut().zip(tail) {
            *slot = value;
        }
        let mut reversed = field_buffer(D)?;
        for (slot, &value) in reversed.iter_mut().zip(v.iter().rev()) {
            *slot = value;
        }
        let low = exact_convolution(g0, &reversed)?;
        let high = exact_convolution(&g1, &reversed)?;
        let mut h = field_buffer(D)?;
        for (s, slot) in h.iter_mut().enumerate() {
            *slot = *low.get(D - 1 + s).ok_or(AkitaError::InvalidProof)?;
            if s > 0 {
                *slot += *high.get(s - 1).ok_or(AkitaError::InvalidProof)?;
            }
        }
        Ok(h)
    }
}

/// Ordinary exact convolution of equally sized, nonempty field vectors.
///
/// Uses Karatsuba with a schoolbook leaf of at most 16 entries. Odd lengths
/// split into floor/ceiling halves; only the half sums are zero-padded.
/// Invalid lengths, size overflow, or allocation failure return an error.
pub fn exact_convolution<F: Field>(lhs: &[F], rhs: &[F]) -> Result<Vec<F>, AkitaError> {
    if lhs.is_empty() || lhs.len() != rhs.len() {
        return Err(AkitaError::InvalidInput(
            "convolution needs equal nonempty lengths".into(),
        ));
    }
    karatsuba(lhs, rhs)
}

// Fallible allocation checks both byte size and allocator capacity before
// initialization. Every caller validates the input shape before reaching here.
fn field_buffer<F: Field>(len: usize) -> Result<Vec<F>, AkitaError> {
    let bytes = checked::product([len, core::mem::size_of::<F>()])
        .ok_or_else(|| AkitaError::InvalidInput("adjoint allocation size overflow".into()))?;
    if bytes > isize::MAX as usize {
        return Err(AkitaError::InvalidInput(
            "adjoint allocation exceeds address space".into(),
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("adjoint allocation failed".into()))?;
    values.resize(len, F::zero());
    Ok(values)
}

// Recursive operands always have equal positive lengths, established by the
// public boundary and preserved by the three splits below.
fn karatsuba<F: Field>(lhs: &[F], rhs: &[F]) -> Result<Vec<F>, AkitaError> {
    let n = lhs.len();
    let length = checked::product([2, n])
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| AkitaError::InvalidInput("convolution size overflow".into()))?;
    let mut output = field_buffer(length)?;
    if n <= 16 {
        for (i, &a) in lhs.iter().enumerate() {
            for (j, &b) in rhs.iter().enumerate() {
                #[cfg(test)]
                MULTIPLICATIONS.with(|count| count.set(count.get() + 1));
                *output.get_mut(i + j).ok_or(AkitaError::InvalidProof)? += a * b;
            }
        }
        return Ok(output);
    }
    let half = n / 2;
    let (a0, a1) = lhs.split_at_checked(half).ok_or(AkitaError::InvalidProof)?;
    let (b0, b1) = rhs.split_at_checked(half).ok_or(AkitaError::InvalidProof)?;
    let mut a_sum = field_buffer(n - half)?;
    let mut b_sum = field_buffer(n - half)?;
    for (slot, &value) in a_sum.iter_mut().zip(a1) {
        *slot = value;
    }
    for (slot, &value) in a_sum.iter_mut().zip(a0) {
        *slot += value;
    }
    for (slot, &value) in b_sum.iter_mut().zip(b1) {
        *slot = value;
    }
    for (slot, &value) in b_sum.iter_mut().zip(b0) {
        *slot += value;
    }
    let low = karatsuba(a0, b0)?;
    let high = karatsuba(a1, b1)?;
    let mut middle = karatsuba(&a_sum, &b_sum)?;
    for (slot, &value) in middle.iter_mut().zip(&low) {
        *slot -= value;
    }
    for (slot, &value) in middle.iter_mut().zip(&high) {
        *slot -= value;
    }
    for (offset, values) in [(0, &low), (half, &middle), (2 * half, &high)] {
        for (i, &value) in values.iter().enumerate() {
            *output.get_mut(offset + i).ok_or(AkitaError::InvalidProof)? += value;
        }
    }
    Ok(output)
}

/// One row block of setup weights, factored as `eq(r_j,j) * h(s)`.
///
/// Stores only `D` coefficients and borrows the column point. Equality points
/// use little-endian bit order. Padded coefficient slots are omitted from the
/// Hankel sum; matrix column and coefficient indices must address real slots.
#[derive(Debug)]
pub struct SetupAdjointWeight<'a, F: Field> {
    h: Vec<F>,
    r_j: &'a [F],
    columns: usize,
}

impl<'a, F: Field> SetupAdjointWeight<'a, F> {
    /// Build one row factor with `v[k] = eq(r_k,k)` for `k < D`.
    ///
    /// Points must have exactly `ceil(log2(columns))` and `ceil(log2(D))`
    /// coordinates, respectively. No table proportional to `columns` is built.
    pub fn new<const D: usize, M: TrinomialModulus>(
        adjoint: &TrinomialAdjoint<D, M>,
        rho: &[F],
        r_j: &'a [F],
        r_k: &[F],
        columns: usize,
    ) -> Result<Self, AkitaError> {
        for (size, point) in [(columns, r_j), (D, r_k)] {
            let expected = checked::ceil_log2(size)
                .ok_or_else(|| AkitaError::InvalidInput("invalid setup adjoint domain".into()))?;
            if point.len() != expected {
                return Err(AkitaError::InvalidSize {
                    expected,
                    actual: point.len(),
                });
            }
        }
        if rho.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: rho.len(),
            });
        }
        let mut v = field_buffer(D)?;
        for (k, slot) in v.iter_mut().enumerate() {
            *slot = eq_eval_at_index(r_k, k);
        }
        Ok(Self {
            h: adjoint.apply(rho, &v)?,
            r_j,
            columns,
        })
    }

    /// The coefficient factor `h`, with exactly the ring degree entries.
    pub fn coefficients(&self) -> &[F] {
        &self.h
    }

    /// Evaluate one real setup coefficient weight without expanding the matrix.
    pub fn weight(&self, j: usize, s: usize) -> Result<F, AkitaError> {
        if j >= self.columns {
            return Err(AkitaError::InvalidInput(
                "setup adjoint column out of range".into(),
            ));
        }
        let h = self.h.get(s).ok_or_else(|| {
            AkitaError::InvalidInput("setup adjoint coefficient out of range".into())
        })?;
        Ok(eq_eval_at_index(self.r_j, j) * *h)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
std::thread_local! {
    static MULTIPLICATIONS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
#[test]
fn full_degree_adjoint_multiplication_count() {
    use super::MinusTrinomial;
    use jolt_field::{Prime128Offset275, Ring};
    let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new().expect("valid degree");
    let rho = vec![Prime128Offset275::from_u64(7); 648];
    let v = vec![Prime128Offset275::from_u64(11); 648];
    MULTIPLICATIONS.with(|count| count.set(0));
    let h = adjoint.apply(&rho, &v).expect("valid vectors");
    let count = MULTIPLICATIONS.with(core::cell::Cell::get);
    assert_eq!(h.len(), 648);
    assert_eq!(count, 154_872);
    let dense = 648usize.pow(2);
    assert_eq!(dense, 419_904);
    assert!(count < dense);
    eprintln!("one D=648 adjoint: {count} field multiplications; dense: {dense}");
}
