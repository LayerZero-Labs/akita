//! Exact coefficient adjoints in `Y^D + middle*Y^(D/2) + 1`.
//!
//! A row extends to `g(l) = <rho, rem(Y^l)>` for `l < 2D-1`.
//! The Hankel product `h(s) = sum_k v[k] g(s+k)` is the middle of
//! `g * reverse(v)`, whose full length would be `3D-2`. Splitting `g`
//! gives two products of length `2D-1`, with no roots of unity required.

use core::marker::PhantomData;

use akita_error::AkitaError;

use jolt_algebra::checked;
use jolt_algebra::Field;

use super::TrinomialModulus;
use crate::offset_eq::eq_eval_at_index;

mod integer;
mod trace;

/// An internal bound failure: unreachable for inputs that passed validation.
fn internal(what: &str) -> AkitaError {
    AkitaError::Internal(format!("trinomial adjoint: {what} out of bounds"))
}

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
            let high = *g
                .get(l - D / 2)
                .ok_or_else(|| internal("extended row index"))?;
            let low = *g.get(l - D).ok_or_else(|| internal("extended row index"))?;
            *g.get_mut(l).ok_or_else(|| internal("extended row index"))? =
                if M::MIDDLE_COEFFICIENT < 0 {
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
        let (g0, tail) = g
            .split_at_checked(D)
            .ok_or_else(|| internal("extended row split"))?;
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
            *slot = *low
                .get(D - 1 + s)
                .ok_or_else(|| internal("low convolution index"))?;
            if s > 0 {
                *slot += *high
                    .get(s - 1)
                    .ok_or_else(|| internal("high convolution index"))?;
            }
        }
        Ok(h)
    }

    /// Compute [`Self::apply`] for `v[k] = eq(point, k)`, `k < D`, without
    /// convolutions.
    ///
    /// `point` has `ceil(log2(D))` coordinates in little-endian bit order.
    /// The set bits of `D` split `0..D` into aligned blocks `b + [0, 2^m)`,
    /// on which `v[b + j] = c_b * eq(point[..m], j)`. On one block the
    /// correlation `X_m(s) = sum_j eq(point[..m], j) g(b + s + j)` follows
    /// from `X_0(t) = g(b + t)` and
    /// `X_(l+1)(t) = X_l(t) + point[l] * (X_l(t + 2^l) - X_l(t))`,
    /// one multiplication per entry and level: 18,505 multiplications
    /// at `D = 648`, against about 155,000 for the two convolutions.
    pub fn apply_equality<F: Field>(&self, rho: &[F], point: &[F]) -> Result<Vec<F>, AkitaError> {
        let bits = checked::ceil_log2(D)
            .ok_or_else(|| AkitaError::InvalidInput("trinomial adjoint size overflow".into()))?;
        if point.len() != bits {
            return Err(AkitaError::InvalidSize {
                expected: bits,
                actual: point.len(),
            });
        }
        let g = self.extend_row(rho)?;
        let mut h = field_buffer(D)?;
        let mut block = Vec::new();
        for m in (0..=bits).rev().filter(|&m| (D >> m) & 1 == 1) {
            let start = D >> (m + 1) << (m + 1);
            let width = 1usize << m;
            let window = g
                .get(start..start + D + width - 1)
                .ok_or_else(|| internal("equality block window"))?;
            block.clear();
            block.extend_from_slice(window);
            for (level, &coordinate) in point.iter().take(m).enumerate() {
                let stride = 1usize << level;
                let live = block.len() - stride;
                for t in 0..live {
                    let next = *block
                        .get(t + stride)
                        .ok_or_else(|| internal("equality block level"))?;
                    let slot = block
                        .get_mut(t)
                        .ok_or_else(|| internal("equality block level"))?;
                    *slot += coordinate * (next - *slot);
                }
                block.truncate(live);
            }
            let scale =
                point
                    .iter()
                    .enumerate()
                    .skip(m)
                    .fold(F::one(), |acc, (bit, &coordinate)| {
                        acc * if (start >> bit) & 1 == 1 {
                            coordinate
                        } else {
                            F::one() - coordinate
                        }
                    });
            for (slot, &value) in h.iter_mut().zip(&block) {
                *slot += scale * value;
            }
        }
        Ok(h)
    }
}

/// Ordinary exact convolution of equally sized, nonempty field vectors.
///
/// Uses Karatsuba with a schoolbook leaf of at most 16 entries. Odd lengths
/// split into floor/ceiling halves; only the half sums are zero-padded.
/// One output buffer and one checked scratch arena serve the entire recursive
/// tree. Invalid lengths, size overflow, or allocation failure return an error.
pub fn exact_convolution<F: Field>(lhs: &[F], rhs: &[F]) -> Result<Vec<F>, AkitaError> {
    if lhs.is_empty() || lhs.len() != rhs.len() {
        return Err(AkitaError::InvalidInput(
            "convolution needs equal nonempty lengths".into(),
        ));
    }
    let length = checked::product([2, lhs.len()])
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| AkitaError::InvalidInput("convolution size overflow".into()))?;
    let scratch_len = karatsuba_scratch_len(lhs.len())?;
    let mut output = field_buffer(length)?;
    let mut scratch = field_buffer(scratch_len)?;
    karatsuba(lhs, rhs, &mut output, &mut scratch)?;
    Ok(output)
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
    #[cfg(test)]
    if len > 0 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
    }
    Ok(values)
}

// Recursive operands always have equal positive lengths, established by the
// public boundary and preserved by the three splits below.
struct KaratsubaSplit {
    half: usize,
    long: usize,
    short_output_len: usize,
    long_output_len: usize,
    high_offset: usize,
    local_fields: usize,
}

impl KaratsubaSplit {
    fn new(n: usize) -> Result<Self, AkitaError> {
        let half = n / 2;
        let long = n - half;
        let high_offset =
            checked::product([2, half]).ok_or_else(|| internal("Karatsuba output offset"))?;
        let short_output_len = high_offset
            .checked_sub(1)
            .ok_or_else(|| internal("Karatsuba short product length"))?;
        let long_output_len = checked::product([2, long])
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| internal("Karatsuba long product length"))?;
        let local_fields = checked::sum([
            long,
            long,
            short_output_len,
            long_output_len,
            long_output_len,
        ])
        .ok_or_else(|| internal("Karatsuba scratch length"))?;
        Ok(Self {
            half,
            long,
            short_output_len,
            long_output_len,
            high_offset,
            local_fields,
        })
    }
}

// Each child reuses the same arena after its sibling returns. The longer child
// bounds the shorter child's workspace: every local size is monotone in n.
fn karatsuba_scratch_len(mut n: usize) -> Result<usize, AkitaError> {
    let mut fields = 0;
    while n > 16 {
        let split = KaratsubaSplit::new(n)?;
        fields = checked::sum([fields, split.local_fields])
            .ok_or_else(|| internal("Karatsuba scratch length"))?;
        n = split.long;
    }
    Ok(fields)
}

fn karatsuba<F: Field>(
    lhs: &[F],
    rhs: &[F],
    output: &mut [F],
    scratch: &mut [F],
) -> Result<(), AkitaError> {
    output.fill(F::zero());
    let n = lhs.len();
    if n <= 16 {
        for (i, &a) in lhs.iter().enumerate() {
            for (j, &b) in rhs.iter().enumerate() {
                #[cfg(test)]
                MULTIPLICATIONS.with(|count| count.set(count.get() + 1));
                *output
                    .get_mut(i + j)
                    .ok_or_else(|| internal("schoolbook output index"))? += a * b;
            }
        }
        return Ok(());
    }
    let split = KaratsubaSplit::new(n)?;
    let (a0, a1) = lhs
        .split_at_checked(split.half)
        .ok_or_else(|| internal("Karatsuba split"))?;
    let (b0, b1) = rhs
        .split_at_checked(split.half)
        .ok_or_else(|| internal("Karatsuba split"))?;
    let (a_sum, tail) = scratch
        .split_at_mut_checked(split.long)
        .ok_or_else(|| internal("Karatsuba operand scratch"))?;
    let (b_sum, tail) = tail
        .split_at_mut_checked(split.long)
        .ok_or_else(|| internal("Karatsuba operand scratch"))?;
    let (low, tail) = tail
        .split_at_mut_checked(split.short_output_len)
        .ok_or_else(|| internal("Karatsuba product scratch"))?;
    let (high, tail) = tail
        .split_at_mut_checked(split.long_output_len)
        .ok_or_else(|| internal("Karatsuba product scratch"))?;
    let (middle, nested) = tail
        .split_at_mut_checked(split.long_output_len)
        .ok_or_else(|| internal("Karatsuba product scratch"))?;
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
    karatsuba(a0, b0, low, nested)?;
    karatsuba(a1, b1, high, nested)?;
    karatsuba(a_sum, b_sum, middle, nested)?;
    for (slot, &value) in middle.iter_mut().zip(low.iter()) {
        *slot -= value;
    }
    for (slot, &value) in middle.iter_mut().zip(high.iter()) {
        *slot -= value;
    }
    for (offset, values) in [
        (0, &*low),
        (split.half, &*middle),
        (split.high_offset, &*high),
    ] {
        for (i, &value) in values.iter().enumerate() {
            *output
                .get_mut(offset + i)
                .ok_or_else(|| internal("Karatsuba output index"))? += value;
        }
    }
    Ok(())
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
        Ok(Self {
            h: adjoint.apply_equality(rho, r_k)?,
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
    static ALLOCATIONS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
#[test]
fn full_degree_adjoint_multiplication_count() {
    use super::MinusTrinomial;
    use jolt_algebra::{Prime128Offset275, Ring};
    let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new().expect("valid degree");
    let rho = vec![Prime128Offset275::from_u64(7); 648];
    let v = vec![Prime128Offset275::from_u64(11); 648];
    MULTIPLICATIONS.with(|count| count.set(0));
    ALLOCATIONS.with(|count| count.set(0));
    let h = adjoint.apply(&rho, &v).expect("valid vectors");
    let count = MULTIPLICATIONS.with(core::cell::Cell::get);
    assert_eq!(h.len(), 648);
    assert_eq!(count, 154_872);
    assert_eq!(ALLOCATIONS.with(core::cell::Cell::get), 8);
    let dense = 648usize.pow(2);
    assert_eq!(dense, 419_904);
    assert!(count < dense);
    eprintln!("one D=648 adjoint: {count} field multiplications; dense: {dense}");
}
