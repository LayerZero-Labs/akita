//! D-agnostic flat vector storage with typed ring-element views.
//!
//! [`FlatMatrix`] stores ring elements as raw field elements in a single
//! contiguous 1D vector, independent of any ring dimension. Each role
//! (A, B, D) interprets a prefix of this vector as its own matrix with
//! role-specific `(num_rows, num_cols)` dimensions.
//!
//! A [`RingMatrixView`] borrows a prefix of the flat data and interprets it
//! as a `rows × cols` matrix of `CyclotomicRing<F, D>` elements, enabling
//! the same underlying vector to serve multiple roles with different shapes.

use akita_algebra::CyclotomicRing;
use akita_field::FieldCore;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use std::io::{Read, Write};

/// Flat 1D vector of field elements, independent of ring dimension.
///
/// Stores `total_ring_elements * gen_ring_dim` contiguous field elements.
/// Each role matrix (A, B, D) views a prefix of this vector reshaped into
/// its own `(num_rows, num_cols)` dimensions via [`RingMatrixView`].
///
/// Any prefix of a uniformly random vector is uniformly random, so role matrices
/// derived from prefixes of the same flat vector are binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatMatrix<F: FieldCore> {
    data: Vec<F>,
    /// Ring dimension used when generating (D_max).
    gen_ring_dim: usize,
}

impl<F: FieldCore> FlatMatrix<F> {
    /// Total number of ring elements at the generation dimension.
    #[inline]
    pub fn total_ring_elements(&self) -> usize {
        if self.gen_ring_dim == 0 {
            0
        } else {
            self.data.len() / self.gen_ring_dim
        }
    }

    /// Ring dimension used during generation.
    #[inline]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    /// Total number of ring elements when viewed at dimension D.
    #[inline]
    pub fn total_ring_elements_at<const D: usize>(&self) -> usize {
        debug_assert!(D > 0 && self.gen_ring_dim.is_multiple_of(D));
        self.total_ring_elements() * (self.gen_ring_dim / D)
    }

    /// Build from pre-flattened field-element data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a multiple of `gen_ring_dim`.
    pub fn from_flat_data(data: Vec<F>, gen_ring_dim: usize) -> Self {
        debug_assert!(
            gen_ring_dim > 0 && data.len().is_multiple_of(gen_ring_dim),
            "data length {} must be a positive multiple of gen_ring_dim={}",
            data.len(),
            gen_ring_dim,
        );
        Self { data, gen_ring_dim }
    }

    /// Build from a flat slice of ring elements.
    pub fn from_ring_slice<const D: usize>(elements: &[CyclotomicRing<F, D>]) -> Self {
        let mut data = Vec::with_capacity(elements.len() * D);
        for ring_elem in elements {
            data.extend_from_slice(&ring_elem.coeffs);
        }
        Self {
            data,
            gen_ring_dim: D,
        }
    }

    /// Create a typed matrix view at ring dimension D with the given shape.
    ///
    /// The view interprets the first `num_rows * num_cols` ring elements
    /// (at dimension D) as a `num_rows × num_cols` matrix.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not divide `gen_ring_dim` or if the requested
    /// shape exceeds the available data.
    pub fn ring_view<const D: usize>(
        &self,
        num_rows: usize,
        num_cols: usize,
    ) -> RingMatrixView<'_, F, D> {
        assert!(D > 0, "ring dimension must be positive");
        assert!(
            self.gen_ring_dim.is_multiple_of(D),
            "D={D} does not divide gen_ring_dim={}",
            self.gen_ring_dim
        );
        let total_at_d = self.total_ring_elements_at::<D>();
        let needed = num_rows * num_cols;
        assert!(
            needed <= total_at_d,
            "requested {num_rows}×{num_cols}={needed} ring elements at D={D}, \
             but only {total_at_d} available"
        );
        let field_len = needed * D;
        RingMatrixView {
            data: &self.data[..field_len],
            num_rows,
            num_cols,
        }
    }
}

impl<F: FieldCore + Valid> Valid for FlatMatrix<F> {
    fn check(&self) -> Result<(), SerializationError> {
        for f in &self.data {
            f.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore> AkitaSerialize for FlatMatrix<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.total_ring_elements()
            .serialize_with_mode(&mut writer, compress)?;
        self.gen_ring_dim
            .serialize_with_mode(&mut writer, compress)?;
        for f in &self.data {
            f.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        2 * std::mem::size_of::<usize>()
            + self
                .data
                .iter()
                .map(|f| f.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F: FieldCore + Valid> AkitaDeserialize for FlatMatrix<F> {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let total_ring = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let gen_ring_dim = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let total_fields = total_ring * gen_ring_dim;
        let mut data = Vec::with_capacity(total_fields);
        for _ in 0..total_fields {
            data.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self { data, gen_ring_dim };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Typed read-only view of a [`FlatMatrix`] prefix at a specific ring
/// dimension D, interpreted as a `num_rows × num_cols` matrix.
///
/// Provides zero-copy access to rows as `&[CyclotomicRing<F, D>]` by
/// transmuting the underlying `&[F]` slice (safe because `CyclotomicRing`
/// is `#[repr(transparent)]` over `[F; D]`).
#[derive(Debug, Clone, Copy)]
pub struct RingMatrixView<'a, F: FieldCore, const D: usize> {
    data: &'a [F],
    num_rows: usize,
    num_cols: usize,
}

impl<'a, F: FieldCore, const D: usize> RingMatrixView<'a, F, D> {
    /// Number of rows in the view.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of ring-element columns per row.
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Borrow a single row as a slice of ring elements (zero-copy).
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows`.
    #[inline]
    pub fn row(&self, row: usize) -> &'a [CyclotomicRing<F, D>] {
        assert!(row < self.num_rows, "row {row} out of bounds");
        let row_field_len = self.num_cols * D;
        let start = row * row_field_len;
        let field_slice = &self.data[start..start + row_field_len];
        // SAFETY: CyclotomicRing<F, D> is #[repr(transparent)] over [F; D],
        // so a contiguous &[F] of length num_cols*D has the same layout as
        // &[CyclotomicRing<F, D>] of length num_cols.
        unsafe {
            std::slice::from_raw_parts(
                field_slice.as_ptr() as *const CyclotomicRing<F, D>,
                self.num_cols,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::fields::Prime128Offset275;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Prime128Offset275;

    #[test]
    fn ring_view_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let rows = 3usize;
        let cols = 5usize;
        let elements: Vec<CyclotomicRing<F, 64>> = (0..rows * cols)
            .map(|_| CyclotomicRing::random(&mut rng))
            .collect();

        let flat = FlatMatrix::from_ring_slice(&elements);
        assert_eq!(flat.total_ring_elements(), rows * cols);
        assert_eq!(flat.gen_ring_dim(), 64);

        let view = flat.ring_view::<64>(rows, cols);
        assert_eq!(view.num_rows(), rows);
        assert_eq!(view.num_cols(), cols);

        for r in 0..rows {
            let view_row = view.row(r);
            for c in 0..cols {
                assert_eq!(view_row[c], elements[r * cols + c]);
            }
        }
    }

    #[test]
    fn ring_view_at_smaller_d() {
        let mut rng = StdRng::seed_from_u64(99);
        let total = 8usize;
        let elements: Vec<CyclotomicRing<F, 64>> = (0..total)
            .map(|_| CyclotomicRing::random(&mut rng))
            .collect();

        let flat = FlatMatrix::from_ring_slice(&elements);
        assert_eq!(flat.total_ring_elements_at::<32>(), total * 2);

        let view32 = flat.ring_view::<32>(2, total);
        assert_eq!(view32.num_rows(), 2);
        assert_eq!(view32.num_cols(), total);

        let view64 = flat.ring_view::<64>(2, 4);
        for r in 0..2 {
            let row64 = view64.row(r);
            let row32 = view32.row(r);
            for (j, orig_ring) in row64.iter().enumerate() {
                let lo = &row32[j * 2];
                let hi = &row32[j * 2 + 1];
                assert_eq!(&orig_ring.coeffs[..32], lo.coefficients());
                assert_eq!(&orig_ring.coeffs[32..], hi.coefficients());
            }
        }
    }

    #[test]
    fn different_role_views_from_same_flat() {
        let mut rng = StdRng::seed_from_u64(7);
        let total = 64usize;
        let elements: Vec<CyclotomicRing<F, 64>> = (0..total)
            .map(|_| CyclotomicRing::random(&mut rng))
            .collect();

        let flat = FlatMatrix::from_ring_slice(&elements);

        let view_a = flat.ring_view::<64>(2, 16);
        let view_b = flat.ring_view::<64>(4, 8);

        assert_eq!(view_a.row(0)[0], elements[0]);
        assert_eq!(view_a.row(1)[0], elements[16]);
        assert_eq!(view_b.row(0)[0], elements[0]);
        assert_eq!(view_b.row(1)[0], elements[8]);
    }
}
