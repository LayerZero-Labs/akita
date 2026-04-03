//! D-agnostic flat matrix storage with typed ring-element views.
//!
//! [`FlatMatrix`] stores matrix entries as raw field elements, independent of
//! any ring dimension. A [`RingMatrixView`] borrows the flat data and
//! interprets it as `CyclotomicRing<F, D>` slices, enabling the same
//! underlying matrix to be viewed at different ring dimensions.

use crate::algebra::CyclotomicRing;
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use std::io::{Read, Write};

/// Row-major matrix of field elements, independent of ring dimension.
///
/// Each row contains `cols_ring * gen_ring_dim` contiguous field elements,
/// where `cols_ring` is the number of ring elements per row at the dimension
/// (`gen_ring_dim`) used when the matrix was generated.
///
/// To view with a smaller ring dimension D' (where D' divides `gen_ring_dim`),
/// each row is re-chunked into `cols_ring * gen_ring_dim / D'` ring elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatMatrix<F: FieldCore> {
    data: Vec<F>,
    num_rows: usize,
    /// Number of ring elements per row at the generation dimension.
    cols_ring: usize,
    /// Ring dimension used when generating (D_max).
    gen_ring_dim: usize,
}

impl<F: FieldCore> FlatMatrix<F> {
    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of ring-element columns at the generation dimension.
    #[inline]
    pub fn cols_ring(&self) -> usize {
        self.cols_ring
    }

    /// Ring dimension used during generation.
    #[inline]
    pub fn gen_ring_dim(&self) -> usize {
        self.gen_ring_dim
    }

    /// Number of field elements per row.
    #[inline]
    pub fn row_field_len(&self) -> usize {
        self.cols_ring * self.gen_ring_dim
    }

    /// Build from pre-flattened field-element data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_rows * cols_ring * gen_ring_dim`.
    pub(crate) fn from_flat_data(
        data: Vec<F>,
        num_rows: usize,
        cols_ring: usize,
        gen_ring_dim: usize,
    ) -> Self {
        debug_assert_eq!(data.len(), num_rows * cols_ring * gen_ring_dim);
        Self {
            data,
            num_rows,
            cols_ring,
            gen_ring_dim,
        }
    }

    /// Build from a `Vec<Vec<CyclotomicRing<F, D>>>`, flattening ring elements
    /// into contiguous field-element storage.
    pub fn from_ring_matrix<const D: usize>(mat: &[Vec<CyclotomicRing<F, D>>]) -> Self {
        let num_rows = mat.len();
        let cols_ring = if num_rows > 0 { mat[0].len() } else { 0 };
        let row_len = cols_ring * D;
        let mut data = Vec::with_capacity(num_rows * row_len);
        for row in mat {
            debug_assert_eq!(row.len(), cols_ring);
            for ring_elem in row {
                data.extend_from_slice(&ring_elem.coeffs);
            }
        }
        Self {
            data,
            num_rows,
            cols_ring,
            gen_ring_dim: D,
        }
    }

    /// Create a typed view at ring dimension D.
    ///
    /// D must divide `gen_ring_dim`. The view re-chunks each row so that
    /// `cols_at_d = cols_ring * gen_ring_dim / D`.
    ///
    /// # Panics
    ///
    /// Panics if `D == 0`, D does not divide `gen_ring_dim`, or the matrix is
    /// empty with inconsistent metadata.
    pub fn view<const D: usize>(&self) -> RingMatrixView<'_, F, D> {
        assert!(D > 0, "ring dimension must be positive");
        assert!(
            self.gen_ring_dim.is_multiple_of(D),
            "D={D} does not divide gen_ring_dim={}",
            self.gen_ring_dim
        );
        let scale = self.gen_ring_dim / D;
        let cols_at_d = self.cols_ring * scale;
        RingMatrixView {
            data: &self.data,
            num_rows: self.num_rows,
            num_cols: cols_at_d,
        }
    }

    /// Borrow the raw field-element data.
    #[inline]
    pub fn raw_data(&self) -> &[F] {
        &self.data
    }

    /// Number of ring-element columns when viewed at dimension D.
    #[inline]
    pub fn num_cols_at<const D: usize>(&self) -> usize {
        debug_assert!(D > 0 && self.gen_ring_dim.is_multiple_of(D));
        self.cols_ring * (self.gen_ring_dim / D)
    }

    /// Borrow a single row as a slice of ring elements at dimension D (zero-copy).
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows` or D does not divide `gen_ring_dim`.
    #[inline]
    pub fn row<const D: usize>(&self, row: usize) -> &[CyclotomicRing<F, D>] {
        assert!(D > 0 && self.gen_ring_dim.is_multiple_of(D));
        assert!(row < self.num_rows, "row {row} out of bounds");
        let row_field_len = self.cols_ring * self.gen_ring_dim;
        let start = row * row_field_len;
        let field_slice = &self.data[start..start + row_field_len];
        let num_cols = row_field_len / D;
        // SAFETY: CyclotomicRing<F, D> is #[repr(transparent)] over [F; D].
        unsafe {
            std::slice::from_raw_parts(
                field_slice.as_ptr() as *const CyclotomicRing<F, D>,
                num_cols,
            )
        }
    }

    /// Whether the matrix has zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Convenience: number of ring-element columns in the first row at dimension D,
    /// or 0 if empty.
    #[inline]
    pub fn first_row_len<const D: usize>(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.num_cols_at::<D>()
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

impl<F: FieldCore> HachiSerialize for FlatMatrix<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.num_rows.serialize_with_mode(&mut writer, compress)?;
        self.cols_ring.serialize_with_mode(&mut writer, compress)?;
        self.gen_ring_dim
            .serialize_with_mode(&mut writer, compress)?;
        for f in &self.data {
            f.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        3 * std::mem::size_of::<usize>()
            + self
                .data
                .iter()
                .map(|f| f.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for FlatMatrix<F> {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let num_rows = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let cols_ring = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let gen_ring_dim = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let total = num_rows * cols_ring * gen_ring_dim;
        let mut data = Vec::with_capacity(total);
        for _ in 0..total {
            data.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self {
            data,
            num_rows,
            cols_ring,
            gen_ring_dim,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Typed read-only view of a [`FlatMatrix`] at a specific ring dimension D.
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

    /// Iterate over all rows.
    pub fn rows(&self) -> impl Iterator<Item = &'a [CyclotomicRing<F, D>]> + '_ {
        (0..self.num_rows).map(move |i| self.row(i))
    }

    /// Take a sub-view: first `n_rows` rows, first `n_cols` ring-element columns.
    ///
    /// This cannot produce a contiguous sub-view because rows are not
    /// contiguous after column truncation. Instead it returns a
    /// [`SubMatrixView`] that copies on access.
    ///
    /// # Panics
    ///
    /// Panics if `n_rows > self.num_rows` or `n_cols > self.num_cols`.
    pub fn submatrix(&self, n_rows: usize, n_cols: usize) -> SubMatrixView<'a, F, D> {
        assert!(n_rows <= self.num_rows);
        assert!(n_cols <= self.num_cols);
        SubMatrixView {
            parent: *self,
            n_rows,
            n_cols,
        }
    }

    /// Collect into the legacy `Vec<Vec<CyclotomicRing<F, D>>>` representation.
    pub fn to_vec_vec(&self) -> Vec<Vec<CyclotomicRing<F, D>>> {
        (0..self.num_rows).map(|i| self.row(i).to_vec()).collect()
    }
}

/// A non-contiguous sub-view that yields column-truncated rows.
#[derive(Debug, Clone, Copy)]
pub struct SubMatrixView<'a, F: FieldCore, const D: usize> {
    parent: RingMatrixView<'a, F, D>,
    n_rows: usize,
    n_cols: usize,
}

impl<'a, F: FieldCore, const D: usize> SubMatrixView<'a, F, D> {
    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of ring-element columns.
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.n_cols
    }

    /// Borrow a row, truncated to `n_cols` ring elements.
    ///
    /// # Panics
    ///
    /// Panics if `row >= n_rows`.
    #[inline]
    pub fn row(&self, row: usize) -> &'a [CyclotomicRing<F, D>] {
        assert!(row < self.n_rows, "row {row} out of bounds");
        &self.parent.row(row)[..self.n_cols]
    }

    /// Iterate over rows.
    pub fn rows(&self) -> impl Iterator<Item = &'a [CyclotomicRing<F, D>]> + '_ {
        (0..self.n_rows).map(move |i| self.row(i))
    }

    /// Collect into the legacy `Vec<Vec<CyclotomicRing<F, D>>>` representation.
    pub fn to_vec_vec(&self) -> Vec<Vec<CyclotomicRing<F, D>>> {
        self.rows().map(|r| r.to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Prime128Offset275;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Prime128Offset275;

    #[test]
    fn roundtrip_from_ring_matrix_and_view() {
        let mut rng = StdRng::seed_from_u64(42);
        let rows = 3usize;
        let cols = 5usize;
        let mat: Vec<Vec<CyclotomicRing<F, 64>>> = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| CyclotomicRing::random(&mut rng))
                    .collect()
            })
            .collect();

        let flat = FlatMatrix::from_ring_matrix(&mat);
        assert_eq!(flat.num_rows(), rows);
        assert_eq!(flat.cols_ring(), cols);
        assert_eq!(flat.gen_ring_dim(), 64);

        let view = flat.view::<64>();
        assert_eq!(view.num_rows(), rows);
        assert_eq!(view.num_cols(), cols);

        for (i, orig_row) in mat.iter().enumerate() {
            let view_row = view.row(i);
            assert_eq!(view_row, orig_row.as_slice());
        }
    }

    #[test]
    fn view_at_smaller_d_rechunks_correctly() {
        let mut rng = StdRng::seed_from_u64(99);
        let rows = 2usize;
        let cols = 4usize;
        let mat: Vec<Vec<CyclotomicRing<F, 64>>> = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| CyclotomicRing::random(&mut rng))
                    .collect()
            })
            .collect();

        let flat = FlatMatrix::from_ring_matrix(&mat);

        // View at D=32: each D=64 element becomes 2 D=32 elements
        let view32 = flat.view::<32>();
        assert_eq!(view32.num_rows(), rows);
        assert_eq!(view32.num_cols(), cols * 2);

        // Verify field elements are the same
        for r in 0..rows {
            let ring32_row = view32.row(r);
            let orig_row = flat.view::<64>().row(r);
            for (j, orig_ring) in orig_row.iter().enumerate() {
                let lo = &ring32_row[j * 2];
                let hi = &ring32_row[j * 2 + 1];
                assert_eq!(&orig_ring.coeffs[..32], lo.coefficients());
                assert_eq!(&orig_ring.coeffs[32..], hi.coefficients());
            }
        }
    }

    #[test]
    fn submatrix_truncates_correctly() {
        let mut rng = StdRng::seed_from_u64(7);
        let mat: Vec<Vec<CyclotomicRing<F, 64>>> = (0..4)
            .map(|_| (0..8).map(|_| CyclotomicRing::random(&mut rng)).collect())
            .collect();

        let flat = FlatMatrix::from_ring_matrix(&mat);
        let view = flat.view::<64>();
        let sub = view.submatrix(2, 5);

        assert_eq!(sub.num_rows(), 2);
        assert_eq!(sub.num_cols(), 5);
        for (r, row) in mat.iter().enumerate().take(2) {
            assert_eq!(sub.row(r), &row[..5]);
        }
    }
}
