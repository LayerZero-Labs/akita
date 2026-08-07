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
use akita_error::AkitaError;
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::Field;
use std::io::{Read, Write};

/// Flat 1D vector of field elements, independent of ring dimension.
///
/// Stores one exact contiguous prefix of public field elements.
/// Each role matrix (A, B, D) views a prefix of this vector reshaped into
/// its own `(num_rows, num_cols)` dimensions via [`RingMatrixView`].
///
/// Any prefix of a uniformly random vector is uniformly random, so role matrices
/// derived from prefixes of the same flat vector are binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatMatrix<F: Field> {
    data: Vec<F>,
}

impl<F: Field> FlatMatrix<F> {
    /// Number of stored base-field elements.
    #[inline]
    pub fn num_field_elements(&self) -> usize {
        self.data.len()
    }

    /// Borrow the backing field-element coefficients.
    #[inline]
    pub fn as_field_slice(&self) -> &[F] {
        &self.data
    }

    /// Build from pre-flattened field-element data.
    pub fn from_flat_data(data: Vec<F>) -> Self {
        Self { data }
    }

    /// Build from a flat slice of ring elements.
    pub fn from_ring_slice<const D: usize>(elements: &[CyclotomicRing<F, D>]) -> Self {
        let mut data = Vec::with_capacity(elements.len() * D);
        for ring_elem in elements {
            data.extend_from_slice(&ring_elem.coeffs);
        }
        Self { data }
    }

    /// Create a typed matrix view at ring dimension D with the given shape.
    ///
    /// The view interprets the first `num_rows * num_cols` ring elements
    /// (at dimension D) as a `num_rows × num_cols` matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested field footprint overflows or exceeds
    /// the stored prefix.
    pub fn ring_view<const D: usize>(
        &self,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<RingMatrixView<'_, F, D>, AkitaError> {
        let needed = num_rows
            .checked_mul(num_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("matrix view shape overflow".to_string()))?;
        let field_len = needed.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidSetup("matrix view field length overflow".to_string())
        })?;
        let data = self.data.get(..field_len).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "requested {field_len} field elements for a {num_rows}x{num_cols} D={D} matrix, but setup only has {}",
                self.data.len()
            ))
        })?;
        RingMatrixView {
            data,
            num_rows,
            num_cols,
        }
        .check_layout()
    }

    /// Runtime-dimension counterpart of [`Self::ring_view`].
    ///
    /// Views the matrix prefix as `num_rows x num_cols` ring elements of
    /// `ring_d` field coefficients each, without a compile-time ring
    /// dimension. Element `(row, col)` is the coefficient slice
    /// `data[(row * num_cols + col) * ring_d ..][.. ring_d]` — identical
    /// layout to the typed view.
    ///
    /// # Errors
    ///
    /// Returns an error when `ring_d` is zero or the requested field footprint
    /// exceeds the stored prefix.
    pub fn ring_view_dyn(
        &self,
        num_rows: usize,
        num_cols: usize,
        ring_d: usize,
    ) -> Result<FlatRingMatrixView<'_, F>, AkitaError> {
        if ring_d == 0 {
            return Err(AkitaError::InvalidSetup(
                "ring dimension must be non-zero".to_string(),
            ));
        }
        let needed = num_rows
            .checked_mul(num_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("matrix view shape overflow".to_string()))?;
        let field_len = needed.checked_mul(ring_d).ok_or_else(|| {
            AkitaError::InvalidSetup("matrix view field length overflow".to_string())
        })?;
        let data = self.data.get(..field_len).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "requested {field_len} field elements for a {num_rows}x{num_cols} D={ring_d} matrix, but setup only has {}",
                self.data.len()
            ))
        })?;
        Ok(FlatRingMatrixView {
            data,
            num_cols,
            ring_d,
        })
    }
}

/// Borrowed runtime-dimension ring-matrix view over flat field coefficients.
///
/// The runtime counterpart of [`RingMatrixView`]: rows of `num_cols` ring
/// elements, each element `ring_d` consecutive field coefficients.
#[derive(Clone, Copy, Debug)]
pub struct FlatRingMatrixView<'a, F> {
    data: &'a [F],
    num_cols: usize,
    ring_d: usize,
}

impl<'a, F: Field> FlatRingMatrixView<'a, F> {
    /// Ring dimension of every element in this view.
    #[must_use]
    pub fn ring_d(&self) -> usize {
        self.ring_d
    }

    /// Flat coefficients of one whole row (`num_cols * ring_d` field
    /// elements); element `col` of the row is the sub-slice
    /// `row_flat(row)[col * ring_d ..][.. ring_d]`.
    ///
    /// # Errors
    ///
    /// Returns an error when `row` lies outside the view.
    pub fn row_flat(&self, row: usize) -> Result<&'a [F], AkitaError> {
        let row_len = self.num_cols.checked_mul(self.ring_d).ok_or_else(|| {
            AkitaError::InvalidInput("ring matrix row length overflow".to_string())
        })?;
        let start = row
            .checked_mul(row_len)
            .ok_or_else(|| AkitaError::InvalidInput("ring matrix row overflow".to_string()))?;
        let end = start
            .checked_add(row_len)
            .ok_or_else(|| AkitaError::InvalidInput("ring matrix row overflow".to_string()))?;
        self.data
            .get(start..end)
            .ok_or_else(|| AkitaError::InvalidInput(format!("ring matrix row {row} out of range")))
    }

    /// Coefficients of the ring element at `(row, col)`.
    ///
    /// # Errors
    ///
    /// Returns an error when the position lies outside the view.
    pub fn elem(&self, row: usize, col: usize) -> Result<&'a [F], AkitaError> {
        if col >= self.num_cols {
            return Err(AkitaError::InvalidInput(format!(
                "ring matrix column {col} out of range (num_cols {})",
                self.num_cols
            )));
        }
        let idx = row
            .checked_mul(self.num_cols)
            .and_then(|base| base.checked_add(col))
            .and_then(|elem| elem.checked_mul(self.ring_d))
            .ok_or_else(|| {
                AkitaError::InvalidInput("ring matrix element index overflow".to_string())
            })?;
        let end = idx.checked_add(self.ring_d).ok_or_else(|| {
            AkitaError::InvalidInput("ring matrix element index overflow".to_string())
        })?;
        self.data
            .get(idx..end)
            .ok_or_else(|| AkitaError::InvalidInput(format!("ring matrix row {row} out of range")))
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> FlatMatrix<F> {
    /// Deserialize a flat matrix whose shape is already fixed by trusted
    /// metadata.
    ///
    /// The serialized matrix header is checked against
    /// `expected_num_field_elements` before allocating the backing vector. This
    /// is the safe verifier-facing setup path: the setup seed is read first,
    /// then the matrix is bounded by that seed rather than by untrusted matrix
    /// header sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the expected shape is invalid, the serialized header
    /// does not match it, allocation fails, or a field element is malformed.
    pub fn deserialize_with_expected_shape<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        expected_num_field_elements: usize,
        max_field_elements: usize,
    ) -> Result<Self, SerializationError> {
        if expected_num_field_elements == 0 {
            return Err(SerializationError::InvalidData(
                "expected flat matrix field count must be non-zero".to_string(),
            ));
        }
        if expected_num_field_elements > max_field_elements {
            return Err(SerializationError::LengthLimitExceeded {
                len: u64::try_from(expected_num_field_elements).unwrap_or(u64::MAX),
                max: max_field_elements,
            });
        }

        let num_field_elements =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        if num_field_elements != expected_num_field_elements {
            return Err(SerializationError::InvalidData(
                "flat matrix field count does not match expected setup shape".to_string(),
            ));
        }

        Self::deserialize_data(reader, compress, validate, num_field_elements)
    }

    fn deserialize_data<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        num_field_elements: usize,
    ) -> Result<Self, SerializationError> {
        let mut data = Vec::new();
        data.try_reserve_exact(num_field_elements).map_err(|_| {
            SerializationError::InvalidData("flat matrix allocation failed".to_string())
        })?;
        for _ in 0..num_field_elements {
            data.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self { data };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Field + Valid> Valid for FlatMatrix<F> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.data.is_empty() {
            return Err(SerializationError::InvalidData(
                "flat matrix field count must be non-zero".to_string(),
            ));
        }
        for f in &self.data {
            f.check()?;
        }
        Ok(())
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for FlatMatrix<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.num_field_elements()
            .serialize_with_mode(&mut writer, compress)?;
        for f in &self.data {
            f.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        std::mem::size_of::<usize>()
            + self
                .data
                .iter()
                .map(|f| f.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for FlatMatrix<F> {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let num_field_elements =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        if num_field_elements == 0 {
            return Err(SerializationError::InvalidData(
                "flat matrix field count must be non-zero".to_string(),
            ));
        }
        Self::deserialize_data(reader, compress, validate, num_field_elements)
    }
}

/// Typed read-only view of a [`FlatMatrix`] prefix at a specific ring
/// dimension D, interpreted as a `num_rows × num_cols` matrix.
///
/// Provides zero-copy access to rows as `&[CyclotomicRing<F, D>]` by
/// transmuting the underlying `&[F]` slice (safe because `CyclotomicRing`
/// is `#[repr(transparent)]` over `[F; D]`).
#[derive(Debug, Clone, Copy)]
pub struct RingMatrixView<'a, F: Field, const D: usize> {
    data: &'a [F],
    num_rows: usize,
    num_cols: usize,
}

impl<'a, F: Field, const D: usize> RingMatrixView<'a, F, D> {
    fn check_layout(self) -> Result<Self, AkitaError> {
        let row_field_len = self.num_cols.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidSetup("matrix row field length overflow".to_string())
        })?;
        let expected_len = self.num_rows.checked_mul(row_field_len).ok_or_else(|| {
            AkitaError::InvalidSetup("matrix view field length overflow".to_string())
        })?;
        if self.data.len() != expected_len {
            return Err(AkitaError::InvalidSetup(
                "matrix view backing length mismatch".to_string(),
            ));
        }
        Ok(self)
    }

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
    /// # Errors
    ///
    /// Returns an error if `row >= num_rows`.
    #[inline]
    pub fn row(&self, row: usize) -> Result<&'a [CyclotomicRing<F, D>], AkitaError> {
        if row >= self.num_rows {
            return Err(AkitaError::InvalidSetup(format!(
                "matrix row {row} out of bounds for {} rows",
                self.num_rows
            )));
        }
        let row_field_len = self.num_cols * D;
        let start = row * row_field_len;
        let field_slice = &self.data[start..start + row_field_len];
        Ok(Self::rings_from_fields(field_slice, self.num_cols))
    }

    /// Iterate rows without per-row bounds checks after the view is validated.
    #[inline]
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &'a [CyclotomicRing<F, D>]> + '_ {
        let row_field_len = self.num_cols * D;
        self.data
            .chunks_exact(row_field_len)
            .map(move |field_slice| Self::rings_from_fields(field_slice, self.num_cols))
    }

    /// Borrow the whole view as row-major ring elements.
    #[inline]
    pub fn as_slice(&self) -> &'a [CyclotomicRing<F, D>] {
        Self::rings_from_fields(self.data, self.num_rows * self.num_cols)
    }

    #[inline]
    fn rings_from_fields(field_slice: &'a [F], num_cols: usize) -> &'a [CyclotomicRing<F, D>] {
        // SAFETY: CyclotomicRing<F, D> is #[repr(transparent)] over [F; D],
        // so a contiguous &[F] of length num_cols*D has the same layout as
        // &[CyclotomicRing<F, D>] of length num_cols.
        unsafe {
            std::slice::from_raw_parts(
                field_slice.as_ptr() as *const CyclotomicRing<F, D>,
                num_cols,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::Zero;
    use jolt_field::Prime128Offset275;
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
        assert_eq!(flat.num_field_elements(), rows * cols * 64);

        let view = flat.ring_view::<64>(rows, cols).unwrap();
        assert_eq!(view.num_rows(), rows);
        assert_eq!(view.num_cols(), cols);

        for r in 0..rows {
            let view_row = view.row(r).unwrap();
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
        assert_eq!(flat.num_field_elements(), total * 64);

        let view32 = flat.ring_view::<32>(2, total).unwrap();
        assert_eq!(view32.num_rows(), 2);
        assert_eq!(view32.num_cols(), total);

        let view64 = flat.ring_view::<64>(2, 4).unwrap();
        for r in 0..2 {
            let row64 = view64.row(r).unwrap();
            let row32 = view32.row(r).unwrap();
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

        let view_a = flat.ring_view::<64>(2, 16).unwrap();
        let view_b = flat.ring_view::<64>(4, 8).unwrap();

        assert_eq!(view_a.row(0).unwrap()[0], elements[0]);
        assert_eq!(view_a.row(1).unwrap()[0], elements[16]);
        assert_eq!(view_b.row(0).unwrap()[0], elements[0]);
        assert_eq!(view_b.row(1).unwrap()[0], elements[8]);
    }

    #[test]
    fn malformed_ring_view_returns_error() {
        let flat = FlatMatrix::<F>::from_flat_data(vec![F::zero(); 3]);
        assert!(flat.ring_view::<2>(1, 1).is_ok());
        assert!(flat.ring_view::<3>(2, 1).is_err());
        assert!(flat.ring_view::<3>(usize::MAX, usize::MAX).is_err());

        let dynamic = flat.ring_view_dyn(1, 1, 1).unwrap();
        assert!(dynamic.row_flat(usize::MAX).is_err());
        assert!(dynamic.elem(usize::MAX, 0).is_err());
    }

    #[test]
    fn deserialization_rejects_zero_field_count_before_allocation() {
        let mut bytes = Vec::new();
        0usize.serialize_uncompressed(&mut bytes).unwrap();

        let err = FlatMatrix::<F>::deserialize_uncompressed(&*bytes, &()).unwrap_err();
        assert!(matches!(err, SerializationError::InvalidData(_)));
    }

    #[test]
    fn stored_prefix_need_not_be_divisible_by_view_dimension() {
        let flat = FlatMatrix::<F>::from_flat_data(vec![F::zero(); 65]);

        assert_eq!(flat.ring_view::<64>(1, 1).unwrap().as_slice().len(), 1);
        assert!(flat.ring_view::<64>(1, 2).is_err());
        assert_eq!(flat.num_field_elements(), 65);
    }
}
