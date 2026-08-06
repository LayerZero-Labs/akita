//! Coefficient-first storage for materialized sumcheck evaluations.

use akita_field::{AkitaError, ExtField, FieldCore};
use core::fmt;
use core::marker::PhantomData;

/// Materialized field evaluations stored by base-field coefficient.
///
/// Each coefficient of `E` occupies one contiguous slab of `F` values. The
/// initial row count is retained as the slab stride while `len` tracks the live
/// prefix after folds. Row order belongs to the caller. Dense multilinear
/// constructors convert Akita's logical LSB-first order into binding order;
/// row-preserving constructors are used for sparse values and API boundaries.
#[derive(Clone)]
pub struct EvaluationTable<F, E> {
    coefficients: Box<[F]>,
    len: usize,
    stride: usize,
    marker: PhantomData<fn() -> E>,
}

impl<F, E> EvaluationTable<F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    /// Build a table while preserving the input row order.
    pub fn from_evaluations(evaluations: &[E]) -> Self {
        Self::from_evaluation_fn(evaluations.len(), |row| evaluations[row])
    }

    /// Build a table from a row generator while preserving generated row order.
    pub fn from_evaluation_fn<G>(len: usize, mut evaluation: G) -> Self
    where
        G: FnMut(usize) -> E,
    {
        let mut coefficients = Self::uninitialized_coefficients(len);
        for row in 0..len {
            let value = evaluation(row);
            for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
                coefficients[coefficient * len + row].write(value.base_coefficient(coefficient));
            }
        }

        // SAFETY: every slot is written exactly once by the nested loops above.
        let coefficients = unsafe { coefficients.assume_init() };
        Self::from_initialized(coefficients, len)
    }

    /// Build a table from a base-coefficient generator while preserving rows.
    pub fn from_coefficient_fn<G>(len: usize, mut coefficient_at: G) -> Self
    where
        G: FnMut(usize, usize) -> F,
    {
        let mut coefficients = Self::uninitialized_coefficients(len);
        for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
            for row in 0..len {
                coefficients[coefficient * len + row].write(coefficient_at(row, coefficient));
            }
        }

        // SAFETY: every slot is written exactly once by the nested loops above.
        let coefficients = unsafe { coefficients.assume_init() };
        Self::from_initialized(coefficients, len)
    }

    /// Build a dense multilinear table from logical LSB-first evaluations.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of evaluations is zero or is not a power
    /// of two.
    pub fn from_multilinear_evaluations(evaluations: &[E]) -> Result<Self, AkitaError> {
        Self::from_multilinear_evaluation_fn(evaluations.len(), |row| evaluations[row])
    }

    /// Build a dense multilinear table from logical LSB-first generated rows.
    ///
    /// The stored rows are in binding order, so the next variable selects
    /// between two contiguous halves.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` is zero or is not a power of two.
    pub fn from_multilinear_evaluation_fn<G>(
        len: usize,
        mut evaluation: G,
    ) -> Result<Self, AkitaError>
    where
        G: FnMut(usize) -> E,
    {
        Self::validate_multilinear_len(len)?;
        Ok(Self::from_evaluation_fn(len, |stored_row| {
            evaluation(Self::logical_row(stored_row, len))
        }))
    }

    /// Build a dense multilinear table from logical LSB-first coefficients.
    ///
    /// The stored rows are in binding order, so the next variable selects
    /// between two contiguous halves.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` is zero or is not a power of two.
    pub fn from_multilinear_coefficient_fn<G>(
        len: usize,
        mut coefficient_at: G,
    ) -> Result<Self, AkitaError>
    where
        G: FnMut(usize, usize) -> F,
    {
        Self::validate_multilinear_len(len)?;
        Ok(Self::from_coefficient_fn(len, |stored_row, coefficient| {
            coefficient_at(Self::logical_row(stored_row, len), coefficient)
        }))
    }

    /// Return the field evaluation at one stored row.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.len()`.
    #[inline]
    pub fn evaluation(&self, row: usize) -> E {
        assert!(row < self.len, "evaluation row out of range");
        E::from_base_fn(|coefficient| self.coefficients[coefficient * self.stride + row])
    }

    /// Return one coefficient's live stored rows.
    ///
    /// # Panics
    ///
    /// Panics if `coefficient >= E::EXT_DEGREE`.
    #[inline]
    pub fn coefficient_slice(&self, coefficient: usize) -> &[F] {
        assert!(
            coefficient < <E as ExtField<F>>::EXT_DEGREE,
            "extension coefficient out of range"
        );
        let start = coefficient * self.stride;
        &self.coefficients[start..start + self.len]
    }

    /// Return one coefficient's mutable live stored rows.
    ///
    /// # Panics
    ///
    /// Panics if `coefficient >= E::EXT_DEGREE`.
    #[inline]
    pub fn coefficient_slice_mut(&mut self, coefficient: usize) -> &mut [F] {
        assert!(
            coefficient < <E as ExtField<F>>::EXT_DEGREE,
            "extension coefficient out of range"
        );
        let start = coefficient * self.stride;
        &mut self.coefficients[start..start + self.len]
    }

    /// Return the number of live stored rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the table has no live rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Shorten the live stored rows without reallocating.
    ///
    /// # Panics
    ///
    /// Panics if `new_len > self.len()`.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        assert!(new_len <= self.len, "evaluation table cannot grow");
        self.len = new_len;
    }

    /// Convert the live stored rows back into ordinary extension values.
    ///
    /// This preserves stored row order. It does not undo dense binding order.
    pub fn into_evaluations(self) -> Vec<E> {
        (0..self.len)
            .map(|row| {
                E::from_base_fn(|coefficient| self.coefficients[coefficient * self.stride + row])
            })
            .collect()
    }

    fn uninitialized_coefficients(len: usize) -> Box<[core::mem::MaybeUninit<F>]> {
        assert!(
            <E as ExtField<F>>::EXT_DEGREE > 0,
            "extension degree must be nonzero"
        );
        let stored_len = <E as ExtField<F>>::EXT_DEGREE
            .checked_mul(len)
            .expect("evaluation table storage length overflow");
        Box::<[F]>::new_uninit_slice(stored_len)
    }

    fn from_initialized(coefficients: Box<[F]>, len: usize) -> Self {
        debug_assert_eq!(coefficients.len(), <E as ExtField<F>>::EXT_DEGREE * len);
        Self {
            coefficients,
            len,
            stride: len,
            marker: PhantomData,
        }
    }

    fn validate_multilinear_len(len: usize) -> Result<(), AkitaError> {
        if len == 0 || !len.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "multilinear evaluation table length must be a nonzero power of two".to_string(),
            ));
        }
        Ok(())
    }

    #[inline]
    fn logical_row(stored_row: usize, len: usize) -> usize {
        let bits = len.trailing_zeros();
        if bits == 0 {
            0
        } else {
            stored_row.reverse_bits() >> (usize::BITS - bits)
        }
    }
}

impl<F, E> fmt::Debug for EvaluationTable<F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvaluationTable")
            .field("len", &self.len)
            .field("stride", &self.stride)
            .field("extension_degree", &<E as ExtField<F>>::EXT_DEGREE)
            .finish_non_exhaustive()
    }
}

impl<F, E> PartialEq for EvaluationTable<F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && (0..<E as ExtField<F>>::EXT_DEGREE).all(|coefficient| {
                self.coefficient_slice(coefficient) == other.coefficient_slice(coefficient)
            })
    }
}

impl<F, E> Eq for EvaluationTable<F, E>
where
    F: FieldCore,
    E: ExtField<F>,
{
}

#[cfg(test)]
mod tests {
    use super::EvaluationTable;
    use akita_field::{Ext2, ExtField, FpExt4, FpExt8, Prime32Offset99};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn value(row: usize) -> E {
        E::from_base_fn(|coefficient| F::from_u64((100 * coefficient + row) as u64))
    }

    #[test]
    fn row_preserving_constructors_match() {
        let evaluations: Vec<_> = (0..7).map(value).collect();
        let from_slice = EvaluationTable::<F, E>::from_evaluations(&evaluations);
        let from_values = EvaluationTable::<F, E>::from_evaluation_fn(7, value);
        let from_coefficients = EvaluationTable::<F, E>::from_coefficient_fn(7, |row, c| {
            F::from_u64((100 * c + row) as u64)
        });

        assert_eq!(from_slice, from_values);
        assert_eq!(from_slice, from_coefficients);
        assert_eq!(from_slice.clone().into_evaluations(), evaluations);
    }

    #[test]
    fn all_supported_extension_degrees_round_trip() {
        fn check<T: ExtField<F>>() {
            let evaluations: Vec<_> = (0..5)
                .map(|row| {
                    T::from_base_fn(|coefficient| F::from_u64((100 * coefficient + row) as u64))
                })
                .collect();
            let table = EvaluationTable::<F, T>::from_evaluations(&evaluations);
            assert_eq!(table.into_evaluations(), evaluations);
        }

        check::<F>();
        check::<Ext2<F>>();
        check::<FpExt4<F>>();
        check::<FpExt8<F>>();
    }

    #[test]
    fn coefficients_are_contiguous() {
        let table = EvaluationTable::<F, E>::from_evaluation_fn(7, value);
        for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
            let expected: Vec<_> = (0..7)
                .map(|row| F::from_u64((100 * coefficient + row) as u64))
                .collect();
            assert_eq!(table.coefficient_slice(coefficient), expected);
        }
    }

    #[test]
    fn multilinear_constructors_write_binding_order() {
        let evaluations: Vec<_> = (0..8).map(value).collect();
        let from_slice = EvaluationTable::<F, E>::from_multilinear_evaluations(&evaluations)
            .expect("valid multilinear length");
        let from_values = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(8, value)
            .expect("valid multilinear length");
        let from_coefficients =
            EvaluationTable::<F, E>::from_multilinear_coefficient_fn(8, |row, coefficient| {
                F::from_u64((100 * coefficient + row) as u64)
            })
            .expect("valid multilinear length");
        let logical_rows = [0, 4, 2, 6, 1, 5, 3, 7];

        assert_eq!(from_slice, from_values);
        assert_eq!(from_slice, from_coefficients);
        for (stored_row, logical_row) in logical_rows.into_iter().enumerate() {
            assert_eq!(from_slice.evaluation(stored_row), evaluations[logical_row]);
        }
    }

    #[test]
    fn truncate_keeps_one_allocation_and_live_prefixes() {
        let mut table = EvaluationTable::<F, E>::from_evaluation_fn(8, value);
        let stored_len = table.coefficients.len();
        table.truncate(3);

        assert_eq!(table.len(), 3);
        assert_eq!(table.stride, 8);
        assert_eq!(table.coefficients.len(), stored_len);
        assert_eq!(table.coefficient_slice(2).len(), 3);
        assert_eq!(table.evaluation(2), value(2));
    }

    #[test]
    fn equality_ignores_inactive_rows() {
        let mut left = EvaluationTable::<F, E>::from_evaluation_fn(8, value);
        let mut right = EvaluationTable::<F, E>::from_evaluation_fn(8, |row| {
            if row < 3 {
                value(row)
            } else {
                value(row + 20)
            }
        });
        left.truncate(3);
        right.truncate(3);
        assert_eq!(left, right);
    }

    #[test]
    fn empty_sparse_value_table_is_supported() {
        let table = EvaluationTable::<F, E>::from_evaluations(&[]);
        assert!(table.is_empty());
        assert_eq!(table.coefficients.len(), 0);
        for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
            assert!(table.coefficient_slice(coefficient).is_empty());
        }
    }

    #[test]
    fn multilinear_length_must_be_nonzero_power_of_two() {
        assert!(EvaluationTable::<F, E>::from_multilinear_evaluation_fn(0, value).is_err());
        assert!(EvaluationTable::<F, E>::from_multilinear_evaluation_fn(3, value).is_err());
    }

    #[test]
    #[should_panic(expected = "extension coefficient out of range")]
    fn coefficient_index_is_checked() {
        let table = EvaluationTable::<F, E>::from_evaluation_fn(1, value);
        let _ = table.coefficient_slice(<E as ExtField<F>>::EXT_DEGREE);
    }

    #[test]
    #[should_panic(expected = "evaluation table cannot grow")]
    fn truncate_cannot_grow() {
        let mut table = EvaluationTable::<F, E>::from_evaluation_fn(1, value);
        table.truncate(2);
    }
}
