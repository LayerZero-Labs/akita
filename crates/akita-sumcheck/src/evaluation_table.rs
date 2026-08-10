//! Coefficient-first storage for materialized sumcheck evaluations.

use crate::accum::{
    DelayedProductRoundAccumulator, DirectProductRoundAccumulator, ProductRoundAccumulator,
};
use akita_field::parallel::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, ExtField, FieldCore};
use core::fmt;
use core::marker::PhantomData;

fn initialize_multilinear_table<F, E, G>(
    coefficients: &mut [core::mem::MaybeUninit<F>],
    len: usize,
    table_index: usize,
    evaluation: &G,
) where
    F: FieldCore,
    E: ExtField<F>,
    G: Fn(usize, usize) -> E + Sync,
{
    #[cfg(feature = "parallel")]
    if akita_field::parallel::__rayon_current_num_threads() > 1 {
        initialize_multilinear_table_in_parallel::<F, E, G>(
            coefficients,
            len,
            table_index,
            evaluation,
        );
        return;
    }

    let mut coefficient_slices: Vec<_> = coefficients.chunks_mut(len).collect();
    initialize_multilinear_rows_sequential::<F, E, G>(
        &mut coefficient_slices,
        0,
        len,
        table_index,
        evaluation,
    );
}

fn initialize_multilinear_rows_sequential<F, E, G>(
    coefficient_slices: &mut [&mut [core::mem::MaybeUninit<F>]],
    start: usize,
    table_len: usize,
    table_index: usize,
    evaluation: &G,
) where
    F: FieldCore,
    E: ExtField<F>,
    G: Fn(usize, usize) -> E + Sync,
{
    for row in 0..coefficient_slices[0].len() {
        let logical_row = EvaluationTable::<F, E>::logical_row(start + row, table_len);
        let value = evaluation(table_index, logical_row);
        for (coefficient, destination) in coefficient_slices.iter_mut().enumerate() {
            destination[row].write(value.base_coefficient(coefficient));
        }
    }
}

fn initialize_evaluation_table<F, E, G>(
    coefficients: &mut [core::mem::MaybeUninit<F>],
    len: usize,
    evaluation: &G,
) where
    F: FieldCore,
    E: ExtField<F>,
    G: Fn(usize) -> E + Sync,
{
    if len == 0 {
        return;
    }

    #[cfg(feature = "parallel")]
    if akita_field::parallel::__rayon_current_num_threads() > 1 {
        let target_tasks = akita_field::parallel::__rayon_current_num_threads()
            .saturating_mul(4)
            .max(1);
        let minimum_rows_per_task = len.div_ceil(target_tasks).max(1_024);
        let coefficient_slices = coefficients.chunks_mut(len).collect();
        initialize_evaluation_rows(coefficient_slices, 0, evaluation, minimum_rows_per_task);
        return;
    }

    let mut coefficient_slices: Vec<_> = coefficients.chunks_mut(len).collect();
    initialize_evaluation_rows_sequential::<F, E, G>(&mut coefficient_slices, 0, evaluation);
}

fn initialize_evaluation_rows_sequential<F, E, G>(
    coefficient_slices: &mut [&mut [core::mem::MaybeUninit<F>]],
    start: usize,
    evaluation: &G,
) where
    F: FieldCore,
    E: ExtField<F>,
    G: Fn(usize) -> E + Sync,
{
    for row in 0..coefficient_slices[0].len() {
        let value = evaluation(start + row);
        for (coefficient, destination) in coefficient_slices.iter_mut().enumerate() {
            destination[row].write(value.base_coefficient(coefficient));
        }
    }
}

#[cfg(feature = "parallel")]
fn initialize_evaluation_rows<F, E, G>(
    mut coefficient_slices: Vec<&mut [core::mem::MaybeUninit<F>]>,
    start: usize,
    evaluation: &G,
    minimum_rows_per_task: usize,
) where
    F: FieldCore,
    E: ExtField<F>,
    G: Fn(usize) -> E + Sync,
{
    let len = coefficient_slices[0].len();
    if len <= minimum_rows_per_task {
        initialize_evaluation_rows_sequential::<F, E, G>(
            &mut coefficient_slices,
            start,
            evaluation,
        );
        return;
    }

    let mid = len / 2;
    let (left, right): (Vec<_>, Vec<_>) = coefficient_slices
        .into_iter()
        .map(|coefficient| coefficient.split_at_mut(mid))
        .unzip();
    akita_field::parallel::__rayon_join(
        || initialize_evaluation_rows::<F, E, G>(left, start, evaluation, minimum_rows_per_task),
        || {
            initialize_evaluation_rows::<F, E, G>(
                right,
                start + mid,
                evaluation,
                minimum_rows_per_task,
            )
        },
    );
}

#[cfg(feature = "parallel")]
fn initialize_multilinear_table_in_parallel<F, E, G>(
    coefficients: &mut [core::mem::MaybeUninit<F>],
    len: usize,
    table_index: usize,
    evaluation: &G,
) where
    F: FieldCore,
    E: ExtField<F>,
    G: Fn(usize, usize) -> E + Sync,
{
    let target_tasks = akita_field::parallel::__rayon_current_num_threads()
        .saturating_mul(4)
        .max(1);
    let minimum_rows_per_task = len.div_ceil(target_tasks).max(1_024);
    let coefficient_slices = coefficients.chunks_mut(len).collect();
    initialize_multilinear_rows(
        coefficient_slices,
        0,
        len,
        table_index,
        evaluation,
        minimum_rows_per_task,
    );
}

#[cfg(feature = "parallel")]
fn initialize_multilinear_rows<F, E, G>(
    mut coefficient_slices: Vec<&mut [core::mem::MaybeUninit<F>]>,
    start: usize,
    table_len: usize,
    table_index: usize,
    evaluation: &G,
    minimum_rows_per_task: usize,
) where
    F: FieldCore,
    E: ExtField<F>,
    G: Fn(usize, usize) -> E + Sync,
{
    let len = coefficient_slices[0].len();
    if len <= minimum_rows_per_task {
        initialize_multilinear_rows_sequential::<F, E, G>(
            &mut coefficient_slices,
            start,
            table_len,
            table_index,
            evaluation,
        );
        return;
    }

    let mid = len / 2;
    let (left, right): (Vec<_>, Vec<_>) = coefficient_slices
        .into_iter()
        .map(|coefficient| coefficient.split_at_mut(mid))
        .unzip();
    akita_field::parallel::__rayon_join(
        || {
            initialize_multilinear_rows::<F, E, G>(
                left,
                start,
                table_len,
                table_index,
                evaluation,
                minimum_rows_per_task,
            )
        },
        || {
            initialize_multilinear_rows::<F, E, G>(
                right,
                start + mid,
                table_len,
                table_index,
                evaluation,
                minimum_rows_per_task,
            )
        },
    );
}

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

/// Fold a dense evaluation table by binding its first variable to `challenge`.
///
/// This is the portable scalar reference for runtime-selected prover kernels.
/// Dense tables are already in binding order, so row `i` is paired with row
/// `i + len / 2`. The result replaces the first half of every coefficient slab,
/// and the table's live length is halved without reallocating.
///
/// # Panics
///
/// Panics if the live table length is not a power of two or has fewer than two
/// rows. Prover construction establishes these dense-table invariants.
pub fn fold_first_variable_scalar<F, E>(table: &mut EvaluationTable<F, E>, challenge: E)
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold,
{
    assert!(
        table.len().is_power_of_two(),
        "evaluation table length must be a power of two"
    );
    assert!(
        table.len() >= 2,
        "evaluation table must have at least two rows"
    );

    let half = table.len() / 2;
    let context = E::precompute_fold(challenge);
    for row in 0..half {
        let folded = E::fold_one(
            &context,
            table.evaluation(row),
            table.evaluation(row + half),
        );
        for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
            table.coefficient_slice_mut(coefficient)[row] = folded.base_coefficient(coefficient);
        }
    }
    table.truncate(half);
}

/// Compute the constant and quadratic coefficients for a product round.
///
/// Both dense tables must have the same binding order and live length. For each
/// row pair this computes `w0 * f0` for the constant coefficient and
/// `(w1 - w0) * (f1 - f0)` for the quadratic coefficient. The caller derives
/// the linear coefficient from the previous claim.
///
/// # Panics
///
/// Panics if the table lengths differ, are not powers of two, or have fewer
/// than two rows.
pub fn compute_product_round_scalar<F, E>(
    witness: &EvaluationTable<F, E>,
    factor: &EvaluationTable<F, E>,
) -> (E, E)
where
    F: FieldCore,
    E: ExtField<F> + HasUnreducedOps,
{
    validate_product_round_tables(witness, factor);
    assert!(
        witness.len() >= 2,
        "product round tables must have at least two rows"
    );

    if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        compute_product_round_with::<F, E, DelayedProductRoundAccumulator<E>>(witness, factor)
    } else {
        compute_product_round_with::<F, E, DirectProductRoundAccumulator<E>>(witness, factor)
    }
}

/// Fold two tables and compute their next product round in one pass.
///
/// The operation binds the first variable to `challenge`, writes both folded
/// tables in place, and returns the constant and quadratic coefficients for the
/// next variable. It avoids reading the folded tables in a second pass.
///
/// # Panics
///
/// Panics if the table lengths differ, are not powers of two, or have fewer
/// than four rows.
pub fn fold_and_compute_product_round_scalar<F, E>(
    witness: &mut EvaluationTable<F, E>,
    factor: &mut EvaluationTable<F, E>,
    challenge: E,
) -> (E, E)
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold + HasUnreducedOps,
{
    validate_product_round_tables(witness, factor);
    assert!(
        witness.len() >= 4,
        "fused product round tables must have at least four rows"
    );

    if E::DELAYED_PRODUCT_SUM_IS_EXACT {
        fold_and_compute_product_round_with::<F, E, DelayedProductRoundAccumulator<E>>(
            witness, factor, challenge,
        )
    } else {
        fold_and_compute_product_round_with::<F, E, DirectProductRoundAccumulator<E>>(
            witness, factor, challenge,
        )
    }
}

fn validate_product_round_tables<F, E>(
    witness: &EvaluationTable<F, E>,
    factor: &EvaluationTable<F, E>,
) where
    F: FieldCore,
    E: ExtField<F>,
{
    assert_eq!(
        witness.len(),
        factor.len(),
        "product round tables must have equal lengths"
    );
    assert!(
        witness.len().is_power_of_two(),
        "product round table length must be a power of two"
    );
}

fn compute_product_round_with<F, E, A>(
    witness: &EvaluationTable<F, E>,
    factor: &EvaluationTable<F, E>,
) -> (E, E)
where
    F: FieldCore,
    E: ExtField<F> + HasUnreducedOps,
    A: ProductRoundAccumulator<E>,
{
    let half = witness.len() / 2;
    let mut accumulator = A::zero();
    for row in 0..half {
        let witness_0 = witness.evaluation(row);
        let witness_1 = witness.evaluation(row + half);
        let factor_0 = factor.evaluation(row);
        let factor_1 = factor.evaluation(row + half);
        accumulator.add_constant_product(witness_0, factor_0);
        accumulator.add_quadratic_product(witness_1 - witness_0, factor_1 - factor_0);
    }
    accumulator.finish()
}

fn fold_and_compute_product_round_with<F, E, A>(
    witness: &mut EvaluationTable<F, E>,
    factor: &mut EvaluationTable<F, E>,
    challenge: E,
) -> (E, E)
where
    F: FieldCore,
    E: ExtField<F> + HasOptimizedFold + HasUnreducedOps,
    A: ProductRoundAccumulator<E>,
{
    let half = witness.len() / 2;
    let quarter = half / 2;
    let fold = E::precompute_fold(challenge);
    let mut accumulator = A::zero();

    for row in 0..quarter {
        let witness_0 = E::fold_one(
            &fold,
            witness.evaluation(row),
            witness.evaluation(row + half),
        );
        let witness_1 = E::fold_one(
            &fold,
            witness.evaluation(row + quarter),
            witness.evaluation(row + quarter + half),
        );
        let factor_0 = E::fold_one(&fold, factor.evaluation(row), factor.evaluation(row + half));
        let factor_1 = E::fold_one(
            &fold,
            factor.evaluation(row + quarter),
            factor.evaluation(row + quarter + half),
        );

        accumulator.add_constant_product(witness_0, factor_0);
        accumulator.add_quadratic_product(witness_1 - witness_0, factor_1 - factor_0);

        for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
            let witness_coefficient = witness.coefficient_slice_mut(coefficient);
            witness_coefficient[row] = witness_0.base_coefficient(coefficient);
            witness_coefficient[row + quarter] = witness_1.base_coefficient(coefficient);

            let factor_coefficient = factor.coefficient_slice_mut(coefficient);
            factor_coefficient[row] = factor_0.base_coefficient(coefficient);
            factor_coefficient[row + quarter] = factor_1.base_coefficient(coefficient);
        }
    }

    witness.truncate(half);
    factor.truncate(half);
    accumulator.finish()
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
    pub fn from_evaluation_fn<G>(len: usize, evaluation: G) -> Self
    where
        G: Fn(usize) -> E + Sync,
    {
        let mut coefficients = Self::uninitialized_coefficients(len);
        initialize_evaluation_table::<F, E, G>(&mut coefficients, len, &evaluation);

        // SAFETY: every coefficient slot is written exactly once by the
        // disjoint stored-row traversal.
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

    /// Build a table from complete coefficient-first storage.
    ///
    /// The storage contains one contiguous `len`-element slab per base-field
    /// coefficient. Rows are preserved exactly as supplied. This constructor
    /// lets bulk producers write the canonical representation directly rather
    /// than materializing and transposing extension-field values first.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` is not a nonzero power of two or if the
    /// storage length is not exactly `E::EXT_DEGREE * len`.
    pub fn from_coefficient_storage(
        coefficients: Box<[F]>,
        len: usize,
    ) -> Result<Self, AkitaError> {
        Self::validate_multilinear_len(len)?;
        let expected = <E as ExtField<F>>::EXT_DEGREE
            .checked_mul(len)
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "evaluation table coefficient storage length overflow".to_string(),
                )
            })?;
        if coefficients.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: coefficients.len(),
            });
        }
        Ok(Self::from_initialized(coefficients, len))
    }

    /// Build a dense multilinear table from logical LSB-first evaluations.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of evaluations is zero or is not a power
    /// of two.
    pub fn from_multilinear_evaluations(evaluations: &[E]) -> Result<Self, AkitaError> {
        let len = evaluations.len();
        Self::validate_multilinear_len(len)?;
        let mut coefficients = Self::uninitialized_coefficients(len);
        for stored_row in 0..len {
            let value = evaluations[Self::logical_row(stored_row, len)];
            for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
                coefficients[coefficient * len + stored_row]
                    .write(value.base_coefficient(coefficient));
            }
        }

        // SAFETY: bit reversal permutes the source rows, and every destination
        // coefficient slot is written exactly once in stored-row order.
        let coefficients = unsafe { coefficients.assume_init() };
        Ok(Self::from_initialized(coefficients, len))
    }

    /// Build a dense multilinear table from logical LSB-first generated rows.
    ///
    /// The stored rows are in binding order, so the next variable selects
    /// between two contiguous halves. One-worker construction calls the
    /// generator in ascending logical-row order for sequential source access.
    /// Multi-worker construction partitions logical rows so every worker reads
    /// sequential source ranges, then permutes the initialized coefficient
    /// slabs in place into binding order.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` is zero or is not a power of two.
    pub fn from_multilinear_evaluation_fn<G>(len: usize, evaluation: G) -> Result<Self, AkitaError>
    where
        G: Fn(usize) -> E + Sync,
    {
        Self::validate_multilinear_len(len)?;
        let mut coefficients = Self::uninitialized_coefficients(len);

        #[cfg(feature = "parallel")]
        if akita_field::parallel::__rayon_current_num_threads() > 1 {
            initialize_evaluation_table::<F, E, _>(&mut coefficients, len, &evaluation);

            // SAFETY: every coefficient slot is written exactly once by the
            // disjoint logical-row traversal.
            let mut coefficients = unsafe { coefficients.assume_init() };
            for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
                let start = coefficient * len;
                let values = &mut coefficients[start..start + len];
                for logical_row in 0..len {
                    let stored_row = Self::logical_row(logical_row, len);
                    if logical_row < stored_row {
                        values.swap(logical_row, stored_row);
                    }
                }
            }
            return Ok(Self::from_initialized(coefficients, len));
        }

        for logical_row in 0..len {
            let value = evaluation(logical_row);
            let stored_row = Self::logical_row(logical_row, len);
            for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
                coefficients[coefficient * len + stored_row]
                    .write(value.base_coefficient(coefficient));
            }
        }

        // SAFETY: bit reversal permutes the rows, so every coefficient slot is
        // written exactly once by the nested loops above.
        let coefficients = unsafe { coefficients.assume_init() };
        Ok(Self::from_initialized(coefficients, len))
    }

    /// Build several dense multilinear tables in one logical-row traversal.
    ///
    /// The generator returns one value per table. Every table is written
    /// directly in binding order and owns the same canonical coefficient-first
    /// representation as the single-table constructor.
    ///
    /// # Errors
    ///
    /// Returns an error if `len` is zero or is not a power of two.
    pub fn from_multilinear_evaluation_array_fn<const N: usize, G>(
        len: usize,
        evaluation: G,
    ) -> Result<[Self; N], AkitaError>
    where
        G: Fn(usize, usize) -> E + Sync,
    {
        Self::validate_multilinear_len(len)?;
        let tables = cfg_into_iter!(0..N)
            .map(|table_index| {
                let mut coefficients = Self::uninitialized_coefficients(len);
                initialize_multilinear_table::<F, E, G>(
                    &mut coefficients,
                    len,
                    table_index,
                    &evaluation,
                );

                // SAFETY: every coefficient slot for this table is written
                // exactly once by its stored-row traversal.
                let coefficients = unsafe { coefficients.assume_init() };
                Self::from_initialized(coefficients, len)
            })
            .collect::<Vec<_>>();
        match tables.try_into() {
            Ok(tables) => Ok(tables),
            Err(_) => unreachable!("table iterator length matches the requested array length"),
        }
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

    /// Replace the field evaluation at one stored row.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.len()`.
    #[inline]
    pub fn set_evaluation(&mut self, row: usize, value: E) {
        assert!(row < self.len, "evaluation row out of range");
        for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
            self.coefficients[coefficient * self.stride + row] =
                value.base_coefficient(coefficient);
        }
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

    /// Return all live coefficient slices as separate borrows.
    ///
    /// The const length must equal `E::EXT_DEGREE`. This form lets a kernel
    /// load several coefficient slabs together while checking the table shape
    /// once outside the row loop.
    ///
    /// # Panics
    ///
    /// Panics if `N != E::EXT_DEGREE`.
    pub fn coefficient_slices<const N: usize>(&self) -> [&[F]; N] {
        assert_eq!(
            N,
            <E as ExtField<F>>::EXT_DEGREE,
            "coefficient slice count must match the extension degree"
        );
        let len = self.len;
        let stride = self.stride;
        std::array::from_fn(|coefficient| {
            let start = coefficient * stride;
            &self.coefficients[start..start + len]
        })
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

    /// Return all live coefficient slices as separate mutable borrows.
    ///
    /// The const length must equal `E::EXT_DEGREE`. This form lets a kernel
    /// process several coefficient slabs together without exposing the table's
    /// backing allocation, stride, or inactive rows.
    ///
    /// # Panics
    ///
    /// Panics if `N != E::EXT_DEGREE`.
    pub fn coefficient_slices_mut<const N: usize>(&mut self) -> [&mut [F]; N] {
        assert_eq!(
            N,
            <E as ExtField<F>>::EXT_DEGREE,
            "coefficient slice count must match the extension degree"
        );
        let len = self.len;
        let stride = self.stride;
        let mut remaining: &mut [F] = &mut self.coefficients;
        std::array::from_fn(|_| {
            let (slab, rest) = core::mem::take(&mut remaining).split_at_mut(stride);
            remaining = rest;
            &mut slab[..len]
        })
    }

    /// Return all live coefficient slices when the extension degree is dynamic.
    ///
    /// This form lets a generic traversal divide every coefficient slab into
    /// the same disjoint row ranges. Fixed-degree SIMD kernels should use
    /// [`Self::coefficient_slices_mut`] instead.
    pub fn all_coefficient_slices_mut(&mut self) -> Vec<&mut [F]> {
        let len = self.len;
        let stride = self.stride;
        let mut remaining: &mut [F] = &mut self.coefficients;
        (0..<E as ExtField<F>>::EXT_DEGREE)
            .map(|_| {
                let (slab, rest) = core::mem::take(&mut remaining).split_at_mut(stride);
                remaining = rest;
                &mut slab[..len]
            })
            .collect()
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

    /// Remove the first variable from a binding-order equality table.
    ///
    /// For `eq(tau, x)`, the two halves are `(1 - tau[0]) * suffix` and
    /// `tau[0] * suffix`. Adding them recovers the unscaled suffix table.
    ///
    /// # Panics
    ///
    /// Panics unless the live length is a power of two with at least two rows.
    pub fn sum_first_variable_halves(&mut self) {
        assert!(
            self.len.is_power_of_two() && self.len >= 2,
            "equality table must have a nontrivial power-of-two length"
        );
        let half = self.len / 2;
        for coefficient in 0..<E as ExtField<F>>::EXT_DEGREE {
            let values = self.coefficient_slice_mut(coefficient);
            for row in 0..half {
                values[row] += values[row + half];
            }
        }
        self.truncate(half);
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
    use super::{
        compute_product_round_scalar, fold_and_compute_product_round_scalar,
        fold_first_variable_scalar, EvaluationTable,
    };
    use akita_algebra::poly::fold_evals_in_place;
    use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
    use akita_field::{
        Ext2, ExtField, FieldCore, FpExt4, FpExt8, Prime128Offset275, Prime32Offset99,
    };

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    fn value(row: usize) -> E {
        E::from_base_fn(|coefficient| F::from_u64((100 * coefficient + row) as u64))
    }

    fn check_product_round<B, T>(witness: Vec<T>, factor: Vec<T>)
    where
        B: FieldCore,
        T: ExtField<B> + HasUnreducedOps,
    {
        let mut constant = T::zero();
        let mut quadratic = T::zero();
        for row in 0..witness.len() / 2 {
            let witness_0 = witness[2 * row];
            let witness_1 = witness[2 * row + 1];
            let factor_0 = factor[2 * row];
            let factor_1 = factor[2 * row + 1];
            constant += witness_0 * factor_0;
            quadratic += (witness_1 - witness_0) * (factor_1 - factor_0);
        }

        let witness = EvaluationTable::<B, T>::from_multilinear_evaluations(&witness)
            .expect("valid witness table");
        let factor = EvaluationTable::<B, T>::from_multilinear_evaluations(&factor)
            .expect("valid factor table");
        assert_eq!(
            compute_product_round_scalar(&witness, &factor),
            (constant, quadratic)
        );
    }

    fn check_fused_product_round<B, T>(witness: Vec<T>, factor: Vec<T>, challenge: T)
    where
        B: FieldCore,
        T: ExtField<B> + HasOptimizedFold + HasUnreducedOps,
    {
        let mut expected_witness = EvaluationTable::<B, T>::from_multilinear_evaluations(&witness)
            .expect("valid witness table");
        let mut expected_factor = EvaluationTable::<B, T>::from_multilinear_evaluations(&factor)
            .expect("valid factor table");
        fold_first_variable_scalar(&mut expected_witness, challenge);
        fold_first_variable_scalar(&mut expected_factor, challenge);
        let expected = compute_product_round_scalar(&expected_witness, &expected_factor);

        let mut actual_witness = EvaluationTable::<B, T>::from_multilinear_evaluations(&witness)
            .expect("valid witness table");
        let mut actual_factor = EvaluationTable::<B, T>::from_multilinear_evaluations(&factor)
            .expect("valid factor table");
        let actual = fold_and_compute_product_round_scalar(
            &mut actual_witness,
            &mut actual_factor,
            challenge,
        );

        assert_eq!(actual, expected);
        assert_eq!(actual_witness, expected_witness);
        assert_eq!(actual_factor, expected_factor);
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
    fn table_array_constructor_matches_independent_tables() {
        let actual = EvaluationTable::<F, E>::from_multilinear_evaluation_array_fn::<3, _>(
            16,
            |table, row| value(row + 100 * table),
        )
        .unwrap();
        let expected = std::array::from_fn(|table| {
            EvaluationTable::<F, E>::from_multilinear_evaluation_fn(16, |row| {
                value(row + 100 * table)
            })
            .unwrap()
        });
        assert_eq!(actual, expected);
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
    fn mutable_coefficient_slices_are_disjoint_and_live() {
        let mut table = EvaluationTable::<F, E>::from_evaluation_fn(8, value);
        table.truncate(3);
        let coefficients = table.coefficient_slices_mut::<4>();
        for (coefficient, slice) in coefficients.into_iter().enumerate() {
            assert_eq!(slice.len(), 3);
            slice[1] = F::from_u64((900 + coefficient) as u64);
        }

        for coefficient in 0..4 {
            assert_eq!(
                table.coefficient_slice(coefficient)[1],
                F::from_u64((900 + coefficient) as u64)
            );
        }
    }

    #[test]
    fn stored_evaluation_can_be_replaced() {
        let mut table = EvaluationTable::<F, E>::from_evaluation_fn(5, value);
        table.set_evaluation(2, value(19));
        assert_eq!(table.evaluation(2), value(19));
        assert_eq!(table.evaluation(1), value(1));
        assert_eq!(table.evaluation(3), value(3));
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
    fn scalar_fold_matches_logical_order_for_every_round() {
        let mut logical: Vec<_> = (0..16).map(value).collect();
        let mut table = EvaluationTable::<F, E>::from_multilinear_evaluations(&logical)
            .expect("valid multilinear length");
        let challenges = [value(31), value(37), value(41), value(43)];

        for challenge in challenges {
            fold_evals_in_place(&mut logical, challenge);
            fold_first_variable_scalar(&mut table, challenge);

            assert_eq!(table.len(), logical.len());
            for stored_row in 0..table.len() {
                let logical_row = EvaluationTable::<F, E>::logical_row(stored_row, table.len());
                assert_eq!(table.evaluation(stored_row), logical[logical_row]);
            }
        }

        assert_eq!(table.len(), 1);
        assert_eq!(table.evaluation(0), logical[0]);
    }

    #[test]
    fn scalar_fold_matches_random_logical_tables_through_twenty_variables() {
        let mut state = 0x8b8b_8b8b_3141_5926u64;
        let mut random_value = || {
            E::from_base_fn(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                F::from_u64(state)
            })
        };

        for num_vars in 1..=20 {
            let mut logical: Vec<_> = (0..1usize << num_vars).map(|_| random_value()).collect();
            let challenges: Vec<_> = (0..num_vars).map(|_| random_value()).collect();
            let mut table = EvaluationTable::<F, E>::from_multilinear_evaluations(&logical)
                .expect("valid multilinear length");

            for challenge in challenges {
                fold_evals_in_place(&mut logical, challenge);
                fold_first_variable_scalar(&mut table, challenge);
                assert_eq!(table.len(), logical.len());
                for stored_row in 0..table.len() {
                    let logical_row = EvaluationTable::<F, E>::logical_row(stored_row, table.len());
                    assert_eq!(table.evaluation(stored_row), logical[logical_row]);
                }
            }
        }
    }

    #[test]
    fn product_round_matches_logical_pairs_with_delayed_reduction() {
        const { assert!(E::DELAYED_PRODUCT_SUM_IS_EXACT) };
        for len in [2, 4, 8, 16, 32] {
            check_product_round::<F, E>(
                (0..len).map(value).collect(),
                (0..len).map(|row| value(row + 37)).collect(),
            );
        }
    }

    #[test]
    fn product_round_matches_logical_pairs_with_direct_reduction() {
        type G = Prime128Offset275;
        const { assert!(!G::DELAYED_PRODUCT_SUM_IS_EXACT) };
        for len in [2, 4, 8, 16, 32] {
            check_product_round::<G, G>(
                (0..len)
                    .map(|row| G::from_u64((3 * row + 11) as u64))
                    .collect(),
                (0..len)
                    .map(|row| G::from_u64((7 * row + 19) as u64))
                    .collect(),
            );
        }
    }

    #[test]
    fn fused_product_round_matches_separate_delayed_operations() {
        for len in [4, 8, 16, 32] {
            check_fused_product_round::<F, E>(
                (0..len).map(value).collect(),
                (0..len).map(|row| value(row + 37)).collect(),
                value(len + 73),
            );
        }
    }

    #[test]
    fn fused_product_round_matches_separate_direct_operations() {
        type G = Prime128Offset275;
        for len in [4, 8, 16, 32] {
            check_fused_product_round::<G, G>(
                (0..len)
                    .map(|row| G::from_u64((3 * row + 11) as u64))
                    .collect(),
                (0..len)
                    .map(|row| G::from_u64((7 * row + 19) as u64))
                    .collect(),
                G::from_u64((len + 29) as u64),
            );
        }
    }

    #[test]
    #[should_panic(expected = "product round tables must have equal lengths")]
    fn product_round_rejects_mismatched_tables() {
        let witness = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(4, value)
            .expect("valid witness table");
        let factor = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(2, value)
            .expect("valid factor table");
        compute_product_round_scalar(&witness, &factor);
    }

    #[test]
    #[should_panic(expected = "evaluation table must have at least two rows")]
    fn scalar_fold_rejects_one_row_table() {
        let mut table = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(1, value)
            .expect("valid one-row multilinear table");
        fold_first_variable_scalar(&mut table, value(3));
    }

    #[test]
    #[should_panic(expected = "evaluation table length must be a power of two")]
    fn scalar_fold_rejects_truncated_non_power_of_two_table() {
        let mut table = EvaluationTable::<F, E>::from_multilinear_evaluation_fn(4, value)
            .expect("valid multilinear table");
        table.truncate(3);
        fold_first_variable_scalar(&mut table, value(3));
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

    #[test]
    fn summing_equality_halves_removes_the_first_coordinate() {
        let point = [value(31), value(47), value(59), value(71)];
        let logical = akita_algebra::eq_poly::EqPolynomial::evals(&point).unwrap();
        let suffix = akita_algebra::eq_poly::EqPolynomial::evals(&point[1..]).unwrap();
        let mut table = EvaluationTable::<F, E>::from_multilinear_evaluations(&logical).unwrap();
        table.sum_first_variable_halves();
        let expected = EvaluationTable::<F, E>::from_multilinear_evaluations(&suffix).unwrap();
        assert_eq!(table, expected);
    }
}
