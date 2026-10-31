//! Factorized evaluation of the recomposition and column compression events.
//!
//! The builder emits one event per (gadget bit, source row, coefficient block)
//! for every recomposition, and one event per column for every compression
//! row. Each such family is an arithmetic progression of aligned events whose
//! scalars are a common row factor times a shared table (the lifted gadget
//! powers, or one evaluated matrix's columns). This module records each family
//! as one [`CompressionEventGroup`] and evaluates the multilinear extension of
//! its events by factor:
//!
//! `sum_j sigma * y_j * L * eq(R, h0 + j T) = sigma * L * S`,
//!
//! where `sigma` is the row factor, `y_j` the table, `L` the shared alpha low
//! factor over the `log2(count)` low coordinates, `R` the remaining (high)
//! coordinates, `h0` the high index of the first event and `T` the high
//! stride. `S` is computed by dyadic blocks when `T` is a power of two (a
//! geometric table closes each block as a product of `log` factors, any other
//! table caches the block's multilinear extension) or otherwise by a loop over
//! the per-event equality window.
//!
//! Every factorization splits disjoint digit positions of one index, so the
//! value is exactly the multilinear extension of the dense table at arbitrary
//! field points; it never multiplies the extensions of two tables over the
//! same variables. A family that fails any precondition is pushed event by
//! event, so the accepted inputs and the errors are those of the per-event
//! path. Prover-side materialization expands the groups back into events.

use super::*;

/// Cached dyadic block extensions keyed by (table, offset, block bits, first
/// point coordinate).
type BlockCache<E> = Vec<((usize, usize, usize, usize), E)>;

/// Shared value table of one or more groups.
#[derive(Clone, Debug)]
pub(super) struct GroupTable<E: Field> {
    values: Vec<E>,
    /// `beta^(2^i)` for `i <= floor(log2(len))` when the table is verified
    /// geometric (`values[0] == 1`, `values[j] == values[j - 1] * beta`).
    ratio_squares: Option<Vec<E>>,
    /// Every entry is nonzero.
    all_nonzero: bool,
}

impl<E: Field> GroupTable<E> {
    fn new(values: Vec<E>, check_geometric: bool) -> Self {
        let all_nonzero = values.iter().all(|value| !value.is_zero());
        let ratio_squares = if check_geometric {
            Self::geometric_squares(&values)
        } else {
            None
        };
        Self {
            values,
            ratio_squares,
            all_nonzero,
        }
    }

    fn geometric_squares(values: &[E]) -> Option<Vec<E>> {
        let first = *values.first()?;
        if first != E::one() {
            return None;
        }
        let beta = values.get(1).copied().unwrap_or_else(E::one);
        let mut previous = first;
        for &value in values.iter().skip(1) {
            if value != previous * beta {
                return None;
            }
            previous = value;
        }
        let levels = usize::BITS as usize - values.len().leading_zeros() as usize;
        let mut squares = Vec::with_capacity(levels.max(1));
        let mut square = beta;
        for _ in 0..levels.max(1) {
            squares.push(square);
            let next = square * square;
            square = next;
        }
        Some(squares)
    }
}

/// One arithmetic progression of aligned compression events:
/// `(physical_start + j * stride, coefficient_count, alpha_exponent_start,
/// scalar * table[j])` for `j < len`.
#[derive(Clone, Debug)]
pub(super) struct CompressionEventGroup<E: Field> {
    physical_start: usize,
    stride: usize,
    len: usize,
    coefficient_count: usize,
    alpha_exponent_start: usize,
    scalar: E,
    table: usize,
}

/// Group storage carried by [`CompressionRelationWeights`].
#[derive(Clone, Debug)]
pub(super) struct Groups<E: Field> {
    groups: Vec<CompressionEventGroup<E>>,
    tables: Vec<GroupTable<E>>,
    /// `(gadget levels, table index)`.
    gadget_tables: Vec<(usize, usize)>,
    /// `(evaluated matrix index, table index)`.
    column_tables: Vec<(usize, usize)>,
}

#[allow(clippy::derivable_impls)]
impl<E: Field> Default for Groups<E> {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            tables: Vec::new(),
            gadget_tables: Vec::new(),
            column_tables: Vec::new(),
        }
    }
}

/// Equality factor of `value` over `coordinates` (little-endian digits); zero
/// when `value` is outside the coordinates' domain, as `eq_eval_at_index`.
fn eq_bits<E: Field>(coordinates: &[E], value: usize) -> E {
    if coordinates.len() < usize::BITS as usize && value >> coordinates.len() != 0 {
        return E::zero();
    }
    coordinates
        .iter()
        .enumerate()
        .fold(E::one(), |acc, (bit, &coordinate)| {
            if bit < usize::BITS as usize && (value >> bit) & 1 == 1 {
                acc * coordinate
            } else {
                acc * (E::one() - coordinate)
            }
        })
}

impl<E: Field> CompressionRelationWeights<E> {
    /// Canonical admission test of one event (the checks of `push`), without
    /// recording it.
    fn admits_event(
        &self,
        physical_start: usize,
        coefficient_count: usize,
        alpha_exponent_start: usize,
    ) -> bool {
        let (Some(physical_end), Some(alpha_end)) = (
            physical_start.checked_add(coefficient_count),
            alpha_exponent_start.checked_add(coefficient_count),
        ) else {
            return false;
        };
        coefficient_count != 0
            && coefficient_count.is_power_of_two()
            && physical_start.is_multiple_of(self.coefficient_block_len)
            && coefficient_count.is_multiple_of(self.coefficient_block_len)
            && alpha_exponent_start.is_multiple_of(self.coefficient_block_len)
            && physical_end <= self.physical_field_len
            && alpha_end <= self.alpha_powers.len()
    }

    /// Whether every element `j < len` of a progression would be admitted by
    /// `push` and takes the aligned evaluation path. The admission checks are
    /// affine or monotone in `j`, so the first two and the last elements
    /// decide them.
    fn admits_progression(
        &self,
        physical_start: usize,
        stride: usize,
        len: usize,
        coefficient_count: usize,
        alpha_exponent_start: usize,
    ) -> bool {
        let Some(last_offset) = len.checked_sub(1).and_then(|last| last.checked_mul(stride)) else {
            return len == 0;
        };
        let Some(last_start) = physical_start.checked_add(last_offset) else {
            return false;
        };
        let second_ok = len < 2
            || physical_start.checked_add(stride).is_some_and(|second| {
                self.admits_event(second, coefficient_count, alpha_exponent_start)
            });
        coefficient_count.is_power_of_two()
            && physical_start.is_multiple_of(coefficient_count)
            && (len < 2 || stride.is_multiple_of(coefficient_count))
            && self.admits_event(physical_start, coefficient_count, alpha_exponent_start)
            && second_ok
            && self.admits_event(last_start, coefficient_count, alpha_exponent_start)
    }

    fn gadget_table<F>(&mut self, levels: usize) -> usize
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        if let Some(&(_, index)) = self
            .groups
            .gadget_tables
            .iter()
            .find(|(cached, _)| *cached == levels)
        {
            return index;
        }
        let values = gadget_row_scalars::<F>(levels, 1)
            .into_iter()
            .map(<E as ExtField<F>>::lift_base)
            .collect::<Vec<_>>();
        self.groups.tables.push(GroupTable::new(values, true));
        let index = self.groups.tables.len() - 1;
        self.groups.gadget_tables.push((levels, index));
        index
    }

    /// Record one compression row's column events (`row_weight * columns[j]` at
    /// `span_start + j * ring_dimension`) as one group, or by the canonical loop.
    pub(super) fn push_column_group(
        &mut self,
        span_start: usize,
        ring_dimension: usize,
        input_width: usize,
        row_weight: E,
        matrix_index: usize,
        columns: &[E],
    ) -> Result<(), AkitaError> {
        #[cfg(feature = "compression-weights-crosscheck")]
        self.crosscheck_columns(span_start, ring_dimension, input_width, row_weight, columns)?;
        let len = columns
            .get(..input_width)
            .and_then(|live| live.iter().rposition(|value| !value.is_zero()))
            .map_or(0, |last| last + 1);
        let grouped = columns.len() >= input_width
            && (row_weight.is_zero()
                || len == 0
                || self.admits_progression(span_start, ring_dimension, len, ring_dimension, 0));
        if !grouped {
            return self.push_columns_canonical(
                span_start,
                ring_dimension,
                input_width,
                row_weight,
                columns,
            );
        }
        if row_weight.is_zero() || len == 0 {
            // Every canonical scalar is zero, so `push` records nothing.
            return Ok(());
        }
        let table = if let Some(&(_, index)) = self
            .groups
            .column_tables
            .iter()
            .find(|(cached, _)| *cached == matrix_index)
        {
            index
        } else {
            self.groups.tables.push(GroupTable::new(
                columns.get(..input_width).unwrap_or(columns).to_vec(),
                false,
            ));
            let index = self.groups.tables.len() - 1;
            self.groups.column_tables.push((matrix_index, index));
            index
        };
        self.groups.groups.push(CompressionEventGroup {
            physical_start: span_start,
            stride: ring_dimension,
            len,
            coefficient_count: ring_dimension,
            alpha_exponent_start: 0,
            scalar: row_weight,
            table,
        });
        Ok(())
    }

    /// The per-event column loop of `build_compression_relation_weights`: one
    /// event per column.
    fn push_columns_canonical(
        &mut self,
        span_start: usize,
        ring_dimension: usize,
        input_width: usize,
        row_weight: E,
        columns: &[E],
    ) -> Result<(), AkitaError> {
        for column in 0..input_width {
            let start = column * ring_dimension;
            self.push(
                span_start + start,
                ring_dimension,
                0,
                row_weight
                    * columns.get(column).copied().ok_or_else(|| {
                        AkitaError::Internal(
                            "compression column exceeds the evaluated column table".into(),
                        )
                    })?,
            )?;
        }
        Ok(())
    }

    /// Canonical events of every group, zero scalars dropped as `push` does.
    pub(super) fn expanded_group_events(
        &self,
    ) -> Result<Vec<CompressionRelationEvent<E>>, AkitaError> {
        let mut events = Vec::new();
        for group in &self.groups.groups {
            let table =
                self.groups.tables.get(group.table).ok_or_else(|| {
                    AkitaError::Internal("compression group table is missing".into())
                })?;
            let values = table.values.get(..group.len).ok_or_else(|| {
                AkitaError::Internal("compression group exceeds its table".into())
            })?;
            for (index, &value) in values.iter().enumerate() {
                let scalar = group.scalar * value;
                if scalar.is_zero() {
                    continue;
                }
                events.push(CompressionRelationEvent {
                    physical_start: group.physical_start + index * group.stride,
                    coefficient_count: group.coefficient_count,
                    alpha_exponent_start: group.alpha_exponent_start,
                    scalar,
                });
            }
        }
        Ok(events)
    }

    /// Sum of every group's events' extension at `point`, by factor.
    pub(super) fn evaluate_groups_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        let mut low_factors = Vec::<((usize, usize), E)>::new();
        let mut windows = Vec::<(usize, OffsetEqWindow<E>)>::new();
        let mut blocks = BlockCache::<E>::new();
        let mut evaluation = E::zero();
        for group in &self.groups.groups {
            let count = group.coefficient_count;
            let low_bits = count.trailing_zeros() as usize;
            let key = (group.alpha_exponent_start, count);
            let low_factor =
                if let Some((_, value)) = low_factors.iter().find(|(cached, _)| *cached == key) {
                    *value
                } else {
                    let powers = group
                        .alpha_exponent_start
                        .checked_add(count)
                        .and_then(|end| self.alpha_powers.get(group.alpha_exponent_start..end))
                        .ok_or_else(|| {
                            AkitaError::Internal(
                                "compression group exceeds the alpha power table".into(),
                            )
                        })?;
                    let low_point = point.get(..low_bits).ok_or_else(|| {
                        AkitaError::Internal("compression group low point exceeds the point".into())
                    })?;
                    let value = multilinear_eval(powers, low_point)?;
                    low_factors.push((key, value));
                    value
                };
            let high_point = point.get(low_bits..).ok_or_else(|| {
                AkitaError::Internal("compression group high point exceeds the point".into())
            })?;
            let table =
                self.groups.tables.get(group.table).ok_or_else(|| {
                    AkitaError::Internal("compression group table is missing".into())
                })?;
            let values = table.values.get(..group.len).ok_or_else(|| {
                AkitaError::Internal("compression group exceeds its table".into())
            })?;
            let first_high = group.physical_start >> low_bits;
            let high_stride = group.stride >> low_bits;
            let sum = if group.len == 1 || high_stride.is_power_of_two() {
                let stride_bits = if group.len == 1 {
                    0
                } else {
                    high_stride.trailing_zeros() as usize
                };
                Self::dyadic_sum(
                    table,
                    values,
                    group.table,
                    high_point,
                    low_bits,
                    first_high,
                    stride_bits,
                    &mut blocks,
                )?
            } else {
                let window_index = if let Some(index) =
                    windows.iter().position(|(cached, _)| *cached == low_bits)
                {
                    index
                } else {
                    let balanced_low_bits = high_point.len().div_ceil(2);
                    windows.push((
                        low_bits,
                        OffsetEqWindow::with_low_bits(high_point, balanced_low_bits)?,
                    ));
                    windows.len() - 1
                };
                let window = &windows
                    .get(window_index)
                    .ok_or_else(|| {
                        AkitaError::Internal("compression group equality window is missing".into())
                    })?
                    .1;
                let mut sum = E::zero();
                let mut index = first_high;
                for &value in values {
                    sum += value * window.eval(index);
                    index = index.wrapping_add(high_stride);
                }
                sum
            };
            evaluation += group.scalar * low_factor * sum;
        }
        Ok(evaluation)
    }

    /// `sum_j values[j] * eq(high_point, first_high + j * 2^stride_bits)` by
    /// dyadic blocks.
    ///
    /// The low `stride_bits` digits of every index equal those of
    /// `first_high`, so their equality factor is shared. The run
    /// `(first_high >> stride_bits) + j` is split into maximal aligned blocks of
    /// `2^b` indices; inside a block the low `b` digits range over all values
    /// and the high digits are fixed, so the block contributes
    /// `eq(rest, block index) * MLE(block values)(block point)`. For a
    /// geometric table `values[j] = beta^j` the block extension is
    /// `beta^offset * prod_i ((1 - r_i) + r_i * beta^(2^i))`.
    #[allow(clippy::too_many_arguments)]
    fn dyadic_sum(
        table: &GroupTable<E>,
        values: &[E],
        table_index: usize,
        high_point: &[E],
        low_bits: usize,
        first_high: usize,
        stride_bits: usize,
        blocks: &mut BlockCache<E>,
    ) -> Result<E, AkitaError> {
        let (stride_point, run_point) = if stride_bits <= high_point.len() {
            high_point.split_at(stride_bits)
        } else {
            return Err(AkitaError::Internal(
                "compression group stride exceeds the point".into(),
            ));
        };
        let low_mask = (1usize << stride_bits) - 1;
        let stride_factor = eq_bits(stride_point, first_high & low_mask);
        let start = first_high >> stride_bits;
        let end = start
            .checked_add(values.len())
            .ok_or_else(|| AkitaError::Internal("compression group run overflow".into()))?;
        let mut sum = E::zero();
        let mut position = start;
        let mut ratio_power = E::one();
        while position < end {
            let remaining = end - position;
            let size_bits = (usize::BITS - 1 - remaining.leading_zeros()) as usize;
            let block_bits = if position == 0 {
                size_bits
            } else {
                size_bits.min(position.trailing_zeros() as usize)
            };
            let block_len = 1usize << block_bits;
            let offset = position - start;
            let (block_point, rest_point) = if block_bits <= run_point.len() {
                run_point.split_at(block_bits)
            } else {
                return Err(AkitaError::Internal(
                    "compression group block exceeds the point".into(),
                ));
            };
            let high_factor = eq_bits(rest_point, position >> block_bits);
            let block = if let Some(squares) = &table.ratio_squares {
                let product = block_point.iter().zip(squares).fold(
                    E::one(),
                    |acc, (&coordinate, &square)| {
                        acc * ((E::one() - coordinate) + coordinate * square)
                    },
                );
                ratio_power * product
            } else {
                let key = (table_index, offset, block_bits, low_bits + stride_bits);
                if let Some((_, value)) = blocks.iter().find(|(cached, _)| *cached == key) {
                    *value
                } else {
                    let slice = values.get(offset..offset + block_len).ok_or_else(|| {
                        AkitaError::Internal("compression group block exceeds its table".into())
                    })?;
                    let value = multilinear_eval(slice, block_point)?;
                    blocks.push((key, value));
                    value
                }
            };
            sum += high_factor * block;
            if let Some(squares) = &table.ratio_squares {
                // Advance beta^(position - start) by beta^(2^block_bits) when
                // another block follows.
                if position + block_len < end {
                    let square = squares.get(block_bits).ok_or_else(|| {
                        AkitaError::Internal("compression group ratio square is missing".into())
                    })?;
                    ratio_power *= *square;
                }
            }
            position += block_len;
        }
        Ok(stride_factor * sum)
    }
}

/// Recomposition events grouped over the gadget bits, or the canonical loop on any failure.
#[cfg(not(feature = "compression-weights-crosscheck"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn push_recomposition<F, E>(
    weights: &mut CompressionRelationWeights<E>,
    span_start: usize,
    digit_ring_dim: usize,
    source_coefficients: usize,
    source_ring_dim: usize,
    source_row_start: usize,
    source_row_count: usize,
    field_bits: usize,
    row_weights: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    let planned = plan_recomposition_groups::<F, E>(
        weights,
        span_start,
        digit_ring_dim,
        source_coefficients,
        source_ring_dim,
        source_row_start,
        source_row_count,
        field_bits,
        row_weights,
    );
    match planned {
        Some(groups) => {
            weights.groups.groups.extend(groups);
            Ok(())
        }
        None => push_recomposition_canonical::<F, E>(
            weights,
            span_start,
            digit_ring_dim,
            source_coefficients,
            source_ring_dim,
            source_row_start,
            source_row_count,
            field_bits,
            row_weights,
        ),
    }
}

/// Groups for one canonical `push_recomposition` call, or `None` when the
/// canonical loop must run (any error, overflow, misalignment, or a table
/// with a zero entry).
#[allow(clippy::too_many_arguments)]
fn plan_recomposition_groups<F, E>(
    weights: &mut CompressionRelationWeights<E>,
    span_start: usize,
    digit_ring_dim: usize,
    source_coefficients: usize,
    source_ring_dim: usize,
    source_row_start: usize,
    source_row_count: usize,
    field_bits: usize,
    row_weights: &[E],
) -> Option<Vec<CompressionEventGroup<E>>>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    // A zero source ring dimension records no events, but the canonical loop
    // still reads every row weight; it runs so a short table fails the same way.
    if digit_ring_dim == 0
        || source_ring_dim == 0
        || !source_ring_dim.is_multiple_of(digit_ring_dim)
        || source_row_count.checked_mul(source_ring_dim) != Some(source_coefficients)
    {
        return None;
    }
    let mut groups = Vec::new();
    if field_bits == 0 || source_row_count == 0 {
        return Some(groups);
    }
    // The canonical loop computes every start with checked additions after an
    // unchecked `bit * source_coefficients`; the largest one must not overflow.
    let largest = (field_bits - 1)
        .checked_mul(source_coefficients)
        .and_then(|offset| span_start.checked_add(offset))
        .and_then(|start| start.checked_add((source_row_count - 1).checked_mul(source_ring_dim)?))
        .and_then(|start| start.checked_add(source_ring_dim - digit_ring_dim));
    largest?;
    let table = weights.gadget_table::<F>(field_bits);
    if !weights.groups.tables.get(table)?.all_nonzero {
        return None;
    }
    for source_row in 0..source_row_count {
        let row_weight = *row_weights.get(source_row_start.checked_add(source_row)?)?;
        if row_weight.is_zero() {
            continue;
        }
        for coefficient_start in (0..source_ring_dim).step_by(digit_ring_dim) {
            let physical_start = span_start
                .checked_add(source_row * source_ring_dim)?
                .checked_add(coefficient_start)?;
            if !weights.admits_progression(
                physical_start,
                source_coefficients,
                field_bits,
                digit_ring_dim,
                coefficient_start,
            ) {
                return None;
            }
            groups.push(CompressionEventGroup {
                physical_start,
                stride: source_coefficients,
                len: field_bits,
                coefficient_count: digit_ring_dim,
                alpha_exponent_start: coefficient_start,
                scalar: -row_weight,
                table,
            });
        }
    }
    Some(groups)
}

/// Recomposition events grouped over the gadget bits, or the canonical loop on
/// any failure, as without `compression-weights-crosscheck`. Each call also
/// records its outcome (with the reason for a fallback) and compares the
/// grouped events with the canonical loop's on the same arguments.
#[cfg(feature = "compression-weights-crosscheck")]
#[allow(clippy::too_many_arguments)]
pub(super) fn push_recomposition<F, E>(
    weights: &mut CompressionRelationWeights<E>,
    span_start: usize,
    digit_ring_dim: usize,
    source_coefficients: usize,
    source_ring_dim: usize,
    source_row_start: usize,
    source_row_count: usize,
    field_bits: usize,
    row_weights: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    use super::crosscheck::BuilderOutcome;
    let planned = plan_recomposition_groups::<F, E>(
        weights,
        span_start,
        digit_ring_dim,
        source_coefficients,
        source_ring_dim,
        source_row_start,
        source_row_count,
        field_bits,
        row_weights,
    );
    let (outcome, compared) = match &planned {
        Some(groups) => {
            let mut grouped = weights.crosscheck_twin(true);
            grouped.groups.groups.clone_from(groups);
            let mut canonical = weights.crosscheck_twin(false);
            let canonical_ok = push_recomposition_canonical::<F, E>(
                &mut canonical,
                span_start,
                digit_ring_dim,
                source_coefficients,
                source_ring_dim,
                source_row_start,
                source_row_count,
                field_bits,
                row_weights,
            )
            .is_ok();
            (
                BuilderOutcome::RecompositionGrouped {
                    groups: groups.len(),
                },
                Some(canonical_ok && grouped.crosscheck_same_events(&canonical)),
            )
        }
        None => (
            BuilderOutcome::RecompositionFallback(recomposition_fallback_reason::<F, E>(
                weights,
                span_start,
                digit_ring_dim,
                source_coefficients,
                source_ring_dim,
                source_row_start,
                source_row_count,
                field_bits,
                row_weights,
            )),
            None,
        ),
    };
    super::crosscheck::record_builder(weights.crosscheck_key(), outcome, compared)?;
    match planned {
        Some(groups) => {
            weights.groups.groups.extend(groups);
            Ok(())
        }
        None => push_recomposition_canonical::<F, E>(
            weights,
            span_start,
            digit_ring_dim,
            source_coefficients,
            source_ring_dim,
            source_row_start,
            source_row_count,
            field_bits,
            row_weights,
        ),
    }
}

/// Why `plan_recomposition_groups` declined a call: its preconditions, checked
/// in its order.
#[cfg(feature = "compression-weights-crosscheck")]
#[allow(clippy::too_many_arguments)]
fn recomposition_fallback_reason<F, E>(
    weights: &CompressionRelationWeights<E>,
    span_start: usize,
    digit_ring_dim: usize,
    source_coefficients: usize,
    source_ring_dim: usize,
    source_row_start: usize,
    source_row_count: usize,
    field_bits: usize,
    row_weights: &[E],
) -> &'static str
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    if digit_ring_dim == 0
        || source_ring_dim == 0
        || !source_ring_dim.is_multiple_of(digit_ring_dim)
        || source_row_count.checked_mul(source_ring_dim) != Some(source_coefficients)
    {
        return "recomposition_geometry_malformed";
    }
    let largest = field_bits
        .saturating_sub(1)
        .checked_mul(source_coefficients)
        .and_then(|offset| span_start.checked_add(offset))
        .and_then(|start| {
            start.checked_add(
                source_row_count
                    .saturating_sub(1)
                    .checked_mul(source_ring_dim)?,
            )
        })
        .and_then(|start| start.checked_add(source_ring_dim - digit_ring_dim));
    if largest.is_none() {
        return "recomposition_address_overflow";
    }
    if gadget_row_scalars::<F>(field_bits, 1)
        .into_iter()
        .any(|power| <E as ExtField<F>>::lift_base(power).is_zero())
    {
        return "recomposition_zero_gadget_entry";
    }
    for source_row in 0..source_row_count {
        let Some(row_weight) = source_row_start
            .checked_add(source_row)
            .and_then(|index| row_weights.get(index))
        else {
            return "recomposition_row_weight_missing";
        };
        if row_weight.is_zero() {
            continue;
        }
        for coefficient_start in (0..source_ring_dim).step_by(digit_ring_dim) {
            let physical_start = source_row
                .checked_mul(source_ring_dim)
                .and_then(|offset| span_start.checked_add(offset))
                .and_then(|start| start.checked_add(coefficient_start));
            let Some(physical_start) = physical_start else {
                return "recomposition_address_overflow";
            };
            if !weights.admits_progression(
                physical_start,
                source_coefficients,
                field_bits,
                digit_ring_dim,
                coefficient_start,
            ) {
                return "recomposition_progression_not_admitted";
            }
        }
    }
    "recomposition_unclassified"
}

/// Nonzero scalar sums keyed by (start, coefficient count, alpha start).
#[cfg(feature = "compression-weights-crosscheck")]
type EventSums<E> = Vec<((usize, usize, usize), E)>;

#[cfg(feature = "compression-weights-crosscheck")]
impl<E: Field> CompressionRelationWeights<E> {
    /// The same geometry with no events and no groups; `with_tables` keeps the
    /// group tables and their caches.
    fn crosscheck_twin(&self, with_tables: bool) -> Self {
        let groups = if with_tables {
            Groups {
                groups: Vec::new(),
                tables: self.groups.tables.clone(),
                gadget_tables: self.groups.gadget_tables.clone(),
                column_tables: self.groups.column_tables.clone(),
            }
        } else {
            Groups::default()
        };
        Self {
            events: Vec::new(),
            alpha_powers: self.alpha_powers.clone(),
            coefficient_block_len: self.coefficient_block_len,
            physical_field_len: self.physical_field_len,
            groups,
        }
    }

    /// Identity of this table while it lives: the address of its alpha power
    /// table and its domain size. Builder outcomes are attributed by it.
    pub(super) fn crosscheck_key(&self) -> (usize, usize) {
        (self.alpha_powers.as_ptr().addr(), self.physical_field_len)
    }

    /// The same table with every group expanded into its events.
    pub(super) fn crosscheck_expanded(&self) -> Result<Self, AkitaError> {
        let mut expanded = self.crosscheck_twin(false);
        expanded.events.clone_from(&self.events);
        expanded.events.extend(self.expanded_group_events()?);
        Ok(expanded)
    }

    /// Whether two tables hold the same events, groups expanded, compared as
    /// the nonzero sums of scalars per (start, coefficient count, alpha start).
    fn crosscheck_same_events(&self, other: &Self) -> bool {
        match (self.crosscheck_event_sums(), other.crosscheck_event_sums()) {
            (Ok(mine), Ok(theirs)) => mine == theirs,
            _ => false,
        }
    }

    fn crosscheck_event_sums(&self) -> Result<EventSums<E>, AkitaError> {
        let mut events = self
            .events
            .iter()
            .cloned()
            .chain(self.expanded_group_events()?)
            .map(|event| {
                (
                    (
                        event.physical_start,
                        event.coefficient_count,
                        event.alpha_exponent_start,
                    ),
                    event.scalar,
                )
            })
            .collect::<Vec<_>>();
        events.sort_unstable_by_key(|(key, _)| *key);
        let mut sums = EventSums::<E>::with_capacity(events.len());
        for (key, scalar) in events {
            if let Some((last, sum)) = sums.last_mut() {
                if *last == key {
                    *sum += scalar;
                    continue;
                }
            }
            sums.push((key, scalar));
        }
        sums.retain(|(_, sum)| !sum.is_zero());
        Ok(sums)
    }

    /// Groups and events per evaluation tier, selected as
    /// `evaluate_groups_at_point` and `evaluate_events_at_point` select them.
    pub(super) fn crosscheck_tiers(&self) -> super::crosscheck::Tiers {
        let mut tiers = super::crosscheck::Tiers::default();
        for event in &self.events {
            if event.physical_start & event.coefficient_count.wrapping_sub(1) == 0 {
                tiers.ungrouped_aligned += 1;
            } else {
                tiers.ungrouped_unaligned += 1;
            }
        }
        for group in &self.groups.groups {
            let low_bits = group.coefficient_count.trailing_zeros();
            let high_stride = group.stride.checked_shr(low_bits).unwrap_or(0);
            let tier = if group.len == 1 || high_stride.is_power_of_two() {
                let geometric = self
                    .groups
                    .tables
                    .get(group.table)
                    .is_some_and(|table| table.ratio_squares.is_some());
                if geometric {
                    &mut tiers.geometric_dyadic
                } else {
                    &mut tiers.cached_dyadic
                }
            } else {
                &mut tiers.window
            };
            tier.groups += 1;
            tier.entries += group.len as u64;
        }
        tiers
    }

    /// Record the outcome of one `push_column_group` call before it runs and,
    /// unless it falls back, compare the grouped events with the canonical
    /// loop's on the same arguments.
    fn crosscheck_columns(
        &self,
        span_start: usize,
        ring_dimension: usize,
        input_width: usize,
        row_weight: E,
        columns: &[E],
    ) -> Result<(), AkitaError> {
        use super::crosscheck::{BuilderOutcome, ObserverGuard};
        let Some(_observing) = ObserverGuard::enter() else {
            // The grouped twin below re-enters `push_column_group`.
            return Ok(());
        };
        let len = columns
            .get(..input_width)
            .and_then(|live| live.iter().rposition(|value| !value.is_zero()))
            .map_or(0, |last| last + 1);
        let outcome = if columns.len() < input_width {
            BuilderOutcome::ColumnFallback("column_table_shorter_than_width")
        } else if row_weight.is_zero() || len == 0 {
            BuilderOutcome::ColumnZero
        } else if self.admits_progression(span_start, ring_dimension, len, ring_dimension, 0) {
            BuilderOutcome::ColumnGrouped
        } else {
            BuilderOutcome::ColumnFallback("column_progression_not_admitted")
        };
        let compared = if matches!(outcome, BuilderOutcome::ColumnFallback(_)) {
            None
        } else {
            let mut grouped = self.crosscheck_twin(false);
            let grouped_ok = grouped
                .push_column_group(
                    span_start,
                    ring_dimension,
                    input_width,
                    row_weight,
                    0,
                    columns,
                )
                .is_ok();
            let mut canonical = self.crosscheck_twin(false);
            let canonical_ok = canonical
                .push_columns_canonical(
                    span_start,
                    ring_dimension,
                    input_width,
                    row_weight,
                    columns,
                )
                .is_ok();
            let expected_groups = usize::from(matches!(outcome, BuilderOutcome::ColumnGrouped));
            Some(
                grouped_ok
                    && canonical_ok
                    && grouped.events.is_empty()
                    && grouped.groups.groups.len() == expected_groups
                    && grouped.crosscheck_same_events(&canonical),
            )
        };
        super::crosscheck::record_builder(self.crosscheck_key(), outcome, compared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128OffsetA7F7 as F;
    use jolt_field::{One, Zero};

    /// Deterministic full-width field elements (no external RNG).
    struct Stream(u64);

    impl Stream {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn field(&mut self) -> F {
            let high = F::from_u64(self.next_u64());
            high * F::from_u64(self.next_u64()) + F::from_u64(self.next_u64())
        }

        fn point(&mut self, len: usize) -> Vec<F> {
            (0..len).map(|_| self.field()).collect()
        }
    }

    fn empty(
        physical_field_len: usize,
        alpha_len: usize,
        block: usize,
        stream: &mut Stream,
    ) -> CompressionRelationWeights<F> {
        CompressionRelationWeights {
            events: Vec::new(),
            alpha_powers: (0..alpha_len).map(|_| stream.field()).collect(),
            coefficient_block_len: block,
            physical_field_len,
            groups: Groups::default(),
        }
    }

    /// The same table built through the canonical loops only.
    fn canonical_twin(weights: &CompressionRelationWeights<F>) -> CompressionRelationWeights<F> {
        let mut twin = weights.clone();
        twin.events.extend(weights.expanded_group_events().unwrap());
        twin.groups = Groups::default();
        twin
    }

    /// Points with random, zero and one coordinates.
    fn points(n: usize, stream: &mut Stream) -> Vec<Vec<F>> {
        let mut out = vec![
            stream.point(n),
            stream.point(n),
            vec![F::zero(); n],
            vec![F::one(); n],
        ];
        let mut mixed = stream.point(n);
        for (index, coordinate) in mixed.iter_mut().enumerate() {
            if index % 3 == 0 {
                *coordinate = F::zero();
            } else if index % 3 == 1 {
                *coordinate = F::one();
            }
        }
        out.push(mixed);
        out
    }

    fn assert_all_agree(weights: &CompressionRelationWeights<F>, stream: &mut Stream, label: &str) {
        let n = weights.physical_field_len.trailing_zeros() as usize;
        let dense = weights.materialize_dense().unwrap();
        let twin = canonical_twin(weights);
        assert_eq!(twin.materialize_dense().unwrap(), dense, "{label}: dense");
        for point in points(n, stream) {
            let expected = multilinear_eval(&dense, &point).unwrap();
            assert_eq!(
                weights.evaluate_at_point(&point).unwrap(),
                expected,
                "{label}: factorized"
            );
            assert_eq!(
                twin.evaluate_events_at_point(&point).unwrap(),
                expected,
                "{label}: canonical"
            );
        }
        assert_eq!(
            weights.clone().into_sparse_entries().unwrap(),
            twin.into_sparse_entries().unwrap(),
            "{label}: sparse entries"
        );
    }

    /// Recomposition calls of many geometries agree with the canonical loop
    /// and the dense extension at random, zero, one and mixed points.
    #[test]
    fn recomposition_groups_match_canonical() {
        let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
        // (span_start, digit, source_coefficients, source_ring_dim, rows, field_bits, n)
        let cases = [
            (0usize, 4usize, 16usize, 16usize, 1usize, 8usize, 9usize), // dyadic, aligned
            (32, 4, 16, 16, 1, 8, 9),                                   // unaligned run start
            (0, 4, 48, 16, 3, 5, 10),                                   // stride not a power of two
            (64, 2, 8, 8, 1, 7, 9),                                     // clipped run (7 bits)
            (0, 1, 2, 2, 1, 128, 9),                                    // count 1 (k = 0), 128 bits
            (16, 8, 16, 8, 2, 3, 8),                                    // two rows
            (0, 4, 16, 16, 1, 1, 8),                                    // a single bit
            (0, 4, 16, 16, 1, 0, 8),                                    // no bits at all
        ];
        for (index, &(span, digit, source, ring, rows, bits, n)) in cases.iter().enumerate() {
            let mut weights = empty(1 << n, 64, 1, &mut stream);
            let mut canonical = weights.clone();
            let row_weights = stream.point(rows + 2);
            push_recomposition::<F, F>(
                &mut weights,
                span,
                digit,
                source,
                ring,
                1,
                rows,
                bits,
                &row_weights,
            )
            .unwrap();
            push_recomposition_canonical::<F, F>(
                &mut canonical,
                span,
                digit,
                source,
                ring,
                1,
                rows,
                bits,
                &row_weights,
            )
            .unwrap();
            assert!(
                weights.events.is_empty(),
                "case {index}: every recomposition is grouped"
            );
            // A raw canonical event in the same table.
            let scalar = stream.field();
            weights.push(0, 2, 2, scalar).unwrap();
            canonical.push(0, 2, 2, scalar).unwrap();
            assert_eq!(
                weights.materialize_dense().unwrap(),
                canonical.materialize_dense().unwrap(),
                "case {index}: grouped and canonical tables"
            );
            assert_all_agree(
                &weights,
                &mut stream,
                &format!("recomposition case {index}"),
            );
        }
    }

    /// Zero row weights record nothing, as the canonical loop.
    #[test]
    fn zero_row_weight_records_nothing() {
        let mut stream = Stream(7);
        let mut weights = empty(1 << 9, 64, 1, &mut stream);
        push_recomposition::<F, F>(&mut weights, 0, 4, 16, 16, 0, 1, 8, &[F::zero()]).unwrap();
        assert!(weights.groups.groups.is_empty() && weights.events.is_empty());
        weights
            .push_column_group(0, 4, 8, F::zero(), 0, &stream.point(8))
            .unwrap();
        assert!(weights.groups.groups.is_empty() && weights.events.is_empty());
    }

    /// Column groups: shared tables, interior and trailing zeros, widths that
    /// are not powers of two, unaligned spans, and the canonical fallback.
    #[test]
    fn column_groups_match_canonical() {
        let mut stream = Stream(0x1234_5678_9abc_def1);
        let n = 10;
        let mut weights = empty(1 << n, 64, 2, &mut stream);
        let mut columns = stream.point(12);
        columns[3] = F::zero();
        columns[10] = F::zero();
        columns[11] = F::zero();
        let mut canonical = weights.clone();
        for span in [0usize, 64, 128, 256] {
            let row_weight = stream.field();
            weights
                .push_column_group(span, 4, 12, row_weight, 0, &columns)
                .unwrap();
            canonical
                .push_columns_canonical(span, 4, 12, row_weight, &columns)
                .unwrap();
        }
        let other = stream.point(5);
        let row_weight = stream.field();
        weights
            .push_column_group(512, 8, 5, row_weight, 1, &other)
            .unwrap();
        canonical
            .push_columns_canonical(512, 8, 5, row_weight, &other)
            .unwrap();
        // Span not aligned to the ring dimension: canonical events, aligned
        // to the block length so `push` admits them.
        let misaligned = stream.point(3);
        let row_weight = stream.field();
        weights
            .push_column_group(2, 4, 3, row_weight, 2, &misaligned)
            .unwrap();
        canonical
            .push_columns_canonical(2, 4, 3, row_weight, &misaligned)
            .unwrap();
        assert!(
            !weights.events.is_empty(),
            "the misaligned span takes the canonical path"
        );
        assert!(
            !weights.groups.groups.is_empty(),
            "aligned spans are grouped"
        );
        assert_eq!(
            weights.materialize_dense().unwrap(),
            canonical.materialize_dense().unwrap()
        );
        assert_all_agree(&weights, &mut stream, "columns");
    }

    /// Errors are the canonical errors: a progression leaving the domain or a
    /// misaligned block falls back to the canonical loop and fails the same way.
    #[test]
    fn errors_are_canonical() {
        let mut stream = Stream(99);
        let row_weights = stream.point(4);
        for (span, block) in [(1usize << 9, 1usize), (2, 4)] {
            let mut grouped = empty(1 << 9, 64, block, &mut stream);
            let mut canonical = grouped.clone();
            let a =
                push_recomposition::<F, F>(&mut grouped, span, 4, 16, 16, 0, 1, 8, &row_weights);
            let b = push_recomposition_canonical::<F, F>(
                &mut canonical,
                span,
                4,
                16,
                16,
                0,
                1,
                8,
                &row_weights,
            );
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            assert!(a.is_err());
        }
        let mut grouped = empty(1 << 6, 64, 1, &mut stream);
        let mut canonical = grouped.clone();
        let columns = stream.point(32);
        let a = grouped.push_column_group(0, 4, 32, F::one(), 0, &columns);
        let b = canonical.push_columns_canonical(0, 4, 32, F::one(), &columns);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert!(a.is_err());
        let a = grouped.push_column_group(0, 4, 9, F::one(), 1, &columns[..8]);
        let b = canonical.push_columns_canonical(0, 4, 9, F::one(), &columns[..8]);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    /// A zero source ring dimension records no events on either path, and a
    /// row range past the row-weight table fails with the canonical error.
    #[test]
    fn zero_ring_dimension_errors_are_canonical() {
        let mut stream = Stream(0x5eed);
        let row_weights = stream.point(2);
        for (row_start, row_count, fails) in [(0usize, 3usize, true), (1, 2, true), (0, 2, false)] {
            let mut grouped = empty(1 << 6, 64, 1, &mut stream);
            let mut canonical = grouped.clone();
            let a = push_recomposition::<F, F>(
                &mut grouped,
                0,
                4,
                0,
                0,
                row_start,
                row_count,
                8,
                &row_weights,
            );
            let b = push_recomposition_canonical::<F, F>(
                &mut canonical,
                0,
                4,
                0,
                0,
                row_start,
                row_count,
                8,
                &row_weights,
            );
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            assert_eq!(a.is_err(), fails);
            assert!(grouped.events.is_empty() && grouped.groups.groups.is_empty());
            assert!(canonical.events.is_empty());
        }
    }

    /// Direct groups exercise tiers the producers do not reach: stride zero
    /// (repeated starts), a non-geometric table under the dyadic tier, and a
    /// geometric table with ratio one.
    #[test]
    fn direct_groups_cover_every_tier() {
        let mut stream = Stream(0xdead_beef);
        let n = 9;
        let mut weights = empty(1 << n, 64, 1, &mut stream);
        weights
            .groups
            .tables
            .push(GroupTable::new(stream.point(16), true));
        weights
            .groups
            .tables
            .push(GroupTable::new(vec![F::one(); 16], true));
        assert!(weights.groups.tables[0].ratio_squares.is_none());
        assert!(weights.groups.tables[1].ratio_squares.is_some());
        let scalar = stream.field();
        for (start, stride, len, count, table) in [
            (8usize, 0usize, 5usize, 4usize, 0usize),
            (16, 4, 16, 4, 0),
            (20, 8, 13, 4, 1),
            (0, 12, 9, 4, 1),
            (24, 8, 1, 8, 0),
        ] {
            weights.groups.groups.push(CompressionEventGroup {
                physical_start: start,
                stride,
                len,
                coefficient_count: count,
                alpha_exponent_start: 4,
                scalar,
                table,
            });
        }
        assert_all_agree(&weights, &mut stream, "direct groups");
    }

    /// The product of two Boolean tables' extensions is not the extension of
    /// their product (the pitfall the factorization avoids by splitting only
    /// disjoint digit positions).
    #[test]
    fn same_variable_products_do_not_factor() {
        let table = [F::one(), F::zero()];
        let point = [F::from_u64(5)];
        let product =
            multilinear_eval(&table, &point).unwrap() * multilinear_eval(&table, &point).unwrap();
        let extension = multilinear_eval(&[F::one(), F::zero()], &point).unwrap();
        assert_ne!(product, extension);
        let groups_value = eq_bits(&point[..], 0);
        assert_eq!(groups_value, extension);
    }

    /// The cross-check rejects an evaluation whose closed form disagrees with
    /// the group's own events: a table marked geometric whose values are not.
    #[cfg(feature = "compression-weights-crosscheck")]
    #[test]
    fn crosscheck_rejects_a_wrong_closed_form() {
        let mut stream = Stream(0xc0ff_ee00);
        let mut weights = empty(1 << 9, 64, 1, &mut stream);
        let mut table = GroupTable::new(stream.point(16), false);
        table.ratio_squares = Some((0..5).map(|i| F::from_u64(3 << i)).collect());
        weights.groups.tables.push(table);
        weights.groups.groups.push(CompressionEventGroup {
            physical_start: 0,
            stride: 4,
            len: 16,
            coefficient_count: 4,
            alpha_exponent_start: 0,
            scalar: stream.field(),
            table: 0,
        });
        let error = weights.evaluate_at_point(&stream.point(9)).unwrap_err();
        assert!(format!("{error:?}").contains("compression weights cross-check"));
    }

    /// The cross-check records the builder outcomes (grouped, fallback with
    /// its reason) and the evaluation tiers of every group kind.
    #[cfg(feature = "compression-weights-crosscheck")]
    #[test]
    fn crosscheck_records_outcomes_and_tiers() {
        use crate::proof::compression_relation_weights::crosscheck::{report, CrossCheckReport};
        let mut stream = Stream(0x5eed_1234);
        let mut weights = empty(1 << 10, 64, 1, &mut stream);
        let row_weights = stream.point(3);
        let before = report();
        // Geometric dyadic, window (stride 48), and a misaligned fallback.
        push_recomposition::<F, F>(&mut weights, 0, 4, 16, 16, 0, 1, 8, &row_weights).unwrap();
        push_recomposition::<F, F>(&mut weights, 256, 4, 48, 16, 0, 3, 5, &row_weights).unwrap();
        push_recomposition::<F, F>(&mut weights, 514, 4, 16, 16, 0, 1, 8, &row_weights).unwrap();
        // One grouped compression row (cached dyadic tier).
        let columns = stream.point(6);
        weights
            .push_column_group(640, 4, 6, stream.field(), 0, &columns)
            .unwrap();
        weights.evaluate_at_point(&stream.point(10)).unwrap();
        let after = report();
        let reason = |report: &CrossCheckReport| {
            report
                .builders
                .fallback_reasons
                .iter()
                .find(|(reason, _)| *reason == "recomposition_progression_not_admitted")
                .map_or(0, |(_, count)| *count)
        };
        // Other tests may run concurrently, so every count is a lower bound.
        assert!(after.evaluations > before.evaluations);
        assert!(after.builders.recomposition_grouped >= before.builders.recomposition_grouped + 2);
        assert!(after.builders.column_grouped > before.builders.column_grouped);
        assert!(after.builders.compared >= before.builders.compared + 3);
        assert!(reason(&after) > reason(&before));
        assert!(after.tiers.geometric_dyadic.groups > before.tiers.geometric_dyadic.groups);
        assert!(after.tiers.window.groups > before.tiers.window.groups);
        assert!(after.tiers.cached_dyadic.groups > before.tiers.cached_dyadic.groups);
        assert!(after.tiers.ungrouped_unaligned > before.tiers.ungrouped_unaligned);
    }
}
