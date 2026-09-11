//! Shared compact F/H relation weights for prover materialization and verifier evaluation.

mod reduced;

pub use reduced::{
    build_reduced_compression_relation_weights, evaluate_reduced_compression_map,
    ReducedCompressionRelationWeights,
};

use akita_algebra::eq_poly::EqPolynomial;
use akita_algebra::offset_eq::{
    eval_boolean_pair_tensor_families, EqPairTensorAxis, EqPairTensorFamily, OffsetEqWindow,
};
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use std::ops::Range;

use crate::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, CommittedGroupParams,
    CompressionWitnessSpan, FpExtEncoding, RelationRowFamily, RingRelationInstance,
    RingRelationMode, WitnessLayout,
};

#[derive(Clone, Debug)]
struct CompressionRelationEvent<E: Field> {
    physical_start: usize,
    coefficient_count: usize,
    alpha_exponent_start: usize,
    scalar: E,
}

/// One alpha-evaluated prefix of the universal rank-one compression matrix.
/// F and H rows with identical map geometry share this data.
struct EvaluatedCompressionMatrix<E: Field> {
    input_width: usize,
    ring_dimension: usize,
    powers: Vec<E>,
    columns: Vec<E>,
}

/// Canonical additive relation table for every F/H compression layer.
#[derive(Clone, Debug)]
pub struct CompressionRelationWeights<E: Field> {
    events: Vec<CompressionRelationEvent<E>>,
    alpha_powers: Vec<E>,
    coefficient_block_len: usize,
    physical_field_len: usize,
}

/// Checked sparse support for the negative-binary compression digits.
#[derive(Clone, Debug)]
pub struct NegativeBinarySupport {
    intervals: Vec<Range<usize>>,
    physical_field_len: usize,
}

impl NegativeBinarySupport {
    /// Derive the canonical support from the witness layout.
    pub fn new(
        witness_layout: &WitnessLayout,
        physical_field_len: usize,
    ) -> Result<Self, AkitaError> {
        if !physical_field_len.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "negative-binary support requires a power-of-two field domain".into(),
            ));
        }
        let intervals = witness_layout.negative_binary_support_intervals();
        let mut previous_end = 0;
        for interval in &intervals {
            if interval.start >= interval.end
                || interval.start < previous_end
                || interval.end > physical_field_len
            {
                return Err(AkitaError::InvalidSetup(
                    "negative-binary support interval is malformed".into(),
                ));
            }
            previous_end = interval.end;
        }
        Ok(Self {
            intervals,
            physical_field_len,
        })
    }

    /// Materialize the support indicator for prover-side folding.
    pub fn materialize<E: Field>(&self) -> Vec<E> {
        let mut weights = vec![E::zero(); self.physical_field_len];
        for interval in &self.intervals {
            weights[interval.clone()].fill(E::one());
        }
        weights
    }

    /// Borrow the sorted live intervals of the support indicator.
    #[must_use]
    pub fn intervals(&self) -> &[Range<usize>] {
        &self.intervals
    }

    /// Evaluate the equality table anchored at `equality_point`, restricted to
    /// this support, at one full witness point.
    ///
    /// This is the single MLE
    /// `sum_{index in support} eq(equality_point, index) * eq(point, index)`,
    /// not the product of the two separately extended tables.
    #[tracing::instrument(skip_all, name = "negative_binary_support_mle")]
    pub fn evaluate_restricted_equality_at_point<E: Field>(
        &self,
        equality_point: &[E],
        point: &[E],
    ) -> Result<E, AkitaError> {
        if equality_point.len() != point.len()
            || self.physical_field_len != 1usize.checked_shl(point.len() as u32).unwrap_or(0)
        {
            return Err(AkitaError::InvalidSize {
                expected: self.physical_field_len.trailing_zeros() as usize,
                actual: equality_point.len().max(point.len()),
            });
        }
        let families = self
            .intervals
            .iter()
            .map(|interval| {
                EqPairTensorFamily::new(
                    interval.start,
                    interval.start,
                    E::one(),
                    vec![EqPairTensorAxis::unit(interval.len(), 1, 1)],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        eval_boolean_pair_tensor_families::<_, false, false>(equality_point, point, &families)
    }
}

impl<E: Field> CompressionRelationWeights<E> {
    fn push(
        &mut self,
        physical_start: usize,
        coefficient_count: usize,
        alpha_exponent_start: usize,
        scalar: E,
    ) -> Result<(), AkitaError> {
        if scalar.is_zero() {
            return Ok(());
        }
        let physical_end = physical_start
            .checked_add(coefficient_count)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression relation address overflow".into())
            })?;
        let alpha_end = alpha_exponent_start
            .checked_add(coefficient_count)
            .ok_or_else(|| AkitaError::InvalidSetup("compression alpha range overflow".into()))?;
        if coefficient_count == 0
            || !coefficient_count.is_power_of_two()
            || !physical_start.is_multiple_of(self.coefficient_block_len)
            || !coefficient_count.is_multiple_of(self.coefficient_block_len)
            || !alpha_exponent_start.is_multiple_of(self.coefficient_block_len)
            || physical_end > self.physical_field_len
            || alpha_end > self.alpha_powers.len()
        {
            return Err(AkitaError::InvalidSetup(
                "compression relation event is outside its checked domain".into(),
            ));
        }
        self.events.push(CompressionRelationEvent {
            physical_start,
            coefficient_count,
            alpha_exponent_start,
            scalar,
        });
        Ok(())
    }

    /// Materialize the complete padded linear-weight table.
    pub fn materialize_dense(&self) -> Result<Vec<E>, AkitaError> {
        let mut weights = vec![E::zero(); self.physical_field_len];
        self.accumulate_dense(&mut weights)?;
        Ok(weights)
    }

    fn accumulate_dense(&self, weights: &mut [E]) -> Result<(), AkitaError> {
        if weights.len() != self.physical_field_len {
            return Err(AkitaError::InvalidSize {
                expected: self.physical_field_len,
                actual: weights.len(),
            });
        }
        for event in &self.events {
            let alpha = self
                .alpha_powers
                .get(
                    event.alpha_exponent_start
                        ..event.alpha_exponent_start + event.coefficient_count,
                )
                .ok_or(AkitaError::InvalidProof)?;
            let target = weights
                .get_mut(event.physical_start..event.physical_start + event.coefficient_count)
                .ok_or(AkitaError::InvalidProof)?;
            for (weight, &power) in target.iter_mut().zip(alpha) {
                *weight += event.scalar * power;
            }
        }
        Ok(())
    }

    /// Consume the relation table into sorted nonzero physical entries.
    ///
    /// This is the prover-side representation: compression relations occupy a
    /// small collection of aligned witness intervals, so retaining the padded
    /// full-domain table would turn a sparse addend into an avoidable scan and
    /// allocation at every Stage-2 round.
    pub fn into_sparse_entries(self) -> Result<Vec<(usize, E)>, AkitaError> {
        let total_entries = self.events.iter().try_fold(0usize, |sum, event| {
            sum.checked_add(event.coefficient_count).ok_or_else(|| {
                AkitaError::InvalidSetup("compression sparse-entry count overflow".into())
            })
        })?;
        let mut entries = Vec::with_capacity(total_entries);
        for event in self.events {
            let alpha_end = event
                .alpha_exponent_start
                .checked_add(event.coefficient_count)
                .ok_or(AkitaError::InvalidProof)?;
            let powers = self
                .alpha_powers
                .get(event.alpha_exponent_start..alpha_end)
                .ok_or(AkitaError::InvalidProof)?;
            for (offset, &power) in powers.iter().enumerate() {
                let index = event
                    .physical_start
                    .checked_add(offset)
                    .ok_or(AkitaError::InvalidProof)?;
                entries.push((index, event.scalar * power));
            }
        }
        entries.sort_unstable_by_key(|(index, _)| *index);
        let mut sparse: Vec<(usize, E)> = Vec::with_capacity(entries.len());
        for (index, value) in entries {
            if let Some((last_index, last_value)) = sparse.last_mut() {
                if *last_index == index {
                    *last_value += value;
                    continue;
                }
            }
            sparse.push((index, value));
        }
        sparse.retain(|(_, value)| !value.is_zero());
        Ok(sparse)
    }

    /// Padded physical field domain covered by this table.
    #[must_use]
    pub fn physical_field_len(&self) -> usize {
        self.physical_field_len
    }

    /// Evaluate the table's multilinear extension at one full witness point.
    #[tracing::instrument(skip_all, name = "compression_relation_mle")]
    pub fn evaluate_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        if self.physical_field_len != 1usize.checked_shl(point.len() as u32).unwrap_or(0) {
            return Err(AkitaError::InvalidSize {
                expected: self.physical_field_len.trailing_zeros() as usize,
                actual: point.len(),
            });
        }
        let mut fallback_equality = None;
        let mut low_factor_cache = Vec::new();
        let mut high_equality_cache = Vec::<(usize, OffsetEqWindow<E>)>::new();
        let mut evaluation = E::zero();
        for event in &self.events {
            if !event.physical_start.is_multiple_of(event.coefficient_count) {
                if fallback_equality.is_none() {
                    fallback_equality = Some(OffsetEqWindow::new(point)?);
                }
                let equality = fallback_equality.as_ref().ok_or(AkitaError::InvalidProof)?;
                let alpha_end = event
                    .alpha_exponent_start
                    .checked_add(event.coefficient_count)
                    .ok_or(AkitaError::InvalidProof)?;
                let powers = self
                    .alpha_powers
                    .get(event.alpha_exponent_start..alpha_end)
                    .ok_or(AkitaError::InvalidProof)?;
                let interval =
                    powers
                        .iter()
                        .copied()
                        .enumerate()
                        .fold(E::zero(), |sum, (offset, power)| {
                            sum + power * equality.eval(event.physical_start + offset)
                        });
                evaluation += event.scalar * interval;
                continue;
            }
            let low_bits = event.coefficient_count.trailing_zeros() as usize;
            let cache_key = (event.alpha_exponent_start, event.coefficient_count);
            let low_factor = if let Some((_, value)) =
                low_factor_cache.iter().find(|(key, _)| *key == cache_key)
            {
                *value
            } else {
                let alpha_end = event
                    .alpha_exponent_start
                    .checked_add(event.coefficient_count)
                    .ok_or(AkitaError::InvalidProof)?;
                let powers = self
                    .alpha_powers
                    .get(event.alpha_exponent_start..alpha_end)
                    .ok_or(AkitaError::InvalidProof)?;
                let low_point = point.get(..low_bits).ok_or(AkitaError::InvalidProof)?;
                let value = multilinear_eval(powers, low_point)?;
                low_factor_cache.push((cache_key, value));
                value
            };
            let high_index = event.physical_start >> low_bits;
            let high_point = point.get(low_bits..).ok_or(AkitaError::InvalidProof)?;
            let high_cache_index = if let Some(index) = high_equality_cache
                .iter()
                .position(|(cached_low_bits, _)| *cached_low_bits == low_bits)
            {
                index
            } else {
                let balanced_low_bits = high_point.len().div_ceil(2);
                high_equality_cache.push((
                    low_bits,
                    OffsetEqWindow::with_low_bits(high_point, balanced_low_bits)?,
                ));
                high_equality_cache.len() - 1
            };
            let high_equality = high_equality_cache
                .get(high_cache_index)
                .ok_or(AkitaError::InvalidProof)?;
            evaluation += event.scalar * low_factor * high_equality.1.eval(high_index);
        }
        Ok(evaluation)
    }
}

#[allow(clippy::too_many_arguments)]
fn push_recomposition<F, E>(
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
    if !source_ring_dim.is_multiple_of(digit_ring_dim)
        || source_row_count.checked_mul(source_ring_dim) != Some(source_coefficients)
    {
        return Err(AkitaError::InvalidSetup(
            "compression recomposition geometry is malformed".into(),
        ));
    }
    for (bit, power) in gadget_row_scalars::<F>(field_bits, 1)
        .into_iter()
        .enumerate()
    {
        for source_row in 0..source_row_count {
            let row_weight = *row_weights
                .get(source_row_start + source_row)
                .ok_or(AkitaError::InvalidProof)?;
            for coefficient_start in (0..source_ring_dim).step_by(digit_ring_dim) {
                let physical_start = span_start
                    .checked_add(bit * source_coefficients)
                    .and_then(|start| start.checked_add(source_row * source_ring_dim))
                    .and_then(|start| start.checked_add(coefficient_start))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("compression witness offset overflow".into())
                    })?;
                weights.push(
                    physical_start,
                    digit_ring_dim,
                    coefficient_start,
                    -(row_weight * E::lift_base(power)),
                )?;
            }
        }
    }
    Ok(())
}

fn compression_span_for_row<'a>(
    witness_layout: &'a WitnessLayout,
    family: &RelationRowFamily,
) -> Result<(Option<usize>, usize, &'a CompressionWitnessSpan), AkitaError> {
    match *family {
        RelationRowFamily::CompressionF {
            group_index,
            map_index,
            ..
        } => {
            let span = witness_layout
                .compression_layers()
                .get(map_index)
                .ok_or(AkitaError::InvalidProof)?
                .f_spans()
                .iter()
                .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
                .ok_or(AkitaError::InvalidProof)?;
            Ok((Some(group_index), map_index, span))
        }
        RelationRowFamily::CompressionH { map_index, .. } => {
            let span = witness_layout
                .compression_layers()
                .get(map_index)
                .ok_or(AkitaError::InvalidProof)?
                .h_span();
            Ok((None, map_index, span))
        }
        _ => Err(AkitaError::InvalidInput(
            "relation row is not a compression row".into(),
        )),
    }
}

fn successor_compression_span(
    witness_layout: &WitnessLayout,
    group_index: Option<usize>,
    map_index: usize,
) -> Result<Option<&CompressionWitnessSpan>, AkitaError> {
    let Some(successor_index) = map_index.checked_add(1) else {
        return Err(AkitaError::InvalidSetup(
            "compression map index overflow".into(),
        ));
    };
    if successor_index >= crate::COMPRESSION_MAP_COUNT {
        return Ok(None);
    }
    let layer = witness_layout
        .compression_layers()
        .get(successor_index)
        .ok_or(AkitaError::InvalidProof)?;
    match group_index {
        Some(group_index) => layer
            .f_spans()
            .iter()
            .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
            .map(Some)
            .ok_or(AkitaError::InvalidProof),
        None => Ok(Some(layer.h_span())),
    }
}

#[allow(clippy::too_many_arguments)]
fn push_initial_recompositions<F, E>(
    weights: &mut CompressionRelationWeights<E>,
    relation_layout: &crate::RelationRhsLayout,
    lp: &CommittedGroupParams,
    opening_batch: &crate::OpeningClaimsLayout,
    witness_layout: &WitnessLayout,
    field_bits: usize,
    row_weights: &[E],
    row_families: &[RelationRowFamily],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F>,
{
    for relation_group_index in 0..relation_layout.groups.len() {
        let (group_index, plan) = relation_layout.group_compression_plan(relation_group_index)?;
        let b_range = lp.commitment_row_range(opening_batch, group_index)?;
        let stage = witness_layout
            .compression_layers()
            .first()
            .and_then(|layer| {
                layer
                    .f_spans()
                    .iter()
                    .find_map(|(candidate, span)| (*candidate == group_index).then_some(span))
            })
            .ok_or(AkitaError::InvalidProof)?;
        push_recomposition::<F, E>(
            weights,
            stage.range().start,
            stage.map().ring_dimension(),
            plan.source_coefficients(),
            relation_layout.groups[relation_group_index].role_dims.d_b(),
            b_range.start,
            b_range.len(),
            field_bits,
            row_weights,
        )?;
    }
    let d_start = row_families
        .iter()
        .position(|row| matches!(row, RelationRowFamily::Opening { .. }))
        .ok_or(AkitaError::InvalidProof)?;
    let opening_plan = relation_layout.opening_compression_plan()?;
    let opening_stage = witness_layout
        .compression_layers()
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .h_span();
    push_recomposition::<F, E>(
        weights,
        opening_stage.range().start,
        opening_stage.map().ring_dimension(),
        opening_plan.source_coefficients(),
        relation_layout.d_ring_dimension,
        d_start,
        relation_layout.n_d,
        field_bits,
        row_weights,
    )
}

/// Build the one canonical compact F/H relation table.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "build_compression_relation_weights")]
pub fn build_compression_relation_weights<F, E>(
    setup: &AkitaExpandedSetup<F>,
    instance: &RingRelationInstance<F>,
    alpha: E,
    lp: &CommittedGroupParams,
    tau1: &[E],
    witness_layout: &WitnessLayout,
    outgoing_ring_dimension: usize,
    physical_field_len: usize,
) -> Result<CompressionRelationWeights<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + Ring + ExtField<F> + MulBaseUnreduced<F>,
{
    if lp.ring_relation_mode != RingRelationMode::QuotientLift
        || !matches!(
            witness_layout.relation_quotient_layout(),
            crate::RelationQuotientLayout::QuotientLift { .. }
        )
    {
        return Err(AkitaError::InvalidSetup(
            "lifted compression weights require a quotient relation layout".into(),
        ));
    }
    let opening_batch = instance.opening_batch();
    let relation_geometry =
        crate::RelationWitnessGeometry::for_level(lp, opening_batch, instance.extension_degree())?;
    let relation_layout = relation_geometry.rhs_layout();
    let row_families = relation_layout.row_families()?;
    let row_weights = EqPolynomial::evals_prefix(tau1, row_families.len())?;
    let coefficient_block_len = lp
        .compression_relation_address_geometry(
            opening_batch,
            instance.extension_degree(),
            outgoing_ring_dimension,
            witness_layout.live_coeff_len(),
        )?
        .coefficient_block_len();
    let maximum_dimension = row_families
        .iter()
        .map(|row| row.geometry().polynomial_modulus_dimension())
        .max()
        .ok_or(AkitaError::InvalidProof)?;
    let mut weights = CompressionRelationWeights {
        events: Vec::new(),
        alpha_powers: scalar_powers(alpha, maximum_dimension),
        coefficient_block_len,
        physical_field_len,
    };
    let field_bits = usize::try_from(F::MODULUS_BITS)
        .map_err(|_| AkitaError::InvalidSetup("compression field width overflow".into()))?;
    let mut evaluated_matrices = Vec::<EvaluatedCompressionMatrix<E>>::new();
    push_initial_recompositions::<F, E>(
        &mut weights,
        relation_layout,
        lp,
        opening_batch,
        witness_layout,
        field_bits,
        &row_weights,
        &row_families,
    )?;

    for (row_index, family) in row_families.iter().enumerate() {
        if !matches!(
            family,
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        let (group_index, map_index, span) = compression_span_for_row(witness_layout, family)?;
        let map = span.map();
        let row_weight = *row_weights.get(row_index).ok_or(AkitaError::InvalidProof)?;
        let matrix_index = if let Some(index) = evaluated_matrices.iter().position(|evaluated| {
            evaluated.input_width == map.input_width()
                && evaluated.ring_dimension == map.ring_dimension()
        }) {
            index
        } else {
            let matrix =
                setup
                    .shared_matrix
                    .ring_view_dyn(1, map.input_width(), map.ring_dimension())?;
            let matrix_row = matrix.row_flat(0)?;
            let powers = scalar_powers(alpha, map.ring_dimension());
            let columns = (0..map.input_width())
                .map(|column| {
                    let start = column * map.ring_dimension();
                    let end = start + map.ring_dimension();
                    Ok(eval_flat_ring_at_pows_fast(
                        matrix_row.get(start..end).ok_or(AkitaError::InvalidProof)?,
                        &powers,
                    ))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            evaluated_matrices.push(EvaluatedCompressionMatrix {
                input_width: map.input_width(),
                ring_dimension: map.ring_dimension(),
                powers,
                columns,
            });
            evaluated_matrices.len() - 1
        };
        let evaluated = evaluated_matrices
            .get(matrix_index)
            .ok_or(AkitaError::InvalidProof)?;
        for column in 0..map.input_width() {
            let start = column * map.ring_dimension();
            weights.push(
                span.range().start + start,
                map.ring_dimension(),
                0,
                row_weight
                    * evaluated
                        .columns
                        .get(column)
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?,
            )?;
        }
        if let Some(successor) = successor_compression_span(witness_layout, group_index, map_index)?
        {
            push_recomposition::<F, E>(
                &mut weights,
                successor.range().start,
                successor.map().ring_dimension(),
                map.output_coefficients(),
                map.ring_dimension(),
                row_index,
                1,
                field_bits,
                &row_weights,
            )?;
        }
        let denominator = evaluated
            .powers
            .last()
            .copied()
            .ok_or(AkitaError::InvalidProof)?
            * alpha
            + E::one();
        for (digit, gadget) in gadget_row_scalars::<F>(
            r_decomp_levels::<F>(lp.open().digits.log_basis),
            lp.open().digits.log_basis,
        )
        .into_iter()
        .enumerate()
        {
            weights.push(
                witness_layout.r_coefficient_index(row_index, digit, 0, 0)?,
                map.ring_dimension(),
                0,
                -(row_weight * denominator * E::lift_base(gadget)),
            )?;
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::One;
    use jolt_field::Prime128OffsetA7F7 as F;

    #[test]
    fn sparse_evaluator_matches_dense_materialization() {
        let mut weights = CompressionRelationWeights {
            events: Vec::new(),
            alpha_powers: (1..=16).map(F::from_u64).collect(),
            coefficient_block_len: 2,
            physical_field_len: 16,
        };
        weights.push(0, 4, 0, F::from_u64(3)).unwrap();
        weights.push(6, 2, 4, F::from_u64(5)).unwrap();
        weights.push(8, 8, 0, F::from_u64(7)).unwrap();
        let point = [
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
            F::from_u64(19),
        ];
        assert_eq!(
            weights.evaluate_at_point(&point).unwrap(),
            multilinear_eval(&weights.materialize_dense().unwrap(), &point).unwrap()
        );
    }

    #[test]
    fn negative_binary_support_restricted_equality_matches_dense_table() {
        let support = NegativeBinarySupport {
            intervals: vec![3..11, 16..31, 48..64],
            physical_field_len: 64,
        };
        let equality_point = (0..6)
            .map(|index| F::from_u64(11 + index as u64))
            .collect::<Vec<_>>();
        let point = (0..6)
            .map(|index| F::from_u64(23 + index as u64))
            .collect::<Vec<_>>();
        let equality = EqPolynomial::evals(&equality_point).unwrap();
        let restricted = support
            .materialize::<F>()
            .into_iter()
            .zip(equality)
            .map(|(support, equality)| support * equality)
            .collect::<Vec<_>>();
        assert_eq!(
            support
                .evaluate_restricted_equality_at_point(&equality_point, &point)
                .unwrap(),
            multilinear_eval(&restricted, &point).unwrap()
        );
    }

    #[test]
    fn negative_binary_support_does_not_factor_restricted_equality() {
        let support = NegativeBinarySupport {
            intervals: std::iter::once(0..1).collect(),
            physical_field_len: 2,
        };
        let equality_point = [F::from_u64(3)];
        let point = [F::from_u64(5)];
        let restricted = support
            .evaluate_restricted_equality_at_point(&equality_point, &point)
            .unwrap();
        let factored = multilinear_eval(&support.materialize::<F>(), &point).unwrap()
            * EqPolynomial::mle(&equality_point, &point).unwrap();
        assert_eq!(
            restricted,
            (F::one() - equality_point[0]) * (F::one() - point[0])
        );
        assert_ne!(restricted, factored);
    }
}
