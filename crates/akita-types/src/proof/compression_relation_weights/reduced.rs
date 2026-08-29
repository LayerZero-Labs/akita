use super::*;
use akita_algebra::ring::residue_kernel;
use akita_field::ExtField;

#[derive(Clone, Debug)]
struct ReducedCompressionMapEvent<E: FieldCore> {
    span: CompressionWitnessSpan,
    row_weight: E,
}

/// Checked physical/setup-column view for one canonical compression map.
struct CompressionMapColumns<F> {
    row: Vec<F>,
    physical_start: usize,
    input_width: usize,
    ring_dimension: usize,
}

impl<F: FieldCore> CompressionMapColumns<F> {
    fn new(
        setup: &AkitaExpandedSetup<F>,
        span: &CompressionWitnessSpan,
        physical_field_len: usize,
    ) -> Result<Self, AkitaError> {
        let map = span.map();
        let digit_span = span.range();
        let expected_len = map
            .input_width()
            .checked_mul(map.ring_dimension())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression map digit span overflow".into())
            })?;
        if digit_span.len() != expected_len || digit_span.end > physical_field_len {
            return Err(AkitaError::InvalidSetup(
                "compression witness span disagrees with its canonical map".into(),
            ));
        }
        let matrix =
            setup
                .shared_matrix()
                .ring_view_dyn(1, map.input_width(), map.ring_dimension())?;
        Ok(Self {
            row: matrix.row_flat(0)?.to_vec(),
            physical_start: digit_span.start,
            input_width: map.input_width(),
            ring_dimension: map.ring_dimension(),
        })
    }

    fn column(&self, column: usize) -> Result<(std::ops::Range<usize>, &[F]), AkitaError> {
        if column >= self.input_width {
            return Err(AkitaError::InvalidProof);
        }
        let source_start = column
            .checked_mul(self.ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("compression map column overflow".into()))?;
        let source_end = source_start
            .checked_add(self.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("compression map column extent overflow".into())
            })?;
        let physical_start = self
            .physical_start
            .checked_add(source_start)
            .ok_or_else(|| AkitaError::InvalidSetup("compression map address overflow".into()))?;
        let physical_end = physical_start
            .checked_add(self.ring_dimension)
            .ok_or_else(|| AkitaError::InvalidSetup("compression map address overflow".into()))?;
        Ok((
            physical_start..physical_end,
            self.row
                .get(source_start..source_end)
                .ok_or(AkitaError::InvalidProof)?,
        ))
    }
}

/// Complete reduced-evaluation weights for the retained F/H compression
/// digits.
///
/// Recomposition terms remain ordinary public-linear alpha windows. Universal
/// compression-map products are evaluated through exact negacyclic terminal
/// kernels at the verifier's final Stage-2 point. No quotient-row event is
/// represented by this type.
#[derive(Clone, Debug)]
pub struct ReducedCompressionRelationWeights<E: FieldCore> {
    linear: CompressionRelationWeights<E>,
    maps: Vec<ReducedCompressionMapEvent<E>>,
    alpha: E,
}

/// Evaluate one complete canonical compression map through exact reduced
/// coefficient functionals.
///
/// This is the checked boundary between [`crate::CompressionMapPlan`] geometry
/// and the public setup prefix. It reads map coefficients from that authority
/// instead of reconstructing them from witness offsets. Each setup coefficient
/// is consumed once while one native terminal kernel is retained at a time.
pub fn evaluate_reduced_compression_map<F, E>(
    setup: &AkitaExpandedSetup<F>,
    span: &CompressionWitnessSpan,
    point: &[E],
    physical_field_len: usize,
    alpha: E,
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    if !physical_field_len.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "compression relation domain must be a power of two".into(),
        ));
    }
    let expected_variables = physical_field_len.trailing_zeros() as usize;
    if point.len() != expected_variables {
        return Err(AkitaError::InvalidSize {
            expected: expected_variables,
            actual: point.len(),
        });
    }
    let columns = CompressionMapColumns::new(setup, span, physical_field_len)?;
    let equality = OffsetEqWindow::new(point)?;
    (0..columns.input_width).try_fold(E::zero(), |evaluation, column| {
        let (physical, coefficients) = columns.column(column)?;
        let functional = crate::ReducedCoefficientFunctional::prepare(
            &equality,
            physical_field_len,
            physical.start,
            columns.ring_dimension,
            alpha,
        )?;
        Ok(evaluation + functional.evaluate_multiplier(coefficients)?)
    })
}

impl<E: FieldCore> ReducedCompressionRelationWeights<E> {
    /// Add the complete reduced F/H ring-relation table to one checked padded
    /// Stage-2 destination.
    ///
    /// Linear recomposition events retain their canonical sparse alpha
    /// windows. Each universal compression-map column is read from its typed
    /// [`CompressionWitnessSpan`] and transposed through the shared
    /// negacyclic residue recurrence. No quotient row or independently
    /// reconstructed map geometry participates in this path.
    pub fn accumulate_dense<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        destination: &mut [E],
    ) -> Result<(), AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        if destination.len() != self.linear.physical_field_len {
            return Err(AkitaError::InvalidSize {
                expected: self.linear.physical_field_len,
                actual: destination.len(),
            });
        }
        for (weight, addition) in destination.iter_mut().zip(self.linear.materialize_dense()?) {
            *weight += addition;
        }
        for event in &self.maps {
            let columns = CompressionMapColumns::new(setup, &event.span, destination.len())?;
            for column in 0..columns.input_width {
                let (physical, coefficients) = columns.column(column)?;
                let kernel = residue_kernel::<F, E>(coefficients, self.alpha)?;
                for (weight, kernel) in destination
                    .get_mut(physical)
                    .ok_or(AkitaError::InvalidProof)?
                    .iter_mut()
                    .zip(kernel)
                {
                    *weight += event.row_weight * kernel;
                }
            }
        }
        Ok(())
    }

    /// Evaluate the complete reduced compression relation at one full witness
    /// point.
    pub fn evaluate_at_point<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        point: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let mut evaluation = self.linear.evaluate_at_point(point)?;
        for event in &self.maps {
            evaluation += event.row_weight
                * evaluate_reduced_compression_map(
                    setup,
                    &event.span,
                    point,
                    self.linear.physical_field_len,
                    self.alpha,
                )?;
        }
        Ok(evaluation)
    }

    /// Padded physical field domain covered by this table.
    #[must_use]
    pub fn physical_field_len(&self) -> usize {
        self.linear.physical_field_len()
    }
}

/// Build the complete reduced-evaluation F/H relation program.
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(skip_all, name = "build_reduced_compression_relation_weights")]
pub fn build_reduced_compression_relation_weights<F, E>(
    alpha: E,
    lp: &CommittedGroupParams,
    opening_batch: &crate::OpeningClaimsLayout,
    extension_degree: usize,
    tau1: &[E],
    witness_layout: &WitnessLayout,
    outgoing_ring_dimension: usize,
    physical_field_len: usize,
) -> Result<ReducedCompressionRelationWeights<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F>,
{
    if lp.ring_relation_mode != crate::RingRelationMode::ReducedEvaluation
        || !matches!(
            witness_layout.relation_quotient_layout(),
            crate::RelationQuotientLayout::ReducedEvaluation
        )
    {
        return Err(AkitaError::InvalidSetup(
            "reduced compression weights require a reduced relation layout".into(),
        ));
    }
    let relation_geometry =
        crate::RelationWitnessGeometry::for_level(lp, opening_batch, extension_degree)?;
    let relation_layout = relation_geometry.rhs_layout();
    let row_families = relation_layout.row_families()?;
    let row_weights = EqPolynomial::evals_prefix(tau1, row_families.len())?;
    let coefficient_block_len = lp
        .compression_relation_address_geometry(
            opening_batch,
            extension_degree,
            outgoing_ring_dimension,
            witness_layout.live_coeff_len(),
        )?
        .coefficient_block_len();
    let maximum_dimension = row_families
        .iter()
        .map(|row| row.geometry().polynomial_modulus_dimension())
        .max()
        .ok_or(AkitaError::InvalidProof)?;
    let mut linear = CompressionRelationWeights {
        events: Vec::new(),
        alpha_powers: scalar_powers(alpha, maximum_dimension),
        coefficient_block_len,
        physical_field_len,
    };
    let field_bits = usize::try_from(F::modulus_bits())
        .map_err(|_| AkitaError::InvalidSetup("compression field width overflow".into()))?;
    push_initial_recompositions::<F, E>(
        &mut linear,
        relation_layout,
        lp,
        opening_batch,
        witness_layout,
        field_bits,
        &row_weights,
        &row_families,
    )?;

    let mut maps = Vec::new();
    maps.try_reserve_exact(
        row_families
            .iter()
            .filter(|family| {
                matches!(
                    family,
                    RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
                )
            })
            .count(),
    )
    .map_err(|_| AkitaError::InvalidSetup("too many compression relation rows".into()))?;
    for (row_index, family) in row_families.iter().enumerate() {
        if !matches!(
            family,
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        let (group_index, map_index, span) = compression_span_for_row(witness_layout, family)?;
        let row_weight = *row_weights.get(row_index).ok_or(AkitaError::InvalidProof)?;
        maps.push(ReducedCompressionMapEvent {
            span: span.clone(),
            row_weight,
        });
        if let Some(successor) = successor_compression_span(witness_layout, group_index, map_index)?
        {
            let map = span.map();
            push_recomposition::<F, E>(
                &mut linear,
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
    }
    Ok(ReducedCompressionRelationWeights {
        linear,
        maps,
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommitmentPayloadMode, CompressionMapPlan, RingRelationMode};
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn reduced_compression_program_covers_every_map_without_quotient_events() {
        let mut params = crate::CommittedGroupParams::params_only(
            crate::SisModulusProfileId::Q128OffsetA7F7,
            32,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(1),
        )
        .with_decomp(4, 8, 1, 2, 2)
        .unwrap();
        params.payload_mode = CommitmentPayloadMode::Compressed;
        params.ring_relation_mode = RingRelationMode::ReducedEvaluation;
        let opening_batch = crate::OpeningClaimsLayout::new(0, 2).unwrap();
        let relation_geometry =
            crate::RelationWitnessGeometry::for_level(&params, &opening_batch, 1).unwrap();
        let witness_layout = crate::WitnessLayout::new(
            &params,
            &opening_batch,
            &relation_geometry,
            2,
            crate::RelationQuotientPlan::ReducedEvaluation,
        )
        .unwrap();
        assert!(witness_layout.r_rows().is_empty());
        let row_families = relation_geometry.rhs_layout().row_families().unwrap();
        let tau1 = (0..params.relation_row_index_num_vars(&opening_batch).unwrap())
            .map(|index| F::from_u64(11 + index as u64))
            .collect::<Vec<_>>();
        let row_weights = EqPolynomial::evals_prefix(&tau1, row_families.len()).unwrap();
        let physical_field_len = witness_layout.live_coeff_len().next_power_of_two();
        let alpha = F::from_u64(19);
        let program = build_reduced_compression_relation_weights::<F, F>(
            alpha,
            &params,
            &opening_batch,
            1,
            &tau1,
            &witness_layout,
            32,
            physical_field_len,
        )
        .unwrap();
        assert_eq!(program.physical_field_len(), physical_field_len);

        let compression_rows = row_families
            .iter()
            .enumerate()
            .filter(|(_, family)| {
                matches!(
                    family,
                    RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(program.maps.len(), compression_rows.len());
        for (event, (row, family)) in program.maps.iter().zip(&compression_rows) {
            let (_, _, expected_span) = compression_span_for_row(&witness_layout, family).unwrap();
            assert_eq!(event.span, *expected_span);
            assert_eq!(event.row_weight, row_weights[*row]);
        }

        let field_bits = F::modulus_bits() as usize;
        let relation_layout = relation_geometry.rhs_layout();
        let mut expected_linear_events = 0usize;
        for relation_group_index in 0..relation_layout.groups.len() {
            let (group_index, _) = relation_layout
                .group_compression_plan(relation_group_index)
                .unwrap();
            let source_rows = params
                .commitment_row_range(&opening_batch, group_index)
                .unwrap()
                .len();
            let source_dimension = relation_layout.groups[relation_group_index].role_dims.d_b();
            let digit_dimension = witness_layout.compression_layers()[0]
                .f_spans()
                .iter()
                .find_map(|(candidate, span)| {
                    (*candidate == group_index).then_some(span.map().ring_dimension())
                })
                .unwrap();
            expected_linear_events +=
                field_bits * source_rows * (source_dimension / digit_dimension);
        }
        let opening_start = row_families
            .iter()
            .position(|family| matches!(family, RelationRowFamily::Opening { .. }))
            .unwrap();
        assert!(opening_start < row_weights.len());
        let opening_digit_dimension = witness_layout.compression_layers()[0]
            .h_span()
            .map()
            .ring_dimension();
        expected_linear_events += field_bits
            * relation_layout.n_d
            * (relation_layout.d_ring_dimension / opening_digit_dimension);
        for (_, family) in &compression_rows {
            let (group_index, map_index, span) =
                compression_span_for_row(&witness_layout, family).unwrap();
            if let Some(successor) =
                successor_compression_span(&witness_layout, group_index, map_index).unwrap()
            {
                expected_linear_events +=
                    field_bits * (span.map().ring_dimension() / successor.map().ring_dimension());
            }
        }
        assert_eq!(program.linear.events.len(), expected_linear_events);
        assert!(program
            .linear
            .events
            .iter()
            .all(|event| { event.physical_start + event.coefficient_count <= physical_field_len }));

        let mut wrong_mode = params.clone();
        wrong_mode.ring_relation_mode = RingRelationMode::QuotientLift;
        assert!(build_reduced_compression_relation_weights::<F, F>(
            alpha,
            &wrong_mode,
            &opening_batch,
            1,
            &tau1,
            &witness_layout,
            32,
            physical_field_len,
        )
        .is_err());
    }

    #[test]
    fn reduced_map_uses_canonical_geometry_at_unaligned_window() {
        let map =
            CompressionMapPlan::new(crate::SisModulusProfileId::Q128OffsetA7F7, 8, 16, 1).unwrap();
        let setup_coefficients = map.input_width() * map.ring_dimension();
        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            crate::AkitaSetupDescriptor {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                num_field_elements: setup_coefficients,
                setup_seed: [0u8; 32].into(),
            },
            crate::FlatMatrix::from_flat_data(
                (0..setup_coefficients)
                    .map(|index| F::from_u64(401 + index as u64))
                    .collect(),
            ),
        );
        let point = (0..11)
            .map(|index| F::from_u64(503 + index as u64))
            .collect::<Vec<_>>();
        let digit_span_start = 11;
        let span = CompressionWitnessSpan::new_for_test(
            map,
            digit_span_start..digit_span_start + setup_coefficients,
        );
        let alpha = F::from_u64(19);
        let alpha_powers = scalar_powers(alpha, map.ring_dimension());
        let setup_row = setup.shared_matrix().as_field_slice();
        // Independent literal oracle: multiply each public map column by every
        // possible witness monomial, reduce X^d = -1, and place the resulting
        // scalar at its full physical witness address before the MLE fold.
        let expected = (0..map.input_width()).fold(F::zero(), |evaluation, column| {
            (0..map.ring_dimension()).fold(evaluation, |evaluation, witness_coefficient| {
                let residue = (0..map.ring_dimension()).fold(F::zero(), |sum, map_coefficient| {
                    let exponent = map_coefficient + witness_coefficient;
                    let product = setup_row[column * map.ring_dimension() + map_coefficient]
                        * alpha_powers[exponent % map.ring_dimension()];
                    if exponent < map.ring_dimension() {
                        sum + product
                    } else {
                        sum - product
                    }
                });
                let physical =
                    digit_span_start + column * map.ring_dimension() + witness_coefficient;
                evaluation + akita_algebra::offset_eq::eq_eval_at_index(&point, physical) * residue
            })
        });
        assert_eq!(
            evaluate_reduced_compression_map(&setup, &span, &point, 2048, alpha).unwrap(),
            expected
        );
        let row_weight = F::from_u64(23);
        let program = ReducedCompressionRelationWeights {
            linear: CompressionRelationWeights {
                events: Vec::new(),
                alpha_powers: Vec::new(),
                coefficient_block_len: 1,
                physical_field_len: 2048,
            },
            maps: vec![ReducedCompressionMapEvent {
                span: span.clone(),
                row_weight,
            }],
            alpha,
        };
        let mut dense = vec![F::zero(); 2048];
        program.accumulate_dense(&setup, &mut dense).unwrap();
        assert_eq!(
            akita_algebra::poly::multilinear_eval(&dense, &point).unwrap(),
            row_weight * expected
        );
        let malformed_span = CompressionWitnessSpan::new_for_test(map, 2040..2048);
        assert!(
            evaluate_reduced_compression_map(&setup, &malformed_span, &point, 2048, alpha).is_err()
        );
        let out_of_domain_span =
            CompressionWitnessSpan::new_for_test(map, 2040..2040 + setup_coefficients);
        assert!(
            evaluate_reduced_compression_map(&setup, &out_of_domain_span, &point, 2048, alpha)
                .is_err()
        );
        let oversized_point = vec![F::one(); 4_096];
        assert!(
            evaluate_reduced_compression_map(&setup, &span, &oversized_point, 2048, alpha).is_err()
        );
    }
}
