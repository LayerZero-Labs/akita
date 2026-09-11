use super::*;
use jolt_field::MulBaseUnreduced;

/// Complete the ring switch after the caller has bound the next witness.
///
/// Samples challenges and builds the evaluation tables for the fused sumcheck.
/// The caller must first absorb the next-witness binding into `transcript`.
///
/// The relation reads the exact compact coefficient prefix. Each commitment
/// group contributes events at its native role dimensions.
///
/// # Errors
///
/// Returns an error if the supplied gamma vector does not match the claim
/// count or if matrix expansion or evaluation-table construction fails.
#[tracing::instrument(skip_all, name = "ring_switch_finalize")]
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn ring_switch_finalize<F, E, T>(
    instance: &RingRelationInstance<F>,
    setup: &AkitaExpandedSetup<F>,
    transcript: &mut T,
    level: u32,
    w: &RecursiveWitnessFlat,
    lp: &CommittedGroupParams,
    opening_source_len: usize,
    opening_ring_dim: usize,
    gamma: Option<&[E]>,
    opening_claim_coefficients: &[E],
    prepared_relation_groups: &[crate::protocol::ring_relation::PreparedRelationGroup<F, E>],
) -> Result<RingSwitchFinalization<E>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + Ring + MulBaseUnreduced<F>,
    T: akita_types::ProverTranscriptGrinding<F>,
{
    let default_gamma;
    let gamma = if let Some(gamma) = gamma {
        gamma
    } else {
        default_gamma = instance
            .gamma()
            .iter()
            .copied()
            .map(E::lift_base)
            .collect::<Vec<_>>();
        &default_gamma
    };
    let opening_batch = instance.opening_batch();
    crate::protocol::ring_relation::validate_prepared_relation_groups(
        prepared_relation_groups,
        lp,
        opening_batch,
        instance,
    )?;
    transcript.grind_query(akita_types::GrindingSite::RingSwitchAlpha { level })?;
    let alpha: E = sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_RING_SWITCH);

    let opening_capacity = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening capacity overflow".into()))?;
    if opening_ring_dim == 0
        || !opening_ring_dim.is_power_of_two()
        || w.live_coeff_len() > opening_capacity
    {
        return Err(AkitaError::InvalidInput(format!(
            "witness length {} does not fit opening capacity {} at ring dimension {}",
            w.live_coeff_len(),
            opening_capacity,
            opening_ring_dim,
        )));
    }
    let witness_layout = instance.segment_layout(lp, None).map_err(|err| {
        AkitaError::InvalidInput(format!("relation witness layout failed: {err:?}"))
    })?;
    if w.live_coeff_len() != witness_layout.live_coeff_len() {
        return Err(AkitaError::InvalidSize {
            expected: witness_layout.live_coeff_len(),
            actual: w.live_coeff_len(),
        });
    }
    // Bind the low coefficient block shared by every role first, then the
    // remaining relation lanes. The challenge order is unchanged: the
    // common coefficients are the low Boolean coordinates.
    let geometry = lp.relation_address_geometry(
        opening_batch,
        instance.extension_degree(),
        opening_ring_dim,
        witness_layout.live_coeff_len(),
    )?;
    let coeff_count = geometry.relation_coefficient_block_len();
    if !w.live_coeff_len().is_multiple_of(coeff_count) {
        return Err(AkitaError::InvalidSetup(
            "relation witness is not aligned to the common coefficient block".into(),
        ));
    }
    let live_relation_lane_count = geometry.live_relation_lane_count();
    let col_bits = geometry.relation_lane_variable_count();
    let ring_bits = geometry.relation_coefficient_variable_count();
    // Bind the current roles' shared low coefficient block as the digit
    // range check's ring phase. Outgoing witness packaging determines only
    // the checked flat live length and its zero-padded capacity. Stage 1,
    // Stage 2, and Stage 3 all read the resulting point through this same
    // `col_bits`/`ring_bits` split.
    let digit_range_equality_low_variable_count = ring_bits;
    let num_sc_vars = col_bits + ring_bits;
    let num_i = lp.relation_row_index_num_vars(opening_batch)?;
    let physical_field_len = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;

    transcript.grind_query(akita_types::GrindingSite::Tau0Point { level })?;
    let tau0: Vec<E> = (0..num_sc_vars)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU0))
        .collect();
    transcript.grind_query(akita_types::GrindingSite::Tau1Point { level })?;
    let tau1: Vec<E> = (0..num_i)
        .map(|_| sample_ext_challenge::<F, E, T>(transcript, CHALLENGE_TAU1))
        .collect();
    if gamma.len() != instance.opening_batch().num_total_polynomials() {
        return Err(AkitaError::InvalidInput(
            "ring-switch gamma length does not match claim count".to_string(),
        ));
    }

    let relation_geometry =
        akita_types::RelationWitnessGeometry::for_level(lp, opening_batch, E::DEGREE)?;
    let relation_plan = akita_types::RelationRangeImagePlan::new(
        relation_geometry,
        geometry,
        akita_types::DigitRangePlan::new(1usize << lp.open().digits.log_basis)?,
        witness_layout.clone(),
        opening_batch,
    )?;
    let prepared_coefficient_packing_points;
    let opening_points = match prepared_relation_groups
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .kind()
    {
        akita_types::OpeningFamily::EvaluationTrace(_) => {
            akita_types::OpeningFamily::EvaluationTrace(())
        }
        akita_types::OpeningFamily::SubringCoefficientPacking(_) => {
            prepared_coefficient_packing_points = prepared_relation_groups
                .iter()
                .enumerate()
                .map(|(group_index, group)| match group.kind() {
                    akita_types::OpeningFamily::SubringCoefficientPacking(point) => {
                        Ok((group_index, point))
                    }
                    akita_types::OpeningFamily::EvaluationTrace(_) => Err(
                        AkitaError::InvalidSetup("ring-switch opening families are mixed".into()),
                    ),
                })
                .collect::<Result<Vec<_>, _>>()?;
            akita_types::OpeningFamily::SubringCoefficientPacking(
                prepared_coefficient_packing_points.as_slice(),
            )
        }
    };
    let relation_claim_coefficients = match opening_points {
        akita_types::OpeningFamily::EvaluationTrace(()) => gamma,
        akita_types::OpeningFamily::SubringCoefficientPacking(_) => {
            if opening_claim_coefficients.len() != opening_batch.num_total_polynomials() {
                return Err(AkitaError::InvalidSize {
                    expected: opening_batch.num_total_polynomials(),
                    actual: opening_claim_coefficients.len(),
                });
            }
            opening_claim_coefficients
        }
    };

    let prepare_relation_weights = || {
        let _span = tracing::info_span!("relation_weight_compilation").entered();
        match lp.ring_relation_mode {
            akita_types::RingRelationMode::QuotientLift => {
                let (events, opening_semantics) =
                    build_relation_weight_events(RelationWeightEventInputs {
                        setup: RelationSetupSource::Matrix(setup),
                        instance,
                        alpha,
                        level_params: lp,
                        relation_row_point: &tau1,
                        claim_coefficients: relation_claim_coefficients,
                        opening_source_len,
                        opening_ring_dim,
                        relation_plan: &relation_plan,
                        opening_points,
                    })?;
                let ordinary = events.factor_common_alpha()?;
                let compression = if lp.payload_mode.is_compressed() {
                    super::RingSwitchCompression::QuotientLift {
                        weights: akita_types::build_compression_relation_weights(
                            setup,
                            instance,
                            alpha,
                            lp,
                            &tau1,
                            &witness_layout,
                            opening_ring_dim,
                            physical_field_len,
                        )?,
                        support: akita_types::NegativeBinarySupport::new(
                            &witness_layout,
                            physical_field_len,
                        )?,
                    }
                } else {
                    super::RingSwitchCompression::Raw
                };
                Ok::<_, AkitaError>((
                    crate::protocol::sumcheck::RelationWeightOracle::QuotientFactored(ordinary),
                    compression,
                    opening_semantics,
                ))
            }
            akita_types::RingRelationMode::ReducedEvaluation => {
                if !matches!(
                    opening_points,
                    akita_types::OpeningFamily::EvaluationTrace(())
                ) {
                    return Err(AkitaError::InvalidSetup(
                        "reduced relation weights require evaluation-trace openings".into(),
                    ));
                }
                let dense = relation_weights::build_reduced_dense_relation_weights(
                    setup,
                    instance,
                    alpha,
                    lp,
                    &tau1,
                    opening_source_len,
                    opening_ring_dim,
                    &relation_plan,
                )?;
                let compression = if lp.payload_mode.is_compressed() {
                    super::RingSwitchCompression::ReducedEvaluation {
                        support: akita_types::NegativeBinarySupport::new(
                            &witness_layout,
                            physical_field_len,
                        )?,
                    }
                } else {
                    super::RingSwitchCompression::Raw
                };
                Ok((
                    crate::protocol::sumcheck::RelationWeightOracle::ReducedDense(dense),
                    compression,
                    akita_types::OpeningFamily::EvaluationTrace(()),
                ))
            }
        }
    };

    #[cfg(feature = "parallel")]
    let (relation_weights_result, w_result) = rayon::join(prepare_relation_weights, || {
        build_w_evals_compact(w.packed_digits(), coeff_count, 1, live_relation_lane_count)
    });
    #[cfg(not(feature = "parallel"))]
    let (relation_weights_result, w_result) = {
        let relation_weights = prepare_relation_weights();
        let w_compact =
            build_w_evals_compact(w.packed_digits(), coeff_count, 1, live_relation_lane_count);
        (relation_weights, w_compact)
    };

    let (relation_weights, compression, opening_semantics) =
        relation_weights_result.map_err(|err| {
            AkitaError::InvalidInput(format!("relation-weight compilation failed: {err:?}"))
        })?;
    let (w_evals_compact, witness_col_bits, witness_ring_bits) = w_result.map_err(|err| {
        AkitaError::InvalidInput(format!("witness opening preparation failed: {err:?}"))
    })?;
    if witness_col_bits != col_bits || witness_ring_bits != ring_bits {
        return Err(AkitaError::InvalidSetup(
            "prepared witness geometry disagrees with the current relation split".into(),
        ));
    }
    Ok(RingSwitchFinalization {
        output: RingSwitchOutput {
            w_evals_compact,
            relation_address_geometry: geometry,
            relation_weights,
            compression,
            digit_range_equality_low_variable_count,
            tau0,
            tau1,
            b: 1usize << lp.open().digits.log_basis,
            alpha,
        },
        relation_plan,
        opening_semantics,
    })
}
