use super::*;
use akita_error::checked;
use jolt_poly::UnivariatePoly;

fn stage2_geometry(
    lane_bits: usize,
    coefficient_bits: usize,
) -> Result<(usize, usize), AkitaError> {
    let lane_capacity = checked::pow2(lane_bits)
        .ok_or_else(|| AkitaError::Internal("stage-2 lane width overflow".to_string()))?;
    let coeff_count = checked::pow2(coefficient_bits)
        .ok_or_else(|| AkitaError::Internal("stage-2 coefficient width overflow".to_string()))?;
    Ok((lane_capacity, coeff_count))
}

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[cfg(test)]
    pub(crate) fn new_virtual_only(
        witness: Vec<i8>,
        stage1_point: &[E],
        range_image_evaluation: E,
        basis: usize,
        live_lanes: usize,
        lane_bits: usize,
        coefficient_bits: usize,
    ) -> Result<Self, AkitaError>
    where
        E: 'static,
    {
        let (lane_capacity, coeff_count) = stage2_geometry(lane_bits, coefficient_bits)?;
        Self::new(
            E::one(),
            PackedSignedDigits::from_i8_digits_auto(witness),
            stage1_point,
            range_image_evaluation,
            basis,
            RelationWeightOracle::QuotientFactored(RelationWeightFactorization::new(
                vec![E::zero(); coeff_count],
                vec![E::zero(); lane_capacity],
            )?),
            live_lanes,
            lane_bits,
            coefficient_bits,
            E::zero(),
            PreparedProverLinearTerms::zero(live_lanes, coeff_count),
            None,
        )
    }

    /// Create a fused stage-2 virtual-claim + relation sumcheck prover.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "RelationRangeImageProver::new")]
    pub(crate) fn new(
        batching_coeff: E,
        w_evals_compact: PackedSignedDigits,
        stage1_point: &[E],
        range_image_evaluation: E,
        b: usize,
        relation_weights: RelationWeightOracle<E>,
        live_lane_count: usize,
        lane_bits: usize,
        coefficient_bits: usize,
        relation_linear_claim: E,
        linear_terms: PreparedProverLinearTerms<E>,
        additional_relation_terms: Option<AdditionalRelationTerms<E>>,
    ) -> Result<Self, AkitaError>
    where
        E: 'static,
    {
        let num_vars = lane_bits
            .checked_add(coefficient_bits)
            .ok_or_else(|| AkitaError::Internal("stage-2 challenge width overflow".to_string()))?;
        if live_lane_count == 0 {
            return Err(AkitaError::Internal(
                "live_lane_count must be at least 1".to_string(),
            ));
        }
        let (lane_capacity, coeff_count) = stage2_geometry(lane_bits, coefficient_bits)?;
        if live_lane_count > lane_capacity {
            return Err(AkitaError::Internal(format!(
                "stage-2 live lane count: expected {lane_capacity}, actual {live_lane_count}"
            )));
        }
        let witness_len = live_lane_count
            .checked_mul(coeff_count)
            .ok_or_else(|| AkitaError::Internal("stage-2 witness size overflow".to_string()))?;
        if w_evals_compact.len() != witness_len {
            return Err(AkitaError::Internal(format!(
                "stage-2 compact witness length: expected {witness_len}, actual {}",
                w_evals_compact.len(),
            )));
        }
        if stage1_point.len() != num_vars {
            return Err(AkitaError::Internal(format!(
                "stage-2 replay point length: expected {num_vars}, actual {}",
                stage1_point.len(),
            )));
        }
        match &relation_weights {
            RelationWeightOracle::QuotientFactored(factorization) => {
                if factorization.common_alpha_factor().len() != coeff_count {
                    return Err(AkitaError::Internal(format!(
                        "stage-2 common alpha factor length: expected {coeff_count}, actual {}",
                        factorization.common_alpha_factor().len(),
                    )));
                }
                if factorization.relation_lane_weights().len() != lane_capacity {
                    return Err(AkitaError::Internal(format!(
                        "stage-2 relation lane weight count: expected {lane_capacity}, actual {}",
                        factorization.relation_lane_weights().len(),
                    )));
                }
            }
            RelationWeightOracle::ReducedDense(dense) => {
                let domain_len = lane_capacity.checked_mul(coeff_count).ok_or_else(|| {
                    AkitaError::Internal("stage-2 relation domain overflow".into())
                })?;
                if dense.evaluations().len() != domain_len {
                    return Err(AkitaError::Internal(format!(
                        "stage-2 dense relation domain length: expected {domain_len}, actual {}",
                        dense.evaluations().len(),
                    )));
                }
                if dense.live_len() != witness_len {
                    return Err(AkitaError::Internal(format!(
                        "stage-2 dense relation live length: expected {witness_len}, actual {}",
                        dense.live_len(),
                    )));
                }
            }
        }
        linear_terms.validate_len(witness_len)?;

        // Self-consistency check: the materialized ordinary relation weights
        // plus the structured linear weights must reproduce the combined
        // relation claim. Packing keeps its Z contribution in the structured
        // representation, so checking the ordinary table in isolation would
        // reject a valid factored relation. This is a full-domain
        // `O(lane_capacity * coeff_count)` pass, so it is gated to
        // debug/test builds and never runs in release proving.
        #[cfg(debug_assertions)]
        {
            let (ordinary_relation_sum, structured_relation_sum) =
                (0..witness_len).fold((E::zero(), E::zero()), |(ordinary, structured), index| {
                    let lane = index / coeff_count;
                    let coefficient = index % coeff_count;
                    let w = w_evals_compact
                        .view()
                        .get(index)
                        .expect("debug relation witness index is in bounds");
                    let witness = E::from_i64(i64::from(w));
                    let relation_weight = match &relation_weights {
                        RelationWeightOracle::QuotientFactored(factorization) => {
                            factorization.common_alpha_factor()[coefficient]
                                * factorization.relation_lane_weights()[lane]
                        }
                        RelationWeightOracle::ReducedDense(dense) => dense.evaluations()[index],
                    };
                    (
                        ordinary + witness * relation_weight,
                        structured + witness * linear_terms.get(lane, coefficient, coeff_count),
                    )
                });
            let additional_claim = additional_relation_terms
                .as_ref()
                .map_or_else(E::zero, |terms| terms.input_claim(&w_evals_compact));
            if ordinary_relation_sum + structured_relation_sum + additional_claim
                != relation_linear_claim
            {
                return Err(AkitaError::Internal(
                    "materialized relation weights do not match the combined relation claim".into(),
                ));
            }
        }

        let input_claim = batching_coeff * range_image_evaluation + relation_linear_claim;
        let split_eq = GruenSplitEq::with_initial_scalar(stage1_point, batching_coeff)?;
        let mut phase = match relation_weights {
            RelationWeightOracle::QuotientFactored(weights) => {
                let engine = CompactQuotientPrefix::new(
                    &w_evals_compact,
                    weights.relation_lane_weights(),
                    &linear_terms,
                    &split_eq,
                    stage1_point,
                    batching_coeff,
                    b,
                    live_lane_count,
                    coefficient_bits,
                )?;
                match engine {
                    Some(engine) => Phase::CompactPrefix {
                        witness: w_evals_compact,
                        weights,
                        engine: Box::new(engine),
                    },
                    None => {
                        // Moments reuse the norm kernels for partially live lanes.
                        let relation_moments =
                            if b == 32 && coefficient_bits > 0 && live_lane_count < lane_capacity {
                                CoefficientRelationMoments::from_basis32(
                                    &w_evals_compact,
                                    weights.relation_lane_weights(),
                                    &linear_terms,
                                    live_lane_count,
                                    coeff_count,
                                )
                            } else {
                                None
                            };
                        Phase::Coefficient {
                            witness: WitnessState::CompactPrefix(w_evals_compact),
                            relation: CoefficientRelation::Factored(weights),
                            relation_moments,
                        }
                    }
                }
            }
            RelationWeightOracle::ReducedDense(weights) => Phase::Coefficient {
                witness: WitnessState::CompactPrefix(w_evals_compact),
                relation: CoefficientRelation::ReducedDense(weights),
                relation_moments: None,
            },
        };

        let mut state = RelationRoundState {
            input_claim,
            split_eq,
            additional_relation_terms,
            linear_terms,
            live_lane_count,
            lane_bits,
            num_vars,
            prev_norm_claim: batching_coeff * range_image_evaluation,
            prev_norm_poly: None,
            cached_round_message: None,
            rounds_completed: 0,
        };
        state.advance_phase(&mut phase);
        Ok(Self { phase, state })
    }

    /// Return the fully folded witness evaluation after the final round.
    ///
    /// Returns an internal error unless the folded suffix contains one field element.
    pub(crate) fn final_w_eval(&self) -> Result<E, AkitaError> {
        let witness = match &self.phase {
            Phase::Lane { witness, .. } => witness,
            Phase::Coefficient {
                witness: WitnessState::FoldedSuffix(witness),
                ..
            } => witness,
            Phase::CompactPrefix { .. }
            | Phase::Coefficient {
                witness: WitnessState::CompactPrefix(_),
                ..
            } => {
                return Err(AkitaError::Internal(
                    "final witness remained compact after final fold".into(),
                ));
            }
        };
        match witness.as_slice() {
            [value] => Ok(*value),
            _ => Err(AkitaError::Internal(format!(
                "final witness suffix length: expected 1, actual {}",
                witness.len(),
            ))),
        }
    }

    pub(crate) fn expected_final_claim(&self) -> Result<E, AkitaError> {
        let phase = &self.phase;
        let witness = self.final_w_eval()?;
        let virtual_claim = self.state.split_eq.current_scalar() * witness * (witness + E::one());
        let relation_weight = match phase {
            Phase::CompactPrefix { weights, .. }
            | Phase::Coefficient {
                relation: CoefficientRelation::Factored(weights),
                ..
            } => {
                match (
                    weights.common_alpha_factor(),
                    weights.relation_lane_weights(),
                ) {
                    ([alpha], [lane]) => *alpha * *lane + self.state.linear_terms.final_value()?,
                    _ => {
                        return Err(AkitaError::Internal(
                            "terminal factored relation weights are not singletons".into(),
                        ))
                    }
                }
            }
            Phase::Coefficient {
                relation: CoefficientRelation::ReducedDense(weights),
                ..
            } => weights.terminal_weight()? + self.state.linear_terms.final_value()?,
            Phase::Lane { lane, .. } => lane.final_weight()?,
        };
        let additional = self
            .state
            .additional_relation_terms
            .as_ref()
            .map_or(Ok(E::zero()), |terms| terms.final_claim(witness))?;
        Ok(virtual_claim + witness * relation_weight + additional)
    }

    pub(super) fn additional_round_message(&self) -> Option<RoundMessage<E>> {
        let additional = self.state.additional_relation_terms.as_ref()?;
        Some(match &self.phase {
            Phase::CompactPrefix {
                witness, engine, ..
            } => additional.round_message_compact(witness.view(), engine.challenges()),
            Phase::Coefficient {
                witness: WitnessState::CompactPrefix(witness),
                ..
            } => additional.round_message_compact(witness.view(), &[]),
            Phase::Coefficient {
                witness: WitnessState::FoldedSuffix(witness),
                ..
            }
            | Phase::Lane { witness, .. } => additional.round_message_folded(witness),
        })
    }
}

impl<E: Field + Ring + Unreduced> RelationRoundState<E> {
    pub(super) fn advance_phase(&mut self, phase: &mut Phase<E>) {
        if let Phase::CompactPrefix {
            witness,
            weights,
            engine,
        } = phase
        {
            if engine.challenges().len() >= engine.last_round() {
                let (folded, norm) = engine.materialize(
                    witness.view(),
                    &self.split_eq,
                    self.split_eq.prepare_linear_q_recovery(),
                );
                let mut relation =
                    engine.relation_message(weights.common_alpha_factor(), &self.linear_terms);
                let norm_poly = self.norm_poly_from_prefix(norm);
                relation.add_assign(RoundMessage::from_polynomial(&norm_poly));
                self.cached_round_message = Some(relation);
                self.prev_norm_poly = Some(norm_poly);
                *phase = Phase::Coefficient {
                    witness: WitnessState::FoldedSuffix(folded),
                    relation: CoefficientRelation::Factored(weights.take()),
                    relation_moments: None,
                };
            }
        }

        if self.rounds_completed < self.coefficient_bits() {
            return;
        }
        if let Phase::Coefficient {
            witness, relation, ..
        } = phase
        {
            let weight_scale = match relation {
                CoefficientRelation::Factored(weights) => {
                    assert_eq!(
                        weights.common_alpha_factor().len(),
                        1,
                        "lane transition requires every coefficient weight to be bound"
                    );
                    weights.common_alpha_factor()[0]
                }
                CoefficientRelation::ReducedDense(_) => E::one(),
            };
            // Materialize before taking any vector; allocation and field work
            // must not leave the installed phase with an empty witness.
            let witness = match witness {
                WitnessState::CompactPrefix(witness) => witness
                    .view()
                    .iter()
                    .map(|digit| E::from_i64(i64::from(digit)))
                    .collect(),
                WitnessState::FoldedSuffix(witness) => mem::take(witness),
            };
            let weights = match relation {
                CoefficientRelation::Factored(weights) => weights.take_lane_weights(),
                CoefficientRelation::ReducedDense(weights) => weights.take_evaluations(),
            };
            *phase = Phase::Lane {
                witness,
                lane: LaneProduct::new(weights),
            };
            // The transition is installed before scanning or modifying weights.
            if let Phase::Lane { lane, .. } = phase {
                lane.prepare_weights(self.live_lane_count, weight_scale, &mut self.linear_terms);
            }
        }
    }

    #[inline]
    pub(super) fn coefficient_bits(&self) -> usize {
        self.num_vars - self.lane_bits
    }

    #[inline]
    fn coefficient_rounds_completed(&self) -> usize {
        self.rounds_completed.min(self.coefficient_bits())
    }

    #[inline]
    pub(super) fn in_coefficient_round(&self) -> bool {
        self.rounds_completed < self.coefficient_bits()
    }

    #[inline]
    pub(super) fn current_coefficient_width(&self) -> usize {
        self.coefficient_bits()
            .saturating_sub(self.coefficient_rounds_completed())
    }

    #[inline]
    pub(super) fn use_partial_lane_coefficient_round(&self) -> bool {
        self.in_coefficient_round() && self.live_lane_count < (1usize << self.lane_bits)
    }

    #[inline]
    pub(super) fn norm_poly_from_terms(
        &self,
        virt_terms: NormRoundTerms<'_, E>,
    ) -> UnivariatePoly<E> {
        match virt_terms {
            NormRoundTerms::Full(virt_q_coeffs) => {
                self.split_eq.gruen_mul(&coeffs_to_poly(virt_q_coeffs))
            }
            NormRoundTerms::SkipLinear([constant, quadratic], recovery) => {
                recovery.gruen_poly_deg_3(constant, quadratic, self.prev_norm_claim)
            }
        }
    }

    #[inline]
    pub(super) fn combine_terms(
        &self,
        virt_terms: NormRoundTerms<'_, E>,
        relation_message: RoundMessage<E>,
    ) -> (RoundMessage<E>, UnivariatePoly<E>) {
        let norm_poly = self.norm_poly_from_terms(virt_terms);
        let mut message = RoundMessage::from_polynomial(&norm_poly);
        message.add_assign(relation_message);
        (message, norm_poly)
    }
}

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[cfg(test)]
    pub(super) fn disable_factored_relation_moments(&mut self) {
        if let Phase::Coefficient {
            relation_moments, ..
        } = &mut self.phase
        {
            *relation_moments = None;
        }
    }

    #[cfg(test)]
    pub(super) fn has_factored_relation_moments(&self) -> bool {
        matches!(
            &self.phase,
            Phase::Coefficient {
                relation_moments: Some(_),
                ..
            }
        )
    }

    #[cfg(test)]
    pub(super) fn disable_compact_quotient_prefix(&mut self) {
        if let Phase::CompactPrefix {
            witness, weights, ..
        } = &mut self.phase
        {
            // The test disables the engine before any challenge is ingested.
            let witness = witness.clone();
            self.phase = Phase::Coefficient {
                witness: WitnessState::CompactPrefix(witness),
                relation: CoefficientRelation::Factored(weights.take()),
                relation_moments: None,
            };
        }
    }
}

#[cfg(test)]
mod final_witness_tests {
    use super::*;
    use jolt_field::{One, Prime128Offset275 as E};

    fn prover() -> RelationRangeImageProver<E> {
        RelationRangeImageProver::new_virtual_only(
            vec![0; 8],
            &[E::one(); 3],
            E::zero(),
            8,
            2,
            1,
            2,
        )
        .unwrap()
    }

    #[test]
    fn final_witness_rejects_compact_phase() {
        let mut prover = prover();
        assert!(
            matches!(prover.final_w_eval(), Err(AkitaError::Internal(message))
            if message == "final witness remained compact after final fold")
        );
        prover.disable_compact_quotient_prefix();
        assert!(
            matches!(prover.final_w_eval(), Err(AkitaError::Internal(message))
            if message == "final witness remained compact after final fold")
        );
    }

    #[test]
    fn final_witness_rejects_nonterminal_suffix() {
        let mut prover = prover();
        for actual in [0, 2] {
            prover.phase = Phase::Lane {
                witness: vec![E::zero(); actual],
                lane: LaneProduct::new(vec![E::one()]),
            };
            assert!(
                matches!(prover.final_w_eval(), Err(AkitaError::Internal(message))
                if message == format!("final witness suffix length: expected 1, actual {actual}"))
            );
        }
    }
}
