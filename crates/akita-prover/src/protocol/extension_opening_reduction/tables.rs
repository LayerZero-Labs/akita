use super::*;

/// Dense EOR group with one transparent factor shared by all witness members.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionGroup<E: FieldCore> {
    pub(in crate::protocol::extension_opening_reduction) terms:
        Vec<ExtensionOpeningReductionTerm<E>>,
    pub(in crate::protocol::extension_opening_reduction) factor: Vec<E>,
    extra_point: Vec<E>,
    extra_round: usize,
    extra_factor_eval: E,
}

impl<E: FieldCore> ExtensionOpeningReductionGroup<E> {
    /// Construct a group whose members share one transparent factor table.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no members or any witness/factor table is
    /// malformed.
    pub fn new(
        terms: Vec<ExtensionOpeningReductionTerm<E>>,
        factor_evals: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if terms.is_empty() {
            return Err(AkitaError::InvalidInput(
                "extension-opening reduction group requires at least one term".to_string(),
            ));
        }
        for term in &terms {
            validate_reduction_tables(&term.witness, &factor_evals)?;
        }
        Ok(Self {
            terms,
            factor: factor_evals,
            extra_point: Vec::new(),
            extra_round: 0,
            extra_factor_eval: E::one(),
        })
    }

    /// Extend this group over additional high variables without copying its
    /// witness or factor tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the combined virtual table length overflows.
    pub fn extend_cylindrically(mut self, extra_point: Vec<E>) -> Result<Self, AkitaError> {
        let native_rounds = num_rounds_from_table_len(self.factor.len())?;
        let total_rounds = native_rounds
            .checked_add(extra_point.len())
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "extension-opening cylindrical domain overflow".to_string(),
                )
            })?;
        reduction_table_len(total_rounds)?;
        self.extra_point = extra_point;
        Ok(self)
    }

    /// Current Boolean-domain table length, including virtual high variables.
    pub(crate) fn domain_len(&self) -> usize {
        self.factor
            .len()
            .checked_shl(
                u32::try_from(self.extra_point.len().saturating_sub(self.extra_round))
                    .unwrap_or(u32::MAX),
            )
            .unwrap_or(0)
    }

    /// Number of witness members sharing this group's factor.
    pub(crate) fn num_terms(&self) -> usize {
        self.terms.len()
    }

    pub(in crate::protocol::extension_opening_reduction) fn claim(&self) -> Result<E, AkitaError> {
        self.terms.iter().try_fold(E::zero(), |acc, term| {
            extension_opening_reduction_claim(&term.witness, &self.factor)
                .map(|claim| acc + term.coeff * claim)
        })
    }

    pub(in crate::protocol::extension_opening_reduction) fn final_terms(
        &self,
    ) -> Option<Vec<(E, E, E)>> {
        if self.factor.len() != 1
            || self.extra_round != self.extra_point.len()
            || self.terms.iter().any(|term| term.witness.len() != 1)
        {
            return None;
        }
        let factor = self.factor[0] * self.extra_factor_eval;
        Some(
            self.terms
                .iter()
                .map(|term| (term.coeff, term.witness[0], factor))
                .collect(),
        )
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningReductionGroup<E> {
    pub(in crate::protocol::extension_opening_reduction) fn accumulate_into(
        &mut self,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        if self.factor.len() > 1 {
            for term in &mut self.terms {
                match term.cached_accumulate.take() {
                    Some((cached_constant, cached_quadratic)) => {
                        *constant += cached_constant;
                        *quadratic += cached_quadratic;
                    }
                    None => {
                        let (round_constant, round_quadratic) =
                            accumulate_dense_round(&term.witness, &self.factor, term.coeff);
                        *constant += round_constant;
                        *quadratic += round_quadratic;
                    }
                }
            }
            return;
        }

        if let Some(&point) = self.extra_point.get(self.extra_round) {
            let factor = self.factor[0] * self.extra_factor_eval * (E::one() - point);
            for term in &self.terms {
                *constant += term.coeff * term.witness[0] * factor;
            }
        }
    }

    pub(in crate::protocol::extension_opening_reduction) fn ingest_challenge(
        &mut self,
        r_round: E,
    ) {
        if self.domain_len() <= 1 {
            return;
        }
        if self.factor.len() > 1 {
            let previous_len = self.factor.len();
            if previous_len >= 4 {
                let (first, remaining) = self
                    .terms
                    .split_first_mut()
                    .expect("validated EOR groups are nonempty");
                let (constant, quadratic) = fused_fold_group_head_and_accumulate(
                    &mut first.witness,
                    &mut self.factor,
                    r_round,
                );
                first.cached_accumulate = Some((first.coeff * constant, first.coeff * quadratic));
                for term in remaining {
                    let (constant, quadratic) =
                        fused_fold_witness_and_accumulate(&mut term.witness, &self.factor, r_round);
                    term.cached_accumulate = Some((term.coeff * constant, term.coeff * quadratic));
                }
            } else {
                fold_evals_in_place(&mut self.factor, r_round);
                for term in &mut self.terms {
                    fold_evals_in_place(&mut term.witness, r_round);
                    term.cached_accumulate = None;
                }
            }
            return;
        }

        if let Some(&point) = self.extra_point.get(self.extra_round) {
            self.extra_factor_eval *= (E::one() - point) * (E::one() - r_round) + point * r_round;
            self.extra_round += 1;
        }
    }
}
