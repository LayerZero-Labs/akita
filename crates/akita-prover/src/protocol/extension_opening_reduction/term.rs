use super::*;

/// One dense suffix term in an extension-opening reduction sumcheck.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionTerm<E: FieldCore> {
    pub(in crate::protocol::extension_opening_reduction) tables: ExtensionOpeningTables<E>,
    pub(in crate::protocol::extension_opening_reduction) coeff: E,
    /// Coefficient-scaled next-round values produced by the fused dense fold.
    pub(in crate::protocol::extension_opening_reduction) cached_accumulate: Option<(E, E)>,
}

impl<E: FieldCore> ExtensionOpeningReductionTerm<E> {
    /// Construct one term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness or factor table is malformed.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>, coeff: E) -> Result<Self, AkitaError> {
        validate_reduction_tables(&witness_evals, &factor_evals)?;
        Ok(Self {
            tables: ExtensionOpeningTables::Dense {
                witness: witness_evals,
                factor: DenseEorFactor::Owned(factor_evals),
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Construct one term with the full transparent factor shared by its group.
    ///
    /// The first fold replaces the shared table with an owned half-size table,
    /// so later rounds retain the existing in-place representation.
    pub fn new_with_shared_factor(
        witness_evals: Vec<E>,
        factor_evals: std::sync::Arc<Vec<E>>,
        coeff: E,
    ) -> Result<Self, AkitaError> {
        validate_reduction_tables(&witness_evals, &factor_evals)?;
        Ok(Self {
            tables: ExtensionOpeningTables::Dense {
                witness: witness_evals,
                factor: DenseEorFactor::Shared(factor_evals),
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Extend this term over additional high variables without copying its
    /// witness table.
    ///
    /// # Errors
    ///
    /// Returns an error if the combined virtual table length overflows.
    pub fn extend_cylindrically(mut self, extra_point: Vec<E>) -> Result<Self, AkitaError> {
        let native_rounds = num_rounds_from_table_len(self.tables.len())?;
        let total_rounds = native_rounds
            .checked_add(extra_point.len())
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "extension-opening cylindrical domain overflow".to_string(),
                )
            })?;
        reduction_table_len(total_rounds)?;
        if !extra_point.is_empty() {
            self.tables = ExtensionOpeningTables::Cylindrical {
                inner: Box::new(self.tables),
                extra_point,
                extra_round: 0,
                extra_factor_eval: E::one(),
            };
        }
        Ok(self)
    }

    /// Batching coefficient multiplying this term.
    pub fn coeff(&self) -> E {
        self.coeff
    }

    /// Current Boolean-domain table length, including virtual high variables.
    pub(crate) fn domain_len(&self) -> usize {
        self.tables.len()
    }

    /// Return final folded witness and factor values after all challenges.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        self.tables.final_witness_and_factor_evals()
    }
}

impl<E: FieldCore + HasUnreducedOps + HasOptimizedFold> ExtensionOpeningReductionTerm<E> {
    pub(in crate::protocol::extension_opening_reduction) fn accumulate_into(
        &mut self,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        match self.cached_accumulate.take() {
            Some((cached_constant, cached_quadratic)) => {
                *constant += cached_constant;
                *quadratic += cached_quadratic;
            }
            None => self
                .tables
                .accumulate_round(self.coeff, constant, quadratic),
        }
    }

    pub(in crate::protocol::extension_opening_reduction) fn ingest_challenge(
        &mut self,
        r_round: E,
    ) {
        if self.tables.len() <= 1 {
            return;
        }
        let fused = match &mut self.tables {
            ExtensionOpeningTables::Dense { witness, factor } if witness.len() >= 4 => {
                Some(fused_fold_and_accumulate(witness, factor, r_round))
            }
            _ => None,
        };
        match fused {
            Some((constant, quadratic)) => {
                self.cached_accumulate = Some((self.coeff * constant, self.coeff * quadratic));
            }
            None => {
                self.tables.fold_in_place(r_round);
                self.cached_accumulate = None;
            }
        }
    }
}
