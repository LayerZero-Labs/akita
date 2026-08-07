use super::*;

/// One term in an extension-opening reduction sumcheck.
///
/// A single dense term is the degenerate `1`-term case; the prover treats the
/// dense and batched paths uniformly.
#[derive(Debug, Clone)]
pub struct ExtensionOpeningReductionTerm<F: FieldCore, E: ExtField<F>> {
    pub(in crate::protocol::extension_opening_reduction) tables: ExtensionOpeningTables<F, E>,
    pub(in crate::protocol::extension_opening_reduction) coeff: E,
    /// `coeff`-scaled `(constant, quadratic)` for the next round, pre-computed
    /// by the fused fold in [`Self::ingest_challenge`] for the dense path.
    pub(in crate::protocol::extension_opening_reduction) cached_accumulate: Option<(E, E)>,
}

impl<F: FieldCore, E: ExtField<F>> ExtensionOpeningReductionTerm<F, E> {
    /// Construct one term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness/factor tables are malformed.
    pub fn new(witness_evals: Vec<E>, factor_evals: Vec<E>, coeff: E) -> Result<Self, AkitaError> {
        validate_reduction_tables(&witness_evals, &factor_evals)?;
        Ok(Self {
            tables: ExtensionOpeningTables::Dense {
                witness: EvaluationTable::from_multilinear_evaluations(&witness_evals)?,
                factor: EvaluationTable::from_multilinear_evaluations(&factor_evals)?,
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Construct one dense term with the transparent tensor equality factor.
    ///
    /// The factor is written directly into its coefficient-first multilinear
    /// table, so this ownership boundary never materializes a temporary factor
    /// vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the witness length, tail point, or packed head point
    /// does not describe one valid tensor factor table.
    pub(crate) fn new_tensor(
        witness: EvaluationTable<F, E>,
        tail_point: &[E],
        eta: &[E],
        coeff: E,
    ) -> Result<Self, AkitaError>
    where
        E: MulBaseUnreduced<F> + SumcheckTableOperations<F>,
    {
        let expected = checked_table_len(tail_point.len())?;
        if witness.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: witness.len(),
            });
        }
        let projection = TensorFactorProjection::<F, E>::new(eta)?;
        let (factor, first_round) = {
            let _span = tracing::debug_span!("extension_opening_factor_table", expected).entered();
            E::materialize_tensor_factor_and_compute_product_round(
                SumcheckKernelPlan::detect(),
                &witness,
                tail_point,
                &projection,
            )?
        };
        Ok(Self {
            tables: ExtensionOpeningTables::Dense { witness, factor },
            coeff,
            cached_accumulate: first_round
                .map(|(constant, quadratic)| (coeff * constant, coeff * quadratic)),
        })
    }

    /// Construct one sparse-witness term `coeff * sum_x witness(x) * factor(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the sparse witness and factor table shapes differ.
    pub fn new_sparse(
        witness_evals: SparseExtensionOpeningWitness<F, E>,
        factor_evals: Vec<E>,
        coeff: E,
    ) -> Result<Self, AkitaError> {
        if witness_evals.table_len() != factor_evals.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor_evals.len(),
            });
        }
        Ok(Self {
            tables: ExtensionOpeningTables::Sparse {
                witness: witness_evals,
                factor: SparseFactor::Dense(factor_evals),
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Construct one sparse-witness term with a lazy transparent tensor factor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor factor shape and sparse witness domain
    /// differ, or if the tensor opening parameters are malformed.
    pub fn new_sparse_tensor_factor(
        witness_evals: SparseExtensionOpeningWitness<F, E>,
        tail_point: Vec<E>,
        eta: Vec<E>,
        coeff: E,
        materialize_at: usize,
    ) -> Result<Self, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let factor = TensorEqualityFactor::new::<F>(tail_point, eta, materialize_at)?;
        if witness_evals.table_len() != factor.len() {
            return Err(AkitaError::InvalidSize {
                expected: witness_evals.table_len(),
                actual: factor.len(),
            });
        }
        let factor = if factor.is_ready_to_materialize() {
            SparseFactor::Dense(factor.materialize_dense())
        } else {
            SparseFactor::Tensor(factor)
        };
        Ok(Self {
            tables: ExtensionOpeningTables::Sparse {
                witness: witness_evals,
                factor,
            },
            coeff,
            cached_accumulate: None,
        })
    }

    /// Extend this term over additional high variables without materializing
    /// repeated witness evaluations.
    ///
    /// The added transparent factor is `eq(extra_point, ·)`. Its Boolean sum is
    /// one, so the input claim is unchanged while the term joins a larger
    /// sumcheck domain.
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
        checked_table_len(total_rounds)?;
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

    /// Return final folded witness/factor evaluations after all challenges.
    pub fn final_witness_and_factor_evals(&self) -> Option<(E, E)> {
        self.tables.final_witness_and_factor_evals()
    }
}

impl<F, E> ExtensionOpeningReductionTerm<F, E>
where
    F: FieldCore,
    E: SumcheckTableOperations<F>,
{
    /// Add this term's `coeff`-scaled `(constant, quadratic)` round
    /// contribution into the shared accumulators.
    ///
    /// Consumes the cache filled by the previous round's fused fold when
    /// present; otherwise accumulates directly from the current tables (the
    /// first round, and every round of the sparse/tensor paths).
    pub(in crate::protocol::extension_opening_reduction) fn accumulate_into(
        &mut self,
        plan: SumcheckKernelPlan,
        constant: &mut E,
        quadratic: &mut E,
    ) {
        match self.cached_accumulate.take() {
            Some((cached_constant, cached_quadratic)) => {
                *constant += cached_constant;
                *quadratic += cached_quadratic;
            }
            None => {
                self.tables
                    .accumulate_round(plan, self.coeff, constant, quadratic);
            }
        }
    }

    /// Fold this term's tables by one sumcheck challenge.
    ///
    /// Representations with a fused fold and next-round operation cache the
    /// `coeff`-scaled result. This includes dense tables, grouped sparse tensor
    /// factors, and the simpler merge-free sparse path.
    ///
    /// Every other shape folds in place and clears the cache.
    pub(in crate::protocol::extension_opening_reduction) fn ingest_challenge(
        &mut self,
        plan: SumcheckKernelPlan,
        r_round: E,
    ) {
        if self.tables.len() <= 1 {
            return;
        }
        let fused = self.tables.fold_and_accumulate(plan, r_round);
        match fused {
            Some((constant, quadratic)) => {
                self.cached_accumulate = Some((self.coeff * constant, self.coeff * quadratic));
            }
            None => {
                self.tables.fold_in_place(plan, r_round);
                self.cached_accumulate = None;
            }
        }
    }
}
