//! Allocation accounting and native lane compilation share the same dimension contract.
use super::*;
use akita_algebra::offset_eq::{materialize_eq_tensor_left, OffsetEqWindow};

impl<E: Field> SetupQuery<E> {
    /// Upper bound for retained factor tables and lane records, including
    /// coefficient preparation and lane-group metadata. Shared index tables
    /// are charged once per native term, not once per lane.
    pub fn compilation_bytes(
        &self,
        polynomial: &SetupPolynomialId,
        coefficient_len: usize,
    ) -> Result<usize, AkitaError> {
        self.validate(polynomial)?;
        polynomial.validate_split(coefficient_len)?;
        self.validate_split(coefficient_len)?;
        let term_bytes = |dimension| {
            let index_len = checked::exact_div(polynomial.field_len(), dimension)
                .ok_or(AkitaError::InvalidProof)?;
            let lanes =
                checked::exact_div(dimension, coefficient_len).ok_or(AkitaError::InvalidProof)?;
            let fields =
                checked::mul_add(3, dimension, index_len).ok_or(AkitaError::InvalidProof)?;
            checked::sum([
                checked::product([fields, size_of::<E>()]).ok_or(AkitaError::InvalidProof)?,
                checked::product([
                    lanes,
                    size_of::<SetupWeightTerm<E>>() + size_of::<Vec<SetupWeightTerm<E>>>(),
                ])
                .ok_or(AkitaError::InvalidProof)?,
            ])
            .ok_or(AkitaError::InvalidProof)
        };
        match &self.0 {
            Query::Evaluation(_) => term_bytes(coefficient_len),
            Query::Contribution {
                plan,
                functional: PreparedCoefficientFunctional::LiftedPower { .. },
            } => term_bytes(plan.projection_geometry().base_ring_dim()),
            Query::Native(terms) => terms.iter().try_fold(0, |bytes, term| {
                checked::sum([bytes, term_bytes(term.coefficient.len())?])
                    .ok_or(AkitaError::InvalidProof)
            }),
            Query::Contribution { plan, .. } => plan.groups().iter().try_fold(0, |bytes, group| {
                // Match the native compiler's three role dimensions without
                // preparing coefficient functionals or materializing tensors.
                let mut bytes = bytes;
                for (dimension, present) in [
                    (
                        group.role_dims().d_d(),
                        plan.d_rows() != 0
                            && plan.d_physical_cols() != 0
                            && !group.d_tensors().is_empty(),
                    ),
                    (
                        group.role_dims().d_b(),
                        group.physical_b().physical_rows() != 0,
                    ),
                    (
                        group.role_dims().d_a(),
                        group.n_a() != 0 && !group.a_tensors().is_empty(),
                    ),
                ] {
                    if present {
                        bytes = checked::sum([bytes, term_bytes(dimension)?])
                            .ok_or(AkitaError::InvalidProof)?;
                    }
                }
                Ok(bytes)
            }),
        }
    }

    /// Compile structured native factors, preserving the lifted index materializer.
    /// Charge `remaining_bytes` before allocation when supplied. `None` leaves
    /// the prover's existing memory policy unrestricted.
    pub fn compile(
        &self,
        polynomial: &SetupPolynomialId,
        coefficient_len: usize,
        remaining_bytes: Option<&mut usize>,
    ) -> Result<Vec<SetupWeightTerm<E>>, AkitaError> {
        if let Some(remaining_bytes) = remaining_bytes {
            let required = self.compilation_bytes(polynomial, coefficient_len)?;
            *remaining_bytes =
                remaining_bytes
                    .checked_sub(required)
                    .ok_or(AkitaError::InvalidSize {
                        expected: *remaining_bytes,
                        actual: required,
                    })?;
        } else {
            self.validate(polynomial)?;
            polynomial.validate_split(coefficient_len)?;
            self.validate_split(coefficient_len)?;
        }
        let native = match &self.0 {
            Query::Evaluation(point) => {
                let bits = coefficient_len.trailing_zeros() as usize;
                let (low, high) = point
                    .split_at_checked(bits)
                    .ok_or(AkitaError::InvalidProof)?;
                vec![(
                    EqPolynomial::evals(low)?,
                    EqPolynomial::evals(high)?,
                    polynomial.field_len() / coefficient_len,
                )]
            }
            Query::Contribution {
                plan,
                functional: PreparedCoefficientFunctional::LiftedPower { alpha },
            } => {
                let dim = plan.projection_geometry().base_ring_dim();
                let mut index = plan.materialize_setup_index_weights(*alpha)?;
                index.resize(polynomial.field_len() / dim, E::zero());
                vec![(scalar_powers(*alpha, dim).to_vec(), index, plan.required())]
            }
            Query::Contribution { .. } | Query::Native(_) => self
                .native_terms()?
                .into_iter()
                .map(|term| {
                    let equality = OffsetEqWindow::new(&term.point)?;
                    let index = materialize_eq_tensor_left(
                        &equality,
                        &term.tensors,
                        polynomial.field_len() / term.coefficient.len(),
                    )?;
                    let active = index
                        .iter()
                        .rposition(|weight| !weight.is_zero())
                        .map_or(0, |last| last + 1);
                    Ok((term.coefficient, index, active))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?,
        };
        let mut terms = Vec::new();
        for (coefficient, index, active) in native {
            let lanes = checked::exact_div(coefficient.len(), coefficient_len)
                .ok_or(AkitaError::InvalidProof)?;
            let index: Arc<[E]> = index.into();
            checked::product([index.len(), lanes]).ok_or(AkitaError::InvalidProof)?;
            for (lane, coefficient) in coefficient.chunks_exact(coefficient_len).enumerate() {
                terms.push(SetupWeightTerm {
                    coefficient: coefficient.to_vec(),
                    index: Arc::clone(&index),
                    lane,
                    lanes,
                    active_index_len: checked::product([active, lanes])
                        .ok_or(AkitaError::InvalidProof)?,
                });
            }
        }
        Ok(terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AkitaExpandedSetup, AkitaSetupDescriptor};
    use jolt_field::{Prime128Offset6A23801F as F, Ring};

    #[test]
    fn compilation_budget_rejects_large_index_tables_before_allocation() {
        let setup = AkitaExpandedSetup::<F>::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                num_field_elements: 64,
                setup_seed: [19; 32].into(),
            },
            crate::derive_public_matrix_prefix(64, &[19; 32].into()),
        );
        let polynomial = SetupPolynomialId::new::<F, F>(&setup, 1 << 24).unwrap();
        let query = SetupQuery::evaluation(vec![F::from_u64(3); 24]).unwrap();
        let required = query.compilation_bytes(&polynomial, 2).unwrap();
        let mut budget = 1024;
        assert!(matches!(query.compile(&polynomial, 2, Some(&mut budget)),
            Err(AkitaError::InvalidSize { expected: 1024, actual }) if actual == required));
        assert_eq!(budget, 1024, "rejection must not consume the budget");
    }
}
