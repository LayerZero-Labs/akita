use super::{point_domain, SetupPolynomialId};
use crate::setup_contribution::setup_index_weight::{
    build_group_b_setup_tensors, lift_role_tensor,
};
use crate::{
    factor_aligned_role_tensors, prepare_reduced_role_coefficient_state, project_role_tensors,
    role_projection_evaluation, role_tensors_are_aligned, PreparedCoefficientFunctional,
    SetupContributionPlan, SetupIndexWeightMle,
};
use akita_algebra::{
    eq_poly::EqPolynomial,
    offset_eq::{
        eval_boolean_pair_tensor_families, materialize_eq_tensor_left, EqPairTensorFamily,
        EqPairTensorWeights, OffsetEqWindow,
    },
    poly::multilinear_eval,
    ring::{evaluate_power_sequence_mle, scalar_powers},
};
use akita_error::{checked, AkitaError};
use akita_serialization::AkitaSerialize;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field};
use std::sync::Arc;

#[derive(Clone)]
enum Query<E: Field> {
    Contribution {
        plan: Arc<SetupContributionPlan<E>>,
        functional: PreparedCoefficientFunctional<E>,
    },
    Evaluation(Vec<E>),
}

/// An exact checked functional of the canonical flat setup stream.
#[derive(Clone)]
pub struct SetupQuery<E: Field>(Query<E>);

/// A rectangular term with a structured lane selector, never a coefficient-by-index table.
pub struct SetupWeightTerm<E: Field> {
    coefficient: Vec<E>,
    index: Arc<[E]>,
    lane: usize,
    lanes: usize,
    active_index_len: usize,
}
impl<E: Field> SetupWeightTerm<E> {
    pub fn coefficient(&self) -> &[E] {
        &self.coefficient
    }
    pub fn index_len(&self) -> usize {
        self.index.len() * self.lanes
    }
    pub const fn active_index_len(&self) -> usize {
        self.active_index_len
    }
    pub fn dense_index(&self) -> Option<&[E]> {
        (self.lanes == 1).then_some(&self.index)
    }
    /// Add only the selected native lane, preserving sparse views without allocating masks.
    pub fn add_scaled_index(&self, scale: E, output: &mut [E]) -> Result<(), AkitaError> {
        if output.len() != self.index_len() {
            return Err(AkitaError::InvalidProof);
        }
        let output = output
            .get_mut(self.lane..)
            .ok_or(AkitaError::InvalidProof)?;
        cfg_chunks_mut!(output, self.lanes)
            .zip(cfg_iter!(self.index.as_ref()))
            .for_each(|(chunk, &weight)| {
                // Nonempty chunks and their extent follow from the checked split.
                if let Some(value) = chunk.first_mut() {
                    *value += scale * weight;
                }
            });
        Ok(())
    }
    pub fn index_weight(&self, index: usize) -> E {
        if index % self.lanes == self.lane {
            self.index
                .get(index / self.lanes)
                .copied()
                .unwrap_or_else(E::zero)
        } else {
            E::zero()
        }
    }
}

struct NativeTerm<E: Field> {
    coefficient: Vec<E>,
    point: Vec<E>,
    tensors: Vec<EqPairTensorFamily<E>>,
}

impl<E: Field> SetupQuery<E> {
    pub fn contribution(
        plan: SetupContributionPlan<E>,
        functional: PreparedCoefficientFunctional<E>,
    ) -> Result<Self, AkitaError> {
        if let PreparedCoefficientFunctional::ReducedEvaluation {
            coefficient_point, ..
        } = &functional
        {
            let expected = plan
                .relation_address_geometry()
                .relation_coefficient_variable_count();
            if coefficient_point.len() != expected {
                return Err(AkitaError::InvalidSize {
                    expected,
                    actual: coefficient_point.len(),
                });
            }
        }
        Ok(Self(Query::Contribution {
            plan: Arc::new(plan),
            functional,
        }))
    }
    pub fn evaluation(point: Vec<E>) -> Result<Self, AkitaError> {
        point_domain(point.len())?;
        Ok(Self(Query::Evaluation(point)))
    }
    pub fn validate(&self, polynomial: &SetupPolynomialId) -> Result<(), AkitaError> {
        let fits = match &self.0 {
            Query::Contribution { plan, .. } => {
                plan.projection_geometry().natural_field_len() <= polynomial.field_len()
            }
            Query::Evaluation(point) => point.len() == polynomial.num_variables(),
        };
        if !fits {
            return Err(AkitaError::InvalidInput(
                "setup query exceeds its domain".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn validate_split(&self, coefficient_len: usize) -> Result<(), AkitaError> {
        if let Query::Contribution { plan, .. } = &self.0 {
            if coefficient_len > plan.projection_geometry().base_ring_dim() {
                return Err(AkitaError::InvalidInput(
                    "setup batch coefficient split exceeds a native role".into(),
                ));
            }
        }
        Ok(())
    }

    /// Evaluate the exact Boolean weight table's MLE without materializing it.
    pub fn evaluate_weight(
        &self,
        polynomial: &SetupPolynomialId,
        point: &[E],
    ) -> Result<E, AkitaError> {
        self.validate(polynomial)?;
        if point.len() != polynomial.num_variables() {
            return Err(AkitaError::InvalidProof);
        }
        match &self.0 {
            Query::Evaluation(query_point) => {
                Ok(query_point.iter().zip(point).fold(E::one(), |v, (&a, &b)| {
                    v * (a * b + (E::one() - a) * (E::one() - b))
                }))
            }
            Query::Contribution {
                plan,
                functional: PreparedCoefficientFunctional::LiftedPower { alpha },
            } => {
                let geometry = plan.projection_geometry();
                let own_bits = geometry.rounds();
                let (local, high) = point
                    .split_at_checked(own_bits)
                    .ok_or(AkitaError::InvalidProof)?;
                let (coefficient, index) = local
                    .split_at_checked(geometry.ring_bits())
                    .ok_or(AkitaError::InvalidProof)?;
                Ok(evaluate_power_sequence_mle(*alpha, coefficient)
                    * SetupIndexWeightMle::new(plan)?.evaluate(index, *alpha)?
                    * high.iter().fold(E::one(), |v, &r| v * (E::one() - r)))
            }
            Query::Contribution { .. } => {
                self.native_terms()?
                    .iter()
                    .try_fold(E::zero(), |sum, term| {
                        let bits = term.coefficient.len().trailing_zeros() as usize;
                        let (low, high) = point
                            .split_at_checked(bits)
                            .ok_or(AkitaError::InvalidProof)?;
                        Ok(sum
                            + multilinear_eval(&term.coefficient, low)?
                                * eval_boolean_pair_tensor_families::<_, false, false>(
                                    high,
                                    &term.point,
                                    &term.tensors,
                                )?)
                    })
            }
        }
    }

    /// Compile structured native factors, preserving the existing lifted index materializer.
    pub fn compile(
        &self,
        polynomial: &SetupPolynomialId,
        coefficient_len: usize,
    ) -> Result<Vec<SetupWeightTerm<E>>, AkitaError> {
        self.validate(polynomial)?;
        polynomial.validate_split(coefficient_len)?;
        self.validate_split(coefficient_len)?;
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
            Query::Contribution { .. } => self
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

    fn native_terms(&self) -> Result<Vec<NativeTerm<E>>, AkitaError> {
        let Query::Contribution { plan, functional } = &self.0 else {
            return Err(AkitaError::InvalidInput(
                "evaluation has no native role tensors".into(),
            ));
        };
        let mut terms = Vec::new();
        let mut coefficient_cache: Vec<(usize, Vec<E>)> = Vec::new();
        for group in plan.groups() {
            let d = if plan.d_rows() == 0 || plan.d_physical_cols() == 0 {
                Vec::new()
            } else {
                group
                    .d_tensors()
                    .iter()
                    .map(|t| {
                        lift_role_tensor(
                            t,
                            group.d_col_range().start,
                            plan.d_physical_cols(),
                            plan.d_weights(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
            };
            let b = if group.physical_b().physical_rows() == 0 {
                Vec::new()
            } else {
                build_group_b_setup_tensors(plan.relation_address_geometry(), group)?
            };
            let a = if group.n_a() == 0 {
                Vec::new()
            } else {
                group
                    .a_tensors()
                    .iter()
                    .map(|t| lift_role_tensor(t, 0, group.z_cols(), group.a_row_weights()))
                    .collect::<Result<Vec<_>, _>>()?
            };
            for (dim, ratio, mut tensors) in [
                (group.role_dims().d_d(), group.d_relation_ratio(), d),
                (group.role_dims().d_b(), group.b_relation_ratio(), b),
                (group.role_dims().d_a(), group.a_relation_ratio(), a),
            ] {
                if tensors.is_empty() {
                    continue;
                }
                let point = plan.relation_address().point();
                let bits = ratio.trailing_zeros() as usize;
                let (low, high) = point
                    .split_at_checked(bits)
                    .ok_or(AkitaError::InvalidProof)?;
                let coefficient;
                let relation_point;
                match functional {
                    PreparedCoefficientFunctional::LiftedPower { alpha } => {
                        coefficient = scalar_powers(*alpha, dim).to_vec();
                        if role_tensors_are_aligned(&tensors, ratio) {
                            if ratio != 1 {
                                factor_aligned_role_tensors(&mut tensors, ratio)?;
                            }
                            let scale = role_projection_evaluation(
                                *alpha,
                                plan.relation_address_geometry()
                                    .relation_coefficient_block_len(),
                                low,
                            )?;
                            for tensor in &mut tensors {
                                tensor.scalar *= scale;
                            }
                            relation_point = high.to_vec();
                        } else {
                            tensors = project_role_tensors(
                                &tensors,
                                ratio,
                                *alpha,
                                plan.relation_address_geometry()
                                    .relation_coefficient_block_len(),
                            )?;
                            relation_point = point.to_vec();
                        }
                    }
                    PreparedCoefficientFunctional::ReducedEvaluation {
                        alpha,
                        coefficient_point,
                    } => {
                        if !role_tensors_are_aligned(&tensors, ratio) {
                            return Err(AkitaError::InvalidProof);
                        }
                        if ratio != 1 {
                            factor_aligned_role_tensors(&mut tensors, ratio)?;
                        }
                        coefficient = if let Some((_, values)) = coefficient_cache
                            .iter()
                            .find(|(dimension, _)| *dimension == dim)
                        {
                            values.clone()
                        } else {
                            let values = prepare_reduced_role_coefficient_state(
                                plan,
                                dim,
                                *alpha,
                                coefficient_point,
                            )?
                            .functional
                            .to_vec();
                            coefficient_cache.push((dim, values.clone()));
                            values
                        };
                        relation_point = high.to_vec();
                    }
                }
                terms.push(NativeTerm {
                    coefficient,
                    point: relation_point,
                    tensors,
                });
            }
        }
        Ok(terms)
    }

    pub fn append_descriptor_bytes<F>(&self, bytes: &mut Vec<u8>) -> Result<(), AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        fn integer(bytes: &mut Vec<u8>, value: usize) -> Result<(), AkitaError> {
            value
                .serialize_compressed(bytes)
                .map_err(|_| AkitaError::InvalidProof)
        }
        fn fields<F: Field + CanonicalEncoding, E: ExtField<F>>(
            bytes: &mut Vec<u8>,
            values: &[E],
        ) -> Result<(), AkitaError> {
            integer(bytes, values.len())?;
            for value in values {
                for limb in value.to_base_vec() {
                    bytes.extend_from_slice(&limb.to_bytes_le_vec());
                }
            }
            Ok(())
        }
        match &self.0 {
            Query::Evaluation(point) => {
                integer(bytes, 0)?;
                fields::<F, E>(bytes, point)
            }
            Query::Contribution { .. } => {
                integer(bytes, 1)?;
                let terms = self.native_terms()?;
                integer(bytes, terms.len())?;
                for term in terms {
                    fields::<F, E>(bytes, &term.coefficient)?;
                    fields::<F, E>(bytes, &term.point)?;
                    integer(bytes, term.tensors.len())?;
                    for tensor in term.tensors {
                        integer(bytes, tensor.left_offset)?;
                        integer(bytes, tensor.right_offset)?;
                        fields::<F, E>(bytes, &[tensor.scalar])?;
                        integer(bytes, tensor.axes.len())?;
                        for axis in tensor.axes {
                            for n in [axis.len, axis.left_stride, axis.right_stride] {
                                integer(bytes, n)?;
                            }
                            match axis.weights {
                                EqPairTensorWeights::Unit => integer(bytes, 0)?,
                                EqPairTensorWeights::Dense(values) => {
                                    integer(bytes, 1)?;
                                    fields::<F, E>(bytes, &values)?;
                                }
                                EqPairTensorWeights::BitProduct(values) => {
                                    integer(bytes, 2)?;
                                    integer(bytes, values.len())?;
                                    for pair in values.iter() {
                                        fields::<F, E>(bytes, pair)?;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }
}
