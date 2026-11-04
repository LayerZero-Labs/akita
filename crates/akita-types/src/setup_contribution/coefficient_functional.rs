//! Coefficient functionals shared by setup scans and external accumulation.
use crate::{factor_aligned_role_tensors, role_tensors_are_aligned, SetupContributionPlan};
use akita_error::AkitaError;
use akita_params::RelationAddressGeometry;
use jolt_field::Field;
use std::sync::Arc;
/// Coefficient functional shared by direct setup scans and external accumulation.
///
/// Lifted evaluation preserves the existing alpha-power factorization.
/// Reduced evaluation instead contracts each native setup ring against the
/// terminal residue kernel prepared from the exact Stage-2 coefficient point.
/// The point contains only the common low coefficient coordinates; role-lane
/// coordinates remain owned by the setup plan's checked relation tensors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparedCoefficientFunctional<E: Field> {
    LiftedPower {
        alpha: E,
    },
    ReducedEvaluation {
        alpha: E,
        coefficient_point: Arc<[E]>,
    },
}

impl<E: Field> PreparedCoefficientFunctional<E> {
    #[must_use]
    pub const fn lifted_power(alpha: E) -> Self {
        Self::LiftedPower { alpha }
    }

    pub fn reduced_evaluation(
        alpha: E,
        coefficient_point: &[E],
        geometry: RelationAddressGeometry,
    ) -> Result<Self, AkitaError> {
        let expected = geometry.relation_coefficient_variable_count();
        if coefficient_point.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: coefficient_point.len(),
            });
        }
        Ok(Self::ReducedEvaluation {
            alpha,
            coefficient_point: coefficient_point.to_vec().into(),
        })
    }

    #[must_use]
    pub const fn alpha(&self) -> E {
        match self {
            Self::LiftedPower { alpha } | Self::ReducedEvaluation { alpha, .. } => *alpha,
        }
    }
}

/// Exact native-ring functional and equality weights shared by direct and batched evaluation.
#[derive(Clone)]
pub struct ReducedRoleCoefficientState<E> {
    pub functional: Arc<[E]>,
    pub equality: Arc<[E]>,
}

use akita_algebra::offset_eq::{materialize_eq_tensor_left, EqPairTensorFamily, OffsetEqWindow};
use akita_algebra::ring::terminal_residue_kernel;

pub fn materialize_reduced_role_tensor_weights<E: Field>(
    plan: &SetupContributionPlan<E>,
    ratio: usize,
    role_dimension: usize,
    tensors: &[EqPairTensorFamily<E>],
    output_len: usize,
) -> Result<Vec<E>, AkitaError> {
    let coefficient_dimension = plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    let expected_ratio = role_dimension
        .checked_div(coefficient_dimension)
        .filter(|&value| {
            coefficient_dimension != 0
                && role_dimension.is_multiple_of(coefficient_dimension)
                && value.is_power_of_two()
        })
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "reduced role dimension does not decompose over the relation base".into(),
            )
        })?;
    if ratio != expected_ratio {
        return Err(AkitaError::InvalidSetup(
            "reduced role projection disagrees with its native dimension".into(),
        ));
    }
    if !role_tensors_are_aligned(tensors, ratio) {
        return Err(AkitaError::InvalidSetup(
            "reduced setup tensor is not aligned to its native coefficient window".into(),
        ));
    }

    let lane_variable_count = ratio.trailing_zeros() as usize;
    let high_point = plan
        .relation_address()
        .point()
        .get(lane_variable_count..)
        .ok_or(AkitaError::InvalidProof)?;
    let mut factored = tensors.to_vec();
    if ratio != 1 {
        factor_aligned_role_tensors(&mut factored, ratio)?;
    }
    let high_equality = OffsetEqWindow::new(high_point)?;
    materialize_eq_tensor_left(&high_equality, &factored, output_len)
}

pub fn prepare_reduced_role_coefficient_state<E: Field>(
    plan: &SetupContributionPlan<E>,
    role_dimension: usize,
    alpha: E,
    coefficient_point: &[E],
) -> Result<ReducedRoleCoefficientState<E>, AkitaError> {
    let coefficient_dimension = plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    let ratio = role_dimension
        .checked_div(coefficient_dimension)
        .filter(|&value| {
            coefficient_dimension != 0
                && role_dimension.is_multiple_of(coefficient_dimension)
                && value.is_power_of_two()
        })
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "reduced role dimension does not decompose over the relation base".into(),
            )
        })?;
    let expected_coefficient_variables = coefficient_dimension.trailing_zeros() as usize;
    if coefficient_point.len() != expected_coefficient_variables {
        return Err(AkitaError::InvalidSize {
            expected: expected_coefficient_variables,
            actual: coefficient_point.len(),
        });
    }
    let lane_variable_count = ratio.trailing_zeros() as usize;
    let lane_point = plan
        .relation_address()
        .point()
        .get(..lane_variable_count)
        .ok_or(AkitaError::InvalidProof)?;
    let native_variable_count = role_dimension.trailing_zeros() as usize;
    let mut native_point = Vec::new();
    native_point
        .try_reserve_exact(native_variable_count)
        .map_err(|_| AkitaError::InvalidSetup("native coefficient point is too large".into()))?;
    native_point.extend_from_slice(coefficient_point);
    native_point.extend_from_slice(lane_point);
    if native_point.len() != native_variable_count {
        return Err(AkitaError::InvalidSize {
            expected: native_variable_count,
            actual: native_point.len(),
        });
    }
    let native_equality = OffsetEqWindow::new(&native_point)?;
    let mut native_equality_weights = vec![E::zero(); role_dimension];
    native_equality.fill_interval(0, &mut native_equality_weights)?;
    let functional = terminal_residue_kernel(&native_equality_weights, alpha)?;
    Ok(ReducedRoleCoefficientState {
        functional: functional.into(),
        equality: native_equality_weights.into(),
    })
}
