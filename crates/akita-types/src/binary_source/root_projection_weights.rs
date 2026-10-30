//! Direct evaluation of the projected binary A relation.

use super::BinaryRootWitnessPlan;
use crate::binary_source::{binary_scalar_challenges, CANONICAL_IMAGE_WINDOW};
use crate::native_root_relation::{binary_commitment_group, NativeRootWeights};
use akita_algebra::ring::trinomial::{
    adjoint::TrinomialAdjoint, module_map::pack_signed_components,
};
use akita_algebra::MinusTrinomial;
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

/// Evaluate the binary A term directly from the typed q24 setup matrix.
///
/// The coefficient functional `rho[i][t]` batches seven exact projections.
/// It compiles `sum_i <rho_i, rem_Psi(A*z-sum c*T)_i>`, with no quotient.
/// The caller supplies the authenticated public projection target.
/// The field E must extend the admitted proof field.
/// Setup uniformity belongs to the typed setup producer; this boundary checks
/// its dimensions and centered representatives before compiling public weights.
pub fn binary_direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    matrix: &[Vec<[i128; 648]>],
    challenges: &Challenges,
    rho: &[[E; 648]],
    padding_weights: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let descriptor = plan.descriptor();
    let columns = descriptor.index_map().columns();
    let rows = checked::product([4, columns])
        .and_then(|divisor| checked::exact_div(descriptor.index_map().padded_carriers(), divisor))
        .ok_or(AkitaError::InvalidProof)?;
    if matrix.len() != plan.rank
        || matrix.iter().any(|row| row.len() != rows)
        || matrix.iter().flatten().flatten().any(|&coefficient| {
            coefficient < i128::from(CANONICAL_IMAGE_WINDOW.0)
                || coefficient > i128::from(CANONICAL_IMAGE_WINDOW.1)
        })
    {
        return Err(AkitaError::InvalidSetup(
            "binary A matrix differs from its declared geometry or centered range".into(),
        ));
    }
    let scalar = binary_scalar_challenges(descriptor, challenges)?;
    let embedded = scalar
        .into_iter()
        .map(|scalar| {
            Ok(pack_signed_components(&[scalar, [0; 162], [0; 162], [0; 162]])?.map(E::from_i128))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let matrix = matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|coefficients| coefficients.map(E::from_i128))
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let row_count = checked::product([descriptor.polynomials().len(), plan.rank])
        .ok_or(AkitaError::InvalidProof)?;
    let mut residuals = Vec::new();
    residuals
        .try_reserve_exact(row_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    residuals.resize(row_count, [E::zero(); 648]);
    if rho.len() != row_count {
        return Err(AkitaError::InvalidInput(
            "binary projection row count mismatch".into(),
        ));
    }
    let group = binary_commitment_group(descriptor.clone(), &matrix, &embedded, 0, 1, &residuals)?;
    let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
    let mut terms = Vec::new();
    for (row, rho) in group.rows.iter().zip(rho) {
        for term in &row.terms {
            for (coefficient, weight) in adjoint
                .apply(rho, &term.multiplier)?
                .into_iter()
                .enumerate()
            {
                terms.extend(plan.repacking().coefficient_weights(
                    term.segment,
                    term.element,
                    coefficient,
                    weight,
                )?);
            }
        }
    }
    let mut padding = padding_weights.iter();
    for address in 0..plan.repacking().domain_len() {
        if plan.repacking().is_padding_address(address)? {
            terms.push((address, *padding.next().ok_or(AkitaError::InvalidProof)?));
        }
    }
    if padding.next().is_some() {
        return Err(AkitaError::InvalidProof);
    }
    NativeRootWeights::new(plan.repacking().domain_len(), terms, E::zero())
}

#[cfg(test)]
#[path = "root_projection_weights_tests.rs"]
mod tests;
