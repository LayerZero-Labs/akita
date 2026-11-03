//! Projected A relation on canonical digit addresses, with an explicit rho.

use akita_algebra::ring::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::ring::trinomial::module_map::pack_signed_components;
use akita_algebra::ring::trinomial::MinusTrinomial;
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

use crate::binary_source::{
    binary_scalar_challenges, BinaryRootWitnessPlan, CANONICAL_IMAGE_WINDOW,
};
use crate::native_root_relation::NativeRootWeights;

/// Compile the A identity's coefficient weights. Proving supplies A; online
/// verification omits it and authenticates its contribution through Stage 3.
/// `rho[polynomial * rank + row][coefficient]` batches all seven public
/// projections over the joined residual. This compiles the signed remainder
/// `sum_i <rho_i, rem_Psi(A*z-sum c*T)_i>`; the caller supplies its projection
/// target. The T terms and every padding equation remain in the root oracle.
pub fn binary_direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan,
    matrix: Option<&[Vec<[i128; 648]>]>,
    challenges: &Challenges,
    rho: &[[E; 648]],
    padding_weights: &[E],
) -> Result<NativeRootWeights<E>, AkitaError> {
    let descriptor = plan.descriptor();
    let columns = descriptor.index_map().columns();
    let rows = checked::product([4, columns])
        .and_then(|d| checked::exact_div(descriptor.index_map().padded_carriers(), d))
        .ok_or(AkitaError::InvalidProof)?;
    let owners = descriptor.polynomials().len();
    if rho.len() != checked::product([owners, plan.rank()]).ok_or(AkitaError::InvalidProof)? {
        return Err(AkitaError::InvalidInput(
            "A coefficient functional has the wrong rank".into(),
        ));
    }
    if let Some(matrix) = matrix {
        if matrix.len() != plan.rank()
            || matrix.iter().any(|row| row.len() != rows)
            || matrix.iter().flatten().flatten().any(|&c| {
                c < i128::from(CANONICAL_IMAGE_WINDOW.0) || c > i128::from(CANONICAL_IMAGE_WINDOW.1)
            })
        {
            return Err(AkitaError::InvalidInput(
                "A differs from its centered typed geometry".into(),
            ));
        }
    }
    let embedded = binary_scalar_challenges(descriptor, challenges)?
        .into_iter()
        .map(|scalar| {
            Ok(pack_signed_components(&[scalar, [0; 162], [0; 162], [0; 162]])?.map(E::from_i128))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let adjoint = TrinomialAdjoint::<648, MinusTrinomial>::new()?;
    let response_digits = plan
        .repacking()
        .segments()
        .first()
        .ok_or(AkitaError::InvalidProof)?
        .digits;
    let image_digits = plan
        .repacking()
        .segments()
        .get(1)
        .ok_or(AkitaError::InvalidProof)?
        .digits;
    let response_terms = if matrix.is_some() {
        checked::product([owners, plan.rank(), rows, 648, response_digits])
            .ok_or(AkitaError::InvalidProof)?
    } else {
        0
    };
    let image_terms = checked::product([owners, plan.rank(), columns, 648, image_digits])
        .ok_or(AkitaError::InvalidProof)?;
    let term_count = checked::sum([response_terms, image_terms, padding_weights.len()])
        .ok_or(AkitaError::InvalidProof)?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(term_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    for owner in 0..owners {
        for i in 0..plan.rank() {
            let row_index =
                checked::mul_add(owner, plan.rank(), i).ok_or(AkitaError::InvalidProof)?;
            let weights = rho.get(row_index).ok_or(AkitaError::InvalidProof)?;
            if let Some(matrix) = matrix {
                for (j, a) in matrix
                    .get(i)
                    .ok_or(AkitaError::InvalidProof)?
                    .iter()
                    .enumerate()
                {
                    let element =
                        checked::mul_add(owner, rows, j).ok_or(AkitaError::InvalidProof)?;
                    for (t, w) in adjoint
                        .apply(weights, &a.map(E::from_i128))?
                        .into_iter()
                        .enumerate()
                    {
                        terms.extend(plan.repacking().coefficient_weights(0, element, t, w)?);
                    }
                }
            }
            for column in 0..columns {
                let challenge_index =
                    checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
                let challenge = embedded
                    .get(challenge_index)
                    .ok_or(AkitaError::InvalidProof)?;
                let element = checked::mul_add(challenge_index, plan.rank(), i)
                    .ok_or(AkitaError::InvalidProof)?;
                for (t, w) in adjoint.apply(weights, challenge)?.into_iter().enumerate() {
                    terms.extend(plan.repacking().coefficient_weights(1, element, t, -w)?);
                }
            }
        }
    }
    let mut padding = padding_weights.iter();
    for address in 0..plan.repacking().domain_len() {
        if plan.repacking().is_padding_address(address)? {
            terms.push((
                address,
                *padding
                    .next()
                    .ok_or_else(|| AkitaError::InvalidInput("missing padding weight".into()))?,
            ));
        }
    }
    if padding.next().is_some() {
        return Err(AkitaError::InvalidInput("excess padding weights".into()));
    }
    NativeRootWeights::new(plan.repacking().domain_len(), terms, E::zero())
}

#[cfg(test)]
#[path = "root_projection_weights_tests.rs"]
mod tests;
